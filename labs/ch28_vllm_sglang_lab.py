#!/usr/bin/env python3
"""CPU-only serving-runtime toy for chapter 28.

The simulator models scheduler token budgets, paged versus radix-prefix KV
accounting, speculative draft/verify, and structured-generation rejections.
It is a deterministic protocol/queue probe, not a vLLM/SGLang or GPU benchmark.
Python 3.10+ and the standard library are sufficient.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    input_len: int
    output_len: int
    arrival_ms: float


@dataclass
class RequestResult:
    request_id: str
    engine: str
    input_len: int
    output_len: int
    prefill_tokens: int = 0
    cache_hit_tokens: int = 0
    allocated_pages: int = 0
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    grammar_rejects: int = 0
    preemptions: int = 0
    status: str = "WAITING"


@dataclass(frozen=True)
class Config:
    seed: int = 7
    requests: int = 40
    concurrency: int = 8
    input_len: int = 256
    output_len: int = 64
    shared_prefix: int = 0
    engine: str = "paged"
    page_tokens: int = 16
    max_batch_tokens: int = 128
    speculative: bool = False
    draft_tokens: int = 4
    accept_rate: float = 0.75
    grammar_reject_rate: float = 0.0
    max_pages: int = 1_000_000


def percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def make_requests(cfg: Config) -> list[RequestSpec]:
    # Stagger arrivals enough to exercise admission while remaining deterministic.
    return [RequestSpec(f"req-{i:04d}", cfg.input_len, cfg.output_len, float(i * 0.5)) for i in range(cfg.requests)]


def simulate(cfg: Config) -> dict:
    if cfg.engine not in {"paged", "radix"}:
        raise ValueError("engine must be paged or radix")
    if cfg.page_tokens <= 0 or cfg.max_batch_tokens <= 0:
        raise ValueError("page_tokens and max_batch_tokens must be positive")
    if not 0 <= cfg.accept_rate <= 1 or not 0 <= cfg.grammar_reject_rate <= 1:
        raise ValueError("accept_rate and grammar_reject_rate must be in [0, 1]")
    rng = random.Random(cfg.seed)
    specs = make_requests(cfg)
    results = [RequestResult(s.request_id, cfg.engine, s.input_len, s.output_len) for s in specs]
    waiting = list(range(len(specs)))
    running: list[int] = []
    prefilled = set()
    clock_ms = 0.0
    max_pages_live = 0
    live_pages = 0
    step_count = 0
    # Prefix is considered shareable only after the first request commits it.
    radix_prefix_ready = False
    while waiting or running:
        step_count += 1
        # Admit in arrival order, respecting concurrency.
        while waiting and len(running) < cfg.concurrency:
            idx = waiting.pop(0)
            results[idx].status = "PREFILL"
            running.append(idx)

        if not running:
            clock_ms += 0.5
            continue

        budget = cfg.max_batch_tokens
        selected_decode: list[int] = []
        # Reserve one decode token (or a draft/verify group) for each running req.
        for idx in list(running):
            r = results[idx]
            if budget <= 0:
                break
            if r.status not in {"DECODING", "PREFILL"}:
                continue
            # Newly admitted requests use prefill first; decode starts next step.
            if r.output_tokens == 0 and r.prefill_tokens < r.input_len:
                continue
            take = 1
            if cfg.speculative and r.output_tokens < r.output_len:
                take = min(cfg.draft_tokens + 1, r.output_len - r.output_tokens)
            if take <= budget:
                selected_decode.append(idx)
                budget -= take

        # Prefill consumes remaining budget. Radix can skip a common prefix after warmup.
        prefill_total = 0
        prefill_items: list[tuple[int, int, int]] = []
        for idx in list(running):
            if budget <= 0:
                break
            r = results[idx]
            if r.prefill_tokens >= r.input_len:
                continue
            logical_remaining = r.input_len - r.prefill_tokens
            hit = 0
            if cfg.engine == "radix" and cfg.shared_prefix and radix_prefix_ready and r.prefill_tokens == 0:
                hit = min(cfg.shared_prefix, logical_remaining)
                r.cache_hit_tokens += hit
                r.prefill_tokens += hit
                logical_remaining -= hit
            take = min(logical_remaining, budget)
            if take <= 0:
                continue
            prefill_items.append((idx, take, hit))
            r.prefill_tokens += take
            r.status = "PREFILL"
            prefill_total += take
            budget -= take
            if r.prefill_tokens >= r.input_len:
                radix_prefix_ready = radix_prefix_ready or cfg.engine == "radix"
                r.status = "DECODING"

        # Allocate pages for newly visible logical tokens. A deliberately simple
        # high-water mark makes page-size and preemption effects inspectable.
        for idx, _, _ in prefill_items:
            r = results[idx]
            needed = math.ceil((r.prefill_tokens + r.output_tokens) / cfg.page_tokens)
            delta = max(0, needed - r.allocated_pages)
            r.allocated_pages = needed
            live_pages += delta
        for idx in selected_decode:
            r = results[idx]
            if r.status == "PREFILL":
                r.status = "DECODING"
            # A page may be needed for the newly accepted tokens.
            proposed = r.output_tokens + (min(cfg.draft_tokens + 1, r.output_len - r.output_tokens) if cfg.speculative else 1)
            needed = math.ceil((r.input_len + proposed) / cfg.page_tokens)
            delta = max(0, needed - r.allocated_pages)
            r.allocated_pages = needed
            live_pages += delta
        max_pages_live = max(max_pages_live, live_pages)

        # If capacity is exceeded, preempt the last selected request (toy policy).
        if live_pages > cfg.max_pages and selected_decode:
            victim = selected_decode[-1]
            results[victim].preemptions += 1
            results[victim].status = "DECODING"
            live_pages -= max(1, results[victim].allocated_pages // 4)

        # Logical step cost. It is a model of queue causality, not hardware time.
        decode_tokens = 0
        draft_tokens = 0
        accepted_tokens = 0
        grammar_rejects = 0
        for idx in selected_decode:
            r = results[idx]
            remaining = r.output_len - r.output_tokens
            if remaining <= 0:
                continue
            if cfg.speculative:
                draft = min(cfg.draft_tokens, remaining)
                accepted = sum(1 for _ in range(draft) if rng.random() < cfg.accept_rate)
                # Target always commits one corrective token after accepted prefix.
                advance = min(remaining, accepted + 1)
                r.draft_tokens += draft
                r.accepted_tokens += accepted
            else:
                draft = 0
                accepted = 0
                advance = 1
            rejects = sum(1 for _ in range(advance) if rng.random() < cfg.grammar_reject_rate)
            r.grammar_rejects += rejects
            # Grammar rejection is a retry/check cost; it does not add output tokens.
            r.output_tokens += advance
            decode_tokens += advance
            draft_tokens += draft
            accepted_tokens += accepted
            grammar_rejects += rejects
            if r.ttft_ms is None:
                r.ttft_ms = clock_ms + 0.6
            if r.output_tokens >= r.output_len:
                r.output_tokens = r.output_len
                r.status = "FINISHED"
                r.e2e_ms = clock_ms + 0.6 - specs[idx].arrival_ms
                live_pages = max(0, live_pages - r.allocated_pages)
                running.remove(idx)

        # Prefill and decode overlap in one iteration; cost grows with work.
        step_ms = 0.6 + 0.004 * prefill_total + 0.09 * decode_tokens
        step_ms += 0.04 * draft_tokens + 0.03 * grammar_rejects
        if prefill_items and cfg.engine == "radix":
            step_ms += 0.02 * len(prefill_items)  # lookup/node bookkeeping
        clock_ms += step_ms
        for idx in running:
            if results[idx].ttft_ms is None and results[idx].prefill_tokens >= results[idx].input_len:
                results[idx].ttft_ms = clock_ms

        # A safety guard protects against malformed arguments.
        if step_count > 1_000_000:
            raise RuntimeError("simulation did not converge")

    finished = [r for r in results if r.e2e_ms is not None]
    ttft = [r.ttft_ms or 0.0 for r in finished]
    e2e = [r.e2e_ms or 0.0 for r in finished]
    itl_values: list[float] = []
    # Approximate ITL from decode step cost; this remains a logical metric.
    for r in finished:
        if r.output_tokens > 1:
            itl_values.extend([0.69] * (r.output_tokens - 1))
    total_out = sum(r.output_tokens for r in finished)
    return {
        "schema_version": 1,
        "experiment": "ch28-vllm-sglang-serving-toy",
        "config": asdict(cfg),
        "summary": {
            "requests": len(results),
            "completed": len(finished),
            "steps": step_count,
            "clock_ms": round(clock_ms, 6),
            "ttft_ms": {"p50": round(percentile(ttft, 0.50), 6), "p95": round(percentile(ttft, 0.95), 6), "p99": round(percentile(ttft, 0.99), 6)},
            "e2e_ms": {"p50": round(percentile(e2e, 0.50), 6), "p95": round(percentile(e2e, 0.95), 6), "p99": round(percentile(e2e, 0.99), 6)},
            "itl_ms": {"p50": round(percentile(itl_values, 0.50), 6), "p95": round(percentile(itl_values, 0.95), 6)},
            "output_tokens": total_out,
            "output_tokens_per_s": round(total_out / (clock_ms / 1000.0), 6) if clock_ms else 0.0,
            "prefill_tokens": sum(r.prefill_tokens - r.cache_hit_tokens for r in results),
            "cache_hit_tokens": sum(r.cache_hit_tokens for r in results),
            "allocated_pages": sum(r.allocated_pages for r in results),
            "max_pages_live": max_pages_live,
            "draft_tokens": sum(r.draft_tokens for r in results),
            "accepted_tokens": sum(r.accepted_tokens for r in results),
            "grammar_rejects": sum(r.grammar_rejects for r in results),
            "preemptions": sum(r.preemptions for r in results),
            "errors": sum(1 for r in results if r.status == "FAILED"),
        },
        "requests": [asdict(r) for r in results],
        "notes": [
            "Logical-time CPU toy; elapsed and throughput are not GPU or framework benchmarks.",
            "Radix cache shares only the configured synthetic prefix after the first committed request.",
            "Page accounting is approximate and intentionally omits CUDA allocator, logits, weights, and network buffers.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--requests", type=int, default=40)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--shared-prefix", type=int, default=0)
    p.add_argument("--engine", choices=("paged", "radix"), default="paged")
    p.add_argument("--page-tokens", type=int, default=16)
    p.add_argument("--max-batch-tokens", type=int, default=128)
    p.add_argument("--speculative", action="store_true")
    p.add_argument("--draft-tokens", type=int, default=4)
    p.add_argument("--accept-rate", type=float, default=0.75)
    p.add_argument("--grammar-reject-rate", type=float, default=0.0)
    p.add_argument("--max-pages", type=int, default=1_000_000)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> int:
    ns = parse_args()
    cfg = Config(**{k: getattr(ns, k.replace("-", "_")) for k in Config.__dataclass_fields__})
    if cfg.requests < 0 or cfg.concurrency <= 0 or cfg.input_len < 0 or cfg.output_len <= 0:
        raise SystemExit("requests >= 0, concurrency > 0, input-len >= 0, output-len > 0 required")
    payload = simulate(cfg)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if ns.output:
        ns.output.parent.mkdir(parents=True, exist_ok=True)
        ns.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
