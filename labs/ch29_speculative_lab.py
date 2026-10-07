#!/usr/bin/env python3
"""CPU-only speculative + constrained decoding lab for chapter 29.

This is a protocol simulator. It implements a small exact accept/reject sampler,
JSON-like finite-state masking, a tree/lookahead proposal toy, and a scheduler
that charges target and draft work. It is deliberately not a GPU benchmark.
Python 3.10+; standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence
from pathlib import Path

VOCAB = tuple(range(8))


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    output_len: int
    arrival_ms: float
    grammar: bool = False


@dataclass(frozen=True)
class Config:
    seed: int = 29
    requests: int = 24
    concurrency: int = 6
    output_len: int = 32
    draft_len: int = 4
    accept_rate: float = 0.72
    grammar_reject_rate: float = 0.0
    max_batch_tokens: int = 32
    mode: str = "speculative"  # baseline/speculative/lookahead
    grammar: bool = False


def normalize(xs: Sequence[float]) -> list[float]:
    total = sum(xs)
    if total <= 0:
        raise ValueError("distribution has no mass")
    return [x / total for x in xs]


def masked_distribution(probs: Sequence[float], allowed: Iterable[int]) -> list[float]:
    allow = set(allowed)
    out = [p if i in allow else 0.0 for i, p in enumerate(probs)]
    return normalize(out)


def json_allowed(position: int) -> set[int]:
    """Tiny deterministic grammar: { "a" : digit } represented by token classes.

    Tokens 0..7 stand for {, ", a, :, digits 0..3, }. Position is modulo 7.
    The grammar is intentionally tiny so tests can assert every committed token.
    """
    pattern = ({0}, {1}, {2}, {3}, {4, 5, 6, 7}, {1}, {7})
    return set(pattern[position % len(pattern)])


def sample(rng: random.Random, probs: Sequence[float]) -> int:
    x = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if x <= acc:
            return i
    return len(probs) - 1


def target_distribution(position: int, drift: float = 0.0) -> list[float]:
    # A smooth, position-dependent target distribution. No model claim.
    weights = [1.0 + ((position + i) % 5) * 0.15 + drift * (i % 2) for i in VOCAB]
    return normalize(weights)


def draft_distribution(position: int, quality: float) -> list[float]:
    target = target_distribution(position)
    # Interpolate draft toward uniform; quality=1 matches target.
    uniform = [1 / len(VOCAB)] * len(VOCAB)
    return normalize([quality * p + (1.0 - quality) * u for p, u in zip(target, uniform)])


def exact_verify(
    rng: random.Random,
    target: Sequence[float],
    draft_tokens: Sequence[int],
    draft_probs: Sequence[Sequence[float]],
) -> tuple[int, int, int]:
    """Verify a draft chain with Leviathan accept/reject.

    Returns (accepted prefix length, committed token count, rejection count).
    On rejection, one residual target sample is committed and the chain ends.
    """
    accepted = 0
    # Accept either one target distribution (legacy/unit-test convenience) or
    # one distribution per candidate position (the real chain case).
    nested = bool(target) and isinstance(target[0], (list, tuple))  # type: ignore[index]
    for j, (token, q) in enumerate(zip(draft_tokens, draft_probs)):
        p_dist = target[j] if nested else target  # type: ignore[index]
        p = p_dist[token]
        # A token with q=0 cannot be sampled by a valid draft. Treat an
        # externally supplied zero-q candidate as a rejection, never as an
        # automatic acceptance.
        ratio = min(1.0, p / q[token]) if q[token] > 0 else 0.0
        if rng.random() < ratio:
            accepted += 1
            continue
        residual = [max(0.0, p_i - q_i) for p_i, q_i in zip(p_dist, q)]
        if sum(residual) <= 1e-12:
            residual = list(p_dist)
        _ = sample(rng, normalize(residual))
        return accepted, accepted + 1, 1
    # Draft chain exhausted; target samples the one lookahead token.
    p_last = target[len(draft_tokens)] if nested and len(target) > len(draft_tokens) else target  # type: ignore[index]
    _ = sample(rng, p_last)
    return accepted, accepted + 1, 0


def constrained_speculative(
    rng: random.Random,
    output_len: int,
    draft_len: int,
    quality: float,
    grammar: bool,
    grammar_reject_rate: float,
) -> dict:
    """Generate one sequence and report exactness/grammar accounting."""
    pos = 0
    accepted = committed = drafts = rejects = grammar_rejects = 0
    committed_tokens: list[int] = []
    while committed < output_len:
        remaining = output_len - committed
        k = min(draft_len, remaining)
        d_tokens: list[int] = []
        d_probs: list[list[float]] = []
        for j in range(k):
            p = draft_distribution(pos + j, quality)
            if grammar:
                p = masked_distribution(p, json_allowed(pos + j))
            tok = sample(rng, p)
            # Optional synthetic rejection models a grammar engine rejecting a
            # parser-invalid candidate before target verification.
            if grammar and rng.random() < grammar_reject_rate:
                rejects += 1
                grammar_rejects += 1
                tok = next(iter(json_allowed(pos + j)))
            d_tokens.append(tok)
            d_probs.append(p)
        p_targets = [target_distribution(pos + j) for j in range(k + 1)]
        if grammar:
            p_targets = [masked_distribution(p, json_allowed(pos + j)) for j, p in enumerate(p_targets)]
        a, advance, rej = exact_verify(rng, p_targets, d_tokens, d_probs)
        # Materialize committed prefix plus a grammar-valid correction/bonus.
        # The exact sampler above draws the correction internally; this toy
        # records a deterministic allowed representative so state transitions
        # remain inspectable without exposing floating-point logits.
        advance = min(advance, remaining)
        accepted += min(a, advance)
        committed_part = list(d_tokens[:a])
        correction_pos = pos + len(committed_part)
        if len(committed_part) < advance:
            committed_part.append(min(json_allowed(correction_pos)) if grammar else 0)
        committed_tokens.extend(committed_part)
        committed += len(committed_part)
        drafts += k
        pos += len(committed_part)
    return {
        "output_tokens": committed,
        "draft_tokens": drafts,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafts if drafts else 0.0,
        "verify_rejections": rejects,
        "grammar_rejects": grammar_rejects,
        "tokens": committed_tokens,
    }


def make_requests(cfg: Config) -> list[RequestSpec]:
    return [RequestSpec(f"req-{i:03d}", cfg.output_len, i * 0.25, cfg.grammar and i % 2 == 0) for i in range(cfg.requests)]


def simulate(cfg: Config) -> dict:
    if cfg.mode not in {"baseline", "speculative", "lookahead"}:
        raise ValueError("mode must be baseline, speculative, or lookahead")
    if cfg.draft_len <= 0 or cfg.max_batch_tokens <= 0 or cfg.concurrency <= 0:
        raise ValueError("draft_len, max_batch_tokens, concurrency must be positive")
    if not 0 <= cfg.accept_rate <= 1 or not 0 <= cfg.grammar_reject_rate <= 1:
        raise ValueError("rates must be in [0,1]")
    rng = random.Random(cfg.seed)
    specs = make_requests(cfg)
    waiting = list(range(len(specs)))
    running: list[int] = []
    done: list[dict] = []
    clock = 0.0
    steps = 0
    draft_total = accepted_total = grammar_total = target_tokens = 0
    ttfts: list[float] = []
    while waiting or running:
        steps += 1
        while waiting and len(running) < cfg.concurrency:
            running.append(waiting.pop(0))
        budget = cfg.max_batch_tokens
        selected: list[int] = []
        for idx in running:
            if budget <= 0:
                break
            take = 1 if cfg.mode == "baseline" else min(cfg.draft_len + 1, cfg.output_len)
            if take <= budget:
                selected.append(idx)
                budget -= take
        step_target = 0
        for idx in list(selected):
            spec = specs[idx]
            if cfg.mode == "baseline":
                result = {"output_tokens": spec.output_len, "draft_tokens": 0, "accepted_tokens": 0, "grammar_rejects": 0, "verify_rejections": 0}
                target = spec.output_len
            else:
                q = min(0.999, max(0.001, cfg.accept_rate))
                result = constrained_speculative(rng, spec.output_len, cfg.draft_len, q, spec.grammar, cfg.grammar_reject_rate)
                target = max(1, result["output_tokens"] - result["accepted_tokens"] + 1)
            step_target += target
            draft_total += result["draft_tokens"]
            accepted_total += result["accepted_tokens"]
            grammar_total += result["grammar_rejects"]
            target_tokens += target
            # Track a request as completed when this toy step reaches its full
            # output; this keeps scheduler accounting deterministic.
            if cfg.mode == "baseline" or result["output_tokens"] >= cfg.output_len:
                running.remove(idx)
                ttfts.append(clock + 0.6 - spec.arrival_ms)
                done.append({"request_id": spec.request_id, **result, "status": "FINISHED"})
        step_ms = 0.55 + 0.08 * step_target + 0.025 * draft_total / max(1, len(selected))
        if cfg.mode == "lookahead":
            step_ms += 0.015 * len(selected)  # tree merge/verification bookkeeping
        step_ms += 0.02 * grammar_total / max(1, len(selected))
        clock += step_ms
        if steps > 100000:
            raise RuntimeError("simulation did not converge")
    return {
        "schema_version": 1,
        "experiment": "ch29-speculative-structured-decoding-toy",
        "config": asdict(cfg),
        "summary": {
            "requests": len(specs),
            "completed": len(done),
            "steps": steps,
            "logical_time_ms": round(clock, 6),
            "ttft_ms": {"p50": round(percentile(ttfts, 0.5), 6), "p95": round(percentile(ttfts, 0.95), 6)},
            "target_tokens": target_tokens,
            "draft_tokens": draft_total,
            "accepted_tokens": accepted_total,
            "acceptance_rate": round(accepted_total / draft_total, 6) if draft_total else 0.0,
            "grammar_rejects": grammar_total,
            "exact_output_token_count": sum(r["output_tokens"] for r in done),
        },
        "requests": done,
        "notes": [
            "Logical simulator only; no model logits, GPU kernels, CUDA graphs, tokenizer or network",
            "Acceptance ratio is a toy proposal quality parameter and not a benchmark result",
            "Grammar masks are finite-state and intentionally smaller than JSON Schema",
        ],
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--accept-rate", type=float, default=0.72)
    parser.add_argument("--grammar-reject-rate", type=float, default=0.0)
    parser.add_argument("--max-batch-tokens", type=int, default=32)
    parser.add_argument("--mode", choices=("baseline", "speculative", "lookahead"), default="speculative")
    parser.add_argument("--grammar", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = simulate(Config(seed=args.seed, requests=args.requests, concurrency=args.concurrency,
                              output_len=args.output_len, draft_len=args.draft_len,
                              accept_rate=args.accept_rate, grammar_reject_rate=args.grammar_reject_rate,
                              max_batch_tokens=args.max_batch_tokens, mode=args.mode, grammar=args.grammar))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
