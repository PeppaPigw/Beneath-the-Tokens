#!/usr/bin/env python3
"""CPU-only toy for online inference scheduling and SLOs.

The simulator is an auditable protocol model, not a vLLM/SGLang benchmark. It
models arrivals, admission control, prefill/decode batching, queueing, TTFT,
inter-token latency (ITL), end-to-end latency, cancellation, weighted tenant
fairness, and streaming backpressure with deterministic arithmetic.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Request:
    request_id: str
    tenant: str
    arrival_ms: float
    prompt_tokens: int
    max_new_tokens: int
    priority: int = 0
    cancel_ms: float | None = None

    def validate(self) -> None:
        if not self.request_id or not self.tenant:
            raise ValueError("request_id and tenant must be non-empty")
        if self.arrival_ms < 0 or self.prompt_tokens <= 0 or self.max_new_tokens <= 0:
            raise ValueError("arrival_ms must be non-negative; token counts must be positive")
        if self.cancel_ms is not None and self.cancel_ms < self.arrival_ms:
            raise ValueError("cancel_ms cannot precede arrival_ms")


@dataclass(frozen=True)
class SchedulerConfig:
    max_batch_requests: int = 4
    max_batch_prompt_tokens: int = 32
    prefill_ms_per_token: float = 0.08
    prefill_overhead_ms: float = 0.30
    decode_overhead_ms: float = 0.25
    decode_ms_per_sequence: float = 0.12
    disaggregated: bool = False
    policy: str = "fifo"
    tenant_weights: tuple[tuple[str, float], ...] = ()
    tenant_concurrency: tuple[tuple[str, int], ...] = ()
    stream_buffer_tokens: int = 4

    def validate(self) -> None:
        if self.max_batch_requests <= 0 or self.max_batch_prompt_tokens <= 0:
            raise ValueError("batch limits must be positive")
        for name in ("prefill_ms_per_token", "prefill_overhead_ms", "decode_overhead_ms", "decode_ms_per_sequence"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.policy not in {"fifo", "priority", "weighted_fair"}:
            raise ValueError("policy must be fifo, priority or weighted_fair")
        if self.stream_buffer_tokens <= 0:
            raise ValueError("stream_buffer_tokens must be positive")
        for tenant, weight in self.tenant_weights:
            if not tenant or weight <= 0:
                raise ValueError("tenant weights must be positive")
        for tenant, limit in self.tenant_concurrency:
            if not tenant or limit <= 0:
                raise ValueError("tenant concurrency limits must be positive")

    def weights(self) -> dict[str, float]:
        return dict(self.tenant_weights)

    def limits(self) -> dict[str, int]:
        return dict(self.tenant_concurrency)


@dataclass
class RequestResult:
    request_id: str
    tenant: str
    status: str
    arrival_ms: float
    prefill_start_ms: float | None = None
    prefill_finish_ms: float | None = None
    first_token_ms: float | None = None
    finish_ms: float | None = None
    ttft_ms: float | None = None
    itl_ms: float | None = None
    e2e_ms: float | None = None
    queue_ms: float | None = None
    generated_tokens: int = 0
    cancel_ms: float | None = None
    stream_dropped_tokens: int = 0


@dataclass(frozen=True)
class StreamResult:
    emitted: tuple[int, ...]
    dropped: int
    cancelled: bool
    producer_blocked: bool


def pctl(values: Iterable[float], percentile: float) -> float | None:
    xs = sorted(float(x) for x in values)
    if not xs:
        return None
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0,100]")
    if len(xs) == 1:
        return round(xs[0], 6)
    rank = (len(xs) - 1) * percentile / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    value = xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)
    return round(value, 6)


def make_requests(*, burst: bool = False) -> tuple[Request, ...]:
    """Small deterministic workload with two tenants and one cancellation."""
    if burst:
        return tuple(
            Request(f"r{i}", "gold" if i % 3 == 0 else "bronze", 0.0, 8 + (i % 4), 4 + (i % 3), priority=i % 2)
            for i in range(12)
        )
    return (
        Request("r0", "gold", 0.0, 8, 5, priority=2),
        Request("r1", "bronze", 0.4, 4, 4, priority=0),
        Request("r2", "gold", 0.7, 12, 3, priority=1, cancel_ms=4.0),
        Request("r3", "bronze", 1.1, 6, 6, priority=0),
        Request("r4", "gold", 2.0, 5, 2, priority=3),
    )


def admission_control(
    requests: Sequence[Request], *, max_inflight: int, per_tenant_limit: dict[str, int] | None = None
) -> tuple[tuple[Request, ...], tuple[dict, ...]]:
    """Accept a bounded prefix and return explicit reject reasons.

    This is an admission decision, not a scheduler guarantee: accepted work
    can still wait in a queue and callers should receive a retry-after hint.
    """
    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    limits = per_tenant_limit or {}
    accepted: list[Request] = []
    counts: dict[str, int] = {}
    rejected: list[dict] = []
    for req in sorted(requests, key=lambda r: (r.arrival_ms, r.request_id)):
        req.validate()
        reason = None
        if len(accepted) >= max_inflight:
            reason = "global_inflight_limit"
        elif req.tenant in limits and counts.get(req.tenant, 0) >= limits[req.tenant]:
            reason = "tenant_concurrency_limit"
        if reason:
            rejected.append({"request_id": req.request_id, "tenant": req.tenant, "reason": reason})
        else:
            accepted.append(req)
            counts[req.tenant] = counts.get(req.tenant, 0) + 1
    return tuple(accepted), tuple(rejected)


def stream_tokens(tokens: Sequence[int], *, consumer_capacity: int, buffer_capacity: int, cancel_after: int | None = None) -> StreamResult:
    """Model a bounded stream buffer; producer stops when consumer is slower."""
    if consumer_capacity <= 0 or buffer_capacity <= 0:
        raise ValueError("stream capacities must be positive")
    if cancel_after is not None and cancel_after < 0:
        raise ValueError("cancel_after must be non-negative")
    emitted: list[int] = []
    dropped = 0
    buffered: list[int] = []
    blocked = False
    cancelled = False
    for index, token in enumerate(tokens):
        if cancel_after is not None and index >= cancel_after:
            cancelled = True
            break
        if len(buffered) >= buffer_capacity:
            # Backpressure means the producer waits; this toy records a blocked
            # producer and drains what the consumer can take before retrying.
            blocked = True
            take = min(consumer_capacity, len(buffered))
            emitted.extend(buffered[:take])
            buffered = buffered[take:]
        if len(buffered) < buffer_capacity:
            buffered.append(int(token))
        else:
            dropped += 1
    emitted.extend(buffered[:consumer_capacity])
    dropped += max(0, len(buffered) - consumer_capacity)
    return StreamResult(tuple(emitted), dropped, cancelled, blocked)


def _pick_batch(waiting: list[Request], config: SchedulerConfig, served: dict[str, int]) -> list[Request]:
    if not waiting:
        return []
    if config.policy == "priority":
        ordered = sorted(waiting, key=lambda r: (-r.priority, r.arrival_ms, r.request_id))
    elif config.policy == "weighted_fair":
        weights = config.weights()
        ordered = sorted(waiting, key=lambda r: (served.get(r.tenant, 0) / weights.get(r.tenant, 1.0), r.arrival_ms, r.request_id))
    else:
        ordered = sorted(waiting, key=lambda r: (r.arrival_ms, r.request_id))
    selected: list[Request] = []
    tokens = 0
    for req in ordered:
        if len(selected) >= config.max_batch_requests:
            break
        if tokens + req.prompt_tokens > config.max_batch_prompt_tokens:
            continue
        selected.append(req)
        tokens += req.prompt_tokens
    return selected


def simulate(requests: Sequence[Request] | None = None, *, config: SchedulerConfig | None = None) -> dict:
    """Run a deterministic event-like scheduler and return JSON-ready evidence."""
    cfg = config or SchedulerConfig()
    cfg.validate()
    reqs = tuple(requests or make_requests())
    for req in reqs:
        req.validate()
    if len({r.request_id for r in reqs}) != len(reqs):
        raise ValueError("request_id values must be unique")
    waiting = sorted(reqs, key=lambda r: (r.arrival_ms, r.request_id))
    results = {r.request_id: RequestResult(r.request_id, r.tenant, "queued", r.arrival_ms, cancel_ms=r.cancel_ms) for r in reqs}
    served: dict[str, int] = {}
    active: list[dict] = []
    ready_later: list[dict] = []
    clock = 0.0
    guard = 0
    while waiting or active or ready_later:
        guard += 1
        if guard > 100000:
            raise RuntimeError("scheduler did not converge")
        # Make future arrivals visible, and activate finished disaggregated prefill.
        if not active and not ready_later and waiting and clock < waiting[0].arrival_ms:
            clock = waiting[0].arrival_ms
        arrived = [r for r in waiting if r.arrival_ms <= clock + 1e-9]
        waiting = [r for r in waiting if r.arrival_ms > clock + 1e-9]
        waiting.extend(arrived)
        for item in list(ready_later):
            if item["ready_at"] <= clock + 1e-9:
                active.append(item)
                ready_later.remove(item)
        # Cancellation before service is an explicit terminal state.
        for req in list(waiting):
            if req.cancel_ms is not None and req.cancel_ms <= clock + 1e-9:
                waiting.remove(req)
                results[req.request_id].status = "cancelled"
                results[req.request_id].finish_ms = clock
                results[req.request_id].e2e_ms = round(clock - req.arrival_ms, 6)
        batch = _pick_batch(waiting, cfg, served)
        if batch:
            for req in batch:
                waiting.remove(req)
            pf_start = clock
            pf_duration = cfg.prefill_overhead_ms + cfg.prefill_ms_per_token * sum(r.prompt_tokens for r in batch)
            pf_finish = pf_start + pf_duration
            for req in batch:
                row = results[req.request_id]
                row.status = "prefilling"
                row.prefill_start_ms = round(pf_start, 6)
                row.prefill_finish_ms = round(pf_finish, 6)
                row.queue_ms = round(pf_start - req.arrival_ms, 6)
                served[req.tenant] = served.get(req.tenant, 0) + req.prompt_tokens
            payload = [{"req": req, "ready_at": pf_finish, "remaining": req.max_new_tokens, "tokens": [], "next_token": None} for req in batch]
            if cfg.disaggregated:
                ready_later.extend(payload)
            else:
                clock = pf_finish
                active.extend(payload)
        # In disaggregated mode, prefill overlaps decode; in coupled mode the
        # clock already includes prefill and therefore exposes head-of-line delay.
        if active:
            decode_duration = cfg.decode_overhead_ms + cfg.decode_ms_per_sequence * len(active)
            next_active: list[dict] = []
            for item in active:
                req: Request = item["req"]
                row = results[req.request_id]
                if req.cancel_ms is not None and req.cancel_ms <= clock + decode_duration + 1e-9:
                    row.status = "cancelled"
                    row.finish_ms = round(min(req.cancel_ms, clock + decode_duration), 6)
                    row.e2e_ms = round(row.finish_ms - req.arrival_ms, 6)
                    continue
                item["tokens"].append(clock + decode_duration)
                item["remaining"] -= 1
                if item["remaining"] <= 0:
                    row.status = "completed"
                    row.generated_tokens = len(item["tokens"])
                    row.first_token_ms = round(item["tokens"][0], 6)
                    row.finish_ms = round(item["tokens"][-1], 6)
                    row.ttft_ms = round(row.first_token_ms - req.arrival_ms, 6)
                    row.itl_ms = round((item["tokens"][-1] - item["tokens"][0]) / max(1, len(item["tokens"]) - 1), 6) if len(item["tokens"]) > 1 else 0.0
                    row.e2e_ms = round(row.finish_ms - req.arrival_ms, 6)
                else:
                    next_active.append(item)
            active = next_active
            clock += decode_duration
        elif ready_later:
            clock = min(item["ready_at"] for item in ready_later)
        elif waiting:
            clock = max(clock, waiting[0].arrival_ms)
    rows = [asdict(results[r.request_id]) for r in sorted(reqs, key=lambda x: x.request_id)]
    completed = [r for r in rows if r["status"] == "completed"]
    cancelled = [r for r in rows if r["status"] == "cancelled"]
    ttft = [r["ttft_ms"] for r in completed if r["ttft_ms"] is not None]
    itl = [r["itl_ms"] for r in completed if r["itl_ms"] is not None]
    e2e = [r["e2e_ms"] for r in completed if r["e2e_ms"] is not None]
    tenant_stats: dict[str, dict] = {}
    for row in rows:
        stat = tenant_stats.setdefault(row["tenant"], {"requests": 0, "completed": 0, "cancelled": 0, "ttft_ms": []})
        stat["requests"] += 1
        stat["completed"] += row["status"] == "completed"
        stat["cancelled"] += row["status"] == "cancelled"
        if row["ttft_ms"] is not None:
            stat["ttft_ms"].append(row["ttft_ms"])
    for stat in tenant_stats.values():
        stat["ttft_p95_ms"] = pctl(stat.pop("ttft_ms"), 95)
    return {
        "schema_version": 1,
        "experiment": "ch35-serving-scheduling-cpu-toy",
        "config": asdict(cfg),
        "requests": rows,
        "metrics": {"completed": len(completed), "cancelled": len(cancelled), "ttft_p50_ms": pctl(ttft, 50), "ttft_p95_ms": pctl(ttft, 95), "itl_p50_ms": pctl(itl, 50), "itl_p95_ms": pctl(itl, 95), "e2e_p50_ms": pctl(e2e, 50), "e2e_p95_ms": pctl(e2e, 95)},
        "tenant_stats": tenant_stats,
        "invariants": {"all_terminal": all(r["status"] in {"completed", "cancelled"} for r in rows), "no_negative_latency": all((r["e2e_ms"] or 0) >= 0 for r in rows), "tokens_accounted": all(r["generated_tokens"] <= next(x.max_new_tokens for x in reqs if x.request_id == r["request_id"]) for r in rows)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burst", action="store_true")
    parser.add_argument("--disaggregated", action="store_true")
    parser.add_argument("--policy", choices=["fifo", "priority", "weighted_fair"], default="fifo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = SchedulerConfig(disaggregated=args.disaggregated, policy=args.policy, tenant_weights=(("gold", 2.0), ("bronze", 1.0)))
    payload = simulate(make_requests(burst=args.burst), config=config)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
