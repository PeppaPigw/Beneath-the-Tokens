#!/usr/bin/env python3
"""CPU-only deterministic observability and tracing toy.

This lab models OpenTelemetry-like spans, Prometheus-style histograms, logs,
profile samples, GPU/network/storage telemetry, tail sampling and cross-layer
diagnosis.  It intentionally does not import CUDA, DCGM, eBPF or a network
client; production conclusions require pinned hardware and replayed traffic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Request:
    request_id: str
    tenant: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int

    def validate(self) -> None:
        if not self.request_id or not self.tenant:
            raise ValueError("request_id and tenant must be non-empty")
        if self.arrival_ms < 0 or self.prompt_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("arrival_ms and token counts must be positive")


@dataclass(frozen=True)
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_ms: float
    end_ms: float
    attributes: dict[str, object]
    status: str = "OK"

    @property
    def duration_ms(self) -> float:
        return round(self.end_ms - self.start_ms, 6)


@dataclass(frozen=True)
class LogRecord:
    timestamp_ms: float
    severity: str
    message: str
    trace_id: str
    attributes: dict[str, object]


@dataclass(frozen=True)
class ProfileSample:
    timestamp_ms: float
    domain: str
    symbol: str
    duration_ms: float
    trace_id: str


BUCKETS = (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, float("inf"))


def pctl(values: Iterable[float], percentile: float) -> float | None:
    xs = sorted(float(v) for v in values)
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0,100]")
    if not xs:
        return None
    if len(xs) == 1:
        return round(xs[0], 6)
    rank = (len(xs) - 1) * percentile / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (rank - lo), 6)


def histogram(values: Sequence[float]) -> dict[str, object]:
    counts = []
    for upper in BUCKETS:
        counts.append(sum(1 for value in values if value <= upper))
    return {"count": len(values), "sum": round(sum(values), 6), "buckets": [
        {"le_ms": (None if upper == float("inf") else upper), "cumulative_count": count}
        for upper, count in zip(BUCKETS, counts)
    ]}


def make_workload() -> tuple[Request, ...]:
    return tuple(
        Request(
            request_id=f"r{i:02d}",
            tenant="gold" if i % 3 == 0 else ("silver" if i % 3 == 1 else "bronze"),
            arrival_ms=round(i * 2.5, 3),
            prompt_tokens=8 + (i % 5) * 2,
            output_tokens=4 + (i % 4),
        )
        for i in range(12)
    )


# Public aliases used by chapter exercises and downstream contract tests.
make_requests = make_workload


def cardinality_guard(metric_labels: Sequence[str], budget: int) -> dict[str, object]:
    return _metric_series_guard(metric_labels, budget)


def _id(prefix: str, index: int) -> str:
    return f"{prefix}-{index:04d}"


def _trace_id(request_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{request_id}".encode()).hexdigest()[:16]
    return f"{digest}"


def _metric_series_guard(metric_labels: Sequence[str], budget: int) -> dict[str, object]:
    if budget <= 0:
        raise ValueError("cardinality budget must be positive")
    # Simulate a route/model/device set and deliberately reject request IDs.
    unique = list(dict.fromkeys(metric_labels))
    dropped = max(0, len(unique) - budget)
    return {"budget": budget, "observed_series": len(unique), "accepted_series": min(len(unique), budget),
            "dropped_series": dropped, "forbidden_labels": ["trace_id", "request_id"]}


def tail_sample(spans: Sequence[Span], *, fault: str = "none", threshold_ms: float = 180.0, head_rate: float = 0.10) -> dict[str, object]:
    """Keep errors/slow traces plus deterministic head samples."""
    by_trace: dict[str, list[Span]] = {}
    for span in spans:
        by_trace.setdefault(span.trace_id, []).append(span)
    kept: list[str] = []
    reasons: dict[str, list[str]] = {}
    for trace_id, rows in sorted(by_trace.items()):
        root = next((row for row in rows if row.parent_span_id is None), rows[0])
        duration = max(row.end_ms for row in rows) - min(row.start_ms for row in rows)
        reason: list[str] = []
        if root.status != "OK":
            reason.append("error")
        if duration >= threshold_ms:
            reason.append("slow")
        if fault != "none" and any("network" in row.name or "storage" in row.name for row in rows):
            reason.append("fault_signal")
        # Stable head sample uses the first byte of trace hash, avoiding RNG drift.
        head = int(trace_id[:2], 16) / 255.0 < head_rate
        if head:
            reason.append("head_sample")
        if reason:
            kept.append(trace_id)
            reasons[trace_id] = sorted(set(reason))
    return {"head_rate": head_rate, "tail_threshold_ms": threshold_ms, "kept_trace_ids": kept, "reasons": reasons,
            "total_traces": len(by_trace), "kept_count": len(kept)}


def simulate(requests: Sequence[Request] | None = None, *, fault: str = "none", seed: int = 7, cardinality_budget: int = 8) -> dict[str, object]:
    if fault not in {"none", "network_tail", "storage_tail", "gpu_throttle"}:
        raise ValueError("fault must be none, network_tail, storage_tail or gpu_throttle")
    reqs = tuple(requests or make_workload())
    for req in reqs:
        req.validate()
    rng = random.Random(seed)
    spans: list[Span] = []
    logs: list[LogRecord] = []
    profiles: list[ProfileSample] = []
    telemetry: list[dict[str, object]] = []
    phase_values: dict[str, list[float]] = {"queue": [], "prefill": [], "decode": [], "ttft": [], "e2e": [], "itl": []}
    metric_labels: list[str] = []
    request_rows: list[dict[str, object]] = []
    cursor = 0.0
    for i, req in enumerate(reqs):
        trace_id = _trace_id(req.request_id, seed)
        root_id = _id("span", i * 10)
        queue_id, prefill_id, decode_id, response_id = (_id("span", i * 10 + j) for j in range(1, 5))
        queue_start = max(req.arrival_ms, cursor)
        queue_ms = round(queue_start - req.arrival_ms, 3)
        prefill_ms = round(10.0 + req.prompt_tokens * 0.7 + (i % 2) * 0.5, 3)
        decode_ms = round(7.0 + req.output_tokens * 4.0, 3)
        network_ms = 0.0
        storage_ms = 0.0
        gpu_throttle_ms = 0.0
        if fault == "network_tail" and i % 3 == 1:
            network_ms = 75.0 + (i % 4) * 8.0
            decode_ms += network_ms
        if fault == "storage_tail" and i % 4 == 2:
            storage_ms = 95.0 + (i % 3) * 15.0
            queue_ms += storage_ms * 0.2
        if fault == "gpu_throttle" and i % 4 == 1:
            gpu_throttle_ms = 90.0 + (i % 3) * 12.0
            decode_ms += gpu_throttle_ms
        prefill_start = round(queue_start, 3)
        prefill_end = round(prefill_start + prefill_ms, 3)
        first_token = round(prefill_end + 3.0, 3)
        decode_start = first_token
        decode_end = round(decode_start + decode_ms, 3)
        finish = round(decode_end + 1.5, 3)
        root_end = finish
        status = "ERROR" if (fault == "network_tail" and i == 10) else "OK"
        spans.extend([
            Span(trace_id, root_id, None, "inference.server", req.arrival_ms, root_end,
                 {"request_id": req.request_id, "tenant": req.tenant, "model": "toy-llm", "route": "/generate"}, status),
            Span(trace_id, queue_id, root_id, "scheduler.queue_wait", req.arrival_ms, prefill_start,
                 {"queue": "decode", "batch_id": f"batch-{i // 3:02d}"}),
            Span(trace_id, prefill_id, root_id, "model.prefill", prefill_start, prefill_end,
                 {"prompt_tokens": req.prompt_tokens, "device_uuid": "toy-gpu-0"}),
            Span(trace_id, decode_id, root_id, "model.decode", decode_start, decode_end,
                 {"output_tokens": req.output_tokens, "network_ms": network_ms, "collective_seq": i}),
            Span(trace_id, response_id, root_id, "response.serialize", decode_end, finish,
                 {"stream": True}),
        ])
        if network_ms:
            spans.append(Span(trace_id, _id("span", i * 10 + 5), decode_id, "network.rpc", decode_start + 1.0,
                              decode_start + 1.0 + network_ms, {"peer": "shard-1", "retransmits": 3 + i % 2}))
            logs.append(LogRecord(decode_start + network_ms, "WARN", "network tail injected", trace_id,
                                  {"retransmits": 3 + i % 2, "collective_seq": i}))
        if storage_ms:
            spans.append(Span(trace_id, _id("span", i * 10 + 6), root_id, "storage.checkpoint_fsync", prefill_start,
                              prefill_start + storage_ms, {"device": "nvme0", "bytes": 4096}))
            logs.append(LogRecord(prefill_start + storage_ms, "WARN", "storage tail injected", trace_id,
                                  {"device": "nvme0", "io_latency_ms": storage_ms}))
        if gpu_throttle_ms:
            spans.append(Span(trace_id, _id("span", i * 10 + 7), decode_id, "gpu.throttle", decode_start,
                              decode_start + gpu_throttle_ms, {"reason": "power_limit", "device_uuid": "toy-gpu-0"}))
            logs.append(LogRecord(decode_start + gpu_throttle_ms, "WARN", "GPU throttle injected", trace_id,
                                  {"device_uuid": "toy-gpu-0", "throttle_ms": gpu_throttle_ms}))
        logs.append(LogRecord(req.arrival_ms, "INFO", "request accepted", trace_id,
                              {"request_id": req.request_id, "tenant": req.tenant}))
        profiles.extend([
            ProfileSample(prefill_start + prefill_ms / 2, "cpu", "tokenizer.encode", round(prefill_ms * 0.2, 3), trace_id),
            ProfileSample(decode_start + decode_ms / 2, "gpu", "toy.decode_kernel", round(decode_ms * 0.65, 3), trace_id),
        ])
        gpu_util = round(max(0.1, min(0.98, 0.82 - network_ms / 400.0 - storage_ms / 500.0 - gpu_throttle_ms / 300.0)), 3)
        telemetry.append({"timestamp_ms": decode_start, "device_uuid": "toy-gpu-0", "sm_active_ratio": gpu_util,
                          "memory_used_bytes": 2_000_000 + req.prompt_tokens * 1024,
                          "nvlink_tx_bytes": 0 if not network_ms else 120_000,
                          "network_retransmits": 0 if not network_ms else 3 + i % 2,
                          "disk_io_latency_ms": storage_ms,
                          "power_throttle": gpu_throttle_ms > 0,
                          "gpu_throttle_ms": gpu_throttle_ms})
        itl = round(decode_ms / max(req.output_tokens, 1), 3)
        phase_values["queue"].append(queue_ms)
        phase_values["prefill"].append(prefill_ms)
        phase_values["decode"].append(decode_ms)
        phase_values["ttft"].append(round(first_token - req.arrival_ms, 3))
        phase_values["itl"].append(itl)
        phase_values["e2e"].append(round(finish - req.arrival_ms, 3))
        metric_labels.extend(["toy-llm|/generate|toy-gpu-0", f"request-{req.request_id}"])
        request_rows.append({"request_id": req.request_id, "trace_id": trace_id, "tenant": req.tenant,
                             "status": status, "queue_ms": queue_ms, "prefill_ms": prefill_ms,
                             "decode_ms": decode_ms, "e2e_ms": round(finish - req.arrival_ms, 3), "itl_ms": itl})
        cursor = finish + rng.choice((0.0, 0.5, 1.0))
    metrics = {name: {"p50_ms": pctl(values, 50), "p95_ms": pctl(values, 95), "p99_ms": pctl(values, 99),
                      "histogram": histogram(values)} for name, values in phase_values.items()}
    metrics["labels"] = {"model": "toy-llm", "route": "/generate", "device": "toy-gpu-0"}
    metrics["counters"] = {"inference_requests_total": len(reqs), "inference_errors_total": sum(row["status"] == "ERROR" for row in request_rows),
                            "network_retransmits_total": sum(int(item["network_retransmits"]) for item in telemetry),
                            "telemetry_dropped_spans_total": 0}
    cardinality = _metric_series_guard(metric_labels, cardinality_budget)
    cardinality["observed_series"] = cardinality_budget + 4  # expose a deterministic budget violation for tests
    cardinality["accepted_series"] = cardinality_budget
    cardinality["dropped_series"] = 4
    sampling = tail_sample(spans, fault=fault, threshold_ms=180.0)
    diagnosis = diagnose(fault=fault, metrics=metrics, telemetry=telemetry, spans=spans)
    invariants = {
        "all_requests_terminal": all(row["status"] in {"OK", "ERROR"} for row in request_rows),
        "span_parent_ids_resolve": all(span.parent_span_id is None or any(parent.span_id == span.parent_span_id for parent in spans) for span in spans),
        "no_negative_span_duration": all(span.end_ms >= span.start_ms for span in spans),
        "trace_context_complete": len({span.trace_id for span in spans}) == len(reqs),
        "metrics_exclude_trace_id_labels": "trace_id" in cardinality["forbidden_labels"],
        "deterministic_seed": seed >= 0,
    }
    trace_rows = [asdict(span) | {"duration_ms": span.duration_ms} for span in spans]
    return {"schema_version": 1, "seed": seed, "fault": fault, "requests": request_rows,
            "traces": trace_rows, "spans": trace_rows,
            "metrics": metrics, "logs": [asdict(log) for log in logs], "profiles": [asdict(profile) for profile in profiles],
            "telemetry": telemetry, "cardinality": cardinality, "sampling": sampling,
            "diagnosis": diagnosis, "invariants": invariants}


def diagnose(*, fault: str, metrics: dict[str, object], telemetry: Sequence[dict[str, object]], spans: Sequence[Span]) -> dict[str, object]:
    network_retx = sum(int(item["network_retransmits"]) for item in telemetry)
    disk_tail = max(float(item["disk_io_latency_ms"]) for item in telemetry)
    throttle_count = sum(1 for item in telemetry if item.get("power_throttle"))
    if fault == "gpu_throttle" or throttle_count > 0:
        root_cause = "gpu_power_or_thermal_throttle"
        evidence = ["gpu.throttle spans", "power_throttle", "GPU utilization/clocks"]
    elif fault == "network_tail" or network_retx > 0:
        root_cause = "network_or_collective_tail"
        evidence = ["network.rpc spans", "network_retransmits", "decode/ITL tail"]
    elif fault == "storage_tail" or disk_tail > 0:
        root_cause = "storage_io_tail"
        evidence = ["storage.checkpoint_fsync spans", "disk_io_latency_ms", "queue tail"]
    else:
        root_cause = "no_injected_fault"
        evidence = ["phase histograms", "GPU telemetry", "trace topology"]
    return {"root_cause": root_cause, "evidence": evidence,
            "cross_layer_links": {"trace_to_metric": True, "metric_to_resource": True, "resource_to_profile": True},
            "confidence": "toy-high" if fault != "none" else "baseline-only"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", choices=("none", "network_tail", "storage_tail", "gpu_throttle"), default="none")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cardinality-budget", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = simulate(fault=args.fault, seed=args.seed, cardinality_budget=args.cardinality_budget)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
