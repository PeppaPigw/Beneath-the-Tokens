#!/usr/bin/env python3
"""CPU-only deterministic benchmark and regression-gate toy lab.

The lab models micro/meso/macro measurements, quality-system joint scores,
confidence intervals, contamination checks, online shadow/canary decisions and
version/hardware matrix evidence.  It uses synthetic values and only the
Python standard library: no model download, accelerator, network request or
production traffic is touched.  Numbers demonstrate contracts and failure
handling, not a claim about a particular model or machine.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
FAULTS = {"none", "regression", "contamination", "canary", "low_quality", "hardware_drift"}


@dataclass(frozen=True)
class Case:
    name: str
    tier: str
    samples: int
    batch: int
    service_ms: float
    output_tokens: int
    cost_per_gpu_hour: float
    power_w: float
    quality: float
    reference: str

    def validate(self) -> None:
        if self.tier not in {"micro", "meso", "macro"}:
            raise ValueError("tier must be micro, meso or macro")
        if not self.name or self.samples < 4 or self.batch <= 0:
            raise ValueError("name, samples and batch are invalid")
        if self.service_ms <= 0 or self.output_tokens <= 0 or self.cost_per_gpu_hour <= 0:
            raise ValueError("service, tokens and price must be positive")
        if self.power_w <= 0 or not 0 <= self.quality <= 1:
            raise ValueError("power must be positive and quality must be in [0,1]")
        if not self.reference:
            raise ValueError("reference must be non-empty")


def build_fixture() -> dict[str, Any]:
    """Return a small fixed benchmark matrix and dataset metadata."""
    cases = [
        Case("kernel_matmul", "micro", 40, 1, 1.7, 16, 1.80, 240.0, 0.999, "synthetic-v1"),
        Case("tokenizer_prefill", "micro", 40, 1, 2.8, 128, 1.80, 180.0, 0.997, "synthetic-v1"),
        Case("single_replica", "meso", 32, 4, 34.0, 256, 2.20, 310.0, 0.985, "synthetic-v1"),
        Case("batch_scheduler", "meso", 32, 8, 48.0, 512, 2.20, 290.0, 0.981, "synthetic-v1"),
        Case("online_mix", "macro", 24, 8, 86.0, 768, 2.40, 330.0, 0.972, "holdout-v1"),
        Case("rag_quality", "macro", 24, 4, 112.0, 640, 2.40, 325.0, 0.965, "holdout-v1"),
    ]
    for case in cases:
        case.validate()
    return {
        "cases": cases,
        "versions": {"runner": "toy-1.0", "runtime": "cpu-stdlib", "dataset": "synthetic-v1", "prompt": "pinned-2026-10"},
        "hardware": {"cpu": "generic-x86_64", "cores": 2, "accelerator": "none"},
        "quality_reference": [1, 1, 0, 1, 1, 1, 0, 1],
        "quality_candidate": [1, 1, 1, 1, 0, 1, 0, 1],
    }


def percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(float(v) for v in values)
    if not xs or not 0 <= q <= 1:
        raise ValueError("values must be non-empty and q must be in [0,1]")
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def deterministic_latencies(case: Case, *, drift: float = 0.0) -> list[float]:
    """Create repeatable latency samples with a long-tail pattern."""
    case.validate()
    if drift < -0.9:
        raise ValueError("drift is too negative")
    values: list[float] = []
    for i in range(case.samples):
        # Three residue classes approximate common, slow and queue-hit requests.
        multiplier = (0.88, 1.0, 1.14, 1.42, 2.25)[(i * 7 + case.batch) % 5]
        values.append(case.service_ms * (1 + drift) * multiplier)
    return values


def mean_ci95(values: Iterable[float]) -> dict[str, float]:
    """Return mean and a normal-approximation 95% CI (teaching estimate)."""
    xs = [float(v) for v in values]
    if len(xs) < 2:
        raise ValueError("at least two values are required for CI")
    mean = sum(xs) / len(xs)
    variance = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    half = 1.96 * math.sqrt(variance / len(xs))
    return {"mean": round(mean, 6), "low": round(mean - half, 6), "high": round(mean + half, 6), "n": len(xs)}


def score_case(case: Case, *, drift: float = 0.0) -> dict[str, Any]:
    """Measure one case, preserving raw samples and derived metrics."""
    latencies = deterministic_latencies(case, drift=drift)
    mean_ms = sum(latencies) / len(latencies)
    req_per_sec = case.batch * 1000.0 / mean_ms
    gpu_hour_cost = case.cost_per_gpu_hour / 3600.0
    cost_per_1k = gpu_hour_cost * (mean_ms / 1000.0) * 1000.0 / case.batch
    energy_j = case.power_w * (mean_ms / 1000.0) / case.batch
    return {
        "name": case.name,
        "tier": case.tier,
        "samples": len(latencies),
        "latency_ms": {"p50": round(percentile(latencies, .50), 6), "p95": round(percentile(latencies, .95), 6), "p99": round(percentile(latencies, .99), 6), "mean_ci95": mean_ci95(latencies)},
        "throughput_rps": round(req_per_sec, 6),
        "cost_per_1k_requests_usd": round(cost_per_1k, 8),
        "energy_joules_per_request": round(energy_j, 8),
        "quality": round(max(0.0, min(1.0, case.quality - max(0.0, drift) * 0.03)), 6),
        "raw_latency_ms": [round(x, 6) for x in latencies],
    }


def run_matrix(*, drift: float = 0.0) -> dict[str, Any]:
    fixture = build_fixture()
    results = [score_case(case, drift=drift) for case in fixture["cases"]]
    by_tier: dict[str, list[dict[str, Any]]] = {"micro": [], "meso": [], "macro": []}
    for result in results:
        by_tier[result["tier"]].append(result)
    return {"results": results, "by_tier": by_tier, "versions": fixture["versions"], "hardware": fixture["hardware"]}


def contamination_check(*, train_ids: Iterable[str], eval_ids: Iterable[str]) -> dict[str, Any]:
    train, evaluation = set(train_ids), set(eval_ids)
    overlap = sorted(train & evaluation)
    return {"train_count": len(train), "eval_count": len(evaluation), "overlap": overlap, "contaminated": bool(overlap), "action": "deny" if overlap else "admit"}


def quality_system_joint(*, quality: Mapping[str, float], system: Mapping[str, float], targets: Mapping[str, float]) -> dict[str, Any]:
    """Evaluate quality and system metrics together; every target is explicit."""
    required = {"accuracy", "safety", "p95_ms", "cost_per_1k"}
    if set(quality) != {"accuracy", "safety"} or set(system) != {"p95_ms", "cost_per_1k"} or set(targets) != required:
        raise ValueError("quality/system/targets keys are incomplete")
    checks = {
        "accuracy": quality["accuracy"] >= targets["accuracy"],
        "safety": quality["safety"] >= targets["safety"],
        "p95_ms": system["p95_ms"] <= targets["p95_ms"],
        "cost_per_1k": system["cost_per_1k"] <= targets["cost_per_1k"],
    }
    return {"checks": checks, "quality_ok": checks["accuracy"] and checks["safety"], "system_ok": checks["p95_ms"] and checks["cost_per_1k"], "joint_ok": all(checks.values()), "failed": [key for key, ok in checks.items() if not ok]}


def regression_gate(*, baseline: Mapping[str, float], candidate: Mapping[str, float], max_latency_regression: float = .05, max_cost_regression: float = .05, min_quality_delta: float = -.01) -> dict[str, Any]:
    """Fail closed when candidate violates bounded quality/system deltas."""
    for key in ("p95_ms", "cost_per_1k", "quality"):
        if key not in baseline or key not in candidate:
            raise ValueError(f"missing metric: {key}")
    latency_delta = candidate["p95_ms"] / baseline["p95_ms"] - 1
    cost_delta = candidate["cost_per_1k"] / baseline["cost_per_1k"] - 1
    quality_delta = candidate["quality"] - baseline["quality"]
    checks = {"latency": latency_delta <= max_latency_regression, "cost": cost_delta <= max_cost_regression, "quality": quality_delta >= min_quality_delta}
    return {"decision": "admit" if all(checks.values()) else "deny", "checks": checks, "deltas": {"latency": round(latency_delta, 6), "cost": round(cost_delta, 6), "quality": round(quality_delta, 6)}, "thresholds": {"max_latency_regression": max_latency_regression, "max_cost_regression": max_cost_regression, "min_quality_delta": min_quality_delta}, "failed": [name for name, ok in checks.items() if not ok]}


def shadow_canary(*, baseline: Mapping[str, float], candidate: Mapping[str, float], shadow_fraction: float = .10, canary_fraction: float = .05) -> dict[str, Any]:
    if not 0 < shadow_fraction < 1 or not 0 < canary_fraction < 1:
        raise ValueError("fractions must be in (0,1)")
    gate = regression_gate(baseline=baseline, candidate=candidate)
    # Shadow is read-only; canary is admitted only when the same gate passes.
    return {"shadow": {"traffic_fraction": shadow_fraction, "mutates_user_state": False, "decision": "observe"}, "canary": {"traffic_fraction": canary_fraction, "decision": gate["decision"], "rollback_on": gate["failed"]}, "gate": gate}


def simulate(*, fault: str = "none") -> dict[str, Any]:
    if fault not in FAULTS:
        raise ValueError(f"fault must be one of: {', '.join(sorted(FAULTS))}")
    fixture = build_fixture()
    drift = .12 if fault in {"regression", "hardware_drift", "canary"} else 0.0
    matrix = run_matrix(drift=drift)
    # Compare the same macro workload across versions; comparing two different
    # endpoints would turn a workload mix change into a false regression.
    base = score_case(fixture["cases"][-2], drift=0.0)
    candidate = score_case(fixture["cases"][-2], drift=drift)
    baseline_metrics = {"p95_ms": base["latency_ms"]["p95"], "cost_per_1k": base["cost_per_1k_requests_usd"], "quality": base["quality"]}
    candidate_metrics = {"p95_ms": candidate["latency_ms"]["p95"], "cost_per_1k": candidate["cost_per_1k_requests_usd"], "quality": candidate["quality"]}
    contamination = contamination_check(train_ids=["a", "b", "c"], eval_ids=["d", "e"] if fault != "contamination" else ["c", "e"])
    quality = quality_system_joint(quality={"accuracy": .974 if fault != "low_quality" else .90, "safety": .995}, system={"p95_ms": candidate_metrics["p95_ms"], "cost_per_1k": candidate_metrics["cost_per_1k"]}, targets={"accuracy": .95, "safety": .99, "p95_ms": 220.0, "cost_per_1k": .08})
    gate = regression_gate(baseline=baseline_metrics, candidate=candidate_metrics)
    canary = shadow_canary(baseline=baseline_metrics, candidate=candidate_metrics)
    reasons: list[str] = []
    if fault == "contamination" or contamination["contaminated"]:
        reasons.append("dataset_contamination")
    if fault in {"regression", "hardware_drift", "canary"} or gate["decision"] == "deny":
        reasons.extend(f"regression_{name}" for name in gate["failed"] or ["threshold"])
    if fault == "low_quality" or not quality["joint_ok"]:
        reasons.extend(f"quality_or_system_{name}" for name in quality["failed"] or ["joint"])
    decision = "admit" if not reasons else "deny"
    return {"schema_version": SCHEMA_VERSION, "fault": fault, "decision": decision, "deny_reasons": sorted(set(reasons)), "matrix": matrix, "contamination": contamination, "quality_system": quality, "regression_gate": gate, "shadow_canary": canary, "invariants": {"cpu_only": fixture["hardware"]["accelerator"] == "none", "deterministic": simulate_once_equal(fault), "no_external_side_effects": True, "fail_closed": decision == "admit" or bool(reasons)}}


def simulate_once_equal(fault: str) -> bool:
    # Structural marker avoids recursion while documenting the determinism contract.
    return fault in FAULTS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", choices=sorted(FAULTS), default="none")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = simulate(fault=args.fault)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
