#!/usr/bin/env python3
"""CPU-only deterministic AI Infra production-capstone toy lab.

This lab turns a service contract into a capacity plan, SLO/error-budget
check, rollout decision, fault-injection record and evidence manifest.  It is
an executable teaching model: all values are synthetic, standard-library
only, and no accelerator, network, cloud billing system or user data is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
FAULTS = {"none", "traffic_spike", "gpu_loss", "network_partition", "schema_drift", "budget_breach", "rollback"}


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    peak_qps: float
    per_node_qps: float
    node_count: int
    headroom: float
    p95_target_ms: float
    availability_target: float
    monthly_budget_usd: float
    node_monthly_usd: float
    energy_wh_per_request: float

    def validate(self) -> None:
        if not self.name or self.peak_qps <= 0 or self.per_node_qps <= 0:
            raise ValueError("name and rates must be positive")
        if self.node_count < 1 or not 0 <= self.headroom < 1:
            raise ValueError("node_count/headroom invalid")
        if self.p95_target_ms <= 0 or not 0 < self.availability_target <= 1:
            raise ValueError("SLO targets invalid")
        if self.monthly_budget_usd <= 0 or self.node_monthly_usd <= 0 or self.energy_wh_per_request <= 0:
            raise ValueError("budget and energy must be positive")


def build_fixture() -> dict[str, Any]:
    spec = ServiceSpec(
        name="summarize-v1", peak_qps=60.0, per_node_qps=24.0, node_count=4,
        headroom=0.30, p95_target_ms=300.0, availability_target=0.995,
        monthly_budget_usd=12_000.0, node_monthly_usd=2_100.0,
        energy_wh_per_request=1.8,
    )
    spec.validate()
    return {
        "spec": spec,
        "versions": {"platform": "capstone-toy-1.0", "model": "model-v7", "schema": "request-v3", "runner": "stdlib"},
        "hardware": {"cpu": "generic-x86_64", "accelerator": "none", "nodes": spec.node_count},
        "owners": {"service": "inference-platform", "oncall": "ai-infra-primary", "data": "evaluation-owner"},
    }


def capacity_plan(spec: ServiceSpec, *, peak_qps: float | None = None, node_count: int | None = None) -> dict[str, Any]:
    spec.validate()
    demand = spec.peak_qps if peak_qps is None else float(peak_qps)
    nodes = spec.node_count if node_count is None else int(node_count)
    if demand <= 0 or nodes < 1:
        raise ValueError("demand and nodes must be positive")
    protected_demand = demand * (1 + spec.headroom)
    required = math.ceil(protected_demand / spec.per_node_qps)
    utilization = demand / (nodes * spec.per_node_qps)
    return {
        "peak_qps": round(demand, 6), "headroom": spec.headroom,
        "protected_qps": round(protected_demand, 6), "required_nodes": required,
        "provisioned_nodes": nodes, "utilization": round(utilization, 6),
        "capacity_ok": nodes >= required and utilization <= (1 - spec.headroom),
        "scale_action": "hold" if nodes >= required else f"add_{required - nodes}_nodes",
    }


def slo_budget(spec: ServiceSpec, *, observed: Mapping[str, float]) -> dict[str, Any]:
    spec.validate()
    required = {"p95_ms", "availability", "monthly_requests"}
    if set(observed) != required:
        raise ValueError(f"observed keys must be {sorted(required)}")
    if observed["monthly_requests"] <= 0:
        raise ValueError("monthly_requests must be positive")
    checks = {
        "latency": observed["p95_ms"] <= spec.p95_target_ms,
        "availability": observed["availability"] >= spec.availability_target,
    }
    total_budget_requests = observed["monthly_requests"] * (1 - spec.availability_target)
    used_error_requests = observed["monthly_requests"] * (1 - observed["availability"])
    remaining = total_budget_requests - used_error_requests
    return {
        "targets": {"p95_ms": spec.p95_target_ms, "availability": spec.availability_target},
        "observed": dict(observed), "checks": checks,
        "error_budget_requests": round(total_budget_requests, 3),
        "used_error_requests": round(used_error_requests, 3),
        "remaining_error_budget": round(remaining, 3),
        "slo_ok": all(checks.values()),
    }


def cost_energy(spec: ServiceSpec, *, nodes: int, monthly_requests: float) -> dict[str, Any]:
    if nodes < 1 or monthly_requests <= 0:
        raise ValueError("nodes and requests must be positive")
    monthly_cost = nodes * spec.node_monthly_usd
    energy_kwh = monthly_requests * spec.energy_wh_per_request / 1000.0
    return {"monthly_node_cost_usd": round(monthly_cost, 2), "monthly_energy_kwh": round(energy_kwh, 3),
            "budget_ok": monthly_cost <= spec.monthly_budget_usd}


def rollout_gate(*, baseline: Mapping[str, float], canary: Mapping[str, float], limits: Mapping[str, float] | None = None) -> dict[str, Any]:
    limits = dict(limits or {"max_p95_regression": 0.10, "max_error_rate": 0.005, "min_quality": 0.95})
    for key in ("p95_ms", "error_rate", "quality"):
        if key not in baseline or key not in canary:
            raise ValueError(f"missing metric: {key}")
    deltas = {"p95_regression": canary["p95_ms"] / baseline["p95_ms"] - 1,
              "error_rate": canary["error_rate"], "quality": canary["quality"]}
    checks = {"latency": deltas["p95_regression"] <= limits["max_p95_regression"],
              "errors": deltas["error_rate"] <= limits["max_error_rate"],
              "quality": deltas["quality"] >= limits["min_quality"]}
    return {"decision": "promote" if all(checks.values()) else "rollback", "checks": checks,
            "deltas": {k: round(v, 6) for k, v in deltas.items()}, "limits": limits,
            "rollback_reasons": [name for name, ok in checks.items() if not ok]}


def evidence_manifest(result: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"schema_version": SCHEMA_VERSION, "manifest_sha256": digest,
            "required_artifacts": ["request_contract", "capacity_plan", "slo_snapshot", "rollout_gate", "fault_record", "owner_ack"],
            "reproducible": True, "external_side_effects": False}


def simulate(*, fault: str = "none") -> dict[str, Any]:
    if fault not in FAULTS:
        raise ValueError(f"fault must be one of: {', '.join(sorted(FAULTS))}")
    fixture = build_fixture(); spec: ServiceSpec = fixture["spec"]
    peak = spec.peak_qps * (1.8 if fault == "traffic_spike" else 1.0)
    nodes = spec.node_count - (1 if fault == "gpu_loss" else 0)
    plan = capacity_plan(spec, peak_qps=peak, node_count=max(1, nodes))
    baseline = {"p95_ms": 220.0, "error_rate": .001, "quality": .972}
    canary = dict(baseline)
    if fault in {"network_partition", "rollback"}: canary.update(p95_ms=420.0, error_rate=.025)
    if fault == "schema_drift": canary.update(quality=.70, error_rate=.010)
    if fault == "budget_breach": canary.update(p95_ms=280.0)
    rollout = rollout_gate(baseline=baseline, canary=canary)
    availability = .999 if fault not in {"network_partition", "schema_drift"} else .970
    p95 = 240.0 if fault not in {"traffic_spike", "gpu_loss"} else 390.0
    observed = {"p95_ms": p95, "availability": availability, "monthly_requests": 2_000_000.0}
    slo = slo_budget(spec, observed=observed)
    cost = cost_energy(spec, nodes=max(1, nodes), monthly_requests=observed["monthly_requests"])
    if fault == "budget_breach": cost["budget_ok"] = False; cost["monthly_node_cost_usd"] = spec.monthly_budget_usd * 1.2
    reasons: list[str] = []
    if not plan["capacity_ok"]: reasons.append("capacity_insufficient")
    if not slo["slo_ok"]: reasons.extend(f"slo_{k}" for k, ok in slo["checks"].items() if not ok)
    if not cost["budget_ok"]: reasons.append("budget_breach")
    if rollout["decision"] == "rollback": reasons.extend(f"rollout_{x}" for x in rollout["rollback_reasons"])
    if fault == "gpu_loss": reasons.append("node_loss")
    decision = "admit" if not reasons else "deny"
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "fault": fault, "decision": decision,
        "deny_reasons": sorted(set(reasons)), "capacity": plan, "slo": slo, "cost_energy": cost,
        "rollout": rollout, "versions": fixture["versions"], "hardware": fixture["hardware"],
        "invariants": {"cpu_only": fixture["hardware"]["accelerator"] == "none", "deterministic": True, "no_external_side_effects": True, "fail_closed": decision == "admit" or bool(reasons)}}
    result["evidence"] = evidence_manifest(result)
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--fault", choices=sorted(FAULTS), default="none"); p.add_argument("--output", type=Path); return p.parse_args()


def main() -> int:
    args = _parse_args(); payload = json.dumps(simulate(fault=args.fault), ensure_ascii=False, indent=2, sort_keys=True); print(payload)
    if args.output: args.output.write_text(payload + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
