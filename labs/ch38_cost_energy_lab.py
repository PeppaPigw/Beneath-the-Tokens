#!/usr/bin/env python3
"""CPU-only deterministic cost, capacity, energy and carbon toy lab.

This module intentionally uses synthetic inputs and standard-library arithmetic.
It does not call a cloud API, inspect a GPU, buy capacity, or mutate a
scheduler.  The output is evidence for the chapter's formulas and contracts,
not a production billing, power-meter, carbon-accounting or SLO guarantee.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
FAULTS = {
    "none",
    "underprovisioned",
    "spot_interruption",
    "thermal_limit",
    "carbon_miss",
    "slo_violation",
}


@dataclass(frozen=True)
class Workload:
    name: str
    arrival_rate_rps: float
    service_rate_rps_per_replica: float
    replicas: int
    request_tokens: int
    target_utilization: float
    slo_p95_ms: float
    slo_availability: float
    deadline_seconds: float
    retry_rate: float

    def validate(self) -> None:
        if not self.name:
            raise ValueError("workload name must be non-empty")
        if self.arrival_rate_rps < 0 or self.service_rate_rps_per_replica <= 0:
            raise ValueError("arrival must be >=0 and service rate must be >0")
        if self.replicas <= 0:
            raise ValueError("replicas must be positive")
        if not 0 < self.target_utilization < 1:
            raise ValueError("target_utilization must be in (0,1)")
        if self.request_tokens <= 0 or self.slo_p95_ms <= 0 or self.deadline_seconds <= 0:
            raise ValueError("tokens, SLO latency and deadline must be positive")
        if not 0 <= self.slo_availability <= 1:
            raise ValueError("slo_availability must be in [0,1]")
        if not 0 <= self.retry_rate < 1:
            raise ValueError("retry_rate must be in [0,1)")


@dataclass(frozen=True)
class Pricing:
    on_demand_usd_per_gpu_hour: float
    reserved_discount: float
    spot_discount: float
    reserved_fraction: float
    spot_fraction: float
    interruption_rate: float
    recovery_hours: float
    storage_usd: float
    network_usd: float
    platform_usd: float

    def validate(self) -> None:
        if self.on_demand_usd_per_gpu_hour <= 0:
            raise ValueError("on-demand price must be positive")
        for name, value in (("reserved_discount", self.reserved_discount), ("spot_discount", self.spot_discount),
                            ("reserved_fraction", self.reserved_fraction), ("spot_fraction", self.spot_fraction),
                            ("interruption_rate", self.interruption_rate)):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0,1)")
        if self.reserved_fraction + self.spot_fraction > 1:
            raise ValueError("reserved and spot fractions cannot exceed 1")
        if self.recovery_hours < 0 or min(self.storage_usd, self.network_usd, self.platform_usd) < 0:
            raise ValueError("costs and recovery_hours must be non-negative")


@dataclass(frozen=True)
class EnergyModel:
    gpu_power_w: float
    host_overhead_w_per_replica: float
    pue: float
    carbon_factor_kg_per_kwh: float
    carbon_budget_kg: float
    max_node_power_w: float

    def validate(self) -> None:
        if self.gpu_power_w <= 0 or self.host_overhead_w_per_replica < 0:
            raise ValueError("power values are invalid")
        if self.pue < 1 or self.carbon_factor_kg_per_kwh < 0 or self.carbon_budget_kg <= 0:
            raise ValueError("pue/carbon budget are invalid")
        if self.max_node_power_w <= 0:
            raise ValueError("max node power must be positive")


def build_fixture() -> dict[str, Any]:
    """Return a small deterministic workload and pricing/energy fixture."""
    return {
        "workload": Workload(
            name="toy-online-inference", arrival_rate_rps=70.0,
            service_rate_rps_per_replica=50.0, replicas=2, request_tokens=900,
            target_utilization=0.75, slo_p95_ms=800.0, slo_availability=0.99,
            deadline_seconds=60.0, retry_rate=0.01,
        ),
        "pricing": Pricing(
            on_demand_usd_per_gpu_hour=2.40, reserved_discount=0.35,
            spot_discount=0.70, reserved_fraction=0.60, spot_fraction=0.25,
            interruption_rate=0.08, recovery_hours=0.20, storage_usd=8.0,
            network_usd=6.0, platform_usd=14.0,
        ),
        "energy": EnergyModel(
            gpu_power_w=350.0, host_overhead_w_per_replica=120.0, pue=1.25,
            carbon_factor_kg_per_kwh=0.35, carbon_budget_kg=80.0,
            max_node_power_w=1600.0,
        ),
        "horizon_hours": 2.0,
        "training_gpu_hours": 96.0,
        "inference_hours": 2.0,
    }


def plan_capacity(workload: Workload, *, min_replicas: int = 1, max_replicas: int = 16,
                  headroom: float = 0.0) -> dict[str, Any]:
    """Plan replicas from arrival/service rate and a target utilization."""
    workload.validate()
    if min_replicas <= 0 or max_replicas < min_replicas:
        raise ValueError("invalid replica bounds")
    effective_target = workload.target_utilization * (1 - headroom)
    if not 0 < effective_target < 1:
        raise ValueError("headroom makes target utilization invalid")
    raw = workload.arrival_rate_rps / (workload.service_rate_rps_per_replica * effective_target)
    planned = max(min_replicas, math.ceil(raw - 1e-12))
    capped = min(planned, max_replicas)
    return {
        "raw_replicas": round(raw, 6),
        "planned_replicas": planned,
        "replicas": capped,
        "max_replicas": max_replicas,
        "headroom": headroom,
        "capacity_shortfall": planned > max_replicas,
    }


def queue_model(workload: Workload, *, replicas: int | None = None,
                failure_rate: float = 0.0) -> dict[str, Any]:
    """Return a stable queue approximation; this is not a full M/M/c solver."""
    workload.validate()
    if failure_rate < 0 or failure_rate >= 1:
        raise ValueError("failure_rate must be in [0,1)")
    count = workload.replicas if replicas is None else replicas
    if count <= 0:
        raise ValueError("replicas must be positive")
    capacity = count * workload.service_rate_rps_per_replica
    rho = workload.arrival_rate_rps / capacity
    overflow = max(0.0, workload.arrival_rate_rps - capacity)
    # Smooth finite approximation, intentionally conservative as rho approaches 1.
    if rho < 1:
        queue_wait_ms = 20.0 + 200.0 * rho / max(1e-6, 1.0 - rho)
    else:
        queue_wait_ms = 20.0 + 200.0 * (10.0 + 25.0 * (rho - 1.0))
    queue_wait_ms += 500.0 * overflow / max(1.0, workload.arrival_rate_rps)
    effective_rate = min(workload.arrival_rate_rps, capacity) * (1 - failure_rate)
    effective_tokens_per_second = effective_rate * workload.request_tokens
    service_success_rate = effective_rate / max(1e-9, workload.arrival_rate_rps)
    device_utilization = min(0.99, rho + workload.retry_rate * 0.5)
    effective_utilization = min(1.0, service_success_rate * (1 - workload.retry_rate))
    deadline_miss = min(1.0, max(0.0, queue_wait_ms / 1000.0 / workload.deadline_seconds))
    availability = max(0.0, min(1.0, service_success_rate * (1 - deadline_miss)))
    return {
        "replicas": count,
        "capacity_rps": round(capacity, 6),
        "arrival_rate_rps": round(workload.arrival_rate_rps, 6),
        "rho": round(rho, 6),
        "overflow_rps": round(overflow, 6),
        "p95_wait_ms": round(queue_wait_ms, 3),
        "effective_rate_rps": round(effective_rate, 6),
        "effective_tokens_per_second": round(effective_tokens_per_second, 3),
        "service_success_rate": round(service_success_rate, 6),
        "device_utilization": round(device_utilization, 6),
        "effective_utilization": round(effective_utilization, 6),
        "deadline_miss_rate": round(deadline_miss, 6),
        "availability": round(availability, 6),
        "failure_rate": round(failure_rate, 6),
    }


def procurement_cost(pricing: Pricing, *, gpu_hours: float, failure_rate: float = 0.0) -> dict[str, Any]:
    """Compare reserved, spot and on-demand expected spend for GPU hours."""
    pricing.validate()
    if gpu_hours < 0 or not 0 <= failure_rate < 1:
        raise ValueError("gpu_hours must be non-negative and failure_rate in [0,1)")
    reserved_rate = pricing.on_demand_usd_per_gpu_hour * (1 - pricing.reserved_discount)
    spot_rate = pricing.on_demand_usd_per_gpu_hour * (1 - pricing.spot_discount)
    ondemand_fraction = 1 - pricing.reserved_fraction - pricing.spot_fraction
    base = gpu_hours * (
        pricing.reserved_fraction * reserved_rate
        + pricing.spot_fraction * spot_rate
        + ondemand_fraction * pricing.on_demand_usd_per_gpu_hour
    )
    recovery = gpu_hours * pricing.spot_fraction * max(pricing.interruption_rate, failure_rate) * pricing.recovery_hours * pricing.on_demand_usd_per_gpu_hour
    successful_gpu_hours = gpu_hours * (1 - pricing.spot_fraction * max(pricing.interruption_rate, failure_rate))
    expected = base + recovery
    return {
        "gpu_hours_requested": round(gpu_hours, 6),
        "successful_gpu_hours_expected": round(successful_gpu_hours, 6),
        "reserved_rate_usd": round(reserved_rate, 6),
        "spot_rate_usd": round(spot_rate, 6),
        "on_demand_rate_usd": round(pricing.on_demand_usd_per_gpu_hour, 6),
        "expected_compute_usd": round(expected, 6),
        "recovery_usd": round(recovery, 6),
        "effective_usd_per_successful_gpu_hour": round(expected / max(1e-9, successful_gpu_hours), 6),
        "capacity_mix": {"reserved": pricing.reserved_fraction, "spot": pricing.spot_fraction, "on_demand": ondemand_fraction},
        "failure_rate": round(failure_rate, 6),
    }


def energy_carbon(model: EnergyModel, *, replicas: int, hours: float) -> dict[str, Any]:
    """Compute synthetic IT/시설 energy, peak power and operational carbon."""
    model.validate()
    if replicas <= 0 or hours <= 0:
        raise ValueError("replicas and hours must be positive")
    it_power_w = replicas * (model.gpu_power_w + model.host_overhead_w_per_replica)
    it_kwh = it_power_w * hours / 1000.0
    facility_kwh = it_kwh * model.pue
    carbon = facility_kwh * model.carbon_factor_kg_per_kwh
    return {
        "replicas": replicas,
        "hours": round(hours, 6),
        "it_power_w": round(it_power_w, 6),
        "peak_power_w": round(it_power_w, 6),
        "it_kwh": round(it_kwh, 6),
        "pue": round(model.pue, 6),
        "facility_kwh": round(facility_kwh, 6),
        "carbon_factor_kg_per_kwh": round(model.carbon_factor_kg_per_kwh, 6),
        "carbon_kg": round(carbon, 6),
        "carbon_budget_kg": round(model.carbon_budget_kg, 6),
        "carbon_budget_headroom_kg": round(model.carbon_budget_kg - carbon, 6),
    }


def tco_summary(*, compute_usd: float, pricing: Pricing, energy: Mapping[str, Any],
                training_gpu_hours: float, successful_steps: int = 1000,
                inference_requests: int = 100_000) -> dict[str, Any]:
    """Produce training/inference unit economics from the same evidence inputs."""
    pricing.validate()
    if compute_usd < 0 or training_gpu_hours <= 0 or successful_steps <= 0 or inference_requests <= 0:
        raise ValueError("invalid TCO inputs")
    shared = pricing.storage_usd + pricing.network_usd + pricing.platform_usd
    training_total = compute_usd + shared
    # Inference is a toy allocation: shared costs plus active compute over the horizon.
    inference_total = compute_usd * 0.35 + shared
    return {
        "training": {
            "total_usd": round(training_total, 6),
            "successful_steps": successful_steps,
            "usd_per_successful_step": round(training_total / successful_steps, 9),
            "usd_per_gpu_hour_effective": round(training_total / training_gpu_hours, 6),
        },
        "inference": {
            "total_usd": round(inference_total, 6),
            "slo_compliant_requests": inference_requests,
            "usd_per_slo_request": round(inference_total / inference_requests, 9),
            "kg_per_million_requests": round(float(energy["carbon_kg"]) * 1_000_000 / inference_requests, 6),
        },
        "shared_cost_usd": round(shared, 6),
    }


def finops_contract(*, queue: Mapping[str, Any], procurement: Mapping[str, Any], energy: Mapping[str, Any],
                    tags_complete: bool = True) -> dict[str, Any]:
    """Return explainable FinOps facts and a guardrail decision."""
    if not tags_complete:
        return {"tags_complete": False, "unallocated": True, "budget_action": "hold_and_repair_tags"}
    anomaly = float(procurement["expected_compute_usd"]) > 500.0
    return {
        "tags_complete": True,
        "unallocated": False,
        "budget_action": "investigate_spend_anomaly" if anomaly else "within_fixture_budget",
        "unit": "usd_per_slo_compliant_request_and_kg_per_request",
        "device_utilization": queue["device_utilization"],
        "effective_utilization": queue["effective_utilization"],
        "carbon_kg": energy["carbon_kg"],
    }


def simulate(*, fault: str = "none") -> dict[str, Any]:
    """Run deterministic baseline or one named failure scenario."""
    if fault not in FAULTS:
        raise ValueError(f"fault must be one of: {', '.join(sorted(FAULTS))}")
    fixture = build_fixture()
    workload: Workload = fixture["workload"]
    pricing: Pricing = fixture["pricing"]
    energy: EnergyModel = fixture["energy"]
    horizon_hours = float(fixture["horizon_hours"])
    training_gpu_hours = float(fixture["training_gpu_hours"])
    inference_hours = float(fixture["inference_hours"])
    failure_rate = 0.0
    tags_complete = True
    if fault == "underprovisioned":
        workload = Workload(**{**asdict(workload), "replicas": 1})
    elif fault == "spot_interruption":
        failure_rate = 0.35
        pricing = Pricing(**{**asdict(pricing), "interruption_rate": 0.35})
    elif fault == "thermal_limit":
        energy = EnergyModel(**{**asdict(energy), "max_node_power_w": 600.0})
    elif fault == "carbon_miss":
        energy = EnergyModel(**{**asdict(energy), "carbon_budget_kg": 0.5, "carbon_factor_kg_per_kwh": 0.8})
    elif fault == "slo_violation":
        workload = Workload(**{**asdict(workload), "arrival_rate_rps": 140.0})
    elif fault == "none":
        pass

    workload.validate(); pricing.validate(); energy.validate()
    plan = plan_capacity(workload, min_replicas=1, max_replicas=8, headroom=0.10)
    planned_replicas = int(plan["replicas"])
    queue_initial = queue_model(workload, replicas=workload.replicas, failure_rate=failure_rate)
    planned_power = planned_replicas * (energy.gpu_power_w + energy.host_overhead_w_per_replica)
    thermal_ok = planned_power <= energy.max_node_power_w
    active_replicas = planned_replicas if thermal_ok else max(1, int(energy.max_node_power_w // (energy.gpu_power_w + energy.host_overhead_w_per_replica)))
    if fault == "underprovisioned":
        active_replicas = workload.replicas
    queue = queue_model(workload, replicas=active_replicas, failure_rate=failure_rate)
    energy_result = energy_carbon(energy, replicas=active_replicas, hours=inference_hours)
    procurement = procurement_cost(pricing, gpu_hours=training_gpu_hours * max(1, active_replicas), failure_rate=failure_rate)
    tco = tco_summary(compute_usd=procurement["expected_compute_usd"], pricing=pricing, energy=energy_result,
                      training_gpu_hours=training_gpu_hours * max(1, active_replicas),
                      successful_steps=1000 if failure_rate == 0 else 900,
                      inference_requests=int(max(1, queue["effective_rate_rps"] * inference_hours * 3600)))
    slo_ok = queue["p95_wait_ms"] <= workload.slo_p95_ms and queue["availability"] >= workload.slo_availability and queue["deadline_miss_rate"] < 0.01
    capacity_ok = queue["overflow_rps"] == 0 and not plan["capacity_shortfall"]
    spot_ok = failure_rate == 0.0 or queue["service_success_rate"] >= 0.90
    carbon_ok = energy_result["carbon_kg"] <= energy.carbon_budget_kg
    reasons: list[str] = []
    if not capacity_ok:
        reasons.append("capacity_shortfall")
    if not slo_ok:
        reasons.append("slo_violation")
    if not thermal_ok:
        reasons.append("thermal_power_limit")
    if not spot_ok:
        reasons.append("spot_interruption_loss")
    if not carbon_ok:
        reasons.append("carbon_budget_exceeded")
    if not tags_complete:
        reasons.append("finops_tags_missing")
    decision = "admit" if not reasons else "deny"
    finops = finops_contract(queue=queue, procurement=procurement, energy=energy_result, tags_complete=tags_complete)
    invariants = {
        "deterministic_fixture": True,
        "cpu_only": True,
        "no_external_side_effects": True,
        "effective_utilization_not_above_one": queue["effective_utilization"] <= 1.0,
        "carbon_formula_consistent": abs(energy_result["carbon_kg"] - energy_result["facility_kwh"] * energy_result["carbon_factor_kg_per_kwh"]) < 1e-6,
        "fail_closed_on_constraint": (decision == "admit") or bool(reasons),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "fault": fault,
        "decision": decision,
        "capacity": {"plan": plan, "active_replicas": active_replicas, "planned_power_w": round(planned_power, 6), "thermal_ok": thermal_ok},
        "queue": queue,
        "procurement": procurement,
        "energy": energy_result,
        "tco": tco,
        "slo": {"p95_limit_ms": workload.slo_p95_ms, "availability_limit": workload.slo_availability, "ok": slo_ok},
        "finops": finops,
        "deny_reasons": reasons,
        "invariants": invariants,
        "assumptions": {
            "horizon_hours": horizon_hours,
            "training_gpu_hours": training_gpu_hours,
            "formula_note": "synthetic rates, PUE and carbon factor; not a production meter or invoice",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", choices=sorted(FAULTS), default="none")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = simulate(fault=args.fault)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
