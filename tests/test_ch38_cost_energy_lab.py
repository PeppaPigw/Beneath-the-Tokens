from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch38_cost_energy_lab import (  # noqa: E402
    EnergyModel,
    Pricing,
    Workload,
    build_fixture,
    energy_carbon,
    plan_capacity,
    procurement_cost,
    queue_model,
    simulate,
)


def test_baseline_is_deterministic_cpu_only_and_admitted():
    first = simulate()
    second = simulate()
    assert first == second
    assert first["schema_version"] == 1
    assert first["decision"] == "admit"
    assert first["invariants"]["cpu_only"]
    assert first["invariants"]["carbon_formula_consistent"]
    assert first["invariants"]["no_external_side_effects"]
    assert first["slo"]["ok"]


def test_capacity_and_queue_show_utilization_trap():
    workload = build_fixture()["workload"]
    plan = plan_capacity(workload, headroom=0.10)
    assert plan["replicas"] == 3
    healthy = queue_model(workload, replicas=3)
    overloaded = queue_model(workload, replicas=1)
    assert overloaded["rho"] > 1
    assert overloaded["p95_wait_ms"] > healthy["p95_wait_ms"]
    assert overloaded["device_utilization"] >= overloaded["effective_utilization"]


def test_procurement_accounts_for_spot_interruption():
    pricing = build_fixture()["pricing"]
    base = procurement_cost(pricing, gpu_hours=100)
    interrupted = procurement_cost(pricing, gpu_hours=100, failure_rate=0.4)
    assert interrupted["expected_compute_usd"] > base["expected_compute_usd"]
    assert interrupted["successful_gpu_hours_expected"] < base["successful_gpu_hours_expected"]
    assert interrupted["capacity_mix"]["spot"] > 0


def test_energy_formula_and_thermal_carbon_faults():
    model = build_fixture()["energy"]
    energy = energy_carbon(model, replicas=2, hours=2)
    assert abs(energy["facility_kwh"] - energy["it_kwh"] * model.pue) < 1e-6
    assert abs(energy["carbon_kg"] - energy["facility_kwh"] * model.carbon_factor_kg_per_kwh) < 1e-6
    thermal = simulate(fault="thermal_limit")
    assert thermal["decision"] == "deny"
    assert "thermal_power_limit" in thermal["deny_reasons"]
    carbon = simulate(fault="carbon_miss")
    assert carbon["decision"] == "deny"
    assert "carbon_budget_exceeded" in carbon["deny_reasons"]


def test_faults_fail_closed_with_actionable_reasons():
    under = simulate(fault="underprovisioned")
    assert under["decision"] == "deny"
    assert "capacity_shortfall" in under["deny_reasons"] or "slo_violation" in under["deny_reasons"]
    spot = simulate(fault="spot_interruption")
    assert spot["decision"] == "deny"
    assert "spot_interruption_loss" in spot["deny_reasons"]
    slo = simulate(fault="slo_violation")
    assert slo["decision"] == "deny"
    assert "slo_violation" in slo["deny_reasons"]
    for result in (under, spot, slo):
        assert result["invariants"]["fail_closed_on_constraint"]


def test_input_validation():
    try:
        Workload("", 1, 1, 1, 100, 0.8, 500, 0.99, 60, 0).validate()
    except ValueError as exc:
        assert "name" in str(exc)
    else:
        raise AssertionError("empty workload name must fail")
    try:
        Pricing(2, 0.2, 0.2, 0.8, 0.4, 0.1, 0.1, 1, 1, 1).validate()
    except ValueError as exc:
        assert "fractions" in str(exc)
    else:
        raise AssertionError("capacity fractions > 1 must fail")
    try:
        simulate(fault="unknown")
    except ValueError as exc:
        assert "fault must be one of" in str(exc)
    else:
        raise AssertionError("unknown fault must fail")


def test_cli_json_stdout_matches_output_file(tmp_path: Path):
    output = tmp_path / "ch38.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch38_cost_energy_lab.py"), "--fault", "spot_interruption", "--output", str(output)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    stdout_payload = json.loads(proc.stdout)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert stdout_payload["fault"] == "spot_interruption"


if __name__ == "__main__":
    test_baseline_is_deterministic_cpu_only_and_admitted()
    test_capacity_and_queue_show_utilization_trap()
    test_procurement_accounts_for_spot_interruption()
    test_energy_formula_and_thermal_carbon_faults()
    test_faults_fail_closed_with_actionable_reasons()
    test_input_validation()
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test_cli_json_stdout_matches_output_file(Path(directory))
    print("ch38 tests: PASS")
