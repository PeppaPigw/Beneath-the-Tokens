from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch40_capstone_lab import (ServiceSpec, build_fixture, capacity_plan, cost_energy, evidence_manifest, rollout_gate, simulate, slo_budget)

def test_baseline_is_deterministic_cpu_only_and_admitted():
    a, b = simulate(), simulate()
    assert a == b and a["decision"] == "admit"
    assert a["invariants"] == {"cpu_only": True, "deterministic": True, "no_external_side_effects": True, "fail_closed": True}
    assert a["capacity"]["capacity_ok"] and a["slo"]["slo_ok"] and a["cost_energy"]["budget_ok"]
    assert len(a["evidence"]["manifest_sha256"]) == 64

def test_capacity_and_slo_budget_units():
    spec = build_fixture()["spec"]
    plan = capacity_plan(spec)
    assert plan["required_nodes"] == 4 and plan["provisioned_nodes"] == 4 and plan["capacity_ok"]
    assert capacity_plan(spec, peak_qps=200, node_count=4)["capacity_ok"] is False
    budget = slo_budget(spec, observed={"p95_ms": 200, "availability": .999, "monthly_requests": 100_000})
    assert budget["slo_ok"] and budget["error_budget_requests"] == 500.0

def test_rollout_fail_closed_and_cost():
    base = {"p95_ms": 200, "error_rate": .001, "quality": .98}
    bad = {"p95_ms": 260, "error_rate": .02, "quality": .90}
    gate = rollout_gate(baseline=base, canary=bad)
    assert gate["decision"] == "rollback" and set(gate["rollback_reasons"]) == {"latency", "errors", "quality"}
    c = cost_energy(build_fixture()["spec"], nodes=4, monthly_requests=1000)
    assert c["monthly_node_cost_usd"] == 8400.0 and c["monthly_energy_kwh"] > 0

def test_faults_deny_with_reason_codes():
    for fault in ("traffic_spike", "gpu_loss", "network_partition", "schema_drift", "budget_breach", "rollback"):
        result = simulate(fault=fault)
        assert result["decision"] == "deny" and result["deny_reasons"]
        assert result["invariants"]["fail_closed"]

def test_validation_and_cli_round_trip(tmp_path: Path):
    spec = ServiceSpec("x", 1, 1, 1, .3, 100, .99, 1, 1, 1); spec.validate()
    try: ServiceSpec("", 1, 1, 1, .3, 100, .99, 1, 1, 1).validate()
    except ValueError: pass
    else: raise AssertionError("empty name must fail")
    output = tmp_path / "result.json"
    proc = subprocess.run([sys.executable, str(ROOT / "labs/ch40_capstone_lab.py"), "--fault", "network_partition", "--output", str(output)], cwd=ROOT, check=True, text=True, capture_output=True)
    assert json.loads(proc.stdout) == json.loads(output.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import tempfile
    test_baseline_is_deterministic_cpu_only_and_admitted(); test_capacity_and_slo_budget_units(); test_rollout_fail_closed_and_cost(); test_faults_deny_with_reason_codes()
    with tempfile.TemporaryDirectory() as d: test_validation_and_cli_round_trip(Path(d))
    print("ch40 tests: PASS")
