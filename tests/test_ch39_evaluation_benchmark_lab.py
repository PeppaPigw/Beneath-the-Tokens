from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch39_evaluation_benchmark_lab import (  # noqa: E402
    Case,
    build_fixture,
    contamination_check,
    mean_ci95,
    percentile,
    quality_system_joint,
    regression_gate,
    run_matrix,
    shadow_canary,
    simulate,
)


def test_deterministic_cpu_only_baseline_admitted():
    first, second = simulate(), simulate()
    assert first == second
    assert first["decision"] == "admit"
    assert first["invariants"]["cpu_only"]
    assert first["invariants"]["deterministic"]
    assert first["invariants"]["no_external_side_effects"]
    assert first["quality_system"]["joint_ok"]


def test_micro_meso_macro_matrix_has_ci_and_units():
    matrix = run_matrix()
    assert set(matrix["by_tier"]) == {"micro", "meso", "macro"}
    for result in matrix["results"]:
        assert set(result["latency_ms"]) == {"p50", "p95", "p99", "mean_ci95"}
        ci = result["latency_ms"]["mean_ci95"]
        assert ci["low"] <= ci["mean"] <= ci["high"]
        assert result["throughput_rps"] > 0
        assert result["cost_per_1k_requests_usd"] > 0
        assert result["energy_joules_per_request"] > 0


def test_quality_and_system_are_joint_gate():
    okay = quality_system_joint(quality={"accuracy": .97, "safety": .995}, system={"p95_ms": 100, "cost_per_1k": .03}, targets={"accuracy": .95, "safety": .99, "p95_ms": 120, "cost_per_1k": .05})
    assert okay["joint_ok"]
    bad = quality_system_joint(quality={"accuracy": .97, "safety": .995}, system={"p95_ms": 200, "cost_per_1k": .03}, targets={"accuracy": .95, "safety": .99, "p95_ms": 120, "cost_per_1k": .05})
    assert not bad["joint_ok"] and "p95_ms" in bad["failed"]


def test_regression_gate_and_shadow_canary_fail_closed():
    base = {"p95_ms": 100.0, "cost_per_1k": .04, "quality": .98}
    candidate = {"p95_ms": 108.0, "cost_per_1k": .041, "quality": .979}
    assert regression_gate(baseline=base, candidate=candidate)["decision"] == "deny"
    result = shadow_canary(baseline=base, candidate=candidate)
    assert result["shadow"]["decision"] == "observe"
    assert result["shadow"]["mutates_user_state"] is False
    assert result["canary"]["decision"] == "deny"


def test_contamination_and_faults():
    clean = contamination_check(train_ids=["a", "b"], eval_ids=["c"])
    dirty = contamination_check(train_ids=["a", "b"], eval_ids=["b", "c"])
    assert clean["action"] == "admit"
    assert dirty["action"] == "deny" and dirty["overlap"] == ["b"]
    assert simulate(fault="contamination")["decision"] == "deny"
    assert simulate(fault="low_quality")["decision"] == "deny"
    assert simulate(fault="regression")["decision"] == "deny"


def test_statistics_and_input_validation():
    assert percentile([1, 2, 3, 4], .5) == 2.5
    ci = mean_ci95([1, 2, 3, 4])
    assert ci["low"] < ci["mean"] < ci["high"]
    try:
        Case("bad", "unknown", 4, 1, 1, 1, 1, 1, .5, "x").validate()
    except ValueError as exc:
        assert "tier" in str(exc)
    else:
        raise AssertionError("unknown tier must fail")


def test_cli_stdout_and_output_file_match(tmp_path: Path):
    output = tmp_path / "ch39.json"
    proc = subprocess.run([sys.executable, str(ROOT / "labs/ch39_evaluation_benchmark_lab.py"), "--fault", "hardware_drift", "--output", str(output)], cwd=ROOT, check=True, text=True, capture_output=True)
    stdout = json.loads(proc.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == saved
    assert stdout["fault"] == "hardware_drift"


if __name__ == "__main__":
    test_deterministic_cpu_only_baseline_admitted()
    test_micro_meso_macro_matrix_has_ci_and_units()
    test_quality_and_system_are_joint_gate()
    test_regression_gate_and_shadow_canary_fail_closed()
    test_contamination_and_faults()
    test_statistics_and_input_validation()
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test_cli_stdout_and_output_file_match(Path(directory))
    print("ch39 tests: PASS")
