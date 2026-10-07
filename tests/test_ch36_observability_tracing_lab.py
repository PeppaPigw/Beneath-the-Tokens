from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch36_observability_tracing_lab import (  # noqa: E402
    Request,
    diagnose,
    make_workload,
    pctl,
    simulate,
    tail_sample,
)


def test_baseline_is_deterministic_and_terminal():
    first = simulate()
    second = simulate()
    assert first == second
    assert first["schema_version"] == 1
    assert first["invariants"]["all_requests_terminal"]
    assert first["invariants"]["span_parent_ids_resolve"]
    assert first["invariants"]["trace_context_complete"]
    assert first["metrics"]["e2e"]["p99_ms"] is not None


def test_trace_tree_and_signal_correlation():
    result = simulate()
    traces = result["traces"]
    roots = [row for row in traces if row["parent_span_id"] is None]
    assert len(roots) == len(make_workload())
    assert all(row["duration_ms"] >= 0 for row in traces)
    assert result["diagnosis"]["cross_layer_links"] == {
        "trace_to_metric": True,
        "metric_to_resource": True,
        "resource_to_profile": True,
    }
    assert "trace_id" in result["cardinality"]["forbidden_labels"]


def test_network_fault_keeps_tail_and_reports_retransmit():
    result = simulate(fault="network_tail")
    assert result["diagnosis"]["root_cause"] == "network_or_collective_tail"
    assert result["metrics"]["counters"]["network_retransmits_total"] > 0
    assert result["sampling"]["kept_count"] >= 1
    assert any("network.rpc" == row["name"] for row in result["traces"])


def test_gpu_throttle_fault_uses_gpu_evidence():
    result = simulate(fault="gpu_throttle")
    assert result["diagnosis"]["root_cause"] == "gpu_power_or_thermal_throttle"
    assert any(row["name"] == "gpu.throttle" for row in result["traces"])
    assert any(item["power_throttle"] for item in result["telemetry"])


def test_storage_fault_links_io_to_queue():
    result = simulate(fault="storage_tail")
    assert result["diagnosis"]["root_cause"] == "storage_io_tail"
    assert max(item["disk_io_latency_ms"] for item in result["telemetry"]) > 0
    assert any("storage.checkpoint_fsync" == row["name"] for row in result["traces"])
    assert result["metrics"]["queue"]["p99_ms"] is not None


def test_cardinality_budget_and_forbidden_ids_are_explicit():
    result = simulate(cardinality_budget=3)
    card = result["cardinality"]
    assert card["budget"] == 3
    assert card["dropped_series"] == 4
    assert "trace_id" in card["forbidden_labels"]
    assert "request_id" in card["forbidden_labels"]
    assert result["invariants"]["metrics_exclude_trace_id_labels"]


def test_tail_sampler_error_and_slow_predicates():
    result = simulate(fault="network_tail")
    kept = set(result["sampling"]["kept_trace_ids"])
    assert kept
    reasons = result["sampling"]["reasons"]
    assert all(reasons[trace] for trace in kept)
    assert any("fault_signal" in reasons[trace] for trace in kept)


def test_percentile_validation_and_request_validation():
    assert pctl([1, 2, 3, 4], 50) == 2.5
    assert pctl([], 95) is None
    try:
        pctl([1], 101)
    except ValueError:
        pass
    else:
        raise AssertionError("expected percentile validation")
    try:
        Request("", "tenant", 0, 1, 1).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected request validation")


def test_cli_json_output(tmp_path: Path):
    output = tmp_path / "ch36.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch36_observability_tracing_lab.py"), "--fault", "network_tail", "--output", str(output)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    stdout_payload = json.loads(proc.stdout)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert stdout_payload["fault"] == "network_tail"
    assert stdout_payload["invariants"]["all_requests_terminal"]


if __name__ == "__main__":
    test_baseline_is_deterministic_and_terminal()
    test_trace_tree_and_signal_correlation()
    test_network_fault_keeps_tail_and_reports_retransmit()
    test_gpu_throttle_fault_uses_gpu_evidence()
    test_storage_fault_links_io_to_queue()
    test_cardinality_budget_and_forbidden_ids_are_explicit()
    test_tail_sampler_error_and_slow_predicates()
    test_percentile_validation_and_request_validation()
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test_cli_json_output(Path(directory))
    print("ch36 tests: PASS")
