from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch35_serving_scheduling_lab import (  # noqa: E402
    Request,
    SchedulerConfig,
    admission_control,
    make_requests,
    pctl,
    simulate,
    stream_tokens,
)


def test_default_workload_is_terminal_and_reports_slos():
    result = simulate()
    assert result["schema_version"] == 1
    assert result["invariants"]["all_terminal"]
    assert result["invariants"]["no_negative_latency"]
    assert result["metrics"]["completed"] >= 3
    assert result["metrics"]["ttft_p95_ms"] is not None
    assert result["metrics"]["e2e_p95_ms"] is not None


def test_continuous_batching_respects_request_and_prompt_caps():
    reqs = tuple(Request(f"r{i}", "t", 0, 8, 2) for i in range(6))
    result = simulate(reqs, config=SchedulerConfig(max_batch_requests=2, max_batch_prompt_tokens=16))
    assert result["invariants"]["all_terminal"]
    starts = {row["request_id"]: row["prefill_start_ms"] for row in result["requests"]}
    assert len(set(starts.values())) >= 3  # six requests require at least three batches


def test_disaggregated_and_coupled_are_deterministic_and_have_metrics():
    reqs = make_requests(burst=True)
    coupled = simulate(reqs, config=SchedulerConfig(tenant_weights=(("gold", 2.0), ("bronze", 1.0))))
    disagg = simulate(reqs, config=SchedulerConfig(disaggregated=True, policy="weighted_fair", tenant_weights=(("gold", 2.0), ("bronze", 1.0))))
    assert coupled == simulate(reqs, config=SchedulerConfig(tenant_weights=(("gold", 2.0), ("bronze", 1.0))))
    assert disagg["invariants"]["all_terminal"]
    assert disagg["config"]["disaggregated"] is True
    assert disagg["metrics"]["ttft_p95_ms"] is not None


def test_admission_control_exposes_global_and_tenant_reasons():
    reqs = tuple(Request(f"r{i}", "gold" if i < 3 else "bronze", float(i), 2, 1) for i in range(5))
    accepted, rejected = admission_control(reqs, max_inflight=3, per_tenant_limit={"gold": 2})
    assert len(accepted) == 3
    reasons = {item["reason"] for item in rejected}
    assert "tenant_concurrency_limit" in reasons or "global_inflight_limit" in reasons
    assert all("request_id" in item for item in rejected)


def test_cancellation_and_stream_backpressure_are_visible():
    reqs = (Request("cancel", "t", 0, 8, 20, cancel_ms=1.0),)
    result = simulate(reqs)
    assert result["requests"][0]["status"] == "cancelled"
    stream = stream_tokens(range(10), consumer_capacity=1, buffer_capacity=2)
    assert stream.producer_blocked
    assert len(stream.emitted) <= 10
    cancelled = stream_tokens(range(10), consumer_capacity=4, buffer_capacity=4, cancel_after=3)
    assert cancelled.cancelled and cancelled.emitted == (0, 1, 2)


def test_percentile_is_deterministic_and_validates_range():
    assert pctl([1, 2, 3, 4], 50) == 2.5
    assert pctl([], 95) is None
    try:
        pctl([1], 101)
    except ValueError:
        pass
    else:
        raise AssertionError("expected percentile validation")


def test_cli_writes_json(tmp_path: Path):
    out = tmp_path / "ch35.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch35_serving_scheduling_lab.py"), "--burst", "--disaggregated", "--policy", "weighted_fair", "--output", str(out)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert payload["config"]["disaggregated"] is True
    assert json.loads(out.read_text(encoding="utf-8")) == payload


if __name__ == "__main__":
    test_default_workload_is_terminal_and_reports_slos()
    test_continuous_batching_respects_request_and_prompt_caps()
    test_disaggregated_and_coupled_are_deterministic_and_have_metrics()
    test_admission_control_exposes_global_and_tenant_reasons()
    test_cancellation_and_stream_backpressure_are_visible()
    test_percentile_is_deterministic_and_validates_range()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_cli_writes_json(Path(d))
    print("ch35 tests: PASS")
