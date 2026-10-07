from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch31_moe_parallelism_lab import (  # noqa: E402
    RoutingConfig,
    build_all_to_all,
    make_logits,
    route_tokens,
    simulate,
    softmax,
)


def test_softmax_is_normalized_and_stable():
    probs = softmax([1000.0, 1000.0, 999.0])
    assert abs(sum(probs) - 1.0) < 1e-12
    assert probs[0] == probs[1]


def test_capacity_bounds_assignments_and_reports_overflow():
    cfg = RoutingConfig(num_experts=4, top_k=2, capacity_factor=0.5, num_ranks=2)
    result = route_tokens(make_logits(12, 4, seed=2, hot_expert=0, hot_bias=8.0), cfg)
    assert all(load <= result.capacity for load in result.expert_load)
    assert result.overflow_assignments > 0
    assert result.dropped_tokens > 0
    assert sum(result.expert_load) <= len(result.assignments) * cfg.top_k


def test_second_policy_uses_fallback_without_duplicate_expert():
    cfg = RoutingConfig(num_experts=4, top_k=2, capacity_factor=0.5, overflow_policy="second", num_ranks=2)
    result = route_tokens(make_logits(10, 4, seed=3, hot_expert=0, hot_bias=10.0), cfg)
    assert all(len(set(a)) == len(a) for a in result.assignments)
    assert all(load <= result.capacity for load in result.expert_load)
    assert sum(result.expert_load) >= 1


def test_residual_policy_reports_residual_tokens_without_drop():
    cfg = RoutingConfig(num_experts=4, top_k=2, capacity_factor=0.5, overflow_policy="residual", num_ranks=2)
    result = route_tokens(make_logits(10, 4, seed=3, hot_expert=0, hot_bias=10.0), cfg)
    assert result.residual_tokens > 0
    assert result.dropped_tokens == 0
    assert result.dropped_assignments == 0


def test_all_to_all_matrix_conserves_assignments():
    assignments = ((0, 2), (1,), (3, 0), ())
    plan = build_all_to_all(assignments, num_ranks=2, num_experts=4, bytes_per_token=100)
    assert sum(map(sum, plan.send_matrix)) == 5
    assert sum(map(sum, plan.recv_matrix)) == 5
    assert plan.send_matrix[0] == (2, 2)
    assert plan.bytes_by_src == (400, 100)


def test_simulation_is_reproducible_and_exposes_contract_fields():
    cfg = RoutingConfig(seed=31, num_experts=8, top_k=2, num_ranks=4)
    first = simulate(cfg, tokens=32, hidden_bytes=128)
    second = simulate(cfg, tokens=32, hidden_bytes=128)
    assert first == second
    assert first["schema_version"] == 1
    assert first["summary"]["tokens"] == 32
    assert "aux_loss" in first["routing"]
    assert first["all_to_all"]["rounds"] >= 1


def test_cli_writes_json(tmp_path: Path):
    out = tmp_path / "ch31.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch31_moe_parallelism_lab.py"), "--tokens", "16",
         "--experts", "4", "--ranks", "2", "--top-k", "1", "--output", str(out)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert json.loads(out.read_text(encoding="utf-8")) == payload


if __name__ == "__main__":
    test_softmax_is_normalized_and_stable()
    test_capacity_bounds_assignments_and_reports_overflow()
    test_second_policy_uses_fallback_without_duplicate_expert()
    test_residual_policy_reports_residual_tokens_without_drop()
    test_all_to_all_matrix_conserves_assignments()
    test_simulation_is_reproducible_and_exposes_contract_fields()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_cli_writes_json(Path(d))
    print("ch31 tests: PASS")
