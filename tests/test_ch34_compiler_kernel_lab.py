from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch34_compiler_kernel_lab import (  # noqa: E402
    Node,
    autotune,
    capture_graph,
    compile_plan,
    default_graph,
    estimate_cost,
    fuse_graph,
    max_abs_error,
    reduction_graph,
    run_fused,
    run_reference,
    simulate,
)


def test_capture_records_guards_and_breaks_for_dynamic_and_unsupported_graphs():
    clean = capture_graph(default_graph())
    assert clean.graphable
    assert any(item.startswith("shape=") for item in clean.guards)
    dynamic = capture_graph(default_graph(dynamic_shape=True))
    assert dynamic.graphable
    assert "dynamic_shape=true" in dynamic.guards
    broken = capture_graph(default_graph(unsupported_op=True))
    assert not broken.graphable
    assert any("unsupported_or_side_effect" in item for item in broken.breaks)


def test_fusion_only_merges_adjacent_pointwise_nodes():
    groups = fuse_graph(default_graph())
    assert len(groups) == 1 and groups[0].fused
    reduction_groups = fuse_graph(reduction_graph())
    assert len(reduction_groups) == 3
    assert any("reduction" in group.reason for group in reduction_groups)


def test_compile_cache_key_is_deterministic_and_second_run_hits():
    cache: set[str] = set()
    first = compile_plan(default_graph(), cache=cache)
    second = compile_plan(default_graph(), cache=cache)
    assert first.cache_key == second.cache_key
    assert first.cache_hit is False and second.cache_hit is True
    assert len(cache) == 1


def test_fusion_cost_reduces_launches_and_first_run_includes_compile():
    nodes = default_graph()
    groups = fuse_graph(nodes)
    cold = estimate_cost(nodes, groups, cache_hit=False)
    hot = estimate_cost(nodes, groups, cache_hit=True)
    assert cold.launches == 1 < len(nodes)
    assert cold.total_first_ms > hot.total_first_ms
    assert cold.bytes < sum(node.bytes * 2 for node in nodes)


def test_reference_and_fused_outputs_match_with_tiny_error():
    nodes = default_graph()
    values = [(-2.0 + i * 0.25) for i in range(16)]
    assert max_abs_error(run_reference(values, nodes), run_fused(values, nodes)) <= 1e-12


def test_autotune_budget_and_safe_fallback_are_deterministic():
    tune_a = autotune(budget_ms=8.0)
    tune_b = autotune(budget_ms=8.0)
    assert tune_a == tune_b
    assert tune_a.best["proxy_ms"] == min(item["proxy_ms"] for item in tune_a.candidates)
    result = simulate(fail_after=1)
    assert result["autotune"]["fallback"] is True
    assert result["autotune"]["selected"]["mode"] == "reference_fallback"
    assert result["summary"]["safe_fallback"]


def test_simulate_exposes_graph_break_cache_and_correctness_contracts():
    normal = simulate()
    broken = simulate(dynamic_shape=True, unsupported_op=True)
    assert normal["summary"]["cache_hit_on_second"]
    assert normal["summary"]["fusion_reduced_launches"]
    assert normal["summary"]["correctness_pass"]
    assert broken["summary"]["graph_break_count"] >= 1
    assert broken["compile"]["cache_entries"] == 1


def test_cli_writes_json(tmp_path: Path):
    out = tmp_path / "ch34.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch34_compiler_kernel_lab.py"), "--dynamic-shape", "--unsupported-op", "--fail-after", "1", "--output", str(out)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert payload["graph"]["graphable"] is False
    assert json.loads(out.read_text(encoding="utf-8")) == payload


if __name__ == "__main__":
    test_capture_records_guards_and_breaks_for_dynamic_and_unsupported_graphs()
    test_fusion_only_merges_adjacent_pointwise_nodes()
    test_compile_cache_key_is_deterministic_and_second_run_hits()
    test_fusion_cost_reduces_launches_and_first_run_includes_compile()
    test_reference_and_fused_outputs_match_with_tiny_error()
    test_autotune_budget_and_safe_fallback_are_deterministic()
    test_simulate_exposes_graph_break_cache_and_correctness_contracts()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_cli_writes_json(Path(d))
    print("ch34 tests: PASS")
