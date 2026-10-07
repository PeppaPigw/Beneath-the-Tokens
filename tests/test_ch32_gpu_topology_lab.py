from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch32_gpu_topology_lab import (  # noqa: E402
    TopologyConfig,
    failure_injection,
    plan_alltoall,
    plan_ring_allreduce,
    rank_to_node,
    simulate,
    topology_matrix,
)


def test_rank_mapping_and_path_matrix_are_deterministic():
    cfg = TopologyConfig(nodes=2, gpus_per_node=2, replicas=2)
    assert rank_to_node(cfg) == (0, 0, 1, 1)
    matrix = topology_matrix(cfg)
    assert matrix[0][1] == "intra_node"
    assert matrix[0][2] == "inter_node"
    assert matrix[2][3] == "intra_node"
    assert matrix[1][1] == "self"


def test_alltoall_conserves_per_rank_payload_and_exposes_inter_node_bytes():
    cfg = TopologyConfig(nodes=2, gpus_per_node=2)
    plan = plan_alltoall(cfg, payload_bytes=1000)
    assert plan.operation == "alltoall"
    assert plan.rank_send_bytes == (1000, 1000, 1000, 1000)
    assert plan.total_bytes == 4000
    assert plan.inter_node_bytes > 0
    assert plan.intra_node_bytes > 0
    assert plan.bottleneck_ms >= plan.p50_link_ms


def test_ring_allreduce_has_two_traversals_and_crosses_node_boundary():
    cfg = TopologyConfig(nodes=2, gpus_per_node=2)
    plan = plan_ring_allreduce(cfg, payload_bytes=1024)
    assert plan.operation == "ring_allreduce"
    assert len(plan.links) == cfg.world_size * 2 * (cfg.world_size - 1)
    assert plan.inter_node_bytes > 0
    assert plan.total_bytes == sum(plan.rank_send_bytes)
    assert plan.bottleneck_ms > 0


def test_failure_scope_distinguishes_gpu_node_and_switch():
    cfg = TopologyConfig(nodes=3, gpus_per_node=2, replicas=2)
    gpu = failure_injection(cfg, scope="gpu", target=1, requests=10)
    node = failure_injection(cfg, scope="node", target=1, requests=10)
    switch = failure_injection(cfg, scope="switch", target=0, requests=10)
    assert gpu.affected_ranks == (1,)
    assert node.affected_ranks == (2, 3)
    assert gpu.inference_replicas_available == 2
    assert node.inference_replicas_available == 2
    assert switch.inference_replicas_available == 0
    assert switch.dropped_requests == 10
    assert not gpu.training_collective_survives


def test_simulation_reproducible_and_failure_is_visible():
    cfg = TopologyConfig(nodes=2, gpus_per_node=2, replicas=2)
    a = simulate(cfg=cfg, payload_bytes=4096, requests=20, failure_scope="node", failure_target=0)
    b = simulate(cfg=cfg, payload_bytes=4096, requests=20, failure_scope="node", failure_target=0)
    assert a == b
    assert a["schema_version"] == 1
    assert a["failure"]["scope"] == "node"
    assert a["summary"]["training_survives_failure"] is False
    assert a["inference"]["dropped"] == 0


def test_cli_writes_json(tmp_path: Path):
    out = tmp_path / "ch32.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch32_gpu_topology_lab.py"), "--nodes", "2",
         "--gpus-per-node", "2", "--payload-mib", "1", "--requests", "8",
         "--failure-scope", "gpu", "--failure-target", "0", "--output", str(out)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert json.loads(out.read_text(encoding="utf-8")) == payload


if __name__ == "__main__":
    test_rank_mapping_and_path_matrix_are_deterministic()
    test_alltoall_conserves_per_rank_payload_and_exposes_inter_node_bytes()
    test_ring_allreduce_has_two_traversals_and_crosses_node_boundary()
    test_failure_scope_distinguishes_gpu_node_and_switch()
    test_simulation_reproducible_and_failure_is_visible()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_cli_writes_json(Path(d))
    print("ch32 tests: PASS")
