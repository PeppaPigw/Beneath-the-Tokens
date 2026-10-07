from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch30_kv_compression_lab import (  # noqa: E402
    CompressionPlan,
    KVShape,
    compressed_bytes,
    low_rank_error,
    plan_quality_loss,
    quantize,
    simulate,
    sparse_reconstruct,
)


def test_quantize_is_bounded_and_reproducible():
    values = [0.1, -0.7, 1.2, 3.4, -2.2, 0.0]
    a = quantize(values, bits=4, group_size=2)
    b = quantize(values, bits=4, group_size=2)
    assert a == b
    assert len(a[0]) == len(values)
    assert a[1] >= 0.0
    assert a[2] >= 0.0


def test_sparse_and_rank_errors_have_expected_limits():
    values = [1.0, 0.2, -0.9, 0.1, 0.01]
    reconstruction, mse, nonzero = sparse_reconstruct(values, 0.4)
    assert nonzero == 2
    assert mse > 0
    assert low_rank_error([3.0, 2.0, 1.0], 0) == 1.0
    assert low_rank_error([3.0, 2.0, 1.0], 3) == 0.0


def test_compressed_payload_is_smaller_for_int4_and_rank():
    shape = KVShape(layers=2, kv_heads=2, head_dim=8)
    fp16 = compressed_bytes(shape, 16, CompressionPlan("fp16", 16))
    int4_rank = compressed_bytes(shape, 16, CompressionPlan("int4-rank", 4, rank=2))
    assert int4_rank < fp16
    assert plan_quality_loss(CompressionPlan("fp16", 16)) == 0.0
    assert plan_quality_loss(CompressionPlan("int4", 4)) > 0.0


def test_simulation_has_tier_hits_and_is_seed_reproducible():
    kwargs = dict(seed=30, requests=8, pages=14, accesses=100, page_tokens=2,
                  hbm_bytes=600_000, dram_bytes=900_000, ssd_bytes=2_000_000)
    first = simulate(**kwargs)
    second = simulate(**kwargs)
    assert first == second
    summary = first["summary"]
    assert summary["accesses"] == 100
    assert summary["misses"] > 0
    assert sum(summary["hits_by_tier"].values()) > 0
    assert summary["admissions"] > 0
    assert summary["latency_ms"]["p95"] >= summary["latency_ms"]["p50"]


def test_tiny_tier_capacity_records_evictions():
    result = simulate(seed=3, requests=4, pages=10, accesses=80, page_tokens=1,
                      hbm_bytes=1, dram_bytes=1, ssd_bytes=1_000_000)
    assert result["summary"]["evictions"] > 0 or result["summary"]["promotions"] == 0


def test_cli_writes_json(tmp_path: Path):
    out = tmp_path / "ch30.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch30_kv_compression_lab.py"), "--requests", "3",
         "--pages", "8", "--accesses", "20", "--page-tokens", "1", "--output", str(out)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert json.loads(out.read_text(encoding="utf-8")) == payload


if __name__ == "__main__":
    test_quantize_is_bounded_and_reproducible()
    test_sparse_and_rank_errors_have_expected_limits()
    test_compressed_payload_is_smaller_for_int4_and_rank()
    test_simulation_has_tier_hits_and_is_seed_reproducible()
    test_tiny_tier_capacity_records_evictions()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_cli_writes_json(Path(d))
    print("ch30 tests: PASS")
