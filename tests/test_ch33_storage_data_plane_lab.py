from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch33_storage_data_plane_lab import (  # noqa: E402
    BackendSpec,
    assign_shards,
    checkpoint_commit,
    estimate_reads,
    simulate,
)


def test_shard_assignment_is_balanced_unique_and_reproducible():
    a = assign_shards(11, 3, seed=33)
    b = assign_shards(11, 3, seed=33)
    assert a == b
    flat = [item for group in a for item in group]
    assert sorted(flat) == list(range(11))
    assert max(map(len, a)) - min(map(len, a)) <= 1


def test_read_model_exposes_latency_and_bandwidth_ordering():
    object_store = estimate_reads(BackendSpec("object", 180.0, 8.0, 1), 12, 4)
    nvme = estimate_reads(BackendSpec("nvme", 1200.0, 0.08), 12, 4)
    assert object_store.bytes == nvme.bytes == 12 * 4 * 1024 * 1024
    assert object_store.elapsed_ms > nvme.elapsed_ms
    assert nvme.throughput_mb_s > object_store.throughput_mb_s


def test_checkpoint_commit_requires_all_shards_and_marker():
    complete = checkpoint_commit(step=100, shards=4)
    partial = checkpoint_commit(step=100, shards=4, fail_after=3)
    assert complete.committed and complete.recoverable
    assert not partial.committed and not partial.recoverable
    assert partial.written_shards == 3
    assert partial.manifest_hash != complete.manifest_hash


def test_simulation_is_reproducible_and_surfaces_incomplete_checkpoint():
    kwargs = dict(seed=33, shards=9, workers=4, shard_mib=2, checkpoint_step=42, fail_after=5)
    first = simulate(**kwargs)
    second = simulate(**kwargs)
    assert first == second
    assert first["assignment"]["all_shards_unique"]
    assert first["summary"]["checkpoint_committed"] is False
    assert first["summary"]["uncommitted_write_is_visible"] is True
    assert first["summary"]["fastest_backend"] == "nvme"


def test_cli_writes_json(tmp_path: Path):
    out = tmp_path / "ch33.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch33_storage_data_plane_lab.py"), "--shards", "6",
         "--workers", "2", "--shard-mib", "1", "--fail-after", "4", "--output", str(out)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert json.loads(out.read_text(encoding="utf-8")) == payload


if __name__ == "__main__":
    test_shard_assignment_is_balanced_unique_and_reproducible()
    test_read_model_exposes_latency_and_bandwidth_ordering()
    test_checkpoint_commit_requires_all_shards_and_marker()
    test_simulation_is_reproducible_and_surfaces_incomplete_checkpoint()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_cli_writes_json(Path(d))
    print("ch33 tests: PASS")
