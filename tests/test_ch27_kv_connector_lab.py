from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch27_kv_connector_lab import KVShape, PROFILES, make_pages, simulate, transfer_pages
import random


def test_kv_formula_scales_linearly_with_tokens():
    shape = KVShape(layers=2, kv_heads=4, head_dim=8, bytes_per_element=2)
    assert shape.bytes_for_tokens(10) == 2 * 2 * 10 * 4 * 8 * 2
    assert shape.bytes_for_tokens(20) == 2 * shape.bytes_for_tokens(10)


def test_paging_covers_exact_token_count_and_bytes():
    shape = KVShape(layers=1, kv_heads=1, head_dim=4, bytes_per_element=2)
    pages = make_pages("r", tokens=19, page_tokens=8, shape=shape)
    assert [p.token_count for p in pages] == [8, 8, 3]
    assert sum(p.payload_bytes for p in pages) == shape.bytes_for_tokens(19)
    assert all(len(p.digest) == 64 for p in pages)


def test_transfer_protocol_commits_and_records_states():
    shape = KVShape(layers=1, kv_heads=1, head_dim=2, bytes_per_element=2)
    pages = make_pages("r", tokens=8, page_tokens=4, shape=shape)
    result = transfer_pages(pages, PROFILES["memcpy"], rng=random.Random(3))
    assert result.ok
    assert result.states == ["INIT", "NEGOTIATED", "MEMORY_REGISTERED", "TRANSFERRING", "COMMITTED"]
    assert result.bytes_sent == shape.bytes_for_tokens(8)


def test_checksum_retry_is_bounded():
    shape = KVShape(layers=1, kv_heads=1, head_dim=2, bytes_per_element=2)
    pages = make_pages("r", tokens=4, page_tokens=4, shape=shape)
    profile = PROFILES["tcp"]
    result = transfer_pages(pages, profile, rng=random.Random(1))
    assert result.ok
    assert 0 <= result.retries <= profile.max_retries * len(pages)


def test_simulation_cache_hit_sends_no_bytes():
    result = simulate(seed=0, requests=20, tokens=128, page_tokens=32,
                      connector_names=["lmcache"], shape=KVShape(), lmcache_hit_rate=1.0)
    row = result["connectors"]["lmcache"]
    assert row["cache_hits"] == 20
    assert row["bytes_sent"] == 0
    assert row["failed"] == 0


def test_cli_emits_json(tmp_path: Path):
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch27_kv_connector_lab.py"), "--requests", "1",
         "--tokens", "16", "--page-tokens", "8", "--output", str(out)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert out.exists() and json.loads(out.read_text()) == payload

if __name__ == "__main__":
    import tempfile
    test_kv_formula_scales_linearly_with_tokens()
    test_paging_covers_exact_token_count_and_bytes()
    test_transfer_protocol_commits_and_records_states()
    test_checksum_retry_is_bounded()
    test_simulation_cache_hit_sends_no_bytes()
    with tempfile.TemporaryDirectory() as d:
        test_cli_emits_json(Path(d))
    print("ch27 tests: PASS")
