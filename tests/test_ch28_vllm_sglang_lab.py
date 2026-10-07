from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch28_vllm_sglang_lab import Config, simulate


def test_simulation_respects_token_budget_and_completes():
    payload = simulate(Config(seed=1, requests=6, concurrency=3, input_len=32, output_len=12, max_batch_tokens=8))
    assert payload["summary"]["completed"] == 6
    assert payload["summary"]["steps"] > 0
    assert all(r["status"] == "FINISHED" for r in payload["requests"])


def test_radix_shared_prefix_reduces_prefill_tokens():
    cold = simulate(Config(seed=2, requests=4, concurrency=1, input_len=64, output_len=4, engine="radix", shared_prefix=0))
    warm = simulate(Config(seed=2, requests=4, concurrency=1, input_len=64, output_len=4, engine="radix", shared_prefix=32))
    assert warm["summary"]["cache_hit_tokens"] > 0
    assert warm["summary"]["prefill_tokens"] < cold["summary"]["prefill_tokens"]


def test_paged_engine_has_no_synthetic_radix_hits():
    payload = simulate(Config(seed=3, requests=4, concurrency=1, input_len=64, output_len=4, engine="paged", shared_prefix=32))
    assert payload["summary"]["cache_hit_tokens"] == 0


def test_speculative_acceptance_is_bounded_by_draft():
    payload = simulate(Config(seed=4, requests=5, concurrency=2, input_len=24, output_len=20,
                              speculative=True, draft_tokens=4, accept_rate=1.0))
    assert payload["summary"]["accepted_tokens"] <= payload["summary"]["draft_tokens"]
    assert payload["summary"]["draft_tokens"] > 0


def test_grammar_rejections_are_reproducible():
    cfg = Config(seed=5, requests=5, concurrency=2, input_len=24, output_len=8, grammar_reject_rate=0.5)
    a = simulate(cfg)
    b = simulate(cfg)
    assert a["summary"]["grammar_rejects"] == b["summary"]["grammar_rejects"]
    assert a["summary"]["grammar_rejects"] > 0


def test_cli_json_and_output_file(tmp_path: Path):
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch28_vllm_sglang_lab.py"), "--requests", "2",
         "--input-len", "16", "--output-len", "4", "--output", str(out)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert out.exists() and json.loads(out.read_text()) == payload


if __name__ == "__main__":
    test_simulation_respects_token_budget_and_completes()
    test_radix_shared_prefix_reduces_prefill_tokens()
    test_paged_engine_has_no_synthetic_radix_hits()
    test_speculative_acceptance_is_bounded_by_draft()
    test_grammar_rejections_are_reproducible()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_cli_json_and_output_file(Path(d))
    print("ch28 tests: PASS")
