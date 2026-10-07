from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch29_speculative_lab import (  # noqa: E402
    Config,
    VOCAB,
    constrained_speculative,
    draft_distribution,
    exact_verify,
    json_allowed,
    masked_distribution,
    simulate,
)


def test_accept_reject_exact_path_commits_correction():
    # q puts all mass on token 0 while p puts all mass on token 1. The token
    # must be rejected and one corrective target token committed.
    accepted, committed, rejected = exact_verify(
        __import__("random").Random(1),
        [0.0, 1.0] + [0.0] * (len(VOCAB) - 2),
        [0],
        [[1.0, 0.0] + [0.0] * (len(VOCAB) - 2)],
    )
    assert accepted == 0
    assert committed == 1
    assert rejected == 1


def test_masked_distribution_has_no_forbidden_mass():
    probs = masked_distribution([1.0] * len(VOCAB), {2, 3})
    assert sum(probs) == 1.0
    assert all(p == 0.0 for i, p in enumerate(probs) if i not in {2, 3})


def test_constrained_tokens_follow_toy_grammar():
    payload = constrained_speculative(__import__("random").Random(29), 14, 3, 0.8, True, 0.0)
    assert payload["output_tokens"] == 14
    assert all(t in json_allowed(i) for i, t in enumerate(payload["tokens"]))


def test_simulate_baseline_and_speculative_complete():
    base = simulate(Config(seed=1, requests=8, concurrency=4, output_len=12, mode="baseline"))
    spec = simulate(Config(seed=1, requests=8, concurrency=4, output_len=12, mode="speculative"))
    assert base["summary"]["completed"] == 8
    assert spec["summary"]["completed"] == 8
    assert spec["summary"]["draft_tokens"] > 0
    assert spec["summary"]["exact_output_token_count"] == 8 * 12


def test_seed_reproducible_and_grammar_rejects_visible():
    cfg = Config(seed=7, requests=5, concurrency=2, output_len=10, grammar=True, grammar_reject_rate=0.5)
    a, b = simulate(cfg), simulate(cfg)
    assert a == b
    assert a["summary"]["grammar_rejects"] >= 0


def test_cli_writes_same_json(tmp_path: Path):
    out = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch29_speculative_lab.py"), "--requests", "3",
         "--output-len", "8", "--draft-len", "2", "--mode", "lookahead", "--output", str(out)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert json.loads(out.read_text(encoding="utf-8")) == payload


if __name__ == "__main__":
    test_accept_reject_exact_path_commits_correction()
    test_masked_distribution_has_no_forbidden_mass()
    test_constrained_tokens_follow_toy_grammar()
    test_simulate_baseline_and_speculative_complete()
    test_seed_reproducible_and_grammar_rejects_visible()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_cli_writes_same_json(Path(d))
    print("ch29 tests: PASS")
