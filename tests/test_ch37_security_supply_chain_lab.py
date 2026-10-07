from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labs.ch37_security_supply_chain_lab import (  # noqa: E402
    Artifact,
    RuntimeSpec,
    assess_model_data,
    build_fixture,
    evaluate_runtime,
    scan_dependencies,
    simulate,
    verify_artifact,
)


def test_baseline_is_deterministic_and_admitted():
    first = simulate()
    second = simulate()
    assert first == second
    assert first["schema_version"] == 1
    assert first["decision"] == "admit"
    assert first["invariants"]["all_required_evidence"]
    assert first["invariants"]["audit_chain_valid"]
    assert first["invariants"]["no_secret_values_in_events"]


def test_artifact_requires_digest_signature_provenance_and_sbom():
    artifact = build_fixture()["model"]
    assert verify_artifact(artifact)["decision"] == "allow"
    for field in ("digest", "signed", "provenance", "sbom"):
        changed = artifact.__dict__.copy()
        if field == "digest":
            changed[field] = ""
        elif field in {"signed", "provenance", "sbom"}:
            changed[field] = False
        bad = Artifact(**changed)
        result = verify_artifact(bad)
        assert result["decision"] == "deny"
        assert result["missing_or_invalid"]


def test_poisoned_data_is_detected_without_claiming_model_quality():
    result = simulate(fault="poisoned_data")
    assert result["decision"] == "quarantine"
    assert result["model_data"]["poisoning"]["detected"]
    assert result["incident"]["containment"] == "freeze_release_and_revoke_promotion"


def test_dependency_vulnerability_blocks_release():
    result = simulate(fault="vulnerable_dependency")
    assert result["decision"] == "deny"
    assert result["dependencies"]["critical_or_high"] >= 1
    assert "dependency_vulnerability" in result["deny_reasons"]


def test_runtime_isolation_blocks_unsafe_pod_and_gpu_cross_tenant():
    unsafe = simulate(fault="unsafe_runtime")
    assert unsafe["decision"] == "deny"
    assert "runtime_isolation" in unsafe["deny_reasons"]
    gpu = simulate(fault="gpu_cross_tenant")
    assert gpu["decision"] == "deny"
    assert "gpu_multi_tenancy" in gpu["deny_reasons"]


def test_secret_and_network_policies_are_fail_closed():
    secret = simulate(fault="secret_leak")
    assert secret["decision"] == "deny"
    assert "secret_handling" in secret["deny_reasons"]
    assert secret["invariants"]["no_secret_values_in_events"]
    egress = simulate(fault="unexpected_egress")
    assert egress["decision"] == "deny"
    assert "network_policy" in egress["deny_reasons"]


def test_component_contracts_validate_inputs_and_report_reasons():
    artifact = build_fixture()["model"]
    assert assess_model_data(build_fixture()["dataset"], artifact)["decision"] == "allow"
    assert scan_dependencies(build_fixture()["dependencies"])["critical_or_high"] == 0
    runtime = RuntimeSpec(**build_fixture()["runtime"])
    assert evaluate_runtime(runtime)["decision"] == "allow"
    malformed = verify_artifact(Artifact("", "1", "sha256:" + "a" * 64, True, True, True, 0, "registry", "m"))
    assert malformed["decision"] == "deny"
    assert "artifact_schema" in malformed["missing_or_invalid"]


def test_cli_json_output(tmp_path: Path):
    output = tmp_path / "ch37.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "labs/ch37_security_supply_chain_lab.py"), "--fault", "unsafe_runtime", "--output", str(output)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    stdout_payload = json.loads(proc.stdout)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert stdout_payload["fault"] == "unsafe_runtime"


if __name__ == "__main__":
    test_baseline_is_deterministic_and_admitted()
    test_artifact_requires_digest_signature_provenance_and_sbom()
    test_poisoned_data_is_detected_without_claiming_model_quality()
    test_dependency_vulnerability_blocks_release()
    test_runtime_isolation_blocks_unsafe_pod_and_gpu_cross_tenant()
    test_secret_and_network_policies_are_fail_closed()
    test_component_contracts_validate_inputs_and_report_reasons()
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test_cli_json_output(Path(directory))
    print("ch37 tests: PASS")
