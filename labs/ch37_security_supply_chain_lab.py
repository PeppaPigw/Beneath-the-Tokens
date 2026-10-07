#!/usr/bin/env python3
"""CPU-only deterministic security and supply-chain toy lab.

The simulator models evidence gates around a model release: digest/signature/
provenance/SBOM verification, model-data poisoning checks, dependency scanning,
container/Kubernetes runtime isolation, GPU tenancy boundaries, secret handling,
network policy, an append-only audit hash chain, and incident containment.  It
never downloads packages, invokes Kubernetes, mounts a GPU, or handles a real
secret.  Results are protocol evidence, not a security certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Artifact:
    name: str
    version: str
    digest: str
    signed: bool
    provenance: bool
    sbom: bool
    vulnerability_score: float
    source: str
    model_hash: str

    def validate(self) -> None:
        if not self.name or not self.version or not self.source or not self.model_hash:
            raise ValueError("artifact name/version/source/model_hash must be non-empty")
        if not self.digest:
            raise ValueError("artifact digest must be non-empty")
        if not 0 <= self.vulnerability_score <= 10:
            raise ValueError("vulnerability_score must be in [0,10]")


@dataclass(frozen=True)
class RuntimeSpec:
    rootless: bool
    read_only_rootfs: bool
    drop_all_caps: bool
    seccomp_profile: str
    network_policy: str
    secret_refs: tuple[str, ...]
    gpu_partition: str
    sandbox_profile: str
    privileged: bool = False
    host_paths: bool = False

    def validate(self) -> None:
        if self.seccomp_profile not in {"runtime/default", "restricted"}:
            raise ValueError("unknown seccomp profile")
        if self.network_policy not in {"default-deny", "model-registry-only"}:
            raise ValueError("unknown network policy")
        if self.gpu_partition not in {"exclusive", "mig-isolated", "cpu-only", "shared-unisolated"}:
            raise ValueError("unknown gpu partition")
        if self.sandbox_profile not in {"seccomp+no_new_privs", "none"}:
            raise ValueError("unknown sandbox profile")
        for ref in self.secret_refs:
            if not ref or "\n" in ref or "\r" in ref:
                raise ValueError("secret references must be non-empty single-line identifiers")


def build_fixture() -> dict[str, Any]:
    """Return a small signed release with explicit, inspectable evidence."""
    return {
        "model": Artifact(
            name="toy-llm", version="2026.10.0", digest="sha256:" + "a" * 64,
            signed=True, provenance=True, sbom=True, vulnerability_score=0.0,
            source="registry.example/toy-llm", model_hash="sha256:" + "b" * 64,
        ),
        "dataset": {
            "name": "toy-instruction-v3", "manifest_hash": "sha256:" + "c" * 64,
            "expected_manifest_hash": "sha256:" + "c" * 64,
            "poison_rate": 0.0, "baseline_loss": 1.0, "canary_loss": 1.01,
            "label_anomaly_rate": 0.0,
        },
        "dependencies": [
            {"name": "tokenizer", "version": "1.4.2", "severity": "none", "cve": None, "fixed": True},
            {"name": "runtime", "version": "3.12.2", "severity": "low", "cve": "CVE-toy-low", "fixed": True},
        ],
        "runtime": {
            "rootless": True, "read_only_rootfs": True, "drop_all_caps": True,
            "seccomp_profile": "restricted", "network_policy": "model-registry-only",
            "secret_refs": ("secret://registry-pull",), "gpu_partition": "mig-isolated",
            "sandbox_profile": "seccomp+no_new_privs", "privileged": False, "host_paths": False,
        },
    }


def verify_artifact(artifact: Artifact, *, max_score: float = 7.0) -> dict[str, Any]:
    """Verify release evidence; missing evidence fails closed."""
    invalid: list[str] = []
    try:
        artifact.validate()
    except ValueError as exc:
        # Verification is a policy gate: malformed evidence is a deny result.
        return {"decision": "deny", "artifact": {"name": artifact.name, "version": artifact.version, "digest": artifact.digest},
                "missing_or_invalid": ["artifact_schema", str(exc)],
                "evidence": {"digest": bool(artifact.digest), "signature": artifact.signed,
                             "provenance": artifact.provenance, "sbom": artifact.sbom}}
    if not artifact.digest.startswith("sha256:") or len(artifact.digest) != 71:
        invalid.append("digest")
    if not artifact.signed:
        invalid.append("signature")
    if not artifact.provenance:
        invalid.append("provenance")
    if not artifact.sbom:
        invalid.append("sbom")
    if artifact.vulnerability_score > max_score:
        invalid.append("artifact_vulnerability_score")
    return {
        "decision": "allow" if not invalid else "deny",
        "artifact": {"name": artifact.name, "version": artifact.version, "digest": artifact.digest},
        "missing_or_invalid": invalid,
        "evidence": {"digest": bool(artifact.digest), "signature": artifact.signed,
                     "provenance": artifact.provenance, "sbom": artifact.sbom},
    }


def assess_model_data(dataset: Mapping[str, Any], artifact: Artifact, *, poison_limit: float = 0.02,
                      canary_delta_limit: float = 0.05) -> dict[str, Any]:
    """Detect manifest drift and poisoning indicators without judging model quality."""
    artifact.validate()
    required = ("manifest_hash", "expected_manifest_hash", "poison_rate", "baseline_loss", "canary_loss", "label_anomaly_rate")
    missing = [key for key in required if key not in dataset]
    if missing:
        raise ValueError(f"dataset evidence missing: {','.join(missing)}")
    poison_rate = float(dataset["poison_rate"])
    anomaly_rate = float(dataset["label_anomaly_rate"])
    baseline = float(dataset["baseline_loss"])
    canary = float(dataset["canary_loss"])
    if poison_rate < 0 or anomaly_rate < 0 or baseline <= 0 or canary <= 0:
        raise ValueError("invalid data quality evidence")
    reasons: list[str] = []
    if dataset["manifest_hash"] != dataset["expected_manifest_hash"]:
        reasons.append("manifest_hash_mismatch")
    if poison_rate > poison_limit:
        reasons.append("poison_rate_above_limit")
    if anomaly_rate > poison_limit:
        reasons.append("label_anomaly_above_limit")
    delta = (canary - baseline) / baseline
    if delta > canary_delta_limit:
        reasons.append("canary_loss_regression")
    return {"decision": "quarantine" if reasons else "allow", "detected": bool(reasons),
            "reasons": reasons, "poison_rate": poison_rate, "label_anomaly_rate": anomaly_rate,
            "canary_relative_delta": round(delta, 6), "model_hash": artifact.model_hash}


def scan_dependencies(dependencies: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return deterministic findings from a pinned dependency manifest."""
    findings: list[dict[str, Any]] = []
    severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    for dep in dependencies:
        for key in ("name", "version", "severity", "fixed"):
            if key not in dep:
                raise ValueError(f"dependency evidence missing: {key}")
        severity = str(dep["severity"]).lower()
        if severity not in severity_rank:
            raise ValueError(f"unknown dependency severity: {severity}")
        if severity_rank[severity] >= severity_rank["high"] and not bool(dep["fixed"]):
            findings.append({"name": dep["name"], "version": dep["version"], "severity": severity,
                             "cve": dep.get("cve"), "action": "block_release"})
    count = sum(1 for item in findings if item["severity"] in {"critical", "high"})
    return {"decision": "deny" if findings else "allow", "findings": findings,
            "critical_or_high": count, "manifest_entries": len(list(dependencies)) if not isinstance(dependencies, list) else len(dependencies)}


def evaluate_runtime(runtime: RuntimeSpec) -> dict[str, Any]:
    """Check rootless container, Kubernetes, secret, network and GPU boundaries."""
    runtime.validate()
    findings: list[str] = []
    if runtime.privileged or runtime.host_paths or not runtime.rootless:
        findings.append("privileged_or_host_access")
    if not runtime.read_only_rootfs or not runtime.drop_all_caps:
        findings.append("writable_rootfs_or_capabilities")
    if runtime.seccomp_profile != "restricted" or runtime.sandbox_profile != "seccomp+no_new_privs":
        findings.append("sandbox_profile")
    if runtime.network_policy not in {"default-deny", "model-registry-only"}:
        findings.append("network_policy")
    # A reference is metadata; secret values must never be embedded in a pod spec.
    if any(ref.startswith(("value:", "plaintext:", "-----BEGIN")) for ref in runtime.secret_refs):
        findings.append("secret_value_embedded")
    if runtime.gpu_partition == "shared-unisolated":
        findings.append("gpu_partition_not_isolated")
    return {"decision": "allow" if not findings else "deny", "findings": findings,
            "effective_boundary": {"rootless": runtime.rootless, "read_only_rootfs": runtime.read_only_rootfs,
                                   "network_policy": runtime.network_policy, "gpu_partition": runtime.gpu_partition}}


def _audit_events(decision: str, reasons: list[str], fault: str) -> tuple[list[dict[str, Any]], bool]:
    """Create an append-only hash chain; values/secrets are intentionally excluded."""
    events: list[dict[str, Any]] = []
    previous = "GENESIS"
    rows = [("release_check", "policy-engine", decision), ("evidence_review", "security-ci", ",".join(reasons) or "all_gates_pass"),
            ("incident_action", "on-call", "containment_recorded" if decision != "admit" else "no_action")]
    for index, (action, actor, reason) in enumerate(rows):
        payload = {"seq": index, "fault": fault, "action": action, "actor": actor, "reason": reason, "prev_hash": previous}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        event = {**payload, "hash": digest}
        events.append(event)
        previous = digest
    valid = True
    prior = "GENESIS"
    for event in events:
        body = {k: event[k] for k in ("seq", "fault", "action", "actor", "reason", "prev_hash")}
        expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        valid = valid and event["prev_hash"] == prior and event["hash"] == expected
        prior = event["hash"]
    return events, valid


def _contains_secret(events: Iterable[Mapping[str, Any]]) -> bool:
    forbidden = ("supersecret", "plaintext:", "value:", "password=", "token=")
    return any(any(fragment in json.dumps(event, sort_keys=True).lower() for fragment in forbidden) for event in events)


def simulate(*, fault: str = "none") -> dict[str, Any]:
    allowed_faults = {"none", "unsigned_model", "poisoned_data", "vulnerable_dependency", "unsafe_runtime",
                      "gpu_cross_tenant", "secret_leak", "unexpected_egress"}
    if fault not in allowed_faults:
        raise ValueError(f"fault must be one of: {', '.join(sorted(allowed_faults))}")
    fixture = build_fixture()
    model: Artifact = fixture["model"]
    dataset = dict(fixture["dataset"])
    dependencies = [dict(item) for item in fixture["dependencies"]]
    runtime_data = dict(fixture["runtime"])
    if fault == "unsigned_model":
        model = Artifact(**{**asdict(model), "signed": False})
    elif fault == "poisoned_data":
        dataset.update({"poison_rate": 0.15, "label_anomaly_rate": 0.11, "canary_loss": 1.22,
                        "manifest_hash": "sha256:" + "d" * 64})
    elif fault == "vulnerable_dependency":
        dependencies.append({"name": "image-parser", "version": "0.9.0", "severity": "critical",
                             "cve": "CVE-2099-0001", "fixed": False})
    elif fault == "unsafe_runtime":
        runtime_data.update({"rootless": False, "read_only_rootfs": False, "drop_all_caps": False,
                             "seccomp_profile": "runtime/default", "sandbox_profile": "none", "privileged": True, "host_paths": True})
    elif fault == "gpu_cross_tenant":
        runtime_data["gpu_partition"] = "shared-unisolated"
    elif fault == "secret_leak":
        runtime_data["secret_refs"] = ("secret://registry-pull", "value:supersecret")
    elif fault == "unexpected_egress":
        runtime_data["network_policy"] = "allow-all"

    artifact_result = verify_artifact(model)
    data_result = assess_model_data(dataset, model)
    dependency_result = scan_dependencies(dependencies)
    try:
        runtime_result = evaluate_runtime(RuntimeSpec(**runtime_data))
    except ValueError as exc:
        runtime_result = {"decision": "deny", "findings": ["runtime_configuration_invalid", str(exc)],
                          "effective_boundary": {}}
    reasons: list[str] = []
    if artifact_result["decision"] == "deny":
        reasons.append("artifact_evidence")
    if data_result["decision"] == "quarantine":
        reasons.append("model_data_poisoning")
    if dependency_result["decision"] == "deny":
        reasons.append("dependency_vulnerability")
    runtime_findings = runtime_result.get("findings", [])
    if runtime_result["decision"] == "deny":
        if any(item in runtime_findings for item in {"gpu_partition_not_isolated"}):
            reasons.append("gpu_multi_tenancy")
        if "secret_value_embedded" in runtime_findings:
            reasons.append("secret_handling")
        if "network_policy" in runtime_findings or "runtime_configuration_invalid" in runtime_findings:
            invalid_text = " ".join(str(item) for item in runtime_findings)
            reasons.append("network_policy" if "network" in invalid_text else "runtime_isolation")
        if any(item in runtime_findings for item in {"privileged_or_host_access", "writable_rootfs_or_capabilities", "sandbox_profile"}):
            reasons.append("runtime_isolation")
    if "model_data_poisoning" in reasons:
        decision = "quarantine"
    elif reasons:
        decision = "deny"
    else:
        decision = "admit"
    containment = "no_action" if decision == "admit" else ("freeze_release_and_revoke_promotion" if decision == "quarantine" else "block_deploy_and_open_incident")
    events, chain_valid = _audit_events(decision, reasons, fault)
    return {
        "schema_version": 1, "fault": fault, "decision": decision, "deny_reasons": reasons,
        "artifact": artifact_result, "model_data": {"quality": data_result, "poisoning": data_result},
        "dependencies": dependency_result, "runtime": runtime_result, "audit_events": events,
        "incident": {"containment": containment, "revocation": decision != "admit", "preserve_evidence": True},
        "invariants": {
            "all_required_evidence": artifact_result["decision"] == "allow" and data_result["decision"] in {"allow", "quarantine"}
                and dependency_result["decision"] in {"allow", "deny"} and "decision" in runtime_result,
            "audit_chain_valid": chain_valid,
            "no_secret_values_in_events": not _contains_secret(events),
            "fail_closed": decision in {"admit", "deny", "quarantine"},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = simulate(fault=args.fault)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
