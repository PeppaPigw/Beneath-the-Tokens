#!/usr/bin/env python3
"""Chapter 22 CPU-only security lab; Python standard library only.

This toy demonstrates integrity tags, a manifest admission check, tenant quotas,
redaction and incident ordering. It is deliberately not a replacement for
signatures, a KMS, kernel isolation, or a privacy proof.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


def canonical(obj: Any) -> bytes:
    """Serialize with deterministic key ordering for the toy manifest."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def mac(obj: Any, key: bytes) -> str:
    return hmac.new(key, canonical(obj), hashlib.sha256).hexdigest()


def verify_mac(obj: Any, tag: str, key: bytes) -> bool:
    return hmac.compare_digest(mac(obj, key), tag)


EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")


def redact(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    return text


@dataclass
class Job:
    tenant: str
    job_id: str
    cost: int
    text: str


class TenantLimiter:
    def __init__(self, per_tenant: dict[str, int], global_limit: int) -> None:
        self.per_tenant = dict(per_tenant)
        self.used: defaultdict[str, int] = defaultdict(int)
        self.global_limit = global_limit
        self.global_used = 0
        self.accepted: deque[Job] = deque()
        self.rejected: list[Job] = []

    def submit(self, job: Job) -> bool:
        allowed = (
            job.tenant in self.per_tenant
            and self.used[job.tenant] + job.cost <= self.per_tenant[job.tenant]
            and self.global_used + job.cost <= self.global_limit
        )
        if not allowed:
            self.rejected.append(job)
            return False
        self.used[job.tenant] += job.cost
        self.global_used += job.cost
        self.accepted.append(job)
        return True


def main() -> None:
    key = b"toy-demo-key-do-not-use-in-production"
    manifest = {
        "model": "demo-1.0",
        "weights_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "source_commit": "deadbeef",
        "builder": "ci.example/build@v3",
        "sbom": ["python:3.12", "numpy:2.0"],
    }
    tag = mac(manifest, key)
    print("manifest sha256:", digest(manifest))
    print("hmac verified:", verify_mac(manifest, tag, key))
    altered = dict(manifest, weights_sha256="c" * 64)
    print("tampered hmac rejected:", not verify_mac(altered, tag, key))

    limiter = TenantLimiter({"tenant-a": 5, "tenant-b": 5}, global_limit=7)
    jobs = [
        Job("tenant-a", "a1", 3, "alice@example.com asks for 13800138000"),
        Job("tenant-a", "a2", 3, "second expensive request"),
        Job("tenant-b", "b1", 3, "bob@example.com asks for 13900139000"),
    ]
    for job in jobs:
        accepted = limiter.submit(job)
        print("quota", job.job_id, "accepted" if accepted else "rejected")
    print("accepted cost by tenant:", dict(limiter.used))
    print("rejected jobs:", [j.job_id for j in limiter.rejected])

    raw = "contact alice@example.com or 13800138000; tenant=tenant-a"
    clean = redact(raw)
    print("redacted:", clean)
    assert "alice@example.com" not in clean and "13800138000" not in clean

    events = [
        {"ts": 3, "kind": "admission_denied", "reason": "missing_provenance"},
        {"ts": 1, "kind": "token_issued", "subject": "train-a", "ttl_s": 300},
        {"ts": 2, "kind": "tenant_quota_reject", "tenant": "tenant-a"},
    ]
    print("incident timeline:")
    for event in sorted(events, key=lambda e: e["ts"]):
        print(json.dumps(event, ensure_ascii=False, sort_keys=True))

    candidate = {"provenance": True, "sbom": True, "signature": True}
    print("admission:", "allow" if all(candidate.values()) else "deny")
    candidate["provenance"] = False
    print("missing provenance:", "deny" if not all(candidate.values()) else "allow")

    print("BOUNDARY: HMAC is not a digital signature; quotas are not kernel isolation;")
    print("BOUNDARY: regex redaction is not anonymization; a manifest is not a safety proof.")


if __name__ == "__main__":
    main()
