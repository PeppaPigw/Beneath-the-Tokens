#!/usr/bin/env python3
"""CPU-only end-to-end AI infrastructure capstone simulation.

This is a teaching simulation. It models contracts, retries, a circuit breaker,
checkpoint publication/recovery, stage gates, and a small capacity ledger. It is
not a production load test or an SLO guarantee.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class RequestResult:
    request_id: int
    ok: bool
    degraded: bool
    retries: int
    latency_ms: float
    reason: str
    audit_writes: int


class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: int = 2) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self.state = "CLOSED"
        self.failures = 0
        self.opened_at = -1

    def allow(self, tick: int) -> bool:
        if self.state == "OPEN" and tick - self.opened_at >= self.cooldown:
            self.state = "HALF_OPEN"
        return self.state != "OPEN"

    def observe(self, ok: bool, tick: int) -> None:
        if ok:
            self.failures = 0
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
            return
        self.failures += 1
        if self.failures >= self.threshold:
            self.state = "OPEN"
            self.opened_at = tick


class AuditStore:
    """An idempotent write store keyed by request id."""

    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.physical_writes = 0

    def write_once(self, key: str) -> None:
        if key not in self.keys:
            self.keys.add(key)
            self.physical_writes += 1


def run_requests(n: int, seed: int, fault: str) -> Tuple[List[RequestResult], Dict[str, int | str]]:
    rng = random.Random(seed)
    breaker = CircuitBreaker()
    audit = AuditStore()
    results: List[RequestResult] = []
    tick = 0
    for rid in range(n):
        tick += 1
        if not breaker.allow(tick):
            results.append(RequestResult(rid, False, True, 0, 0.2, "circuit_open", 0))
            continue
        retries = 0
        latency = 3.0 + rng.random() * 3.0
        reason = "ok"
        degraded = False
        ok = True
        # A toy dependency failure is sticky enough to exercise breaker behavior.
        retrieval_bad = fault in {"retrieval", "all"} and rng.random() < 0.22
        model_bad = fault in {"model", "all"} and rng.random() < 0.18
        for attempt in range(3):
            if retrieval_bad:
                retries += 1
                latency += 2.0 * (attempt + 1)
                if attempt == 2:
                    # Retrieval is allowed to degrade to a bounded empty context.
                    degraded = True
                    reason = "retrieval_degraded"
                    retrieval_bad = False
                    break
                continue
            if model_bad:
                retries += 1
                latency += 5.0 * (attempt + 1)
                if attempt == 2:
                    ok = False
                    reason = "model_timeout"
                    break
                continue
            break
        if not ok:
            breaker.observe(False, tick)
            results.append(RequestResult(rid, False, False, retries, latency, reason, 0))
            continue
        # Audit write is idempotent even if the caller retries after a timeout.
        before = audit.physical_writes
        audit.write_once(f"req-{rid}")
        # Simulate a client retry after an ambiguous network response.
        if fault in {"audit", "all"} and rng.random() < 0.12:
            audit.write_once(f"req-{rid}")
            retries += 1
            latency += 1.0
        writes = audit.physical_writes - before
        breaker.observe(True, tick)
        results.append(RequestResult(rid, True, degraded, retries, latency, reason, writes))
    ok_count = sum(r.ok for r in results)
    degraded_count = sum(r.degraded for r in results)
    summary: Dict[str, int | str] = {
        "requests": n,
        "ok": ok_count,
        "failed": n - ok_count,
        "degraded": degraded_count,
        "retries": sum(r.retries for r in results),
        "physical_audit_writes": audit.physical_writes,
        "logical_ok": ok_count,
        "breaker_final": breaker.state,
    }
    return results, summary


def checkpoint_drill(root: Path, fault: str) -> Dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    versions = []
    for step in (10, 20, 30):
        tmp = root / f"step-{step}.manifest.tmp"
        final = root / f"step-{step}.manifest"
        tmp.write_text(json.dumps({"step": step, "checksum": f"sum-{step}"}) + "\n")
        if fault in {"checkpoint", "all"} and step == 30:
            # Crash before atomic publication leaves only a temporary file.
            continue
        tmp.replace(final)
        versions.append(step)
    valid = []
    for p in root.glob("step-*.manifest"):
        try:
            obj = json.loads(p.read_text())
            if obj.get("checksum") == f"sum-{obj['step']}":
                valid.append(int(obj["step"]))
        except (ValueError, KeyError, json.JSONDecodeError):
            pass
    recovered = max(valid) if valid else None
    return {"published": versions, "valid": sorted(valid), "recovered_step": recovered,
            "rpo_steps": (30 - recovered) if recovered is not None else None}


def stage_gates(summary: Dict[str, int | str], checkpoint: Dict[str, object]) -> List[Dict[str, object]]:
    gates = [
        ("G0-contract", ["request_schema", "SLO", "owner", "threat_model"]),
        ("G1-data", ["manifest", "lineage", "quality_report", "deletion_policy"]),
        ("G2-model", ["reproducible_run", "eval_report", "checkpoint_manifest"]),
        ("G3-service", ["load_test", "rollback", "dashboards", "runbook"]),
        ("G4-operate", ["chaos_evidence", "cost_report", "postmortem_template"]),
    ]
    # G0-G2 are document stubs in this teaching script. G3/G4 reflect the
    # simulated run, so a fault drill cannot accidentally look production-ready.
    statuses = {name: ("PASS", "document stub") for name, _ in gates}
    if int(summary["failed"]) > 0:
        statuses["G3-service"] = ("BLOCKED", "request failures observed")
    if int(summary["degraded"]) > 0:
        statuses["G3-service"] = ("BLOCKED", "degraded responses observed")
    if checkpoint.get("rpo_steps") not in (0, None):
        statuses["G2-model"] = ("BLOCKED", "checkpoint drill lost the latest step")
    if int(summary["physical_audit_writes"]) != int(summary["logical_ok"]):
        statuses["G4-operate"] = ("BLOCKED", "idempotency mismatch")
    return [{"gate": name, "evidence": evidence, "status": statuses[name][0],
             "reason": statuses[name][1]} for name, evidence in gates]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fault", choices=["none", "retrieval", "model", "audit", "checkpoint", "all"], default="all")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="ch26-capstone-") as td:
        results, summary = run_requests(args.requests, args.seed, args.fault)
        checkpoint = checkpoint_drill(Path(td) / "checkpoints", args.fault)
    report = {
        "teaching_simulation": True,
        "parameters": vars(args),
        "stage_gates": stage_gates(summary, checkpoint),
        "request_summary": summary,
        "checkpoint_drill": checkpoint,
        "sample_requests": [asdict(r) for r in results[:5]],
        "interpretation": [
            "The breaker bounds repeated model failures but does not prove availability.",
            "The manifest drill demonstrates recovery to the last complete version.",
            "Idempotency keeps physical audit writes at most once per request key.",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
