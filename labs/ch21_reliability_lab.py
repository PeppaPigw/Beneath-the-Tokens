#!/usr/bin/env python3
"""CPU-only reliability toy for Chapter 21.

The toy intentionally models two ambiguous failure points:
- fail_before_commit: the worker fails before its side effect is committed.
- fail_after_commit: the side effect is committed, but the response is lost.

An idempotency ledger prevents the second case from duplicating a side effect.
A circuit breaker bounds repeated attempts while a checkpoint manifest protects
recovery from a truncated file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Record:
    batch_id: int
    digest: str
    state: str  # PENDING, COMMITTED, FAILED
    result: Optional[str] = None


class IdempotencyLedger:
    def __init__(self) -> None:
        self.records: Dict[str, Record] = {}
        self.side_effects = 0

    def begin(self, key: str, batch_id: int, digest: str) -> Record:
        old = self.records.get(key)
        if old is not None:
            if old.batch_id != batch_id or old.digest != digest:
                raise ValueError(f"idempotency conflict for {key}")
            return old
        rec = Record(batch_id=batch_id, digest=digest, state="PENDING")
        self.records[key] = rec
        return rec

    def commit(self, key: str, result: str) -> Record:
        rec = self.records[key]
        if rec.state != "COMMITTED":
            # This is the single externally visible side effect in the toy.
            self.side_effects += 1
            rec.state = "COMMITTED"
            rec.result = result
        return rec


class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: float = 0.04) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.state = "CLOSED"
        self.opened_at = 0.0

    def allow(self, now: float) -> bool:
        if self.state == "OPEN" and now - self.opened_at >= self.cooldown:
            self.state = "HALF_OPEN"
            return True
        return self.state != "OPEN"

    def success(self) -> None:
        self.failures = 0
        self.state = "CLOSED"

    def failure(self, now: float) -> None:
        self.failures += 1
        if self.state == "HALF_OPEN" or self.failures >= self.threshold:
            self.state = "OPEN"
            self.opened_at = now


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest: Optional[Path] = None

    @staticmethod
    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def publish(self, step: int, committed: list[int], truncate: bool = False) -> Path:
        tmp = self.root / f"step-{step}.tmp"
        final = self.root / f"step-{step}"
        if tmp.exists():
            shutil.rmtree(tmp)
        if final.exists():
            shutil.rmtree(final)
        tmp.mkdir()
        state = tmp / "state.json"
        state.write_text(json.dumps({"step": step, "committed": committed}, sort_keys=True), encoding="utf-8")
        weights = tmp / "weights.bin"
        weights.write_bytes(("weights-at-step-" + str(step)).encode("utf-8") * 32)
        manifest = {
            "step": step,
            "files": {
                p.name: {"bytes": p.stat().st_size, "sha256": self.digest(p)}
                for p in sorted(tmp.iterdir())
            },
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        if truncate:
            # Simulate a torn write after the manifest was prepared. The
            # checksum must now fail during recovery.
            data = weights.read_bytes()
            weights.write_bytes(data[: max(1, len(data) // 3)])
        # Local rename is the toy's commit point. A real object store needs a
        # conditional manifest/pointer protocol instead.
        tmp.rename(final)
        self.latest = final
        return final

    def recover(self) -> tuple[int, list[int]]:
        candidates = sorted(
            self.root.glob("step-*"),
            key=lambda p: int(p.name.split("-")[1].split(".")[0]) if p.name.split("-")[1].split(".")[0].isdigit() else -1,
            reverse=True,
        )
        for directory in candidates:
            if not directory.is_dir() or ".tmp" in directory.name:
                continue
            try:
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                for name, meta in manifest["files"].items():
                    p = directory / name
                    if p.stat().st_size != meta["bytes"] or self.digest(p) != meta["sha256"]:
                        raise ValueError(f"checksum mismatch: {p}")
                state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
                return int(state["step"]), list(state["committed"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return 0, []


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    ledger = IdempotencyLedger()
    breaker = CircuitBreaker(args.breaker_threshold, args.cooldown)
    attempts = 0
    retries = 0
    rejected = 0
    committed: list[int] = []
    events: list[str] = []
    root = Path(tempfile.mkdtemp(prefix="ch21-ckpt-"))
    store = CheckpointStore(root)
    start = time.monotonic()

    try:
        for batch_id in range(args.batches):
            key = f"batch-{batch_id}"
            digest = hashlib.sha256(f"payload-{batch_id}".encode()).hexdigest()
            done = False
            for retry_no in range(args.max_retries + 1):
                now = time.monotonic()
                if not breaker.allow(now):
                    rejected += 1
                    events.append(f"batch={batch_id} breaker=OPEN reject")
                    time.sleep(args.cooldown)
                    continue
                attempts += 1
                rec = ledger.begin(key, batch_id, digest)
                if rec.state == "COMMITTED":
                    done = True
                    events.append(f"batch={batch_id} idempotent-hit result={rec.result}")
                    breaker.success()
                    break
                # Failure before commit is safe to retry; failure after commit
                # represents a lost response and must be resolved by the ledger.
                if rng.random() < args.fail_before_commit:
                    breaker.failure(time.monotonic())
                    events.append(f"batch={batch_id} try={retry_no} fail=before_commit breaker={breaker.state}")
                else:
                    result = f"result-{batch_id}"
                    ledger.commit(key, result)
                    if rng.random() < args.fail_after_commit:
                        breaker.failure(time.monotonic())
                        events.append(f"batch={batch_id} try={retry_no} fail=after_commit")
                    else:
                        breaker.success()
                        done = True
                        events.append(f"batch={batch_id} try={retry_no} committed")
                if done:
                    break
                retries += 1
                wait = min(args.backoff_cap, args.backoff * (2**retry_no))
                # Full jitter: deterministic under --seed while avoiding herd wakeups.
                time.sleep(rng.uniform(0.0, wait))
            if not done:
                events.append(f"batch={batch_id} terminal=FAILED")
            elif batch_id in [x for x in range(args.batches) if (x + 1) % args.checkpoint_every == 0]:
                store.publish(batch_id + 1, sorted(ledger.records[k].batch_id for k in ledger.records if ledger.records[k].state == "COMMITTED"))
                committed = sorted(ledger.records[k].batch_id for k in ledger.records if ledger.records[k].state == "COMMITTED")

        # Make a deliberately invalid latest checkpoint when requested. The
        # recovery reader should reject it and use the previous valid one.
        if args.truncate_checkpoint:
            store.publish(args.batches + 1, committed, truncate=True)
            events.append("checkpoint=truncated published-for-test")
        recovered_step, recovered_batches = store.recover()
        elapsed = time.monotonic() - start
        print("events:")
        for e in events:
            print("  " + e)
        print("summary:")
        print(json.dumps({
            "batches": args.batches,
            "committed": committed,
            "attempts": attempts,
            "retries": retries,
            "retry_amplification": round(attempts / max(args.batches, 1), 3),
            "rejected_by_breaker": rejected,
            "side_effects": ledger.side_effects,
            "duplicate_side_effects": max(0, ledger.side_effects - len(committed)),
            "recovered_step": recovered_step,
            "recovered_batches": recovered_batches,
            "elapsed_seconds": round(elapsed, 4),
            "checkpoint_root": str(root),
        }, ensure_ascii=False, sort_keys=True))
    finally:
        if not args.keep_tmp:
            shutil.rmtree(root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--fail-before-commit", type=float, default=0.18)
    p.add_argument("--fail-after-commit", type=float, default=0.12)
    p.add_argument("--checkpoint-every", type=int, default=3)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--breaker-threshold", type=int, default=3)
    p.add_argument("--cooldown", type=float, default=0.04)
    p.add_argument("--backoff", type=float, default=0.003)
    p.add_argument("--backoff-cap", type=float, default=0.03)
    p.add_argument("--truncate-checkpoint", action="store_true")
    p.add_argument("--keep-tmp", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
