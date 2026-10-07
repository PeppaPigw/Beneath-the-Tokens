#!/usr/bin/env python3
"""CPU-only storage and data-plane toy for chapter 33.

The lab is a deterministic protocol model.  It does not claim to measure S3,
Lustre, NVMe, POSIX or checkpoint throughput.  Instead it makes shard
assignment, backend latency, manifest commit and failure evidence inspectable
with Python's standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BackendSpec:
    name: str
    bandwidth_mb_s: float
    latency_ms: float
    list_delay_ticks: int = 0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("backend name must not be empty")
        if self.bandwidth_mb_s <= 0 or self.latency_ms < 0:
            raise ValueError("backend bandwidth must be positive and latency non-negative")
        if self.list_delay_ticks < 0:
            raise ValueError("list_delay_ticks must be non-negative")


@dataclass(frozen=True)
class ReadMeasurement:
    backend: str
    objects: int
    bytes: int
    elapsed_ms: float
    throughput_mb_s: float


@dataclass(frozen=True)
class CheckpointResult:
    step: int
    shard_count: int
    written_shards: int
    manifest_hash: str
    committed: bool
    recoverable: bool
    reason: str


OBJECTS_PER_WORKER = 8


def _backends() -> dict[str, BackendSpec]:
    # Transparent proxies, not vendor claims.  Values keep a CPU run quick.
    return {
        "object": BackendSpec("object", bandwidth_mb_s=180.0, latency_ms=8.0, list_delay_ticks=1),
        "posix": BackendSpec("posix", bandwidth_mb_s=420.0, latency_ms=1.4, list_delay_ticks=0),
        "nvme": BackendSpec("nvme", bandwidth_mb_s=1200.0, latency_ms=0.08, list_delay_ticks=0),
    }


def _hash_bytes(label: str, size: int) -> str:
    # Hashing the label plus size represents content identity without allocating
    # large buffers.  Production checksums must hash actual payload bytes.
    return hashlib.sha256(f"{label}:{size}".encode("utf-8")).hexdigest()


def assign_shards(shards: int, workers: int, *, seed: int = 33) -> tuple[tuple[int, ...], ...]:
    """Deterministically assign each shard exactly once using round-robin.

    ``seed`` is part of the audit contract.  This balanced toy intentionally
    does not randomize; production samplers should record a real shuffle seed.
    """
    if shards <= 0 or workers <= 0:
        raise ValueError("shards and workers must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    groups = [[] for _ in range(workers)]
    for shard_id in range(shards):
        groups[shard_id % workers].append(shard_id)
    return tuple(tuple(group) for group in groups)


def estimate_reads(backend: BackendSpec, shard_count: int, shard_mib: int) -> ReadMeasurement:
    """Estimate serialized read time from startup latency plus payload time."""
    backend.validate()
    if shard_count < 0 or shard_mib <= 0:
        raise ValueError("shard_count must be non-negative and shard_mib positive")
    total_bytes = shard_count * shard_mib * 1024 * 1024
    elapsed = shard_count * backend.latency_ms + (shard_count * shard_mib / backend.bandwidth_mb_s) * 1000.0
    throughput = (total_bytes / (1024 * 1024)) / (elapsed / 1000.0) if elapsed > 0 else 0.0
    return ReadMeasurement(backend.name, shard_count, total_bytes, round(elapsed, 6), round(throughput, 6))


def checkpoint_commit(*, step: int, shards: int, fail_after: int | None = None) -> CheckpointResult:
    """Write shard records then commit one manifest marker.

    A reader may recover only if all shard hashes and the manifest are present;
    a directory full of shard files without a commit marker is incomplete.
    """
    if step < 0 or shards <= 0:
        raise ValueError("step must be non-negative and shards positive")
    if fail_after is not None and not 0 <= fail_after <= shards:
        raise ValueError("fail_after must be in [0, shards]")
    written = shards if fail_after is None else fail_after
    entries = [{"name": f"part-{i:05d}", "bytes": 1024 * (i + 1), "sha256": _hash_bytes(f"step-{step}-part-{i}", 1024 * (i + 1))} for i in range(written)]
    manifest = {"step": step, "shards": shards, "entries": entries}
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    committed = written == shards and fail_after is None
    recoverable = committed and len(entries) == shards
    reason = "commit marker durable after all shard hashes" if committed else "incomplete shards; reader must ignore uncommitted prefix"
    return CheckpointResult(step, shards, written, manifest_hash, committed, recoverable, reason)


def simulate(*, seed: int = 33, shards: int = 12, workers: int = 3, shard_mib: int = 4,
             checkpoint_step: int = 100, fail_after: int | None = None) -> dict:
    if shards <= 0 or workers <= 0 or shard_mib <= 0:
        raise ValueError("shards, workers and shard_mib must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    backends = _backends()
    assignments = assign_shards(shards, workers, seed=seed)
    reads = {name: asdict(estimate_reads(spec, shards, shard_mib)) for name, spec in backends.items()}
    ckpt = checkpoint_commit(step=checkpoint_step, shards=shards, fail_after=fail_after)
    flat = [shard for group in assignments for shard in group]
    unique = len(set(flat)) == shards and len(flat) == shards
    return {
        "schema_version": 1,
        "experiment": "ch33-storage-data-plane-cpu-toy",
        "seed": seed,
        "config": {"shards": shards, "workers": workers, "shard_mib": shard_mib,
                   "checkpoint_step": checkpoint_step, "fail_after": fail_after},
        "backends": reads,
        "assignment": {"worker_shards": assignments, "all_shards_unique": unique,
                       "coverage": len(flat)},
        "checkpoint": asdict(ckpt),
        "summary": {
            "fastest_backend": min(reads, key=lambda name: reads[name]["elapsed_ms"]),
            "object_elapsed_ms": reads["object"]["elapsed_ms"],
            "posix_elapsed_ms": reads["posix"]["elapsed_ms"],
            "nvme_elapsed_ms": reads["nvme"]["elapsed_ms"],
            "checkpoint_committed": ckpt.committed,
            "checkpoint_recoverable": ckpt.recoverable,
            "uncommitted_write_is_visible": not ckpt.committed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--shard-mib", type=int, default=4)
    parser.add_argument("--checkpoint-step", type=int, default=100)
    parser.add_argument("--fail-after", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = simulate(seed=args.seed, shards=args.shards, workers=args.workers,
                       shard_mib=args.shard_mib, checkpoint_step=args.checkpoint_step,
                       fail_after=args.fail_after)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
