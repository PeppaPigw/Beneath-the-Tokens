#!/usr/bin/env python3
"""CPU-only KV-cache transfer and connector laboratory.

This is a protocol simulator, not a network benchmark.  It models bytes,
paging, handshakes, checksum retries, and a few connector policies so that a
reader can inspect the causal chain without a GPU, RDMA NIC, or external
service.  Python 3.10+ and the standard library are sufficient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class KVShape:
    layers: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    bytes_per_element: int = 2

    def bytes_for_tokens(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        # K and V are both stored, hence the factor of two.
        return 2 * self.layers * tokens * self.kv_heads * self.head_dim * self.bytes_per_element


@dataclass(frozen=True)
class Page:
    request_id: str
    page_index: int
    token_start: int
    token_count: int
    payload_bytes: int
    digest: str


@dataclass(frozen=True)
class ConnectorProfile:
    name: str
    bandwidth_gbps: float
    setup_ms: float
    per_page_ms: float
    loss_rate: float = 0.0
    max_retries: int = 2


@dataclass
class TransferResult:
    connector: str
    request_id: str
    pages: int
    bytes_sent: int
    retries: int
    elapsed_ms: float
    states: list[str]
    ok: bool
    error: str | None = None


PROFILES = {
    # Values are intentionally toy parameters. They are not hardware claims.
    "memcpy": ConnectorProfile("memcpy", bandwidth_gbps=80.0, setup_ms=0.03, per_page_ms=0.01),
    "tcp": ConnectorProfile("tcp", bandwidth_gbps=12.0, setup_ms=0.35, per_page_ms=0.05, loss_rate=0.02),
    "nixl": ConnectorProfile("nixl", bandwidth_gbps=45.0, setup_ms=0.20, per_page_ms=0.025, loss_rate=0.01),
    "mooncake": ConnectorProfile("mooncake", bandwidth_gbps=32.0, setup_ms=0.25, per_page_ms=0.035, loss_rate=0.015),
    "lmcache": ConnectorProfile("lmcache", bandwidth_gbps=28.0, setup_ms=0.18, per_page_ms=0.03, loss_rate=0.01),
}


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def make_pages(request_id: str, tokens: int, page_tokens: int, shape: KVShape) -> list[Page]:
    if page_tokens <= 0:
        raise ValueError("page_tokens must be positive")
    pages: list[Page] = []
    for page_index, start in enumerate(range(0, tokens, page_tokens)):
        count = min(page_tokens, tokens - start)
        size = shape.bytes_for_tokens(count)
        # Avoid allocating the entire KV payload. The digest models a payload
        # generated from stable request/page metadata.
        seed = f"{request_id}:{page_index}:{size}".encode()
        pages.append(Page(request_id, page_index, start, count, size, _digest(seed)))
    return pages


def _attempt_ok(page: Page, attempt: int, profile: ConnectorProfile, rng: random.Random) -> bool:
    # Deterministic first-attempt corruption switch makes --seed reproducible.
    if profile.loss_rate <= 0:
        return True
    marker = rng.random()
    return marker >= profile.loss_rate or attempt > 0


def transfer_pages(
    pages: Iterable[Page],
    profile: ConnectorProfile,
    *,
    rng: random.Random,
    verify_checksum: bool = True,
) -> TransferResult:
    pages = list(pages)
    request_id = pages[0].request_id if pages else "empty"
    states = ["INIT", "NEGOTIATED", "MEMORY_REGISTERED", "TRANSFERRING"]
    retries = 0
    sent = 0
    elapsed = profile.setup_ms
    for page in pages:
        sent += page.payload_bytes
        transfer_ms = page.payload_bytes * 8 / (profile.bandwidth_gbps * 1e9) * 1e3
        elapsed += profile.per_page_ms + transfer_ms
        ok = False
        for attempt in range(profile.max_retries + 1):
            if _attempt_ok(page, attempt, profile, rng):
                ok = True
                break
            retries += 1
            elapsed += profile.per_page_ms
        if not ok:
            states.append("ABORTED")
            return TransferResult(profile.name, request_id, len(pages), sent, retries, elapsed, states, False, "checksum_mismatch")
        if verify_checksum and len(page.digest) != 64:
            states.append("ABORTED")
            return TransferResult(profile.name, request_id, len(pages), sent, retries, elapsed, states, False, "invalid_digest")
    states.append("COMMITTED")
    return TransferResult(profile.name, request_id, len(pages), sent, retries, elapsed, states, True)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[index]


def simulate(
    *,
    seed: int,
    requests: int,
    tokens: int,
    page_tokens: int,
    connector_names: list[str],
    shape: KVShape,
    lmcache_hit_rate: float = 0.0,
) -> dict:
    rng = random.Random(seed)
    request_rows: list[dict] = []
    by_connector: dict[str, list[TransferResult]] = {name: [] for name in connector_names}
    for i in range(requests):
        request_id = f"req-{i:04d}"
        # A hit means that the prefix is already local and no transfer occurs.
        hit = rng.random() < lmcache_hit_rate
        pages = make_pages(request_id, tokens, page_tokens, shape)
        for name in connector_names:
            if name not in PROFILES:
                raise KeyError(f"unknown connector: {name}")
            if hit:
                result = TransferResult(name, request_id, len(pages), 0, 0, 0.0, ["CACHE_HIT", "COMMITTED"], True)
            else:
                result = transfer_pages(pages, PROFILES[name], rng=rng)
            by_connector[name].append(result)
        request_rows.append({"request_id": request_id, "cache_hit": hit, "pages": len(pages)})
    summary = {}
    for name, rows in by_connector.items():
        elapsed = [r.elapsed_ms for r in rows]
        summary[name] = {
            "requests": len(rows),
            "cache_hits": sum(1 for r in rows if "CACHE_HIT" in r.states),
            "bytes_sent": sum(r.bytes_sent for r in rows),
            "retries": sum(r.retries for r in rows),
            "failed": sum(not r.ok for r in rows),
            "p50_ms": percentile(elapsed, 0.50),
            "p95_ms": percentile(elapsed, 0.95),
            "mean_ms": statistics.fmean(elapsed) if elapsed else 0.0,
        }
    return {
        "schema_version": 1,
        "seed": seed,
        "requests": requests,
        "tokens": tokens,
        "page_tokens": page_tokens,
        "kv_shape": asdict(shape),
        "kv_bytes_per_request": shape.bytes_for_tokens(tokens),
        "lmcache_hit_rate": lmcache_hit_rate,
        "connectors": summary,
        "request_rows": request_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--page-tokens", type=int, default=128)
    parser.add_argument("--connectors", default="memcpy,tcp,nixl,mooncake,lmcache")
    parser.add_argument("--lmcache-hit-rate", type=float, default=0.25)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    if not 0.0 <= args.lmcache_hit_rate <= 1.0:
        parser.error("--lmcache-hit-rate must be in [0, 1]")
    result = simulate(
        seed=args.seed,
        requests=args.requests,
        tokens=args.tokens,
        page_tokens=args.page_tokens,
        connector_names=[x.strip() for x in args.connectors.split(",") if x.strip()],
        shape=KVShape(),
        lmcache_hit_rate=args.lmcache_hit_rate,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
