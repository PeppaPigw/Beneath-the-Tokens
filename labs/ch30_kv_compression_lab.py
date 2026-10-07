#!/usr/bin/env python3
"""CPU-only KV-cache compression, tiering, admission and eviction toy.

This laboratory is a deterministic protocol model, not a GPU benchmark.  It
makes three things inspectable without CUDA: per-group quantisation error,
low-rank/sparse payload accounting, and an online hot/warm/cold cache policy.
Python 3.10+ and the standard library are sufficient.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class KVShape:
    layers: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    bytes_per_element: int = 2

    def values_for_tokens(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        # K and V are separate tensors; hence the factor two.
        return 2 * self.layers * tokens * self.kv_heads * self.head_dim

    def baseline_bytes(self, tokens: int) -> int:
        return self.values_for_tokens(tokens) * self.bytes_per_element


@dataclass(frozen=True)
class CompressionPlan:
    name: str
    bits: int
    group_size: int = 64
    rank: int = 0
    sparsity: float = 0.0
    scale_bytes: int = 2
    zero_point_bytes: int = 0
    dequant_ms_per_token: float = 0.001


@dataclass(frozen=True)
class TierSpec:
    name: str
    capacity_bytes: int
    latency_ms: float
    plan: CompressionPlan


@dataclass
class PageRecord:
    page_id: str
    tokens: int
    accesses: int = 0
    last_access: int = -1
    tier: str | None = None
    evictions: int = 0
    quality_loss: float = 0.0

    def score(self, now: int) -> float:
        age = max(0, now - self.last_access)
        # A deliberately transparent recency/frequency utility.  A real
        # policy should calibrate this to quality and recompute cost.
        return (self.accesses + 0.5) / (1.0 + age)


def _validate_bits(bits: int) -> None:
    if bits not in {2, 4, 8, 16}:
        raise ValueError("bits must be one of 2, 4, 8, 16")


def quantize(values: Sequence[float], bits: int, group_size: int = 64) -> tuple[list[float], float, float]:
    """Symmetric per-group quantise/dequantise; returns reconstruction, MSE, max error."""
    _validate_bits(bits)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    out: list[float] = []
    qmax = (1 << (bits - 1)) - 1
    for start in range(0, len(values), group_size):
        group = list(values[start:start + group_size])
        if not group:
            continue
        scale = max(abs(v) for v in group) / qmax if qmax else 1.0
        if scale == 0:
            out.extend([0.0] * len(group))
            continue
        for value in group:
            integer = max(-qmax, min(qmax, int(round(value / scale))))
            out.append(integer * scale)
    errors = [(a - b) ** 2 for a, b in zip(values, out)]
    return out, statistics.fmean(errors) if errors else 0.0, max((abs(a - b) for a, b in zip(values, out)), default=0.0)


def sparse_reconstruct(values: Sequence[float], keep_ratio: float) -> tuple[list[float], float, int]:
    """Keep the largest-magnitude entries, zeroing the rest."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    count = max(1, math.ceil(len(values) * keep_ratio)) if values else 0
    keep = set(sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)[:count])
    out = [v if i in keep else 0.0 for i, v in enumerate(values)]
    mse = statistics.fmean([(a - b) ** 2 for a, b in zip(values, out)]) if values else 0.0
    return out, mse, sum(1 for x in out if x != 0.0)


def low_rank_error(singular_values: Sequence[float], rank: int) -> float:
    """Normalised squared tail energy for a truncated low-rank representation."""
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if not singular_values:
        return 0.0
    total = sum(v * v for v in singular_values)
    if total == 0:
        return 0.0
    return sum(v * v for v in singular_values[rank:]) / total


def plan_quality_loss(plan: CompressionPlan) -> float:
    """A bounded, synthetic error model used only for policy comparisons."""
    _validate_bits(plan.bits)
    if plan.group_size <= 0:
        raise ValueError("group_size must be positive")
    if not 0.0 <= plan.sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    quant_error = 0.12 * (16 - plan.bits) / 16
    sparse_error = 0.18 * plan.sparsity
    # A decaying toy spectrum makes rank loss explicit; rank=0 means dense.
    rank_error = 0.0 if plan.rank <= 0 else low_rank_error([1.0, 0.55, 0.28, 0.14, 0.07, 0.03], plan.rank)
    return min(0.99, max(0.0, quant_error + sparse_error + 0.35 * rank_error))


def compressed_bytes(shape: KVShape, tokens: int, plan: CompressionPlan) -> int:
    """Payload plus scale/index metadata for one token range."""
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    _validate_bits(plan.bits)
    if plan.group_size <= 0:
        raise ValueError("group_size must be positive")
    values = shape.values_for_tokens(tokens)
    if values == 0:
        return 0
    effective_values = values
    if plan.rank > 0:
        if plan.rank > shape.head_dim:
            raise ValueError("rank cannot exceed head_dim")
        # A shared basis is assumed by this toy, so only rank coefficients are
        # charged.  Basis storage is deliberately reported separately in text.
        effective_values = math.ceil(values * plan.rank / shape.head_dim)
    keep = 1.0 - plan.sparsity
    payload_values = math.ceil(effective_values * keep)
    payload = math.ceil(payload_values * plan.bits / 8)
    groups = math.ceil(payload_values / plan.group_size)
    metadata = groups * (plan.scale_bytes + plan.zero_point_bytes)
    # Two-byte indices approximate sparse coordinate overhead; this is why
    # sparsity does not always reduce bytes at tiny pages.
    index_bytes = math.ceil(payload_values * 2) if plan.sparsity > 0 else 0
    return payload + metadata + index_bytes


def percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def _make_tiers(shape: KVShape, page_tokens: int, hbm_bytes: int, dram_bytes: int, ssd_bytes: int) -> dict[str, TierSpec]:
    return {
        "hbm": TierSpec("hbm", hbm_bytes, 0.08, CompressionPlan("fp16", 16, dequant_ms_per_token=0.0005)),
        "dram": TierSpec("dram", dram_bytes, 0.45, CompressionPlan("int8", 8, dequant_ms_per_token=0.0015)),
        "ssd": TierSpec("ssd", ssd_bytes, 3.5, CompressionPlan("int4", 4, dequant_ms_per_token=0.004)),
    }


def simulate(
    *,
    seed: int = 30,
    requests: int = 64,
    pages: int = 48,
    accesses: int = 600,
    page_tokens: int = 32,
    hbm_bytes: int = 12_000_000,
    dram_bytes: int = 6_000_000,
    ssd_bytes: int = 40_000_000,
    admission_threshold: float = 0.20,
) -> dict:
    if requests <= 0 or pages <= 0 or accesses <= 0:
        raise ValueError("requests, pages and accesses must be positive")
    if page_tokens <= 0:
        raise ValueError("page_tokens must be positive")
    if min(hbm_bytes, dram_bytes, ssd_bytes) <= 0:
        raise ValueError("tier capacities must be positive")
    shape = KVShape()
    tiers = _make_tiers(shape, page_tokens, hbm_bytes, dram_bytes, ssd_bytes)
    records = {f"page-{i:03d}": PageRecord(f"page-{i:03d}", page_tokens) for i in range(pages)}
    rng = random.Random(seed)
    used = {name: 0 for name in tiers}
    hits = {name: 0 for name in tiers}
    misses = 0
    admissions = 0
    evictions = 0
    promotions = 0
    demotions = 0
    latency: list[float] = []
    quality: list[float] = []
    trace: list[dict] = []

    def page_bytes(rec: PageRecord, tier_name: str) -> int:
        return compressed_bytes(shape, rec.tokens, tiers[tier_name].plan)

    def remove(rec: PageRecord) -> None:
        nonlocal evictions
        if rec.tier is None:
            return
        used[rec.tier] -= page_bytes(rec, rec.tier)
        rec.tier = None
        rec.evictions += 1
        evictions += 1

    def choose_victim(tier_name: str, now: int) -> PageRecord | None:
        candidates = [r for r in records.values() if r.tier == tier_name]
        return min(candidates, key=lambda r: (r.score(now), r.page_id), default=None)

    def place(rec: PageRecord, target: str, now: int) -> bool:
        nonlocal promotions, demotions
        size = page_bytes(rec, target)
        if size > tiers[target].capacity_bytes:
            return False
        if rec.tier == target:
            return True
        old = rec.tier
        if old is not None:
            used[old] -= page_bytes(rec, old)
        # First make room. HBM and DRAM spill down one level; SSD simply evicts.
        while used[target] + size > tiers[target].capacity_bytes:
            victim = choose_victim(target, now)
            if victim is None:
                break
            if target == "hbm":
                if not place(victim, "dram", now):
                    remove(victim)
                else:
                    demotions += 1
            elif target == "dram":
                if not place(victim, "ssd", now):
                    remove(victim)
                else:
                    demotions += 1
            else:
                remove(victim)
        if used[target] + size > tiers[target].capacity_bytes:
            if old is not None:
                used[old] += page_bytes(rec, old)
                rec.tier = old
            return False
        rec.tier = target
        used[target] += size
        if old is not None:
            if tiers[target].latency_ms < tiers[old].latency_ms:
                promotions += 1
            elif tiers[target].latency_ms > tiers[old].latency_ms:
                demotions += 1
        return True

    # Zipf-like popularity is generated online: hot pages are repeatedly seen,
    # while the tail still receives enough requests to exercise admission.
    weights = [1.0 / (1.0 + i * 0.16) for i in range(pages)]
    for step in range(accesses):
        request_id = f"req-{step % requests:03d}"
        page_id = rng.choices(list(records), weights=weights, k=1)[0]
        rec = records[page_id]
        rec.accesses += 1
        rec.last_access = step
        if rec.tier is None:
            misses += 1
            base_latency = 6.0  # source recompute/read, not a hardware claim
            score = rec.score(step)
            if score >= admission_threshold:
                # New pages enter the cold tier; repeated accesses promote them.
                if place(rec, "ssd", step):
                    admissions += 1
            latency.append(base_latency)
            quality.append(0.0)
            event = "MISS_ADMIT" if rec.tier else "MISS_BYPASS"
        else:
            tier_name = rec.tier
            tier = tiers[tier_name]
            hits[tier_name] += 1
            latency.append(tier.latency_ms + tier.plan.dequant_ms_per_token * rec.tokens)
            loss = plan_quality_loss(tier.plan)
            rec.quality_loss = loss
            quality.append(loss)
            event = f"HIT_{tier_name.upper()}"
            # Promotion is deliberately gated by repeated observations.
            if tier_name == "ssd" and rec.accesses >= 2:
                place(rec, "dram", step)
            elif tier_name == "dram" and rec.accesses >= 3:
                place(rec, "hbm", step)
        if step < 20:
            trace.append({"step": step, "request_id": request_id, "page_id": page_id, "event": event, "tier": rec.tier})

    baseline = shape.baseline_bytes(page_tokens) * pages
    resident = sum(used.values())
    compressed_capacity = sum(compressed_bytes(shape, page_tokens, tier.plan) * pages for tier in tiers.values())
    return {
        "schema_version": 1,
        "experiment": "ch30-kv-compression-tiering-toy",
        "config": {"seed": seed, "requests": requests, "pages": pages, "accesses": accesses, "page_tokens": page_tokens,
                   "hbm_bytes": hbm_bytes, "dram_bytes": dram_bytes, "ssd_bytes": ssd_bytes,
                   "admission_threshold": admission_threshold},
        "shape": asdict(shape),
        "plans": {name: {"bytes_per_page": compressed_bytes(shape, page_tokens, tier.plan),
                         "quality_loss": plan_quality_loss(tier.plan), "plan": asdict(tier.plan)}
                  for name, tier in tiers.items()},
        "summary": {
            "baseline_bytes_all_pages": baseline,
            "theoretical_compressed_bytes_all_tiers": compressed_capacity,
            "resident_bytes": resident,
            "compression_ratio_vs_fp16": round(resident / baseline, 6) if baseline else 0.0,
            "requests": requests,
            "accesses": accesses,
            "hits_by_tier": hits,
            "misses": misses,
            "hit_rate": round(sum(hits.values()) / accesses, 6),
            "admissions": admissions,
            "evictions": evictions,
            "promotions": promotions,
            "demotions": demotions,
            "latency_ms": {"p50": round(percentile(latency, 0.50), 6), "p95": round(percentile(latency, 0.95), 6)},
            "quality_loss_mean": round(statistics.fmean(quality), 8) if quality else 0.0,
            "quality_loss_p95": round(percentile(quality, 0.95), 8),
            "used_bytes_by_tier": used,
        },
        "trace_first_20": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--pages", type=int, default=48)
    parser.add_argument("--accesses", type=int, default=600)
    parser.add_argument("--page-tokens", type=int, default=32)
    parser.add_argument("--hbm-bytes", type=int, default=12_000_000)
    parser.add_argument("--dram-bytes", type=int, default=6_000_000)
    parser.add_argument("--ssd-bytes", type=int, default=40_000_000)
    parser.add_argument("--admission-threshold", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.admission_threshold < 0:
        parser.error("--admission-threshold must be non-negative")
    payload = simulate(seed=args.seed, requests=args.requests, pages=args.pages, accesses=args.accesses,
                       page_tokens=args.page_tokens, hbm_bytes=args.hbm_bytes, dram_bytes=args.dram_bytes,
                       ssd_bytes=args.ssd_bytes, admission_threshold=args.admission_threshold)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
