#!/usr/bin/env python3
"""CPU-only toy lab for Chapter 25 AI Infra frontiers.

Standard library only. The numbers are illustrative measurements of the toy model,
not hardware claims.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Device:
    name: str
    latency_ms: float
    joule_per_token: float
    capacity: int


def route(tokens: int, experts: int, top_k: int, rng: random.Random) -> list[int]:
    """Random top-k routing; returns per-expert token assignments."""
    if not (1 <= top_k <= experts):
        raise ValueError("top_k must be in [1, experts]")
    loads = [0] * experts
    for _ in range(tokens):
        scores = [rng.random() for _ in range(experts)]
        picks = sorted(range(experts), key=scores.__getitem__, reverse=True)[:top_k]
        for idx in picks:
            loads[idx] += 1
    return loads


def cv(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def schedule(
    tokens: int, devices: Sequence[Device], latency_weight: float, rng: random.Random
) -> dict[str, float | int]:
    """Greedy scheduler with a scalar latency/energy objective."""
    capacities = {d.name: d.capacity for d in devices}
    completion_ms: list[float] = []
    energy_j = 0.0
    rejected = 0
    for _ in range(tokens):
        candidates = [d for d in devices if capacities[d.name] > 0]
        if not candidates:
            rejected += 1
            continue
        # Small random noise avoids deterministic ties while preserving ordering.
        def score(d: Device) -> float:
            return (
                latency_weight * d.latency_ms
                + (1.0 - latency_weight) * d.joule_per_token * 100.0
                + rng.random() * 1e-3
            )

        chosen = min(candidates, key=score)
        capacities[chosen.name] -= 1
        completion_ms.append(chosen.latency_ms)
        energy_j += chosen.joule_per_token
    return {
        "p95_latency_ms": round(_quantile(completion_ms, 0.95), 4) if completion_ms else None,
        "mean_latency_ms": round(statistics.fmean(completion_ms), 4) if completion_ms else None,
        "energy_j": round(energy_j, 6),
        "completed": len(completion_ms),
        "rejected": rejected,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def fedavg(
    clients: Sequence[Sequence[float]], clip: float | None, noise: float, rng: random.Random
) -> tuple[list[float], list[float]]:
    """FedAvg with optional per-client L2 clipping and Gaussian noise."""
    if not clients:
        raise ValueError("clients cannot be empty")
    clipped: list[list[float]] = []
    norms: list[float] = []
    for vec in clients:
        norm = math.sqrt(sum(x * x for x in vec))
        norms.append(norm)
        scale = min(1.0, clip / max(norm, 1e-12)) if clip is not None else 1.0
        clipped.append([x * scale for x in vec])
    dim = len(clipped[0])
    if any(len(v) != dim for v in clipped):
        raise ValueError("client vectors must have equal dimensions")
    n = len(clipped)
    result = [sum(v[j] for v in clipped) / n for j in range(dim)]
    if noise:
        result = [x + rng.gauss(0.0, noise) for x in result]
    return result, norms


def l2(vec: Iterable[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def kv_bytes(layers: int, tokens: int, kv_heads: int, head_dim: int, bytes_per: int) -> int:
    """2 (key/value) * layers * tokens * heads * dim * bytes."""
    return 2 * layers * tokens * kv_heads * head_dim * bytes_per


def carbon_grams(energy_j: float, carbon_g_per_kwh: float) -> float:
    return energy_j / 3_600_000.0 * carbon_g_per_kwh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tokens", type=int, default=20_000)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--clients", type=int, default=12)
    parser.add_argument("--carbon-g-per-kwh", type=float, default=400.0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    # 1) Sparse routing.
    routing = {}
    for top_k in (1, 2):
        loads = route(args.tokens, args.experts, top_k, rng)
        routing[f"top_{top_k}"] = {
            "loads": loads,
            "min_load": min(loads),
            "max_load": max(loads),
            "cv": round(cv(loads), 6),
            "assignments": sum(loads),
        }

    # 2) Heterogeneous scheduling under two objective weights.
    # Each device has enough nominal capacity so the objective changes placement.
    devices = (
        Device("edge_npu", latency_ms=3.0, joule_per_token=0.08, capacity=args.tokens),
        Device("gpu", latency_ms=1.3, joule_per_token=0.20, capacity=args.tokens),
        Device("cpu", latency_ms=8.0, joule_per_token=0.05, capacity=args.tokens),
    )
    scheduling = {
        "latency_priority": schedule(args.tokens, devices, latency_weight=0.9, rng=rng),
        "energy_priority": schedule(args.tokens, devices, latency_weight=0.1, rng=rng),
    }
    pressure_devices = tuple(
        Device(d.name, d.latency_ms, d.joule_per_token, max(1, args.tokens // 5)) for d in devices
    )
    scheduling["capacity_pressure"] = schedule(
        args.tokens, pressure_devices, latency_weight=0.5, rng=rng
    )

    # 3) Federated averaging with non-IID client means.
    clients: list[list[float]] = []
    for client_id in range(args.clients):
        mean = (client_id - (args.clients - 1) / 2.0) * 0.35
        clients.append([rng.gauss(mean, 0.25) for _ in range(8)])
    raw, raw_norms = fedavg(clients, clip=None, noise=0.0, rng=rng)
    clipped, clipped_norms = fedavg(clients, clip=1.0, noise=0.0, rng=rng)
    private, private_norms = fedavg(clients, clip=1.0, noise=0.08, rng=rng)
    federated = {
        "client_norm_min": round(min(raw_norms), 6),
        "client_norm_max": round(max(raw_norms), 6),
        "raw_norm": round(l2(raw), 6),
        "clipped_norm": round(l2(clipped), 6),
        "private_norm": round(l2(private), 6),
        "clip": 1.0,
        "noise_std": 0.08,
    }

    # 4) KV cache and illustrative energy/carbon estimate.
    layers, kv_heads, head_dim, bytes_per = 32, 8, 128, 2
    lengths = [2_048, 8_192, 32_768, 131_072]
    kv = {str(t): kv_bytes(layers, t, kv_heads, head_dim, bytes_per) for t in lengths}
    # Use the latency-priority completed token energy as a transparent toy estimate.
    energy_j = float(scheduling["latency_priority"]["energy_j"])
    memory_energy = {"joules": round(energy_j, 6), "carbon_g": round(carbon_grams(energy_j, args.carbon_g_per_kwh), 8)}

    # Property checks keep the experiment honest.
    checks = {
        "routing_assignments_monotonic": routing["top_2"]["assignments"] >= routing["top_1"]["assignments"],
        "kv_monotonic": all(kv[str(a)] < kv[str(b)] for a, b in zip(lengths, lengths[1:])),
        "clipping_bound": max(min(n, 1.0) for n in clipped_norms) <= 1.0 + 1e-9,
        "capacity_rejects_nonnegative": all(v["rejected"] >= 0 for v in scheduling.values()),
        "capacity_pressure_rejects": scheduling["capacity_pressure"]["rejected"] > 0,
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    result = {
        "seed": args.seed,
        "tokens": args.tokens,
        "experts": args.experts,
        "routing": routing,
        "scheduling": scheduling,
        "federated": federated,
        "kv_cache_bytes": kv,
        "energy_carbon_estimate": memory_energy,
        "checks": checks,
        "note": "Toy CPU measurements; not a hardware performance or privacy guarantee.",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
