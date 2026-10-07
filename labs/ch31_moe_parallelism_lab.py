#!/usr/bin/env python3
"""CPU-only MoE routing, capacity and expert-parallel toy.

This lab models token routing rather than running a transformer.  It keeps the
choices that matter in production inspectable: top-k selection, per-expert
capacity and overflow policy, Switch/GShard-style load-balance loss, and a
rank-local all-to-all schedule.  It is deterministic and standard-library
only; numbers are not GPU benchmarks.
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
class RoutingConfig:
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    overflow_policy: str = "drop"  # drop, second, residual
    num_ranks: int = 4
    seed: int = 31


@dataclass(frozen=True)
class RoutingResult:
    assignments: tuple[tuple[int, ...], ...]
    expert_load: tuple[int, ...]
    expert_prob: tuple[float, ...]
    selected_fraction: tuple[float, ...]
    dropped_tokens: int
    overflow_assignments: int
    capacity: int
    aux_loss: float


@dataclass(frozen=True)
class AllToAllPlan:
    send_matrix: tuple[tuple[int, ...], ...]
    recv_matrix: tuple[tuple[int, ...], ...]
    bytes_by_src: tuple[int, ...]
    bytes_by_dst: tuple[int, ...]
    rounds: int
    max_bytes: int
    imbalance_ratio: float


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("logits must not be empty")
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid logits")
    return [x / total for x in exps]


def _validate_config(cfg: RoutingConfig) -> None:
    if cfg.num_experts <= 0 or cfg.num_ranks <= 0:
        raise ValueError("num_experts and num_ranks must be positive")
    if cfg.num_experts % cfg.num_ranks:
        raise ValueError("num_experts must be divisible by num_ranks")
    if cfg.top_k <= 0 or cfg.top_k > cfg.num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if cfg.capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")
    if cfg.overflow_policy not in {"drop", "second", "residual"}:
        raise ValueError("overflow_policy must be drop, second, or residual")


def route_tokens(logits: Sequence[Sequence[float]], cfg: RoutingConfig | None = None) -> RoutingResult:
    """Route token logits with deterministic tie-breaking and hard capacity.

    Capacity is ceil(capacity_factor * ceil(tokens * top_k / experts)).
    Primary selections are considered in token order; overflow_policy=second
    tries the next selected expert, while residual records a token as dropped
    but models a dense residual path.  The latter still reports all-to-all
    traffic only for accepted expert assignments.
    """
    cfg = cfg or RoutingConfig()
    _validate_config(cfg)
    if not logits:
        raise ValueError("logits must not be empty")
    if any(len(row) != cfg.num_experts for row in logits):
        raise ValueError("each logit row must have num_experts entries")
    tokens = len(logits)
    capacity = max(1, math.ceil(cfg.capacity_factor * math.ceil(tokens * cfg.top_k / cfg.num_experts)))
    probs = [softmax(row) for row in logits]
    selected = [tuple(sorted(range(cfg.num_experts), key=lambda e: (-row[e], e))[:cfg.top_k]) for row in logits]
    accepted: list[list[int]] = [[] for _ in logits]
    load = [0] * cfg.num_experts
    overflow = 0
    dropped = 0
    for token_id, choices in enumerate(selected):
        for slot, expert in enumerate(choices):
            # A fallback selected for an earlier overflowing slot may be the
            # later primary choice; count the token once for that expert.
            if expert in accepted[token_id]:
                continue
            if load[expert] < capacity:
                accepted[token_id].append(expert)
                load[expert] += 1
                continue
            overflow += 1
            fallback = None
            if cfg.overflow_policy == "second":
                for alt in choices[slot + 1:]:
                    if load[alt] < capacity and alt not in accepted[token_id]:
                        fallback = alt
                        break
            if fallback is not None:
                accepted[token_id].append(fallback)
                load[fallback] += 1
            else:
                dropped += 1
    selected_fraction = tuple(sum(1 for choices in accepted if e in choices) / tokens for e in range(cfg.num_experts))
    expert_prob = tuple(sum(p[e] for p in probs) / tokens for e in range(cfg.num_experts))
    # Switch/GShard-style proxy: E * sum(f_i * p_i), where f_i is selected
    # fraction.  It equals one for an ideal uniform assignment in expectation.
    aux = cfg.num_experts * sum(f * p for f, p in zip(selected_fraction, expert_prob))
    return RoutingResult(tuple(tuple(x) for x in accepted), tuple(load), expert_prob,
                         selected_fraction, dropped, overflow, capacity, aux)


def make_logits(tokens: int, experts: int, seed: int = 31, hot_expert: int | None = None, hot_bias: float = 0.0) -> list[list[float]]:
    if tokens <= 0 or experts <= 0:
        raise ValueError("tokens and experts must be positive")
    rng = random.Random(seed)
    rows: list[list[float]] = []
    for t in range(tokens):
        row = [rng.uniform(-1.0, 1.0) + 0.05 * ((t + e) % 3) for e in range(experts)]
        if hot_expert is not None:
            if not 0 <= hot_expert < experts:
                raise ValueError("hot_expert out of range")
            row[hot_expert] += hot_bias
        rows.append(row)
    return rows


def build_all_to_all(assignments: Sequence[Sequence[int]], num_ranks: int, num_experts: int,
                     bytes_per_token: int = 4096) -> AllToAllPlan:
    if num_ranks <= 0 or num_experts <= 0 or num_experts % num_ranks:
        raise ValueError("invalid rank/expert topology")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    experts_per_rank = num_experts // num_ranks
    matrix = [[0 for _ in range(num_ranks)] for _ in range(num_ranks)]
    for src, choices in enumerate(assignments):
        src_rank = src % num_ranks
        for expert in choices:
            if not 0 <= expert < num_experts:
                raise ValueError("expert id out of range")
            dst_rank = expert // experts_per_rank
            matrix[src_rank][dst_rank] += 1
    recv = [list(row) for row in zip(*matrix)]
    bytes_src = tuple(sum(row) * bytes_per_token for row in matrix)
    bytes_dst = tuple(sum(row) * bytes_per_token for row in recv)
    nonzero = [n for row in matrix for n in row if n]
    rounds = max((sum(1 for n in row if n) for row in matrix), default=0)
    max_bytes = max(bytes_src + bytes_dst, default=0)
    mean = statistics.fmean(nonzero) if nonzero else 0.0
    imbalance = (max(nonzero) / mean) if mean else 0.0
    return AllToAllPlan(tuple(tuple(row) for row in matrix), tuple(tuple(row) for row in recv),
                        bytes_src, bytes_dst, rounds, max_bytes, imbalance)


def simulate(cfg: RoutingConfig | None = None, *, tokens: int = 256, hidden_bytes: int = 4096,
             hot_expert: int | None = None, hot_bias: float = 0.0) -> dict:
    cfg = cfg or RoutingConfig()
    _validate_config(cfg)
    logits = make_logits(tokens, cfg.num_experts, cfg.seed, hot_expert, hot_bias)
    routed = route_tokens(logits, cfg)
    plan = build_all_to_all(routed.assignments, cfg.num_ranks, cfg.num_experts, hidden_bytes)
    accepted = sum(routed.expert_load)
    # A transparent latency proxy: local compute plus all-to-all payload and
    # a synchronization penalty for the busiest rank.  Not a benchmark.
    compute_ms = 0.02 * accepted / cfg.num_experts
    comm_ms = plan.max_bytes / (50 * 1024 * 1024)  # 50 MiB/s toy link
    sync_ms = 0.03 * plan.imbalance_ratio
    return {
        "schema_version": 1,
        "experiment": "ch31-moe-routing-expert-parallel-toy",
        "config": asdict(cfg) | {"tokens": tokens, "hidden_bytes": hidden_bytes,
                                  "hot_expert": hot_expert, "hot_bias": hot_bias},
        "routing": asdict(routed),
        "all_to_all": asdict(plan),
        "summary": {
            "tokens": tokens,
            "accepted_assignments": accepted,
            "drop_rate": routed.dropped_tokens / max(1, tokens),
            "overflow_assignments": routed.overflow_assignments,
            "aux_loss": routed.aux_loss,
            "all_to_all_bytes": sum(plan.bytes_by_src),
            "latency_proxy_ms": round(compute_ms + comm_ms + sync_ms, 6),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--capacity-factor", type=float, default=1.25)
    p.add_argument("--overflow-policy", choices=["drop", "second", "residual"], default="drop")
    p.add_argument("--ranks", type=int, default=4)
    p.add_argument("--hidden-bytes", type=int, default=4096)
    p.add_argument("--hot-expert", type=int)
    p.add_argument("--hot-bias", type=float, default=0.0)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    cfg = RoutingConfig(num_experts=args.experts, top_k=args.top_k, capacity_factor=args.capacity_factor,
                        overflow_policy=args.overflow_policy, num_ranks=args.ranks, seed=args.seed)
    payload = simulate(cfg, tokens=args.tokens, hidden_bytes=args.hidden_bytes,
                       hot_expert=args.hot_expert, hot_bias=args.hot_bias)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
