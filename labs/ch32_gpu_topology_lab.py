#!/usr/bin/env python3
"""CPU-only GPU-cluster topology, collective and failure-boundary toy.

This module deliberately models *contracts*, not CUDA/NCCL performance.  It
keeps rank placement, path classes, ring traffic, and failure scopes explicit
so a newcomer can inspect which assumptions are safe to carry into production.
Python 3.10+ and the standard library are sufficient.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TopologyConfig:
    """Logical cluster shape and link assumptions for the toy model."""

    nodes: int = 2
    gpus_per_node: int = 4
    intra_bw_gbps: float = 900.0
    inter_bw_gbps: float = 200.0
    intra_latency_us: float = 2.0
    inter_latency_us: float = 8.0
    replicas: int = 2

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    def validate(self) -> None:
        if self.nodes <= 0 or self.gpus_per_node <= 0:
            raise ValueError("nodes and gpus_per_node must be positive")
        if min(self.intra_bw_gbps, self.inter_bw_gbps) <= 0:
            raise ValueError("bandwidths must be positive")
        if min(self.intra_latency_us, self.inter_latency_us) < 0:
            raise ValueError("latencies must be non-negative")
        if self.replicas <= 0 or self.replicas > self.nodes:
            raise ValueError("replicas must be in [1, nodes]")


@dataclass(frozen=True)
class LinkMetric:
    src: int
    dst: int
    path: str
    bytes: int
    bandwidth_gbps: float
    latency_us: float
    transfer_ms: float


@dataclass(frozen=True)
class CollectivePlan:
    operation: str
    payload_bytes: int
    world_size: int
    links: tuple[LinkMetric, ...]
    rank_send_bytes: tuple[int, ...]
    inter_node_bytes: int
    intra_node_bytes: int
    bottleneck_ms: float
    p50_link_ms: float
    p95_link_ms: float

    @property
    def total_bytes(self) -> int:
        return sum(link.bytes for link in self.links)


@dataclass(frozen=True)
class FailureResult:
    scope: str
    target: int
    affected_ranks: tuple[int, ...]
    training_collective_survives: bool
    inference_replicas_available: int
    dropped_requests: int
    explanation: str


def _path(cfg: TopologyConfig, src: int, dst: int) -> tuple[str, float, float]:
    if src == dst:
        return "self", 0.0, 0.0
    src_node, dst_node = src // cfg.gpus_per_node, dst // cfg.gpus_per_node
    if src_node == dst_node:
        return "intra_node", cfg.intra_bw_gbps, cfg.intra_latency_us
    return "inter_node", cfg.inter_bw_gbps, cfg.inter_latency_us


def rank_to_node(cfg: TopologyConfig) -> tuple[int, ...]:
    cfg.validate()
    return tuple(rank // cfg.gpus_per_node for rank in range(cfg.world_size))


def topology_matrix(cfg: TopologyConfig) -> tuple[tuple[str, ...], ...]:
    """Return a stable path-class matrix, useful for audits and tests."""
    cfg.validate()
    return tuple(tuple(_path(cfg, i, j)[0] for j in range(cfg.world_size)) for i in range(cfg.world_size))


def _link(src: int, dst: int, payload: int, cfg: TopologyConfig) -> LinkMetric:
    path, bw, latency = _path(cfg, src, dst)
    if path == "self":
        return LinkMetric(src, dst, path, 0, 0.0, 0.0, 0.0)
    # Decimal gigabit is intentional: this is a transparent estimate, not a
    # claim about a vendor's usable payload bandwidth.
    transfer_ms = latency / 1000.0 + (payload * 8.0 / (bw * 1_000_000_000.0)) * 1000.0
    return LinkMetric(src, dst, path, payload, bw, latency, transfer_ms)


def _percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def plan_alltoall(cfg: TopologyConfig, payload_bytes: int) -> CollectivePlan:
    """Model one all-to-all phase with equal bytes to each peer.

    The payload is the aggregate bytes each rank must send.  Self traffic is
    omitted; each non-self peer receives an equal share.  This exposes how
    topology alone changes inter-node bytes and the slowest link.
    """
    cfg.validate()
    if payload_bytes <= 0:
        raise ValueError("payload_bytes must be positive")
    world = cfg.world_size
    peer_payload = payload_bytes // (world - 1) if world > 1 else 0
    remainder = payload_bytes - peer_payload * max(0, world - 1)
    links: list[LinkMetric] = []
    rank_send: list[int] = []
    for src in range(world):
        sent = 0
        extra_left = remainder
        for dst in range(world):
            if src == dst:
                continue
            amount = peer_payload
            if extra_left > 0:
                amount += 1
                extra_left -= 1
            links.append(_link(src, dst, amount, cfg))
            sent += amount
        rank_send.append(sent)
    times = [x.transfer_ms for x in links]
    inter = sum(x.bytes for x in links if x.path == "inter_node")
    intra = sum(x.bytes for x in links if x.path == "intra_node")
    return CollectivePlan("alltoall", payload_bytes, world, tuple(links), tuple(rank_send),
                          inter, intra, max(times, default=0.0), _percentile(times, .50),
                          _percentile(times, .95))


def plan_ring_allreduce(cfg: TopologyConfig, payload_bytes: int) -> CollectivePlan:
    """Model a ring all-reduce's two traversals over logical rank order."""
    cfg.validate()
    if payload_bytes <= 0:
        raise ValueError("payload_bytes must be positive")
    world = cfg.world_size
    if world == 1:
        return CollectivePlan("ring_allreduce", payload_bytes, 1, tuple(), (0,), 0, 0, 0.0, 0.0, 0.0)
    chunk = max(1, math.ceil(payload_bytes / world))
    links: list[LinkMetric] = []
    # Reduce-scatter + all-gather: every ring edge is traversed world-1 times
    # in each direction.  We account for two chunks per traversal.
    for src in range(world):
        dst = (src + 1) % world
        for _ in range(2 * (world - 1)):
            links.append(_link(src, dst, chunk, cfg))
    rank_send = tuple(sum(x.bytes for x in links if x.src == rank) for rank in range(world))
    times = [x.transfer_ms for x in links]
    inter = sum(x.bytes for x in links if x.path == "inter_node")
    intra = sum(x.bytes for x in links if x.path == "intra_node")
    return CollectivePlan("ring_allreduce", payload_bytes, world, tuple(links), rank_send,
                          inter, intra, max(times, default=0.0), _percentile(times, .50),
                          _percentile(times, .95))


def failure_injection(cfg: TopologyConfig, *, scope: str, target: int, requests: int = 0) -> FailureResult:
    """Describe blast radius without pretending a job can heal itself.

    ``scope`` is ``gpu`` (one rank), ``node`` (all ranks on a node), or
    ``switch`` (all nodes in the toy fabric).  A synchronous training
    collective requires every rank, while replicated inference can continue
    when at least one replica node remains.
    """
    cfg.validate()
    if requests < 0:
        raise ValueError("requests must be non-negative")
    if scope not in {"gpu", "node", "switch"}:
        raise ValueError("scope must be gpu, node, or switch")
    if scope == "gpu":
        if not 0 <= target < cfg.world_size:
            raise ValueError("gpu target out of range")
        affected = (target,)
        failed_nodes = {target // cfg.gpus_per_node}
        explanation = "one rank is missing; a synchronous all-reduce cannot advance"
    elif scope == "node":
        if not 0 <= target < cfg.nodes:
            raise ValueError("node target out of range")
        affected = tuple(range(target * cfg.gpus_per_node, (target + 1) * cfg.gpus_per_node))
        failed_nodes = {target}
        explanation = "all ranks on the node are unavailable; collective membership changes"
    else:
        affected = tuple(range(cfg.world_size))
        failed_nodes = set(range(cfg.nodes))
        explanation = "the fabric switch failure partitions every node in this toy"
    available = max(0, cfg.nodes - len(failed_nodes))
    available_replicas = min(cfg.replicas, available)
    dropped = requests if available_replicas == 0 else 0
    return FailureResult(scope, target, affected, False, available_replicas, dropped, explanation)


def simulate(*, seed: int = 32, cfg: TopologyConfig | None = None,
             payload_bytes: int = 64 * 1024 * 1024, requests: int = 128,
             failure_scope: str | None = None, failure_target: int = 0) -> dict:
    """Run deterministic training/inference boundary scenarios."""
    cfg = cfg or TopologyConfig()
    cfg.validate()
    if seed < 0 or requests < 0:
        raise ValueError("seed must be non-negative and requests non-negative")
    # Seed is retained in the contract even though the balanced toy has no
    # random placement.  Production schedulers must record placement seed.
    alltoall = plan_alltoall(cfg, payload_bytes)
    allreduce = plan_ring_allreduce(cfg, payload_bytes)
    failure = None
    if failure_scope is not None:
        failure = failure_injection(cfg, scope=failure_scope, target=failure_target, requests=requests)
    available = cfg.replicas if failure is None else failure.inference_replicas_available
    served = requests if available > 0 else 0
    return {
        "schema_version": 1,
        "experiment": "ch32-gpu-cluster-topology-cpu-toy",
        "seed": seed,
        "config": asdict(cfg) | {"payload_bytes": payload_bytes, "requests": requests},
        "placement": {"rank_to_node": rank_to_node(cfg), "path_matrix": topology_matrix(cfg)},
        "collectives": {"alltoall": asdict(alltoall), "ring_allreduce": asdict(allreduce)},
        "failure": asdict(failure) if failure else None,
        "inference": {"replicas_configured": cfg.replicas, "replicas_available": available,
                      "requests": requests, "served": served, "dropped": requests - served},
        "summary": {
            "world_size": cfg.world_size,
            "alltoall_inter_node_bytes": alltoall.inter_node_bytes,
            "alltoall_bottleneck_ms": round(alltoall.bottleneck_ms, 6),
            "ring_inter_node_bytes": allreduce.inter_node_bytes,
            "ring_bottleneck_ms": round(allreduce.bottleneck_ms, 6),
            "training_survives_failure": bool(failure is None),
            "inference_drop_rate": (requests - served) / max(1, requests),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--payload-mib", type=int, default=64)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--failure-scope", choices=["gpu", "node", "switch"])
    parser.add_argument("--failure-target", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = TopologyConfig(nodes=args.nodes, gpus_per_node=args.gpus_per_node, replicas=args.replicas)
    payload = simulate(cfg=cfg, payload_bytes=args.payload_mib * 1024 * 1024,
                       requests=args.requests, failure_scope=args.failure_scope,
                       failure_target=args.failure_target)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
