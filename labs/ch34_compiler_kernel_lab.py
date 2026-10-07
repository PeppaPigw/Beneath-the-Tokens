#!/usr/bin/env python3
"""CPU-only compiler/kernel engineering toy for chapter 34.

This is a deterministic protocol model, not a Triton/CUDA/XLA benchmark.  It
models graph capture guards, pointwise fusion, compile-cache keys, an
interpretable launch/memory/compute cost model, and reference-vs-fused
correctness checks using only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


POINTWISE_OPS = {"add", "mul", "relu", "gelu"}
UNSUPPORTED_OPS = {"custom", "python_side_effect"}


@dataclass(frozen=True)
class Node:
    name: str
    op: str
    shape: tuple[int | None, ...]
    dtype: str = "fp32"
    layout: str = "contiguous"
    bytes: int = 0
    flops: int = 0
    side_effect: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True)
class CaptureResult:
    graphable: bool
    guards: tuple[str, ...]
    breaks: tuple[str, ...]


@dataclass(frozen=True)
class FusionGroup:
    nodes: tuple[str, ...]
    fused: bool
    reason: str


@dataclass(frozen=True)
class CompilePlan:
    cache_key: str
    cache_hit: bool
    backend: str
    shape_key: tuple[int | None, ...]
    dtype: str
    groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Cost:
    launches: int
    bytes: int
    flops: int
    compile_ms: float
    steady_ms: float
    total_first_ms: float


@dataclass(frozen=True)
class TuneResult:
    best: dict
    candidates: tuple[dict, ...]
    compile_budget_ms: float
    spent_ms: float
    stopped_early: bool


def _validate_nodes(nodes: Sequence[Node]) -> None:
    if not nodes:
        raise ValueError("graph must contain at least one node")
    for node in nodes:
        if not node.name or not node.op:
            raise ValueError("node name and op must be non-empty")
        if any(dim is not None and dim <= 0 for dim in node.shape):
            raise ValueError("shape dimensions must be positive or None")
        if node.dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError("toy supports fp32, bf16 and fp16")
        if node.layout not in {"contiguous", "strided"}:
            raise ValueError("toy supports contiguous or strided layout")
        if node.bytes < 0 or node.flops < 0:
            raise ValueError("bytes/flops must be non-negative")


def default_graph(*, dynamic_shape: bool = False, unsupported_op: bool = False) -> tuple[Node, ...]:
    shape: tuple[int | None, ...] = (None,) if dynamic_shape else (16,)
    nodes = (
        Node("add", "add", shape, bytes=16 * 4, flops=16),
        Node("mul", "mul", shape, bytes=16 * 4, flops=16),
        Node("relu", "relu", shape, bytes=16 * 4, flops=16),
    )
    if unsupported_op:
        nodes = nodes + (Node("custom", "custom", shape, bytes=16 * 4, flops=32, side_effect=True),)
    return nodes


def reduction_graph() -> tuple[Node, ...]:
    return (
        Node("add", "add", (16,), bytes=64, flops=16),
        Node("sum", "reduce_sum", (16,), bytes=64, flops=15),
        Node("relu", "relu", (16,), bytes=64, flops=16),
    )


def capture_graph(nodes: Sequence[Node], *, backend: str = "inductor") -> CaptureResult:
    """Return guards and breaks without invoking a real compiler."""
    _validate_nodes(nodes)
    if not backend:
        raise ValueError("backend must be non-empty")
    guards: list[str] = [f"backend={backend}"]
    breaks: list[str] = []
    shape = nodes[0].shape
    dtype = nodes[0].dtype
    layout = nodes[0].layout
    guards.extend([f"shape={shape}", f"dtype={dtype}", f"layout={layout}"])
    if any(dim is None for dim in shape):
        guards.append("dynamic_shape=true")
    for node in nodes:
        if node.op in UNSUPPORTED_OPS or node.side_effect:
            breaks.append(f"{node.name}:unsupported_or_side_effect")
        if node.op not in POINTWISE_OPS and node.op not in {"reduce_sum"} and node.op not in UNSUPPORTED_OPS:
            breaks.append(f"{node.name}:unknown_op")
        if node.shape != shape:
            breaks.append(f"{node.name}:shape_mismatch")
        if node.dtype != dtype:
            breaks.append(f"{node.name}:dtype_mismatch")
        if node.layout != layout:
            breaks.append(f"{node.name}:layout_mismatch")
    return CaptureResult(not breaks, tuple(guards), tuple(dict.fromkeys(breaks)))


def fuse_graph(nodes: Sequence[Node]) -> tuple[FusionGroup, ...]:
    """Fuse only adjacent pointwise nodes with identical metadata."""
    _validate_nodes(nodes)
    groups: list[FusionGroup] = []
    pending: list[str] = []
    pending_meta: tuple[tuple[int | None, ...], str, str] | None = None

    def flush(reason: str = "pointwise_chain") -> None:
        nonlocal pending, pending_meta
        if pending:
            groups.append(FusionGroup(tuple(pending), len(pending) > 1, reason if len(pending) == 1 else "pointwise_chain"))
        pending = []
        pending_meta = None

    for node in nodes:
        meta = (node.shape, node.dtype, node.layout)
        if node.op in POINTWISE_OPS and not node.side_effect and (pending_meta is None or pending_meta == meta):
            pending.append(node.name)
            pending_meta = meta
        else:
            flush("boundary:" + node.op)
            groups.append(FusionGroup((node.name,), False, "reduction_or_side_effect" if node.op not in POINTWISE_OPS or node.side_effect else "single"))
    flush()
    return tuple(groups)


def _key(backend: str, nodes: Sequence[Node], config: dict | None = None) -> str:
    config = config or {}
    descriptor = {
        "compiler": "ch34-toy-v1",
        "backend": backend,
        "nodes": [node.to_dict() for node in nodes],
        "config": config,
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compile_plan(nodes: Sequence[Node], *, backend: str = "inductor", cache: set[str] | None = None, config: dict | None = None) -> CompilePlan:
    _validate_nodes(nodes)
    cache = cache if cache is not None else set()
    groups = fuse_graph(nodes)
    key = _key(backend, nodes, config)
    hit = key in cache
    cache.add(key)
    return CompilePlan(key, hit, backend, nodes[0].shape, nodes[0].dtype, tuple(group.nodes for group in groups))


def estimate_cost(nodes: Sequence[Node], groups: Sequence[FusionGroup], *, cache_hit: bool, bandwidth_gb_s: float = 100.0, compute_gflop_s: float = 500.0, launch_us: float = 12.0) -> Cost:
    _validate_nodes(nodes)
    if bandwidth_gb_s <= 0 or compute_gflop_s <= 0 or launch_us < 0:
        raise ValueError("bandwidth/compute must be positive and launch non-negative")
    launches = len(groups)
    # Separate kernels materialize every node; fused groups keep intermediates in registers.
    separate_bytes = sum(node.bytes * 2 for node in nodes)
    fused_bytes = sum(max((nodes[i].bytes for i, _ in enumerate(nodes)), default=0) * 2 for _ in groups)
    # The max expression is intentionally conservative; payload is tiny in this toy.
    bytes_moved = min(separate_bytes, fused_bytes) if launches < len(nodes) else separate_bytes
    flops = sum(node.flops for node in nodes)
    memory_ms = bytes_moved / (bandwidth_gb_s * 1024**3) * 1000.0
    compute_ms = flops / (compute_gflop_s * 1e9) * 1000.0
    steady = launches * launch_us / 1000.0 + memory_ms + compute_ms
    compile_ms = 0.0 if cache_hit else 4.0 + 0.6 * len(groups)
    return Cost(launches, bytes_moved, flops, round(compile_ms, 6), round(steady, 6), round(compile_ms + steady, 6))


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def run_reference(values: Iterable[float], nodes: Sequence[Node]) -> list[float]:
    """Run a sequential 1-D reference for add/mul/relu/gelu.

    `add` adds 1 and `mul` multiplies by 2; these constants are part of the toy
    graph contract.  Reductions and unsupported ops intentionally fail.
    """
    _validate_nodes(nodes)
    out = [float(v) for v in values]
    for node in nodes:
        if node.op == "add":
            out = [v + 1.0 for v in out]
        elif node.op == "mul":
            out = [v * 2.0 for v in out]
        elif node.op == "relu":
            out = [max(0.0, v) for v in out]
        elif node.op == "gelu":
            out = [_gelu(v) for v in out]
        elif node.op == "reduce_sum":
            raise ValueError("reference toy keeps reduction as a graph boundary")
        else:
            raise ValueError(f"unsupported op in reference: {node.op}")
    return out


def run_fused(values: Iterable[float], nodes: Sequence[Node]) -> list[float]:
    """Compute the same pointwise chain in one loop to mimic a fused kernel."""
    _validate_nodes(nodes)
    if any(node.op not in POINTWISE_OPS for node in nodes):
        raise ValueError("run_fused accepts only pointwise nodes")
    # The loop is deliberately explicit so the toy remains inspectable.
    result: list[float] = []
    for original in values:
        x = float(original)
        for node in nodes:
            if node.op == "add":
                x += 1.0
            elif node.op == "mul":
                x *= 2.0
            elif node.op == "relu":
                x = max(0.0, x)
            elif node.op == "gelu":
                x = _gelu(x)
        result.append(x)
    return result


def max_abs_error(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    if not a:
        return 0.0
    return max(abs(x - y) for x, y in zip(a, b))


def autotune(*, budget_ms: float = 8.0, candidates: Sequence[dict] | None = None) -> TuneResult:
    """Choose a deterministic proxy winner and stop when the compile budget ends."""
    if budget_ms <= 0:
        raise ValueError("budget_ms must be positive")
    candidates = tuple(candidates or (
        {"block": 64, "warps": 1, "stages": 1},
        {"block": 128, "warps": 2, "stages": 1},
        {"block": 256, "warps": 4, "stages": 2},
        {"block": 512, "warps": 8, "stages": 3},
    ))
    seen: list[dict] = []
    spent = 0.0
    for candidate in candidates:
        cost = 0.8 + 0.25 * candidate["warps"] + 0.15 * candidate["stages"] + (abs(candidate["block"] - 192) / 512)
        if spent + cost > budget_ms and seen:
            break
        measured = dict(candidate)
        measured["proxy_ms"] = round(cost, 6)
        seen.append(measured)
        spent += cost
    if not seen:
        raise ValueError("budget too small to measure one candidate")
    best = min(seen, key=lambda item: item["proxy_ms"])
    return TuneResult(best, tuple(seen), round(budget_ms, 6), round(spent, 6), len(seen) < len(candidates))


def simulate(*, backend: str = "inductor", dynamic_shape: bool = False, unsupported_op: bool = False, fail_after: int | None = None, budget_ms: float = 8.0) -> dict:
    nodes = default_graph(dynamic_shape=dynamic_shape, unsupported_op=unsupported_op)
    capture = capture_graph(nodes, backend=backend)
    groups = fuse_graph(nodes)
    cache: set[str] = set()
    first = compile_plan(nodes, backend=backend, cache=cache)
    second = compile_plan(nodes, backend=backend, cache=cache)
    cost_first = estimate_cost(nodes, groups, cache_hit=first.cache_hit)
    cost_hot = estimate_cost(nodes, groups, cache_hit=second.cache_hit)
    values = [(-2.0 + i * 0.25) for i in range(16)]
    pointwise = tuple(node for node in nodes if node.op in POINTWISE_OPS)
    ref = run_reference(values, pointwise)
    opt = run_fused(values, pointwise)
    error = max_abs_error(ref, opt)
    tuning = autotune(budget_ms=budget_ms)
    failed_candidates = 0 if fail_after is None else max(0, min(fail_after, len(tuning.candidates)))
    fallback = failed_candidates > 0
    if fallback:
        # A failed candidate is rejected; reference remains the safe path.
        selected = {"mode": "reference_fallback", "failed_candidates": failed_candidates}
    else:
        selected = {"mode": "autotuned", **tuning.best}
    return {
        "schema_version": 1,
        "experiment": "ch34-compiler-kernel-cpu-toy",
        "config": {"backend": backend, "dynamic_shape": dynamic_shape, "unsupported_op": unsupported_op, "fail_after": fail_after, "budget_ms": budget_ms},
        "graph": {"nodes": [node.to_dict() for node in nodes], "guards": list(capture.guards), "breaks": list(capture.breaks), "graphable": capture.graphable},
        "fusion": {"groups": [asdict(group) for group in groups], "fused_group_count": sum(group.fused for group in groups)},
        "compile": {"first": asdict(first), "second": asdict(second), "cache_entries": len(cache)},
        "cost": {"first": asdict(cost_first), "hot": asdict(cost_hot)},
        "correctness": {"max_abs_error": error, "within_tolerance": error <= 1e-12, "values": len(values)},
        "autotune": {"result": asdict(tuning), "selected": selected, "fallback": fallback},
        "summary": {"cache_hit_on_second": second.cache_hit, "graph_break_count": len(capture.breaks), "fusion_reduced_launches": cost_first.launches < len(nodes), "correctness_pass": error <= 1e-12, "safe_fallback": fallback or error <= 1e-12},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--dynamic-shape", action="store_true")
    parser.add_argument("--unsupported-op", action="store_true")
    parser.add_argument("--fail-after", type=int, default=None)
    parser.add_argument("--budget-ms", type=float, default=8.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    payload = simulate(backend=args.backend, dynamic_shape=args.dynamic_shape, unsupported_op=args.unsupported_op, fail_after=args.fail_after, budget_ms=args.budget_ms)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
