#!/usr/bin/env python3
"""CPU-only incident toy for chapter 20.

Simulates a bounded worker pool where a shared lock can create a long tail.
This is a teaching model; it does not represent GPU or production capacity.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Full


@dataclass
class Event:
    request_id: int
    arrived: float
    started: float
    finished: float
    status: str


def run_case(
    n: int = 200,
    workers: int = 4,
    slow_ratio: float = 0.0,
    lock_hold: float = 0.0,
    queue_limit: int = 64,
    seed: int = 7,
) -> dict:
    rng = random.Random(seed)
    gate = threading.Lock()
    work: Queue[int] = Queue(maxsize=queue_limit)
    events: list[Event] = []
    events_lock = threading.Lock()

    def worker() -> None:
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                return
            rid, arrived, slow = item
            with gate:
                started = time.monotonic()
                if slow:
                    time.sleep(lock_hold)
                time.sleep(0.002)
            finished = time.monotonic()
            with events_lock:
                events.append(Event(rid, arrived, started, finished, "ok"))
            work.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()

    rejected = 0
    with ThreadPoolExecutor(max_workers=1) as producer:
        for rid in range(n):
            arrived = time.monotonic()
            try:
                work.put_nowait((rid, arrived, rng.random() < slow_ratio))
            except Full:
                rejected += 1
            # Keep arrivals deterministic but nonzero, making queueing visible.
            time.sleep(0.0005)

    work.join()
    for _ in threads:
        work.put(None)
    for t in threads:
        t.join()

    latencies = sorted(e.finished - e.arrived for e in events)
    if not latencies:
        return {"accepted": 0, "rejected": rejected}
    p95_idx = max(0, int(0.95 * len(latencies)) - 1)
    return {
        "accepted": len(latencies),
        "rejected": rejected,
        "p50_s": statistics.median(latencies),
        "p95_s": latencies[p95_idx],
        "max_s": max(latencies),
        "lock_hold_s": lock_hold,
        "slow_ratio": slow_ratio,
        "workers": workers,
        "queue_limit": queue_limit,
        "events": [e.__dict__ for e in sorted(events, key=lambda x: x.request_id)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incident", action="store_true", help="increase slow-task ratio and lock hold")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    result = run_case(
        n=args.requests,
        workers=args.workers,
        slow_ratio=0.35 if args.incident else 0.0,
        lock_hold=0.02 if args.incident else 0.0,
        seed=args.seed,
    )
    # Keep stdout machine-readable for reproducible comparisons.
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
