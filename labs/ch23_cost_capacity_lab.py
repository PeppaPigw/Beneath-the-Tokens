#!/usr/bin/env python3
"""Chapter 23 toy lab: queueing, autoscaling, cost and carbon.
Standard library only; no production billing or power API calls.
"""
import argparse
import math
import random
from dataclasses import dataclass


@dataclass
class Job:
    arrival: float
    service: float
    done: float | None = None


def poisson_arrivals(rate: float, horizon: float, rng: random.Random):
    t = 0.0
    out = []
    while t < horizon:
        t += rng.expovariate(rate)
        if t <= horizon:
            out.append(t)
    return out


def simulate(rate=3.0, base_workers=2, max_workers=6,
             service_rate=2.0, horizon=120.0, seed=7):
    rng = random.Random(seed)
    arrivals = poisson_arrivals(rate, horizon, rng)
    queue = []
    workers = [0.0] * base_workers
    jobs = []
    scale_events = []
    i = 0
    now = 0.0
    while i < len(arrivals) or queue or any(x > now for x in workers):
        next_arrival = arrivals[i] if i < len(arrivals) else math.inf
        next_done = min((x for x in workers if x > now), default=math.inf)
        now = min(next_arrival, next_done)
        if now == math.inf:
            break
        while i < len(arrivals) and arrivals[i] <= now:
            j = Job(arrivals[i], rng.expovariate(service_rate))
            queue.append(j)
            jobs.append(j)
            i += 1
        # Simple autoscaler: add one worker when queue persists; remove at idle.
        if queue and len(workers) < max_workers:
            free = sum(1 for x in workers if x <= now)
            if free == 0:
                workers.append(now)
                scale_events.append((now, len(workers), "scale_up"))
        for wi, available in enumerate(workers):
            if available <= now and queue:
                j = queue.pop(0)
                workers[wi] = now + j.service
                j.done = workers[wi]
        while len(workers) > base_workers and all(x <= now for x in workers[-1:]):
            workers.pop()
            scale_events.append((now, len(workers), "scale_down"))
    completed = [j for j in jobs if j.done is not None]
    waits = [j.done - j.arrival for j in completed]
    gpu_seconds = sum(j.service for j in completed)  # one toy worker = one GPU
    billed_seconds = sum(max(j.done, horizon) - j.arrival
                         for j in completed) if completed else 0.0
    return jobs, waits, gpu_seconds, billed_seconds, scale_events


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rate", type=float, default=3.0)
    args = ap.parse_args()
    jobs, waits, gpu_s, billed_s, events = simulate(rate=args.rate, seed=args.seed)
    # Illustrative assumptions, not vendor prices or grid factors.
    usd_per_gpu_hour = 2.40
    pue = 1.35
    watts_per_gpu = 320.0
    kgco2_per_kwh = 0.40
    useful = len(jobs)
    cost = gpu_s / 3600 * usd_per_gpu_hour
    kwh = gpu_s * watts_per_gpu / 1000 / 3600 * pue
    carbon = kwh * kgco2_per_kwh
    print(f"jobs={useful} p50_s={percentile(waits,.50):.3f} "
          f"p95_s={percentile(waits,.95):.3f} p99_s={percentile(waits,.99):.3f}")
    print(f"gpu_hours={gpu_s/3600:.3f} illustrative_cost_usd={cost:.2f}")
    print(f"facility_kwh={kwh:.3f} illustrative_kgco2={carbon:.3f}")
    print(f"scale_events={len(events)} first_events={events[:5]}")


if __name__ == "__main__":
    main()
