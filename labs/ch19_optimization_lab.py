from __future__ import annotations
import math, random, statistics, time


def baseline(xs, a, b):
    u = [a * x for x in xs]
    v = [x + b for x in u]
    return [max(0.0, x) for x in v]


def fused(xs, a, b):
    return [max(0.0, a * x + b) for x in xs]


def symmetric_int8(xs):
    peak = max(abs(x) for x in xs) or 1.0
    scale = peak / 127.0
    q = [max(-127, min(127, round(x / scale))) for x in xs]
    deq = [scale * z for z in q]
    return q, deq, scale


class PagePool:
    def __init__(self, page_tokens, pages):
        self.page_tokens = page_tokens
        self.free = set(range(pages))
        self.owner = {}

    def grow(self, seq, tokens):
        need = math.ceil(tokens / self.page_tokens)
        have = len(self.owner.get(seq, []))
        if need <= have:
            return True
        extra = need - have
        if len(self.free) < extra:
            return False
        ids = sorted(self.free)[:extra]
        self.free.difference_update(ids)
        self.owner.setdefault(seq, []).extend(ids)
        return True

    def release(self, seq):
        for p in self.owner.pop(seq, []):
            self.free.add(p)

    def stats(self):
        used = sum(len(v) for v in self.owner.values())
        waste = sum(self.page_tokens - (tokens % self.page_tokens or self.page_tokens)
                    for tokens in self._tokens.values()) if hasattr(self, "_tokens") else 0
        return used, len(self.free), waste


def run(seed=7, n=200_000):
    rng = random.Random(seed)
    xs = [rng.uniform(-2, 2) for _ in range(n)]
    for fn in (baseline, fused):
        fn(xs, 1.7, -0.2)
    rows = []
    for fn in (baseline, fused):
        times = []
        for _ in range(5):
            t0 = time.perf_counter(); ys = fn(xs, 1.7, -0.2)
            times.append(time.perf_counter() - t0)
        rows.append((fn.__name__, statistics.median(times), max(ys)))
    q, deq, scale = symmetric_int8(xs)
    err = max(abs(x-y) for x, y in zip(xs, deq))
    pool = PagePool(page_tokens=16, pages=128)
    admissions = []
    for seq, tokens in enumerate([7, 19, 33, 61, 9, 42]):
        admissions.append((seq, tokens, pool.grow(seq, tokens)))
    return rows, scale, err, admissions


if __name__ == "__main__":
    for row in run()[0]:
        print("kernel", row[0], "median_s", f"{row[1]:.6f}", "max", f"{row[2]:.4f}")
    rows, scale, err, admissions = run()
    print("int8_scale", f"{scale:.6f}", "max_abs_error", f"{err:.6f}")
    print("page_admissions", admissions)
