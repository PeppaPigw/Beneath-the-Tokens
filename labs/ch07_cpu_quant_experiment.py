import math
import time
import numpy as np

def symmetric_quantize(x, bits=8, clip_percentile=100.0):
    x = np.asarray(x, dtype=np.float32)
    qmax = (1 << (bits - 1)) - 1
    if clip_percentile < 100:
        amax = np.percentile(np.abs(x), clip_percentile)
    else:
        amax = np.max(np.abs(x))
    amax = max(float(amax), 1e-12)
    scale = amax / qmax
    q = np.clip(np.rint(x / scale), -qmax - 1, qmax).astype(np.int8)
    return q, np.float32(scale)

def affine_quantize(x, bits=8, clip_percentile=None):
    x = np.asarray(x, dtype=np.float32)
    qmin, qmax = 0, (1 << bits) - 1
    if clip_percentile is None:
        lo, hi = float(x.min()), float(x.max())
    else:
        p = float(clip_percentile)
        lo, hi = np.percentile(x, [100 - p, p])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8), np.float32(1.0), 0
    scale = (hi - lo) / (qmax - qmin)
    zp = int(np.rint(qmin - lo / scale))
    zp = max(qmin, min(qmax, zp))
    q = np.clip(np.rint(x / scale + zp), qmin, qmax).astype(np.uint8)
    return q, np.float32(scale), zp

def dequant_sym(q, scale):
    return q.astype(np.float32) * np.float32(scale)

def dequant_affine(q, scale, zp):
    return (q.astype(np.float32) - np.float32(zp)) * np.float32(scale)

def report(name, ref, approx, q=None):
    err = approx - ref
    abs_err = np.abs(err)
    denom = np.maximum(np.abs(ref), 1e-6)
    rel = abs_err / denom
    sat = 0.0 if q is None else float(np.mean((q == q.min()) | (q == q.max())))
    print(f"{name:26s} MAE={abs_err.mean():.6g} RMSE={np.sqrt(np.mean(err*err)):.6g} p99={np.percentile(abs_err,99):.6g} rel_p99={np.percentile(rel,99):.6g} sat={sat:.4%}")

def matmul_error(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((256, 384), dtype=np.float32)
    b = rng.standard_normal((384, 192), dtype=np.float32)
    a[rng.random(a.shape) < 2e-4] *= 25
    b[rng.random(b.shape) < 2e-4] *= 25
    ref = a @ b
    print("-- activation/weight reconstruction --")
    for p in [100.0, 99.99, 99.9, 99.0]:
        qa, sa = symmetric_quantize(a, 8, p)
        qb, sb = symmetric_quantize(b, 8, p)
        ah, bh = dequant_sym(qa, sa), dequant_sym(qb, sb)
        report(f"INT8 symmetric p={p}", a, ah, qa)
        out = ah @ bh
        report(f"matmul p={p}", ref, out)
    pos = np.abs(a)
    q, s, z = affine_quantize(pos, 8, 99.9)
    recon = dequant_affine(q, s, z)
    print("-- affine activation --")
    print("scale=", float(s), "zero_point=", z)
    report("UINT8 affine p=99.9", pos, recon, q)

def timing(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2048, 2048), dtype=np.float32)
    q, s = symmetric_quantize(x)
    for _ in range(2):
        _ = x @ x.T
    t0 = time.perf_counter(); _ = x @ x.T; t1 = time.perf_counter()
    t2 = time.perf_counter(); _ = dequant_sym(q, s); t3 = time.perf_counter()
    print("fp32 matmul_ms=", (t1 - t0) * 1e3, "int8 dequant_ms=", (t3 - t2) * 1e3, "bytes fp32/int8=", x.nbytes, q.nbytes)
    print("注意：NumPy 的这段代码没有调用专用 INT8 GEMM，不能把它当作硬件加速结论")

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    matmul_error(); timing()
