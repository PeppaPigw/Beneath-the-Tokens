"""第6章最小 PyTorch 实验。CPU 可执行；CUDA 专项在无 GPU 时 SKIP。"""
from __future__ import annotations

import sys

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch 未安装：pip install torch")
    sys.exit(0)


def eager_compile() -> None:
    print("\n[1] eager vs compile")
    torch.manual_seed(0)

    def f(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ w + 0.1)

    x, w = torch.randn(32, 64), torch.randn(64, 16)
    eager = f(x, w)
    if not hasattr(torch, "compile"):
        print("SKIP: torch.compile unavailable")
        return
    compiled = torch.compile(f, backend="eager")
    out = compiled(x, w)
    print("shape=", tuple(out.shape), "allclose=", torch.allclose(eager, out, atol=1e-6))


def autograd_demo() -> None:
    print("\n[2] autograd")
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    loss = (x * x).sum()
    loss.backward()
    print("loss=", loss.item(), "grad=", x.grad)


def dispatcher_demo() -> None:
    print("\n[3] dispatcher trace")
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception as exc:  # pragma: no cover - version dependent
        print("SKIP: TorchDispatchMode unavailable:", type(exc).__name__)
        return

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            print("dispatch:", func)
            return func(*args, **(kwargs or {}))

    with Trace():
        z = torch.ones(2) + torch.ones(2)
    print("result=", z)


def memory_demo() -> None:
    print("\n[4] memory/allocator")
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable (CPU: use torch.profiler.profile(profile_memory=True))")
        return

    def report(tag: str) -> None:
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 2**20
        reserved = torch.cuda.memory_reserved() / 2**20
        print(f"{tag}: allocated={alloc:.1f} MiB reserved={reserved:.1f} MiB")

    report("before")
    x = torch.randn(4096, 4096, device="cuda")
    report("x")
    y = x @ x
    report("y")
    del y
    report("del y")
    torch.cuda.empty_cache()
    report("empty_cache")
    del x


def streams_demo() -> None:
    print("\n[5] CUDA streams")
    if not torch.cuda.is_available():
        print("SKIP: CUDA required")
        return

    device = torch.device("cuda")
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn_like(a)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(s1):
        c = a @ b
    # c 在 s1 上产生；s2 使用前显式建立依赖
    s2.wait_stream(s1)
    with torch.cuda.stream(s2):
        out = c.relu()
    s2.synchronize()
    print("shape=", tuple(out.shape), "mean finite=", bool(torch.isfinite(out.mean())))


def graph_break_demo() -> None:
    print("\n[6] graph breaks")
    if not hasattr(torch, "compile"):
        print("SKIP: torch.compile unavailable")
        return

    import torch._dynamo as dynamo

    def g(x: torch.Tensor) -> torch.Tensor:
        y = torch.sin(x)
        # Tensor.item() 将标量拉回 Python，通常触发 graph break
        if y.sum().item() > 0:
            y = y * 2
        return y

    cg = torch.compile(g, backend="eager")
    print("output=", cg(torch.ones(4)))
    try:
        exp = dynamo.explain(g)(torch.ones(4))
        print("graphs=", exp.graph_count, "breaks=", exp.graph_break_count)
    except Exception as exc:  # explain API 版本差异
        print("explain unavailable:", type(exc).__name__)


def checkpoint_demo() -> None:
    print("\n[7] activation checkpointing")
    from torch.utils.checkpoint import checkpoint

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(8, 8)
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.sin(self.lin(x)).relu()

    block = Block()
    x = torch.randn(4, 8, requires_grad=True)
    block(x).sum().backward()
    print("normal calls=", block.calls)
    block.calls = 0
    x.grad = None
    # use_reentrant=False 是新代码推荐实现；旧 torch 可移除此参数
    checkpoint(block, x, use_reentrant=False).sum().backward()
    print("checkpoint calls=", block.calls)


def main() -> None:
    print("torch=", torch.__version__, "cuda=", torch.cuda.is_available())
    eager_compile()
    autograd_demo()
    dispatcher_demo()
    memory_demo()
    streams_demo()
    graph_break_demo()
    checkpoint_demo()


if __name__ == "__main__":
    main()
