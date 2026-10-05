---
id: ch06-pytorch-execution
title: PyTorch 执行内部：从 Python 调用到 GPU kernel
slug: /chapters/06-pytorch-execution
description: 解释 eager、autograd、dispatcher、allocator、streams、torch.compile 与 checkpointing 的执行链，并用最小实验定位性能和内存故障
sidebar_position: 6
level: systems
prerequisites:
  - ch01-ai-infrastructure
  - ch02-linux
  - ch04-performance-math
learning_objectives:
  - 能从一次 Python 运算追踪到 dispatcher、设备实现和 kernel
  - 能解释 eager、torch.compile、TorchDynamo、AOTAutograd、Inductor 的边界
  - 能用 autograd 图、版本计数器和 saved tensors 诊断反向传播错误
  - 能区分 CUDA stream、event、同步点和异步错误
  - 能解释 caching allocator、reserved/allocated、碎片与 OOM 的关系
  - 能选择 activation checkpointing、梯度累积和混合精度的组合
  - 能运行最小实验，记录图断裂、编译缓存、内存峰值和 kernel 时间
  - 能根据版本边界和硬件后端判断结论是否可迁移
estimated_hours: 20
hardware: CPU-only baseline; CUDA GPU optional
risk_level: L2
last_verified: 2026-10-05
---

# 第6章　PyTorch 执行内部：从 Python 调用到 GPU kernel

> PyTorch 代码看起来像普通 Python，但一行 `y = x @ w + b` 会经过张量元数据检查、算子分派、设备队列、内存分配、自动求导记录，甚至可能被编译成新的图和 kernel。本章建立一条可以反复使用的追踪路径：先问“调用在什么层”，再问“谁拥有这块内存”“哪个 stream 在执行”“哪里发生了同步”，最后用实验验证猜测。

## 6.1 先建立执行心智模型

### 6.1.1 一行代码的六层含义

设代码如下：

```python
loss = torch.nn.functional.cross_entropy(logits, target)
loss.backward()
optimizer.step()
```

它至少同时发生六件事：

1. **Python API 层**：解析参数，处理默认值，调用绑定到 C++ 的入口。
2. **Tensor 抽象层**：读取 dtype、device、shape、stride、requires_grad 和 layout，决定输入是否为连续内存或 view。
3. **Dispatcher 层**：根据算子 schema 和 dispatch key 选择 CPU、CUDA、Autograd、Functionalize、Batched、Meta 等实现或包装器。
4. **执行后端层**：调用 ATen/C++ kernel、cuBLAS、cuDNN、Triton、NCCL 或其他设备库，把工作排到 stream。
5. **状态层**：更新 storage 的引用、版本计数器、CUDA caching allocator 的块表，以及 autograd 图的节点和 saved tensors。
6. **同步与可见性层**：Python 何时返回、设备工作何时完成、异常在哪个 API 才会暴露，取决于后端的异步模型。

[事实] “Python 函数返回”不等于“GPU 已经完成”。在 CUDA 后端，许多算子只把 kernel 入队到当前 stream；真正读取结果或显式同步时，才会等待设备。

[机制] 每层只负责一部分契约。Dispatcher 不负责内存生命周期，allocator 不决定算子语义，autograd 不知道 kernel 的线程布局。性能和正确性问题通常发生在层与层之间的边界。

[设计判断] 调试时不要从“这行很慢”直接跳到“GPU 不够快”。先记录调用栈、设备、stream、分配量、是否建立反向图、是否触发编译或同步，再定位瓶颈。

### 6.1.2 eager、compile 和手写 kernel 的对照

**Eager 模式**逐个执行 Python 发起的算子。它的优点是语义直接、调试容易、动态控制流自然；代价是 Python 调度开销、kernel 碎片和中间张量读写较多。

**`torch.compile`**不改变用户可见的 Tensor API，而是在运行时捕获一段 Python，经过图变换和后端编译，生成可复用的执行计划。第一次调用通常包含追踪、守卫检查、代码生成和编译成本；后续命中缓存时才体现稳定性能。

**手写 kernel**（CUDA、Triton 或 C++ 扩展）把线程映射、内存访问、同步和布局交给开发者。它可能解决通用算子无法覆盖的瓶颈，也更容易引入越界、竞态、非确定性和版本耦合。

三者不是“新旧替代”关系。稳定的热路径可以 compile，稀疏且动态的控制流可以保留 eager，经过 profiling 证明有收益的热点再考虑自定义 kernel。

### 6.1.3 观察执行需要四类证据

为了让结论可复现，最小记录集包括：

- **语义证据**：输入 shape、dtype、requires_grad、随机种子、模型模式（train/eval）。
- **时间证据**：CPU wall time、CUDA event 时间、kernel 时间线、编译时间和同步点。
- **内存证据**：`allocated`、`reserved`、峰值、活跃块、临时张量存活区间。
- **版本证据**：Python、PyTorch、CUDA runtime、驱动、GPU 型号、后端开关和环境变量。

只报一个“吞吐提升 1.4 倍”是不够的。必须说明是否把首次编译摊入平均值，是否使用固定 shape，是否在每次迭代前同步，是否把数据拷贝算入端到端时间。

## 6.2 从 Python 调用到 kernel：一条可追踪路径

### 6.2.1 Python 绑定与 ATen 算子

PyTorch 的许多张量方法最终落到 ATen 算子。算子拥有 schema，例如输入类型、输出类型、默认参数和别名关系。Python 端的 `torch.add`、`Tensor.add_` 和复合函数可能共享底层算子，也可能经过多个分解步骤。

一个算子调用大致经历：

```text
Python bytecode
  -> Python/C++ binding
  -> aten::op schema
  -> Dispatcher 选择 key
  -> 后端实现（CPU/CUDA/库调用）
  -> stream 上的 kernel 或 host 代码
  -> TensorImpl/Storage 更新
  -> Autograd wrapper 记录历史（若启用）
```

这里的箭头不是每次都严格一层一层出现。例如纯元数据操作可以不发 kernel，复合算子可能递归调用多个 ATen 算子，`torch.compile` 还可能把多个节点融合成一个生成代码。

### 6.2.2 Tensor、TensorImpl、Storage 与 view

用户看到的 Tensor 是一个带有元数据的句柄。元数据包括 sizes、strides、dtype、device、layout、requires_grad 和 dispatch 相关信息。实际数据通常位于 Storage 指向的分配块中。

`view`、切片和转置常常只新建元数据，不复制数据。例如：

```python
x = torch.arange(12, device="cpu").reshape(3, 4)
y = x.t()                 # 共享 storage，stride 改变
z = y.contiguous()        # 必要时新分配并复制
```

[事实] 共享 storage 意味着原张量和 view 的写入可能互相影响；in-place 操作还会更新版本计数器。Autograd 为了保证反向结果正确，会检查保存张量的版本是否被非法修改。

[故障模式] 把 `reshape` 当成总是零拷贝，或把 `contiguous` 当成便宜的元数据操作，都会导致错误的内存和性能估计。遇到 stride 异常时，应显式打印 `size、stride、storage_offset、is_contiguous`。

### 6.2.3 Dispatcher 与 DispatchKey

Dispatcher 是算子调用的交通枢纽。它接收 schema、参数和 dispatch key 集合，然后选择最高优先级的实现。常见 key 包括 CPU、CUDA、Autograd、CompositeImplicitAutograd、Meta、Functionalize、Batched 和 Python。

可以把它想成两步：

1. **包装器步骤**：Autograd、Functionalize、AMP 等 key 可能先截获调用，记录图、规范化 in-place 或做 dtype 策略。
2. **设备步骤**：CPU/CUDA/XPU/MPS 等后端执行真正的算子实现，可能进一步调用 BLAS、DNN 库或自定义 kernel。

Dispatch key 并不是“设备类型的字符串”。同一个 CUDA Tensor 在训练时可能同时携带 CUDA 与 Autograd 语义；禁用梯度后，Autograd 包装层就不再建立新节点，但 CUDA kernel 仍然执行。

排查算子落点时，可以先做低风险检查：

```python
print(torch.__version__)
print(torch.backends.cuda.matmul.allow_tf32)
print(x.device, x.dtype, x.layout, x.stride())
print(torch.is_grad_enabled())
```

需要更深入时，使用 profiler 的 operator view 或 `torch._C._dispatch_dump_table("aten::add")`（内部 API，版本可能变化）查看注册表。不要把内部 dump 当作长期稳定接口。

### 6.2.4 CPU 与 CUDA 的时间边界

CPU 算子通常在调用返回前完成，虽然线程池可能在内部并行。CUDA 算子多数是异步入队：

```python
start = time.perf_counter()
y = x @ w
elapsed = time.perf_counter() - start  # 只测到入队，不是 kernel 完成时间
```

正确的微基准至少要预热并用 CUDA event：

```python
for _ in range(10):
    y = x @ w
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    y = x @ w
end.record()
end.synchronize()
print(start.elapsed_time(end) / 100, "ms")
```

事件记录在 stream 上，因此比围绕 Python 计时器调用 `synchronize()` 更能避免把 CPU 端调度噪声混进 kernel 时间。端到端服务仍要单独测量，因为数据拷贝、排队和同步是用户实际等待的一部分。

## 6.3 Autograd：反向图、保存值与版本计数器

### 6.3.1 正向传播时发生什么

当 `requires_grad=True` 且梯度模式开启时，每个可微算子会创建一个 autograd Node。Node 保存反向所需的信息，例如输入 shape、某些中间值、缩放因子和对父节点的引用。输出 Tensor 通过 `grad_fn` 指向这个 Node，形成从结果回溯到叶子张量的有向无环图。

```python
x = torch.randn(4, requires_grad=True)
y = (x * x).sum()
print(y.grad_fn)
y.backward()
print(x.grad)
```

图是动态构建的。每次 forward 都可能得到不同结构，因此 Python 的 `if`、循环和早退天然可表达；代价是每次需要重新创建 Node 和保存信息。

### 6.3.2 backward 的调度与梯度累积

调用 `backward()` 后，autograd engine 从根节点开始反向调度。一个节点只有在所有依赖的梯度到达后才可执行。引擎可以使用多个 CPU worker 处理独立分支，但 CUDA 计算仍受 stream 和设备语义约束。

叶子参数的 `.grad` 默认是累积的：

```python
for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
```

若不清零，第二个 batch 的梯度会叠加。`set_to_none=True` 通常可以减少一次填零写入，并让未参与本次计算的参数保持 `None`，但某些旧代码假设 `.grad` 总是 Tensor，需要检查兼容性。

### 6.3.3 saved tensors 与显存占用

反向并不一定保存所有正向输出。每个 backward 实现只保存它需要的值，但这些值可能占据大量显存，例如注意力中的激活、卷积输入或归一化统计量。查看显存时，不能只看模型参数；激活和临时 workspace 常常是训练峰值的主要来源。

自定义 autograd Function 时，`ctx.save_for_backward` 会把 Tensor 纳入保存列表。保存一个大 Tensor 的 view 也可能因为共享 storage 而保留整块底层分配，导致“看起来只保存一小段，实际释放不了大块”的现象。

### 6.3.4 版本计数器与 in-place 错误

每个可变 Tensor 都有版本计数器。Autograd 保存一个 Tensor 后，如果该 Tensor 或其共享 storage 被 in-place 修改，版本计数会改变。反向阶段发现版本不一致时，会报类似“one of the variables needed for gradient computation has been modified in-place”的错误。

典型问题：

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
# x.add_(1)  # 对需要梯度的叶子原地修改通常直接报错
loss = y.sum()
loss.backward()
```

修复不是盲目地把所有操作改成复制。先判断是否真的需要原地更新；如果只为节省显存，应先测量激活峰值，再评估 checkpoint、低精度或算子融合等方案。原地操作可能阻碍编译器的 alias 分析，也可能让 graph capture 产生额外限制。

### 6.3.5 `no_grad`、`inference_mode` 和 detach

- `torch.no_grad()` 暂停梯度记录，适合验证和推理上下文，但仍保留较完整的 Tensor 语义。
- `torch.inference_mode()` 进一步使用 inference tensor 优化元数据和版本开销，适合确定不需要 autograd 的纯推理；把 inference tensor 带回训练图可能失败。
- `detach()` 返回共享 storage 的无梯度视图，不复制数据。后续 in-place 写入仍可能影响原 Tensor。

```python
with torch.inference_mode():
    pred = model(x)
```

不要把 `detach()` 当成内存释放指令。只要某个引用还在，底层 storage 就不会释放；要释放中间结果，需要缩短引用生命周期或删除容器中的引用。

## 6.4 `torch.compile` 执行链：Dynamo、AOTAutograd、Inductor

### 6.4.1 编译不是一次函数调用

`torch.compile(fn)` 返回一个包装后的 callable。第一次执行时，TorchDynamo 观察 Python 字节码和 Tensor 操作，构建 FX 图并安装 guards。图交给后端，例如 TorchInductor，后端再生成 Triton/C++/设备库调用并编译。训练场景可能先由 AOTAutograd 分离正向和反向图，再分别优化。

典型路径：

```text
Python frame
  -> TorchDynamo bytecode analysis
  -> FX graph + guards
  -> AOTAutograd（训练时）
  -> decomposition / pattern rewrite
  -> TorchInductor scheduling + fusion
  -> Triton/C++/库代码生成
  -> 编译缓存
  -> 后续调用检查 guards 并复用
```

[边界] 这是概念路径，不保证每个版本、每个后端都包含相同步骤。后端可以被替换，某些算子可能直接回退 eager，编译缓存也可能因环境变化而失效。

### 6.4.2 Guards 与形状专门化

编译图需要假设。常见 guard 包括 Tensor 的 dtype、device、rank、某些尺寸、Python 全局变量、模块属性和对象身份。如果下一次调用违反 guard，Dynamo 可能重新编译、选择另一个缓存版本或回退 eager。

固定 batch 和固定序列长度时，专门化通常更激进，生成代码更容易融合；动态 shape 可以减少编译版本数量，但可能增加运行时判断，或者限制某些融合。选择 `dynamic=True`、`dynamic=False` 或默认策略前，应先测量形状分布，而不是凭感觉打开“动态”。

### 6.4.3 图断裂（graph break）

图断裂表示编译器无法把某段 Python 转换进当前图，于是先运行已捕获部分，再回到 Python eager，之后尝试继续捕获。常见原因：

- 对 Tensor 调用依赖具体值的 `.item()`，需要把设备结果同步到 CPU。
- 使用不受支持的第三方 Python 库或数据结构。
- 依赖不可追踪的副作用，例如修改全局列表、打印对象、随机访问文件。
- 数据依赖控制流没有使用可追踪的写法。

图断裂不一定是错误，但会带来 Python 调度、同步和中间张量物化成本。诊断时设置日志而不是直接猜：

```bash
TORCH_LOGS="graph_breaks,recompiles" python train.py
```

也可在小函数上使用 `torch._dynamo.explain` 查看断裂原因。日志接口和字段属于版本敏感区域，脚本应锁定 PyTorch 版本并在升级时重跑。

### 6.4.4 编译缓存、首次延迟与公平基准

编译实验至少报告三种时间：

1. 首次调用时间，包含捕获和编译。
2. 缓存命中后的稳态时间。
3. 形状或控制流变化触发重新编译时的时间。

若服务启动后只处理一个请求，compile 可能得不偿失；若服务长时间处理稳定形状，编译成本可以摊薄。基准测试应预热到缓存命中，并单独报告编译时间，否则会把两种不同产品体验混成一个平均值。

## 6.5 动态控制流、可变状态与编译边界

### 6.5.1 Tensor 控制流和 Python 控制流

```python
def f(x):
    if x.sum() > 0:
        return x * 2
    return x - 2
```

这里 `x.sum() > 0` 需要读取 Tensor 值来决定 Python 分支，通常会形成图断裂或 guard。改写为张量级选择：

```python
def f_compilable(x):
    return torch.where(x.sum() > 0, x * 2, x - 2)
```

不过 `torch.where` 可能先计算两边表达式，分支很重时要权衡。可编译性不是唯一目标，需比较总成本和语义。

### 6.5.2 可变模块状态

BatchNorm 的 running statistics、缓存字典、随机数生成器和自定义计数器都可能成为 guard 或断裂来源。使用 compile 前，明确哪些状态是模型语义，哪些只是调试副作用。把日志、统计和指标收集移到图外，通常比强行塞进图更稳。

### 6.5.3 失败回退策略

工程上可以采用三层策略：

- 先让整个模型 eager 正确运行并有基准。
- 对稳定子模块做 `torch.compile`，把动态或第三方部分保留 eager。
- 对热点进行算子级分解或自定义 kernel，建立针对性测试。

不要一开始就把所有断裂标记为错误。先计算断裂每秒发生次数、是否触发同步、对 p50/p99 的影响，再决定重构是否值得。

## 6.6 内存系统：Storage、caching allocator 与碎片

### 6.6.1 allocated、reserved 和实际占用

CUDA caching allocator 从驱动申请较大的块，再在进程内切分和复用。常见指标：

- `memory_allocated()`：当前由活跃 Tensor 使用的字节数。
- `memory_reserved()`：allocator 向驱动保留、可复用的总字节数。
- `max_memory_allocated()`：记录窗口内活跃 Tensor 峰值。
- `max_memory_reserved()`：记录窗口内保留块峰值。

因此 `reserved > allocated` 通常不是泄漏，而是缓存。真正的 OOM 可能发生在 reserved 尚有空间但找不到合适大小的连续块时，也可能是驱动层、其他进程或库 workspace 抢占了容量。

```python
torch.cuda.reset_peak_memory_stats()
# 运行一个 step
print("alloc", torch.cuda.memory_allocated()/2**20, "MiB")
print("reserved", torch.cuda.memory_reserved()/2**20, "MiB")
print("peak", torch.cuda.max_memory_allocated()/2**20, "MiB")
```

### 6.6.2 分配生命周期和碎片

每次临时 Tensor 都有生命周期。若大小和释放顺序相近，缓存可以高效复用；若出现很多不同尺寸、跨 stream 的长短块，可能留下无法满足大分配的碎片。动态 batch、变长序列和频繁创建临时列表是常见诱因。

`torch.cuda.empty_cache()` 只把未使用的缓存块归还给驱动，不能释放仍被 Tensor 引用的内存，也不是每一步都应调用的“清理按钮”。频繁调用会丢失缓存复用优势，增加驱动分配开销。

### 6.6.3 跨 stream 的释放延迟

一个 Tensor 在非当前 stream 上使用后，allocator 需要知道该 stream 何时完成，才能安全复用底层块。若代码在多个 stream 间传递 Tensor，没有正确记录 stream 使用关系，可能出现释放延迟、峰值升高，极端情况下会有数据竞态。使用官方 stream API 和事件，避免手工猜测生命周期。

### 6.6.4 OOM 分类

看到 `CUDA out of memory` 时先分类：

1. **活跃 Tensor 真超容量**：参数、激活、梯度或 optimizer state 本身超过显存。
2. **临时峰值**：单个算子 workspace 或融合前后的中间张量瞬时过大。
3. **碎片**：总 reserved 足够，但无法满足连续块或特定 stream 的请求。
4. **泄漏式引用**：把带图 Tensor 追加到 Python list，导致每步图都被保留。
5. **外部占用**：其他进程、上下文、通信库或图形桌面占用显存。

修复路径不同。不要看到 OOM 就直接减 batch；先用 memory snapshot、峰值统计和引用审计确认类别。

### 6.6.5 典型“伪泄漏”

```python
history.append(loss)          # loss 带 grad_fn，整张图可能被保留
```

应改为：

```python
history.append(float(loss.detach()))
# 或 history.append(loss.detach().cpu())，视需求而定
```

同样，日志框架若缓存 GPU Tensor、闭包捕获中间变量、异常对象持有局部栈，都可能延长生命周期。使用 `gc.get_referrers`、显存快照和缩小复现来确认，而不是反复 `empty_cache()`。

## 6.7 Streams、events 与异步执行

### 6.7.1 默认 stream 和当前 stream

每个 CUDA 设备有当前 stream 语义。普通算子通常入队到当前 stream；同一 stream 内按顺序执行，不同 stream 可以并发，但只有满足依赖时才安全。PyTorch 的 current stream 是线程和设备相关的抽象，不应把“默认 stream”简单理解成全局唯一队列。

```python
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    y = model(x)
# 在需要读取 y 前建立依赖或同步
s.synchronize()
```

### 6.7.2 事件用于计时和依赖

CUDA event 可以记录 stream 上的时间点，也可以用于跨 stream 依赖：

```python
ready = torch.cuda.Event()
with torch.cuda.stream(producer):
    y = f(x)
    ready.record()
with torch.cuda.stream(consumer):
    consumer.wait_event(ready)
    z = g(y)
```

事件等待不会必然让 CPU 阻塞，但会约束设备执行顺序。把 `torch.cuda.synchronize()` 放在每个小算子后面会破坏并发并放大延迟；调试阶段可以这样做来定位错误，性能基准阶段必须移除或明确报告。

### 6.7.3 异步错误为什么在下一行爆出

设备 kernel 的越界、非法访问或断言，可能在异步执行期间发生，直到下一个同步点才报告。于是堆栈看起来像是无辜的 `tensor.cpu()` 或日志打印。调试时可使用：

```bash
CUDA_LAUNCH_BLOCKING=1 python script.py
```

这会强制更同步的执行，便于定位，但会显著改变性能和时序。只用于复现和缩小问题，不能把它当成生产配置。

### 6.7.4 数据加载和计算重叠

Pinned memory、非阻塞拷贝和独立 stream 可以让 CPU 准备下一批数据时，GPU 计算当前批次。前提是：

- DataLoader 使用合理的 worker 数和 `pin_memory=True`。
- H2D 拷贝使用 `non_blocking=True`，并且源内存确实 pinned。
- 不在每个 batch 后无意调用 `.item()` 或 `.cpu()` 触发同步。
- 设备显存和 host pinned memory 都有上限，队列过长会把压力转移而非消除。

## 6.8 Activation checkpointing、梯度累积与训练内存

### 6.8.1 Checkpointing 的交换关系

Activation checkpointing 在正向阶段只保存少量输入，反向阶段重新执行被包裹的 forward，以换取更低的激活显存。它减少的是保存激活，不减少参数、梯度和 optimizer state；重算会增加计算时间并可能改变随机数、通信和副作用。

```python
from torch.utils.checkpoint import checkpoint

def block(x):
    return torch.relu(x @ w + b)

y = checkpoint(block, x, use_reentrant=False)
```

优先把计算密集、激活占用大且副作用少的 block 作为 checkpoint 边界。过细的切分会引入大量重算和调度开销，过粗则无法释放足够激活。

### 6.8.2 RNG、dropout 与确定性

Checkpoint 为了让重算结果与原 forward 一致，通常需要保存和恢复随机数状态。跨 CPU、CUDA 设备或自定义 RNG 时要检查语义；如果关闭 RNG 保存，可能得到不同 dropout mask，训练统计会变化。开启确定性算法也可能牺牲性能，必须在实验记录中写明。

### 6.8.3 梯度累积不是免费扩容

梯度累积把多个 micro-batch 的梯度合并后再更新参数，可以在显存不足时保持较大的有效 batch。每个 micro-batch 在自己的 backward 前需要保存激活，backward 完成后通常即可释放；若把多个 loss 图留到最后统一 backward，或错误地保留带 grad_fn 的 loss，累积步数越多越容易 OOM。

推荐模式：

```python
optimizer.zero_grad(set_to_none=True)
for micro in range(accum_steps):
    with autocast_context:
        loss = model(x[micro]).loss / accum_steps
    scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

确认 `loss / accum_steps`，否则累积后梯度会按步数放大。混合精度、GradScaler 和 checkpoint 的组合还要验证溢出恢复和随机数一致性。

### 6.8.4 参数、梯度、optimizer state 的账本

训练显存粗略由四部分组成：参数、梯度、优化器状态、激活/临时。以 Adam 为例，每个参数通常还要维护一阶和二阶矩，混合精度可能同时保留 FP16 参数和 FP32 master copy。只按“参数量 × dtype 字节”估算会严重低估峰值。

建立账本时列出每一项的 dtype、设备、生命周期和是否分片。ZeRO/FSDP 等分片方法减少每卡持有的参数或状态，但增加通信和重组阶段；结论不能直接外推到单卡训练。

## 6.9 最小可运行实验

本节实验都可以在 CPU 上运行，检测到 CUDA 时自动启用 GPU。每个实验都强调一个可观察事实，而不是追求绝对性能数字。

### 实验 A：eager 与 compile 的首次/稳态差异

```python
import time, torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Sequential(
    torch.nn.Linear(1024, 2048), torch.nn.GELU(),
    torch.nn.Linear(2048, 1024)
).to(device)
x = torch.randn(64, 1024, device=device)

def run(m, n=20):
    for _ in range(3):
        m(x)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): m(x)
    if device == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

print("eager", run(model))
if hasattr(torch, "compile"):
    compiled = torch.compile(model)
    t0 = time.perf_counter()
    compiled(x)
    if device == "cuda": torch.cuda.synchronize()
    print("compile first", time.perf_counter() - t0)
    print("compile steady", run(compiled))
```

记录首次编译时间和稳态时间，改变 batch 或最后一维后再次运行，观察是否出现重新编译。CPU 上编译后端和版本差异较大，不能拿结果直接和 CUDA 结论互换。

### 实验 B：autograd 图与保存值

```python
import torch
x = torch.randn(8, 8, requires_grad=True)
h = torch.sin(x) * x
loss = h.square().mean()
print("grad_fn", type(loss.grad_fn).__name__)
print("leaf", x.is_leaf, "requires_grad", x.requires_grad)
loss.backward()
print("grad norm", x.grad.norm().item())
```

扩展实验：在 `h` 后加入 `h.detach()`，比较 `grad_fn` 和 `x.grad`；在循环中把 `loss` 加到列表，观察内存是否随步数上升，再改成 `loss.detach().item()`。

### 实验 C：view、stride 和隐式 contiguous

```python
x = torch.randn(1024, 1024)
y = x.t()
print(y.stride(), y.is_contiguous())
z = y @ x
print(z.shape)
yc = y.contiguous()
print(yc.stride(), yc.is_contiguous())
```

使用 profiler 比较直接转置输入和显式 contiguous 的时间。某些矩阵乘法库可以处理非连续布局，某些路径会先复制；不要仅凭 `is_contiguous=False` 断言一定慢。

### 实验 D：CUDA 异步计时

```python
import time, torch
if torch.cuda.is_available():
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    for _ in range(10): torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.mm(a, b)
    cpu_seen = time.perf_counter() - t0
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); torch.mm(a, b); e.record(); e.synchronize()
    print("enqueue", cpu_seen, "event_ms", s.elapsed_time(e))
```

如果 `enqueue` 远小于 `event_ms`，说明 Python 计时器只看到了入队。把 `torch.cuda.synchronize()` 加在前后可得到端到端单次时间，但会阻塞 CPU。

### 实验 E：allocator 峰值和伪泄漏

```python
import torch
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    xs = []
    for _ in range(20):
        x = torch.randn(2048, 2048, device="cuda", requires_grad=True)
        loss = (x * x).mean()
        xs.append(loss)  # 故意保留计算图
    print(torch.cuda.max_memory_allocated()/2**20)
    xs.clear()
    torch.cuda.synchronize()
    print("after clear", torch.cuda.memory_allocated()/2**20)
```

将 `xs.append(loss)` 改为 `xs.append(float(loss.detach()))`，比较峰值。实验说明 allocator 缓存和 Python 引用是两个不同问题。

### 实验 F：graph break 观察

```python
import torch

def f(x):
    s = x.sum().item()       # 可能触发设备到主机同步
    if s > 0:
        return x * 2
    return x - 2

if hasattr(torch, "compile"):
    g = torch.compile(f)
    for _ in range(3): print(g(torch.randn(16, device="cuda" if torch.cuda.is_available() else "cpu")))
```

用 `TORCH_LOGS="graph_breaks,recompiles"` 运行，记录断裂位置。再改成 `torch.where` 版本，比较日志和时间。不同 PyTorch 版本的日志文本不同，实验报告应保存版本号。

### 实验 G：checkpoint 的显存与时间交换

```python
import torch
from torch.utils.checkpoint import checkpoint

class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__(); self.l = torch.nn.Linear(d, d)
    def forward(self, x): return torch.relu(self.l(x))

if not torch.cuda.is_available():
    print("SKIP: CUDA required")
else:
    blocks = torch.nn.Sequential(*[Block(4096) for _ in range(4)]).cuda()
    x = torch.randn(8, 4096, device="cuda", requires_grad=True)
    def plain(x): return blocks(x).sum()
    def ckpt(x):
        y = x
        for b in blocks: y = checkpoint(b, y, use_reentrant=False)
        return y.sum()
    for fn in (plain, ckpt):
        torch.cuda.reset_peak_memory_stats(); blocks.zero_grad(set_to_none=True)
        fn(x).backward(); torch.cuda.synchronize()
        print(fn.__name__, torch.cuda.max_memory_allocated()/2**20)
```

记录 forward+backward 总时间和峰值显存，不要只报其中一个。显存不足时可缩小维度或层数，保持相同趋势即可。

## 6.10 从一次真实调用追踪到 kernel

下面用 `linear + gelu` 作为追踪练习。目标不是记住某个函数名，而是建立证据链。

### 步骤 1：固定输入契约

记录 `shape=(batch, hidden)`、dtype、device、stride、是否 requires grad。输入如果来自 DataLoader，还要记录 pinned 状态、H2D 拷贝和 batch 拼接是否产生额外副本。

### 步骤 2：区分复合函数和原生算子

`nn.Linear` 可能调用矩阵乘加；`gelu` 可能选择精确或近似实现。使用 profiler 的 `record_shapes=True` 可以看到实际 aten 操作。不要根据模块名称假设只有一个 kernel。

### 步骤 3：观察 dispatcher 和编译边界

在 eager 下，多个 aten 操作各自经过 dispatcher。在 compile 下，Dynamo 可能把它们收集成 FX 图，Inductor 再决定融合成一个或多个 kernel。若遇到 unsupported op，时间线会出现 compiled segment、eager segment、再次 compiled segment 的交替。

### 步骤 4：观察 stream 和 kernel

在 CUDA profiler 中，查看 kernel 所在 stream、开始时间、持续时间以及前后的 memcpy。若 CPU 线程出现长空洞，可能在等待编译、同步或内存分配；若 GPU stream 处于空闲，可能是 Python 调度或数据准备不足。

### 步骤 5：将结果映射回代码

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available()
               else [torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(5):
        y = model(x)
        loss = y.square().mean()
        loss.backward()
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

`with_stack=True` 会增加开销，且源代码路径可能包含机器信息。分享 trace 前应清理路径和输入内容。Profiler 看到的是被采样和插桩的执行，不等于所有驱动层开销；必要时用 Nsight Systems/Compute 进行更底层验证。

## 6.11 性能故障案例：平均吞吐变快但 p99 变差

### 6.11.1 症状

团队把模型包在 `torch.compile` 中，稳态吞吐从 900 提升到 1200 samples/s，平均延迟下降；上线后 p99 从 120 ms 升到 2 s，偶发请求卡住十几秒。

### 6.11.2 排查路径

先按 shape 和请求序列分桶，而不是只看总平均。发现线上有很多罕见序列长度，触发新的 guards 和重新编译。编译线程与请求线程共享 CPU，重新编译时 GPU 反而空闲；编译完成后缓存版本太多，部分 frame 触发重新编译上限并回退 eager。

修复方案是：

1. 对长度做有限分桶，减少 guard 组合。
2. 预热高频 shape，编译阶段从服务流量中隔离。
3. 监控 compile time、recompile count、graph break count，而不只看 kernel time。
4. 对长尾 shape 设置 eager fallback 或专用队列，避免阻塞常规请求。

[教训] compile 的收益是“缓存命中后的执行计划”，不是无条件的全局加速。p99 需要把编译、排队和同步纳入边界。

## 6.12 性能故障案例：GPU 利用率低但 kernel 不慢

### 6.12.1 症状

`nvidia-smi` 显示 GPU 利用率只有 30%，单个矩阵乘 kernel 已接近理论带宽，服务却达不到吞吐目标。

### 6.12.2 证据

时间线显示每个 batch 之间有 2 ms CPU 空洞，空洞来自 Python 数据预处理和频繁的 `.item()`。`.item()` 需要把标量从 GPU 复制到 CPU，形成同步；每个请求调用一次，多个小 batch 的同步成本累计成主要延迟。

改进：将指标计算留在 GPU，批量异步拷贝结果；把预处理放入 DataLoader worker；合并小算子或使用 compile；只在需要日志时低频同步。修复后 GPU 利用率上升，但仍需用端到端时间确认用户真的受益。

## 6.13 性能故障案例：误把 contiguous 当作万能优化

### 症状

开发者对每个输入调用 `.contiguous()`，某个模型速度略升，另一个模型速度下降且显存峰值增加。

### 原因

连续化会复制数据。若后端库本身支持给定 stride，复制只是额外开销；若后端必须连续化，显式提前复制可以让时间更可控。两种情况下都不能仅凭经验判断。

### 诊断

对比三条路径：原始 view、显式 `contiguous`、重新设计上游布局。记录复制字节数、kernel 时间、峰值显存和总 wall time。对于转置后立即矩阵乘的场景，选择取决于矩阵尺寸和库版本。把“是否连续”加入基准参数，而不是写死全局策略。

## 6.14 内存故障案例：每步多一点，几百步后 OOM

### 现象

训练开始时每步约 8 GiB，运行 500 步后显存涨到 20 GiB。`empty_cache()` 无法解决。

### 根因推断

最常见是 Python 容器持有带图的 loss 或输出：

```python
metrics.append({"loss": loss, "logits": logits})
```

每个字典都延长整张图和底层 storage 的生命周期。另一个可能是 `retain_graph=True` 被永久打开，或 hook 闭包保存了激活。

### 修复和验证

- 指标转为 Python 标量或 detached CPU Tensor。
- 只在确实需要高阶导时使用 `retain_graph=True`。
- 对 hook 明确 remove，避免重复注册。
- 用 `gc` 和 memory snapshot 检查活跃块对应的 Python 引用。
- 修复后运行固定步数，比较每步峰值而非只看最终值。

如果 `allocated` 稳定而 `reserved` 上升，重点看碎片和分配模式；如果 `allocated` 逐步上升，优先审计引用。

## 6.15 内存故障案例：reserved 很大但分配失败

### 症状

日志显示 `allocated=10 GiB, reserved=15 GiB`，申请 4 GiB 临时张量时 OOM。开发者认为还有 5 GiB 可用。

### 解释

reserved 是 allocator 管理的总块，不代表存在 4 GiB 连续可复用块。活跃张量可能把大块切成碎片，跨 stream 的 pending 使用也会延迟回收。驱动和其他进程占用也可能让真正可用容量更小。

### 处理顺序

1. 导出 memory snapshot，查看大块分配和释放历史。
2. 固定或分桶动态 shape，减少尺寸抖动。
3. 缩短临时张量生命周期，避免同时保留多个大中间结果。
4. 评估 allocator 配置（例如 `PYTORCH_ALLOC_CONF` 的 backend 或 split 策略），每次只改一个变量。
5. 在受控环境比较 `cudaMallocAsync` 等后端，但不要把配置直接复制到所有驱动和容器。

## 6.16 调试与观测工作流

### 6.16.1 先做最小复现

把模型缩减成一个函数、一个 shape 和固定随机种子。保留能复现错误的最短代码，记录 PyTorch/CUDA/驱动版本。最小复现比完整训练日志更容易判断是语义问题、编译问题还是后端问题。

### 6.16.2 再分层开关

按以下顺序切换，避免一次改太多：

1. eager + CPU，确认数学结果和 autograd。
2. eager + CUDA，加入显式同步，确认设备结果。
3. `no_grad` 或 inference_mode，区分图构建成本。
4. compile backend='eager'，只验证 Dynamo 捕获。
5. backend='aot_eager'，加入 AOTAutograd。
6. backend='inductor'，再分析融合、代码生成和 kernel。

每一步都比较数值误差、图数量、断裂数、耗时和显存。如果在 backend='eager' 就失败，问题多半不是 Inductor kernel；如果只在 Inductor 失败，再收集最小 FX 图和版本信息。

### 6.16.3 常用工具和边界

- `torch.profiler`：跨 CPU/CUDA 的算子级时间、形状、内存和调用栈。
- `torch.utils.benchmark`：控制线程、重复和统计的微基准。
- `torch._dynamo.explain`、`TORCH_LOGS`：图断裂、guards、recompile，接口可能变化。
- `torch.cuda.memory_summary()`：快速查看分配统计。
- `_record_memory_history` + `_dump_snapshot`：分析 allocator 历史，只覆盖 PyTorch allocator 可见的分配。
- Nsight Systems/Compute：驱动、kernel、PCIe、SM 和内存管线；需要匹配 CUDA 工具链。

### 6.16.4 结果报告模板

每次性能实验建议包含：

```text
代码提交/脚本：
PyTorch、Python、CUDA runtime、驱动、硬件：
输入 shape/dtype/layout、batch 与随机种子：
模式：eager/compile，backend，dynamic，train/eval：
预热次数、重复次数、同步边界：
编译时间、稳态 p50/p95/p99、吞吐：
allocated/reserved/峰值，是否记录 snapshot：
数值误差与失败样本：
结论、适用范围、未验证假设：
```

## 6.17 版本边界与迁移注意

本章以 2026-10-05 可访问的 PyTorch 2.x 文档概念为主，示例尽量覆盖 2.0 之后的公共 API；具体环境仍需锁定 `torch.__version__`、CUDA runtime、驱动和后端。以下项目明确存在版本边界：

1. `torch.compile` 的默认 backend、dynamic shape 策略、recompile 上限和日志字段会变化。不要依赖未公开的缓存目录结构或日志正则。
2. Dispatcher 的 DispatchKey 顺序和内部注册表是实现细节；`torch._C._dispatch_dump_table`、`TorchDispatchMode` 适合教学和诊断，不宜作为稳定业务接口。
3. Activation checkpointing 从 2.9 起要求显式传 `use_reentrant`；推荐 `use_reentrant=False`，旧版本示例可能省略该参数。非重入实现支持更完整的 autograd 语义，并可能提前停止重算。
4. `torch.compile` 训练时通常由 AOTAutograd 生成 forward/backward segment，而用户的 `.backward()` 仍由 autograd engine 调度；compiled autograd 是更晚引入的扩展，不能和基础 AOTAutograd 混为一谈。
5. CUDA allocator 配置变量曾使用 `PYTORCH_CUDA_ALLOC_CONF` 别名；新文档偏向 `PYTORCH_ALLOC_CONF`。部署前查看目标版本文档，避免拼写存在但不生效。
6. `memory_allocated/reserved` 只描述 PyTorch caching allocator 可见的部分，NCCL、直接 `cudaMalloc`、图形驱动和其他进程可能不在统计中。
7. MPS、ROCm、XPU、CPU 后端有不同的异步、库调用和 allocator 语义。CUDA stream 结论不能未经验证迁移到其他设备。
8. CUDA Graphs 要求更稳定的 shape、指针和内存生命周期，适合低 launch 开销场景；它牺牲部分 eager 动态性，不是 `torch.compile` 的自动等价物。

## 6.18 六个理解检查（含答案）

### 检查 1：为什么 `time.perf_counter()` 可能低估 CUDA 算子时间？

**答案**：CUDA kernel 通常先入当前 stream，Python 调用可在 kernel 完成前返回。计时器只覆盖 CPU 入队时间；应使用 CUDA event 测量设备时间，或在边界显式 synchronize 测端到端时间。event 时间和端到端时间回答不同问题，不能互换。

### 检查 2：`memory_reserved` 比 `memory_allocated` 大是否必然泄漏？

**答案**：不是。reserved 包括 allocator 为复用而缓存的未使用块，通常会大于活跃 Tensor 的 allocated。泄漏更像是 allocated 随步数上升，或 Python 引用持续持有图；碎片和跨 stream 生命周期则可能让 reserved 很大且分配仍失败。

### 检查 3：图断裂为什么会影响性能，即使结果仍然正确？

**答案**：断裂段回到 Python/eager，可能增加调度开销、物化中间结果并触发设备到主机同步。编译器只能在连续图内做融合和内存规划。正确性由回退保证，但吞吐和尾延迟可能恶化。

### 检查 4：`detach()`、`no_grad()`、`inference_mode()` 的共同点和差异是什么？

**答案**：都可减少某些梯度记录，但作用域不同。`detach()` 是共享 storage 的无梯度视图；`no_grad()` 是上下文级别的图记录开关；`inference_mode()` 进一步关闭部分 view tracking/version counter，适合纯推理，但其 Tensor 不能随意带回后续 autograd。三者都不会自动删除已有 Python 引用。

### 检查 5：checkpoint 为什么节省激活显存却增加计算？

**答案**：正向只保存少量输入，反向需要时重新执行被包裹的 forward，因此释放了原本保存的激活，代价是重算和 RNG 状态管理。参数、梯度和 optimizer state 不会因 checkpoint 自动消失。

### 检查 6：为什么 `torch.compile` 的首次调用不应直接和 eager 单次调用比较？

**答案**：首次 compile 包含 Dynamo 捕获、AOTAutograd 分区（训练时）、Inductor 代码生成和编译缓存写入；eager 单次调用没有这些固定成本。应分开报告首次、缓存命中稳态和重新编译情形，并按真实请求形状分布计算摊销。

## 6.19 练习

1. **画调用链**：对 `x.relu().sum().backward()` 画出 Python、dispatcher、Autograd Node、CUDA stream、allocator 和 optimizer 的顺序。标注哪些步骤可能是异步的。
2. **复现图断裂**：写一个包含 `.item()`、`print` 和第三方数学库的函数，分别用 eager、compile、`fullgraph=True` 运行。记录 graph break 数、同步点和数值结果。
3. **形状分桶实验**：为变长序列设置 4 个长度桶，比较固定桶、完全动态和逐样本 compile 的编译次数、稳态吞吐和 p99。
4. **allocator 压力测试**：交替申请 64、96、128 MiB Tensor，随机释放其中一部分，记录 allocated、reserved 和 snapshot。改变释放顺序，观察碎片迹象。
5. **stream 依赖实验**：在两个非默认 stream 中生产和消费 Tensor，故意去掉 `wait_stream`，再加回依赖。使用结果校验和时间线验证差异，并说明为什么竞态不一定每次复现。
6. **checkpoint 预算**：给一个 12 层 MLP，测量不 checkpoint、每 3 层 checkpoint、每层 checkpoint 的峰值显存和训练时间，选择满足显存上限的最优点。
7. **编译分层诊断**：同一个训练函数依次使用 backend='eager'、'aot_eager'、'inductor'，将失败和性能差异归因到 Dynamo、AOTAutograd 或后端代码生成。
8. **写性能报告**：使用本章模板提交一页报告，明确硬件、版本、预热、同步、shape 分布、统计方法和未验证假设。

## 6.20 来源地图

以下来源用于核对 API 语义和版本边界，阅读时应选择与环境匹配的文档版本：

- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)：动态反向图、grad mode、inference mode、saved tensors 和版本计数器。
- [torch.autograd.backward](https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html)：梯度计算、叶子 `.grad` 累积和 `retain_graph`。
- [Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html)：DispatchKey、redispatch、`__torch_dispatch__` 和自定义算子边界。
- [Dispatcher tutorial](https://docs.pytorch.org/tutorials/advanced/dispatcher)：dispatcher 选择 CPU/CUDA/Autograd 实现的概念路径。
- [torch.compile API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)：`fullgraph`、`dynamic`、backend 和 guard/recompile 语义。
- [torch.compiler programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html)：graph break、guards、动态 shape 和日志诊断。
- [torch.compiler FAQ](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html)：训练时 Dynamo、AOTAutograd、Inductor 与 eager autograd engine 的关系。
- [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)：异步执行、stream/event、缓存 allocator、`record_stream` 和 CUDA Graphs。
- [Tensor.record_stream](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)：跨 stream 使用时的 allocator 生命周期约束。
- [Understanding CUDA memory usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)：memory history、snapshot 和可见性边界。
- [Activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)：非重入/重入实现、RNG 保存、early-stop 和 `use_reentrant` 版本要求。
- [Compiled Autograd tutorial](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial)：比 AOTAutograd 捕获更大 backward 图的扩展边界。
- [Graph transformations performance blog](https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations/)：eager 与 graph 模式的性能权衡背景。

## 6.21 小结

PyTorch 的“执行”不是单一引擎，而是一条协作链：Python 产生语义，dispatcher 选择实现，autograd 记录可微历史，backend kernel 在 stream 上异步运行，allocator 管理 storage 生命周期，compile 则在部分边界上把多个调用变成可缓存的图。性能优化的关键不是背诵某个开关，而是保持测量边界和层次对应。

遇到慢、OOM 或结果不一致时，按以下顺序行动：先固定版本与输入契约；再用 eager 建立正确性基线；随后用 profiler、event、memory stats 和日志确认时间、内存、图断裂及同步位置；最后才选择 compile、checkpoint、布局调整、stream 并发或自定义 kernel。每个优化都写出收益、额外成本和失效边界，下一次升级 PyTorch 时重新验证，而不是把一次机器上的偶然数字当成普遍规律。

## 6.22 执行链深潜：把“黑盒”拆成可测量的状态机

前面的章节给出主线，下面补足几个经常被忽略、却会决定故障形态的细节。把一次训练 step 看成状态机更有用：每个状态都有进入条件、可观察计数器和退出事件。典型状态包括 Python 准备、图捕获、CPU 调度、设备执行、同步等待、梯度累积、参数更新和缓存回收。优化的目标不是让某个状态单独最短，而是减少无谓状态转换和相互等待。

### 6.22.1 ATen 算子、复合算子与分解

一个高层 API 可能是复合算子，也可能直接映射到原生 ATen 实现。复合算子会在 Python 或 C++ 中调用一串更基础的算子；原生实现则可能一次完成更多工作。两者对 eager、autograd 和 compile 的影响不同：复合路径更透明、可组合，原生路径可能调用高度优化的库 kernel，但对输入布局和版本更敏感。

编译器常把算子分解（decomposition）为更容易优化的原语。例如把一个复杂激活函数拆成加、乘、指数等节点，可能暴露融合机会，也可能增加数值误差和临时节点。分解不是无条件更快；要同时看算术强度、内存读写、误差和 kernel launch 数量。对需要严格复现的模型，记录是否启用了特定 decomposition，并在升级后重新比较数值。

使用 profiler 时，区分三种名称：用户模块名、aten operator 名和实际 kernel 名。模块名回答“代码写在哪里”，aten 名回答“dispatcher 收到了什么”，kernel 名回答“设备执行了什么”。只看其中一种容易误诊。例如多个 aten 节点可能融合为一个 Triton kernel，或者一个 aten 节点可能因布局和 dtype 选择不同库 kernel。

### 6.22.2 融合、临时张量和内存带宽

逐元素链 `y = (x + a).relu() * b` 在 eager 中可能产生两个或三个中间 Tensor，每个 Tensor 都需要读写显存。若编译器把它们融合，理论上可以减少中间写回；但融合后的 kernel 可能寄存器压力更大，降低 occupancy，或因形状不规则而退化。

可以用一个粗略账本判断方向。设每个元素需要读写 `R` 字节、做 `F` 次浮点操作，算术强度为 `I=F/R`。当 `I` 很低时，减少内存往返常比增加算力更重要；当 `I` 很高时，融合是否减少读写的收益可能被寄存器和指令吞吐抵消。实际判断仍要用设备时间线和内存吞吐计数器验证。

不要把融合数量当成功指标。更有意义的是：端到端时间、显存峰值、kernel 数、内存读写量、SM occupancy、数值误差和长尾行为。某个版本的 Inductor 可能选择不同调度，升级后需要重新基准。

### 6.22.3 CPU 线程池、NUMA 与数据准备

GPU 不是唯一的执行资源。PyTorch CPU 算子、DataLoader worker、tokenizer、图编译线程和日志线程会争抢 CPU。`torch.set_num_threads`、`set_num_interop_threads`、OpenMP/MKL 环境变量以及进程绑核都会改变结果。多路 NUMA 机器上，如果数据在错误的节点分配，内存带宽和 H2D 拷贝会出现难以解释的抖动。

CPU 基准应记录线程数和绑核方式，避免把“线程更多”误认为“算子更高效”。DataLoader worker 数过少会饿死 GPU，过多则增加上下文切换、页缓存和 pinned memory 压力。对端到端吞吐，分别测数据准备、拷贝和计算时间，再决定要不要加 worker 或预取深度。

### 6.22.4 分布式训练中的 stream 与通信

DDP、FSDP 和张量并行会引入通信 stream、通信库 workspace 和额外的同步边界。梯度 bucket 可能在部分梯度就绪后启动 all-reduce，与后续反向计算重叠；如果 bucket 太小，通信启动开销变大，如果太大，重叠窗口又太晚。观察 GPU 利用率时，应区分计算 kernel 和 NCCL kernel，不能只看总百分比。

常见故障是某个 rank 先进入同步，其他 rank 因数据 shape、图断裂或异常分支仍在计算，最终表现为“随机卡住”。排查时记录每个 rank 的 step、shape、collective 名称和 stream；先在单卡 eager 复现数学问题，再回到多卡分析通信。不要用全局 `cuda.synchronize()` 掩盖真正的 rank 间依赖，它会让性能基线失真。

### 6.22.5 数值稳定性、精度和执行顺序

AMP、TF32、bf16、fp16 和 fp32 不只是存储格式，还会改变 kernel 选择、累加精度和溢出行为。融合后改变加法顺序，也可能让最后几位不同；分布式归约的顺序变化会放大差异。验收应区分“位级一致”“容差内一致”和“任务指标一致”，并为每种精度定义容差。

GradScaler 的动态缩放遇到 inf/nan 会跳过 step 并调整 scale。若把跳过的 step 当成正常吞吐，训练速度统计会失真；若只监控 loss，不监控 scale 和溢出计数，可能错过长期退化。checkpoint 重算还涉及 RNG 状态，关闭 RNG 保存会改变 dropout 轨迹，即使最终 loss 看起来相近。

### 6.22.6 生产基线与回滚开关

把性能特性做成可回滚开关，至少包括：eager/compile、compile backend、dynamic shape、autocast dtype、checkpoint 策略、allocator backend 和通信重叠。每个开关都应有启动日志，打印实际生效值，而不是只打印配置文件中的期望值。

发布前执行三组测试：

1. **正确性组**：固定 seed、边界 shape、空 batch、极端长度、nan/inf 输入，比较 eager 与优化路径的输出和梯度。
2. **稳态性能组**：预热到 compile cache 命中，测 p50/p95/p99、吞吐、CPU/GPU 利用率和峰值显存。
3. **恢复组**：触发 OOM、编译失败、kernel 异常、通信超时和进程重启，验证能否回退 eager 或清理缓存。

回滚不仅意味着换回旧代码，也包括清理编译缓存、重启持有坏状态的 worker、停止继续增长的 recompile 计数和恢复 allocator 配置。把这些动作写入运行手册，值班人员才能在 p99 恶化时快速止损。

### 6.22.7 一个可复用的三分钟诊断脚本

```python
import os, platform, torch
print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name())
    print("runtime", torch.version.cuda)
    print("alloc", torch.cuda.memory_allocated(),
          "reserved", torch.cuda.memory_reserved())
print("grad enabled", torch.is_grad_enabled())
print("threads", torch.get_num_threads())
print("compile", hasattr(torch, "compile"))
print("alloc conf", os.getenv("PYTORCH_ALLOC_CONF") or
      os.getenv("PYTORCH_CUDA_ALLOC_CONF"))
```

脚本不替代 profiler，但能在报告开头锁定最常见的环境差异。若结果异常，下一步分别运行 eager CPU、eager 设备、compile backend='eager'，把故障定位到语义、设备或编译层。诊断输出中不要包含凭据、路径中的个人信息或生产请求内容。

## 6.23 章节完成标准

读者完成本章后，应能对任意一段 PyTorch 代码回答五个问题：

- 它在 eager、compile 还是自定义后端中运行？
- dispatcher 选择了哪些功能层和设备实现？
- autograd 保存了什么，何时释放，是否存在 in-place 或引用问题？
- kernel 排到了哪个 stream，哪里发生同步，allocator 如何管理其 storage？
- 结论依赖哪些版本、shape、dtype、硬件和测量边界？

如果这五个问题还答不出来，继续添加日志通常只会产生更多噪声。回到最小复现，明确一个假设，设计一个能证伪它的实验，再决定是否改代码。这样的执行链思维比记住某个版本的内部函数名更能跨设备和跨版本迁移。

## 6.24 自定义算子与扩展的安全边界

当公共算子无法满足需求时，可以用 C++/CUDA 扩展、Triton kernel 或 `torch.library` 注册自定义算子。注册算子至少要考虑 schema、CPU/CUDA 实现、Meta 实现、autograd 规则、复合分解、别名和可变性。只实现 CUDA kernel 而没有 Meta 或 fake 实现，可能导致 compile 无法推断 shape；只在 eager 中测试而没有 backward，训练时才会暴露梯度断裂。

自定义算子的最小验收矩阵包括：连续和非连续输入、空维度、不同 dtype、CPU 与目标设备、requires_grad 开关、`vmap`（若需要）、compile 捕获、异常输入和多 stream 使用。对 in-place 算子还要验证版本计数器和 alias 关系，避免静默覆盖用户仍在使用的 storage。

性能上，先测算子本身，再测包含数据准备和同步的端到端路径。Triton kernel 的块大小、warp 数、寄存器和共享内存会随设备架构变化；把某一张 GPU 上的最佳参数硬编码到所有型号通常不稳。发布扩展时锁定 ABI、编译器、CUDA 和 PyTorch 主次版本，并准备 eager fallback。升级 PyTorch 后先跑正确性矩阵，再跑性能基线，最后才打开生产流量。

安全上，不要从不受信任的仓库直接加载预编译扩展或 pickle 模型。扩展拥有进程级权限，错误的指针算术既可能导致数据损坏，也可能让驱动复位。把编译、测试和生产加载分离，记录构建哈希和来源，出现 kernel 异常时优先回退到官方算子路径。
