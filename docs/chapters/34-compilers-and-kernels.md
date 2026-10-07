---
id: ch34-compilers-and-kernels
title: 编译器、Kernel 与算子工程：从图捕获到 Triton/CUDA/融合内核
slug: /chapters/34-compilers-and-kernels
description: 从 PyTorch 2.x 编译器与 Inductor、XLA/MLIR、Triton/CUDA，到图捕获、算子融合、自动调优和正确性边界
sidebar_position: 34
level: advanced
prerequisites:
  - ch04-performance-math
  - ch05-gpu-cuda
  - ch06-pytorch-execution
  - ch07-numerical-precision
  - ch15-inference-execution
  - ch19-inference-optimization-accelerator-stack
  - ch20-observability-debugging-incident-response
  - ch21-ai-reliability-engineering
learning_objectives:
  - 能区分 eager、graph capture、IR lowering、code generation 和 kernel launch 的边界
  - 能解释 PyTorch 2.x Dynamo/AOTAutograd/PrimTorch/Inductor 以及 XLA/MLIR 的职责分工
  - 能从 shape、dtype、布局、内存带宽和同步点推导算子融合的收益与反例
  - 能用 Triton/CUDA 的线程块、warp、shared memory、register 与异步拷贝心智模型阅读 kernel
  - 能设计 kernel autotuning 的搜索空间、测量协议、缓存键和回滚策略
  - 能运行 CPU-only toy lab，验证图捕获、融合、缓存命中、正确性误差和性能边界
estimated_hours: 48
hardware: CPU-only lab required; CUDA/Triton/XLA measurements optional and must be pinned to target hardware
risk_level: L3
last_verified: 2026-10-07
---

# 第34章　编译器、Kernel 与算子工程：从图捕获到 Triton/CUDA/融合内核

> 深度学习性能经常被一句“换个 kernel”概括。真正发生的事情更长：Python/eager 程序先被捕获为图，图中的高阶算子经过分解、自动微分和规范化，再 lowering 到某个中间表示（IR），编译器选择布局、融合边界和调度，代码生成器产出 Triton、CUDA、LLVM 或设备专用程序，运行时把它编译、缓存、加载并提交到 stream。任意一个阶段都可能因为动态控制流、别名、shape、dtype、设备能力或数值约束而退回 eager。一个看似“快”的融合 kernel 也可能让寄存器压力、编译时间、缓存失效和错误定位成本上升。

本章把 compiler、kernel 和算子工程放到同一条可审计链路中。我们会解释 PyTorch 2.x 的 TorchDynamo、AOTAutograd、PrimTorch 和 TorchInductor，比较 XLA/StableHLO/MLIR 的模块化路径，拆读 Triton/CUDA 的执行模型，推导融合收益的边界，设计 autotuning 与正确性门禁，最后用不依赖 GPU 的 toy lab 模拟图捕获、融合决策、编译缓存和误差检查。实验数字是协议代理，不是 NVIDIA、AMD、TPU 或任何云实例的性能承诺。

## 34.1 问题/边界：为什么“编译一下”不是答案

### 34.1.1 五个不同的问题

当有人说“模型没有跑满”，至少可能在问五件事：

1. **捕获问题：** Python 分支、数据依赖 shape、未支持的算子或副作用让图在某处 break；
2. **表示问题：** 图已捕获，但 layout、dtype、alias 或动态维度让优化器无法安全重写；
3. **调度问题：** IR 已 lower，融合和 tile 选择不合适，产生过多内存往返或同步；
4. **代码生成问题：** Triton/CUDA/LLVM 生成代码的 occupancy、vectorization、shared-memory bank conflict 或寄存器使用不佳；
5. **运行时问题：** 编译缓存 miss、首次编译、stream 依赖、内存分配、通信或 host launch 成本主导。

因此，单看 GPU 利用率不能定位根因。一个 kernel 可能只花 50 微秒，但每个 step 有 2,000 次 launch；另一个大型 GEMM 利用率很高，却因前后 layout copy 和同步损失收益。要把 trace 中的 `graph_id、guard_id、kernel_id、shape_key、compile_cache_key、stream、dtype、layout` 关联起来，才知道优化的是哪一段。

### 34.1.2 本章覆盖和不覆盖

本章覆盖训练和推理中常见的 pointwise、reduction、matmul 周边算子，动态图捕获、静态/动态 shape、融合、自动调优、验证和回退。不会给出某个 GPU 型号的“最佳 tile”，不会替代目标框架的 release note，也不讨论完整分布式编译集群调度。CUDA Graph、NCCL、专用 attention kernel 和量化 kernel 会作为边界案例出现；若要部署，必须在目标 driver、runtime、GPU 架构和模型版本上复测。

### 34.1.3 先写性能预算

设一个请求的端到端延迟为

\[
T= T_{capture}+T_{compile}+T_{launch}+T_{memory}+T_{compute}+T_{sync}+T_{fallback}.
\]

`T_capture` 和 `T_compile` 可能只在 warm-up 或新 shape 首次出现，也可能在服务进程中周期性出现。`T_memory` 是 DRAM/HBM/L2/共享内存之间的移动；`T_compute` 是算术；`T_sync` 包括 stream、event、barrier 和 host/device 等待；`T_fallback` 是图 break 后回到 eager 的额外开销。优化前先测冷启动、热启动、shape 变化和 p50/p99，避免把 compile time 隐藏在“第一次请求不算”的脚本里。

## 34.2 心智模型：一条带守卫和证据的编译流水线

一句话模型：**编译器不是把 Python 变成更快的 Python，而是在一组可验证的守卫（guards）下，把带布局和数值契约的算子图改写成设备程序；每次改写都要留下可重放的 IR、配置、测量和回退证据。**

可以把运行时看成下图：

```text
user code
  │  Python bytecode / tensor metadata
  ▼
Dynamo capture ── guard(shape,dtype,device,alias,grad)
  │ graph breaks + resume points
  ▼
AOTAutograd / functional graph
  │ forward + backward, saved tensors
  ▼
PrimTorch / ATen normalized ops
  │ explicit broadcasts, reductions, layouts
  ▼
Inductor / XLA / MLIR backend IR
  │ fusion, tiling, scheduling, memory planning
  ▼
Triton/CUDA/LLVM/TPU codegen ── autotune/cache
  │ launch config + stream dependencies
  ▼
device kernel(s) + fallback eager path
```

图捕获不是一次性“录屏”。守卫描述了图成立的前提，例如 `x.shape[1] == 4096`、dtype 是 `bf16`、设备是 `cuda:0`、某个对象没有被别名写入。守卫失败会触发重新编译或 graph break。编译缓存的 key 至少要包含代码/IR digest、设备 capability、dtype、shape/layout 和调优配置；否则不同模型或 GPU 可能错误复用二进制。

### 34.2.1 四种证据

- **语义证据：** eager 与编译路径的输出/梯度在容差内一致，随机种子、异常和 alias 行为符合契约；
- **编译证据：** 捕获图、break 原因、lowered IR、生成源码、编译器版本和 guard 集合；
- **性能证据：** warm/cold、shape 分布、batch、并发、p50/p99、编译 amortization 和峰值内存；
- **运维证据：** cache 命中/失效、编译失败率、回退率、kernel 版本、驱动和可回滚配置。

没有这些证据，单次 `time()` 只能说明某次调用花了多久。

## 34.3 机制：PyTorch 2.x compiler 栈

### 34.3.1 TorchDynamo：在 Python 边界捕获

TorchDynamo 通过 CPython frame evaluation hook 观察字节码，将可追踪的 Tensor 操作转成 FX Graph。它不是完整 Python 编译器：对数据依赖分支、任意副作用、未注册 dispatcher、对象身份和动态容器，可能插入 graph break。每个 break 把程序切成多个子图，子图间回到 Python/eager，带来额外 launch、同步或 tensor materialization。

`torch.compile(model, backend="inductor")` 默认会在首次遇到新 guard 集合时编译。`fullgraph=True` 可把 break 视为错误，适合找出不可捕获点；`dynamic=True` 允许某些维度符号化，但扩大守卫和代码生成空间。工程上不要盲目打开 dynamic：如果真实请求只有几个固定 shape，静态 specialization 可能更快、更易缓存；如果 shape 高度离散，静态版本会造成编译风暴。

建议为每个 graph break 保存：文件/行号、触发的 op 或 Python 构造、输入 metadata、是否可通过重写消除、回退耗时。`torch._dynamo.explain`、TORCH_LOGS 和 `torch._dynamo.config` 是诊断入口，但这些内部 API 会随版本变化，生产脚本应锁定 PyTorch 版本。

### 34.3.2 AOTAutograd：把反向也变成图

AOTAutograd（Ahead-Of-Time Autograd）在捕获阶段生成 forward 和 backward graph，允许编译器统一做融合、内存规划和 saved-tensor 选择。反向图不是“免费复制 forward”：它可能需要保存中间激活，或重新计算（recompute）来换内存。一个前向融合若改变了中间精度、舍入或 NaN 传播，反向梯度也可能不同。

训练编译的预算要把 `saved_bytes、recompute_flops、optimizer step、gradient accumulation` 一起算。只比较 forward kernel 时间会低估 compile 和内存的代价。对有自定义 autograd Function、随机算子、in-place 写和跨设备通信的模型，应先用 eager 与 compiled 的小 batch 做逐层梯度检查，再扩大规模。

### 34.3.3 PrimTorch/ATen：规范化算子语义

PrimTorch 把大量 ATen 算子分解成较小的 primitive 集，降低后端需要实现的语义数量。例如广播、view、slice、reduction 的边界被显式化，后端可以识别 pointwise 链和内存别名。分解并不保证更快：若把一个设备原生高效算子拆成许多 primitive，可能丢失专用实现；后端通常保留“opaque”调用给 cuBLAS、cuDNN、FlashAttention 等库。

阅读 IR 时要区分：

- `view/reshape/expand` 多数只改 metadata，但不连续 layout 时后续 kernel 可能需要 copy；
- `permute/transpose` 改 strides，可能让向量化变差；
- `sum/max/softmax` 是 reduction，有数据依赖和数值顺序；
- `mm/conv` 常由库调用，tile/algorithm 由 cuBLASLt、cuDNN 或后端选择；
- `copy/convert` 往往是隐藏的带宽成本。

### 34.3.4 TorchInductor：融合、调度和代码生成

TorchInductor 接收 FX/ATen 图，先做布局和内存规划，再把可融合的 pointwise/reduction 区域生成 Triton（GPU）或 C++/OpenMP（CPU）代码；矩阵乘、卷积等通常调用外部库。Inductor 的“融合”是受约束的：有副作用、不同设备、别名、layout 冲突、过大的中间结果或需要不同 launch 形状时会拆开。

一个典型链是 `add -> mul -> gelu -> dropout`。若每个 op 独立，可能产生 4 次读、4 次写和 4 次 launch；融合后一次 kernel 读输入、在寄存器中计算并写输出，减少 HBM 往返。可是当链很长、每线程寄存器超过阈值，occupancy 降低甚至溢出到 local memory；当 `dropout` 需要 RNG 状态、训练/推理分支时，融合边界会不同。查看 generated code、kernel launch shape 和显存分配，而不是只看图上“fusion group”数量。

### 34.3.5 XLA、StableHLO 与 MLIR：另一条模块化路线

XLA 的 HLO/StableHLO 图强调设备无关的高层算子和静态/符号 shape，编译器再做 layout assignment、fusion、collective 和设备 lowering。StableHLO 作为可移植的 MLIR 方言，定义了版本化的算子语义，便于 JAX、TensorFlow、PyTorch/XLA 之间交换模块。MLIR 通过 dialect、pass、conversion 和 pattern rewrite 组合前端与后端；一个项目可从 linalg/tensor/arith 逐步转到 LLVM、GPU 或 TPU dialect。

XLA/MLIR 的优势是模块化和跨设备重用，代价是编译 pipeline、shape polymorphism 和调试层次更多。`HLO dump` 看见的 fusion 不等于最终一个物理 kernel：后端可能再拆分、调用库或插入 copy。比较 XLA 与 Inductor 时，要统一 batch、shape、编译 warm-up、cache、精度和 layout；不要把不同的“首次运行”混为一谈。

## 34.4 Kernel 机制：从线程到内存事务

### 34.4.1 CUDA 的层级

CUDA kernel 以 grid 启动 block；每个 block 含 threads，通常以 warp（NVIDIA 上 32 threads）锁步执行。线程拥有 registers；block 共享 shared memory 和同步原语；所有 block 通过 global memory、L2 和设备调度器交换数据。一个 kernel 的性能受计算吞吐、内存带宽、延迟隐藏、分支分歧、occupancy、同步和 launch 开销共同限制。

定义算术强度 \(I=F/B\)，其中 `F` 是浮点操作数，`B` 是从慢速内存读写的字节。若设备峰值计算为 `P`、带宽为 `M`，roofline 上限近似为 `min(P, I·M)`。融合通常减少 `B`、提高 `I`；但若增加寄存器和 instruction count，实际可达值可能下降。用 Nsight Compute 或 CUPTI 采集 dram bytes、achieved occupancy、sm throughput、branch efficiency，才能判断瓶颈。

### 34.4.2 Triton 的块级编程

Triton 让开发者以 program instance 的块为单位写 kernel，编译器负责映射到 threads/warps。例如向量加法可让每个 program 处理 `BLOCK_SIZE` 元素，使用 `tl.load`/`tl.store` 和 mask 处理尾部。`num_warps`、`num_stages`、`BLOCK_SIZE` 影响寄存器、共享内存、并行度和 pipeline。Triton 代码短并不代表语义简单：指针算术、stride、mask、dtype promotion 和原子操作仍决定正确性。

减少 global-memory 往返是 pointwise 融合的主要动机。对 reduction、softmax、attention，需处理跨元素依赖、数值稳定性和临时空间。比如 log-sum-exp 要先求最大值再求指数和，两个 pass 可能比一个不稳定的 pass 慢一点但正确；在 bf16 下把累加器设为 fp32 常是必要的设计选择。

### 34.4.3 异步拷贝与 shared memory

新架构支持把 global 到 shared 的拷贝与计算流水化。tile `K` 分段，stage 0 在计算 tile 0 时预取 tile 1，靠 barrier 保证可见。增加 stages 可隐藏内存延迟，却占用更多 shared memory；超过每 SM 预算会降低同时驻留的 block 数。所有“异步”都要有依赖图：缺少 wait/barrier 会读到旧数据，过多 barrier 则失去重叠。

### 34.4.4 CPU kernel 与向量化

CPU-only 环境不是 GPU 的低配替代。Inductor CPU 后端可能生成 C++/OpenMP，利用 SIMD、线程池、cache blocking；Triton CPU 支持和版本需单独验证。融合在 CPU 上同样减少 cache miss 和中间数组，但过大的融合可能超过 L1/L2，降低向量化或增加分支。toy lab 用抽象 cost model 表示这些方向，不声称测量 AVX/AMX。

## 34.5 融合工程：收益、反例和边界

### 34.5.1 融合收益方程

设未融合链有 `K` 个 kernel，第 `i` 个读写字节 `B_i`、计算 `F_i`、launch 开销 `L_i`。粗略成本

\[
T_{sep}=\sum_i (L_i + B_i/M + F_i/P).
\]

融合成 `G` 组后，组内中间张量不落地，成本近似

\[
T_{fused}=\sum_g (L_g + B_g/M + F_g/P + C_g),
\]

`C_g` 表示额外寄存器、同步、索引和编译复杂度。只有当省下的内存/launch 成本大于 `C_g` 才值得融合。这个模型解释了三个反例：

- 链很短、tensor 很小，launch 优势不足以摊平 compile；
- reduction/softmax 的工作集大，融合后寄存器或 shared memory 爆炸；
- 一个 op 可调用高度优化的库 kernel，手写融合反而降低矩阵乘效率。

### 34.5.2 别名与 in-place

若 `y = x.view(...); z = y + 1`，view 可能与 `x` 共享 storage。编译器若错误地重排写入，会改变后续读取。需要 alias analysis、mutation tracking 和正确的 storage offset/stride。in-place op 对 eager 有可见副作用，捕获时可能被 functionalize 成 out-of-place；这会增加内存，也可能改变 OOM 边界。测试必须覆盖非连续 tensor、零 stride expand、负 stride（若支持）和共享 storage。

### 34.5.3 动态 shape 和分支

动态 batch、序列长度和 MoE token count 让一个 kernel 覆盖多个 shape。完全多态的 kernel 常包含更多边界判断和保守 layout；完全特化则产生许多编译缓存条目。实务做法是按生产分布聚类 shape，设置 top-N specialization，其余走 dynamic/fallback，并给编译缓存设置上限和淘汰策略。不能只在单一 `1024x4096` 形状上报告 speedup。

### 34.5.4 随机性与确定性

Dropout、随机采样和某些 atomic reduction 的顺序会影响 bitwise 结果。确定性模式可能禁用某些融合或使用更慢算法。定义可接受契约：训练 loss/梯度在统计容差内，还是要求 bitwise reproducibility。保存 RNG seed、offset、Philox counter（如果框架暴露）和 kernel 版本；否则“同 seed”并不足以重现。对混合精度，比较时用相对/绝对容差并检查 NaN/Inf。

## 34.6 图捕获实验：从 eager 到编译缓存

### 34.6.1 捕获协议

一个最小实验应包含三阶段：

1. **warm-up/capture：** 固定输入 metadata，记录 guard 与 graph break；
2. **compile/cache：** 首次编译计时，保存 IR/codegen digest，第二次调用区分 cache hit；
3. **steady state：** 至少 30 次热运行，报告中位数、p95、p99 和输出误差。

输入要覆盖 contiguous/non-contiguous、不同 batch、dtype、requires_grad 和设备。若使用 `torch.compile`，可用 `torch._dynamo.explain` 查看 break；若使用 XLA，导出 StableHLO/HLO 并记录编译选项。实验报告需注明是否包括同步（CUDA 上常用 `torch.cuda.synchronize()`）、是否包括首次 allocator 和 CPU 到 GPU 拷贝。

### 34.6.2 CPU toy 的抽象

本章 lab 不导入 PyTorch、Triton 或 CUDA。它把算子图表示为带 shape、dtype、bytes 和 FLOPs 的节点，把相邻 pointwise 节点融合，把 shape/dtype/backend 组成缓存 key，用公式估计 launch、memory 和 compile 成本，并用 Python 计算 eager/fused 的真实输出检查误差。这样在没有 GPU 的环境也能审计决策和失败边界；它不能证明真实 kernel 的 occupancy 或 HBM 带宽。

## 34.7 Kernel autotuning：搜索不是魔法

### 34.7.1 搜索空间

一个 autotuner 的配置向量可以是

\[
\theta=(B_M,B_N,B_K,W,S,vec,algo,dtype),
\]

其中 `B_*` 是 tile，`W` 是 warps/threads，`S` 是 pipeline stages，`vec` 是向量宽度，`algo` 是库算法。先用设备能力和资源约束过滤：tile 必须覆盖 shape，shared memory、register 估算不能超过上限，线程数不超过 block 上限，dtype 与指令集兼容。候选数过大时采用分层搜索：先粗粒度 tile，再在前几名上调整 warps/stages。

### 34.7.2 测量协议和噪声

每个候选至少 warm-up 若干次，随后采样固定次数，用 median/p90 而不是单次最小值。锁定 CPU governor、GPU clocks（若许可）、并发、stream、输入分布和 allocator；在不同时间重复测量，记录温度和 throttling。若候选 A 在 1% 内快于 B，但方差更大，应考虑稳定性和尾延迟。autotune 时间本身是成本：服务启动若有 500 个 shape，每个 shape 20 个候选，编译/测量可能超过业务 SLA。

### 34.7.3 缓存键、持久化和回滚

缓存 key 至少包含 kernel source digest、compiler/runtime 版本、GPU compute capability、driver、shape/layout/dtype、调优约束和确定性模式。把 `best_config` 单独存储并带测量 evidence；升级驱动或改动代码时应使旧 key 失效。在线 autotune 不应直接覆盖生产配置：先在影子流量验证，再通过 feature flag 灰度；发现 NaN、p99 回归或 OOM 时一键回滚到库 kernel/未融合路径。

### 34.7.4 过拟合和目标函数

仅优化平均 latency 可能选择对大 batch 最佳、对小 batch 极差的配置。目标函数可写为

\[
J=\operatorname{p50}(T)+\lambda\operatorname{p99}(T)+\mu\,\text{compile\_seconds}+\nu\,\text{memory\_bytes},
\]

其中权重来自产品 SLO。对训练，吞吐/step 和总 wall-clock（含编译）更重要；对在线 serving，p99、冷启动和并发下的显存更重要。报告必须给出目标函数，而不是只说“autotune 找到最快”。

## 34.8 正确性：编译优化的安全网

### 34.8.1 参考实现和容差

为每个新 kernel 保留清晰的 reference（通常是 eager/PyTorch 或 NumPy）。测试维度包括：极小/非整除 shape、零长度（若允许）、大值/小值、NaN/Inf、负值、不同 stride、不同 dtype、随机 seed 和 gradient。对输出 `y_ref,y_opt`，计算

\[
\text{max\_abs}=\max |y_{opt}-y_{ref}|,\quad
\text{rel}=\max \frac{|y_{opt}-y_{ref}|}{\max(|y_{ref}|,\epsilon)}.
\]

容差应绑定 dtype 和算法，而不是统一写 `1e-5`。bf16/fp16 累加 fp32 仍可能有 reduction 顺序差异；需要检查相对误差分布和最终任务指标。

### 34.8.2 梯度、随机数和异常

训练路径要做 finite-difference 或 autograd gradcheck 的小规模验证，覆盖 forward/backward 各分支。若 kernel 产生 NaN，捕获时应保留最小复现输入、shape key、配置和生成源码 digest。异常路径（除零、越界 mask、空 reduction）必须与 eager 一致或明确记录差异。不要因为“正常样本没错”就跳过毒性输入。

### 34.8.3 Differential、metamorphic 和 fuzz

Differential testing 在多个实现间比较：eager、compiled、库 kernel、自定义 kernel。Metamorphic testing 则检查变换不变量，例如给 pointwise 输入加零、按 batch 置换后输出同样置换、缩放输入时线性层比例关系保持。shape/dtype/stride fuzz 可以发现只在尾块、非连续内存或动态维度出现的错误。每次失败都写入 corpus，作为回归测试。

### 34.8.4 性能回归门禁

正确性通过不代表性能通过。CI 可在固定 CPU toy 或受控 GPU runner 上给出宽松门槛：编译时间不能超过上限，缓存命中路径不能重新编译，p95 不得恶化超过阈值，峰值内存有预算。硬件噪声大时使用基线区间和多次 run；不要因为一次快 2% 就更新黄金值。

## 34.9 机制对照：PyTorch Inductor、XLA/MLIR、Triton/CUDA

| 层次 | PyTorch 2.x / Inductor | XLA / StableHLO / MLIR | Triton / CUDA |
| --- | --- | --- | --- |
| 输入 | Python + ATen/FX | HLO/StableHLO、JAX/TF/PyTorch 前端 | 手写 kernel 与 launch |
| 优化粒度 | 图、fusion group、布局和内存规划 | HLO fusion、layout、设备 pass | block/warp、指针、同步 |
| 主要守卫 | shape、dtype、device、alias、Python 对象 | shape、layout、编译选项、设备 | launch 参数、指针、mask、资源 |
| 代码生成 | Triton、C++/OpenMP、库调用 | LLVM、GPU/TPU backend、库调用 | PTX/CUBIN 或后端目标 |
| 优势 | 低侵入、自动融合、与 PyTorch 集成 | 跨前端/设备、IR 生态和可移植性 | 精细控制、快速迭代、自定义算子 |
| 风险 | graph break、版本敏感、编译缓存 | 编译层多、调试复杂、shape 爆炸 | 越界/同步/数值错误、维护成本 |

这不是互斥选择。一个系统可能由 PyTorch Dynamo 捕获，Inductor 把 matmul 交给 cuBLASLt，把 pointwise 生成 Triton；另一个系统由 JAX 导出 StableHLO，经 MLIR 和 XLA 调度；关键是记录每个 op 的 owner、IR、库版本和 fallback。

## 34.10 实验：CPU-only 编译/融合 toy lab

实验脚本为 `labs/ch34_compiler_kernel_lab.py`，只用 Python 标准库，默认 Python 3.10+。它提供：

- `capture_graph`：检查节点输入输出 metadata，标记动态 shape 和不支持的副作用；
- `fuse_graph`：只融合相邻且同 dtype/layout、无 reduction/alias 的 pointwise 节点；
- `compile_plan`：按 backend、shape、dtype、配置生成确定性 cache key，区分 hit/miss；
- `estimate_cost`：以 launch + bytes/bandwidth + flops/compute 的代理模型比较 eager/fused；
- `run_reference` 与 `max_abs_error`：对一维数据执行 add/mul/relu/gelu，检查融合前后数值；
- `autotune`：在有限 tile/warps/stages 候选中用确定性代理目标选择配置，并暴露 compile budget；
- `simulate`：一次输出图、融合组、缓存、成本、误差、失败边界和 summary JSON。

默认实验故意包含两个图：一个可融合的 `add -> mul -> relu` 链，一个含 reduction 的边界图。通过 `--dynamic-shape` 或 `--unsupported-op` 可观察 graph break；通过重复运行可观察 cache hit；通过 `--fail-after` 模拟候选正确性失败并回滚到 reference 配置。toy 的 bytes、FLOPs 和带宽数字是可解释的代理，不应写成真实 GPU 性能。

### 34.6.3 生产实现阅读法：从 API 走到二进制

阅读一个 compiler backend 时，不要从“最神奇的 pass”开始。先画一条最小路径并为每段指定 owner：

```text
Python/API -> dispatcher -> capture IR -> decomposition -> scheduler
          -> layout/memory planner -> codegen -> compiler driver
          -> cache/load -> stream launch -> profiler counters
```

对每个箭头问三个问题：输入和输出的 schema 是什么，失败时回退到哪里，怎样拿到可重放证据。比如一个 `aten.add` 可能在 dispatcher 处根据 dtype/device 选择 CPU、CUDA 或 meta kernel；meta kernel 只传播 shape/stride，不能证明真实数据正确。Inductor scheduler 可能把多个 FX 节点分成 pointwise、reduction、extern kernel 三类；extern kernel 再调用 cuBLASLt 或 oneDNN。若只截取最终 kernel 名，无法解释为什么某个 view 触发了 copy。

建立版本矩阵很有帮助。行是 PyTorch、Triton、CUDA driver、GPU architecture，列是模型 commit、shape bucket、dtype、determinism 和 cache schema。每次升级只改变一格并保留旧基线；若性能或误差变化，先查 IR 和 generated source，再查硬件 counters。编译器日志可能含用户代码路径、shape 或输入摘要，生产环境应做脱敏和访问控制。

### 34.6.4 图捕获与服务生命周期

在线服务把编译器放进请求生命周期后，还要考虑并发和部署：编译锁是否阻塞所有请求，cache 是否写共享目录，两个 pod 是否生成相同 binary，滚动升级时旧 kernel 是否仍被引用。常见做法是把编译拆成发布阶段和运行阶段：发布阶段预编译高频 shape，运行阶段只允许有限的新 key，并把新 key 放到隔离 worker；超出预算就回退 eager 或库实现。

CUDA Graph capture 还要求地址、stream、控制流和某些 allocator 状态稳定；它适合重复执行的静态 batch，动态请求则需要 graph pool 或 padding。XLA/StableHLO 服务通常按 shape 编译 executable，shape explosion 同样需要 bucket 和 LRU。无论哪条路线，编译状态都应通过指标暴露：`compile_inflight`、`compile_seconds`、`cache_entries`、`cache_hit_ratio`、`fallback_total`、`guard_fail_total`。

## 34.6.5 形状、布局与内存规划的审计清单

当融合没有预期收益时，依次检查：

- shape 是否来自真实请求，是否在 benchmark 中偷偷固定了 padding；
- dtype 是否在某个转换节点被提升到 fp32，又在输出处转回；
- stride 是否连续，是否产生隐式 contiguous copy；
- alias 是否允许 in-place，是否因安全而复制；
- 中间张量是否被其他分支使用，导致不能释放或不能融合；
- allocator 是否因不同生命周期产生碎片；
- reduction 的临时空间和 workspace 是否计入峰值；
- 设备间 copy、host callback、通信和同步是否隐藏在 kernel 时间之外。

把这些字段写入每条 benchmark 记录，才能区分“编译器没融合”和“融合了但 layout/内存规划抵消了收益”。

## 34.11 故障诊所/失败模式

### 症状 A：第一次请求很慢，后续很快

**可能原因：** compile/cache miss、CUDA context 初始化、allocator warm-up、autotune。**证据：** 分离 `capture_ms、compile_ms、load_ms、steady_ms`，检查 cache key 和编译日志。**修复：** 预热常见 shape、持久化缓存、限制 autotune 候选，或把编译放到发布阶段。**边界：** 预热会消耗资源并不能覆盖长尾动态 shape。

### 症状 B：`torch.compile` 后速度更慢

**可能原因：** graph break、过度特化、融合导致寄存器溢出、库 kernel 被自定义版本替换、动态 shape guard 开销。**诊断：** 对比 break 前后 kernel count、显存读写、occupancy、compile amortization；用 `fullgraph=True` 找 break。**回滚：** 针对问题模块禁用 compile 或设置 backend=eager，保留其余模块；不要全局关闭后失去证据。

### 症状 C：GPU 利用率 90%，p99 仍恶化

**可能原因：** kernel 长尾、并发请求互相阻塞、stream/event 排队、编译锁、显存碎片。**证据：** trace 中看 kernel duration 分布、队列等待、allocator、cache miss；平均 SM busy 不能证明每个请求公平。**修复：** 限制大 shape 并发、拆分过大融合、按请求 class 缓存专用 kernel、设置编译和内存预算。

### 症状 D：误差只在非整除 shape 出现

**可能原因：** mask/边界索引、尾块读取越界、stride 误算。**修复：** 对 `n=0,1,BLOCK_SIZE-1,BLOCK_SIZE,BLOCK_SIZE+1` 做测试，开启 device sanitizer（目标平台），保留最小输入。**不要做：** 用 padding 掩盖错误后直接上线，除非 padding 语义被写入契约。

### 症状 E：训练 loss 偶尔 NaN，重跑又正常

**可能原因：** 未初始化 masked lane、atomic reduction 顺序、fp16 overflow、随机数状态或未同步异步拷贝。**证据：** 固定 seed、启用 anomaly/NaN 检测、记录 kernel config 和输入摘要。**修复：** masked load 使用安全值，累加提升精度，增加必要 barrier，或回退确定性库实现。

### 症状 F：编译缓存越积越多，服务 OOM

**可能原因：** shape/dtype/layout/设备 key 高基数，未设置 LRU/TTL，版本 digest 未归并。**证据：** 按 key 维度统计 cardinality、每条 binary 大小和命中率。**修复：** shape bucket、缓存上限、发布时清理旧版本、对低频 shape 使用 eager；任何缓存清理都不能删除仍在使用的映射。

### 症状 G：CPU 上 toy 显示融合收益，GPU 上却没有

**原因：** toy 只模拟线性 bytes/FLOPs，没有真实 cache、warp、occupancy、库 kernel 和 PCIe。**正确解释：** toy 验证协议方向和边界，不验证硬件性能。必须在目标 GPU 用固定输入、同步和 profiler 复测。

## 34.12 失败复盘模板

每次 kernel/编译事故至少记录：

1. **触发条件：** model commit、PyTorch/Triton/CUDA/XLA/driver 版本、shape、dtype、并发、是否 warm；
2. **观察：** graph break、guard miss、cache key、kernel 名、p50/p99、显存、NaN/错误码；
3. **最小复现：** 输入生成器、seed、layout/stride、期望输出和 reference；
4. **因果链：** 哪个 pass 或配置改变了语义/资源，为什么监控没有提前报警；
5. **缓解：** feature flag、eager/library fallback、缓存回滚、限制 shape 或关闭 autotune；
6. **长期修复：** 新 regression test、性能门禁、版本 pin、告警和 runbook。

“把 `torch.compile` 关掉”是缓解，不是根因。事故报告应保留失败 kernel 的源码/IR digest 和 guard，确保未来升级能重放。

## 34.13 理解检查（含答案）

1. **为什么 graph break 会让一个看似简单的 pointwise 链变慢？** 参考答案：链被切成多个子图，子图间回到 Python/eager，产生额外 launch、同步、临时张量和可能的 layout copy；应通过 break 日志和 kernel trace 证实。
2. **何时动态 shape 优于静态 specialization？** 参考答案：shape 高度离散且编译缓存/发布预算有限时，多态图减少编译条目；若 shape 集合小且热点明显，静态特化通常能获得更好的布局和调度。必须用真实 shape 分布测量。
3. **融合为什么可能降低 occupancy？** 参考答案：融合把更多中间值同时保留在寄存器或 shared memory，单线程资源上升，SM 可驻留 block 减少，甚至寄存器溢出到 local memory；需看 register/thread、shared bytes/block 和 achieved occupancy。
4. **为什么 `ETag`/kernel cache key 不能单独当内容身份？** 参考答案：ETag 或缓存摘要的生成语义可能受 multipart、加密、编译器版本影响；内容身份应绑定真实 payload/源码 hash、版本、shape、dtype、设备能力及配置。
5. **autotune 的“最快一次”为什么不是可靠结论？** 参考答案：计时含噪声、warm-up、频率变化和编译成本；应固定环境、多次采样，用 median/p99 和总 wall-clock 目标，并保存候选证据。
6. **CPU-only toy 通过了，能否宣称 Triton kernel 正确且更快？** 参考答案：不能。toy 只验证抽象图语义、融合规则、缓存协议和误差计算；真实正确性需目标设备的 reference/differential/边界测试，真实性能需 profiler 和同步后的 warm/cold 测量。

## 34.14 练习

### 练习 1：回忆
列出一次 `torch.compile` 调用从 Python frame 到设备 kernel 的至少六个可观察证据，并指出每个证据属于语义、编译、性能还是运维层。

### 练习 2：推导
给定三层 pointwise 链，每层读 1 个输入、写 1 个输出，元素数 `N=16M`，设备有效带宽 `M=800 GB/s`，launch 20 µs。分别估算分离与完全融合的内存/launch下界；再加入 30 µs 的融合寄存器开销，判断是否仍有收益。说明忽略了什么。

### 练习 3：实现
在 toy lab 中增加 `sigmoid` 节点，并为连续 `mul -> sigmoid -> add` 设计融合合法性规则。添加 shape 尾部和 NaN 的 reference 测试，写出误差阈值选择理由。

### 练习 4：诊断
某服务 p50 从 4 ms 变为 3 ms，但 p99 从 12 ms 变为 80 ms，cache miss 从 2% 变为 25%。列出你会先查询的五个字段和两条临时回滚策略。

### 练习 5：设计
设计一个支持 4 种 batch bucket、fp16/bf16、两代 GPU 的编译缓存 schema。要求旧 binary 不会在新 driver 上误用，缓存有容量上限并可灰度回滚。

### 练习 6：审阅
审阅一段声称“融合后 GPU 利用率 95%，所以吞吐提升 2 倍”的报告。指出至少四个缺失的控制变量或反例，并给出一个更可信的实验矩阵。

## 34.15 来源与版本化证据

以下链接是本章写作时可复核的官方文档、论文或成熟源码。链接本身不替代目标版本验收；实验记录应保存访问日期、commit/release、设备和配置。

- PyTorch `torch.compile` 用户文档：https://pytorch.org/docs/stable/torch.compiler.html
- PyTorch Dynamo 概览：https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html
- TorchDynamo 源码：https://github.com/pytorch/pytorch/tree/main/torch/_dynamo
- AOTAutograd 文档：https://pytorch.org/functorch/stable/aot_autograd.html
- AOTAutograd 源码：https://github.com/pytorch/pytorch/tree/main/torch/_functorch
- PrimTorch 设计与源码：https://github.com/pytorch/pytorch/tree/main/torch/_prims
- TorchInductor 文档：https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html
- TorchInductor 源码：https://github.com/pytorch/pytorch/tree/main/torch/_inductor
- PyTorch 编译器故障排查：https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html
- PyTorch profiler：https://pytorch.org/docs/stable/profiler.html
- Triton 官方文档：https://triton-lang.org/main/index.html
- Triton tutorials：https://github.com/triton-lang/triton/tree/main/python/tutorials
- Triton 源码：https://github.com/triton-lang/triton
- NVIDIA CUDA C Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA CUDA Best Practices：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- CUDA Graphs 文档：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- NVIDIA Nsight Compute：https://docs.nvidia.com/nsight-compute/
- NVIDIA CUTLASS：https://github.com/NVIDIA/cutlass
- StableHLO specification：https://github.com/openxla/stablehlo
- OpenXLA 文档：https://openxla.org/xla
- XLA 源码：https://github.com/openxla/xla
- OpenXLA GPU 编译：https://openxla.org/xla/gpu
- MLIR 官方文档：https://mlir.llvm.org/docs/
- MLIR Linalg dialect：https://mlir.llvm.org/docs/Dialects/Linalg/
- LLVM Project：https://github.com/llvm/llvm-project
- IREE 编译器：https://iree.dev/
- IREE 源码：https://github.com/iree-org/iree
- TVM 编译器：https://tvm.apache.org/docs/
- TVM 源码：https://github.com/apache/tvm
- FlashAttention 论文：https://arxiv.org/abs/2205.14135
- FlashAttention-2 论文：https://arxiv.org/abs/2307.08691
- Ansor 自动调优论文：https://arxiv.org/abs/2006.06762
- Halide 论文与文档：https://halide-lang.org/
- MLIR：论文《MLIR: Scaling Compiler Infrastructure for Domain Specific Computation》https://arxiv.org/abs/2002.11054
- XLA 论文：《Accelerating Large-Scale Deep Learning by Offloading Computation to a TPU》https://arxiv.org/abs/1704.04760
- PyTorch Inductor 设计讨论：https://dev-discuss.pytorch.org/
- 本章 CPU-only 实验：`labs/ch34_compiler_kernel_lab.py`
- 本章测试合同：`tests/test_ch34_compiler_kernel_lab.py`

## 34.16 小结与下一依赖

编译器、kernel 和算子工程的核心不是“写一个更短的 kernel”，而是维护一条有守卫、有证据、有回退的语义链：捕获什么、为何能融合、如何 lowering、哪种资源成为瓶颈、缓存何时有效、误差和异常如何验证。PyTorch 2.x 让 eager 程序可以渐进式进入 Dynamo/AOTAutograd/Inductor；XLA/StableHLO/MLIR 提供另一种可组合的 IR 和设备路线；Triton/CUDA 把 tile、warp、内存和同步暴露给 kernel 工程师。融合和 autotune 只有在真实 shape 分布、编译摊销、p99、内存和正确性都纳入预算时才有意义。

下一步可把本章的 kernel 证据接到第20章的 trace/metrics，把编译缓存和回滚接到第21章可靠性，把实际 GPU profiler 读法接到第5章 CUDA 与第19章加速器栈。任何“更快”结论都应携带版本、硬件、输入、热身、同步、误差和失败边界；否则它只是一次不可复现的演示。
