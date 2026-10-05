---
id: ch08-distributed-collectives
title: 分布式通信：AllReduce、AllGather、ReduceScatter 与拓扑
slug: /chapters/08-distributed-collectives
description: 从 collective 语义、代价模型和环树算法出发，解释 NCCL/MPI、点对点、重叠、故障隔离与可运行 CPU 多进程实验
sidebar_position: 8
level: systems
prerequisites:
  - ch02-linux
  - ch04-performance-math
  - ch06-pytorch-execution
learning_objectives:
  - 能准确区分 AllReduce、AllGather、ReduceScatter、Reduce、Gather、Scatter 与 Broadcast 的输入输出契约
  - 能用 alpha-beta-gamma 模型估算消息延迟、带宽和归约计算成本
  - 能手算 ring、tree、recursive doubling 等算法的轮次、流量和瓶颈
  - 能解释节点内 NVLink/PCIe、节点间 NIC/InfiniBand/RoCE 拓扑对路径的影响
  - 能选择并诊断 NCCL、Gloo、MPI 等通信后端，理解进程组和 rank 约束
  - 能设计通信与计算重叠，区分异步句柄、stream 依赖和真正的端到端并行
  - 能通过 CPU 多进程实验验证 collective 语义、顺序要求、超时与故障隔离
  - 能在超时、数据不一致、性能退化和进程崩溃时建立最小复现与安全回滚路径
estimated_hours: 18
hardware: CPU-only baseline; CUDA/NCCL/MPI optional
risk_level: L2
last_verified: 2026-10-05
---

# 第8章　分布式通信：AllReduce、AllGather、ReduceScatter 与拓扑

> 当模型从一张卡扩展到多张卡，计算图中的“加法”会变成跨进程、跨设备、跨交换机的协议。分布式通信并不是一个隐藏的黑盒：每个 collective 都有明确的输入输出契约、调用顺序和故障语义。本章先建立 collective 的语义，再用 α-β-γ 代价模型分析 ring、tree 与分层算法，最后把这些概念落到 NCCL、MPI、PyTorch ProcessGroup 和 CPU 多进程实验。目标不是背某个库的环境变量，而是学会回答三件事：数据经过哪些链路、等待发生在哪里、某个 rank 失败后系统会怎样。

## 8.1 集体通信的共同契约

### 8.1.1 rank、world size 与 communicator

分布式程序通常运行多个进程（process）。每个进程在一个通信域（communicator 或 process group）里有唯一 `rank`，通信域总进程数是 `world_size`。同一 collective 的参与者必须使用同一个通信域，并按相同的调用顺序进入操作。

```text
communicator C = {rank 0, rank 1, ..., rank P-1}
P = world_size
```

`rank` 只是通信域内的逻辑编号，不等于物理机器编号、GPU 编号或网络地址。一个节点可以运行多个 rank，一个 rank 也可以绑定一个或多个设备，具体映射由启动器和应用决定。调试时同时打印 `hostname、pid、rank、local_rank、device`，否则“rank 2 挂了”很可能只是把机器编号误当成了 rank。

[事实] collective 是“所有参与者共同完成的一次操作”。调用缺一个 rank，其他 rank 往往会阻塞、超时或收到连接错误；它不会像普通函数一样自动跳过缺席者。

[机制] communicator 保存成员列表、传输通道、算法选择和错误状态。NCCL communicator、MPI communicator 与 PyTorch `ProcessGroup` 不完全同义，但都需要定义参与者集合与顺序。

[设计判断] 把通信域当成显式依赖：先初始化，再创建子组，最后在相同顺序销毁。不要让某个 rank 根据本地随机条件偷偷跳过 collective。

### 8.1.2 数据契约：count、dtype、shape 和 in-place

一个 collective 至少有四个契约字段：参与 rank 集合、元素数量（或 shape）、数据类型和操作（如 sum、max、product）。对 NCCL，构成一次完整 collective 的各 rank 通常必须以一致的 count 和 datatype 调用；MPI 则通过 datatype、count 和缓冲区描述数据布局。PyTorch distributed 通常要求参与者传入兼容的 Tensor shape/dtype，具体容忍度取决于后端。

“兼容”不代表每个 rank 的本地值相同。例如 AllGather 允许每个 rank 的 send buffer 内容不同；AllReduce 要求逻辑长度一致，但输入值可以不同。某些变长操作（如 `all_gather_object` 或 `all_gather_into_tensor` 的变体）有额外的长度交换或 padding 规则，不能把固定长度契约套过去。

集合操作常支持 in-place 形式：

```python
dist.all_reduce(x, op=dist.ReduceOp.SUM)
```

调用完成后 `x` 被写成归约结果。in-place 只描述缓冲区复用，不改变 collective 的同步要求。若同一 storage 还有其他 stream 或线程在读，必须建立正确的事件或 stream 依赖；不能因为“函数是异步的”就假设写入已经安全。

### 8.1.3 顺序契约与匹配

假设三个 rank 的代码分别执行：

```text
rank 0: all_reduce(A); all_gather(B)
rank 1: all_reduce(A); all_gather(B)
rank 2: all_gather(B); all_reduce(A)
```

即使每个调用的参数单独看都合法，整体也可能永久等待或触发未定义错误。大多数实现按 communicator 上的调用序列匹配操作；不同顺序会让某个 rank 在等待 `all_reduce` 时，另一个 rank 已经在等待 `all_gather`。

[故障模式] 条件分支、异常路径、数据迭代器长度不一致和提前 `break` 是最常见的顺序破坏源。日志里“最后一条打印”只说明 rank 走到了某行，不说明其他 rank 的调用已经匹配。

[诊断动作] 给每个 collective 分配单调递增的序号和名字，打印 `seq、op、tensor.numel、dtype、shape、rank、时间戳`。出现超时后比较各 rank 最后一个相同序号，通常比盲目提高 timeout 有用。

## 8.2 语义图：六类基本 collective

为了统一思考，用 (P) 表示 rank 数，rank (r) 的输入为 (x_r)。下面的“数据量”按元素数描述，实际字节数还要乘 dtype 宽度。

### 8.2.1 Reduce：所有输入归约到 root

`Reduce(op)` 把 (x_0, x_1, ..., x_{P-1}) 按逐元素操作 (op) 组合，只有 root rank 获得结果：

\[
 y_{root}[i] = op(x_0[i], x_1[i], ..., x_{P-1}[i]).
\]

非 root 的接收缓冲区是否被修改取决于 API；不要读取它。Reduce 适合把局部统计量汇总到一个控制节点，但 root 可能成为带宽和内存热点。

### 8.2.2 AllReduce：归约结果广播给所有 rank

`AllReduce(op)` 先做 Reduce，再把结果分发给所有 rank：

\[
 y_r[i] = op(x_0[i], ..., x_{P-1}[i]),\quad r=0..P-1.
\]

数据并行训练中的梯度同步是典型用例。对浮点 sum，归约顺序不同会造成舍入差异；“每张卡 bitwise 完全相等”不是跨实现的默认保证，验收应定义容差或任务指标。

### 8.2.3 Broadcast：root 的值复制到所有 rank

`Broadcast(root)` 以 root 的缓冲区为源，完成后每个 rank 都拥有同样的数据：

\[
 y_r = x_{root}.
\]

模型参数初始化、配置和随机种子同步常用 broadcast。源缓冲区的含义由 API 规定；某些后端允许 in-place，某些封装要求显式 send/recv buffer。

### 8.2.4 Gather 与 AllGather

`Gather(root)` 把每个 rank 的 (x_r) 收集到 root，root 得到有序序列：

\[
 y_{root} = [x_0, x_1, ..., x_{P-1}].
\]

`AllGather` 则让所有 rank 都获得这个序列：

\[
 y_r = [x_0, x_1, ..., x_{P-1}],\quad \forall r.
\]

AllGather 的输出大小是输入的 (P) 倍。序列顺序通常按 rank 排列，但必须以具体 API 文档为准；不要把网络到达顺序当成结果顺序。

### 8.2.5 Scatter：root 分发不同片段

`Scatter(root)` 由 root 持有 (P) 个逻辑片段，rank (r) 获得第 (r) 个：

\[
 y_r = x_{root,r}.
\]

Scatter 与 Gather 在数据流上互为方向，但不是“自动可逆”操作；中间是否发生 padding、排序或不同 datatype 会影响结果。

### 8.2.6 ReduceScatter：归约后分片

`ReduceScatter(op)` 先对所有 rank 的输入逐元素归约，再把结果切成 (P) 个不重叠片段，每个 rank 只得到一片：

\[
 z = op(x_0, ..., x_{P-1}),\quad y_r = z[r\cdot K:(r+1)K].
\]

输入总长度为 (P K)，每个 rank 输出 (K)（均匀版本）。ReduceScatter 与 AllReduce 的关系是：AllReduce 得到完整 (z)，ReduceScatter 只保留本 rank 的片段。优化器分片、ZeRO/FSDP 的梯度分片经常使用它以减少每个 rank 的显存。

### 8.2.7 All-to-All 和点对点的边界

`AllToAll` 让每个 rank 向每个其他 rank 发送一个片段，并收到来自所有 rank 的片段。它适合专家并行和稀疏路由，但通信量、连接数和缓冲区管理比 AllGather 更复杂。

点对点 `send/recv`、`isend/irecv` 只有指定的发送者和接收者。它能表达环、树和复杂流水线，也把匹配、标签、死锁责任交给应用。collective 提供全局契约和库级算法选择；点对点提供局部灵活性。不要用一长串点对点消息替代成熟 collective，除非有明确的拓扑或稀疏模式收益。

## 8.3 代价模型：α、β、γ 与内存占用

### 8.3.1 单链路 α-β 模型

经典通信模型把一次消息传输时间近似为：

\[
 T_{msg}(m) = \alpha + \beta m,
\]

其中 (m) 是字节数，\(\alpha\) 是启动延迟（包括队列、协议和软件开销），\(\beta\) 是每字节传输时间（带宽倒数）。小消息受 α 限制，大消息受 β 限制。

归约还要进行计算。若每个元素的操作成本是 \(\gamma\)，总计算时间近似为 \(\gamma n\)，于是可以写成：

\[
 T \approx \text{通信启动次数}\times\alpha + \text{传输字节数}\times\beta + \text{归约元素数}\times\gamma.
\]

这个模型忽略协议切换、拥塞、NUMA、PCIe 共享、交换机 credit 和 kernel launch，但足以比较算法量级。实测时要用 p50/p95 和端到端时间校准 α、β，而不是把链路标称带宽当作应用可用带宽。

### 8.3.2 ring AllReduce 的分解

ring AllReduce 把每个 rank 排成环。对总数据 (N) 字节，分成 (P) 个 chunk。算法分两阶段：

1. **Reduce-Scatter**：进行 (P-1) 轮，每轮向下一个 rank 发送一个 chunk，并把收到的 chunk 与本地对应 chunk 归约。结束后 rank (r) 拥有归约结果的第 (r) 块。
2. **AllGather**：再进行 (P-1) 轮，每轮转发已经归约的 chunk。结束后每个 rank 拥有全部块。

每个 rank 在整个操作中发送和接收约 \(2(P-1)N/P\) 字节，启动轮数约 \(2(P-1)\)。粗略时间：

\[
 T_{ring-allreduce} \approx 2(P-1)\alpha + 2\frac{P-1}{P}\beta N + \gamma\frac{P-1}{P}N.
\]

归约项具体系数取决于操作和实现（每个收到 chunk 只归约一次或多次）。ring 的优点是带宽利用率高、每个 rank 负载均衡；缺点是小消息需要 (P-1) 轮，尾部等待明显，环中任一慢链路都会影响完整环。

### 8.3.3 tree AllReduce 与递归加倍

tree 算法把 rank 组织为树。Reduce 阶段从叶子向根聚合，Broadcast 阶段从根向叶子传播，深度约 \(\lceil\log_2 P\rceil\)。粗略时间：

\[
 T_{tree} \approx 2\lceil\log_2 P\rceil\alpha + 2\beta N + \gamma N.
\]

树的轮数对小消息更友好，但根或上层节点可能成为带宽热点；若每个内部节点只能处理一条链路，通信总时间和并发度受树形限制。递归倍增（recursive doubling）在 P 为 2 的幂时每轮与距离 \(2^k\) 的 rank 交换，常用于小消息 AllReduce/AllGather。非幂次 P 需要折叠或不均匀处理，不能简单套公式。

现代库往往按消息大小、拓扑和协议在 ring、tree、分层或混合算法之间选择。不要看到日志里“使用 tree”就断定一定更快；算法选择还受 chunk、channel、NIC 数量和并发流影响。

### 8.3.4 分层（hierarchical）模型

多节点 GPU 集群通常分两级或三级：节点内 GPU 通过 NVLink/PCIe 聚合，节点间通过 NIC/交换机。可用如下步骤：

1. 每节点内部做 ReduceScatter 或 AllReduce；
2. 每节点选择一个或多个代表 rank，通过 NIC 做跨节点 collective；
3. 把跨节点结果广播或 AllGather 回节点内。

若节点内带宽 (B_{in}) 远大于节点间 (B_{out})，分层方案可以显著减少昂贵的跨节点流量。节点内多 NIC、多轨（rail）和 NUMA 绑定会改变代表 rank 的选择。分层不是无条件最优：节点内互联弱、节点间网络强，或数据量很小而额外的阶段启动占主导时，扁平算法可能更好。

### 8.3.5 内存与临时缓冲成本

时间模型之外，还要估算每个 rank 的额外内存：输入输出缓冲、chunk 双缓冲、通信库 workspace、CUDA graph 捕获池和 staging buffer。in-place AllReduce 可以复用输入，但 ring 仍可能需要临时 chunk；AllGather 的输出本身就是 (P) 倍大小；ReduceScatter 可把输出缩小到 (1/P)，却需要输入按分片布局排列。

记录 `allocated/reserved` 时，NCCL 直接申请的内存可能不出现在框架 allocator 统计中。CPU MPI 也会有 eager/rendezvous buffer、注册内存和 pinned buffer。性能调优必须同时记录内存峰值，否则把通信速度提升建立在 OOM 风险上是不完整的优化。

## 8.4 ring、tree 与拓扑映射

### 8.4.1 逻辑环不等于物理环

ring 算法需要逻辑上的“下一个 rank”，但物理链路可能是 NVLink、PCIe switch、主机内存、NIC、交换机再到远端节点。库会根据拓扑把逻辑边映射到物理路径，尽量避免把两个相邻 ring 边都放到同一条拥塞链路上。

在有多个 ring/channel 时，不同 ring 可能使用不同方向或不同 NIC。通道数增加可以提升大消息吞吐，也会增加 kernel、队列和调度开销。NCCL 的拓扑发现会读取 PCIe/NVLink/NVSwitch/NIC 关系；MPI/UCX 则可能结合网卡亲和、NUMA 和 fabric 路由选择路径。

### 8.4.2 节点内：NVLink、NVSwitch、PCIe

NVLink 提供 GPU 间高带宽点对点访问，NVSwitch 让多 GPU 更接近全互联，但实际路径仍受代际、链路数、交换芯片和 P2P 能力影响。PCIe 拓扑可能有多个 root complex；跨 root 访问经过 CPU/UPI，带宽和延迟与同一 switch 下的 GPU 不同。

使用 `nvidia-smi topo -m` 或库提供的拓扑工具先确认路径，再解释“某两张卡慢”。不要仅凭 GPU 编号猜测拓扑。容器里还可能屏蔽 P2P、设备节点或 ACS 设置，使理论上存在的链路在运行时不可用。

### 8.4.3 节点间：NIC、InfiniBand、RoCE 与 TCP

InfiniBand 通常提供低延迟、RDMA 和硬件流控；RoCE 依赖以太网配置（PFC、ECN、队列和交换机策略）；TCP 是兼容性最好的路径，但 CPU 协议栈和拷贝开销可能更高。NCCL、MPI/UCX 会依据可用接口和环境变量选择 transport。错误地把管理网卡选为通信网卡，会出现吞吐极低、拥塞或连接失败。

多 NIC 主机需要考虑 GPU-NIC 亲和：GPU 通过哪个 PCIe root 连接到哪张网卡，跨 NUMA 访问会增加 H2D/D2H 和 DMA 延迟。绑定 rank 与 CPU core、GPU、NIC 时，先记录物理拓扑，再做一个变量的 A/B 测试。

### 8.4.4 从拓扑到算法选择

可以把拓扑特征压缩成四个问题：

1. 节点内是否有远高于节点间的带宽？
2. 每个 GPU 是否有独立 NIC，还是共享 PCIe/root？
3. 是否存在链路不对称、机架 oversubscription 或多租户拥塞？
4. 消息是小而多，还是大而少？

小消息优先减少轮数（tree、recursive doubling）；大消息优先填满带宽（ring、多 ring、分层）。拓扑不对称时，库可能选择不同 ring 或树；手动固定 `NCCL_ALGO`、`NCCL_PROTO` 前必须做矩阵基准，并保留恢复默认的开关。

## 8.5 NCCL、MPI、Gloo 与 PyTorch ProcessGroup

### 8.5.1 NCCL 的职责和边界

NCCL 是面向 NVIDIA GPU 的 collective 库，提供 AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll、点对点和 group 操作等。它负责拓扑发现、通信 kernel、channel/protocol 选择和 CUDA stream 上的异步执行。NCCL 不是通用进程管理器；rank/world size、地址端口和进程启动通常由 torchrun、MPI、Slurm 或用户代码提供。

NCCL 的 collective 调用必须在所有 rank 上匹配。一次 API 返回通常表示工作已入队到指定 stream，是否完成要依据 `ncclGroupEnd`、事件或上层 Work 对象的语义判断。把返回即完成写入 host 可见内存会造成竞态。

常见环境变量（名称和语义会随版本扩展）包括算法/协议选择、调试日志、网卡过滤、P2P/SHM/IB 开关和超时。调试时只改变一个变量，记录完整环境；生产环境避免长期保留 `NCCL_DEBUG=TRACE`，否则日志量和时序变化本身会扰动性能。

### 8.5.2 MPI 的 communicator 和非阻塞 collective

MPI 规范定义 communicator、datatype、collective、点对点和拓扑等语义。`MPI_Allreduce`、`MPI_Allgather`、`MPI_Reduce_scatter` 的结果与本章语义一致；`MPI_Iallreduce` 等非阻塞版本返回 request，应用可以在通信期间做独立计算，之后用 `MPI_Wait/Test` 检查完成。

MPI 的性能取决于实现（Open MPI、MPICH、厂商 MPI）、传输层（UCX、OFI 等）、进程绑定和网络配置。MPI 不等于“只能 CPU”：CUDA-aware MPI 可直接处理 GPU buffer，但必须确认目标实现、CUDA 支持和 GPUDirect RDMA 配置。若未启用，MPI 可能隐式把数据拷贝到 host staging buffer，导致性能和内存行为完全不同。

### 8.5.3 Gloo 与 CPU 基线

Gloo 常用于 CPU collective 和开发环境。它适合作为 CPU 正确性基线，也可以帮助区分“模型语义错误”和“NCCL/GPU 环境错误”。Gloo 的传输与线程模型不同于 NCCL，不能把 CPU 时间直接当作 GPU 通信时间的替代。

### 8.5.4 PyTorch `torch.distributed`

PyTorch distributed 通过 `init_process_group` 创建默认 ProcessGroup，再用 `all_reduce`、`all_gather_into_tensor`、`reduce_scatter_tensor`、`broadcast`、`send/recv` 等 API 发起操作。CUDA 场景通常选择 NCCL，CPU 场景可选 Gloo 或 MPI（如果构建时支持）。

每个 collective 可返回 `Work` 句柄（`async_op=True`）。`work.is_completed()`、`work.wait()` 的完成语义只覆盖通信库定义的工作；对 CUDA Tensor，仍需理解 stream 之间的依赖和 host 访问同步。上层训练框架可能使用专用通信 stream、bucket、gradient hook 或异步错误处理，不能只看一处 API 就推断完整执行顺序。

[版本边界] 后端可用性、默认 device 选择、`all_gather_into_tensor` 的 shape 限制和 ProcessGroupNCCL 选项会随 PyTorch 版本变化。发布脚本应打印 `torch.__version__`、后端、CUDA runtime、NCCL version、启动器参数和实际 rank-device 映射。

## 8.6 点对点通信：灵活性与死锁

### 8.6.1 阻塞 send/recv 的匹配

最简单的点对点模式是：

```python
# sender
comm.send(buf, dest=1, tag=7)
# receiver
comm.recv(buf, source=0, tag=7)
```

阻塞是否返回，取决于实现何时可以复用发送缓冲区或填充接收缓冲区。发送方和接收方必须在 communicator、source/dest、tag、datatype 和长度上匹配。通配 `ANY_SOURCE`/`ANY_TAG` 提高灵活性，却会让调试和确定性变难。

### 8.6.2 非阻塞与缓冲区生命周期

`isend/irecv` 或 `MPI_Isend/Irecv` 返回 request。发起后，应用不能在 request 完成前修改发送缓冲区，也不能在未完成接收前读取接收缓冲区。常见错误是把临时 Tensor 交给异步 send 后立即释放，或循环中复用同一 buffer，导致数据竞态。

在 GPU 上，通信 stream 与计算 stream 之间需要事件依赖。框架的 `Work.wait()` 可能只让 host 等待提交完成，不代表另一个 CUDA stream 上的数据已经对当前计算可见。要使用库规定的 `wait_stream`、event 或 `record_stream` 机制，不要靠睡眠或全局同步猜测。

### 8.6.3 双向发送与死锁

两个 rank 都先阻塞发送可能形成循环等待，尤其当消息超过 eager buffer、协议转入 rendezvous 时。安全模式包括：一方先接收、使用非阻塞 `Irecv` 后 `Isend`、或调用经过验证的 `Sendrecv`。在环或 halo exchange 中，先发布所有接收，再发布发送，再等待全部 request，是易于审计的顺序。

## 8.7 通信与计算重叠

### 8.7.1 非阻塞不等于重叠

要实现重叠，必须同时满足：

1. 通信调用是异步的，返回后通信仍在进行；
2. 计算使用的数据与通信未完成部分无依赖；
3. 设备有独立执行资源或通信 kernel 不完全占用计算资源；
4. host 线程没有在关键路径立即 `wait`；
5. 调度器、stream 和内存生命周期没有隐式同步。

只把 `async_op=True` 加上，再马上调用 `wait()`，得到的只是异步 API 的同步使用。正确的模式是把参数或梯度分块：对已就绪的 bucket 发起 all-reduce，同时继续计算尚未归约的层。

### 8.7.2 DDP 梯度 bucket 的基本逻辑

数据并行反向传播按层产生梯度。框架把多个参数梯度放入 bucket；当 bucket 就绪，就在通信 stream 发起 AllReduce。后续层的反向计算可继续，形成计算与通信流水线。bucket 太小会增加 α 和 kernel launch；太大则要等很久才能启动，减少可重叠窗口。

重叠测量应看时间线：计算 kernel、NCCL kernel、事件和空洞是否交错。总 GPU 利用率高不代表有效重叠，两个 kernel 可能只是串行占满不同时间段。报告至少给出“无重叠端到端时间”“启用重叠端到端时间”“通信占用”“计算占用”和“未重叠尾部”。

### 8.7.3 FSDP/ZeRO 中的 AllGather 与 ReduceScatter

参数分片训练常在前向前 AllGather 参数，反向后 ReduceScatter 梯度。AllGather 将分片恢复为完整参数，ReduceScatter 将梯度归约并保留本 rank 的片段。二者可以在不同 CUDA stream 上排队，但同一 communicator 的内部进度可能仍然串行；必须以目标版本和配置为准。

重叠的安全边界是“数据依赖先于优化”。参数 AllGather 未完成，不能执行对应层；梯度 ReduceScatter 未完成，不能更新本地分片。把通信提前到没有依赖的窗口，可能减少尾延迟；把它提前到有依赖的窗口，则会出现错误或隐式同步。

### 8.7.4 CPU 通信与计算重叠

CPU MPI 的非阻塞 collective 可能由通信线程、进展线程或调用 `MPI_Test` 推进。若实现没有异步 progress，CPU 在做计算时通信可能没有前进，最终 `Wait` 仍要支付全部时间。检查 MPI 实现文档和进展线程配置，测量“计算期间字节数是否增长”，不要仅凭代码结构宣称重叠。

## 8.8 一致性、确定性与数值误差

### 8.8.1 浮点归约的非结合性

浮点加法不满足严格结合律：

\[
 (a+b)+c \neq a+(b+c).
\]

ring、tree、分层算法的归约顺序不同，结果最后几位可能不同。更换 rank 数、chunk 大小、NCCL 算法或 MPI 实现都可能改变顺序。需要 bitwise 复现时，应固定算法、拓扑、进程映射并承担性能代价；通常更实用的是定义 dtype 相关的绝对/相对误差和训练指标阈值。

### 8.8.2 异步错误与可见性

GPU kernel 或通信错误可能在后续 API 才报告。一个 rank 先检测到异常，其他 rank 可能仍在 collective 中等待。超时日志不一定能指出最初的错误点。开发环境可启用更严格的同步和 debug 日志，生产环境则需要结构化错误传播、超时和进程组重建策略。

### 8.8.3 屏障不是修复同步

`barrier()` 只保证参与者都到达屏障（具体是 host、device 还是通信 stream 的语义取决于后端），不会修复错误的 collective 顺序、错误的数据 shape 或未建立的 stream 依赖。滥加 barrier 会增加 α、掩盖真正的竞态并使性能基线失真。

## 8.9 故障模型与隔离策略

### 8.9.1 四类常见故障

1. **契约错误**：shape、dtype、count、操作或调用顺序不一致。
2. **传输故障**：网卡、交换机、P2P、RDMA、端口或容器权限异常。
3. **进程故障**：OOM、非法 kernel、Python 异常、节点重启或被调度器杀死。
4. **性能故障**：拓扑映射错误、拥塞、NUMA 远端内存、bucket 配置或算法选择不佳。

同一表象“all_reduce 卡住”可能来自四类中的任意一种。先检查契约和 rank 进度，再检查 transport，最后才调算法参数。

### 8.9.2 超时、abort 与重建

ProcessGroup 通常有初始化和 collective timeout。超时后，通信库可能把 communicator 标记为错误；继续在同一 communicator 上发起新 collective 可能得到连锁失败。安全做法是记录失败序号和 rank，停止依赖该 communicator 的工作，按框架支持的流程 abort 并重建，而不是在坏状态上无限重试。

重建需要重新确认：所有旧进程已退出或不再访问旧连接；新 rank/world size 一致；数据和 optimizer state 从一致 checkpoint 恢复。仅重启一个 rank 往往不能让其他 rank 自动恢复，因为 collective 的成员集合已改变。

### 8.9.3 故障隔离实验原则

故障实验应在临时作业和无敏感数据环境中进行：

- 先做单机 CPU baseline，确认模型和通信语义；
- 再注入一个可控延迟或提前退出；
- 设定短 timeout，收集每个 rank 的结构化日志；
- 验证其余 rank 是否退出、是否释放进程组和端口；
- 恢复后检查是否出现僵尸进程、残留共享内存或 NCCL socket。

不要在生产集群直接杀网卡、修改交换机 QoS 或关闭 P2P。故障注入的目标是验证恢复路径，不是让基础设施进入未知状态。

### 8.9.4 失败边界的逐层排查

collective 故障通常不是单点错误，而是错误在不同层被延迟放大。可以按“契约、进度、传输、资源、恢复”五层排查，每层都要有可观察证据。

第一层是契约。比较所有 rank 的序号、操作名、元素数量、数据类型、设备和参与组。不要只比较 Python 分支，因为隐式的张量重排、自动类型提升和空张量也会改变底层 count。一个 rank 使用长度为零的张量，另一个 rank 使用长度为非零的张量，某些后端可能立即报错，另一些后端可能等到通信 kernel 才失败。排查时把每次调用的元数据在进入 collective 前记录下来，避免错误发生后只能猜测。

第二层是进度。为每个 rank 维护最后进入和最后完成的序号，并区分“已入队”“设备完成”“主机收到完成通知”。这三个时刻可能相差很远。某 rank 打印了“开始 all_reduce”，不代表它已把缓冲区交给通信库；某 rank 打印了“返回”，也不代表另一个计算 stream 可以安全读取结果。把这些状态混在一条日志里，会把真正的等待位置隐藏起来。

第三层是传输。先做小消息、单节点、单 NIC 的测试，再逐步增加消息大小和节点数。若单节点成功而跨节点失败，优先检查网卡选择、RDMA 能力、端口策略和交换机拥塞；若只有某一对 GPU 失败，检查 P2P 可达性、PCIe ACS、IOMMU 和容器设备权限。不要一开始就关闭所有传输路径，因为这样会失去判断哪一层出错的对照组。

第四层是资源。collective 可能因为临时缓冲、注册内存、文件描述符或共享内存不足而表现为超时。AllGather 的输出会按参与者数量增长，ReduceScatter 的输入布局也可能让每个 rank 暂时保留完整梯度。对每个阶段记录主机内存、设备内存、通信库 workspace 和进程句柄数量；发现资源逼近上限时先缩小消息或 rank 数，而不是继续增加 timeout。

第五层是恢复。超时发生后，旧 communicator 是否还能使用取决于后端实现和错误状态；不能把“下一次调用偶尔成功”当作安全证据。恢复流程应明确谁负责停止剩余 rank、谁回收端口和共享内存、谁选择 checkpoint，以及如何验证所有 rank 读取的是同一版本。若只重启失败的进程，其余 rank 可能仍持有旧连接和旧序号，新的 rank 即使使用相同编号，也不一定能加入原 collective。

还有三类容易被忽视的边界。其一，空 collective 或零长度张量是否允许，取决于后端和 API，测试矩阵要把空 batch、空专家路由和过滤后无样本列为独立案例。其二，异步错误可能跨越多个序号才显现，必须保存最近若干个 collective 的上下文，而不是只留最后一行日志。其三，进程正常退出并不代表设备工作已完成；退出前要按后端要求等待或销毁通信域，避免驱动仍在访问即将释放的缓冲区。

在性能边界上，慢 rank 与坏 rank 的症状相似。慢 rank 仍会按顺序完成，只是每个序号的完成时间逐渐落后；坏 rank 则通常在固定序号停止、断开连接或返回错误。把每个 rank 的完成时间画成序列，能区分数据加载抖动、CPU 绑核错误和通信链路故障。只有确认所有 rank 都在相同序号推进后，才值得调整 ring/tree、channel 或 bucket 大小。

## 8.10 可运行 CPU 多进程实验：语义、顺序、超时和点对点

下面脚本只依赖 Python 标准库 `multiprocessing`，不需要 MPI、GPU 或外部服务。它用 TCP socket 实现一个教学版 collective：每个 rank 连接 coordinator，发送带序号的消息；coordinator 检查顺序和长度，返回 AllReduce/AllGather/ReduceScatter 结果。实现不是高性能库，只用于观察契约和故障行为。

保存为 `ch08_cpu_collectives.py`：

```python
#!/usr/bin/env python3
import argparse, multiprocessing as mp, socket, struct, time, json, os, sys

HDR = struct.Struct("!I")

def send_msg(sock, obj):
    data = json.dumps(obj).encode()
    sock.sendall(HDR.pack(len(data)) + data)

def recv_msg(sock):
    raw = sock.recv(HDR.size)
    if len(raw) != HDR.size:
        raise RuntimeError("short header")
    (n,) = HDR.unpack(raw)
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("peer closed")
        buf.extend(chunk)
    return json.loads(buf)

def coordinator(host, port, world, timeout):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen(world)
    srv.settimeout(timeout)
    conns = []
    try:
        while len(conns) < world:
            s, _ = srv.accept(); s.settimeout(timeout); conns.append(s)
        ranks = []
        for s in conns:
            hello = recv_msg(s)
            ranks.append((hello["rank"], s))
        ranks.sort()
        if [r for r, _ in ranks] != list(range(world)):
            raise RuntimeError("ranks are not 0..world-1")
        for seq in range(3):
            msgs = []
            for rank, s in ranks:
                msg = recv_msg(s)
                if msg["seq"] != seq:
                    raise RuntimeError(f"sequence mismatch rank={rank} got={msg}")
                msgs.append(msg)
            op = msgs[0]["op"]
            if any(m["op"] != op for m in msgs):
                raise RuntimeError("operation mismatch")
            if op == "allreduce":
                lengths = {len(m["data"]) for m in msgs}
                if len(lengths) != 1: raise RuntimeError("length mismatch")
                out = [sum(m["data"][i] for m in msgs)
                       for i in range(len(msgs[0]["data"]))]
                result = [out for _ in range(world)]
            elif op == "allgather":
                result = [[m["data"] for m in msgs] for _ in range(world)]
            elif op == "reducescatter":
                n = len(msgs[0]["data"])
                if any(len(m["data"]) != n for m in msgs):
                    raise RuntimeError("length mismatch")
                reduced = [sum(m["data"][i] for m in msgs)
                           for i in range(n)]
                if n % world: raise RuntimeError("need equal chunks")
                k = n // world
                result = [reduced[r*k:(r+1)*k] for r in range(world)]
            else:
                raise RuntimeError(f"unknown op {op}")
            for (rank, s), value in zip(ranks, result):
                send_msg(s, {"seq": seq, "data": value})
    except Exception as e:
        for s in conns:
            try: send_msg(s, {"error": repr(e)})
            except Exception: pass
        print("coordinator error:", repr(e), file=sys.stderr)
    finally:
        for s in conns: s.close()
        srv.close()

def worker(rank, world, host, port, bad_order=False, die=False):
    time.sleep(0.05 * rank)  # make connection order visibly different
    s = socket.socket(); s.settimeout(5); s.connect((host, port))
    send_msg(s, {"rank": rank})
    plans = [
        ("allreduce", [rank + 1, 10 + rank]),
        ("allgather", [rank]),
        # 让总长度始终是 world 的整数倍，每个 rank 输出 2 个元素
        ("reducescatter", [rank + i for i in range(world * 2)]),
    ]
    if bad_order and rank == world - 1:
        plans[1], plans[2] = plans[2], plans[1]
    for seq, (op, data) in enumerate(plans):
        if die and rank == world - 1 and seq == 1:
            print("rank", rank, "exits before seq", seq, flush=True)
            os._exit(7)
        send_msg(s, {"seq": seq, "op": op, "data": data})
        reply = recv_msg(s)
        if "error" in reply: raise RuntimeError(reply["error"])
        print(json.dumps({"rank": rank, "seq": seq, "op": op,
                          "result": reply["data"]}), flush=True)
    s.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=3)
    ap.add_argument("--bad-order", action="store_true")
    ap.add_argument("--die", action="store_true")
    ap.add_argument("--port", type=int, default=29591)
    args = ap.parse_args()
    host = "127.0.0.1"
    c = mp.Process(target=coordinator, args=(host, args.port, args.world, 5))
    c.start()
    ps = [mp.Process(target=worker, args=(r, args.world, host, args.port,
                                          args.bad_order, args.die))
          for r in range(args.world)]
    for p in ps: p.start()
    for p in ps: p.join(8)
    for p in ps:
        if p.is_alive(): p.terminate(); p.join()
    c.join(2)
    print("exitcodes", [p.exitcode for p in ps], "coordinator", c.exitcode)

if __name__ == "__main__":
    main()
```

运行正常路径：

```bash
python3 ch08_cpu_collectives.py --world 3
```

你会看到：

- AllReduce 每个 rank 都收到按 rank 求和的两元素向量；
- AllGather 每个 rank 都收到按 rank 排列的 `[[0],[1],...]`；
- ReduceScatter 的总和按 rank 等分，每个 rank 收到 2 个元素（输入长度由 `world*2` 自动决定）。

推荐依次运行：

```bash
python3 ch08_cpu_collectives.py --world 2
python3 ch08_cpu_collectives.py --world 3
python3 ch08_cpu_collectives.py --world 3 --bad-order
python3 ch08_cpu_collectives.py --world 3 --die
```

`--bad-order` 让最后一个 rank 交换 seq=1/2 的操作，coordinator 会报告 operation/sequence mismatch；`--die` 让一个 rank 在第二个 collective 前退出，coordinator 会看到 peer closed（`short header`），其他 rank 收到错误并退出。真实后端通常把同类情况报告为 timeout 或 connection reset。实验重点不是 socket 性能，而是观察三条原则：

1. collective 的顺序比单次 API 参数更重要；
2. 缺席 rank 必须通过 timeout/abort 暴露，而不是永久等待；
3. 数据长度检查应在真正传输前发生，避免半完成状态难以恢复。

### 8.10.1 用 PyTorch Gloo 复刻同一实验（可选）

如果环境已安装 PyTorch，可以用 Gloo 运行真实 ProcessGroup。保存为 `ch08_torch_gloo.py`：

```python
import os, torch, torch.distributed as dist

def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    x = torch.tensor([rank + 1., 10 + rank])
    dist.all_reduce(x)
    print(rank, "all_reduce", x.tolist(), flush=True)
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, torch.tensor([float(rank), float(rank)]))
    print(rank, "all_gather", [p.tolist() for p in parts], flush=True)
    inp = torch.arange(world * 2, dtype=torch.float32) + rank
    out = torch.empty(2)
    dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
    print(rank, "reduce_scatter", out.tolist(), flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__": main()
```

用 `torchrun --standalone --nproc-per-node=2 ch08_torch_gloo.py` 启动。不要在没有 `torch.distributed` 支持的构建上直接复制命令；先检查 `torch.__version__` 和 `dist.is_available()`。实验输出的打印顺序不保证按 rank 排列，语义由 Tensor 内容而不是 stdout 顺序决定。

## 8.11 观测与基准方法

### 8.11.1 端到端、通信内核和有效带宽

基准至少分三层：

1. **纯 collective**：预分配 Tensor，预热，测单次 AllReduce/AllGather 的设备时间；
2. **通信+计算**：模拟真实 bucket 或层级流水，测 overlap；
3. **端到端 step**：含数据准备、forward、backward、optimizer、checkpoint 和日志。

有效带宽可按算法定义计算。对 ring AllReduce，每 rank 发送/接收约 \(2(P-1)N/P\) 字节；用这个“算法字节数”除以通信时间得到 bus bandwidth。不要用 `N / time` 直接与链路标称带宽比较，因为 AllReduce 的逻辑结果复制和双向流量会改变字节计数。

报告固定 shape、dtype、rank 数、节点数、GPU-NIC 映射、预热次数、重复次数和同步边界。对非阻塞操作，测量必须在所有 rank 和正确的 stream/event 上确认完成后结束。

### 8.11.2 诊断日志

每个 rank 建议输出结构化 JSON 字段：`timestamp、rank、local_rank、host、pid、seq、op、numel、dtype、device、stream 标识、start、enqueue_end、complete、error`。日志中不要写入训练样本、访问令牌或完整环境变量（其中可能包含凭据）。

NCCL/MPI 的 debug 日志是辅助证据，不是唯一事实来源。把日志级别、版本和环境开关纳入实验报告。生产故障时先收集小窗口，再恢复默认日志，避免磁盘写满和时序扰动。

### 8.11.3 性能回归阈值

为每个消息规模和 rank 数设置基线区间，而不是一个单点阈值。例如 p50 通信时间回归超过 10%、p99 超过 20%、或 timeout 次数非零就触发调查。区分噪声来源：多租户网络、频率变化、CPU governor、GPU clock、JIT/compile 首次开销和缓存冷启动。

## 8.12 设计模式：从语义到实现的五步法

1. **写出数学契约**：输入来自哪个 rank，输出在哪些 rank，可否 in-place，归约操作是否结合/交换。
2. **画数据分片**：标出每个 rank 持有哪些 chunk，AllGather/ReduceScatter 的切片边界和输出布局。
3. **估算代价**：用 α、β、γ 写出轮数、字节数和归约量，明确小消息还是大消息主导。
4. **映射拓扑**：节点内、节点间、NIC/NUMA、P2P 和共享链路，决定分层、ring 或 tree 候选。
5. **验证与回滚**：CPU 小规模正确性、真实后端基准、故障注入和超时恢复，最后才把开关带到生产。

这五步可以应用于梯度同步、参数分片、专家路由、检索服务的分布式缓存和多节点统计，不局限于深度学习训练。

## 8.13 案例推演：从一次梯度同步到可验证的时间线

为了把前面的语义、代价和拓扑连接起来，设有 4 个 rank、每个 rank 一张 GPU，训练一个数据并行模型。每个 rank 在反向结束时得到 256 MiB 梯度 bucket。目标是让 optimizer 在每个 rank 上使用相同的梯度，同时尽量隐藏通信时间。

### 8.13.1 先写出不带重叠的基线

最直接的实现是：反向全部完成后，对 256 MiB 梯度执行一次 AllReduce，等待通信完成，再除以 world size 并更新参数。若通信库使用 ring，每个 rank 的算法发送量约为

\[
2(P-1)N/P = 2\times3/4\times256\text{ MiB}=384\text{ MiB}.
\]

若端到端测得 AllReduce 用时 6 ms，算法带宽约为 384 MiB / 6 ms ≈ 64 GiB/s。这个数字不能直接与单条 NVLink 或 NIC 的标称带宽比较，因为它包含了协议、chunk、kernel launch、拓扑和等待最慢 rank 的时间。基线还要记录反向计算耗时、optimizer 耗时、峰值显存和每个 rank 的 p95。

基线的时间线可以写成：

```text
backward (所有 bucket)
  -> all_reduce(bucket 0..K)
  -> wait
  -> optimizer.step
```

如果反向耗时 40 ms、通信 6 ms，理想情况下 step 至少约 46 ms（忽略其他成本）。此时通信没有隐藏，且所有梯度都要在通信开始前占据显存。

### 8.13.2 把 bucket 变成流水线

将 256 MiB 切成 8 个 32 MiB bucket。反向产生 bucket 7 时，立即在通信 stream 发起它的 AllReduce；主计算 stream 继续计算更早或尚未完成的层。理想时间线如下：

```text
计算:  b7  b6  b5  b4  b3  b2  b1  b0  optimizer
通信:      AR7 AR6 AR5 AR4 AR3 AR2 AR1 AR0  tail
```

真实系统不会严格按字符图排列：bucket 的就绪顺序、kernel 资源冲突和通信 stream 的进度都会改变间隔。评估重叠要测三种时间：

- **纯计算时间**：临时关闭通信或用单 rank，得到反向和 optimizer 的基线；
- **纯通信时间**：固定相同 bucket，在空闲 stream 上测 AllReduce；
- **流水端到端时间**：保留真实依赖，等待最后一个 bucket 完成后再更新。

如果通信和计算完全重叠，step 时间接近 `max(计算,通信)+不可隐藏尾部`，而不是二者简单相加。不可隐藏尾部包括最后一个 bucket 的通信、stream 事件和 optimizer 依赖。报告中不要用“通信占用率高”代替这个结论。

### 8.13.3 依赖图和缓冲区生命周期

每个 bucket 至少有四个事件：梯度写入完成、通信读入开始、通信写回完成、optimizer 读取开始。可用如下依赖表示：

```text
backward writes bucket_i
  -> event_compute_i
  -> comm_stream waits event_compute_i
  -> all_reduce_i (in-place or out-of-place)
  -> event_comm_i
  -> optimizer_stream waits event_comm_i
  -> optimizer reads bucket_i
```

如果使用 in-place AllReduce，通信开始后反向不能再写同一 bucket；如果使用 out-of-place，必须确保输出 Tensor 的生命周期覆盖 optimizer。框架通常通过 autograd hook、bucket 状态和 `record_stream` 管理这些关系，但自定义通信代码不能假设临时 Tensor 会自动存活。

一个常见错误是把 bucket 放入 Python 列表后立刻删除引用，以为通信库已经复制了数据。异步通信可能仍在读取原 storage；正确做法是等待 Work 完成，或使用后端规定的 buffer 生命周期 API。另一个错误是把同一个 bucket 交给两个 communicator，未建立跨 communicator 的顺序；两个 collective 可能在不同 stream 上竞争同一内存。

### 8.13.4 拓扑变化如何改变结论

假设机器 A 有 NVSwitch 和 8 张 GPU，机器 B 有 4 张 GPU、每张卡通过 PCIe 连接一张 NIC。相同 256 MiB bucket 在两台机器上的最佳算法可能不同：

- A 的节点内带宽高且路径近似对称，多 ring 可以填满 NVSwitch；tree 可能只在小消息上占优；
- B 的跨节点带宽和 GPU-NIC 亲和更关键，分层 AllReduce 可能先在节点内聚合，再用 2 张代表 GPU 跨节点通信；
- 如果 B 的某张 NIC 位于远端 NUMA，绑定错误会让某一条 ring 边变慢，其他 rank 在每轮都等待它；
- 当消息降到几十 KiB，算法启动延迟和进程调度会主导，减少轮数比追求峰值带宽更重要。

因此不能把 A 上测得的“ring 比 tree 快 20%”写成通用规则。至少用消息大小、rank 数、拓扑和算法四维矩阵重新测量，并记录环境变量是否覆盖了库的自动选择。

### 8.13.5 故障时的时间线

继续上面的流水线，若 rank 2 在 bucket 4 的 kernel 中触发非法访问，可能发生：

1. rank 2 的 CUDA 错误直到下一次同步才被发现；
2. rank 0、1、3 已经进入 bucket 4 的 AllReduce，等待 rank 2；
3. ProcessGroup timeout 触发，日志显示“collective seq=4 未完成”；
4. 其他 rank 需要停止后续 bucket 和 optimizer，避免在不完整梯度上更新；
5. 调度器或上层恢复逻辑 abort communicator，清理进程和临时端口；
6. 从一致 checkpoint 重启，不能只让 rank 2 单独回来。

故障演练应验证每一步都有可观察证据：seq 号、最后完成 bucket、错误码、abort 时间、进程退出码和 checkpoint 版本。若只验证“程序最终退出”，却没有检查数据是否来自同一 optimizer step，恢复逻辑仍可能静默损坏训练。

### 8.13.6 将案例迁移到 ReduceScatter

若模型使用参数分片，每个 rank 不需要完整梯度。把 AllReduce 改为 ReduceScatter：256 MiB 梯度按 4 份切分，每个 rank 最终只保留 64 MiB。ring ReduceScatter 只执行前 (P-1) 轮，算法发送量约为 \((P-1)N/P=192\) MiB，比完整 AllReduce 少一个阶段。随后 optimizer 只更新本地参数分片；下一次前向需要完整参数时，再按依赖 AllGather。

这种方案减少了持久显存，但把通信依赖分散到前向和反向两个窗口。若 AllGather 没有及时完成，前向会在第一层等待；若 ReduceScatter 尚未完成，optimizer 不能读取梯度分片。衡量收益时要把两次通信、参数分片内存、重叠尾部和 checkpoint 格式一起计算，不能只比较单次 ReduceScatter 的时间。

### 8.13.7 案例结论

一次“梯度同步变慢”至少可能对应五个不同问题：bucket 太大导致启动晚、bucket 太小导致 α 占主导、拓扑映射把流量压到慢链路、通信 stream 依赖错误造成隐式同步、或某个 rank 的计算/数据加载落后造成 straggler。只有把语义序号、时间线、拓扑和 rank 进度放在同一份报告里，才知道应该改 bucket、改绑定、改算法还是修复数据迭代器。

在实践中还应记录“没有发生什么”：没有等待的阶段、没有失败的 rank、没有变化的网络配置都属于证据。将正常路径和故障路径放在同一张时序图中，可以发现某些所谓优化只是把等待从通信 API 移到了隐式 stream 同步。报告结尾写出仍未验证的假设，例如交换机拥塞是否可重复、NIC 是否共享 PCIe root、非阻塞 MPI 是否启用 progress 线程，以及 checkpoint 是否包含完整 optimizer state。这样下一次迁移到新 GPU、新驱动或新调度器时，测试计划可以直接从假设列表生成，而不是重新猜测问题。

## 8.14 六个理解检查（含答案）

### 检查 1：AllReduce 与 ReduceScatter 的主要差异是什么？

**答案**：AllReduce 归约后让每个 rank 获得完整结果；ReduceScatter 归约后只把不重叠的结果分片给各 rank。若结果总长度为 (N)，均匀 ReduceScatter 每个 rank 只保存 (N/P)，因此可节省输出显存，但输入布局和后续计算必须按分片契约设计。

### 检查 2：为什么 ring AllReduce 的每个 rank 发送量接近 (2(P-1)N/P) 而不是 (N)？

**答案**：ReduceScatter 和 AllGather 各有 (P-1) 轮，每轮传输一个 (N/P) chunk；两阶段合计 (2(P-1)N/P)。当 P 较大时趋近 (2N)，但每个 rank 的负载均衡，能较好利用带宽。

### 检查 3：两个 rank 的 collective 顺序不一致，为什么不一定立即报错？

**答案**：实现通常按通信域上的调用序列匹配；如果某个调用尚未触发严格参数检查，rank 可能先阻塞在不同操作上，表现为无输出或 timeout。只有当超时、连接关闭或库进行一致性检查时，错误才会显现。给 collective 加序号和名字能缩短定位时间。

### 检查 4：`async_op=True` 为什么不保证通信和计算重叠？

**答案**：它只说明 API 返回异步工作句柄。若随后立即 `wait()`、计算依赖未完成数据、通信和计算争抢同一执行资源，或后端没有独立 progress，实际仍可能串行。必须用时间线和事件确认两个阶段确实交错。

### 检查 5：浮点 AllReduce 为什么会因 ring/tree 选择不同而产生微小差异？

**答案**：浮点加法不满足严格结合律，ring 和 tree 的归约顺序不同，舍入误差也不同。固定算法可改善复现，但跨 rank、拓扑和版本的 bitwise 一致不能默认保证。应定义 dtype 相关容差和任务级验收。

### 检查 6：all_reduce 超时后为什么不能只重试一次调用？

**答案**：超时可能已让 communicator 进入错误状态，部分 rank 仍持有未完成请求或已退出。继续在同一通信域重试会造成连锁阻塞和数据不一致。应停止依赖它的工作，记录失败序号，按后端支持的流程 abort/rebuild，并从一致 checkpoint 恢复。

## 8.15 练习

1. **语义表**：为 Reduce、AllReduce、Gather、AllGather、Scatter、ReduceScatter、AllToAll 各画一张 rank×buffer 表，标出输入、输出和数据量。
2. **代价比较**：设 P=8、N=64 MiB、α=2 μs、β=1/(100 GB/s)，分别估算 ring 和 tree AllReduce 的通信项。说明何时应加入 γ。
3. **非 2 的幂**：手算 P=6 的 recursive doubling 或折叠策略，解释为什么不能直接使用三轮 `xor`。
4. **拓扑实验**：在有多张 GPU 的机器上运行 `nvidia-smi topo -m`，把 GPU、PCIe root、NIC 和 NUMA 画成图，比较相邻和跨 root 的 P2P 带宽。
5. **bucket 调优**：用一个多层 MLP 模拟梯度 bucket，比较 1 MiB、8 MiB、64 MiB bucket 的启动次数、重叠比例和 p99。
6. **点对点死锁**：写两个 rank 都先阻塞 send 的程序，分别用小消息和大消息运行；再改成 `Irecv`→`Isend`→`Waitall`，解释行为差异。
7. **CPU 故障注入**：扩展本章 socket 脚本，加入 rank 延迟、错误 dtype 和错误长度，验证 coordinator 如何在传输前拒绝。
8. **Gloo/MPI 对照**：在同一 CPU 节点用 Gloo 和 MPI（若可用）跑 AllReduce，记录 p50/p95、线程数和绑定策略，写出不可迁移的假设。
9. **NCCL 算法矩阵**：在隔离 GPU 作业中比较默认、ring、tree（仅在目标版本支持的环境变量下），报告消息大小、拓扑和恢复默认方法。
10. **故障恢复设计**：为训练服务画出 timeout、abort、checkpoint、重建 process group 和回滚的状态机，标出哪些步骤必须由调度器执行。

## 8.16 安全边界与运维清单

### 8.15.1 不把 collective 当作访问控制

AllReduce、Broadcast 等只提供数据移动和归约，不提供身份认证、加密或租户隔离。通信组中的任何 rank 都可以读取其收到的数据。敏感数据跨节点前应使用受控网络、作业隔离和最小权限；不要把调试 socket 暴露到公共接口。

### 8.15.2 环境变量和启动参数

NCCL/MPI 环境变量会改变网卡、P2P、共享内存、算法和日志。只在隔离作业中测试，记录修改前后的完整差异，结束后恢复默认。不要把包含凭据的启动环境复制到日志或 issue；对外分享日志前脱敏主机名、路径、令牌和样本标识。

### 8.15.3 资源与拒绝服务

错误的 world size、端口冲突、未清理进程和超大 AllGather 都可能耗尽 CPU、GPU、共享内存或网络。为作业设置内存、文件描述符、进程数和超时上限；对用户输入的 shape 做白名单或预算检查，避免恶意请求触发 (P) 倍内存分配。

### 8.15.4 自定义通信 kernel 与第三方库

不要从不受信任来源加载预编译 NCCL 插件、MPI hook 或 CUDA extension。通信 kernel 具有设备级权限，越界写可能损坏其他 Tensor 或导致 GPU reset。锁定库版本、构建哈希和容器镜像；先在 CPU 小规模和单节点 GPU 上做正确性，再扩大到生产拓扑。

### 8.15.5 故障时的最小披露

故障报告包含足够诊断信息即可：rank/seq/op、shape/dtype、版本、拓扑摘要、错误码和时间线。不要上传训练样本、用户内容、私钥、完整环境变量或未脱敏网络配置。需要供应商支持时，先用可合成数据复现，再分享经过审查的日志。

## 8.17 版本边界与迁移注意

1. PyTorch `torch.distributed` 的 collective 名称和参数在 2.x 版本间持续增加；`all_gather_into_tensor`、`reduce_scatter_tensor`、DeviceMesh 和新 ProcessGroup 选项在旧版本可能不存在。先查目标版本文档。
2. NCCL 的算法、协议、channel 和环境变量会随版本、GPU 架构与拓扑改变。默认自动选择通常比长期硬编码更可迁移；手动固定只用于经过基准和回滚验证的部署。
3. NCCL 的异步错误处理、communicator abort、CUDA Graph 集成和多 communicator 排序要求依赖具体版本。遇到 hang 时，按目标版本 troubleshooting 文档收集日志，不要套用旧博客的变量名。
4. MPI 非阻塞 collective 的 progress、CUDA-aware 支持和线程安全级别由实现和构建选项决定。`MPI_THREAD_MULTIPLE`、UCX transport、GPUDirect RDMA 不能仅凭函数名推断已启用。
5. 硬件拓扑变化会改变最佳 ring/tree。更换 GPU 代际、NVSwitch、NIC、交换机 oversubscription 或 NUMA 绑定后，应重新做消息大小×rank 数×算法矩阵。
6. 浮点归约的确定性和误差阈值受 dtype、库 kernel、FMA、TF32、压缩/量化传输影响。升级驱动或通信库后重新运行数值验收。
7. ProcessGroup 的 `Work.wait()`、barrier 和 CUDA stream 可见性语义应以目标后端文档为准。不能把 CPU Gloo 的“返回即完成”直推到 CUDA NCCL。

## 8.18 来源地图

以下优先使用官方文档和标准，链接用于核对语义、API 和版本边界；阅读时选择与环境匹配的版本。

- [PyTorch Distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html)：ProcessGroup、NCCL/Gloo/MPI 后端、collective API、异步 Work 和环境变量。
- [PyTorch 分布式训练概览](https://docs.pytorch.org/docs/stable/accelerator/distributed.html)：后端选择、rank/device 模型和 collective 列表。
- [PyTorch Writing Distributed Applications 教程](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)：点对点、Gather/Scatter、AllReduce 示例和启动方式。
- [PyTorch FSDP fully_shard 文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)：参数 AllGather、梯度 ReduceScatter、stream 与 ProcessGroup 交互。
- [NVIDIA NCCL 用户指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)：collective、group 操作、拓扑发现、算法/协议和故障排查。
- [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2292/user-guide/docs/usage/collectives.html)：调用匹配、count/datatype 契约和 AllReduce/AllGather 语义。
- [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)：P2P、拓扑检查、GPU/NIC 与常见 hang 原因。
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)：算法、协议、网卡和调试变量；不要脱离目标版本使用。
- [MPI Forum 标准文档](https://www.mpi-forum.org/docs/)：MPI communicator、collective、datatype、非阻塞通信和进程拓扑的规范定义。
- [MPI-4.1 标准 PDF](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf)：`MPI_Allreduce`、`MPI_Allgather`、`MPI_Reduce_scatter`、`MPI_I*` 语义与线程模型。
- [Open MPI Collective Communication](https://docs.open-mpi.org/en/main/tuning-apps/collective-communication.html)：算法选择和调优边界，适用于 Open MPI 版本文档。
- [UCX 文档](https://openucx.readthedocs.io/)：RDMA、RoCE、传输层和进程间通信配置背景。
- [NVIDIA GPU 拓扑工具说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html)：`nvidia-smi topo` 输出和 P2P 能力检查。

[方法说明] α-β-γ 模型和 ring/tree 轮数是教学用近似，不是任何特定库的性能承诺。真实结果必须在目标硬件、驱动、库版本和进程映射上测量，并报告未验证假设。

## 8.19 章节完成标准

读者完成本章后，应能对一次分布式 collective 回答八个问题：

1. 参与者是谁，rank/world size 和 communicator 如何定义？
2. 输入输出 shape、dtype、count、归约操作和 in-place 规则是什么？
3. 各 rank 的调用顺序、序号和异常路径是否匹配？
4. 算法是 ring、tree、recursive doubling、分层还是库自动选择？
5. 每个 rank 的 α、β、γ 项、发送字节和临时内存是多少？
6. 逻辑边映射到哪些 NVLink/PCIe/NIC/交换机/NUMA 路径？
7. 异步句柄、通信 stream、计算 stream 和 buffer 生命周期在哪里建立依赖？
8. timeout、abort、重建和 checkpoint 恢复的安全边界是什么？

如果其中任何一项答不出来，先缩小到 CPU 两进程、固定 shape 和单个 collective，再逐步加入真实拓扑和重叠。不要用更多 barrier、无限 timeout 或盲目切换算法掩盖未验证的契约。

## 8.20 小结

Collective 的核心不是“把 Tensor 发到别的机器”，而是一个由参与集合、调用顺序、数据契约、归约语义和完成规则共同定义的协议。AllReduce 给所有 rank 完整归约结果，AllGather 复制分片，ReduceScatter 归约后分片；ring 以均衡带宽换取更多轮次，tree 以较少轮次换取上层链路压力，分层算法则利用节点内外拓扑差异。NCCL、MPI、Gloo 和 PyTorch ProcessGroup 提供不同的实现和异步边界，但都不能替应用修复错误顺序或错误 shape。

真正可迁移的工作流是：先写数学契约，再用 α-β-γ 估算，读取物理拓扑，建立 CPU 正确性基线，测量真实 stream/通信时间线，最后在隔离作业中做故障和回滚演练。把版本、拓扑、消息规模、数值容差和安全边界写进报告，下一次升级库或更换集群时重新验证，才能让“多卡更快”成为可解释、可恢复的工程结论。
