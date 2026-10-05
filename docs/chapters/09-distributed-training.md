---
id: ch09-distributed-training
title: 分布式训练：DP/TP/PP、ZeRO/FSDP 与容错
slug: /chapters/09-distributed-training
description: 用统一的并行度、内存和通信模型解释数据/张量/流水线/序列/专家并行，比较 ZeRO 与 FSDP，并在 CPU 上验证梯度、检查点一致性和故障恢复
sidebar_position: 9
level: systems
prerequisites:
  - ch04-performance-math
  - ch06-pytorch-execution
  - ch08-distributed-collectives
learning_objectives:
  - 能区分 DP、TP、PP、SP/CP 与 EP 的切分维度、通信契约和适用边界
  - 能从参数、梯度、优化器状态和激活四项估算每个 rank 的显存/内存
  - 能解释 ZeRO-1/2/3、FSDP full_shard、reshard-after-forward 与 all-gather/reduce-scatter 的关系
  - 能计算流水线 bubble、micro-batch 数和 1F1B 调度的吞吐影响
  - 能设计跨节点拓扑映射，避免把频繁通信放在慢链路上
  - 能建立一致性检查点、故障检测、重试与恢复后的数据游标协议
  - 能用 CPU 多进程实验复现梯度同步、分片优化器、激活重计算和两节点 toy 训练
  - 能在训练不收敛、hang、OOM、rank 丢失或检查点损坏时建立最小复现和安全回滚路径
estimated_hours: 24
hardware: CPU-only baseline; CUDA/NCCL optional
risk_level: L2
last_verified: 2026-10-05
---

# 第9章　分布式训练：DP/TP/PP、ZeRO/FSDP 与容错

> 分布式训练的难点不在于“把 batch 分给多张卡”。真实系统必须同时决定参数放在哪里、激活沿哪条链路移动、梯度何时归约、优化器状态由谁持有，以及某个进程死掉时从哪个一致状态继续。一个看似简单的 `loss.backward()`，可能触发张量并行的多次 all-reduce、流水线的 send/recv、专家路由的 all-to-all、数据并行的 reduce-scatter，以及分片参数的 all-gather。若不把这些动作画成显式协议，增加 GPU 数量往往只是把单机 bug 变成跨机器的 hang。
>
> 本章采用四个统一视角。第一，**切分维度**：数据、参数张量、层、序列 token、专家分别沿哪个轴分布。第二，**生命周期**：参数、梯度、优化器状态和激活何时驻留、何时释放或重计算。第三，**通信时序**：collective/点到点的参与者、顺序、消息量和是否能与计算重叠。第四，**故障边界**：一个 rank、一个节点、一条链路或一个检查点损坏时，系统如何发现、隔离和恢复。后续所有公式和实验都围绕这四个问题展开。

## 9.1 先定义目标：吞吐、时延、规模和可恢复性

### 9.1.1 全局 batch 与有效步长

设数据并行副本数为 (D)，每个副本每次处理 micro-batch 大小 (b)，梯度累积步数为 (K)，则一次 optimizer step 的全局样本数为

\[
B_{global}=D\times b\times K.
\]

若使用序列长度 (S)，每个样本每步的 token 数为 (S)，则 token 级有效 batch 是 (B_{tok}=B_{global}S)。比较不同并行配置时，要同时报告 samples/s 和 tokens/s；只报告样本数会掩盖序列长度变化造成的工作量差异。梯度累积并不免费：它减少 optimizer 更新频率，却让激活在多个 micro-batch 上重复产生，并可能让流水线需要更多 micro-batch 才能填满。

数据并行同步梯度时，通常将本地梯度求和再除以 (D)，或者先求平均再使用学习率。二者只要缩放一致即可。常见错误是每个 rank 已经做了 mean，通信后又除以 (D)，导致有效学习率缩小 (D) 倍。实验日志应记录 `loss_reduction=sum|mean`、`world_size`、`grad_accum_steps` 和最终梯度范数。

### 9.1.2 端到端吞吐的粗模型

把一次 step 分为计算时间 (T_c)、通信时间 (T_m)、输入/输出时间 (T_i) 和同步等待 (T_w)。如果通信完全无法重叠，(T_{step}=T_c+T_m+T_i+T_w)；若通信与计算可重叠，理想上是

\[
T_{step}\approx \max(T_c,T_m)+T_i+T_{residual},
\]

其中 (T_{residual}) 是启动、依赖和尾部同步无法隐藏的部分。扩展效率可写成

\[
E(P)=\frac{T_1}{P\,T_P},\quad
\text{或}\quad E_{tok}(P)=\frac{\text{tokens/s}_P}{P\,\text{tokens/s}_1}.
\]

在不同并行度下保持全局 batch、序列长度和精度不变，才能比较 (E(P))。如果为了塞进显存而降低序列长度，吞吐提升未必代表规模效率提高。

### 9.1.3 选择并行维度的三个问题

1. **哪个对象最大？** 参数/优化器状态太大，优先考虑 ZeRO/FSDP；单层矩阵太大，考虑 TP；层数多且激活占主导，考虑 PP；专家参数多但每 token 只访问少数专家，考虑 EP。
2. **哪条链路最慢？** 同一节点的 NVLink/PCIe、跨节点 NIC、跨机架交换机带宽差异可达数量级。高频 all-reduce/all-gather 应尽量局限在快域。
3. **故障后需要保留什么语义？** 如果要求精确恢复，必须保存 RNG、数据游标、梯度累积阶段和 optimizer step；如果只要求近似恢复，允许从最近 checkpoint 重新取样，但要记录样本重复范围。

## 9.2 并行度地图：DP、TP、PP、SP/CP、EP

设总设备数 (P=D\times T\times Q\times E)，其中 (D) 为数据并行度，(T) 为张量并行度，(Q) 为流水线 stage 数，(E) 为专家并行相关的设备组大小。实际系统还可能有上下文并行或副本维度，公式只是帮助检查乘积是否与 world size 匹配，不代表所有维度都必须独立。

### 9.2.1 数据并行（Data Parallel, DP）

每个副本持有完整模型参数和优化器状态，不同 rank 处理不同样本。反向传播后对梯度做 AllReduce 或 ReduceScatter。优点是编程模型直观、单副本计算高效；缺点是每张卡都复制参数、梯度和状态，模型规模受单卡内存限制。DP 的通信量与梯度总大小 (G) 成正比，与 batch 和层数无关，但通信发生在每个 step 的许多 bucket 上。

经典 ring AllReduce 的每 rank 发送和接收约 (2(P-1)G/P) 字节（忽略协议开销）。当 (G) 很大、跨节点带宽有限时，DP 可能被通信主导。梯度 bucket 化、通信压缩和与反向重叠只能隐藏一部分成本，不能消除全量复制本身。

### 9.2.2 张量并行（Tensor Parallel, TP）

TP 把单个线性层或注意力头沿矩阵维度切分到 (T) 个 rank。以 (Y=XW) 为例：

- **列并行（column parallel）**：(W=[W_0,\dots,W_{T-1}])，每 rank 计算 (Y_t=XW_t)。若下一层需要拼接完整 (Y)，执行 AllGather；若后续层也采用相容的行切分，可以避免立即拼接。
- **行并行（row parallel）**：(W=[W_0;\dots;W_{T-1}])，输入切成 (X_t)，每 rank 计算 (X_tW_t)，最终对部分和做 AllReduce。

Transformer 中常见“列并行的 QKV/上投影 + 行并行的输出投影”。一次前向可能包含 AllGather 或 AllReduce，反向还会产生对应的通信。TP 的好处是单层权重和中间激活按 (T) 缩小；代价是每层都可能通信，因此应优先放在高带宽、低延迟互联的同一节点内。把 TP=8 横跨两个只有以太网的节点，通常比 TP=4 每节点再做 DP 更差。

### 9.2.3 流水线并行（Pipeline Parallel, PP）

把连续层划分为 (Q) 个 stage，每个 stage 只保存其层和局部激活。micro-batch 在 stage 间以 send/recv 传递。最简单的 GPipe 调度先执行所有 forward，再执行所有 backward；1F1B（one-forward-one-backward）在 warmup 后交替执行，减少激活驻留。

设每个 micro-batch 的 stage 时间近似为 (t)，stage 数为 (Q)，micro-batch 数为 (M)。理想无气泡时间约为 ((M+Q-1)t) 的一个方向；相对于 (Mt) 的流水线效率上限可粗略写成

\[
\eta_{bubble}\approx \frac{M}{M+Q-1}.
\]

例如 (Q=4,M=4) 时上限约 (4/7=57\%)；增大 (M=16) 时上限约 (16/19=84\%)。但 (M) 增大意味着更多梯度累积和调度开销，且 stage 不均衡会把最快 stage 拖到最慢 stage 的节奏。切分层时应以 profile 的实际 FLOPs、激活大小和通信边界为准，而不是按层数平均。

流水线有两种语义需要区分：**同步更新**要求所有 micro-batch 的梯度在一个 optimizer step 结束时使用同一版本参数；**异步或 stale 更新**允许 stage 使用稍旧参数，吞吐更高但收敛分析不同。大多数现代训练采用 1F1B 的同步变体，若框架标注为 interleaved、zero-bubble 或 asynchronous，要检查其参数版本和梯度归并协议。

### 9.2.4 序列并行与上下文并行（SP/CP）

Sequence Parallel（SP）通常与 TP 一起使用，把 LayerNorm、Dropout 等沿序列维切分，使非线性激活不必在每张 TP 卡完整复制。典型做法是在线性层前后用 AllGather/ReduceScatter 转换布局，在计算 token-wise 操作时保持局部序列片段。SP 的核心不是“减少参数”，而是减少 (B\times S\times H) 激活的复制。

Context Parallel（CP）进一步把长序列的注意力上下文分到多个 rank。朴素注意力需要每个 query 看到完整 key/value，可能通过 ring attention 让 K/V 分块环流，或采用 block-sparse/窗口注意力减少可见范围。CP 的通信量随序列长度和注意力模式变化，不能简单套用 TP 的矩阵公式。实现时要明确 causal mask 的全局位置偏移、KV cache 索引以及跨 rank 的位置编码一致性。

### 9.2.5 专家并行（Expert Parallel, EP）

MoE 层有 (E) 个专家，每个 token 由门控网络选择 top-k 专家。专家权重按 rank 分片，token 需要通过 AllToAll/AllToAllV 路由到拥有目标专家的 rank，再把结果路由回去。与 DP/TP 不同，EP 的通信量取决于 token 路由和容量因子（capacity factor），负载不平衡会让某些专家成为热点。

门控过程通常包含：计算 logits → 取 top-k → 计算每专家 token 数 → capacity 截断/丢弃或 padding → token dispatch → 专家 FFN → combine。训练日志至少要记录专家负载直方图、丢 token 比例、辅助负载均衡损失和 AllToAll 时间。为了避免数据泄漏或不可复现，不应在不同 rank 上独立进行随机 tie-break；使用共享 RNG 或确定性排序。

### 9.2.6 混合并行的笛卡尔网格

实际 LLM 训练常见 (D\times T\times Q) 或 (D\times T\times Q\times E) 网格。例如 16 卡可选 (D=2,T=4,Q=2)。rank 映射要把通信频繁的维度放在物理邻近设备：先按节点内 GPU 编号分配 TP，再跨节点分配 DP，最后按层边界安排 PP。若 rank 线性编号与拓扑不一致，collective 仍能工作，但 NCCL/MPI 可能走更慢路径。

## 9.3 梯度、优化器与内存：为什么单纯 DP 会 OOM

### 9.3.1 四类内存的字节账本

以参数数量 (N) 为单位，设参数存储宽度为 (b_p) 字节，梯度宽度 (b_g)，优化器状态字节数 (b_o)，激活峰值为 (A)，临时通信缓冲为 (C)。单卡 DP 的粗略峰值：

\[
M_{DP}\approx N(b_p+b_g+b_o)+A+C.
\]

以 FP16/BF16 混合精度 Adam 为例，常见配置是模型权重 2B、梯度 2B、FP32 master weight 4B、Adam 的 m/v 各 4B，总计约 16B/参数；具体实现可能保留额外副本或使用 8-bit optimizer，不能只按“模型文件大小”估算。若 N=7B，仅参数相关就可能超过 100GB，再加激活和碎片化，单卡显然无法容纳。

激活与 batch、序列长度、隐藏维、层数近似成正比：(A\propto BSKL)，其中 (K) 表示保存的层间中间量。梯度累积增加同时驻留的 micro-batch 数，流水线调度和 checkpoint 策略会改变 (K)。因此“把 batch 减半”通常能缓解激活 OOM，却不影响参数/优化器 OOM；反之，ZeRO 只切状态，不自动解决长序列激活峰值。

### 9.3.2 梯度桶与通信时机

反向传播按参数拓扑的逆序产生梯度。框架把就绪梯度装入 bucket，达到阈值就启动 ReduceScatter/AllReduce，尝试与更早层的计算重叠。bucket 太小，启动开销和 collective 数量增加；太大，等待最后一个梯度导致重叠变差。改变模型、batch 或参数注册顺序可能改变 bucket 分组，从而让“同一代码”性能波动。

使用梯度累积时，常见两种模式：

1. 每个 micro-batch 都通信并平均，再累积到 optimizer step；通信次数高。
2. 前 (K-1) 个 micro-batch 暂停同步（如 `no_sync`），只在最后一次同步；通信量低，但本地梯度在累积期间占内存，且溢出/异常时重做成本更高。

无论哪种模式，都要明确 loss 是否除以 (K)，否则有效梯度会放大 (K) 倍。

## 9.4 ZeRO 与 FSDP：切分状态和参数生命周期

### 9.4.1 ZeRO 三个阶段

ZeRO（Zero Redundancy Optimizer）的基本思想是：数据并行副本不必各自保存完全相同的状态，可以在 (D) 个 rank 间分片。

- **ZeRO-1**：只分片优化器状态。每 rank 仍持有完整参数和梯度，优化器状态约降为 (1/D)。更新后需要广播或同步参数。
- **ZeRO-2**：分片优化器状态和梯度。梯度通过 ReduceScatter 分到 owner rank，参数仍完整复制。
- **ZeRO-3**：优化器状态、梯度、参数都分片。前向/反向使用某层前临时 AllGather 参数，计算后释放或重新分片。参数峰值接近单层或通信 bucket，而不是整模型。

用 (P) 表示参数字节、(G) 表示梯度字节、(O) 表示优化器状态字节，忽略临时缓冲：

\[
M_{Z1}\approx P+G+O/D,
\quad M_{Z2}\approx P+G/D+O/D,
\quad M_{Z3}\approx (P+G+O)/D + P_{active}+C.
\]

(P_{active}) 取决于同时 all-gather 的层/bucket 数；如果预取过度，峰值会高于公式。ZeRO-3 的通信通常比 ZeRO-2 更多，尤其是前向参数 all-gather 和反向梯度 reduce-scatter，必须让通信与计算重叠，否则节省下的内存换成吞吐下降。

### 9.4.2 FSDP 的 full shard 与 reshard

PyTorch Fully Sharded Data Parallel（FSDP）可以理解为参数/梯度/优化器状态的分片封装，并以“flat parameter”聚合多个原始参数来减少通信调用。一个典型 full-shard 生命周期：

1. rank 只保留本地参数 shard。
2. 进入模块前，AllGather 组成完整 flat parameter。
3. 执行 forward；若 `reshard_after_forward=True`，立即释放完整参数，仅保留 shard。
4. backward 前再次 AllGather（或利用保留的参数，牺牲峰值内存换通信）。
5. 反向产生完整梯度或局部梯度，ReduceScatter 到各 rank 的梯度 shard。
6. 本地 optimizer 更新自己的状态和参数 shard。

FSDP 的 wrapping 策略决定 all-gather 的粒度。把整个模型包成一个 FSDP 单元会减少元数据但导致巨大峰值；按 Transformer block 包装可获得更好的重叠。自动 wrapping 应通过 profile 验证，不能只依赖参数数量阈值。

FSDP 与 ZeRO-3 的数学目标相似，但实现、状态字典和通信时机不同。加载 checkpoint 时要区分 `FULL_STATE_DICT`、`SHARDED_STATE_DICT` 和 `LOCAL_STATE_DICT` 等语义；不要把某个 rank 的本地 shard 当成完整模型发布。跨 world size 恢复需要有明确的重分片规则或使用框架提供的 reshard 工具。

### 9.4.3 参数状态机与内存峰值

可把一个 FSDP/ZeRO-3 模块画成状态机：`SHARDED -> GATHERED -> COMPUTE -> RESHARD -> SHARDED`。每次状态转移都有依赖：

- gather 必须在计算 stream 使用参数前完成；
- reshard 不能早于最后一个使用参数的 kernel；
- gradient reduce-scatter 必须在 optimizer 读取梯度前完成；
- optimizer 更新时参数 shard 与状态 shard 必须属于同一 step。

调试 OOM 时记录每个阶段的 `allocated`、`reserved`、all-gather buffer 大小和当前模块名。只看 step 结束时的显存会错过瞬时峰值。若多个模块预取，峰值近似“当前模块 + 预取模块 + 梯度桶 + 激活”，可通过减小 prefetch/bucket 或关闭 `limit_all_gathers` 的相关选项（按框架版本）验证。

## 9.5 激活检查点（Activation Checkpointing）与重计算

### 9.5.1 核心交换

普通反向会保存每层中间激活以计算梯度。Activation Checkpointing（AC）只保存选定边界，在 backward 时从边界重新执行 forward，换计算时间省内存。若模型有 (L) 层，每段长度为 (s)，保存约 (L/s) 个边界；粗略激活内存从 (O(L)) 降为 (O(L/s+s))，重计算开销约增加一遍或多遍前向，实际取决于分段和算子。

AC 必须保证重算 forward 与原 forward 的随机性和输入一致。Dropout、随机路由、量化噪声和自定义 CUDA kernel 可能依赖 RNG 状态。框架通常保存并恢复 CPU/CUDA RNG，但跨 rank 的 RNG 顺序若被条件分支打乱，仍会产生不同 mask。若使用 checkpoint 的非重入/重入实现，检查其对副作用、in-place 操作和异常的约束。

### 9.5.2 与流水线、FSDP 的组合

PP 中每个 stage 已经持有多个 micro-batch 的激活，AC 能显著降低 stage 内存，但重算可能与后续 micro-batch 的通信争用计算资源。FSDP 中，重算时参数可能再次 all-gather；若 reshard-after-forward 开启，AC 会引入额外的参数通信。要比较三种策略：

1. 保留完整参数、不重算：通信少，显存高。
2. reshard + 重算：显存最低，通信和计算最高。
3. 选择性重算：只对 attention/MLP 的大激活重算，保留便宜或通信敏感的激活。

通过 profile 记录 forward、backward、recompute、all-gather、reduce-scatter 的时间和峰值内存，按目标（吞吐还是可扩展规模）选择。

### 9.5.3 何时不能随意 checkpoint

如果模块有不可逆的外部副作用（写文件、更新全局计数器、修改缓存）或依赖“只执行一次”的随机采样，重算会改变语义。应把副作用移到 checkpoint 区域之外，或显式保存所需状态。对于推理路径和评估路径，AC 通常没有收益，反而会增加延迟。

## 9.6 拓扑与 rank 映射：把通信放到正确的链路

### 9.6.1 两层带宽模型

设节点内带宽 (B_{intra})、跨节点带宽 (B_{inter})，且 (B_{intra}\gg B_{inter})。对于需要 (G) 字节通信的 collective，分层算法先在节点内归约，再跨节点交换节点代表，最后在节点内广播，可近似为

\[
T\approx \alpha_{intra}+\frac{G}{B_{intra}}
+\alpha_{inter}+\frac{G}{B_{inter}N_{node}}
+\alpha'_{intra}+\frac{G}{B_{intra}}.
\]

实际 NCCL/MPI 会根据拓扑、消息大小和进程数选择 ring/tree/分层算法；公式用于发现数量级错误。例如把 TP 的每层 AllReduce 放跨节点，(G/B_{inter}) 会在每层重复出现，而把 DP 的大梯度 reduce-scatter 放在跨节点、在节点内先做聚合通常更合理。

### 9.6.2 rank、设备与网络接口

启动前生成映射表：`global_rank, node_rank, local_rank, hostname, gpu_uuid, numa_node, nic`。验证：

- local_rank 不超出本节点设备数；
- 同一 TP 组尽量共享 NVLink/PCIe root complex；
- 同一 DP 组的跨节点成员使用可达且负载均衡的 NIC；
- PP 相邻 stage 之间的 send/recv 不穿越不必要的交换机；
- CPU 线程和数据加载 worker 绑定到正确 NUMA 节点。

不要通过修改 `CUDA_VISIBLE_DEVICES` 随意“修复”rank；它会改变 local_rank 到物理 GPU 的映射，导致 checkpoint 或监控中的设备 ID 不一致。应在启动器层显式生成 hostfile 和映射，再把映射写入日志。

### 9.6.3 两节点 toy 设计

下面设计一个可在两台普通 CPU 机器上模拟的拓扑（也可在单机用两个进程伪造）：

- 节点 A：rank 0、1；节点 B：rank 2、3；总 (P=4)。
- 每节点内链路成本设为 (\alpha_{intra}=1,\beta_{intra}=1/100)，跨节点成本 (\alpha_{inter}=5,\beta_{inter}=1/10)（单位可理解为任意时间/字节）。
- 并行配置：TP=2（每节点内一组），DP=2（rank 0↔2、rank 1↔3），PP=1。
- 每 step 先对 TP 组做小矩阵 AllReduce，再对 DP 组做大梯度 ReduceScatter。

用脚本计算两种映射：方案 A 让 TP 跨节点，方案 B 让 TP 节点内。若 TP 消息 (g_t=10) MB，DP 消息 (g_d=100) MB，代价模型给出

\[
T_A\approx 2(\alpha_{inter}+g_t\beta_{inter})
 +(\alpha_{inter}+g_d\beta_{inter}),
\]

\[
T_B\approx 2(\alpha_{intra}+g_t\beta_{intra})
 +(\alpha_{inter}+g_d\beta_{inter}).
\]

两者 DP 项相同，方案 B 把每层频繁的小通信留在快链路，通常明显更快。toy 实验不模拟真实 NIC，但能验证“映射改变路径、路径改变代价”的推理。

## 9.7 检查点一致性：不仅是保存一个 `.pt` 文件

### 9.7.1 需要保存的状态集合

一个可精确恢复的训练检查点至少包括：模型参数（完整或分片）、optimizer 状态、学习率调度器、混合精度 scaler、当前 global step、epoch 和数据迭代器游标、梯度累积阶段、随机数状态（Python/NumPy/框架 CPU/GPU）、并行拓扑与 world size、代码/配置版本、词表或数据集版本。若使用 MoE，还要保存专家路由统计或能重建其状态的元数据。

“保存模型权重”只能做推理或近似继续训练。缺少 optimizer 状态会改变 Adam 的动量和二阶矩；缺少数据游标会重复或跳过样本；缺少 RNG 会让 dropout、采样和数据增强路径不同。文档应明确 checkpoint 的恢复等级：`weights_only`、`optimizer_resume` 或 `exact_resume`。

### 9.7.2 两阶段提交式保存

分布式保存时不要让每个 rank 直接覆盖同一个文件。推荐流程：

1. 每 rank 写临时 shard：`ckpt.tmp/step_000123/rank_0003.bin`，写入校验和与长度。
2. `fsync` 文件和目录（若存储系统支持），rank 间 barrier。
3. coordinator 汇总 manifest：step、world_size、每 shard 路径/大小/哈希、配置哈希、随机状态索引。
4. 写 `COMMITTED` 标记或原子 rename 到最终目录。
5. 更新一个小的 `latest` 指针，指向已提交目录；指针更新应原子化。

恢复时先读 manifest，验证所有 shard 存在、长度和校验和匹配，再构造分片状态。若只有部分 rank 写完，目录没有 `COMMITTED`，应视为未完成 checkpoint，不要“尽量加载”。保留最近两个可验证 checkpoint，删除旧版本前先确认新版本可读取。

### 9.7.3 FSDP/ZeRO 状态字典陷阱

分片 state dict 的键名和扁平化方式可能随框架版本、wrap 策略和 world size 改变。发布前应在至少以下场景测试：同 world size 恢复、不同 world size 重分片、单进程加载用于评估、缺一个 shard 时明确失败。不要把 `rank0` 的本地 state dict 当成完整权重；需要完整权重时让框架执行 gather，并估算 coordinator 内存。

## 9.8 容错与恢复：把失败当作协议状态

### 9.8.1 失败分类

- **可重试瞬时错误**：网络超时、文件系统临时不可达、NCCL 操作超时。先记录 collective 序号和通信域状态，再按策略重建进程组。
- **进程级失败**：一个 rank OOM、Python 异常或被杀死。多数 collective 不能安全地继续；需要终止同组进程，从最近一致 checkpoint 重启。
- **节点级失败**：整机断电、GPU/Xid、NIC 不可用。若有弹性作业和足够 spare 节点，可以缩容或替换；否则停止并人工确认。
- **数据/检查点损坏**：哈希不匹配、manifest 缺失。禁止静默跳过；切换到上一个已提交 checkpoint，并记录丢失的 step。

### 9.8.2 心跳、超时与 watchdog

每个 rank 周期性写本地心跳，coordinator 维护最后进度 `last_collective_seq`、`last_step`、hostname 和进程状态。超时阈值应大于正常长尾（例如 checkpoint flush），并把“正在保存”作为显式阶段，避免误杀。watchdog 检测到 rank 失联时，先收集诊断信息（堆栈、GPU memory、网络统计、最近 collective），再触发组级退出。不要让一个 rank 无限等待；无限 timeout 会把节点资源锁死。

### 9.8.3 可重复恢复与样本语义

恢复后有三种数据语义：

1. **精确继续**：恢复数据游标和 RNG，每个样本只处理一次；实现最复杂。
2. **至少一次**：可能重复最近一个 checkpoint 之后的样本，但不跳过；适合多数预训练，需记录重复范围。
3. **尽力继续**：重新打乱数据，可能重复或跳过；适合探索性实验，不适合可审计训练。

选择哪种语义必须写入训练配置和结果报告。不要声称“可恢复”却没有定义样本重复和 optimizer step 的关系。

### 9.8.4 弹性 world size 的边界

world size 变化会改变有效 batch、学习率、数据分片、ZeRO/FSDP shard 划分和 RNG 顺序。若要弹性缩容，必须定义：是否保持 (B_{global})（相应调整每 rank batch/累积）、如何重新分片参数/状态、如何分配数据游标、是否接受非 bitwise 的结果差异。只重启进程而不重建 process group，通常会导致 collective mismatch。

## 9.9 CPU 单机可运行实验

以下实验不依赖 GPU/NCCL，使用 Python 标准库和 PyTorch CPU（若未安装 PyTorch，可先运行纯 Python 代价模型）。目标是验证推导和协议，而非模拟真实 GPU 速度。运行前固定线程数，避免 BLAS 线程把通信时间污染：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu  # 按环境决定是否需要
```

### 9.9.1 实验 A：DP 梯度平均与线性代数推导

创建 `ch09_dp_cpu.py`：

```python
import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29571"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(7 + rank)
    # 相同模型，不同本地样本；y = 2x + 1 的回归
    w = torch.tensor([0.0], requires_grad=True)
    x = torch.tensor([float(rank + 1), float(rank + 2)])
    y = 2 * x + 1
    pred = w * x
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    local_grad = w.grad.detach().clone()
    dist.all_reduce(w.grad, op=dist.ReduceOp.SUM)
    w.grad /= world
    print(f"rank={rank} local={local_grad.item():.6f} "
          f"global={w.grad.item():.6f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
```

手算每个 rank 的局部梯度 (g_r=\frac{1}{2}\sum_i 2(wx_i-y_i)x_i)，再取 ((g_0+g_1)/2)，应与打印的 `global` 一致。把 `w.grad /= world` 删除，观察参数更新会放大 2 倍；把一个 rank 的 `all_reduce` 替换成条件跳过，程序会在 Gloo 等待，说明顺序契约比“代码看起来合理”更重要。

### 9.9.2 实验 B：ZeRO-1 内存账本和通信代价

不需要真正切分大模型，使用整数模拟验证公式：

```python
# ch09_zero_accounting.py
from dataclasses import dataclass

@dataclass
class State:
    params: int       # bytes
    grads: int
    optim: int
    active: int = 0
    comm: int = 0

def estimate(s, D, stage):
    if stage == 1:
        return s.params + s.grads + s.optim / D
    if stage == 2:
        return s.params + s.grads / D + s.optim / D
    if stage == 3:
        return (s.params + s.grads + s.optim) / D + s.active + s.comm
    raise ValueError(stage)

s = State(params=7_000_000_000 * 2,
          grads=7_000_000_000 * 2,
          optim=7_000_000_000 * 8,
          active=250_000_000, comm=64_000_000)
for D in (1, 2, 4, 8):
    print(D, [round(estimate(s, D, k) / 2**30, 2) for k in (1, 2, 3)])
```

解释输出时列出假设：参数/梯度 BF16 各 2B，Adam m/v 及 master 权重合计 8B，active/comm 是假设的临时峰值。将 `active` 加倍可看到 ZeRO-3 仍可能 OOM；将 `optim` 设为 0 则显示推理场景的差异。

### 9.9.3 实验 C：流水线 bubble 的离散事件模拟

用不依赖框架的模拟验证 ((M+Q-1)) 直觉：

```python
# ch09_pipeline_bubble.py

def schedule(Q, M):
    # 每个 stage 每个 micro-batch 用一个时间单位；先完成所有 F，再完成所有 B
    f_end = [[0] * M for _ in range(Q)]
    for q in range(Q):
        for m in range(M):
            deps = [f_end[q][m - 1] if m else 0]
            if q:
                deps.append(f_end[q - 1][m])
            f_end[q][m] = max(deps) + 1
    b_end = [[0] * M for _ in range(Q)]
    for q in reversed(range(Q)):
        for m in reversed(range(M)):
            deps = [f_end[q][M - 1]]       # 同一 stage 的 F 全部完成
            if q + 1 < Q:
                deps.append(b_end[q + 1][m]) # 下游先做 B
            if m + 1 < M:
                deps.append(b_end[q][m + 1]) # micro-batch 逆序
            b_end[q][m] = max(deps) + 1
    return b_end[0][0]

for Q in (2, 4, 8):
    for M in (Q, 2*Q, 4*Q):
        makespan = schedule(Q, M)
        print(f"Q={Q} M={M} makespan={makespan} ideal={2*M}")
```

该简化调度没有实现 1F1B 的所有细节，重点是让 warmup/cooldown 和 bubble 可视化。可在下一步为每个 stage 加入 1F1B 状态机，比较激活驻留数；验收标准是随着 (M/Q) 增大，bubble 比例下降而并非消失。

### 9.9.4 实验 D：两节点拓扑 toy 成本

```python
# ch09_topology_toy.py

def cost(bytes_, alpha, beta):
    return alpha + bytes_ * beta

intra = (1.0, 1/100e6)  # 任意单位，100 MB/s 等比例
inter = (5.0, 1/10e6)
tp, dp = 10e6, 100e6
A = 2*cost(tp, *inter) + cost(dp, *inter)  # TP 跨节点
B = 2*cost(tp, *intra) + cost(dp, *inter)  # TP 节点内
print(f"TP-cross={A:.3f}, TP-local={B:.3f}, speedup={A/B:.2f}x")
```

把 `inter` 的带宽改成 1/2e6，观察跨节点 TP 的代价进一步恶化。再把 DP 消息拆成节点内聚合后跨节点的分层公式，比较减少的跨节点字节。该实验是代价模型练习，不能当作真实 NCCL benchmark；真实系统还受协议、拥塞、并发 stream、NUMA 和消息大小影响。

### 9.9.5 实验 E：检查点原子提交与损坏恢复

```python
# ch09_checkpoint_atomic.py
from pathlib import Path
import hashlib, json, os, shutil, tempfile

root = Path('toy_ckpt'); root.mkdir(exist_ok=True)
step = 3
final = root / f'step_{step:06d}'
if final.exists():
    shutil.rmtree(final)
stage = root / f'step_{step:06d}.tmp'
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(exist_ok=True)
blob = b'parameter-shard-rank0'
(path := stage / 'rank0.bin').write_bytes(blob)
manifest = {'step': step, 'files': [{'name': path.name,
    'size': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}]}
(stage / 'manifest.json').write_text(json.dumps(manifest))
os.replace(stage, root / f'step_{step:06d}')
(root / 'latest.tmp').write_text(f'step_{step:06d}')
os.replace(root / 'latest.tmp', root / 'latest')
print('committed', (root / 'latest').read_text())
```

在 `os.replace` 前中断进程，恢复脚本应忽略 `.tmp` 目录；手工修改 `rank0.bin` 后重新计算哈希，恢复必须失败并回退到上一个 `COMMITTED` step。这个 toy 只验证协议结构，不覆盖对象存储的最终一致性；云存储上要使用其条件写入/版本化能力。

## 9.10 训练系统的诊断清单

### 9.10.1 Hang 或超时

1. 打印每个 rank 最近的 collective 序号、op、shape、dtype、stream/线程。
2. 对照 rank 映射，确认是否有进程提前退出、数据迭代器长度不同或异常被吞掉。
3. 将 batch、层数和 world size 缩到 2 个 rank 的最小复现，关闭异步 overlap。
4. 检查网络接口、端口、防火墙和 NCCL/Gloo/MPI 后端日志；不要先盲目把 timeout 调到数小时。
5. 如果一个 rank OOM，其他 rank 的 hang 是结果而非根因；优先分析该 rank 的峰值内存。

### 9.10.2 OOM 或吞吐突降

按时间轴查看参数 all-gather、激活、梯度桶和 optimizer step 的峰值。逐项做二分：关闭 prefetch、减小 bucket、开启/增加 activation checkpoint、降低 micro-batch、关闭梯度累积、改变 FSDP wrapping。记录每次改变的 tokens/s、峰值内存和通信时间，避免“调参成功但不知为何”。

### 9.10.3 不收敛或 loss 抖动

先在单进程、固定 seed、小数据集上建立基线；再逐层打开 DP、TP、PP、AMP、梯度累积。检查梯度缩放是否重复、loss 是否正确除以 (D) 和 (K)、参数版本是否在流水线中 stale、MoE 是否丢 token、恢复后 optimizer step 是否跳跃。对比若干层的梯度范数和参数 checksum，定位首次分歧，而不是只看最终 loss。

### 9.10.4 Checkpoint 无法加载

读取 manifest 和版本字段，确认 world size、wrap 策略、dtype、词表、代码 commit。先在 CPU 上加载小 shard 并验证哈希，再尝试重分片。若缺少 shard 或哈希不匹配，停止自动训练，报告可恢复的最新 step；不要用“忽略缺失键”把损坏状态伪装成成功恢复。

## 9.11 六个理解检查（含答案）

### 检查 1：为什么 TP 通常放在节点内，而 DP 可以跨节点？

**答案**：TP 在几乎每层都产生 AllReduce/AllGather，消息频繁且延迟敏感；DP 主要在梯度桶就绪后做较大消息的 ReduceScatter/AllReduce，次数相对少，并可通过分层算法先节点内聚合。若跨节点互联异常快或模型结构特殊，结论可改变，但必须用 profile 和拓扑数据证明。

### 检查 2：ZeRO-2 已分片梯度，为什么仍可能 OOM？

**答案**：ZeRO-2 保留完整参数副本，且激活、临时 all-reduce buffer、通信 bucket 和碎片化不受其直接影响。参数或长序列激活足够大时仍会超限；需要 ZeRO-3/FSDP、TP/SP 或 activation checkpoint 等组合。

### 检查 3：流水线增加 micro-batch 是否总能提升吞吐？

**答案**：增加 (M) 会降低 bubble 比例 (Q-1) 相对开销，但同时增加调度、激活驻留和梯度累积开销。stage 不均衡、输入瓶颈或通信未重叠时，吞吐可能不升反降；应测量实际 makespan 和内存峰值。

### 检查 4：为什么恢复 checkpoint 不能只加载模型权重？

**答案**：继续训练还依赖 optimizer 动量/二阶矩、学习率 scheduler、混合精度 scaler、global step、数据游标和 RNG。缺失这些会改变有效学习率、样本序列和随机路径，得到的是从同一初始化附近重新训练，而不是精确继续。

### 检查 5：一个 rank 跳过 all-reduce 会发生什么？

**答案**：其他 rank 会等待匹配参与者，通常超时或收到通信错误；即使某实现意外返回，梯度也不再是定义的全局梯度。条件分支必须在所有 rank 上做同样的 collective 决策，或把不同路径放到不同 process group 并显式同步。

### 检查 6：EP 中专家负载不均衡为什么会降低吞吐？

**答案**：AllToAll 后每个专家处理的 token 数不同，最忙的 rank 决定同步步长，其他 rank 空转；容量截断还会丢 token，改变有效训练信号。应监控负载直方图、capacity factor、丢 token 率和路由通信时间，并调整门控辅助损失或专家布局。

## 9.12 练习题

1. 给定 (N=1\)B 参数、BF16 参数/梯度各 2B、Adam 状态 8B、(D=8)，估算 DP、ZeRO-1/2/3 的参数相关内存；再假设 3GB 激活和 1GB 临时缓冲，判断哪个阶段仍可能 OOM。
2. 设计 (D=2,T=4,Q=2) 的 16 卡 rank 映射，要求 TP 组都在同一节点（每节点 8 卡），并列出每个 DP 组和 PP 相邻边界。
3. 对 (Q=8) 计算 (M=8,32,128) 时的 bubble 上限，说明为什么只增大 (M) 不能修复 stage 负载不均衡。
4. 推导行并行线性层的前向 AllReduce 和反向通信；指出在什么布局下可以用 ReduceScatter 替代完整 AllReduce。
5. 为 MoE 设计一个最小日志 schema，至少包含每专家 token 数、丢 token 数、AllToAll 字节和 step；说明如何在 rank 失败后判断 checkpoint 是否可恢复。
6. 修改实验 A，使 rank 1 在第 3 step 故意退出；实现 watchdog 记录最后 collective 序号，并从实验 E 的最近提交点重启。报告哪些状态可以精确恢复，哪些只能至少一次。
7. 写一个 20 行以内的 manifest 验证器，拒绝缺 shard、长度不符、哈希不符和未提交目录；为其添加两个单元测试。
8. 设计一个 profile 表格，比较 DP、ZeRO-2、FSDP full-shard + AC 三种配置，列出峰值内存、tokens/s、all-gather 时间、reduce-scatter 时间、重算时间和恢复时间。

## 9.13 来源地图与阅读路线

以下来源优先采用官方文档、论文和标准，链接用于核对语义、API、版本边界和实验假设；论文中的渐近公式仍需在目标硬件上复测。

- [PyTorch Distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html)：进程组、collective、启动器和后端选择；用于核对 AllReduce、AllGather、ReduceScatter 语义及 CPU Gloo 实验。
- [PyTorch FSDP 文档](https://docs.pytorch.org/docs/stable/fsdp.html)：`FullyShardedDataParallel` 的参数生命周期、auto-wrap、state-dict 类型、混合精度和限制；用于实现 full-shard 与重分片恢复。
- [PyTorch FSDP `fully_shard` 文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)：参数 AllGather、梯度 ReduceScatter、reshard 和 stream 交互；用于检查新的 composable FSDP API。
- [ZeRO 论文](https://arxiv.org/abs/1910.02054) 与 [DeepSpeed ZeRO 文档](https://www.deepspeed.ai/tutorials/zero/)：ZeRO-1/2/3 的状态分片、通信/内存权衡和 offload 扩展；用于校准内存账本，不把理论峰值当成实测保证。
- [Megatron-LM 技术说明](https://github.com/NVIDIA/Megatron-LM)：Tensor Parallel、Pipeline Parallel、Sequence Parallel 和混合并行实现；用于理解列/行并行、1F1B 调度和 rank 网格。
- [NVIDIA NCCL 用户指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)：拓扑探测、分层 collective、环境变量和故障日志；仅在兼容 CUDA/NCCL 硬件上验证，不把 GPU 开关复制到 CPU。
- [GPipe 论文](https://arxiv.org/abs/1811.06965)、[PipeDream 论文](https://arxiv.org/abs/1806.03377)：流水线气泡、同步/异步调度和 stale 参数语义；用于推导 bubble 近似及其局限。
- [Switch Transformer 论文](https://arxiv.org/abs/2101.03961)：稀疏专家路由、容量因子和负载均衡；用于 EP 的 token dispatch 设计。
- [MPI-4.1 标准](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf)：communicator、collective、datatype、非阻塞通信和线程模型；用于跨后端对照调用契约。
- [NVIDIA `nvidia-smi topo` 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)：GPU/NUMA/NIC 拓扑输出和 P2P 能力检查；用于验证 rank 映射假设。

阅读顺序建议：先用第 9.9 节 CPU 实验验证 collective 与账本，再读 FSDP/ZeRO 文档；最后结合硬件拓扑和 profile 选择并行网格。论文中的 asymptotic 复杂度是上界模型，不能替代目标集群上的小规模 benchmark。

## 9.14 安全边界与运行纪律

1. **不把 toy 结果当生产 SLA**：CPU Gloo 只能验证语义和代价趋势，不能证明 GPU/NCCL 的吞吐或故障恢复时间。
2. **不静默忽略恢复错误**：缺 shard、哈希不符、参数 shape 不符、world size 不兼容都应显式失败并回退到已验证 checkpoint。
3. **不在未授权集群上启动大作业**：先确认节点、GPU、配额、数据许可、存储容量和预估费用；小规模 dry-run 通过后再扩展。
4. **不上传敏感数据到调试服务**：日志和 checkpoint 可能含有训练样本、路径、令牌或用户信息；共享前脱敏并使用最小权限。
5. **不绕过通信安全设置**：不要关闭 TLS、SSH host key 校验或防火墙来“解决”连接问题；遇到证书或安全警告应停下并让管理员处理。
6. **限制自动重试**：对 OOM、数据损坏和确定性逻辑错误无限重试会放大成本和损害；重试次数、退避和停止条件写入作业配置。
7. **保护检查点完整性**：设置写权限最小化、校验和、保留窗口和删除审批；删除唯一可恢复版本属于高风险操作。
8. **记录可审计元数据**：代码版本、配置、数据版本、并行映射、硬件拓扑、step 和恢复原因应进入 manifest，便于复盘和合规。

## 9.15 以单个训练步为单位的纵向剖析

前面的公式给出内存和通信的量级，但生产问题通常表现为“偶发变慢”“某个 rank 卡住”或“恢复后 loss 轻微漂移”。处理这类问题时，应该把一个训练步拆成可排序的事件，而不是只看进程结束时的平均吞吐。建议为每个 rank 记录同一单调时钟上的事件：读取 batch 开始和结束、前向每个 stage 的进入与离开、参数 all-gather、反向计算、梯度 reduce-scatter、optimizer step、写入检查点，以及 collective 的序号和参与者集合。事件应包含 step、micro-batch、layer 或 stage、字节数、开始时间、结束时间和返回状态。这样可以把“等待通信”与“计算本身变慢”区分开，也能发现某一 rank 比其他 rank 多做了一次同步的协议错误。

一个可操作的时间线如下。首先记录数据游标确认样本是否真正就绪；若读取结束时间晚于其他 rank 的前向开始时间，问题属于输入尾部而不是 GPU 算子。然后检查每个 collective 的进入时间差。若最早进入与最晚进入相差很大，而 collective 持续时间正常，说明上游计算不均衡；若进入时间接近但持续时间很长，才应调查拓扑、消息大小、拥塞或算法选择。对流水线，分别计算每个 stage 的有效计算时长和空闲时长，并将气泡拆成“等待上游激活”“等待下游反向”“显式调度空槽”三类。仅报告总体 bubble 比例会掩盖某一个 stage 的参数加载或缓存抖动。

常见的 all-reduce 挂起有三种模式。第一种是一个 rank 在条件分支中没有调用 collective，其他 rank 会一直等待；日志中能看到同一 collective 序号只在部分 rank 出现。第二种是各 rank 调用顺序不同，例如 rank 零先做梯度桶 A、再做桶 B，而 rank 一先做 B、再做 A；每个进程看起来都在通信，但匹配不到对端。第三种是某个 rank 已经因为数据异常退出，剩余进程继续发送，最终由 watchdog 报超时。诊断时应保留最后一个成功完成的 collective 序号、每个 rank 的进程组标识和退出码，不能只打印“通信超时”。重试前先确认是否需要重新初始化整个 process group；在同一坏状态上重复调用 collective 往往会扩大损坏范围。

FSDP 或 ZeRO 的 OOM 也需要时间线。若 OOM 发生在前向开始前，检查参数 all-gather 和预取是否重叠失败，尤其要看是否同时保留了前一个模块的完整参数和当前模块的梯度桶。若 OOM 发生在反向结束，重点查看梯度 reduce-scatter 是否延迟释放、通信 bucket 是否过大，以及 optimizer 状态是否在同一时刻 materialize。可以逐步关闭 prefetch、减小 bucket、减少同时在飞的 all-gather 数量，再比较峰值显存和 tokens/s。每次改变只动一个变量，并保存事件快照；否则“降低 batch 后不再 OOM”无法说明到底是激活、参数还是通信缓存得到缓解。

数值异常应沿着第一次分歧向前追踪。为固定的一小批样本保存若干层的参数校验和、梯度范数、损失缩放因子、溢出标志和 optimizer step。先在单进程得到基线，再开启 DP，再开启混合精度，最后加入 checkpoint 恢复。若 DP 开启后梯度范数约为单进程的 D 倍，通常是 loss 或梯度平均漏除；若只有恢复后分歧，检查 scaler、学习率 scheduler、数据游标和 RNG 状态；若只有流水线分歧，检查 micro-batch 的归一化和参数版本。不要用放宽容差掩盖协议错误，应该标注“可接受的浮点非确定性”和“不可接受的状态缺失”两条边界。

## 9.16 受控实验矩阵与结果解释

当需要比较 DP、ZeRO、FSDP、激活检查点或不同拓扑时，先写出实验矩阵。固定模型结构、词表、数据快照、随机种子、编译选项和 CPU 线程数，只改变一个主因素；若必须同时改变 world size 与 batch，应把全局 batch、梯度累积步数和学习率换算写在实验记录中。每个点至少运行若干个热身步和若干个测量步，分别报告中位数、p95、标准差和样本数。启动阶段的 JIT、缓存构建和首个 all-gather 不应混入稳态吞吐，否则大模型和小模型的比较会产生系统性偏差。

扩展性实验不能只画 GPU 数量对 tokens/s。至少同时给出强扩展和弱扩展两组：强扩展保持全局 batch 和序列长度不变，观察通信和调度开销如何吞掉计算；弱扩展保持每卡 batch 不变，观察有效吞吐是否近似线性。用串行比例和并行效率解释拐点，而不是把拐点归因于“硬件不够快”。当增加 DP 后吞吐下降，先检查梯度桶大小和网络分层；当增加 TP 后下降，检查每层小消息的延迟以及是否跨越 NUMA 或节点；当增加 PP 后下降，检查 stage 不平衡和激活驻留。每个结论都要附带测量的消息字节、collective 次数和链路位置。

故障注入实验需要区分“故障被检测”与“状态被正确恢复”。可以在安全的 CPU toy 环境里注入四类故障：某个 rank 在 collective 前退出、写入 shard 后在 manifest 提交前终止、对象存储返回短读、恢复时 world size 改变。对每种故障记录检测延迟、停止了多少额外 step、恢复到的提交点、重复样本数量和 optimizer step 是否连续。若恢复点之前的 shard 完整但数据游标未提交，语义通常是至少一次；若数据游标与 optimizer 状态在同一 manifest 中原子提交，才有机会接近有效一次。实验报告应明确哪些状态是精确恢复，哪些状态只能保证不越过已提交边界。

通信实验还要做负向对照。把同一 collective 替换为无操作或本地复制，得到纯计算下界；把网络带宽人为限制，得到通信上界；再恢复真实设置。若真实时间接近计算下界，优化通信收益有限；若接近通信上界，应优先优化 bucket、拓扑或重叠，而不是继续改 kernel。对 MoE，除了平均 tokens/s，还应报告每个专家的 token 直方图、最大与平均负载比、容量截断率和 AllToAll 字节。平均值正常但尾部 rank 过载时，step 时间仍由最慢专家决定。

结果解释要保留不确定性。短实验中 GC、文件系统抖动、时钟同步误差和后台作业都可能造成百分之几的变化；如果两个配置差异小于测量噪声，不应宣称有稳定收益。建议使用同一节点重复运行，随机化配置顺序，并在报告中列出异常样本而不是静默删除。对关键结论增加一个独立实现或第二种后端的交叉验证，例如用 CPU Gloo 验证 collective 顺序，再用目标后端验证吞吐趋势。交叉验证不能证明绝对性能，但能降低把单一实现缺陷误判为算法结论的风险。

## 9.17 组合策略的边界与选择流程

选择并行策略应从最紧约束开始。若模型参数加优化器状态在单卡已超过显存，先考虑 ZeRO-3 或 FSDP full-shard；若参数可以容纳但序列长度造成激活峰值，优先考虑序列并行、上下文并行和激活检查点；若单层矩阵乘法本身太大，再引入 TP；只有当单卡和单节点仍无法容纳完整模型时，才把 PP 扩展到跨节点。这个顺序减少了同时引入多个通信维度的复杂度，也便于在每一步建立可比较的数值基线。

拓扑是组合策略的硬边界。TP 组要求高带宽低延迟，通常放在同一节点的互联域内；DP 可以通过分层 reduce-scatter 跨节点，但必须确认节点间链路和交换机端口没有过载；PP 的相邻 stage 需要持续传输激活，若跨节点链路抖动，会把气泡放大。建立 rank 网格时不仅记录 rank 编号，还要保存 GPU、NUMA、NIC、机架和可用链路的映射。任何自动放置器变更后都应重新跑拓扑 toy 和短 profile，不能假设 rank 连续就代表物理相邻。

混合精度和检查点策略也有边界。BF16 通常比 FP16 有更宽的指数范围，但并不意味着所有归一化、归约和损失计算都可直接降精度；对 softmax、归一化统计和梯度范数应保留必要的累加精度。启用梯度缩放时，记录 scaler 的增长、回退和溢出步，否则恢复后可能出现看似随机的 loss 跳变。检查点间隔应由可接受的恢复点目标、写入带宽和失败率共同决定；只按时间间隔保存而忽略写入队列，会在高峰期形成同步长尾。

当 world size 需要变化时，先区分允许改变的实验和要求连续语义的生产训练。允许改变时，应显式重算全局 batch、学习率、梯度累积和数据分片，并在 manifest 记录 membership epoch；要求连续语义时，应等待安全边界，保存所有 rank 的数据游标和 RNG，再按新网格重分片。即便参数值可以重分片，Adam 动量、混合精度 scaler 和采样器状态也可能需要转换。若转换规则没有经过小规模对照实验，应把恢复标记为“近似继续”而不是“精确继续”。

最后建立一个上线前决策门槛：一是单进程与双 rank 的 loss、梯度范数和校验和通过；二是目标拓扑下稳态吞吐和峰值内存有重复测量；三是 rank 退出、短读和未提交 manifest 的恢复探针通过；四是日志包含足够的 collective 序号、状态版本和硬件信息；五是停止条件、重试上限和回滚点已写入作业配置。缺少任一门槛时，扩大规模只会把未定义的语义变成更昂贵的故障。


## 9.18 小结：从“多卡运行”到“可解释的分布式训练”

DP、TP、PP、SP/CP 和 EP 是沿不同对象维度切分工作；ZeRO/FSDP 则进一步切分状态并管理参数的 gather/reshard 生命周期。选择方案时要同时看四类量：单卡内存、通信字节和频率、流水线 bubble/激活峰值、故障后的恢复语义。拓扑决定这些量真正经过哪条链路，checkpoint 协议决定失败后能否回到一个定义明确的状态。

在工程上，最可靠的路径通常是：先用单进程建立数值基线；用 2 个 CPU rank 验证 collective、梯度缩放和顺序；用 toy 代价模型比较 rank 映射；再在目标硬件上做小 batch profile；最后才增加并行维度和作业规模。任何“吞吐提升”“显存下降”“可恢复”的结论都应附带测量条件、状态语义和失败边界。这样，分布式训练就不再是神秘的启动命令，而是一组可以计算、实验、观测和安全回滚的协议。
