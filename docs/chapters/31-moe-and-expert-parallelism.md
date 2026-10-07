---
id: ch31-moe-and-expert-parallelism
title: Mixture-of-Experts 与 Expert Parallelism：路由、容量和 All-to-All 的工程边界
slug: /chapters/31-moe-and-expert-parallelism
description: 从 Top-k router、capacity factor 与 load balancing 到 Expert Parallel all-to-all，建立训练和 serving 的可观测故障边界
sidebar_position: 31
level: advanced
prerequisites:
  - ch08-distributed-collectives
  - ch09-distributed-training
  - ch10-training-ops
  - ch15-inference-execution
  - ch16-model-serving-system
  - ch19-inference-optimization-accelerator-stack
  - ch20-observability-debugging-incident-response
  - ch21-ai-reliability-engineering
learning_objectives:
  - 能从 router logits、top-k、capacity factor 和 token 数推导 expert capacity、overflow 与 drop/residual 语义
  - 能推导 Switch/GShard 风格的 load-balancing auxiliary loss，并区分概率均衡、实际 token 均衡和吞吐均衡
  - 能画出 Expert Parallel 的 dispatch、all-to-all、expert compute、combine 路径，计算 send/recv bytes、rank skew 和尾延迟
  - 能比较 DeepSpeed-MoE、Megatron-Core、vLLM 与 SGLang 中 MoE 的并行和 serving 边界，并按版本和源码验证而非猜测 flag
  - 能设计 capacity overflow、expert hot spot、collective timeout、silent token drop、权重版本不一致的故障检测与回滚合同
  - 能运行 CPU-only toy lab，复现实验数据、测试边界，并把结果写入 evidence manifest、报告和发布清单
estimated_hours: 44
hardware: CPU-only lab required; CUDA/NCCL/GPU optional and cost-bounded
risk_level: L4
last_verified: 2026-10-07
---

# 第31章　Mixture-of-Experts 与 Expert Parallelism：路由、容量和 All-to-All 的工程边界

> MoE 的核心不是“用更少的激活参数得到更大的模型”，而是在每一个 token 上做一次分布式的选择、搬运、计算和合并。router 的一个错误 tie-break、capacity 的一次取整、all-to-all 的一个 rank skew，都可能把理论上的稀疏计算变成实际的 token 丢失、通信尾延迟或不可重现的质量回归。本章把 MoE 看成一个协议：token 有 route id 和 position，expert 有容量和版本，collective 有顺序和故障边界，服务系统还必须知道哪些 token 在 overflow 时走 residual、重试或明确失败。

本章与第8、9章的边界是：第8章讲集体通信的抽象和正确性，第9章讲数据/张量/流水并行的组合；这里深入 MoE 的 token dispatch 和 expert parallel（EP）数据面。与第15、16章的边界是：那里描述模型执行和 serving 调度；这里关注稀疏 FFN、路由状态、专家装载及其 all-to-all 生命周期。与第30章一样，本章所有性能数字只属于给定配置的 toy 或受控测量，不是任何厂商的通用保证。

## 31.1 先写协议：一个 token 在 MoE 层里经过什么状态

对第 \(l\) 个 MoE layer，输入 hidden states 为 \(X\in\mathbb{R}^{T\times d}\)，router 产生 \(R=XW_r\in\mathbb{R}^{T\times E}\)，其中 \(E\) 是 expert 数。softmax 后得到 \(P\)，每个 token 选择 \(K\) 个 expert，形成 `(token_id, expert_id, gate, slot)` 记录。记录不能只保存 expert id：`slot` 决定它在目标 expert buffer 中的行号，`gate` 决定 combine 时的加权，`route_version` 和 `model_revision` 决定它能否与另一批 hidden states 混合。

一个可审计的状态机如下：

1. **ROUTED：** router logits、概率、top-k 和 deterministic tie-break 已确定；请求仍在源 rank。
2. **DISPATCHED：** token 按目标 rank 和 expert 的 `(expert_id, slot)` 排列，写入 send buffer；未越过 collective fence 不能复用 buffer。
3. **RECEIVED：** all-to-all 或等价通信完成，目标 rank 的每个 expert 拥有局部 token 和反向映射。
4. **COMPUTED：** expert FFN 完成，输出保留 route id、slot、gate 和 batch epoch；不同模型 revision 的输出禁止合并。
5. **COMBINED：** token 的 K 路输出按 gate 加权，恢复原 token 顺序；drop、residual 和空路由都需要显式标志。
6. **COMMITTED：** MoE 层输出送入下一层；trace 写入容量、实际路由、通信字节和异常原因。

取消请求或 collective 超时时，DISPATCHED/RECEIVED/COMPUTED 的临时 buffer 必须可回收，不能把旧 batch 的 route id 当成下一批的 slot。生产系统常用 generation 或 microbatch sequence 做隔离；仅靠 Python 对象地址或 CUDA stream 顺序不构成协议。

### 31.1.1 路由实验的四类证据

本章把陈述标成四类。**事实**来自 Switch Transformer、GShard、DeepSpeed、Megatron-Core、vLLM、SGLang 等论文、官方文档或源码；**机制**是对明确版本控制流的重建；**测量**来自 `labs/ch31_moe_parallelism_lab.py` 的 CPU toy；**设计判断**是为了满足给定 SLO、可回滚和多租户边界的建议。一个日志说“平均 load 很好”属于测量，不等于所有请求分布均衡；一个 README 说“支持 MoE”也不等于当前 checkpoint、量化和 TP/EP 组合能工作。

## 31.2 Router 与 Top-k：从 logits 到可复现的选择

### 31.2.1 softmax、温度与门控概率

最常见的门控概率为：

\[
p_{t,e}=\frac{\exp((r_{t,e}+b_e)/\tau)}{\sum_{j=1}^{E}\exp((r_{t,j}+b_j)/\tau)}.
\]

\(b_e\) 可以是 expert bias 或负载调节项，\(\tau\) 是 router temperature。训练时通常保留完整概率用于 auxiliary loss，dispatch 时只发送 top-k。router logits 的精度和归一化顺序必须版本化：FP16 下先减 max 再 exp 是必要的数值稳定步骤，top-k tie 时要有明确的 expert id 次序，否则不同 rank 或不同 kernel 可能给出不同 route。

温度不是一个无害的超参。温度降低会使 gate 更尖锐，容量压力和 hot expert 风险增加；温度升高会让更多 expert 获得概率质量，却可能让实际 top-k 选择更加敏感于量化噪声。训练期间改变温度而不重新校准 aux loss 权重，常见结果是 loss 看似下降、实际 token drop 上升。

### 31.2.2 Top-1、Top-2 与任意 top-k

Top-1 每个 token 只有一条 expert 路径，dispatch 和 combine 便宜，overflow 语义也简单；Top-2 提供冗余和更平滑的梯度，但 token 数、通信字节和 expert buffer 需求近似翻倍。更大的 k 可能提高表达能力，却把每层的 all-to-all 放大为 \(O(TK d)\)，并增加 combine 的内存读写。

一个 token 的 top-k 选择应使用稳定排序：按 `(-logit, expert_id)` 排序，再截断 K。`argpartition` 的同分行为可能依赖底层实现；如果 checkpoint 在不同硬件上训练/恢复，tie-break 不稳定会改变 expert load 和 optimizer state。route record 必须记录 `router_seed`（若有采样）、`logit_dtype`、`top_k`、`normalization` 和 `tie_break_policy`。

Top-k gate 的 combine 有两种常见语义：一是把 K 个 gate 重新归一化后加权；二是保留原 softmax 概率，未选 expert 的质量视为丢弃。两者输出尺度不同，residual 分支和 LayerNorm 的统计也不同。切换实现时不能只改 `top_k` flag，应在小批量上对 gate sum、输出范数、梯度和 token drop 做 golden test。

### 31.2.3 Token choice 与 expert choice

**Token choice** 由每个 token 选择最喜欢的 K 个 expert，容易解释，容量上限通过 token 计数生效；缺点是大量 token 可能同时拥向一个 expert。**Expert choice** 由每个 expert 选择它愿意接收的 token，能让 buffer 更规整，但同一个 token 可能没有 expert 或被多个 expert 选择，必须定义覆盖率和 combine。两者不能混用 loss 或容量公式。本文实验和大多数 serving 路径假设 token choice，并把 expert choice 作为需要单独验证的算法变体。

## 31.3 Capacity factor 与 Overflow：容量公式之外的语义

### 31.3.1 Capacity 的取整和 buffer 账本

给定 microbatch token 数 T、top-k 为 K、expert 数 E，常用容量近似是：

\[
C=\left\lceil\text{capacity\_factor}\times\left\lceil\frac{T K}{E}\right\rceil\right\rceil.
\]

有的实现使用 \(\lceil T/E\rceil\times K\)，有的按每 expert 的 token 上限分别计算；不能把不同公式的实验结果直接比较。`capacity_factor=1.0` 并不保证零 drop，因为 router 分布不是均匀的，且微批量取整会放大偏差。C 还不是完整内存账本：每行 hidden state 的 bytes、route metadata、padding、FP8/INT8 scale、CUDA allocator 对齐和临时 combine buffer 都要加上。

如果 global batch 被切成 N 个 microbatch，容量是按每个 microbatch 还是整批计算，会改变 drop 和峰值显存。按 microbatch 计算，峰值低但小批次噪声大；按 global batch 计算，通信和缓冲更大，可能降低 drop。配置项要把 `capacity_scope` 写进 manifest，而不是只记录 capacity factor。

### 31.3.2 四种 overflow 合同

发生 overflow 时至少有四种合法策略，各自影响质量和可观测性：

- **drop：** 超过 C 的 route 被丢弃；token 仍可走 dense residual 或 identity。必须记录 `dropped_assignments`，不能把它计入 expert compute 的 accepted token。
- **second-choice fallback：** 第一 expert 满时尝试 token 的下一候选；需要防止同一个 token 重复占用同一 expert，且 fallback 会改变实际负载。
- **residual-only：** 保留原 FFN/residual 路径，不再尝试 expert；吞吐稳定但稀疏层的有效容量随负载变化。
- **reroute/retry：** 在新 microbatch 或更高容量的 slow path 重试；可能增加队列和尾延迟，必须有重试上限和幂等 token id。

“drop 率为零”若只统计最终输出，很可能把 fallback 或 residual 隐藏了。至少报告 `requested_assignments`、`accepted_assignments`、`overflow_assignments`、`dropped_tokens`、`residual_tokens` 和每 expert load。对于 Top-2，一个 token 只有一条路径成功也应说明 combine 的 gate 是否重新归一化。

### 31.3.3 训练和推理的不同边界

训练中 token drop 可能影响梯度和 expert specialization；Switch Transformer 通过 auxiliary loss 和 capacity 调参降低 drop，同时保留可扩展的稀疏计算。推理中 drop 更像质量故障：prompt 的关键 token 进入 residual，可能导致输出变化但无显式错误。serving 需按租户和请求类别设 drop budget，例如工具调用或 JSON schema 请求允许的 drop 必须更严格；超过阈值应回退 dense FFN、提高 capacity 或拒绝请求。

推理 batch 的 token 数会动态变化，静态 C 可能在小 batch 中浪费空间，在大 batch 中导致 overflow。连续 batching 需要在 admission 时锁定 batch-local capacity，不能在 all-to-all 进行中改变 C。否则 expert buffer 的 slot 与 combine index 不再一致。

## 31.4 Auxiliary Loss 与 Load Balancing：均衡什么才有用

### 31.4.1 概率均衡和实际选择

令 \(f_e\) 是实际被选择到 expert e 的 token fraction，\(P_e\) 是 router 概率在 token 上的平均值。常见的 Switch/GShard 风格辅助项为：

\[
L_{aux}=E\sum_{e=1}^{E} f_eP_e.
\]

在理想均匀情况下 \(f_e=P_e=1/E\)，\(L_{aux}=1\)。如果一个 expert 独占概率质量，项会变大。实现有时乘上 batch size、top-k 或不同归一化常数；比较 loss 数字前必须查公式和 reduction。aux loss 不应当被解释为主任务 loss 的同量纲指标。

仅优化 \(P_e\) 不能保证实际 f_e 均匀：top-k 截断、capacity drop、tie-break 和 batch size 会改变 f。反过来，只看 f 也可能漏掉 router 概率已经极端尖锐、稍微改变输入就会崩溃的情况。监控应同时输出 `prob_mean`、`selected_fraction`、`load_cv`、最大/最小 load、overflow 和 entropy。

### 31.4.2 Aux loss 的权重和训练稳定性

aux loss 权重过低，主任务会把所有 token 推向少数“强”expert；权重过高，router 被迫均匀，expert specialization 受损。最佳权重取决于 E、K、batch、模型深度和 optimizer。应做小型扫参并记录：主任务 loss、aux loss、expert load、token drop、梯度范数、router entropy 和下游质量。一个 epoch 的平均均衡不代表长序列、代码、工具调用等子分布均衡。

router z-loss 或 logit clipping 也会影响稳定性。logits 过大时 softmax 饱和，梯度变小，所有 token 更容易共享一个 expert；clip 过强则 gate 近似均匀，可能降低 specialization。把 clipping、temperature、aux 权重作为一个联合版本，而不是上线后分别改动。

### 31.4.3 Batch 统计、跨 rank 统计与跨层统计

EP 场景下每个 rank 只看到一部分 token 或 local expert。aux loss 如果用 local batch 计算，rank 间统计有偏；如果做 all-reduce，需定义在 router 之前还是之后同步，且不要让通信成本吞掉稀疏收益。跨层平均也会掩盖单层 hot expert，建议保存 layer/expert 维度的直方图和 top-N 热点。

当 batch 很小，load 的整数噪声本身就大。可以对长窗口 EMA 或多 microbatch 累积统计，但 EMA 不能用于 capacity slot 决策，slot 必须依赖当前 batch。报告中要区分 `online_window`、`optimizer_step` 和 `evaluation_epoch` 的聚合范围。

## 31.5 Expert Parallel：把 token 送到拥有权重的 rank

### 31.5.1 TP、DP、PP 与 EP 的组合

- **Tensor Parallel（TP）：** 一个 expert 的矩阵分片到多个 rank；每个 token 需要在 TP group 内进行矩阵乘或 reduce。优点是单 expert 可放下更大 hidden/intermediate，代价是 expert 内通信。
- **Expert Parallel（EP）：** 不同 rank 拥有不同 expert；token 通过 all-to-all 到拥有目标 expert 的 rank。优点是激活稀疏和专家权重容量扩展，代价是 dispatch/combine 通信。
- **Data Parallel（DP）：** 每个副本拥有同样的专家集，批次切分；需要同步 dense 权重、router 或 optimizer state。
- **Pipeline Parallel（PP）：** MoE layer 随 stage 切分；microbatch 必须按 pipeline schedule 流动，EP group 可能跨 stage 或局部化。

常见的 3D/4D 拓扑是 `world = DP × PP × TP × EP`，但不同框架对维度顺序和 group 构造不同。任何文档或故障报告都应写出 rank mapping，例如 `(dp, pp, tp, ep)` 到 global rank 的函数，以及 expert owner `expert_id // experts_per_rank`。只说“8 卡 EP=4”无法判断 all-to-all 伙伴。

### 31.5.2 Dispatch 与 combine 的两次 all-to-all

逻辑流程是：源 rank 根据 route table 对 hidden state 做 permutation，按目标 EP rank 分桶，执行 all-to-all，目标 rank 按 expert id 和 slot 再排序；expert FFN 完成后，输出按原 token id 排列并执行反向 all-to-all 或等价 reduce-scatter，最后 inverse permutation 和 gate combine。两次通信的字节通常近似为 `2 × accepted_assignments × hidden_bytes`，但 padding、quantization、梯度和 NCCL protocol 可能改变实际值。

必须同时验证 permutation index 和通信 buffer。一个常见 bug 是源端按 token id 排序、目标端按 expert id 解包，导致输出被送回错误 token；另一个 bug 是 overflow 后删掉 route，却没有更新 inverse map，combine 读取未初始化行。CPU toy 的 `send_matrix`、`recv_matrix` 和 conservation test 只验证计数，不证明 GPU buffer layout 正确，因此真实系统还需要小张量 golden test。

### 31.5.3 All-to-all 的通信代价和尾延迟

设 rank 数为 R，源 rank 到目标 rank 的 token 计数为 \(M_{ij}\)，每 token payload 为 B bytes，则 rank i 的发送字节为 \(S_i=B\sum_jM_{ij}\)，接收字节为 \(Q_i=B\sum_jM_{ji}\)。实际一步的通信时间受最大 rank、链路拥塞、collective 算法和 rendezvous 影响，粗略下界是：

\[
T_{a2a}\geq \max_i(S_i,Q_i)/BW_{eff}+T_{latency}.
\]

均值吞吐会隐藏最忙 rank。应报告 max/mean、p95/p99 all-to-all、非零 peer 数、send/recv imbalance、collective timeout 和重试次数。把专家均匀分布在 NVLink 与跨节点 InfiniBand 上也不是同一件事；拓扑感知的 expert placement 可能把通信保留在节点内，却让某些专家权重无法均匀利用。

### 31.5.4 Local experts、shared experts 与 hybrid MoE

有些架构把 shared/dense expert 放在每个 rank，另一些让 shared expert 跨 TP；它们可能绕过一次 all-to-all，但增加本地矩阵乘和参数副本。把 shared expert 输出和 routed expert 输出相加时，需要定义 gate、LayerNorm 和 residual 顺序。部署时应按组件统计显存：`routed_expert_bytes`、`shared_expert_bytes`、`router_bytes`、`dispatch_buffer_bytes` 和 `optimizer_bytes`，否则“模型参数量”不能解释 OOM。

## 31.6 DeepSpeed-MoE：训练栈里的容量与通信契约

DeepSpeed-MoE 将 expert parallel 与数据/模型并行组合，公开文档和源码提供 MoE layer、top-k gate、capacity、专家分组和通信实现。使用它时必须固定 DeepSpeed commit、PyTorch/CUDA/NCCL 版本和 launch 拓扑；参数名字相似不代表语义相同，例如 capacity factor、min capacity、drop token、eval capacity factor 在训练/评估阶段可能分别生效。

验收路径应从 checkpoint 和 layer 构造开始，而不是只看 launch log：

1. 打印 global batch、microbatch、EP/DP/TP group、expert owner 和 capacity 计算结果。
2. 用固定 logits 输入验证 top-k、tie-break、token permutation、combine 权重；在单卡和 EP 多卡结果逐元素对比。
3. 打开/关闭 drop、residual、second-choice 等策略，检查实际 accepted/dropped token 与日志字段一致。
4. 在一个含 hot expert 的 synthetic batch 上验证 aux loss、最大 load、all-to-all bytes 和 wall time；不要用随机均匀输入当压力测试。
5. 让一个 rank 延迟或退出，确认 timeout、错误传播和 checkpoint/optimizer 恢复语义；不能把 collective error 当作正常 drop。

DeepSpeed 的训练优化器、activation checkpoint、ZeRO 与 MoE 专家参数分片的组合会改变内存峰值。ZeRO 级别解决的是 optimizer/参数状态分片，不会自动消除 dispatch buffer；要单独测量 peak allocator 和通信 buffer 生命周期。

## 31.7 Megatron-Core：并行 group、token dispatcher 与 Transformer Engine

Megatron-Core 提供 MoE module、token dispatcher、expert parallel、和 sequence/tensor/pipeline 并行组合。源码审计要从 `MoELayer`、router、token dispatcher、EP group 构造和 fused permutation 路径入手，并锁定 commit；不同版本对 `--expert-model-parallel-size`、`--moe-router-topk`、`--moe-ffn-hidden-size`、`--moe-token-dispatcher-type`、capacity 和 aux loss 的支持会变化。

Megatron 的高性能路径可能使用 fused kernel、grouped GEMM、FP8/Transformer Engine 和多种 dispatcher。CPU toy 的一个 token = 一个均匀 hidden row，无法覆盖：

- grouped GEMM 对每个 expert 的 token count 要求是否被 padding 满足；
- FP8 scale 是 per-tensor、per-channel 还是 per-expert，更新和 all-to-all 是否同步；
- sequence parallel 让 token 在哪一个维度切分，inverse permutation 是否跨 SP/EP group；
- checkpoint 中 expert 参数命名和 global expert id 在 TP/PP/EP 变化时是否可逆。

发布前可把 `--dry-run` 配置和一份小模型 checkpoint 作为 evidence：记录 world size、group ranks、expert count、top-k、capacity、dtype、dispatcher、collective backend，并保存单步 token histogram。任何只贴吞吐数字、不贴这些输入的报告不具备复现性。

## 31.8 vLLM 与 SGLang MoE Serving：从 checkpoint 到动态 batch

### 31.8.1 vLLM 的验收边界

vLLM 的模型执行、scheduler、paged KV 和多 GPU worker 共同决定 MoE serving 的行为。具体 MoE 支持、expert parallel、量化和 kernel 依版本、模型实现与硬件而变；不能把博客中的 flag 视为当前安装包的合同。验证流程是固定 tag，执行 `python -m vllm.entrypoints.openai.api_server --help` 或对应 engine config，记录可用参数，再从 worker/model runner 源码追踪：router 输出在哪里产生、token dispatcher 采用何种 collective、专家权重何时加载、请求取消如何清理临时 buffer。

连续 batching 会让每个 iteration 的 token 数和 capacity 改变。若 runtime 使用静态 expert capacity，需观察 overflow 进入何种 fallback；若使用 padding，需把 padding token 的通信和 GEMM 费用计入吞吐。vLLM 的 paged KV 只管理 attention cache，不会自动管理 MoE dispatch buffer；监控应区分 KV blocks、activation workspace、expert weights 和 all-to-all staging buffers。

负载测试至少包含四组：均匀 prompts、同一模板重复 prompts、故意 hot router 的 synthetic hidden states、以及多租户不同输出长度的混合批。每组报告 TTFT、ITL、E2E p50/p95/p99、tokens/s、accepted/dropped/fallback token、EP all-to-all bytes、GPU memory high-water mark 和 cancellation cleanup。

### 31.8.2 SGLang 的 Radix/调度与 MoE 状态

SGLang 的 RadixAttention、prefix cache、continuous batching 和结构化输出调度会改变 token 到达 MoE layer 的时间分布。前缀命中减少了 prefill token，却可能让 decode token 占比上升；decode 的每步 token 数更小，静态 capacity 更容易出现整数浪费或 hot expert。部署时要把 Radix cache 命中、MoE route histogram、EP 通信和 request queue 一起观测，不能把 prefix cache 命中率当成 MoE 均衡证据。

SGLang 的具体 MoE backend、量化和 EP 支持同样需要 pin commit 并查官方文档/源码。测试结构化输出时，schema mask 可能改变 hidden states 分布，不能只用自然语言 prompt 的 load 结果推断工具调用 workload。请求取消、grammar dead-end 与 all-to-all timeout 的错误传播应有 integration test，确保不会把部分 expert 输出拼成可发送的 JSON。

### 31.8.3 两个 serving 框架的共同验证合同

无论 runtime 选 vLLM 还是 SGLang，至少要保存：镜像 digest、框架 commit、模型 revision、tokenizer hash、GPU 型号、CUDA/NCCL、TP/EP/DP mapping、router dtype、capacity/fallback 配置、并发和数据集摘要。结果 JSON 需包括请求级 route 摘要和 aggregate，不要上传原始 prompt 或敏感输出。若框架不暴露某字段，记录“unavailable”并说明替代信号，不能填零。

## 31.9 量化与 MoE：权重、激活、通信三个不同问题

MoE 有至少三条量化边界：expert 权重、router/logits/hidden activation、以及 dispatch payload。权重 INT4 可能减少显存和带宽，但 grouped GEMM 的 dequant kernel 与专家 token 数相关；hidden INT8/FP8 可减少 all-to-all 字节，却要求两端共享 scale 和版本；router 低精度可能改变 top-k，导致 load/quality 变化。

建议把三个 dtype 分开命名，例如 `expert_weight_dtype`、`dispatch_dtype`、`router_compute_dtype`，并将 scale layout、accumulation dtype、rounding mode 写入 checkpoint manifest。混合 batch 中不同专家使用不同 dtype 时，dispatcher 必须分桶；分桶太多会降低 GEMM occupancy 和通信合并，可能抵消节省的字节。

量化验证不是只看 weight error。应建立三层 golden：

1. 固定 hidden 输入，比较 FP32 router 的 top-k、gate sum、expert load 与量化 router。
2. 固定 route table，比较 expert 输出和 combine 误差，隔离权重量化影响。
3. 端到端固定 seed，比对 logits、任务分数、drop/fallback 和 p99。

若只有第3层变差，不能直接归因于权重；可能是 route 改变、collective reorder、padding 或随机数顺序。每层都需要独立 trace。

## 31.10 训练/推理故障边界与恢复

### 31.10.1 Hot expert 与容量雪崩

Hot expert 可能来自数据分布漂移、router checkpoint 恢复错误、温度/aux loss 改动、tokenizer 版本、或者某个 rank 的 expert 权重损坏。雪崩模式通常是：一个 expert load 上升 → capacity overflow → fallback/residual 增多 → hidden 分布进一步偏移 → router 更集中。仅看平均 tokens/s 会错过它。

检测信号包括 per-layer/per-expert load CV、最大 load/C、overflow rate、router entropy、expert GEMM occupancy、all-to-all max/mean、fallback quality proxy 和队列 p99。触发阈值后可按策略处理：冻结新请求、提高 capacity、降低 top-k/temperature、回退 dense checkpoint、迁移 expert 或隔离异常租户。任何自动动作都要有版本化 feature flag 和 cooldown，避免负载抖动。

### 31.10.2 Collective timeout、rank failure 与 partial output

all-to-all 是同步边界；一个 rank 卡在 expert kernel、OOM 或网络抖动，其他 rank 可能等待。timeout 后必须让整个 microbatch 失败并清理所有参与 rank 的 buffer；只让源 rank 重试会造成重复 token、乱序 combine 或 NCCL communicator 污染。训练中要与 checkpoint/optimizer step 对齐，serving 中要回到请求级可重试边界。

推荐的状态字段是 `collective_id`、`microbatch_id`、`route_checksum`、`send_bytes`、`recv_bytes`、`deadline_ms`、`error_rank`。日志不要只写“all-to-all failed”；需要能判定是 route table 不一致、通信超时、expert kernel error 还是 OOM。恢复后用 route checksum 和模型 revision 验证，不匹配就丢弃临时输出而不是尝试 combine。

### 31.10.3 权重版本与专家迁移

EP 弹性扩缩容或滚动升级时，某些 rank 可能已经加载新 expert 权重，另一些仍是旧 revision。route record 必须包含 `expert_revision` 或全局 model revision；combine 不能把不同 revision 的 K 路输出无提示相加。迁移时应先加载并校验权重 checksum，再原子切换 expert owner；在切换窗口，旧 owner 可继续服务已有 batch，新 batch 使用新 mapping。

## 31.11 可观测性、成本与 SLO：把稀疏性换成可解释指标

MoE 的“激活参数少”不等于成本低。要建立一份 per-request/per-step 账本：

- router：logits latency、top-k latency、entropy、load histogram、aux/fallback/drop；
- communication：dispatch/combine bytes、peer 数、all-to-all p50/p95/p99、max/mean skew、retries；
- expert compute：每 expert token 数、GEMM time、occupancy、padding rows、weight read bytes；
- memory：expert weights、dispatch/receive/combine buffer、allocator high-water mark、KV blocks；
- service：queue time、TTFT、ITL、E2E、tokens/s、cancel/error、quality proxy。

成本模型可写成：

\[
T_{step}=T_{router}+T_{permute}+T_{a2a}^{dispatch}+T_{expert}+T_{a2a}^{combine}+T_{unpermute},
\]

其中每项都应有观测值，而不是用平均值填充。`T_expert` 可能因 padding 和 hot expert 远大于按 accepted token 估算的理论 FLOPs；`T_a2a` 受最忙 rank 而非总字节决定。容量 factor 增加后，drop 下降但 buffer 和通信上界上升，应该画 throughput/quality/p99 Pareto，而不是只选 drop 最低的点。

质量代理也需分层。离线可比较 logits KL、perplexity、任务分数；在线可抽样与 dense/FP16 shadow 对比、工具调用成功率、schema parse error、拒答率和业务约束错误。shadow 只取小样本，并记录 denominator、采样率、模型 revision；否则“0 次失败”可能只是没有请求进入 shadow。

## 31.12 CPU-only Lab：复现路由、容量和 All-to-All 账本

实验脚本 `labs/ch31_moe_parallelism_lab.py` 生成 deterministic logits，并实现：

- stable softmax 与按 `(-logit, expert_id)` 的 top-k；
- capacity factor 取整、drop、second-choice fallback、residual 标志；
- expert probability、selected fraction 和 \(E\sum f_eP_e\) auxiliary proxy；
- token 到 EP rank 的 send/recv matrix、bytes、非零 peer rounds 和 imbalance；
- 均匀与 hot-expert 输入的通信/延迟 proxy JSON。

运行示例：

```bash
python3 labs/ch31_moe_parallelism_lab.py \
  --seed 31 --tokens 256 --experts 8 --ranks 4 --top-k 2 \
  --capacity-factor 1.25 --overflow-policy drop --hidden-bytes 4096 \
  --output reports/ch31-moe-routing.json
```

结果中的 `routing.capacity` 是每 expert 的 toy slot 上限；`routing.expert_load` 是 accepted assignment；`summary.drop_rate` 以 token 为分母，不是 assignment。`all_to_all.send_matrix[i][j]` 表示源 rank i 发给目标 rank j 的 assignment 数，`bytes_by_src` 乘了 hidden bytes。`latency_proxy_ms` 只用于比较配置方向，不能当 GPU/NCCL benchmark。

建议至少跑三组：

1. `top-k=1, capacity-factor=1.0`，观察简单 token choice 的 overflow；
2. `top-k=2, capacity-factor=1.25`，对比 drop 和通信翻倍方向；
3. 给 expert 0 加 `--hot-expert 0 --hot-bias 8`，观察 fallback/drop、最大 load 和 all-to-all skew。

测试文件固定了六个合同：softmax 归一化、capacity 上界、second-choice 不重复 expert、send/recv assignment conservation、seed reproducibility、CLI JSON schema。它们不会证明 CUDA kernel 或 NCCL 正确性，真实部署仍需单步小张量 golden、跨 rank collective test、故障注入和端到端质量集。

## 31.13 发布前 Checklist：从 toy 到生产的最小证据

### 路由与质量

- [ ] 固定模型/tokenizer revision、router dtype、temperature、top-k、tie-break 和 aux loss reduction。
- [ ] 在均匀、长上下文、代码、工具调用和 hot-expert 数据上记录 per-layer/per-expert load。
- [ ] 明确 capacity scope、overflow policy、residual/second-choice、drop budget 和回退行为。
- [ ] 运行 FP32/低精度 router golden，比较 top-k、gate sum、load、logits 和业务质量。

### EP 与通信

- [ ] 保存 DP/PP/TP/EP rank mapping、expert owner、dispatcher 类型和 collective backend。
- [ ] 验证 dispatch/combine permutation、send/recv matrix、route checksum 和 slot 回收。
- [ ] 记录 all-to-all bytes、max/mean skew、p95/p99、timeout、重试和失败 rank。
- [ ] 在单节点、跨节点、不同 NCCL protocol 和网络拥塞下执行小规模故障注入。

### 框架与运行时

- [ ] 对 DeepSpeed-MoE、Megatron-Core、vLLM 或 SGLang 锁定版本/commit，保存实际 `--help`/config。
- [ ] 验证专家 checkpoint 命名、quantization scale、grouped GEMM、activation workspace 和滚动升级。
- [ ] 在 continuous batching、prefix cache、取消、schema mask 和多租户负载下测 TTFT/ITL/E2E。
- [ ] 明确缺少指标时的替代信号，禁止把 unavailable 填成零。

### 故障与回滚

- [ ] hot expert、OOM、collective timeout、rank crash、权重 checksum mismatch 都能触发请求级或作业级安全失败。
- [ ] 临时 dispatch/receive/combine buffer 在取消和异常后可证明释放，旧 route 不会污染新 batch。
- [ ] 回退到 dense、提高 capacity、迁移 expert 或降级 top-k 的动作有 feature flag、cooldown 和审计日志。
- [ ] evidence manifest 包含来源 URL、版本、命令、报告路径和明确的 toy/non-production 边界。

## 31.14 小结：MoE 的收益由最窄的边界决定

MoE 用稀疏激活换参数容量，但每个 token 都要付 router、容量、置换、all-to-all、expert compute 和 combine 的协议成本。Top-k 选择决定计算和通信上界；capacity factor 决定 overflow 的概率和显存；aux loss 改变 router 的偏好却不自动保证实际均衡；EP 决定 token 是否跨节点搬运；DeepSpeed-MoE 和 Megatron-Core 负责训练时的 group、dispatcher 与 checkpoint 组合，vLLM 和 SGLang 还要把动态 batching、KV/prefix cache、取消和多租户加入运行时边界。

可部署的 MoE 不是“把 dense FFN 替换成 MoE layer”就完成。最低要求是：route record 可重放、capacity/overflow 可解释、dispatch/combine 可验证、collective 失败可隔离、专家版本可校验、指标能分解到 rank/expert/request，且在质量和 p99 越界时能安全回退。CPU toy 只能帮助我们检查公式和账本；真正上线前必须用固定版本、真实拓扑、真实 checkpoint 和故障注入补齐证据。

## 31.15 端到端算例：把一个 microbatch 的数字对上

假设一个 microbatch 有 128 个 token、8 个 expert、top-2、capacity factor 为 1.25，hidden payload 为 4096 bytes，4 个 EP rank，每个 rank 负责 2 个 expert。先算请求的逻辑 assignment 数：\(T\times K=256\)。理想均匀平均每个 expert 32 个 assignment；toy lab 的容量是 \(\lceil1.25\times\lceil256/8\rceil\rceil=40\)。如果 router 产生的最大 expert load 是 57，至少 17 个 assignment 会在该 expert 溢出；它们是否最终丢弃，取决于 fallback/residual policy，不能仅从容量数推导。

若 accepted assignment 为 240，则 dispatch payload 的逻辑下界是 \(240\times4096=983{,}040\) bytes，combine 还要再读写一遍；若实现使用 FP8 hidden 加每 128 个值一组的 scale，payload 和 metadata 需要分别计账。源 rank 的 send matrix 可能是 `[31, 29, 30, 30]`，另一个输入分布可能是 `[18, 62, 17, 15]`；总字节几乎相同，但第二个 rank 负责的最大 peer/专家显著更忙，尾延迟会由它决定。因而“all-to-all 总字节不变”不是“延迟不变”。

把以上数字写成报告时，至少同时列出：`requested=256`、`accepted=240`、`overflow=31`、`dropped=16`、`residual=0`（如果 fallback 接受了 15 个）、每 expert load、capacity=40、aux loss、send/recv matrix、max/mean skew 和通信 p99。若只写 `drop_rate=6.25%`，读者不知道是 16 个 token 还是 16 条 assignment，也不知道 K 路中有多少路径成功。

### 31.15.1 训练梯度和 serving 计数不要混写

训练中的一个 token 可能在 K 个 expert 都产生梯度，gate 和 router 也参与反向；serving 的计数常以最终 accepted assignment 或输出 token 为主。训练报告的 FLOPs、通信和显存要包括反向 dispatch/combine、activation 保存与 optimizer state；推理报告则要包括权重加载、KV、batch queue 和 cancellation。将训练阶段的 `tokens/s` 与 serving 的 `output tokens/s` 直接比较，会把 prompt/prefill、decode、drop 和 padding 混在一起。

### 31.15.2 最小 golden tensor

即使使用成熟框架，也建议保留一个 2 个 token、4 个 expert、top-2 的 golden tensor：手工指定 logits，其中一个 token 有相同的两个最高值以测试 tie-break，另一个 token 让第一 expert 超过容量。保存 route table、slot、gate、send/recv index、expert 输出和 combine 结果。这个样本应在单卡、TP/EP、多 dtype、checkpoint reload、框架升级后逐字段对比。它的作用不是 benchmark，而是把“路由/置换/合并”从复杂的性能路径中分离出来；一旦 golden 失败，先不要用吞吐优化掩盖正确性问题。

### 31.15.3 用小实验校准大系统的假设

toy lab 可以回答三个可迁移的问题：容量取整是否有清晰语义、overflow 计数是否守恒、rank skew 是否随 hot expert 增加。它不能回答 expert GEMM 在 H100、MI300 或其他 GPU 上的 occupancy，也不能模拟 NCCL/ROCm 的协议选择、节点拓扑、网络拥塞和故障恢复。生产实验应沿着同一字段名扩展：把 toy 的 `expert_load` 对应到框架的真实计数，把 `send_matrix` 对应到 collective profiler，把 `latency_proxy_ms` 替换为分段 wall time，再在报告中并列 toy 与真实结果，避免把简化数字误写成硬件承诺。
