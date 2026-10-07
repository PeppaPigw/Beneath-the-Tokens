---
id: ch30-kv-compression-and-tiering
title: KV Cache 压缩与冷热分层：在质量、容量和延迟边界上做在线决策
slug: /chapters/30-kv-compression-and-tiering
description: 从逐组量化、低秩与稀疏表示到 HBM/DRAM/SSD 冷热分层，建立可观测的 admission、eviction 与质量—容量—延迟边界
sidebar_position: 30
level: advanced
prerequisites:
  - ch15-inference-execution
  - ch16-model-serving-system
  - ch19-inference-optimization-accelerator-stack
  - ch20-observability-debugging-incident-response
  - ch21-ai-reliability-engineering
  - ch27-kv-transfer-and-connectors
  - ch28-vllm-and-sglang-serving
  - ch29-speculative-and-structured-decoding
learning_objectives:
  - 能从层数、KV heads、head dimension、token 数和 dtype 推导 KV 字节，并解释页尾碎片和量化元数据为何改变容量收益
  - 能比较 per-tensor、per-channel、per-token、per-group 的 INT8/INT4/FP8 量化，计算重构误差并识别 outlier、scale 和校准数据的边界
  - 能用低秩、稀疏和混合表示分析 payload、索引、基底与重构误差，区分存储压缩和注意力计算加速
  - 能设计 HBM/DRAM/CXL/SSD/远端缓存的冷热分层协议，定义 admission、promotion、demotion、eviction、lease 和回收语义
  - 能把质量损失、容量、解压延迟、带宽、命中率、p95/p99 和 recompute 放到同一 Pareto 前沿，而不是只报告压缩比
  - 能运行 CPU-only toy lab，复现实验数据、在线策略和失败边界，并写出生产系统的证据与回滚合同
estimated_hours: 38
hardware: CPU-only lab required; GPU/CXL/SSD optional and cost-bounded
risk_level: L4
last_verified: 2026-10-07
---

# 第30章　KV Cache 压缩与冷热分层：在质量、容量和延迟边界上做在线决策

> KV cache 的问题不是“能不能把 FP16 变成 INT4”，而是**哪些 token 可以在什么时间、什么介质、以什么误差预算被压缩、晋升、降级或丢弃**。一次请求可能同时拥有 HBM 中的精确热页、DRAM 中的量化温页、SSD 或远端缓存中的冷页，以及因无法恢复而需要重新计算的页。压缩改变了字节数，也改变了解压 kernel、attention 数值、命中收益、迁移带宽和尾延迟。本章把它作为一个在线协议和容量决策问题来处理。

本章与第27章的边界很明确：第27章关心 KV 页怎样通过 connector、RDMA 或 P/D 解耦链路安全到达另一端；这里假定传输协议已经给出一页**可读的 payload**，研究这页在到达前后如何压缩、在哪个层级保存、何时被接受和何时被淘汰。本章也不重复第29章的候选验证；只讨论 speculative 或 grammar 读取压缩 KV 时需要增加的数值和版本合同。

## 30.1 从“显存不够”改写成四个可测问题

一个 serving 团队说“KV 占满了显存”，至少可能指四件不同的事：

1. **物理容量不足：** 活跃请求的逻辑 token 数乘以每 token 的 K、V 字节超过 HBM 可用页。
2. **表示浪费：** 页中很多通道的动态范围很小，却为少量 outlier 预留了 FP16 精度；页尾 padding、scale 和索引甚至可能抵消低 bit 的收益。
3. **热度错配：** 长上下文的前缀很久不被读取，却和下一步要反复访问的 decode 页占用同一种昂贵介质。
4. **延迟预算不允许：** 压缩后虽然多放了请求，但每个 attention step 都要解压、搬运或重算，p99 TTFT/ITL 反而越过 SLO。

因此第一张图不是“INT4 比 FP16 小四倍”，而是每个页的账本：`logical_tokens`、`resident_bytes`、`metadata_bytes`、`dequant_ms`、`recompute_ms`、`quality_loss`、`hit_probability` 和 `last_access`。没有这些字段，压缩比只是营销数字。

### 30.1.1 先写质量与回滚合同

在改 dtype 或 cache policy 之前，把以下合同写进测试和发布单：

- **数值合同：** 在固定模型 revision、tokenizer、RoPE、attention kernel、随机种子和校准集下，压缩 KV 与 FP16 baseline 的 logits 误差、perplexity 或任务分数阈值分别是多少；阈值按短/长上下文和 prompt 类别分层。
- **容量合同：** `resident_bytes` 必须按 payload、scale、zero-point、索引、页对齐和 allocator overhead 统计；不能用理论 bit 数代替实际分配。
- **延迟合同：** 记录压缩、解压、页迁移和重算的 TTFT、ITL、TPOT、p95/p99；命中和未命中分别统计。
- **在线策略合同：** admission、promotion、demotion、eviction 和 recompute 的状态迁移幂等；取消请求后不会留下悬挂 lease 或被其他租户复用的页。
- **回滚合同：** 每个压缩页保留 format/version/checksum；一旦误差超限、解压失败或指标异常，可以按页回退到原精度或重算，而不是清空整个服务池。

这些合同把“质量—容量—延迟边界”变成可审计的证据。一个策略若只节省容量，却没有明确谁负责重算和谁承担质量损失，不能称为可部署优化。

## 30.2 统一记账：一页 KV 到底有多少字节

设模型有 (L) 层，(H_{kv}) 个 KV heads，每个 head 维度 (D)，页中包含 (T) 个 token，元素大小为 (b) bytes。K 与 V 各存一份，因此未压缩 payload 是：


\[
B_{\mathrm{fp}}(T)=2\times L\times T\times H_{kv}\times D\times b.
\]

这里的 2 不是 batch size，也不是 attention 的缩放因子；它是 Key 和 Value 两个张量。若模型使用 GQA/MQA，(H_{kv}) 小于 query heads，不能把 query heads 代入，否则会高估或低估容量。

### 30.2.1 量化页的实际公式

对 bit 宽度 (q)、group size (G)，每组保存 scale (s) bytes 和可选 zero-point (z) bytes。忽略页头时：

\[
N=2LTH_{kv}D,qquad
B_{q}=\left\lceil\frac{Nq}{8}\right\rceil+
\left\lceil\frac{N}{G}\right\rceil(s+z).
\]

如果稀疏表示需要每个非零值的坐标，索引项 (B_{idx}) 还要加上；如果低秩表示用 (r) 个系数替换 (D) 个值，系数 payload 近似乘 (r/D)，但共享基底 (U,V) 的存储、版本和加载时间必须单列。压缩页的 allocator 常按 128B、256B 或更大的 block 对齐，故真实分配是 `ceil(B_q / alignment) * alignment`。

举例：(L=32,H_{kv}=8,D=128,T=32,b=2) 时，一页 FP16 payload 为 4,194,304 bytes。INT4 每 64 个值保存 2-byte scale，理论 payload 加 metadata 约 1,179,648 bytes，约 3.56 倍而不是四倍；若每个非零值再加索引，稀疏页可能反而变大。读者应使用本章 lab 的 `compressed_bytes` 把页尾、group metadata 和索引代价算进去。

### 30.2.2 页粒度是策略的一部分

页越小，admission 和 eviction 越精确，但 scale/索引/页头的比例越大，DMA 和 kernel launch 越频繁；页越大，压缩效率通常更好，却容易把冷 token 与热 token 绑在一起。建议把 `page_tokens` 当成实验变量，至少扫描 8、16、32、64、128，并记录：

- 页尾 padding bytes / payload bytes；
- 每次命中的平均有效 token 数；
- promotion/demotion 的 bytes 与次数；
- HBM 的碎片和 allocator high-water mark；
- 解压 kernel 的 launch 数和 batch 内不同格式的数量。

第27章中的 connector 页序列和 checksum 仍然适用，但压缩页的 digest 应覆盖 `format_id、model_revision、token_start、token_count、scale_layout、payload`，不能只 hash 逻辑 token 范围。

## 30.3 量化不是一个按钮：误差从哪里进入 attention

### 30.3.1 对称与非对称映射

最简单的对称量化把值 (x) 映射为整数 (q=\operatorname{round}(x/s))，再用 (hat{x}=qs) 还原。每组 scale 常取 `max(abs(x))/qmax`。非对称量化额外保存 zero-point (z)：

\[
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max}),
\quad \hat{x}=(q-z)s.
\]

对称映射的元数据更少、kernel 更简单；当一组值明显偏离零时，非对称映射可能降低均方误差。选择不是理论偏好，而要按 K、V、层、head 和 workload 测量。K 的误差会改变 (qK^T) 的注意力分数，V 的误差会直接污染加权和；同样的 MSE 不代表对输出的影响相同。

### 30.3.2 granularity：per-tensor、per-channel、per-token、per-group

- **Per-tensor：** 整个层共用 scale，metadata 最小，但 outlier 会压缩普通值的有效码字。
- **Per-channel/head：** 每个通道或 head 一个 scale，能处理不同 head 的动态范围，读取时要增加广播和索引。
- **Per-token：** 每个 token 一组 scale，适合跨 token 方差大但 metadata 开销更高；页边界和 beam 分支要保留对应 scale。
- **Per-group：** 每 (G) 个相邻值共享 scale，是容量、误差和 kernel 复杂度的常见折中；G 不是越小越好，极小 G 会让 metadata 变成主项。

对比时必须固定 tokenizer、序列长度和 calibration distribution。把 per-token INT4 在短 prompt 上的误差结果宣传为长上下文 decode 的保证，是典型的外推错误。

### 30.3.3 outlier 与通道重排

少数 outlier 可能把一个 group 的 scale 拉大，使大多数值只使用很少码字。常见做法包括：把 outlier 通道保留 FP16、单独存 residual、进行 channel permutation，或者先做旋转/平滑再量化。工程要回答三个问题：

1. outlier mask 是否随模型 revision 和层索引版本化？
2. mask 的查找是否会让每个 attention step 多一次 gather，导致 p99 变差？
3. outlier 页和普通页是否能在同一个 batch/kernel 中合并？

只报告“99.9% 值能 INT4”不够；剩下 0.1% 可能恰好落在对 logits 最敏感的 head。应记录每层、每 head、K/V 分开的 outlier count、最大值、scale 分布和输出误差。

### 30.3.4 FP8 与整数不是同一条曲线

FP8（例如 E4M3 或 E5M2）有指数和尾数，动态范围与舍入行为不同于 INT8/INT4；它往往减少 scale 依赖，却不能自动解决 outlier。不同硬件对 FP8 accumulation、转换和 denormal 的支持不同。比较 FP8 和 INT8 时要锁定：输入/输出 dtype、accumulation dtype、kernel 实现、是否在 HBM 中压缩后才转回，以及 scale 是否 per-tensor 或 per-channel。没有这些信息，`bits=8` 不是同一个格式。

## 30.4 校准与在线误差：离线分数不等于请求质量

### 30.4.1 校准集要覆盖访问分布

量化 scale 可以在离线样本上求得，但在线 KV 的分布取决于语言、代码、工具调用、上下文长度和 system prompt。校准集至少分为短/中/长上下文、自然语言/代码/结构化 JSON、不同租户模板，并记录每组的 token 数、层和 head coverage。把用户的原始 prompt 写入长期校准 artifact 还会引入隐私风险；应保留聚合统计或脱敏样本。

### 30.4.2 在线监控信号

无法直接为每个请求计算真实 perplexity 时，可用便宜的代理信号：

- 解压后的 K/V 与重算采样页的 cosine、max-abs、MSE；
- attention logits 的 KL 或 top-k overlap；
- 命中压缩页与 FP16 shadow 页在采样 token 上的差异率；
- 按请求类别聚合的 refusal、schema parse error、工具调用失败；
- 回退到 FP16/recompute 的比率和原因。

shadow 页要抽样而非全量复制，否则容量和带宽账本会被监控本身污染。每个指标都需要采样率、版本和置信区间；没有 denominator 的“回退次数”无法解释。

### 30.4.3 误差预算的分配

可以把请求级误差预算 (E_{req}) 分给层、页或 tier：

\[
E_{req}\geq\sum_{\ell,t}w_{\ell,t}e_{\ell,t},
\]

其中 (e_{\ell,t}) 是某层某页的重构误差，(w) 由 attention 权重、位置或业务敏感度给出。早期层、sink token、system prompt 和工具参数可能拥有不同权重。在线策略不应仅按页龄淘汰；更合理的是按“预计命中收益 / 字节 / 误差代价”排序，并对关键 prefix 设置保护位。

## 30.5 低秩表示：节省 payload 也可能增加新状态

### 30.5.1 低秩的存储模型

若一组 K 或 V 矩阵 (X\in\mathbb{R}^{T\times D}) 近似为 (AB^T)，其中 (A\in\mathbb{R}^{T\times r})、(B\in\mathbb{R}^{D\times r})，系数 payload 从 (TD) 降到 (r(T+D))。当 (r\ll\min(T,D)) 时有收益，但 B 是共享基底，必须写入模型/层/head/version metadata，且 attention 读取时需要在合适的位置重构或使用低秩乘法。若每个页都有一份 B，rank reduction 可能被基底重复抵消。

SVD tail energy 是离线误差的一个指标：

\[
e_{rank}=\frac{\sum_{i>r}\sigma_i^2}{\sum_i\sigma_i^2}.
\]

它不是 logits 误差证明。attention 的非线性 softmax 会放大某些方向，RoPE 后的相位也会改变奇异值分布。lab 的 `low_rank_error` 只展示 tail energy，不声称它等价于模型质量。

### 30.5.2 共享基底、分层基底与版本

三种工程选择各有代价：

- **模型级基底：** metadata 少，kernel 稳定；不同请求的 KV 统计若变化大，重构误差不可控。
- **层/head 级基底：** 误差更小，但基底数量和加载路径增加；tensor parallel 必须定义 owner 和广播。
- **页级基底：** 适应性最好，metadata 和小矩阵乘法开销也最大；页迁移时需一起传输和校验。

每份基底都要有 `basis_id`、dtype、rank、训练/校准版本和 checksum。缓存命中只能在这些字段匹配时复用。不能因为 token prefix 相同就把不同 rank 或不同 basis 的页拼接到同一 attention batch。

## 30.6 稀疏表示：零值、索引与 attention 访问模式

### 30.6.1 非结构化与结构化稀疏

非结构化稀疏保留绝对值最大的元素，重构简单但索引不规则；block/2:4 等结构化稀疏牺牲部分可压缩性，却能让 GPU kernel 更容易合并加载。KV cache 的 token 访问通常按 head、dimension 和 page 组织，稀疏坐标若与 kernel 的读取顺序不一致，会把节省的 payload 换成随机 gather。

容量公式需要加入索引：若保留比例 (k)，每个非零项使用 (q) bit 和 (i) bit 坐标，则近似：

\[
B_{sparse}\approx N k(q/8+i/8)+B_{mask}+B_{scale}.
\]

当页很小或 (i) 较大时，(B_{sparse}\) 可能大于密集 INT4。必须用真实 page size、block shape 和 allocator alignment 计算。

### 30.6.2 H2O、Scissorhands 与“重要 token”边界

Heavy-Hitter Oracle（H2O）和 Scissorhands 等研究提出根据 attention 或历史使用保留重要 token；SnapKV、PyramidKV 等方法利用 prompt 末端或层级差异压缩/选择上下文。它们提供可测试的启发式，不是生产系统的普适证明。尤其要注意：训练/论文里的 attention 观测窗口、模型版本、任务和序列长度决定了重要性；把一个工作负载的保留比例直接写成全局 cache policy，会造成隐性质量回归。

在在线系统中，保留 token 需要定义“不可删”的语义：system prompt、工具 schema、最近用户消息、KV sink、正在生成的 suffix 和任何受合规政策保护的片段。策略只删除可重算且未被其他分支引用的页，并在 trace 中记录 `protected_reason`。

## 30.7 混合压缩：按层、按页、按热度选择格式

“一种 dtype 适用于所有页”通常不是最优。一个可解释的 tier profile 可能是：

| 层级 | 典型格式 | 目标 | 主要代价 |
| --- | --- | --- | --- |
| HBM 热页 | FP16/BF16 或 FP8 | 最低 ITL，保护近期 decode | 容量最贵，淘汰压力大 |
| DRAM 温页 | INT8，per-group | 保留可晋升上下文 | 拷贝/解压和 NUMA 亲和性 |
| SSD/远端冷页 | INT4/低秩混合 | 大容量、可重用 prefix | 首次命中长尾、写放大 |
| 不保留页 | 无 | 省容量 | 必须重算，TTFT 变差 |

格式切换必须携带 `format_id`，并且每个 batch 的 attention kernel 能处理混合格式，或在 admission 时按格式分桶。否则每页压缩得很好，batch 仍因 kernel 分歧而串行化。第27章的 transfer connector 看到的应是完整页记录：`token_start/end、format_id、metadata_bytes、checksum、lease`。

压缩和迁移的顺序也有选择：先在 HBM 压成 INT8 再发到 DRAM，减少网络字节但增加 CPU/GPU kernel；先传 FP16 再在接收端压缩，降低发送端争用但占用链路。应按总时间：

\[
T_{move}=T_{compress}+B_{compressed}/BW+T_{decompress},
\]

与重算成本 (T_{recompute}) 比较，而不是只比较 (B)。

## 30.8 冷热分层协议：状态机比缓存名称更重要

### 30.8.1 五个状态和合法迁移

建议把页状态明确写成：`ABSENT → COLD → WARM → HOT → PROTECTED`，另有 `COMPRESSING、IN_FLIGHT、EVICTING、RECOMPUTING、CORRUPT` 等瞬态。合法迁移示例：

- 首次计算完成后，按 admission 进入 COLD（SSD/远端 INT4）或直接 WARM；
- COLD 连续命中且收益超过阈值，异步晋升 WARM（DRAM INT8）；
- WARM 在 decode 窗口高频命中时晋升 HOT（HBM FP16）；
- HOT 长时间未命中时降级为 WARM，确认写入和 checksum 后才能释放 HBM；
- 任何 format/version/checksum 不匹配进入 CORRUPT，禁止“尝试解压后继续”；
- 取消或租约过期先标记 EVICTING，等读引用数归零，再释放物理页。

每次迁移带 `generation` 和 `lease_id`。消费方只接受自己声明的 generation；晚到的旧写入不能覆盖新页。迁移完成并不等于 scheduler 可读，必须有 `READY`/`COMMITTED` 可见性事件。

### 30.8.2 HBM、DRAM、CXL、SSD 的不同边界

- **HBM：** 带宽和并发最优；碎片、显存水位和 kernel 可见性是主要限制。
- **DRAM：** 容量大但 NUMA、PCIe、页锁定和 CPU 解压会影响尾延迟；要记录 socket 与 GPU 亲和性。
- **CXL memory：** 访问模型取决于平台和一致性配置，不能把它当“更大的 DRAM”而省略拓扑/带宽测量。
- **SSD：** 持久化和容量好，但随机读取、写放大、寿命和加密/租户隔离成本高；冷页应按大块批量迁移并有写回预算。
- **远端 cache：** 复用范围大，网络抖动和租约安全成为边界；第27章的 connector 状态机、checksum 和 backpressure 仍是前置条件。

分层协议要为每一层记录独立的 `capacity、used、bandwidth、latency、error_rate、encryption、eviction_cost`。一个“缓存命中”如果发生在远端 SSD，并不等价于 HBM 命中。

### 30.8.3 读路径与写路径

读路径应先查本地 HOT/WARM，再查 COLD/远端，最后决定重算；查找结果必须包括 format 和 checksum，而不只是一个 pointer。写路径通常是：`compute → optional quantize → checksum → persist → publish index`。若先发布 index 再写 payload，读者可能拿到半页。异步压缩完成前可以保留 FP16 页，但要把两份 bytes 都计入 high-water mark，不能在容量报表中提前扣除。

## 30.9 Admission：不是所有页都值得缓存

### 30.9.1 从“有空间就放”到效用分数

在线 admission 可以估计：

\[
U(p)=\frac{\hat{P}_{hit}(p)\cdot C_{recompute}(p)-C_{read}(p)-\lambda E(p)}{B(p)},
\]

其中 (hat{P}_{hit}) 是未来命中概率，(C_{recompute}) 是重算成本，(C_{read}) 是从当前 tier 读取和解压成本，(E) 是质量误差，(lambda) 是业务权重，(B) 是该格式占用字节。只有 (U) 超过阈值才放入目标 tier。估计不需要精确到 token 级，但必须记录输入窗口、历史命中、租户权重和误差版本。

新页的第一次访问没有历史，常见策略是 probation：先进入小的 COLD 区，二次命中才晋升；这比一律占用 HBM 更能避免扫描型流量污染 cache。TinyLFU、ARC、2Q 等频率/时间混合策略可作为基线，但要把压缩格式和质量惩罚纳入 score。

### 30.9.2 租户公平与预算

全局最优可能让一个长上下文租户填满所有 DRAM。应给 tenant、模型 revision、schema/业务类别设置 soft/hard budget，并在 admission 时扣除 metadata、索引和预取 bytes。公平不是只按请求数分配：一个请求的 prefix 可能被多次复用，另一个请求是一次性上传。可用 `tenant_bytes、tenant_hit_value、tenant_eviction_count、tenant_quality_loss` 做按效用的配额。

### 30.9.3 预取的反例

从 prefix 预测下一页并异步预取，可能降低命中延迟；但预取页也会挤走真正的热页。预取必须有 cancellation、bytes budget 和 stale 版本检查，并把未被访问的预取视为负收益。报告中单列 `prefetch_requested、prefetch_used、prefetch_wasted_bytes`，不要把预取命中混入自然命中率。

## 30.10 Eviction 与 demotion：回收不是删除按钮

### 30.10.1 代价感知的 victim 选择

LRU 简洁，但它不认识重算成本、压缩误差和页大小。一个可解释的 victim score 是：

\[
V(p)=\frac{\alpha\cdot age(p)+\beta\cdot(1/freq(p))
+\gamma\cdot B(p)}{C_{recompute}(p)+\epsilon}
+\delta E(p).
\]

高 (V) 先被降级或驱逐。对受保护的 system/tool 页，设置 `protected=true`；对有未提交引用的页，设置 `pin_count>0`，只有引用归零才允许进入 EVICTING。实际系统可以用 CLOCK/segmented-LRU 近似，不应为了精确排序把 scheduler 变成全量锁竞争。

### 30.10.2 demotion 优先于丢弃

HBM 满时，优先把可重构的 HOT 页压成 INT8/INT4 写入 DRAM，成功校验后释放 HBM。DRAM 满时再写入 SSD 或远端；如果写入成本大于重算成本，就直接丢弃。demotion 的双写窗口要在容量中算两份：旧页不能在新页 `COMMITTED` 前释放。

### 30.10.3 取消、重复与崩溃恢复

Eviction 与 decode 可以并发。取消请求后，引用计数要递减；若异步压缩线程仍持有 buffer，不能复用其地址。崩溃恢复读取 SSD 页时先校验 schema、model revision、format、digest，再决定重算；“读失败就当全零”会制造静默质量错误。重启后应清空未完成 `COMPRESSING/IN_FLIGHT` 状态，只恢复已提交 manifest。

## 30.11 质量—容量—延迟的 Pareto 边界

### 30.11.1 三轴而非单一压缩比

给定策略 (s)，可测向量为：

\[
Q(s)=\text{质量损失},\quad C(s)=\text{resident bytes},\quad
L(s)=\text{p95/p99 latency}.
\]

策略 (s_1) 支配 (s_2) 当且仅当在至少一轴严格更好、其余不差。保留非支配点就得到 Pareto 前沿。一个 INT4 策略若质量损失略高但把 p99 从 800ms 降到 300ms，可能比 FP16 更适合特定 SLO；另一个 INT4 若让 schema parse error 增加且 p99 无改善，就应被淘汰。

### 30.11.2 延迟拆解

请求的 p99 不应只写“cache hit latency”。建议拆成：

`queue + lookup + transfer + decompress + attention + recompute + serialize + network_flush`。

对每个组件记录命中/未命中、tier、format、page_tokens、batch size 和并发。压缩收益可能降低 transfer，却增加 dequant；分层收益可能降低 HBM pressure，却增加 DRAM/SSD queue。只有事件时间戳对齐，才能知道优化是否真正移动了瓶颈。

### 30.11.3 容量曲线的拐点

扫描 HBM 容量或 admission threshold 时，画出：`hit_rate、p95、quality_loss、evictions、recompute_tokens、resident_bytes`。常见形状是：容量增大初期 p95 快速下降；超过工作集热点后收益变小；某个阈值后更多页进 HBM 但质量和延迟几乎不变。不要把最大容量点当目标，应选择满足 SLO 且留有故障/发布 headroom 的拐点。

## 30.12 与 continuous batching、speculative 和 structured decoding 的接口

### 30.12.1 batching 的格式分歧

Continuous batching 会把不同请求、不同 tier 和不同格式放在同一步。若 attention kernel 只能处理一种 dtype，scheduler 需要按 format 分桶，可能破坏公平和 batch size；若 kernel 支持混合格式，必须定义每个 page 的 dequant stream、scale pointer 和 error flag。记录 `batch_format_groups`、每组 token 数和等待时间，避免把 kernel 分歧隐藏在 ITL p99 中。

### 30.12.2 speculative 的暂存 KV

第29章的 tentative branch 不能直接写入长期冷层。draft KV 应标记 `tentative=true`，只有 target verification 接受后才可压缩并发布；被拒绝的 branch 及其 scale、basis、索引都要回收。若 target 需要读取压缩 draft，必须确保 p/q 在同一逻辑页和同一格式版本，否则接受率下降不能归因于模型质量。

### 30.12.3 grammar/JSON 的敏感页

grammar 状态、工具 schema 和 system prompt 的 KV 往往比普通上下文更敏感，也可能影响后续 token 的合法集合。可以给这些页设置更高精度和 `protected_reason=grammar_or_policy`，而不是为了容量盲目 INT4。若解压误差导致 parser dead-end，应记录 `grammar_state_digest` 和回退原因，不把错误吞成普通 cache miss。

### 30.12.4 位置编码与页拼接

RoPE、ALiBi 或位置缩放使 token position 成为 KV 语义的一部分。压缩/低秩页的 `token_start`、position transform revision 必须随 payload 保存；把相同内容在不同 position 的页合并会改变 attention。页晋升和 demotion 不得重排 token；拼接前检查连续 position、layer/head shape 和 generation。

## 30.13 安全、隐私与供应链边界

KV 可能包含完整 prompt、工具参数和个人数据。压缩并不会降低敏感性，反而可能把数据复制到更多 tier。远端/SSD 层要有租户隔离、加密、密钥轮换、最小访问权限和可审计删除。共享 prefix 只能在授权的身份、model revision、prompt policy 和 cache namespace 下命中。

压缩 kernel、量化库和自定义 CUDA extension 属于供应链输入：锁定来源、版本和 checksum；在升级时运行数值回归和越界测试。不要从不受信任的 payload 读取 scale/shape 后直接分配任意大小，先做上限检查。异常 metadata 应进入 `CORRUPT` 并触发重算或拒绝，不让远端页决定本地指针。

## 30.14 CPU-only Toy Lab：让在线策略可复现

### 30.14.1 实验边界

`labs/ch30_kv_compression_lab.py` 使用标准库模拟三件事：

1. `quantize` 对有限浮点向量做 per-group 对称量化，报告 MSE 和最大误差；
2. `compressed_bytes` 叠加 bit payload、scale/zero-point、低秩系数和稀疏索引，展示理论比率为何偏离整数倍；
3. `simulate` 生成 Zipf-like page 访问，在线执行 COLD(SSD INT4)→WARM(DRAM INT8)→HOT(HBM FP16) 晋升，以及空间不足时的 demotion/eviction/admission。

模拟时钟、延迟常数和 tier 容量是 toy 参数，不代表任何 GPU、NUMA、SSD 或 CXL 性能。它没有真实 attention logits、tokenizer、CUDA kernel、压缩库、网络或多租户安全证明。

### 30.14.2 运行命令

```bash
python3 labs/ch30_kv_compression_lab.py \
  --seed 30 --requests 64 --pages 48 --accesses 600 --page-tokens 32 \
  --hbm-bytes 12000000 --dram-bytes 6000000 --ssd-bytes 40000000 \
  --admission-threshold 0.20 \
  --output reports/ch30-kv-compression.json
python3 tests/test_ch30_kv_compression_lab.py
```

JSON 包含 `plans`（每个 tier 的 bytes/page 和合成质量损失）、`hits_by_tier`、`misses`、`admissions`、`promotions`、`demotions`、`evictions`、延迟分位数和前 20 步 trace。trace 用于检查事件顺序，不应被当作生产日志格式。

### 30.14.3 受控变量矩阵

先固定 seed 和请求数，只改变一个变量：

- `page_tokens=8,16,32,64`：观察 metadata、页粒度和晋升频率；
- HBM 容量从 0.25×、0.5×、1× 工作集扫描：画 p95、hit rate、eviction；
- admission threshold `0.0,0.2,0.5,1.0`：区分“全收”和 probation；
- accesses 的热点分布：把 Zipf 权重改成近似均匀，观察策略是否被扫描流量污染；
- 改变 DRAM/SSD 容量：验证 demotion 是否优于直接丢弃；
- 用 `compressed_bytes` 比较 INT4/INT8、rank=2/4、sparsity=0.25 的 metadata break-even 点。

每次扫描保存命令、Python 版本、git commit、配置 JSON、stdout 和 manifest。不要只保留最好的点；负结果说明策略在该 workload 上不值得。

## 30.15 失败诊所：压缩看似成功却把系统推向错误边界

### 案例一：标称 4×，实际只省 2.7×

团队把 FP16 bytes 除以 4，预估 INT4 能容纳四倍请求，结果 HBM 只多出 2.7 倍空间。复盘发现：per-group scale、页尾对齐、sparse coordinate、双份 in-flight buffer 和 allocator fragmentation 没有计入。修复方法不是修改宣传数字，而是把 `payload、metadata、alignment、staging、free-list` 分字段上报，并按 page size 扫描 break-even。发布门要求实际 high-water mark 与公式相差超过 5% 时阻断。

### 案例二：平均质量不变，少数工具调用全部失败

离线 perplexity 几乎不变，但生产中的 JSON 工具调用 parse error 增加。敏感页恰好包含 schema、系统指令和数字边界，INT4 outlier 误差在长链路中被放大。修复是给 grammar/system/tool 页设置 protected tier，针对结构化请求单独做 logits/top-k overlap 回归，并把业务失败计入质量预算。平均分数不能掩盖 tail task 的灾难。

### 案例三：命中率上升，p99 反而恶化

增加 DRAM 容量后 hit rate 从 70% 升到 92%，但 p99 ITL 变差。原因是 admission 把低价值页写入 DRAM，读取时发生 NUMA 跨 socket 和 CPU 解压，热页的 HBM batch 被格式分裂。修复是使用效用/字节 admission、按 GPU socket 亲和分配、限制每步 mixed-format groups，并比较 `hit_rate × latency` 而不是只看 hit rate。

### 案例四：回收竞态让旧页覆盖新页

请求取消后，旧的异步 INT4 压缩任务完成并把 payload 写入同一物理 page id；新请求拿到旧 generation，输出偶发错误。根因是 eviction 只释放了 allocator，不检查 lease/generation。修复为每次写入带 generation、checksum 和 compare-and-publish；旧 generation 被拒绝，未提交任务在取消时可中止或安全丢弃。

### 案例五：低秩基底版本不一致

滚动升级后，新 worker 读取旧页的 token coefficients，却使用了新基底 B。shape 相同所以没有崩溃，logits 逐渐漂移。修复是把 basis_id、model_revision、rank 和 dtype 纳入 cache key 与 digest；版本不匹配走重算。任何“解压成功”都不能替代语义版本检查。

### 案例六：远端冷层把故障放大成雪崩

SSD/远端 cache 短暂抖动，所有 miss 同时回源重算，GPU prefill 队列爆满，p99 从 200ms 变成数秒。应为冷层设置并发和 bytes budget、single-flight 合并同一页请求、熔断后按优先级重算，并保留一小段 HBM safety reserve。缓存层不可用时，服务仍应有可预期的 degraded mode，而不是无限重试。

## 30.16 生产验证清单：从 toy 到真实模型

### 30.16.1 数值验证

- 锁定模型权重、tokenizer、RoPE/position revision、attention kernel、CUDA/driver 和 quantization library；
- 对每层、每 head、K/V 分别记录 max-abs、MSE、cosine、logits KL、top-k overlap；
- 在短/长、自然语言/代码/JSON/工具调用、不同 batch 和并发下比较 FP16、FP8、INT8、INT4；
- 运行 shadow 或 paired seed，统计质量分位数和置信区间；
- 明确哪些指标是代理，哪些是最终业务质量；代理越过阈值必须触发回退。

### 30.16.2 容量和性能验证

- allocator high-water mark、fragmentation、metadata/index、staging 和双写窗口全部入账；
- 采集 HBM/DRAM/CXL/SSD bytes、bandwidth、queue depth、NUMA、PCIe、解压 kernel 时间；
- 分开 TTFT、ITL、TPOT、E2E 的 p50/p95/p99，并按 hit/miss/tier/format 分组；
- 扫描 page size、group size、rank、sparsity 和 admission threshold，保留完整矩阵；
- 对比压缩、分层和重算三种方案，计算每个请求的总 GPU/CPU/网络资源，而不是只比较显存。

### 30.16.3 状态与故障验证

- 注入 checksum mismatch、metadata 越界、basis/version 不同、SSD timeout、远端断链、GPU OOM、取消竞态和重复迁移；
- 验证 generation/lease、引用计数、single-flight、backpressure 和幂等；
- 在压缩和迁移中途 kill worker，确认只恢复已提交 manifest；
- 验证租户隔离、加密、删除、审计和 cache key namespace；
- 记录 degraded mode 的质量、延迟和恢复时间，不能只测 happy path。

## 30.17 理解检查（含答案）

### 检查一：为什么 INT4 不是 FP16 的四分之一？

因为 scale/zero-point、稀疏索引、页尾对齐和 allocator/staging 也占字节。只有把这些项加入 `compressed_bytes`，才能得到真实 page allocation。若页很小，metadata 可能让 INT4 接近 INT8。

### 检查二：K 和 V 的误差应相同预算吗？

不应默认相同。K 的误差改变 attention logits，V 的误差改变加权输出；敏感度取决于层、head、位置和 workload。应分别测量并按业务质量分配预算。

### 检查三：低秩 rank 越小越好吗？

rank 小节省 payload，但 tail energy、重构 kernel 和共享基底版本风险增加。若基底每页重复存储或 rank 低于 metadata break-even，实际 bytes 可能不降；必须画 rank—误差—延迟曲线。

### 检查四：cache hit 率 95% 是否足以证明分层有效？

不足。95% 可能大多是慢速 SSD 命中，或命中页格式导致 batch 分裂；要同时看 tier 分布、解压/迁移时间、p99、质量和重算成本。HBM、DRAM、SSD 的 hit 不能混成一个数字。

### 检查五：为什么 eviction 需要 generation？

异步压缩、迁移和回收可能乱序完成。generation/lease 让消费者拒绝旧写入，防止已回收物理页被迟到任务覆盖。地址复用本身不是版本正确性。

### 检查六：为什么第一次访问不一定 admission？

一次性扫描的页若占用 HBM，会驱逐高复用页；在线系统可先放 probation COLD，二次命中再晋升。admission 应比较预计命中收益、重算成本、字节和误差，而不是“有空就放”。

### 检查七：toy lab 的质量损失能否外推到模型 perplexity？

不能。toy 的 `plan_quality_loss` 是可解释的合成成本函数；它不包含真实 logits、attention、tokenizer 或任务。真实质量必须用锁定模型和 workload 的数值/业务回归验证。

## 30.18 练习：从复述到独立设计评审

1. 给定 (L=40,H_{kv}=8,D=128,T=64)，比较 FP16、INT8（group=64、2-byte scale）和 INT4（group=32、scale+zero-point 各 2 bytes）的 page bytes；再加入 256B 对齐，解释容量比为何变化。
2. 用一个包含 outlier 的向量运行 `quantize`，比较 per-group size 16、64、256 的 MSE 和 metadata；写出在 K、V 上不同 group size 的风险。
3. 设 HBM 只能容纳 10 页，DRAM 30 页，访问分布从 Zipf 改成均匀。运行 toy lab，判断 probation admission 是否仍有收益，并说明 eviction trace。
4. 设计一个三层 cache key，至少包含 model revision、token range、format、basis/version、tenant namespace 和 position transform。列出任一字段缺失时的错误案例。
5. 将低秩基底作为单独 artifact 管理：写出生成、checksum、发布、回滚和过期步骤，并说明 tensor parallel 下谁拥有 B。
6. 为结构化 JSON 请求定义保护策略：哪些页保留 FP16，哪些页允许 INT8/INT4，遇到 grammar dead-end 时如何回退；把策略写成可测试的状态迁移。
7. 画质量—容量—p99 三轴 Pareto 图，标出 FP16 全驻留、INT8 两层、INT4 三层、积极 eviction 四个点。说明一个不在前沿的点为何应被删除。
8. 设计一次远端冷层故障演练：故障注入、single-flight/backpressure、重算上限、用户可见错误、恢复条件和证据字段都必须明确。

## 30.19 小结：把压缩变成有边界的在线控制

本章的关键不是记住某个 INT4 算法，而是建立一条能被测量和回滚的链：

1. 先按层、head、token、K/V 和 metadata 算真实 bytes；
2. 再按校准集测量量化、低秩、稀疏的误差，并把 outlier 和 basis version 写入格式合同；
3. 把 HBM、DRAM、CXL、SSD/远端视为不同延迟和故障边界，定义状态、lease、generation、checksum 和可见性；
4. 用 admission、promotion、demotion、eviction 和 recompute 把容量决策放进 scheduler，而不是在满了以后临时清空；
5. 用质量—容量—p95/p99 三轴寻找非支配点，按 workload、租户、结构化/普通请求分层；
6. 用 CPU toy 复现因果，再在锁定的真实模型、kernel、硬件和业务评估上验收。

当一个页从 FP16 热层降到 INT4 冷层时，系统并没有“免费得到四倍显存”：它得到的是一项新的协议义务——解释误差、记录 metadata、支付解压/迁移成本、处理版本和故障，并在质量或尾延迟越界时可靠地回到 FP16 或重算。只有把这些义务写进证据账本，KV compression 才是工程能力，而不是一次不可审计的 flag。

## 30.20 来源地图与证据边界

下列来源用于定义概念、算法或生产接口；具体版本、commit、硬件和实验条件必须在复现时重新锁定：

- **KIVI：** 研究非对称 2-bit KV quantization 与 key/value 不同粒度；不要把论文配置当作所有模型的默认值。
- **KVQuant：** 讨论 outlier、per-channel/per-token 量化与长上下文误差；应结合自己的 attention kernel 测量。
- **H2O、Scissorhands：** 提出重要 token/历史注意力启发式；重要性依赖任务和上下文分布。
- **SnapKV、PyramidKV：** 研究 prompt compression 与层级差异；它们减少的是特定阶段/任务的上下文，不等价于通用 KV tier eviction。
- **FlexGen：** 展示 GPU/CPU/SSD 分层推理的系统权衡；其设备、模型和带宽条件不能直接转译到今天的 serving 集群。
- **PagedAttention/vLLM 与 SGLang 文档：** 说明 paged KV、cache 管理和 serving 接口；请检查目标 commit 的格式支持、量化限制和指标语义。

### 可复现记录模板

```text
日期与时区：
代码 commit：
模型/权重 revision：
tokenizer/chat template revision：
GPU、driver、CUDA、kernel/量化库：
workload（语言/代码/JSON、输入输出长度、并发）：
KV shape（L/H_kv/D、page_tokens、K/V dtype）：
format（bits、group、scale、zero-point、rank、sparsity）：
tier（容量、带宽、NUMA、延迟、加密）：
policy（admission/promotion/demotion/eviction/recompute）：
指标（bytes、MSE/KL、hit、TTFT/ITL/TPOT/E2E p50/p95/p99、业务质量）：
异常注入与回滚结果：
artifact（JSON、日志、manifest、报告）路径：
```

本章的 `evidence/ch30-kv-compression-manifest.json` 列出论文、官方文档、toy lab、测试和报告；`reports/ch30-kv-compression-report.md` 只解释本地测量，不把 toy 数字包装成 GPU 或生产承诺。
