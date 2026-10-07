---
id: ch28-vllm-and-sglang-serving
title: vLLM 与 SGLang Serving：调度器、分页注意力与前缀复用的运行时
slug: /chapters/28-vllm-and-sglang-serving
description: 从请求进入到 token 流出的路径，解释 vLLM PagedAttention、SGLang RadixAttention、continuous batching、speculative decoding、structured generation、指标与故障诊断，并用 CPU toy 实验验证因果关系
sidebar_position: 28
level: advanced
prerequisites:
  - ch15-inference-execution
  - ch16-model-serving-system
  - ch19-inference-optimization-accelerator-stack
  - ch20-observability-debugging-incident-response
  - ch21-ai-reliability-engineering
  - ch27-kv-transfer-and-connectors
learning_objectives:
  - 能画出 vLLM 与 SGLang 从 HTTP 请求、tokenizer、scheduler、KV 管理到 GPU kernel 和流式响应的控制流
  - 能从 token、层数、KV 头数、头维度和精度推导 KV 工作集，解释 PagedAttention 的页表、内部碎片和抢占
  - 能解释 continuous batching、chunked prefill、优先级与公平性如何共同决定 TTFT、ITL、TPOT 和 goodput
  - 能比较 vLLM 的 block/page 管理与 SGLang 的 RadixAttention 前缀树，说明命中、插入、淘汰和多租户隔离边界
  - 能设计 speculative decoding 和 structured generation 的验收、回退、指标与故障处理，不把接受长度或语法成功当作质量证明
  - 能运行 CPU-only toy lab，对调度、缓存、草稿 token、结构约束和指标做可复现测量，并写出不越界的 benchmark 结论
estimated_hours: 36
hardware: CPU-only lab required; CUDA GPU optional and cost-bounded
risk_level: L4
last_verified: 2026-10-07
---

# 第28章　vLLM 与 SGLang Serving：调度器、分页注意力与前缀复用的运行时

> 一个 serving runtime 的核心不是“把模型包成 HTTP 接口”，而是在有限显存、有限 kernel launch 和不均匀请求下，持续做三个决定：这一个调度步让哪些序列前进、它们的 KV 放在哪些物理页、以及哪些 token 可以被验证或被语法约束。vLLM 以 block/page 化的 KV 管理和 continuous batching 为代表，SGLang 以程序化请求、RadixAttention 和结构化生成优化为代表。二者都能暴露 OpenAI 兼容 API，但内部的缓存键、调度边界、指标含义和故障模式并不相同。本章把它们放到同一套可审计模型中，所有数字区分事实、机制、测量、推断和设计判断。

## 28.1 为什么需要专门理解 serving runtime

### 28.1.1 “模型很快”不等于“服务很快”

离线地对一个 batch 做一次 forward，通常只测到 kernel 的计算吞吐；在线服务还要承担请求到达的随机性、tokenizer、队列等待、首 token 的长 prefill、每步 decode、流式网络写出、取消和显存回收。用户感知的时间至少有 time to first token（TTFT，首 token 时间）与 inter-token latency（ITL，连续 token 间隔）；平台要关心请求延迟 p50/p95/p99、输出 token/s、排队时间、GPU 利用率、OOM、重试和取消浪费。一个平均输出 token/s 很高的系统，如果 p99 TTFT 被长 prompt 拖到几秒，交互产品仍然不可用。

vLLM 和 SGLang 的价值来自运行时的共同设计：scheduler 不是模型外部的普通队列，而是决定每一次 GPU 调用的 token 数、请求集合、KV 分配和抢占。`max_num_batched_tokens`、最大并发序列、每请求最大长度、块大小、chunked prefill、优先级和 cache policy 互相耦合。改变其中一个参数会同时改变吞吐、尾延迟和可复现性，不能把某个单独 flag 当作“性能开关”。

### 28.1.2 三种容易误判的健康信号

第一，GPU utilization 90% 可能只是大量短 kernel、重复重算或网络等待后的突发，并不代表有效 token throughput 高。第二，KV cache usage 低可能表示命中率差或大量请求在 CPU 队列中，而不是内存有余量。第三，平均 ITL 很好可能因为少数请求在等待分页、结构化 grammar 编译或 speculative rollback，p99 已经失控。指标必须按阶段、请求长度、缓存状态和租户切片。

一个反例是“把 batch size 调到最大”。若 batch 中有一个 32K prompt，scheduler 在 prefill 时消耗大部分 token budget，短请求的 TTFT 变差；若强行打断又频繁产生 partial batch，GPU launch 与 KV 分配开销上升。正确做法不是寻找单一最大 batch，而是先明确 SLO、工作负载分布和可以接受的 preemption 代价，再调度。

### 28.1.3 框架名称不是解释

“vLLM 用 PagedAttention”“SGLang 有 RadixAttention”只是索引词，不能解释一个请求为什么等待、某个前缀为什么命中、一个 JSON 输出为什么退回普通采样。解释必须回答：控制面有哪些队列和状态；数据面哪些 token 被送进哪个 kernel；KV page 或 radix node 何时分配、引用、复制和释放；完成是 DMA 完成、GPU event 完成还是客户端收到 chunk。后文每个结论都标注来源类别，并用 CPU toy 把可验证的因果关系跑出来。

## 28.2 先修知识与学习交付物

### 28.2.1 先修知识

读者需要会计算字节、吞吐和排队的基本量，理解 transformer 的 Q/K/V、因果 mask、tokenization 和自回归循环；能读 Python 3.10+ 标准库脚本、JSON 和简单状态机；了解第 15、16、19 章的推理执行、serving、量化与 kernel；了解第 20、21、27 章的指标、故障和 KV connector。没有 CUDA、NVIDIA GPU 或生产 vLLM/SGLang 集群也可以完成本章实验，因为实验明确模拟调度协议，不声称硬件性能。

### 28.2.2 阅读后应交付的四张纸

1. **请求时序图**：`HTTP → tokenize → admission → prefill/decode scheduler → KV manager → attention/MLP kernels → detokenize → stream`，标出每个等待点和取消边界。
2. **工作集账本**：给出层数、KV 头数、头维度、精度、token 数、页大小和余量，算出每个请求的理想 KV bytes、页数和内部碎片。
3. **策略表**：对 vLLM page/block、SGLang radix prefix、speculative draft、structured grammar 记录命中键、失败动作、可观测指标和隔离边界。
4. **故障报告**：从 toy 的 JSON 记录基线与干预，说明观察、机制解释、不能推出的结论和下一步真实硬件实验。

## 28.3 一句话心智模型与边界

**Serving runtime 是一个按调度步运行的虚拟内存系统：scheduler 选择要推进的 token，KV manager 把逻辑序列映射到物理页或前缀节点，runtime 将可并行的 prefill/decode/verify 工作打包到 GPU，随后以可背压的流式协议交付 token。**

这个模型有五个边界。第一，KV 页或 radix node 只保存中间状态，不决定模型事实性和安全性。第二，cache hit 只表示某个键的中间状态存在，必须再验证模型、tokenizer、位置编码、租户和 lease。第三，continuous batching 让不同请求共享调度步，不保证公平或低尾延迟。第四，speculative decoding 只有在 target verification 接受后才改变输出，接受长度上升不等于语义质量上升。第五，structured generation 保证的是候选 token 的语法集合，schema 本身若设计错误，合法输出仍可能错误。

vLLM 和 SGLang 的共同抽象可写成：请求有逻辑 token 序列 (x_{0:t})、已生成长度 (t)、待生成上限 (M)、KV 引用集合 (R)、约束状态 (G) 和优先级 (p)。每个调度步选择集合 (B)，为每个请求分配本步 token 数 (q_i)，使显存、计算和公平约束成立。vLLM 常把 (R) 实现为可寻址 block/page 的间接表；SGLang 可把共享前缀表示为 radix tree 上的节点和引用。实现细节会随版本变化，心智模型用于对照源码和指标，而不是替代版本检查。

## 28.4 第一性原理：从 attention 到调度目标

### 28.4.1 Decode 的数据依赖

设第 (l) 层的 query 为 (Q_l)，历史 key/value 为 (K_l^{0:t-1},V_l^{0:t-1})，新 token 的 attention 需要读取整个历史工作集。对单个 decode token，逻辑上计算：

\[
A_l = \operatorname{softmax}(Q_l (K_l^{0:t-1})^T / \sqrt{d}) V_l^{0:t-1}.
\]

其中 (d) 是每个 attention head 的维度，softmax 的结果由因果 mask 限制。即使矩阵乘 FLOPs 不大，读取 (K,V) 的字节与非连续地址访问仍可能成为瓶颈。多请求 batch 可以把多个 query 合并进 kernel，摊平 launch 和权重读取；但历史长度不同、页表不同和约束状态不同会增加调度元数据。

### 28.4.2 KV 字节与页数

定义层数为 (L)，每层 KV 头数为 (H_{kv})，头维度为 (D)，每元素字节为 (B)，单序列 token 数为 (T)。K 与 V 各一份，理想 KV bytes 为：

\[
K_{bytes}=2 \times L \times T \times H_{kv} \times D \times B.
\]

设一个物理 page 容纳 (Q) 个 token，则页数为 (P=\lceil T/Q\rceil)。最后一页若只有 (r=T\bmod Q) 个 token，理想内部未使用比例为 ((Q-r)/Q)；实现还会对齐到向量宽度、按层布局、保存 block table 和 allocator metadata。页大小越小，碎片和抢占浪费通常下降，但 descriptor、索引、事件和 kernel 参数增加。页大小越大，长序列访问更连续，却可能浪费显存并使取消粒度粗。

例：(L=32,H_{kv}=8,D=128,B=2,T=2048) 时，理想大小为 (2×32×2048×8×128×2=268{,}435{,}456) bytes，约 256 MiB。若 (Q=16)，需要 128 页；若 (Q=32)，需要 64 页。这个数字是**推断**，真实模型可能有不同 KV layout、量化、滑动窗口、跨层打包、beam 复制和 padding。

### 28.4.3 调度步的资源不等式

对一次调度步，令 (n_i) 为请求 (i) 本步新增 token 数，(C_{tok}) 为 GPU 允许的 token budget，(C_{seq}) 为并发序列预算，(F(n_i)) 为 prefill/decode 对应的计算成本，(M(R)) 为新增 KV 页和临时 buffer 的显存成本。基本可行性是：

\[
\sum_i n_i \le C_{tok},\qquad |B|\le C_{seq},\qquad M(R_{new})+M(R_{resident})\le M_{GPU}.
\]

真正 scheduler 还要加入 prefill chunk、decode 至少一个 token、优先级、公平配额、grammar 状态、draft/verify 的额外 token、通信窗口和 deadline。`C_tok` 增大可能提高吞吐，也可能让单轮 kernel 变长，抬高短请求 TTFT；`C_seq` 增大可能提高并发，也可能因 KV 逼近水位而触发抢占。设计判断应通过受控实验验证，而非从单个 utilization 数值推断。

### 28.4.4 Queueing 和 goodput

设到达率为 (lambda)，有效服务率为 (mu)，利用率 (
ho=\lambda/\mu)。在高 (
ho) 时，等待时间对微小的服务率下降非常敏感；一次 grammar 编译、缓存 miss、speculative 低接受率或远端 KV 拉取都可能把系统推入排队区。goodput 不是原始 token/s，而是在 TTFT、ITL、E2E 和错误率约束内完成的请求或 token 数。报告必须记录 workload、warmup、并发/到达率、输入输出长度分布和 SLO，否则“吞吐提升”不可比较。

## 28.5 两个 runtime 的请求路径

### 28.5.1 vLLM：API server、engine core 与 worker

vLLM 官方 architecture overview 将 API server 与 engine core/worker 分开：前端处理 HTTP/OpenAI 协议、输入预处理、tokenization、取消与流式响应；engine core 维护请求队列和调度；worker/runner 负责把批次送入模型执行。不同版本会拆分进程或线程，V1 engine core 与 model runner 的边界也在演进，因此应以目标 commit 的设计文档和 trace 为准。稳定的机制是前端不会直接决定物理 KV 地址，scheduler 和 worker 通过输入 batch、block table、slot mapping 与完成事件协作。

典型顺序如下：

1. 解析请求、应用 chat template、tokenize，记录 request id、采样参数和停止条件；
2. admission 检查模型实例、最大上下文、租户配额和队列容量；
3. scheduler 选择 waiting/running/swapped 请求，计算 prefill chunk 或 decode token 数；
4. block manager 为新增 token 分配物理 block，必要时 copy-on-write、swap 或 preemption；
5. worker 根据 block table 构建 GPU input，执行 attention、MLP、logits 和采样；
6. runner 返回每请求的 token、logprobs/stop 状态、KV 变更和 scheduler stats；
7. detokenizer 产生 stream chunk，完成或取消时释放 block 和 metrics。

Papers 和官方 docs 直接支持 PagedAttention、block-level memory management、continuous batching 和 preemptive scheduling；“每次调度步都严格按上述七步同步完成”是对代码路径的机制重建，不能当作 API 保证。

### 28.5.2 SGLang：程序执行、scheduler 与 radix cache

SGLang 将 prompt 程序和生成动作作为可执行请求。用户可以描述多轮对话、few-shot、并行采样、约束解码或工具调用，runtime 把这些动作编译/解释成 token 生成任务。官方论文和文档把 RadixAttention、压缩有限状态机/grammar、continuous batching 和 memory pool 作为关键优化。请求进入后，scheduler 先在 radix cache 查找最长可复用前缀，再决定要计算的 suffix；执行器把共享前缀与不同 suffix 组织进 batch，生成结果写回 radix tree 或被 pin 的节点。

与 vLLM 的 page 表相比，radix tree 的逻辑键天然表达“多个请求共享一段 token 前缀”。命中最长前缀减少 prefill，但树的节点拆分、引用计数、LRU/频率策略和多租户命名空间成为新的元数据成本。SGLang 的 `--disable-radix-cache` 和 benchmark 文档可用于做对照实验；默认 cache 是否开启、是否包含采样参数和是否跨请求共享，必须查目标版本配置。

### 28.5.3 共同的控制面与不同的数据面

两者都需要 tokenizer/model revision、采样策略、stop/grammar 状态、KV capacity、水位、取消、超时和 metrics。数据面差异体现在复用索引：vLLM 通常把每条序列分成固定大小的逻辑 block，再映射到非连续物理 block；SGLang 将 token 序列前缀挂在 radix tree 节点上，并将叶/内部节点的 KV 与引用联系起来。两者都可能使用分页、压缩、量化或远端 connector，不能仅凭名字推断所有版本的内存布局。

一个公平的比较应固定模型、权重、tokenizer、GPU、TP/DP、输入输出长度、并发/到达率、warmup、cache policy、结构约束和采样随机种子；同时运行 cache cold、warm、50% shared prefix 三种场景。若只比较一个有利于 radix 命中的 shared-prefix workload，不能宣布 SGLang 在所有任务上更快；若关闭 vLLM 的 prefix cache 或改变 block size，也不能归因于引擎本身。

## 28.6 Continuous batching 与 scheduler/runtime

### 28.6.1 Static、dynamic 与 continuous batching

static batching 等所有请求准备好后一次运行，简单却会被最慢序列拖住；dynamic batching 在短时间窗口聚合请求，减少等待但仍有批次边界；continuous batching（迭代级 batching）在每个调度迭代结束时移除完成序列、加入新序列，并为正在 decode 的请求继续产生 token。它把一个大 batch 变成一串可变形的微批次，让短请求更早退出、空槽更早被填充。

continuous batching 并不消除队列：prefill 可能占用 token budget，长序列占用 KV，grammar/draft verification 增加每步工作，前端网络与 detokenizer 仍可能背压。实现必须定义“加入”的安全点：通常在 GPU step 结束、KV 状态一致、没有未提交的 CUDA event 时加入；若在异步 copy 或 page allocation 中途插入，会读到半完成状态。

### 28.6.2 Prefill、decode 与 chunked prefill

prefill 一次处理输入 token，产生完整前缀 KV；decode 每轮通常每请求产生一个 token。chunked prefill 把长 prompt 分片，使短请求可以在 chunk 之间插入，但会增加 prefill 的调度轮数和可能的 kernel 小批次。策略可用两个 budget：`prefill_budget` 与 `decode_reserved`，先保证 decode 的最小进度，再用剩余预算填充 prefill。对交互服务，这通常比“直到 prompt 完成才让 decode 运行”更符合 TTFT/ITL SLO；对离线吞吐，较大的 prefill batch 可能更优。

chunk 的边界必须跟 KV page、position ids、rope scaling、grammar 状态和 speculative draft 对齐。切在错误 token boundary 会导致 position 偏移；切块后缓存索引若没有把模型 revision 纳入键，会复用不兼容的 KV。测试中应故意把 prompt 切成 1、页大小、页大小+1 和最大块四种边界，检查输出 token 与不切块基线一致。

### 28.6.3 抢占、swap 与公平

当新请求需要页而 GPU 水位逼近上限，scheduler 可以拒绝、抢占 running 请求、把 KV swap 到 CPU/远端、或重算被抢占的 prefix。抢占目标通常考虑剩余长度、优先级、已付费 SLO、重算成本和 cache 共享价值。swap 增加 PCIe/NIC 流量与尾延迟，重算增加 GPU FLOPs；应记录 `preemptions_total`、swap bytes、recompute tokens 和被影响请求的 TTFT。

公平不等于每个请求每轮一个 token。一个请求可能有 16K prompt，另一个只有 32 token；按请求轮询会让大 prompt 长时间占据 admission，按 token 轮询又可能饿死低优先级长上下文。可采用 deficit round robin：每个队列有 token credit，完成 token、输出优先级和 deadline 共同更新 credit。无论策略如何，要有 starvation 检测和最大等待时间，不能只看平均吞吐。

### 28.6.4 结构约束与 scheduler 的耦合

structured generation 的 grammar state 每产生一个 token 就更新；不同请求的状态不同，不能简单把所有 logits mask 拼成一个共享张量。scheduler 需要把 grammar state pointer、允许 token 数、拒绝采样重试、JSON schema 编译状态纳入请求预算。grammar 编译发生在 admission 还是首个 decode step，会影响 TTFT；预热缓存和按 schema hash 共享可减少抖动，但 schema hash 必须包含 tokenizer、grammar engine 和严格模式。

### 28.6.5 Speculative decoding 与 scheduler 的耦合

speculative decoding 一轮先由 draft model、n-gram 或 suffix 方法提出 (k) 个候选，再由 target model 并行验证并接受前缀。对 scheduler 而言，验证轮一次消耗 (k+1) 个逻辑 token 的 KV 与 logits budget，却可能只输出 (a+1) 个 token，其中 (a) 是接受长度。低接受率时，额外 draft 计算和 KV page 分配可能让系统比普通 decode 更慢。应按请求或 prompt 类别动态开启，监测 `draft_tokens`、`accepted_tokens`、`accept_length`、rollback、draft latency 和 target latency。

## 28.7 PagedAttention 与 RadixAttention

### 28.7.1 vLLM PagedAttention 的机制

PagedAttention 借鉴虚拟内存：逻辑 token block 不要求连续物理显存，block table 把每条序列的逻辑块映射到物理块。attention kernel 根据 table 读取 K/V，允许不同序列共享物理 block（例如共享前缀的 copy-on-write）并减少因预分配最大上下文而产生的碎片。论文报告在其模型与硬件实验中相较基线有 2–4× throughput 优势；这属于论文测量，不是本章或任意生产配置的保证。

管理器需处理：首个 token 的 block 分配、最后块的部分使用、共享块引用、写时复制、finished block 的释放、swap/preemption、prefix cache 的命中和坏块回收。一个健康但错误的实现可能只在 block table 中更新了物理地址，未同步 position ids 或 refcount，结果是 GPU 没有 OOM，却出现重复或跨请求 token。单元测试要比较 logits/output 与连续 KV 基线，而非只断言分配器“成功”。

### 28.7.2 SGLang RadixAttention 的机制

RadixAttention 将已计算的 token 前缀组织到 radix tree。每个节点代表一段 token 边，节点可有多个子节点；新请求沿 token 序列从根走到最长匹配点，命中节点的 KV 可复用，剩余 suffix 进入 prefill。生成后，新 token 可以追加到叶节点或触发节点 split；多个请求共享前缀时，引用计数和淘汰决定何时释放。论文的关键观察是程序化生成经常重复 system prompt、few-shot、工具描述或对话前缀，显式复用比单纯 page allocator 更适合这类工作负载。

radix cache 的成本包括 hash/token 比较、节点分裂、锁/并发、LRU metadata 和潜在的热点前缀。一个长公共前缀被很多请求引用时，命中率高但 eviction 可能让少数租户长期占据内存；若 prefix key 未包含租户或 system policy，跨租户共享会成为数据泄露。设计判断是：公共、无敏感、版本固定的前缀才可跨请求共享；用户私密对话和工具返回值默认按租户/会话隔离。

### 28.7.3 分页和 radix 可以共存

“PagedAttention 对立于 RadixAttention”是错误二分。radix tree 解决逻辑前缀查找，节点内部仍可以指向 paged KV blocks；paged manager 解决物理分配，前缀索引仍可以是 hash table 或 radix tree。比较时要拆成三层：逻辑键如何找到共享前缀；物理 KV 如何布局和访问；scheduler 如何为命中/未命中分配 token budget。不同版本可能把其中两层合并，阅读源码时先画层次再看类名。

### 28.7.4 命中率不能单独证明收益

设 cache hit rate 为 (h)，平均命中前缀 token 为 (T_h)，重算单 token 成本为 (c)，查找/锁/管理成本为 (o)，远端加载成本为 (r)。粗略节省是 (h(T_h c-r-o))。当前缀很短、grammar/cache lookup 很慢或远端页拥塞时，h 上升仍可能使 TTFT 变差；当共享前缀只在 warmup 期间命中，长尾请求无收益。报告至少同时给 `prefix_hit_tokens`、`lookup_us`、`prefill_saved_tokens`、`cache_bytes`、`evictions` 和 TTFT 分位数。

## 28.8 Speculative decoding：草稿、验证和回退

### 28.8.1 验证算法的最小模型

令 draft 产生 (k) 个候选 (y_1,ldots,y_k)，target 对 prompt 加候选一次前向，得到对应分布。系统从前往后接受满足验证规则的 token，遇到不接受的 token 时回滚后续候选，并从 target 分布采样替代 token。若接受长度为 (a)，这一轮最终通常前进 (a+1) 个 token；(ain[0,k])。接受长度是测量，不是质量分数；它受 draft/target 相似度、采样温度、top-p、grammar、上下文和任务影响。

draft 可以是小模型、同模型的 n-gram、suffix cache、EAGLE/DFlash 等实现（具体支持随 vLLM/SGLang 版本变化）。n-gram 不需要额外模型却依赖重复文本；小模型提高候选质量但占用额外权重和 compute；suffix 方法适合重复后缀但跨域泛化有限。选择应按目标 workload 的 acceptance、额外显存、target batch 形状和尾延迟测量。

### 28.8.2 与 KV 页的交互

draft 生成候选时可能写入临时 KV 页，target 验证后只提交接受前缀，拒绝部分必须 rollback 或标记不可见。若 page 容量大于候选数，回滚可能只是调整逻辑长度；若候选跨页，需减少 refcount、清除页表、归还 free list。错误实现若把拒绝页留在 prefix cache，下一请求会命中不存在于 target 输出的状态。协议上应把 `draft_generation`、`accepted_len` 和 `committed_generation` 分开记录。

### 28.8.3 Structured generation 下的验证

grammar 允许集合 (V(G,s)) 由语法状态 (s) 决定。draft 候选若包含不在集合的 token，可以提前截断，但 target 仍需按 grammar mask 验证。接受长度因此不只由模型相似度决定，还由 grammar 分支宽度、tokenization 和 schema 设计决定。严格 JSON schema 可能导致候选接受长度短，却仍然减少 target steps；相反，宽松 grammar 接受长度高，但输出可能不满足业务约束。

### 28.8.4 观测与回退策略

建议每个请求记录 draft model/version、候选数 (k)、接受 token、拒绝位置、draft ms、verify ms、rollback pages、grammar rejects 和最终输出一致性。若 acceptance 连续低于阈值、draft queue 堵塞、OOM 水位或 p99 ITL 超标，可按请求类别关闭 speculative，回到普通 decode；回退动作必须保持输出协议和 request id 不变。不要在用户看不到的情况下悄悄改变采样语义，例如 draft 失败后改变 temperature 或 top-p。

## 28.9 Structured generation：从 schema 到 token mask

### 28.9.1 约束层次

结构化生成至少有三层：字符串格式（如 JSON/XML）、语法（JSON Schema、正则、上下文无关 grammar）和语义校验（字段范围、枚举、业务关系）。runtime 的 grammar engine 通常只能保证前两层；“生成了合法 JSON”不表示金额为正、日期存在或工具参数安全。服务端应在 grammar 完成后再做业务 validator，失败时返回可重试的结构化错误，不把业务错误伪装成模型错误。

### 28.9.2 tokenizer 与 grammar 的交叉

grammar 定义的是字符或字节序列，模型输出的是 token。引擎要维护 grammar state 并计算允许 token 集：对每个候选 token，把其字符串展开，检查是否仍存在一条合法路径，再生成 logits mask。一个 Unicode token 可能跨越多个字符，JSON escape、UTF-8、数字小数点和空白都造成边界问题。tokenizer revision 必须进入 grammar cache key；仅用 schema 字符串做 key 会在模型升级后复用错误的 token mask。

### 28.9.3 编译与缓存

schema 编译可能在首个请求、后台预热或进程启动时进行。冷编译延长 TTFT，且多个不同 schema 并发会争抢 CPU；预编译减少延迟，却消耗内存并需要版本失效。可按 `hash(schema, tokenizer_revision, grammar_engine_version, mode)` 缓存，限制条目数和每租户配额，暴露 `grammar_compile_seconds`、命中率、缓存字节、拒绝 token 数和编译错误。schema 内容若包含敏感字段名，metrics label 不应直接暴露全文。

### 28.9.4 Stream 与完成语义

流式 JSON 在中间状态可能不是合法完整 JSON，但每个增量 token 仍应满足 grammar 前缀。客户端不能在每个 chunk 上简单 `json.loads` 并判定失败；应使用增量 parser 或等 `finish_reason=stop` 后校验。runtime 取消时需释放 grammar state、KV 页和 schema cache 引用。若客户端断开但服务端继续生成，浪费 token 和敏感输出，应通过取消传播测试验证。

## 28.10 Metrics 与 benchmark：测量什么才不自欺

### 28.10.1 指标词典

- **TTFT**：从服务接受请求或进入排队的定义点到首个可交付 token。必须声明起点，是 HTTP 到达、tokenize 完成还是 scheduler admission。
- **ITL**：连续流式 token 的时间间隔。首 token、网络 flush、speculative 一次输出多个 token 会影响统计。
- **TPOT**：通常为 `(E2E - TTFT)/(output_tokens-1)`；要说明是否按请求平均或按 token 加权。
- **E2E latency**：请求完成时间减开始时间，包含排队、生成、detokenize、网络和业务校验的范围应固定。
- **throughput**：可按 output token/s、total token/s、request/s；输入和输出不能混为一项。
- **goodput**：满足 TTFT/ITL/E2E/error SLO 的有效请求或 token；公式和 SLO 必须显式写出。
- **KV/cache 指标**：命中请求数、命中 token、lookup latency、页分配、eviction、swap、recompute。
- **speculative/grammar 指标**：draft token、accepted token、accept length、rollback、grammar compile/hit/reject。

vLLM 官方 metrics design 文档列出 request prefill/decode 时间、queue、KV cache usage、eviction 等指标，并提醒 speculative 多 token 输出会影响传统 ITL/TPOT 统计；SGLang `bench_serving` 文档列出 TTFT、ITL、TPOT、E2E、throughput、accept length 以及 JSONL 详情。指标名和默认 labels 随版本变化，导入 Prometheus 时要 pin 版本并防止 request id、schema 或 prompt 进入高基数 label。

### 28.10.2 Benchmark protocol

一个可复现实验至少包括：

1. 记录 commit/tag、容器镜像、GPU 型号/数量、CUDA/driver、tensor/data parallel、量化和模型 revision；
2. 固定 tokenizer/chat template、输入输出长度分布、temperature/top-p、停止条件、grammar/schema、cache cold/warm；
3. 先 warmup，再测固定请求数或固定时间；报告并发、到达率、最大 in-flight、请求超时和失败；
4. 收集每请求 JSONL：request id、input/output tokens、queue、TTFT、ITL 数组、E2E、cache hit、preempt、draft/accept、error；
5. 至少报告 p50/p95/p99、均值、样本数、错误率和有效吞吐；不要只报最佳 run；
6. 做基线/干预成对实验，例如 radix cache on/off、speculative on/off、page size 变化、chunked prefill on/off；
7. 对每个观察写机制解释和不确定性，区分 toy、单机和生产结果。

### 28.10.3 不可直接比较的数字

不同框架如果使用不同 prompt 长度、输出长度、请求率、cache 热身、量化、GPU 时钟或 sampling，token/s 不能直接比较。SGLang bench 文档支持 synthetic、ShareGPT、shared-prefix、图像等数据集，vLLM 也有自己的 benchmark/metrics 工具；命令相似不代表 workload 相同。真实服务还要测连接复用、HTTP/2、TLS、流式 flush、客户端重试和 autoscaling 冷启动。若只跑离线 `generate`，不能证明 OpenAI API 的 p99。

## 28.11 最小实现阅读：伪代码与可验证边界

### 28.11.1 调度器伪代码

下面伪代码只表达资源和状态，不是 vLLM/SGLang API：

```python
def step(state):
    budget = state.max_tokens_per_step
    selected = []
    # 先给运行中的 decode 请求一个最小进度
    for req in fair_order(state.running):
        if budget == 0:
            break
        if req.cancelled or req.done:
            continue
        if req.needs_decode() and state.kv.can_append(req, 1):
            selected.append((req, 1))
            budget -= 1
    # 再把剩余预算给 prefill chunk 或 speculative verify
    for req in fair_order(state.waiting):
        chunk = min(req.remaining_prefill(), budget, state.prefill_cap)
        if chunk and state.kv.can_append(req, chunk):
            selected.append((req, chunk))
            budget -= chunk
    outputs = state.runner.execute(selected)
    for req, out in outputs:
        state.kv.commit(req, out.committed_tokens)
        if out.cancelled or out.stop:
            state.finish(req)
    return outputs
```

这个实现缺少优先级、swap、grammar mask、draft rollback、远端 KV、CUDA stream 和异常处理，因此不能当生产代码。它可以测试三个契约：已完成/取消请求不再被选；`budget` 不为负；只有 runner 报告 committed token 后 KV 长度才增长。生产阅读时把伪代码中的 `can_append` 对照 block allocator/radix manager，把 `execute` 对照 model runner，把 `finish` 对照 detokenizer 和 cleanup。

### 28.11.2 前缀查找伪代码

```python
def longest_prefix(root, token_ids, tenant, model_rev):
    node = root
    matched = 0
    while matched < len(token_ids):
        key = (tenant, model_rev, token_ids[matched])
        child = node.child(key)
        if child is None:
            break
        if not child.kv_ready or child.lease_expired():
            break
        matched += child.edge_len
        node = child
    return node, matched
```

关键点是键里有 tenant/model revision，且 `kv_ready` 不等于“索引中存在”。真实实现还要校验 tokenizer、rope/position、sampling-independent prefix、引用和 eviction。若把用户 prompt 的完整哈希写进日志，可能造成隐私泄露；使用受控摘要和采样日志更安全。

### 28.11.3 可观测事件

建议事件 schema 至少包含 `request_id`（短期采样）、`stage`、`queue_enter_ns`、`admission_ns`、`prefill_start/end`、`decode_steps`、`kv_hit_tokens`、`kv_alloc_pages`、`preemptions`、`draft/accepted`、`grammar_compile_ms`、`finish_reason` 和 `error_class`。不要把 event time 当指标 label；以 histogram/counter 聚合，保留低采样率 trace 关联。一个 trace 若没有 scheduler step、KV page 和 network write 的 span，就无法区分模型慢、排队慢和客户端慢。

## 28.12 CPU-only measured lab

本章实验在 `labs/ch28_vllm_sglang_lab.py`，等级 L0，Python 3.10+ 标准库。它不导入 vLLM、SGLang、PyTorch 或 CUDA，避免把软件安装问题误当成 serving 机制。脚本用确定性离散事件模拟：请求到达与输入/输出长度、continuous batching 的 token budget、page allocator、radix longest-prefix hit、speculative acceptance、grammar rejection、TTFT/ITL/throughput 统计和故障注入。参数是 toy 参数，不是 GPU、NVLink、RDMA 或真实框架 benchmark。

### 28.12.1 基线命令

```bash
cd Beneath-the-Tokens
python3 labs/ch28_vllm_sglang_lab.py \
  --seed 7 --requests 40 --concurrency 8 \
  --input-len 256 --output-len 64 --shared-prefix 0 \
  --engine paged --page-tokens 16 --max-batch-tokens 128 \
  --output reports/ch28-paged-baseline.json

python3 labs/ch28_vllm_sglang_lab.py \
  --seed 7 --requests 40 --concurrency 8 \
  --input-len 256 --output-len 64 --shared-prefix 128 \
  --engine radix --page-tokens 16 --max-batch-tokens 128 \
  --output reports/ch28-radix-shared-prefix.json
```

第一条近似 cache cold、固定页表；第二条让请求共享 128 token 前缀，比较 radix hit 与 prefill 节省。为研究 continuous batching，可改变 `--max-batch-tokens 64` 与 `256`；为研究 speculative，可加 `--speculative --draft-tokens 4 --accept-rate 0.75`；为研究 structured generation，可加 `--grammar-reject-rate 0.10`。每次实验都要保存完整 stdout JSON 和参数，不要只抄一行 throughput。

### 28.12.2 Toy 状态和测量

每条请求有 `WAITING → PREFILL → DECODING → FINISHED`，取消或资源不足可进入 `PREEMPTED`/`FAILED`。每个调度 step 先保留 decode token，再用剩余 budget 处理 prefill；engine=paged 时按固定页计数，engine=radix 时先对共享前缀做 longest-prefix lookup，再为 suffix 分配页。speculative 在 decode step 中提出 draft token，并按随机种子从 acceptance Bernoulli 采样；grammar rejection 增加一次采样检查和固定延迟。脚本输出 request rows 和 summary，summary 包括 TTFT/ITL/E2E p50/p95、output token/s、prefill tokens saved、cache hit tokens、allocated pages、preemptions、draft accepted、grammar rejects 和 error count。

这组测量能验证因果方向：共享前缀增加时，radix engine 的 prefill token 与 TTFT 应下降；页大小从 16 改为 64 时，页数与每页开销下降但内部浪费可能上升；accept-rate 太低时 speculative 的额外 draft/verify 成本会抬高 E2E；grammar rejection 增加时，输出 token 数固定而 step 时间上升。测量不能证明 vLLM 或 SGLang 的绝对速度、GPU 利用率、数值一致性、CUDA event 正确性或多租户安全。

### 28.12.3 预期观察表

| 干预 | 预期方向 | 机制 | 不可推出 |
| --- | --- | --- | --- |
| radix shared-prefix 从 0 提高到 128 | prefill tokens、TTFT 下降 | 最长前缀命中，suffix 重算减少 | 生产 cache 命中率或跨租户安全 |
| page tokens 从 16 提高到 64 | allocated pages 下降，尾页浪费可能上升 | 页粒度变粗，descriptor 减少 | PagedAttention kernel GB/s |
| max batch tokens 从 64 提高到 256 | token throughput 可能上升，短请求 TTFT 可能变差 | 迭代更大，排队和 kernel 时间增加 | 任意模型的最优参数 |
| speculative accept-rate 从 0.9 降到 0.2 | accepted/token 降低，E2E 可能上升 | draft 候选频繁回滚，额外工作不摊平 | 目标模型质量 |
| grammar reject-rate 增大 | grammar rejects 和 latency 上升 | 允许 token 集更窄，采样重试 | schema 语义正确性 |

### 28.12.4 运行测试

```bash
python3 tests/test_ch28_vllm_sglang_lab.py
# 若环境装有 pytest，也可以：
pytest -q tests/test_ch28_vllm_sglang_lab.py
```

测试契约包括 KV 字节和页数、continuous batching 不超过 token budget、radix 共享前缀节省 prefill、speculative 接受数不超过 draft 数、grammar rejects 可复现、CLI JSON schema 与文件输出一致。测试通过只证明这些 toy 契约，不能证明生产 readiness。

## 28.13 Failure clinic：看似健康但实际错误

### 28.13.1 p50 很好，p99 TTFT 爆炸

**症状**：平均 TTFT 120 ms，p99 4 s；GPU utilization 80%，错误率低。**排查**：按输入长度、cache hit、grammar compile、queue time 和 preemption 分桶；看是否长 prefill 连续占满 token budget，或少数 schema 首次编译阻塞 scheduler。若 queue time 占 E2E 大部分，调大 model kernel 并不会解决。**修复候选**：保留 decode budget、限制单轮 prefill chunk、预热常用 grammar、为长 prompt 单独队列或容量配额。**验证**：同一 workload 记录 TTFT p50/p95/p99 和 tail request trace，不能只看均值。

### 28.13.2 GPU utilization 高，throughput 下降

**症状**：开启 speculative 后 GPU 利用率从 65% 到 92%，output token/s 反而下降。**原因**：accept-rate 低，draft model 额外计算与 KV 写入没有被 target verify 摊平；或者 draft 和 target 的 batch 形状导致 kernel 低效。**动作**：比较 draft ms、verify ms、accepted tokens/step、rollback pages、普通 decode 基线；按模型/任务动态关闭或减少 draft (k)。**反例**：接受长度提高但 E2E 仍变差，可能是 draft queue 或 detokenizer 变慢；只看 accept length 会漏掉它。

### 28.13.3 KV usage 低，仍然 OOM

**症状**：监控显示 KV cache 使用 60%，但 runner 报 OOM。**原因候选**：临时 logits、grammar mask、draft buffer、activation、CUDA graph workspace、通信 staging 或 allocator 碎片未计入 KV gauge；另外 metrics 可能延迟刷新。**动作**：读取 allocator reserved/allocated、non-KV memory、pending page allocations、graph capture 状态；在 admission 前预留临时 buffer。**修复**：设显式 watermarks、限制 draft/grammar concurrency、让 gauge 区分 logical KV 与 physical reserved。**验证**：用统一采样时间戳和 OOM 前后快照，不要把 KV 使用率当总显存事实。

### 28.13.4 Radix cache hit 高，答案却过时

**症状**：prefix hit rate 95%，模型升级后部分请求仍返回旧格式。**原因**：cache key 只包含 token IDs，没有包含 model revision、tokenizer/chat template、rope scaling、grammar engine 或 system policy；旧 radix node 未失效。**修复**：namespace key 加入版本与租户，部署时 generation bump，旧 generation drain 后 scrub；对已命中页校验 metadata，失败则重算。**验证**：先用同 prompt 在两个 model revision 上对照输出和 cache key，再用 cold cache 复跑。

### 28.13.5 JSON parser 报错，但 grammar 指标全绿

**症状**：服务报告 `grammar_reject=0`，客户端仍收到无法解析的 JSON。**原因**：流式 chunk 在中间状态不是完整 JSON；或者服务只做了字符级约束，没有做 tokenizer/UTF-8 正确性；又或者 grammar 只包住了模型输出，外层 chat template/stop token 被客户端拼入。**动作**：保存脱敏后的 token 序列、grammar final state、finish reason、增量 parser 结果；区分“前缀合法”与“完成时合法”。**修复**：完成时运行 parser+schema validator，统一 stop token 和编码，明确客户端是否需要等待终止事件。

### 28.13.6 Radix cache 让一个租户饿死其他租户

**症状**：共享 system prompt 的租户吞吐上升，另一租户 p99 queue 飙升。**原因**：热点前缀节点长期 pinned、LRU 以请求数而非字节/租户计费、shared prefix 复用绕过了配额。**修复**：每租户 cache quota、节点 byte cost、最大 pin 时间和公平 scheduler；跨租户共享仅允许明确公开前缀。**验证**：两租户交替注入 shared-prefix 与 unique-prefix，比较命中、eviction、queue 和 p99，而不是只看总 throughput。

### 28.13.7 continuous batching 下重复 token 或漏 token

**症状**：少数请求输出重复片段，服务没有 OOM。**排查**：检查 scheduler step 的 request generation、KV logical length、position ids、block/radix refcount、speculative rollback 和客户端重试 idempotency。常见 bug 是请求完成后仍在 running list，或 cancellation 后旧 future 回写新 request id。**修复**：每个请求有单调 generation；提交 token 与发送 chunk 都带 generation；旧 generation 的回调丢弃。**验证**：随机插入取消、超时和 out-of-order completion，和单请求 greedy baseline 做 token-by-token diff。

### 28.13.8 metrics 自己造成性能退化

**症状**：开启详细 ITL 数组和 request labels 后 p99 变差。**原因**：每 token 记录高基数 label、同步日志锁、JSON 序列化和 trace export 占用 CPU/网络；SGLang bench 的 `--output-details` 适合受控实验，不应无条件线上开启。**修复**：固定 label 集、采样 trace、异步 ring buffer、聚合 histogram；敏感 schema/prompt 不入 label。**验证**：metrics off/normal/debug 三档对照，并测 CPU、GC、日志字节和请求尾延迟。

### 28.13.9 vLLM 与 SGLang benchmark 结果矛盾

**症状**：A 报 vLLM 更快，B 报 SGLang 更快。**原因**：cache warmness、shared-prefix 比例、结构化输出、speculative 配置、tokenizer、TP/DP、请求到达率或统计起点不同。**动作**：导出每请求 JSONL 和版本清单；先复现各自默认 workload，再构建交叉矩阵。**结论**：在相同 workload 下给出条件化结果，禁止只引用 headline。若无法对齐，明确“不可比较”比选择赢家更诚实。

## 28.14 Trade-offs 与被拒绝的替代方案

### 28.14.1 固定大 batch vs 迭代级 batching

固定大 batch 的优点是实现简单、kernel 形状稳定，适合离线任务；缺点是 straggler、长 TTFT 和 KV 峰值。continuous batching 的优点是完成即退出、请求可插入，缺点是调度与 allocator 复杂、可复现性下降。我们选择迭代级 batching 作为在线默认，并保留离线固定 batch 作为基线；理由是交互 SLO 需要 tail control。若业务只关心总 token/s，应在独立池运行离线模式，不能让一个策略兼顾所有目标。

### 28.14.2 Paged-only vs radix-first

paged-only 简化物理分配，对无共享前缀的随机 prompt 稳健；radix-first 对长 system prompt、few-shot 和多轮会话节省 prefill，但引入树元数据、锁和版本失效。选择应由 shared-prefix 分布和租户策略驱动。一个拒绝的替代方案是“永远启用最大 radix cache”：它把低命中或敏感 prompt 也放入内存，增加 eviction 与隐私风险。更好的做法是按前缀长度、命中频率、租户和 TTL 设门槛。

### 28.14.3 Speculative always-on vs adaptive

always-on 配置简单，若 draft 很匹配可提高 token throughput；但低 acceptance、短输出、grammar 严格或 GPU 已接近容量时可能变慢。采用 adaptive policy：冷启动先关闭，收集每类请求的 acceptance/latency，满足收益阈值才启用，连续退化自动回退。保留用户可审计的配置和事件，不能悄悄改变输出随机性。

### 28.14.4 强 grammar vs 后处理重试

强 grammar 在 token 级避免非法输出，减少后处理失败和重试；代价是 grammar compile、mask 和可接受候选减少。仅后处理在 schema 简单时开销低，却可能浪费整段生成并触发重试风暴。选择要看失败代价和 schema 复杂度。业务语义仍需 validator；grammar 不是安全沙箱，也不替代工具权限检查。

### 28.14.5 单一全局指标 vs 分层 metrics

单一总吞吐易于 dashboard，却隐藏 cache hit、prefill、decode、grammar、draft、租户和尾延迟。分层 metrics 增加采集与 cardinality 管理成本，但能支持故障诊断。保留少量稳定 SLO 指标 + 受控 debug 维度；把完整 per-request JSONL 只用于 benchmark/抽样，定期 scrub 敏感字段。

## 28.15 Paper、官方文档与源码综合

### 28.15.1 PagedAttention 论文

Kwon 等人的《Efficient Memory Management for Large Language Model Serving with PagedAttention》（SOSP 2023，arXiv:2309.06180）提出把 KV cache 切成 block、用 block table 映射、和 vLLM scheduler/抢占共同设计。论文支持“连续 KV 分配会浪费显存，分页减少碎片并提升吞吐”的事实；论文中的 2–4× 是特定基线、模型、硬件和 workload 的测量。阅读论文时要把 kernel 机制、block manager 和 benchmark 条件分开，不要把摘要数字作为部署承诺。

### 28.15.2 SGLang 论文与 RadixAttention

Zheng 等人的《SGLang: Efficient Execution of Structured Language Model Programs》（NeurIPS/ArXiv 2312.07104）描述语言程序、RadixAttention、压缩 FSM、continuous batching 和 runtime 优化。论文支持“程序化请求常有共享前缀，显式 radix cache 可复用 KV”的机制与测量；其吞吐倍数依赖任务、模型、共享前缀、硬件和实现版本。论文不是当前 pip 包的 API 文档，部署前应阅读目标 tag 的 launch/server、radix cache、grammar 和 benchmark 文档。

### 28.15.3 vLLM 官方文档与源码入口

vLLM stable docs 的 Architecture Overview 说明 API server、engine core、worker/model runner 的职责；Design Metrics 文档列出 queue/prefill/decode/request/KV 相关 metric，并提醒 speculative 对 ITL/TPOT 解释的影响；Features 页面包含 continuous batching、speculative decoding、structured outputs、disaggregated prefill 等入口。源码阅读建议从目标 tag 的 engine core/scheduler、KV cache manager/block table、model runner、metrics logger 和 tests 开始，再对照 docs 的版本注记。官方文档支持接口/职责事实，不自动保证未标注为稳定的内部类。

### 28.15.4 SGLang 官方文档与源码入口

SGLang docs 的 Bench Serving Guide 说明 `bench_serving` 的数据集、并发/到达率、TTFT、ITL、TPOT、E2E、吞吐和 JSONL 输出；Observability 页面列出生产 metrics；PD disaggregation 页面说明 prefill/decode 分离配置和 DP attention caveat。源码阅读入口包括 scheduler、radix cache、memory pool、grammar backend、launch_server 和 bench_serving。文档支持命令和指标定义；默认值、字段名和后端需以目标版本 `--help` 与 commit 为准。

### 28.15.5 连接器与 disaggregation 的边界

第 27 章介绍 NIXL、Mooncake、LMCache 和 connector；本章只把它们作为 runtime 的可选 data plane。vLLM/SGLang scheduler 决定何时请求远端 KV、为多少 token 保留预算，connector 决定数据如何注册、传输、校验和报告完成。不能因为 connector 传输成功就跳过本章的模型/layout/token sequence/lease 校验。P/D 分离若减少 prefill 干扰，也可能把网络带宽和尾延迟变成新瓶颈；benchmark 必须包含 cold/warm cache、丢包/超时和取消。

## 28.16 六个 comprehension checks 与答案

### 检查 1：PagedAttention 为什么能减少碎片？

**问题**：如果最大上下文是 8K，但请求最终只有 300 token，固定连续分配会发生什么？分页 block 如何改变这一点？

**答案**：连续策略可能按最大长度或增长策略预留大段显存，未使用空间无法被其他请求安全复用；分页只为实际 token 分配 block，最后 block 的内部浪费受 page size 限制，其他空闲 block 可以给别的请求。分页本身仍有 block table、对齐和最后页浪费，不能宣称零开销。

### 检查 2：radix 命中为什么不等于请求完成更快？

**问题**：命中率从 20% 升到 90%，TTFT 仍上升，可能有哪些机制？

**答案**：前缀很短，节省的 prefill 小于 lookup/锁/节点维护；命中节点需要远端加载或 lease 校验；热点节点造成 eviction/公平问题；grammar compile 或 queue wait 主导 TTFT。应同时看命中 token、lookup 时间、加载时间、queue 时间和 p95/p99，而不是只看 hit rate。

### 检查 3：为什么 accept length 不是质量指标？

**问题**：speculative 的平均接受长度提高到 3，能否断言答案质量提高？

**答案**：接受长度表示 target 接受 draft token 的比例，受 draft/target 分布、采样参数、grammar 和任务重复性影响；最终 token 仍由 target 验证规则产生。质量要用任务评估、拒答/事实性和安全测试验证，接受长度只支持性能分析。

### 检查 4：`max_batch_tokens` 增大一定提高 throughput 吗？

**问题**：从 128 调到 512 后总 token/s 提高，但短请求 p99 变差，如何解释？

**答案**：更大的预算让一次迭代处理更多 token，摊平 kernel/权重开销，原始 throughput 可能提高；同一迭代也可能让长 prefill 或 batch 执行更久，短请求在队列中等待，TTFT/ITL tail 变差。调参目标应是 SLO 约束下的 goodput，而非总 token/s。

### 检查 5：grammar cache key 至少包含什么？

**问题**：为什么 schema 字符串本身不够？

**答案**：tokenizer revision、grammar engine/version、模式（严格/宽松）和必要的模型/chat template 约束会改变字符到 token 的映射或允许集合；只用 schema 会跨版本复用错误 mask。还要按租户/敏感级别限制缓存共享和容量。

### 检查 6：如何定位“GPU 利用率高但服务慢”？

**问题**：应先看哪些分层指标？

**答案**：按请求查看 queue、prefill、decode、draft/verify、grammar compile、KV lookup/load、detokenize 和 network flush 时间；同时看 TTFT/ITL/E2E p50/p95/p99、cache hit/eviction/preempt/recompute、accepted tokens 和错误/取消。GPU utilization 只能说明某段时间有 kernel，不说明有效 token、尾延迟或客户端已经收到 token。

## 28.17 Exercises：从复述到设计

### 练习 A：recall

不看正文，写出 vLLM page table、SGLang radix node、continuous batch、TTFT、ITL、TPOT、accept length、grammar state 八个词的定义。然后在一张图上标出每个词位于控制面还是数据面。检查标准：定义包含可观察事件和一个常见误用；只写“更快/更省内存”不算完成。

### 练习 B：derivation

给 (L=40,H_{kv}=8,D=128,B=2,T=4096,Q=32)，计算理想 KV bytes、页数和最后页浪费。再假设 12 个并发请求、平均 25% page 内部未用、10% allocator/metadata 余量，估算 logical 与 physical bytes。写出哪些项是推断，哪些必须通过目标 engine 的 layout 检查。

### 练习 C：implementation

扩展 toy lab，加入每租户 cache quota 和 `tenant_domain` 到 radix key。构造两个租户使用相同 128 token system prompt 的 workload：策略一允许公开前缀共享，策略二完全隔离。测试在相同总容量下，两个租户的 hit tokens、eviction、p99 TTFT 和跨租户可见性。禁止在日志中输出完整 prompt。

### 练习 D：diagnosis

给出一份包含 p50/p99 TTFT、queue、prefill、decode、KV hit、grammar compile、draft acceptance、GPU reserved bytes 的 JSON。设计一棵五步故障树来区分“长 prompt 排队”“radix lookup 锁争用”“grammar 冷编译”“speculative 低 acceptance”“allocator 碎片”。每一步写需要的额外事件和停止条件。

### 练习 E：benchmark design

为 vLLM 和 SGLang 设计三组公平实验：cold random prompts、warm shared prefixes、strict JSON schema。固定模型 revision、tokenizer、GPU、TP、输入输出长度和请求率；定义 TTFT/ITL/E2E/goodput；列出 warmup、超时、失败、cache flush、seed 和原始 JSONL 的保存规则。说明为什么每组都要报告至少 p95/p99 和错误率。

### 练习 F：architecture

为“多租户客服 + 工具调用 + 长系统提示”设计一个 serving runtime。选择 page/radix、prefill/decode 分离、speculative、grammar、远端 KV 和 autoscaling 的启用条件；为每一项写容量、隐私、回退、metrics 和 runbook。至少包括一个“看似健康”的故障，解释如何在不重启全部实例的情况下隔离。

## 28.18 总结与下一依赖

本章把 vLLM 和 SGLang 还原成一个调度、虚拟内存和流式协议问题。vLLM 的 PagedAttention 通过 block table 管理非连续 KV，SGLang 的 RadixAttention 通过前缀树复用程序化请求；两者都需要 continuous batching、chunked prefill、抢占和严格的完成语义。speculative decoding 把 draft 与 target verification 绑定，结构化生成把 grammar state 和 tokenizer 绑定；两者都必须进入 scheduler 预算和 metrics，而不是作为“加速开关”单独开启。TTFT、ITL、TPOT、E2E、goodput、cache hit、accept length 和 GPU utilization 各自只证明一小段事实，只有带 workload、版本、硬件、错误和尾延迟的证据才能支持部署决策。

下一依赖是第 29 章（若课程继续）中的多模型路由、弹性扩缩与成本控制：它会把本章的请求级 scheduler 指标上升到副本级容量、路由和发布策略。继续实践时，先在目标版本跑最小健康检查，再做本章三组 benchmark，最后接入第 27 章的 KV connector 和第 21 章的超时/熔断；不要直接把 CPU toy 或论文 headline 当作生产容量。

## 28.19 Source map 与 reproducibility record

本章的来源按四类标注：论文/技术报告（P）、官方文档/参考实现（O）、成熟源码与测试（S）、本章可复现实验（M）。访问日期统一记录为 2026-10-07；版本敏感的命令应在运行时记录实际 tag/commit。

| ID | 类型 | 来源 | 支持的主张 | 复现/审计动作 |
| --- | --- | --- | --- | --- |
| p-paged | P | Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention”, https://arxiv.org/abs/2309.06180 | block/page KV、PagedAttention、vLLM scheduler co-design、论文条件下的吞吐结果 | 记录模型、GPU、基线、上下文和 workload；不要外推 headline |
| p-sglang | P | Zheng et al., “SGLang: Efficient Execution of Structured Language Model Programs”, https://arxiv.org/abs/2312.07104 | RadixAttention、程序化请求、FSM/structured generation、continuous batching | 记录 prefix sharing、schema、采样和版本 |
| o-vllm-arch | O | vLLM Architecture Overview, https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md | API server、engine core、worker/model runner 请求路径 | pin vLLM tag，比较设计文档和源码 |
| o-vllm-metrics | O | vLLM Metrics Design, https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md | queue/prefill/decode/KV/request metrics、speculative 统计 caveat | 用目标版本 `/metrics` 与 logger 对照字段 |
| o-vllm-docs | O | vLLM stable documentation, https://docs.vllm.ai/en/stable/ | continuous batching、speculative、structured outputs、disaggregation 入口 | 保存 `vllm --version` 和 `--help` |
| o-sglang-bench | O | SGLang Bench Serving Guide, https://docs.sglang.ai/developer_guide/bench_serving | bench_serving 数据集、TTFT/ITL/TPOT/E2E、JSONL 输出、accept length | 保存命令、数据集和 JSONL |
| o-sglang-observe | O | SGLang Observability, https://docs.sglang.ai/advanced_features/observability.html | 生产 metrics 与观测入口 | pin docs/commit，检查实际 exporter |
| o-sglang-pd | O | SGLang PD Disaggregation, https://docs.sglang.ai/backend/pd_disaggregation.html | prefill/decode 分离、DP attention caveat | 在有硬件的环境记录拓扑和带宽 |
| o-sglang-blog | O | LMSYS, “Fast and Expressive LLM Inference with RadixAttention and SGLang”, https://www.lmsys.org/blog/2024-01-17-sglang/ | RadixAttention 直观解释与条件化测量 | 与论文和目标代码交叉检查 |
| s-vllm | S | vLLM source and tests, https://github.com/vllm-project/vllm | scheduler/block manager/model runner/metrics 的实现入口 | pin commit，运行相关 unit/integration tests |
| s-sglang | S | SGLang source and tests, https://github.com/sgl-project/sglang | scheduler/radix cache/grammar/bench 的实现入口 | pin commit，运行目标 backend tests |
| m-toy | M | `labs/ch28_vllm_sglang_lab.py` | CPU toy 的 token budget、page/radix、speculative、grammar、指标因果 | `python3 labs/ch28_vllm_sglang_lab.py --seed 7 ...` |
| m-tests | M | `tests/test_ch28_vllm_sglang_lab.py` | toy 契约：预算、命中、接受、grammar、JSON | 直接执行或 `pytest -q` |

### 可复现命令与边界

```bash
python3 --version
python3 labs/ch28_vllm_sglang_lab.py --seed 7 --requests 40 --concurrency 8 \
  --input-len 256 --output-len 64 --shared-prefix 128 --engine radix \
  --page-tokens 16 --max-batch-tokens 128 \
  --speculative --draft-tokens 4 --accept-rate 0.75 \
  --grammar-reject-rate 0.10 --output reports/ch28-combined.json
python3 tests/test_ch28_vllm_sglang_lab.py
```

Toy 的 `elapsed_ms` 是离散模型参数计算出来的逻辑时间，不等价于 CUDA event；随机种子使相对观察可重复，但不代表真实网络或 kernel 的随机性。实验不覆盖真实 tokenizer、模型 logits、GPU memory visibility、CUDA graph、NCCL、RDMA、HTTP/TLS、autoscaling、工具权限、数据泄露或答案质量。生产验证至少需要目标框架版本的单卡健康请求、分页/前缀 cache cold/warm、取消和超时、speculative 低接受、grammar 编译失败、OOM 水位、metrics 开关和跨租户隔离演练，并保留原始日志与配置。
