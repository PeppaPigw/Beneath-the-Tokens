---
id: ch27-kv-transfer-and-connectors
title: KV Cache 传输与 Connector：把 Prefill、Decode 和分层内存接成可验证的数据面
slug: /chapters/27-kv-transfer-and-connectors
description: 从 KV 字节公式、分页和协议握手出发，理解 NIXL、Mooncake、LMCache 与 DistServe 的边界，并用 CPU toy 实验验证传输、缓存命中、重试和背压
sidebar_position: 27
level: advanced
prerequisites:
  - ch04-performance-math
  - ch08-distributed-collectives
  - ch15-inference-execution
  - ch16-model-serving-system
  - ch19-inference-optimization-accelerator-stack
  - ch21-ai-reliability-engineering
learning_objectives:
  - 能从层数、KV 头数、头维度、精度和 token 数推导 KV cache 字节数，并把带宽与排队写进容量账本
  - 能画出 prefill、decode、缓存存储和 connector 的控制面/数据面，解释 metadata、页、lease、校验和完成语义
  - 能区分 NIXL 的传输抽象、Mooncake 的分布式 KV 存储与传输引擎、LMCache 的缓存管理层、DistServe 的 P/D 资源分离
  - 能设计幂等、可取消、可重试、可回滚的 KV 传输协议，并说明失联、过期、序列不匹配和内存不足时的动作
  - 能运行 CPU-only toy lab，比较不同 connector 的字节、延迟、命中、重试和失败计数，知道实验不能证明真实硬件性能
  - 能用论文、官方文档、源码和可复现实验形成 evidence manifest，审查“传输加速”等主张的适用边界
estimated_hours: 32
hardware: CPU-only lab required; GPU/RDMA/NVLink optional and cost-bounded
risk_level: L4
last_verified: 2026-10-07
---

# 第27章　KV Cache 传输与 Connector：把 Prefill、Decode 和分层内存接成可验证的数据面

> 长上下文推理的瓶颈常常不是“模型算得不够快”，而是已经算出的 Key/Value 没有在正确的时间、正确的内存层级和正确的请求之间到达。KV cache 传输把一个看似简单的内存拷贝变成了带版本、租约、校验、取消、背压和权限的分布式协议。本章用新人可以复算的数字和状态机拆开这个协议，再把 NIXL、Mooncake、LMCache、DistServe 与 vLLM connector 放回各自的层级。文中明确区分**事实**（论文、官方文档或源码直接支持）、**机制**（根据调用和状态还原的因果）、**测量**（本章 CPU toy 得到的数字）、**推断**（依赖假设的计算）和**设计判断**（在约束下的推荐）。Toy 的毫秒数和带宽不是 GPU、RDMA、NVLink 或生产 SLO 的保证。

## 27.1 为什么 KV cache 传输值得单独学习

### 27.1.1 Prefill 与 decode 的工作形状不同

一次自回归请求先把输入 prompt 送入 prefill。prefill 可以在较大的矩阵中并行处理许多 token，主要受算力、批量和权重读取影响；然后 decode 每生成一个 token 都要读取此前 token 的 KV，并追加一个新的 KV。decode 的单步计算量较小，却反复触碰历史工作集，通常更容易受内存带宽、缓存命中和尾延迟影响。这是**机制**，不是“decode 永远带宽受限”的定律；batch、模型结构、精度、kernel 和上下文长度会改变瓶颈。

如果 prefill 和 decode 共用同一组 GPU，短请求的 decode 可能被长 prompt 的 prefill 挤出调度窗口。DistServe 论文把两个阶段放到独立资源上，允许分别选择并行度和副本数；其论文摘要在特定模型、负载和 SLO 设置下报告相对于基线最高 7.4× goodput 或 12.6× 更紧的 SLO。这里的“最高”是作者实验结果，不能外推成任意集群的普遍提升。阶段拆分之后，prefill 产生的 KV 必须可靠地交给 decode，这就是 connector 数据面的任务。

### 27.1.2 一次“拷贝”包含多个不可省略的承诺

把显存 A 复制到显存 B 只是数据平面的一小段。完整流程还要回答：

1. 发送方和接收方是否指向同一 token 序列、同一模型、同一 tokenizer 和同一 KV 布局？
2. 远端地址和访问权限如何交换？交换的 metadata 是否含敏感句柄，多久过期？
3. 一个请求的 KV 是整段、分页还是按层分块？接收方怎样知道第 37 页已经可用？
4. 传输中断时，哪些页可以重试，哪些页必须重新计算？重试会不会放大 decode 队列？
5. 客户端取消、模型回滚或租约到期时，发送方和接收方谁负责释放页？
6. “完成”指 DMA 已提交、目标内存可读，还是 scheduler 已将这段 KV 加入可用批次？

忽略其中任何一项，都可能出现“网络指标健康但答案错误”的事故。一个常见反例是把 token 数作为缓存键，却没有把 tokenizer 版本和模型 revision 纳入键；升级 tokenizer 后，前缀 token 序列看起来长度相同，实际边界已经不同，旧 KV 被错误复用。

### 27.1.3 缓存不是越大越好

KV cache 保存的是特定模型、位置和 token 序列下的中间状态。它比重新计算 prompt 更快，但会占用 HBM、DRAM、SSD 或远端内存，并带来一致性、隐私和删除责任。缓存命中率上升可能降低算力，却增加远端读取和淘汰元数据；把所有请求都转成“持久缓存”可能让备份、租户隔离和过期清理成为新瓶颈。设计缓存前先写工作集和保留界限：哪些前缀允许跨请求共享、哪些只允许同一租户、最长 lease 多久、删除事件如何传播。

## 27.2 前置知识、学习路径和一条心智模型

### 27.2.1 先修知识

读者需要理解矩阵乘和字节/秒的基本估算，知道 token、layer、attention head、batch、p50/p95 的含义；能读 Python 标准库脚本和 JSON；理解第 15、16、19 章中的推理调度、批处理、分页和 GPU 内存；理解第 21 章的超时、重试、熔断和幂等。没有 RDMA、NVLink 或 CUDA 经验也可以完成本章 CPU lab，因为 lab 模拟的是协议状态和容量关系，不模拟硬件内存一致性。

### 27.2.2 本章的一句话心智模型

**Connector 是请求生命周期的搬运工：控制面先确认“谁、搬什么、搬到哪、用哪一版、何时算完成”，数据面再按页异步搬运，接收方只有在校验和版本/权限都通过后才把页交给 scheduler。**

这句话有四个边界：Connector 不决定模型质量；Connector 不是缓存淘汰策略；Connector 不等于 P/D 调度器；Connector 的成功只证明指定字节在指定地址可读，不能证明端到端请求满足 SLO。NIXL 主要提供跨内存类型的传输抽象，Mooncake 还包括分布式 KV 存储和 Transfer Engine，LMCache 管理缓存生命周期和后端，DistServe 负责 P/D 资源拆分及其调度目标。把这些名称当作同义词会让故障归因失焦。

### 27.2.3 完成本章后应能交付什么

读者应能交付一张带箭头的控制面/数据面图、一页容量计算、一个带状态和错误码的协议草案、一个 CPU toy 的原始 JSON 报告、一次失败注入记录，以及一个 evidence manifest。若只能说“某框架支持 connector”而不能说数据如何从页表到完成通知，说明学习目标还未达成。

## 27.3 第一性原理：从 KV 字节到端到端时间

### 27.3.1 每个符号先定义

设模型层数为 (L)，每层参与缓存的 key/value 头数为 (H_{kv})，每个头的维度为 (D)，每个元素占 (B) 字节，序列 token 数为 (T)。一层有 K 和 V 两份张量，所以理想、未分页、未复制的 KV 字节数为：

[
K_{mathrm{bytes}} = 2 	imes L 	imes T 	imes H_{kv} 	imes D 	imes B.
]

“2”只表示 K 与 V，不表示 batch 或 beam。若有 (S) 条同时驻留的序列、beam 复制因子为 (R)、副本或层间复制因子为 (C)，理想总驻留字节可写为 (S R C K_{mathrm{bytes}})。实际还要加分页内碎片、对齐、索引、传输 staging buffer、压缩头和 allocator 碎片。建议用余量系数 (alphageq 1) 表示这些额外开销，并把 (alpha) 的来源写进报告，而不是默认为 1。

例：(L=32)、(H_{kv}=8)、(D=128)、(B=2)、(T=1024) 时，单序列理想大小是 (2×32×1024×8×128×2=134,217,728) 字节，即约 128 MiB。这个数量级解释了为什么“把 KV 从 prefill GPU 直接传给 decode GPU”会遇到带宽问题，也解释了为什么 GQA 减少 (H_{kv}) 会明显改变驻留量。此处是公式**推断**；真实引擎还要验证布局、头数、量化和分页方式。

### 27.3.2 带宽下限不是端到端延迟

设有效链路带宽为 (W) bit/s，单向传输字节为 (K_{mathrm{bytes}})，固定握手和排队时间为 (t_0)，分成 (P) 页，每页软件处理和通知开销为 (t_p)，重试额外字节和等待为 (t_r)。一个保守的时间模型是：

[
T_{mathrm{xfer}} ge t_0 + rac{8 K_{mathrm{bytes}}}{W} + P t_p + t_r.
]

这是下限模型，不是测量。若链路全双工、多个页并行、传输与计算重叠，壁钟时间可能低于“逐页相加”的直观估算；若有拥塞、NUMA 跨 socket、GPU staging 或流控，可能更高。报告要同时给出理论字节下限、链路实测吞吐、应用可见时间和重叠比例。

DistServe 论文给出的一个数量级例子是 OPT-66B 单请求 512 token 的 KV 约 1.13 GB，若每秒 10 个请求，原始搬运需求约 11.3 GB/s，折算约 90 Gb/s。这个数字是论文在特定布局和模型下的**事实**，不应套用到本章的 32 层 toy。它提醒我们：先算字节和请求率，再讨论“用 TCP 还是 RDMA”。如果带宽预算不足，任何 connector 都只能排队、压缩、重新计算或拒绝。

### 27.3.3 分页改变了浪费和调度粒度

把 KV 切成固定 token 数的 page 可以实现按需读取、共享前缀和有限淘汰。页大小太小，descriptor 数、通知和索引开销上升；页太大，短请求也要搬运多余字节，取消和重试粒度变粗。设页容量为 (Q) token，页数为 (P=lceil T/Qceil)。若最后一页只含 (r=Tmod Q) 个 token，内部浪费比例约为 ((Q-r)/Q)；真实引擎还会按层、头和对齐再切分。页表要存 token 起止、模型/策略版本、租户域、校验和、存储位置和 lease，不要只存裸地址。

### 27.3.4 何时传输，何时重算

设一页重算时间为 (t_c)，传输时间（含排队）为 (t_x)，且重算不会阻塞 decode 关键路径。若 (t_c < t_x)，重算可能更快；若远端页已经存在且会被多次复用，传输一次的摊销成本可能低于每次重算。NIXL connector 文档中的 `kv_recompute_threshold` 就体现了这种策略：低于阈值的 token 可以本地重算以摊平握手开销。阈值不是常数真理，应按模型、链路、页大小、命中率和尾延迟压测。失败时默认重算还是失败返回，必须是显式策略；默默重算会隐藏网络退化并放大 GPU 负载。

## 27.4 从请求到硬件：控制面、数据面与完成语义

### 27.4.1 两条平行路径

控制面负责低频、可审计的元数据：agent 身份、内存段、地址范围、长度、设备 ID、权限句柄、模型/布局版本、页索引、lease、校验策略和对端能力。NIXL 官方 Architecture 将已注册的内存抽象为 Memory Section，Transfer Agent 通过 backend interface 选择 UCX、Libfabric、GDS、TCP、NVLink 等路径；初始化时交换远端访问所需 metadata，通常不在每个页重复发送。控制面可以走 side-channel、etcd 或 Redis，但 metadata 本身要有权限和生命周期。

数据面承载 KV payload 和必要的完成通知。一个非阻塞请求通常是：构造读/写描述符 → post transfer → 轮询或等待 status → 校验目标页 → 通知 scheduler。数据面不应把大块 payload 放进控制面消息，否则 control-plane 拥塞会拖慢所有请求。控制面失败时，已提交的数据传输是否继续，要在协议中定义；本章建议把 transfer handle 绑定到 lease 和 cancellation token，控制面失联后在有限 grace period 内完成，过期则释放。

### 27.4.2 Transfer Agent 与 Memory Section

新手可把 agent 理解为“拥有地址空间和能力的端点”，把 Memory Section 理解为“经注册、可被后端访问的一段连续或可描述内存”。注册动作可能准备 RDMA key、GDS handle 或设备指针元数据；远端只得到完成读写所需的最小句柄。地址、长度和 device ID 不是数据完整性证明，也不是租户权限证明。应用层仍需在页表上检查请求、模型、租户和 token 摘要。

NIXL Quick Start 的生命周期顺序是：创建 Transfer Agent，创建后端，注册内存，交换 metadata，创建并 post 非阻塞传输，轮询状态，最后清理。顺序很重要：先 post 再注册可能得到无效句柄；注册后未清理可能泄漏 pinned memory；metadata 延迟更新可能让对端访问过期地址。把每一步写入 trace，才能区分“没有工作”“工作没提交”“工作提交但完成通知丢了”。

### 27.4.3 完成、可见和可消费

定义三个事件：`DMA_DONE` 表示后端报告搬运完成；`MEMORY_VISIBLE` 表示接收方使用正确的 stream/event 后可以读目标内存；`SCHEDULER_READY` 表示页通过校验、版本和权限检查，并加入可消费队列。三者不能混成一个“success”。若 scheduler 在 `DMA_DONE` 之前读取，得到部分页；若只等待 `MEMORY_VISIBLE` 而未更新页表，调度器永远看不到页；若页可读但 token 序列不匹配，答案可能语义错误而不触发硬件故障。

### 27.4.4 Backpressure 是正确性的一部分

接收方的 GPU 页池和 decode 队列是有限资源。发送方不能因为 socket 可写就无界 post，必须使用 credit、窗口或显式 `PAUSE`。可行协议是接收方发放 `credit_pages`，每成功提交一页消耗一个 credit，消费或淘汰后归还；超出窗口进入等待，不应静默丢页。credit 还应区分租户和优先级，避免一个长上下文占满所有页。观测至少记录窗口大小、在途页、等待时长、拒绝数和恢复时间。

## 27.5 协议草案：页、版本、租约和幂等

### 27.5.1 最小消息结构

下面是教学协议，不是某个项目的 wire compatibility。`TRANSFER_INIT` 包含 `request_id`、`transfer_id`、`model_ref`、`tokenizer_ref`、`layout_ref`、`tenant_domain`、`source_agent`、`target_agent`、`deadline_ns`、`page_count` 和能力列表。`PAGE_DESCRIPTOR` 包含 `page_id`、`token_start`、`token_count`、`bytes`、`checksum`、`source_section`、`destination_section`、`lease_expiry`、`idempotency_key` 和 `sequence_digest`。`PAGE_ACK` 返回 `page_id`、`status`、`bytes_received`、`checksum_ok`、`visible_event` 和错误码。`TRANSFER_COMMIT` 只有在所有必需页都 `READY` 后发送，`TRANSFER_ABORT` 可携带取消原因和是否允许重算。

descriptor 不应包含完整 prompt 或原始 token 文本；`sequence_digest` 可以是受控哈希，便于发现不匹配而不泄露内容。校验和用于发现传输损坏，不用于证明数据来自正确租户；身份和授权由 agent 认证、租户策略和 manifest 共同完成。对跨区域或敏感数据，数据面还要有传输加密和最小权限，不能因“只搬中间张量”就跳过安全评审。

### 27.5.2 幂等、重试和取消

每页的 `idempotency_key` 应稳定地由 `transfer_id`、`page_id`、布局版本和目标 generation 组成。重复发送同一 key 时，接收方可以返回已有结果而不重复写入；如果 payload digest 不同，必须报 `CONFLICT`，不能覆盖。重试只针对 `TIMEOUT`、`TRANSIENT_LINK`、`RETRYABLE_BACKEND` 等有限错误；`VERSION_MISMATCH`、`AUTH_DENIED`、`CHECKSUM_MISMATCH`（重复失败后）应终止或转重算。指数退避需有上限，并计入请求 deadline。

取消有三种来源：客户端断开、decode 调度器淘汰、租约或策略过期。发送方收到 `CANCEL` 后停止新页、取消可取消的在途页；接收方标记未消费页为不可见，归还 credit，清除敏感内容。若后端不能中止 DMA，要等待完成后立即 scrub，而不是把“取消已确认”提前返回。取消率和取消后仍完成的字节都应观测，否则会把浪费当吞吐。

### 27.5.3 Lease 与回收

页 lease 防止发送方认为缓存仍可读而接收方已回收。租约记录 owner、到期时间、generation 和 heartbeat；续期失败时，页可在 grace period 内被完成，但不应被新请求复用。NIXL/vLLM 的 connector 文档提到 prefiller KV block lease 和 decoder 双向缓存 TTL 的概念，具体默认秒数属于实现配置而非协议定律。推荐把 lease 变更写入事件流，支持审计“某页何时对谁可读”。

回收要有两个阶段：从索引摘除，阻止新读；随后等待在途引用归零并 scrub 内存。若内存压力触发紧急淘汰，可以优先淘汰可重算页；不可重算或合规保留页需要单独策略。不要把 Python GC 或 allocator free 当作数据擦除证明。

### 27.5.4 版本和序列对齐

最低版本键包括模型权重 revision、tokenizer revision、KV layout（精度、头数、页对齐）、位置编码配置和策略域。若 reasoning token 在客户端被剥离，token 序列可能与 prefiller 看到的序列不同；vLLM NixlConnector 文档明确提醒这会导致 prefix 不对齐。解决办法是服务端基于最终 token IDs 计算摘要，接收方逐页验证 `token_start/token_count` 与摘要链；无法验证时宁可重算。不要只用用户可见文本或字符长度做键。

## 27.6 NIXL：传输抽象如何帮助而不替你做调度

### 27.6.1 NIXL 的职责边界

NIXL（Inference Xfer Library）官方文档把它定位为点到点数据搬运库，覆盖 VRAM、DRAM、文件、块和对象存储等内存类型，并通过插件选择 UCX、Libfabric、GDS、Mooncake、UCCL-P2P 等后端。它提供 agent、memory section、descriptor、异步 post/status 和 metadata handler 等抽象。**事实**是这些对象和接口出现在官方 Overview/Architecture；**设计判断**是把它放在 serving stack 的“传输层”，让上层 scheduler 决定何时传、传哪几页。

NIXL 不知道你的质量阈值、租户配额或 prefix 是否可共享。即使 `get_xfer_status` 返回成功，也要由应用验证 model/layout/sequence digest 和页表状态。把 NIXL 当作“自动 KV cache”会漏掉淘汰、租约、命中和权限；把它当作“RDMA 等于零拷贝”也不严谨，staging、对齐、后端和拓扑可能仍产生拷贝或同步。

### 27.6.2 异步状态机

一个健壮的 NIXL 风格流程如下：

1. 启动时创建 agent，声明支持的后端和 device；
2. 注册源/目的 Memory Section，保存 generation 和 owner；
3. 通过 side-channel 交换最小 metadata，校验 agent 身份、版本和 TTL；
4. 上层把若干 descriptor 按页分组，指定 READ 或 WRITE、通知对象和 deadline；
5. `post_transfer_request` 非阻塞返回 handle；
6. worker 轮询或等待 status，区分 `PENDING`、`DONE`、`FAILED`、`CANCELLED`；
7. `DONE` 后执行 stream/event 同步、checksum 和页表更新；
8. 归还 credit，发送 ack/commit，最后注销过期 section。

步骤 5 到 7 是最容易写错的地方。同步等待每一个小页会把并发搬运变成串行；只看 handle 数量而不看完成状态会让队列假满；没有取消路径会在客户端断开后继续占用 GPU。测量时要记录 post time、真正 xfer time、可见到 ready 的时间，而不是只记函数调用耗时。

### 27.6.3 后端选择与拓扑

在同机 GPU 之间，NVLink 或异步 device copy 可能有低延迟；跨机可选 RoCE/InfiniBand、TCP 或其他 RDMA/EFA 路径；GPU 到文件或对象存储可能需要 GDS 或分层缓存。后端选择必须结合内存类型、NIC affinity、NUMA、拥塞和故障域。官方文档列出的支持不等于你的镜像启用了该插件，也不等于目标驱动/固件兼容。启动自检应报告后端版本、设备句柄、注册结果和最小 ping/read/write，而不是等线上请求才发现不可用。

### 27.6.4 安全与可观测性

metadata side-channel 是控制面，里面的 rkey、地址、端口和 agent ID 仍是敏感操作材料。用 mTLS 或受控认证、最小 TTL 和审计；禁止把 descriptor 原样写入公开日志。指标可记录 descriptor 数、post/完成时间、字节、失败类别、重试、in-flight 和队列，但不要把完整 token 或 prompt 放进 label。不同租户的页命中与失败要按策略域聚合，避免高基数泄露。

## 27.7 Mooncake：分布式 KV 存储和 Transfer Engine

### 27.7.1 它解决的更大问题

Mooncake 论文把 prefill 和 decode 集群分离，并把 CPU、DRAM、SSD、RDMA 组成分布式 KV cache。Conductor 根据 KV 分布和负载调度，Transfer Engine 负责点对点搬运。这个组合比单纯 connector 多两层能力：缓存在哪里、哪些前缀值得保留，以及如何在多个存储节点间放置和迁移。官方代码仓库和设计文档提供 Transfer Engine 的 Segment、BatchTransfer、RDMA/TCP/NVMe-oF/NVLink/SHM 等实现入口。

典型工作流可以拆成四步：prefill 节点收到原始输入、可复用 prefix block ID 和新分配的 full-cache block ID；增量 prefill 先从远端 CPU DRAM 取已有块并计算新增 token，再把新 KV 写回 CPU；独立 Messenger 异步按层将 KV 流式传到 decode 节点的 CPU buffer，和 prefill 重叠；decode 节点再异步把所需页加载到 GPU，并加入连续批处理。每一步都要有页状态和版本，不能只传一个“cache ready”布尔值。

### 27.7.2 Transfer Engine 的段和批量操作

Mooncake 官方 TE 文档将远端可读写的 DRAM/VRAM/NVMe-oF 等区域抽象为 Segment，支持批量、非连续、异步 READ/WRITE。大于某阈值的请求可被切片到多路径，注册 GPUDirect RDMA buffer 时还要考虑权限、rkey 和 NIC affinity。对新人来说，重要的机制是“把许多小页描述合并成一次有明确完成语义的 batch”，而不是把某个峰值 GB/s 当成常数。

批量有三种边界：批次太小，通知和 syscall 占比高；批次太大，取消和重试粒度粗；跨租户批次若共用 descriptor，审计和隔离困难。建议按请求/租户/优先级分组，保留每页 checksum 和 idempotency key，batch 失败时允许只重发坏页。若后端只支持整批重试，要把它的浪费计入成本账本。

### 27.7.3 Prefix hash、共享与隔离

Mooncake 论文描述了带前缀 hash 的 paged block 去重、复制和迁移。前缀 hash 可以减少重复计算和存储，但 hash 命中不等于允许跨租户共享。安全设计把 `tenant_domain`、模型/策略 revision 和访问级别放入逻辑键；同一文本在不同租户也应得到不同命名空间。缓存块退出时按 owner 和 lease 回收，不能因为 hash 相同就跳过删除审计。

### 27.7.4 把论文数字放回条件

论文报告的请求数、SLO 或传输吞吐只在它的模型、硬件、网络、负载、缓存命中和调度策略下成立。Mooncake 官方集成文档里的 8×RoCE 峰值、特定 32K prompt 时间，LMCache 技术报告中的约 400 Gbps 与 1.46× 重叠，都是条件化测量。本章把它们作为“该问什么”的线索：使用什么 NIC、页大小、并发、是否包含控制面和尾延迟？不要把摘要数字复制进容量承诺。

## 27.8 LMCache：把临时 KV 变成可管理的缓存层

### 27.8.1 管理层而非单一链路

LMCache 官方文档将 KV cache 从引擎内临时状态扩展为可持久复用的管理层，支持 CPU/SSD/远端后端，插件包括 Mooncake、NIXL、GDS、Redis 等。论文强调抽取、加载、持久化、跨网络传输要兼容 vLLM/SGLang，并提供批量、计算/I/O pipeline、可配置 chunk 和 zero-copy 等机制。它还描述 pin、lookup、cleanup、move、compress 等 control API。LMCache 的核心问题是“哪些缓存存在、如何查找和淘汰、何时搬运”，而不是取代所有传输后端。

### 27.8.2 Disaggregated prefill 的三组件

LMCache 的官方指南将 disaggregated prefill 拆为 prefiller、decoder 和 proxy/通知组件：prefiller 产生 KV，decoder 消费 KV，proxy 负责发现和握手。配置中可启用 `enable_pd`、选择 `transfer_channel='nixl'`、设置 `pd_role=sender|receiver`、buffer/device、peer 地址；低带宽会抵消收益，高带宽的 NVLink 或 PCIe Gen4/5 才适合某些场景。指南还提到 `pd_bidirectional`：先探测 decoder 已缓存的 KV，再通过 NIXL 读取，避免重复计算。

这说明缓存命中与 P/D transfer 是闭环：若 decoder 已有前缀，prefiller 不应盲目重新发送；若命中页被租约淘汰，必须回到重算或降级。proxy 的通知不能成为单点瓶颈，至少应有超时、重试和重建状态的方式。把 proxy 的“已登记”当作“页可读”会产生健康假象。

### 27.8.3 Chunk、zero-copy 与重叠

KV chunk 是性能和正确性的交叉点。chunk 太大减少 descriptor，但长时间占用 buffer，取消和公平性变差；chunk 太小可以细粒度调度，却增加 API、校验和通知。计算/IO pipeline 可以在第 (i) 块传输时计算第 (i+1) 块，端到端时间近似为 (max(T_{compute},T_{io})) 加上填充/排空，而不是两者简单相加。是否 zero-copy 取决于目标内存注册、布局、alignment、后端和消费者 API；文档中的 “zero-copy” 应理解为某路径的设计目标，而不是所有配置的保证。

### 27.8.4 过期、压缩和合规

缓存管理层必须回答 cleanup：客户端取消是否删除页？用户删除请求是否传播到 SSD 和远端节点？压缩格式是否绑定模型/布局版本？恢复时能否验证 checksum 和 sequence digest？压缩减少字节但消耗 CPU，且在 decode 尾延迟高时可能不值得。设计一个“可重算页”和“不可重算页”标志，前者可在压力下淘汰，后者要求更长保留和更严谨的访问日志。

## 27.9 DistServe：为什么 P/D 拆分会把网络带宽变成 SLO 变量

### 27.9.1 资源分离的目标

DistServe 观察到 prefill 偏 compute-bound，decode 偏 memory-bound 且对延迟敏感；把两阶段放到不同 GPU 后，可以独立设置并行度和副本，降低相互干扰。这个架构把原本同机共享的内存读写变成跨实例的 KV 传输，换来更可控的调度。系统目标是 goodput 和 SLO，而不是单一 tokens/s。prefill 端适合批量和吞吐，decode 端适合稳定的 inter-token latency；connector 必须让交接时间显式进入排队模型。

### 27.9.2 Pull 与 push 的选择

DistServe 描述的 pull-based 设计让 decoder 按需取 KV，避免 decode 内存被过早填满；节点内可用异步 cudaMemcpy，跨节点可用 NCCL 或网络路径。push 的优点是 producer 主动发送、协议简单；缺点是消费者可能尚未有页预算，造成拥塞。pull 需要 decoder 维护需求清单、处理多源和优先级；若 metadata 或 prefix lookup 慢，也会增加首 token 延迟。没有普适答案，应按消费者容量、链路和取消频率做实验。

### 27.9.3 带宽感知放置

当跨节点带宽不足时，DistServe 论文讨论把对应 prefill/decode 层放到同节点，走 NVLink 或更短路径。放置问题可抽象为：对每一层 (l)，选择节点 (n_l)，最小化计算、KV 搬运、排队和故障域成本，同时满足内存容量和租户约束。不要只优化平均带宽；p99 拥塞、网络共享、故障转移和滚动升级会改变最优解。部署前至少测单层 KV、混合请求、多个并发流和一个节点降级。

### 27.9.4 P/D 不是 connector 的替代品

DistServe 决定“为什么拆、拆多少、如何调度”；NIXL/Mooncake/LMCache 负责“具体怎么发现页、搬页、缓存和确认”。没有 connector，P/D 只能通过重新计算或粗糙 RPC 交接；没有调度，connector 可能把链路塞满而让 decode 更慢。架构评审应分别列出 P/D 层的资源模型、缓存层的生命周期和传输层的协议状态。

## 27.10 vLLM connector 视角：scheduler、worker 和 lookup buffer

### 27.10.1 两个 connector 角色

vLLM 官方 disaggregated-prefill 文档区分 scheduler connector 和 worker connector。scheduler 负责决定哪些 token/页要取、何时等待、何时降级；worker 执行实际 send/recv。这个拆分使调度策略不必知道 RDMA descriptor 细节，也使后端可以替换。NixlConnector、MooncakeConnector、LMCacheConnectorV1 等名称代表集成，不代表相同的 lease、失败或缓存策略。

### 27.10.2 LookupBuffer 解决乱序

在多请求并发时，发送完成顺序不一定等于 scheduler 消费顺序。官方文档中的 LookupBuffer 通过 `insert` 非阻塞写入，以 key 关联 token IDs 与 KV；`drop_select` 按请求选择并阻塞等待匹配项，解决 FIFO 到达与请求处理不一致的问题。教学协议可把 key 写成 `hash(model_ref, tokenizer_ref, token_ids, layout_ref, tenant_domain)`；实际实现应避免把原始 token 放进日志。若 key 冲突或序列摘要不匹配，必须拒绝并走重算，而不是拿“最接近”的页。

### 27.10.3 Pipe 与背压

Pipe 是单向 FIFO 的 send_tensor/recv_tensor 抽象，适合明确方向的数据流；它不自动解决跨请求公平、取消或内存上限。scheduler 应维护在途窗口，worker 报告 credit 和完成；当 decoder buffer 满时，producer 暂停或选择重算。监控只看 pipe 吞吐会漏掉等待和堆积，需同时看 `post_time`、`xfer_time`、`ready_time`、inflight pages、lookup wait 和 recompute count。

## 27.11 一张比较表：把职责、状态和证据对齐

| 组件 | 主要职责 | 不应误解为 | 关键控制面 | 关键数据面 | 新人应查的证据 |
|---|---|---|---|---|---|
| NIXL | 统一 VRAM/DRAM/文件/块/对象的异步搬运抽象 | 缓存淘汰器或 P/D scheduler | agent、Memory Section、backend、metadata | descriptor、READ/WRITE、post/status | NVIDIA 官方 Overview/Architecture/Quick Start 与源码 |
| Mooncake | 分布式 KV cache、Conductor、Transfer Engine | 只有一个 RDMA memcpy | block hash、对象映射、Segment、BatchTransfer | P2P/TCP/RDMA/NVMe-oF/NVLink 搬运 | 论文 2407.00079、官方设计和仓库 |
| LMCache | KV lookup、分层存储、生命周期、跨引擎管理 | 单一网络协议 | lookup/pin/move/cleanup、PD 配置、proxy | chunk、pipeline、NIXL/Mooncake 等后端 | 官方 docs、论文 2510.09665 |
| DistServe | P/D 资源拆分、并行度、SLO/goodput 调度 | connector 实现 | placement、pull、SLO 和副本 | prefill→decode KV 交接 | OSDI'24 论文与 repo |
| vLLM connector | 将 scheduler/worker 与 transfer/cache 对接 | 一定提供同样的失败语义 | LookupBuffer、Pipe、role、lease | send/recv tensor、页匹配 | vLLM 官方 disaggregated-prefill/nixl docs |

比较表不是选型排名。若问题是“我的页怎样从 GPU A 到 GPU B”，先看 NIXL/Mooncake TE；若问题是“哪些前缀值得保留”，看 LMCache/Mooncake policy；若问题是“prefill 和 decode 资源如何配”，看 DistServe；若问题是“引擎如何调用”，查 vLLM/SGLang 集成和源码。

## 27.12 CPU-only Toy Lab：用页、状态和随机丢包重现因果

### 27.12.1 实验边界

脚本 `labs/ch27_kv_connector_lab.py` 只用 Python 3.10+ 标准库。它计算 KV 字节、按 token 切页，给 `memcpy`、`tcp`、`nixl`、`mooncake`、`lmcache` 五个 toy profile 设置固定带宽、握手、每页开销和随机丢包率；丢包后按有限重试，成功后记录 `COMMITTED`。`--lmcache-hit-rate` 把命中请求建模为本地已有页，验证命中时不产生传输字节。profile 参数是教学假设，绝不是硬件 benchmark。

运行基线：

```bash
python3 labs/ch27_kv_connector_lab.py \\
  --seed 7 --requests 20 --tokens 1024 --page-tokens 128 \\
  --connectors memcpy,tcp,nixl,mooncake,lmcache \\
  --lmcache-hit-rate 0.25 --output reports/ch27-kv-baseline.json
```

输出 JSON 含 schema 版本、参数、单请求理想 KV 字节、每 connector 的请求数、cache hit、发送字节、重试、失败、p50/p95/mean toy ms，以及每请求页数。保存原始 JSON，不要只复制终端中的 p95。用三个 seed 重复，并在报告中声明随机分布和暖机规则（本 lab 无真实暖机）。

### 27.12.2 实验一：公式和分页

执行：

```bash
python3 - <<'PY'
from labs.ch27_kv_connector_lab import KVShape, make_pages
shape = KVShape(layers=32, kv_heads=8, head_dim=128, bytes_per_element=2)
for tokens in (1, 128, 1024, 2048):
    pages = make_pages('formula', tokens, 128, shape)
    print(tokens, shape.bytes_for_tokens(tokens), len(pages), sum(p.payload_bytes for p in pages))
PY
```

预期：总字节随 token 线性增长；页数为 (lceil T/128ceil)；所有页字节之和等于公式。若不相等，先检查最后一页 token_count、K/V 因子和单位，不要先怀疑网络。把 bytes 转 MiB 时除以 (2^{20})，转 MB 时除以 (10^6)，报告两者不要混用。

### 27.12.3 实验二：connector 和重试

分别运行无缓存与高命中：

```bash
python3 labs/ch27_kv_connector_lab.py --seed 11 --requests 50 --tokens 2048 --page-tokens 128 --lmcache-hit-rate 0 --connectors tcp,nixl,mooncake
python3 labs/ch27_kv_connector_lab.py --seed 11 --requests 50 --tokens 2048 --page-tokens 128 --lmcache-hit-rate 0.8 --connectors tcp,nixl,mooncake
```

在同一 seed 下，toy 的 profile 带宽更高或固定开销更低通常有更小的 mean/p95；TCP 的随机丢包会增加 retries，若重试成功则 bytes_sent 仍按逻辑 payload 统计，而不是把每次失败复制都当作新页。这里的统计简化了重传字节，故不能用来估算真实网络流量。高命中率应降低 bytes_sent 和 mean 时间，但请求数与页数仍不变。若命中并没有降低字节，检查 hit 分支是否在 transfer 前生效。

### 27.12.4 实验三：人为制造失败

把某 profile 的 `max_retries` 改成 0 或把 `loss_rate` 提高到 1.0，运行少量请求，观察 `failed` 与 `ABORTED`。失败时应报告错误类别，不能把失败页当作已提交。然后将 `loss_rate` 恢复，确认重试上限不超过 `max_retries × pages`。这验证的是状态机和计数契约，不是 TCP 或 RDMA 的真实恢复。

### 27.12.5 实验四：重算阈值的 toy 推断

改变 `page_tokens`，比较页数和固定每页开销；小页会增加 `per_page_ms` 的总贡献，大页减少通知却增大取消粒度。用公式算每页传输时间，再假设本地重算 0.2 ms/token，找出 toy 中 (t_c < t_x) 的页大小。把这个阈值写成“在该假设下建议重算”，不要写成某模型的生产默认值。真实系统还要测 GPU kernel、并发、链路共享和 cache 命中。

## 27.13 实验报告：从数字到可证伪结论

### 27.13.1 报告模板

`reports/ch27-kv-connector-report.md` 应记录：日期、主机 CPU、Python 版本、git commit、命令行、随机种子、参数、profile 表、原始 JSON 路径、失败注入、观察、机制解释、限制和下一步。每个结论标记类别。例如“hit=0.8 时 bytes_sent 降低”（**测量**）；“因为命中分支跳过 transfer_pages”（**机制**）；“真实 RDMA 会更快”（未经验证的**推断**，不能写成事实）。

### 27.13.2 一份示例结果

以下数字来自本章脚本在普通 CPU 的 toy profile，作为格式示例而非固定 golden：

| 场景 | connector | requests | hit | bytes_sent | retries | failed | p95 toy ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 无命中 | tcp | 50 | 0 | 6,710,886,400 | 随机 | 0 或少量 | 受 seed 影响 |
| 无命中 | nixl | 50 | 0 | 6,710,886,400 | 随机 | 0 或少量 | 受 seed 影响 |
| 高命中 | lmcache | 50 | 约 0.8 | 约 1,342,177,280 | 0 | 0 | 显著下降 |

表中“约”表示随机命中和 profile，不应当被当成测试断言。真实报告应粘贴脚本输出的精确 JSON。若修改 profile、页大小或脚本版本，必须更新报告和 evidence manifest。

### 27.13.3 测试契约

`tests/test_ch27_kv_connector_lab.py` 覆盖六条契约：KV 公式对 token 线性；分页覆盖准确 token 数和总字节；成功传输状态按 INIT→NEGOTIATED→MEMORY_REGISTERED→TRANSFERRING→COMMITTED；重试有上界；100% cache hit 不发送字节；CLI 输出可解析 JSON 并可写文件。运行环境没有 pytest 时，可直接执行：

```bash
python3 tests/test_ch27_kv_connector_lab.py
```

仓库若安装 pytest，也可运行 `pytest -q tests/test_ch27_kv_connector_lab.py`。测试通过只证明这些 toy 契约，不证明后端插件、驱动、网络、模型质量、跨租户隔离或生产恢复。将单元测试结果与至少一次手工失败注入、原始报告和源码 commit 一起保存。

## 27.14 生产实现阅读法：从文档跳到源码

### 27.14.1 先画调用图再读 API

阅读一个 connector 时，先标出四个调用点：scheduler 何时提出 lookup，worker 何时注册内存，backend 何时 post transfer，完成后谁更新页表和 batch。再追踪取消、超时、重试和版本检查。只读 README 的配置键容易误判默认值，必须查看对应 release 的实现和测试。对 NIXL，关注 descriptor、Memory Section、backend capability 和 status；对 Mooncake，关注 Segment/BatchTransfer、对象映射和 Conductor；对 LMCache，关注 lookup/pin/cleanup/move 和后端；对 vLLM，关注 LookupBuffer、Pipe、scheduler/worker connector。

### 27.14.2 读测试中的负路径

源代码中的 happy path 不能证明恢复。搜索 `timeout`、`cancel`、`lease`、`checksum`、`recompute`、`retry`、`drop_select`、`kv_load_failure_policy` 和 `generation`。vLLM NixlConnector 文档列出 load failure 可选择 fail 或 recompute；默认策略和重算阈值需锁版本。重算可能干扰 decode tail latency，必须在 SLO 报告中分开计数。观察测试是否验证 token 序列不匹配、远端 agent 消失、descriptor 过期、重复 key 和 buffer 满。

### 27.14.3 配置与能力探测

部署时先做 capability probe：后端是否编译、NIC/driver/firmware 是否兼容、目标 device 能否注册、side-channel 端口是否可达、最小页能否写读校验。把 probe 结果写入启动报告和 release manifest。不要只因配置文件里写了 `transfer_channel=nixl` 就宣称 NIXL 已经工作；插件可能回退到 TCP，或者发送方和接收方版本不兼容。指标要暴露实际 backend、页数、post/complete、失败和重算，而不是只暴露“connector enabled”。

## 27.15 失败诊所：健康指标掩盖的五类事故

### 27.15.1 事故一：带宽高，答案却错

现象：网络吞吐和 `DMA_DONE` 正常，用户偶发得到不相关答案。调查发现缓存键只含字符前缀，tokenizer revision 更新后 token 边界不同；页校验和正确，因为搬运的是“正确的旧页”。修复：键加入模型、tokenizer、layout、tenant 和序列摘要；接收方验证页链，无法证明则重算。教训：完整性不等于语义匹配。

### 27.15.2 事故二：p50 变好，p99 雪崩

现象：平均 transfer time 下降，p99 首 token 超时。原因是 decoder buffer 没有 credit，长请求一次 post 了数百页，占满页池；短请求在等待。修复：按租户和优先级设置窗口，限制每次 batch，满载时暂停 producer 或重算短前缀。教训：吞吐与公平性必须一起测，平均值无法证明背压正确。

### 27.15.3 事故三：重算掩盖网络故障

现象：业务成功率看似稳定，GPU 利用率和能耗上升，decode p99 逐步恶化。原因是 connector 在所有 load failure 上自动 recompute，网络断开时大量请求重复 prefill。修复：区分可重试和不可重试错误，设置重算预算与告警，超过阈值转降级或拒绝；报告“重算成功”而非把它混入 transfer success。教训：功能正确不代表成本和 SLO 正确。

### 27.15.4 事故四：lease 过期导致随机空洞

现象：长请求偶发缺页，重试无效。发送方 heartbeat 延迟，接收方按 TTL 回收了尚未消费页；proxy 仍缓存旧 descriptor。修复：generation+lease 原子更新，proxy 在回收时撤销 lookup，消费端收到 `LEASE_EXPIRED` 后整体重算受影响前缀。教训：缓存生命周期必须是协议的一部分。

### 27.15.5 事故五：控制面正常，数据面饥饿

现象：metadata/health endpoint 全绿，transfer queue 不断增长。根因是控制面探针只创建空 descriptor，没有真正搬运大页；数据面 NIC 与控制面共享队列且被日志占满。修复：分离控制/数据资源，加入最小真实页的端到端合成探针，监控 post→ready 和 credit。教训：健康检查要覆盖用户路径的关键数据。

### 27.15.6 事故六：跨租户 cache hit

现象：安全审计发现租户 B 命中租户 A 的公共前缀。原因是 hash 去重键未包含策略域，工程师以为相同 prompt 可安全共享。修复：把租户域和访问级别纳入命名空间，敏感租户硬隔离；已有页按 owner 重新加密或清理。教训：文本相同不代表权限相同，缓存 hit 也要通过授权。

## 27.16 权衡与被拒绝的替代方案

### 27.16.1 全部重算

优点是协议简单、无远端数据泄露、对失联更稳；缺点是长 prompt 重复算力昂贵，decode 资源被 prefill 干扰，质量/延迟可能不可接受。可作为低负载或小前缀 fallback，但应设置重算阈值和预算，并让用户知道 `degraded=true` 或内部指标显示重算。

### 27.16.2 全部用 TCP

优点是部署普遍、调试容易；缺点是高带宽 KV 场景可能成为瓶颈，CPU 拷贝和 socket 缓冲增加尾延迟。TCP 不是“不安全”或“永远慢”，但要在目标链路和并发下测量。若选 TCP，至少加入页化、窗口、校验和、压缩选择与连接复用，不要把一次大 blob 阻塞所有租户。

### 27.16.3 直接使用 RDMA 指针

优点是减少中间拷贝、带宽高；缺点是注册/pinned memory、rkey 生命周期、NIC affinity、驱动和故障恢复复杂，错误隔离和审计更难。NIXL/Mooncake 等抽象的价值是统一注册、descriptor、后端和状态，但并不消除硬件依赖。除非团队能维护驱动/固件、密钥和回滚，否则先用经过验证的 backend。

### 27.16.4 把所有 KV 持久化到 SSD

优点是容量大、可跨重启；缺点是延迟、写放大、加密和删除复杂，SSD 带宽可能比 GPU 需求低很多。分层缓存应按热度、可重算性和合规保留决定，不能把 SSD 当作免费 HBM。对敏感数据，持久化必须有密钥轮换和实际擦除证据。

### 27.16.5 无界预取

预取可以隐藏 latency，但会浪费带宽和页池，尤其在用户取消或 beam 分叉时。拒绝无界预取；用 credit、deadline、优先级和命中概率限制，统计预取命中与浪费字节。对 p99 敏感的 decode，可优先 demand-pull，再对稳定热点做小窗口预取。

### 27.16.6 只看 tokens/s

tokens/s 可能上升但首 token、inter-token、错误预算、每成功请求成本和跨租户风险变差。性能报告至少拆 TTFT、ITL、完整响应、transfer bytes、recompute、p95/p99、命中、失败和质量分层。否则无法知道“加速”来自缓存复用、请求变短、错误请求被排除还是尾部被隐藏。

## 27.17 六个理解检查（含答案）

### 检查一：KV 字节为什么有两个 2？

**问题**：公式 (2×L×T×H_{kv}×D×B) 中的 2 是否已经包含 batch？

**答案**：不是。第一个 2 表示每层有 K 和 V 两份；batch、beam、复制和分页余量要另外乘。若把 batch 误塞进 K/V 因子，会在容量账本中双算或漏算。

### 检查二：传输成功是否等于 scheduler 可用？

**问题**：后端返回 `DONE` 后能否立即把请求加入 decode batch？

**答案**：不能直接假设。还要确认目标内存可见、checksum、model/tokenizer/layout/sequence 版本、租约和权限，并更新页表。`DMA_DONE`、`MEMORY_VISIBLE`、`SCHEDULER_READY` 是不同事件。

### 检查三：NIXL、Mooncake、LMCache 谁负责淘汰？

**问题**：配置 NIXL backend 后，是否自动得到 prefix eviction 和租户隔离？

**答案**：NIXL 主要是搬运抽象；淘汰、lookup、租约、命名空间通常由上层缓存管理（如 LMCache 或 Mooncake 组件）负责。租户隔离必须由应用策略和 manifest 证明，不能从 backend 成功推断。

### 检查四：为什么低带宽可能不该传？

**问题**：已有 KV 就一定比本地重算快吗？

**答案**：不一定。比较页传输的固定握手、排队、带宽、校验和重试与本地重算时间；短前缀或拥塞链路可能更适合重算。阈值要按目标模型、页大小、命中和 SLO 测量。

### 检查五：cache hit 可以跨租户共享吗？

**问题**：两个租户的 prompt 完全相同，是否能用同一 hash 页？

**答案**：只有在明确的共享策略、相同模型/策略版本和授权域下才可以。默认把 tenant_domain/访问级别放入命名空间，避免内容相同掩盖权限不同。hash 只说明相似，不授予访问权。

### 检查六：Toy lab 的 p95 证明了什么？

**问题**：toy 中 nixl profile 比 tcp profile 快，是否证明 NIXL 在生产一定更快？

**答案**：只证明给定随机参数和公式下的模拟输出。它可验证分页、重试、命中和状态计数，不能证明真实 NIC、驱动、拓扑、GPU staging、并发和质量。生产结论需要目标硬件、版本、负载和失败演练的测量。

## 27.18 练习：从复述到设计评审

1. **字节账本**：为一个 40 层、GQA 8 头、头维度 128、FP16、8K 上下文的请求算理想 KV；再加入 32 路并发、10% 页内浪费和 20% 碎片，报告 HBM 预算。
2. **带宽门槛**：给定 12 GB/s 有效链路、512 token 页、固定 0.3 ms 握手，比较传输和 0.1 ms/token 重算；找出在何种 token 数下传输占优。
3. **协议状态机**：画 INIT、NEGOTIATED、REGISTERED、POSTED、VISIBLE、READY、ABORTED、EXPIRED，给每条边写触发事件、超时和可重试性。
4. **乱序消费**：模拟 100 个页随机完成顺序，设计 LookupBuffer key 和 `drop_select` 行为；验证重复 key、缺页和错误摘要不会静默消费。
5. **租户隔离**：为两个租户设计共享与硬隔离两种命名空间，列出 hash、加密、lease、删除和审计字段；说明你为何拒绝其中一种。
6. **P/D 放置**：给三台 prefill、两台 decode 和两条不同带宽网络，设计按层放置；计算单节点失效时的带宽和 SLO 风险。
7. **重算策略**：在 toy lab 中加入 `recompute_threshold`、重算预算和 `degraded` 计数，比较低带宽与高命中场景的成本。
8. **背压演练**：把 decoder credit 降为 1、4、16，观察等待、p95 和失败；写出 credit 归还的正确时机。
9. **源码审查**：在一个指定版本中搜索 lease、checksum、cancel、recompute 和 generation，写出一条“文档承诺—代码证据—测试证据”链。
10. **生产门禁**：设计 1%、5%、25%、100% 灰度，每阶段列质量、TTFT/ITL、transfer bytes、recompute、租户泄露、成本和自动回滚条件。
11. **删除传播**：构造用户删除一个 prompt 的事件，追踪 GPU、CPU、SSD、远端节点、备份和日志中的页；指出任何无法证明的残留。
12. **故障注入**：分别注入 agent 掉线、metadata 过期、NIC 拥塞、checksum 错误、客户端取消，记录状态、错误码和恢复动作。

## 27.19 安全、隐私和伦理边界

KV 是模型中间状态，不是“无意义的临时字节”。它可能编码用户输入、系统提示、检索文档和跨租户上下文。日志只记录哈希、长度、版本和状态，不记录原始 token 或完整 prompt；debug dump 必须加密、限时和受控。跨节点传输使用经批准的认证和加密，rkey、地址、端口和 agent metadata 视作敏感操作材料。不要因传的是 KV 就绕过数据驻留、删除和访问审计。

缓存共享要遵循最小权限：默认按租户和策略域隔离，明确授权的公共前缀才可共享；共享页的命中、读取和淘汰都应可追溯。对医疗、金融、身份、教育和政府请求，load failure 的“自动重算”可能改变决策延迟或使用过期数据，必须由策略决定并在响应中标记降级。未验证的旧 KV 不可在模型或策略回滚后继续使用。

性能实验也有伦理成本。不要上传真实用户 prompt 到公共 benchmark，不要用生产密钥或未授权 RDMA 句柄做实验；网络压测要设流量上限，避免影响同集群租户。把 energy、重算和废弃页纳入成本报告，不能只优化吞吐而隐藏资源浪费。任何跨区域复制都要经过数据驻留和合同审查。

## 27.20 总结：把“搬得快”变成可证伪的系统主张

本章从公式开始：先算每 token、每页和每请求的 KV 字节，再把有效带宽、固定开销、排队、重试和重叠写入时间模型。然后用控制面/数据面分离和状态机定义谁拥有页、何时可读、何时消费、何时回收。NIXL 提供跨内存类型的异步传输抽象；Mooncake 将分布式 KV 存储、Conductor 和 Transfer Engine 组合起来；LMCache 管理 lookup、分层存储、生命周期和跨引擎连接；DistServe 说明 P/D 分离如何改变资源和 SLO；vLLM connector 展示 scheduler/worker、LookupBuffer 和 Pipe 如何接入引擎。它们解决相邻问题，不能互相替代。

可复查的交付链是：冻结模型/tokenizer/layout/策略版本 → 计算容量和带宽账本 → 定义 descriptor、lease、checksum、credit、错误码和取消 → 读官方 docs 与源码的负路径 → 用 CPU toy 验证页、命中、重试和状态计数 → 在目标 GPU/网络上测 TTFT、ITL、p95/p99、字节、重算、成本和租户隔离 → 以小流量灰度和完整 bundle 回滚。任何只展示平均 tokens/s 或一次成功 `DONE` 的报告都不完整。

下一依赖是把本章的 connector 协议接到第 28 章（若课程继续扩展）的跨区域推理与数据驻留审查；若不扩展，读者应把协议、实验报告和 evidence manifest 作为第 26 章 Capstone 的可交付附录。无论选择哪个方向，保持同一原则：事实、机制、测量、推断和设计判断分开，未知量写出来，不能由一个绿色 benchmark 替你签署生产保证。

## 27.21 来源地图与 evidence manifest

下表列出本章引用的一手论文、官方文档和源码。访问日期为 2026-10-07；上线前应锁定具体 commit/tag 并重新核验 API。

| 来源 | 类型与链接 | 本章支持的主张 | 复核动作 |
|---|---|---|---|
| NVIDIA NIXL Overview | 官方文档：[Overview](https://docs.nvidia.com/nixl/getting-started/overview/) | NIXL 统一 VRAM/DRAM/文件/块/对象搬运，插件和异步定位 | 锁文档版本；运行 Quick Start 的 agent/register/post/status |
| NVIDIA NIXL Architecture | 官方文档：[Architecture](https://docs.nvidia.com/nixl/getting-started/architecture/) | Transfer Agent、Memory Section、backend、metadata 控制面与非阻塞传输 | 对照 descriptor、metadata handler 和完成状态源码 |
| NIXL Quick Start/Backend Guide | 官方文档/源码：[Quick Start](https://docs.nvidia.com/nixl/getting-started/quick-start/)、[BackendGuide](https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md) | 创建 agent、注册内存、交换 metadata、READ/WRITE、backend capability | 在目标后端跑最小读写、记录驱动/NIC/版本 |
| Mooncake 论文 | 论文：[arXiv:2407.00079](https://arxiv.org/abs/2407.00079) | P/D 分离、CPU/DRAM/SSD/RDMA 分布式 KV、Conductor、prefix block/hash 工作流 | 复现论文负载或明确 partial reproduction；不要外推摘要数字 |
| Mooncake 源码与设计 | 官方：[GitHub](https://github.com/kvcache-ai/Mooncake)、[Transfer Engine](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/) | Segment、BatchTransfer、P2P、RDMA/TCP/NVMe-oF/NVLink/SHM、zero-copy 目标 | 锁 commit；检查 buffer 注册、批量完成和错误路径 |
| LMCache 文档 | 官方：[docs](https://docs.lmcache.ai/)、[disaggregated prefill](https://docs.lmcache.ai/getting_started/quickstart/disaggregated_prefill.html) | 分层缓存、prefiller/decoder/proxy、NIXL channel、lookup 和双向 transfer | 记录配置、proxy、buffer、带宽和命中率；验证 cleanup/TTL |
| LMCache 论文 | 论文：[arXiv:2510.09665](https://arxiv.org/abs/2510.09665) | 抽取、加载、持久化、跨网络、chunk/pipeline/control API | 绑定论文版本和引擎版本；测量 overlap、失败与压缩开销 |
| vLLM disaggregated prefill | 官方文档：[disagg_prefill](https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md) | LookupBuffer insert/drop_select、Pipe、scheduler/worker connector、Nixl/Mooncake/LMCache 集成 | 锁 vLLM commit；为乱序、缺页、取消和 backpressure 写测试 |
| vLLM NixlConnector | 官方文档：[usage](https://github.com/vllm-project/vllm/blob/main/docs/features/nixl_connector_usage.md) | 全异步 send/recv、side-channel、lease/TTL、fail/recompute、token 对齐提醒 | 核对默认值和配置；注入 agent 掉线与重算 tail latency |
| DistServe 论文 | 论文：[arXiv:2401.09670](https://arxiv.org/abs/2401.09670)、[OSDI PDF](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) | P/D 阶段资源分离、带宽感知放置、pull transfer、goodput/SLO 结果 | 在目标模型、拓扑和 SLO 下重测；记录条件和不确定性 |
| DistServe 源码 | 官方：[GitHub](https://github.com/LLMServe/DistServe) | 论文实现入口和配置线索 | 锁 commit；检查通信、放置、失败和回滚路径 |

`evidence/ch27-kv-connector-manifest.json` 把每个 URL、来源类型、访问日期、claim、证据片段和复核状态机器可读化。manifest 不宣称论文或文档替代生产测试；它只让审阅者知道每个主张应去哪里核对。`reports/ch27-kv-connector-report.md` 保存 toy 命令、原始 JSON、测试结果和限制；`labs/ch27_kv_connector_lab.py` 与 `tests/test_ch27_kv_connector_lab.py` 是可执行入口。

### 27.21.1 可复现记录清单

- 主机 CPU、内存、OS、Python 版本和依赖锁；
- Git commit、脚本 SHA-256、命令行、随机种子和 profile 参数；
- 模型、tokenizer、KV layout、精度、页大小、batch、并发和请求长度分布；
- connector/backend、NIC/driver/firmware、NUMA、网络拓扑、buffer、credit 和 lease 配置；
- 理论 KV 字节、有效链路带宽、post/xfer/ready/commit 分解，p50/p95/p99、重试、重算、命中、取消和失败类别；
- 租户/策略域、加密、日志保留、删除传播和 scrub 证据；
- 论文/官方文档版本、访问日期、复现实验差异、未知量、回滚版本和批准人。

完成门禁：正文（含本章代码、表格和练习）超过 20,000 个中文字符；至少一个 CPU-only 可运行实验、六条自动测试契约、五个以上失败案例、六个理解检查及答案、十二个练习、来源地图和 evidence manifest；每个关键性能数字都标注条件；无 GPU/RDMA 的读者仍可完成状态、容量和协议推导。通过门禁只表示教材交付完整，不表示任何 connector 在生产环境已经达到 SLO。
