---
id: ch35-serving-scheduling-and-slo
title: 在线推理调度与服务 SLO：连续 batching、队列、流式输出与多租户隔离
slug: /chapters/35-serving-scheduling-and-slo
description: 从队列论和 TTFT/ITL/E2E 指标，到 continuous batching、prefill/decode 解耦、准入、流式取消背压与多租户 SLO 的可审计服务设计
sidebar_position: 35
level: advanced
prerequisites:
  - ch15-inference-execution
  - ch16-model-serving-system
  - ch19-inference-optimization-accelerator-stack
  - ch20-observability-debugging-incident-response
  - ch21-ai-reliability-engineering
  - ch28-vllm-and-sglang-serving
  - ch29-speculative-and-structured-decoding
learning_objectives:
  - 能用 Little 定律、排队等待和服务时间分解 TTFT、ITL、E2E 及其 p95/p99 尾部
  - 能解释静态 batching、dynamic batching 与 continuous batching 的状态机、收益和边界
  - 能设计 prefill/decode disaggregation、KV 传输、容量匹配和失败回退协议
  - 能实现带 prompt token 预算、并发上限、deadline、优先级和租户配额的 admission control
  - 能把 SSE/gRPC 流式输出、取消、超时、重试、背压和幂等连接成端到端契约
  - 能为多租户定义公平性、资源隔离、SLO error budget、降级顺序和审计证据
  - 能运行 CPU-only toy lab，复现实验队列、批处理、TTFT/ITL/E2E、取消和公平调度
estimated_hours: 52
hardware: CPU-only lab required; GPU serving benchmarks must be repeated on pinned model, driver, runtime and traffic traces
risk_level: L3
last_verified: 2026-10-07
---

# 第35章　在线推理调度与服务 SLO：连续 batching、队列、流式输出与多租户隔离

> 在线推理不是把一个 `generate()` 函数放进 HTTP handler。请求带着到达时间、租户、优先级、prompt 长度、输出上限、deadline 和取消信号进入系统；它们要排队、等待 GPU 资源、经历一次 prefill，再和别的请求共享 decode 迭代，最后以流或完整响应离开。一个短 prompt 可能很快出首 token，却因为 decode 队列拥塞而完成很慢；一个长 prompt 可能使同一批中所有请求的 TTFT 变差。要把体验、成本和公平都说清楚，必须把 scheduler 的每一次选择和 SLO 证据连接起来。

本章采用“协议优先、测量可重放”的路线。CPU-only toy lab 模拟请求到达、prompt 预算、连续 batching、prefill/decode 耦合或解耦、TTFT/ITL/E2E、取消和租户公平；它不声称测量 vLLM、SGLang、Triton 或任何 GPU 的真实吞吐。生产验收仍需固定模型、KV cache、GPU/驱动、量化、网络、并发分布和真实流量回放。

## 35.1 问题/边界：为什么“平均延迟”会骗人

### 35.1.1 在线请求是有截止时间的工作

离线 batch 可以把 10,000 个样本凑满再启动，在线服务不行。请求在 `t_arrival` 到达，用户可能在 `deadline` 前只关心第一个 token，也可能在完整答案前一直占着连接。至少要保存以下字段：

- `request_id、tenant、model_revision、arrival_monotonic_ns`；
- `prompt_tokens、max_new_tokens、stop_reason、stream`；
- `priority、deadline、admission_epoch、queue_enter/leave`；
- `prefill_start/finish、decode_rounds、first_token、finish`；
- `cancel_received、cancel_applied、tokens_emitted、bytes_dropped`。

没有单调时钟和阶段时间戳，`TTFT=响应时间-请求时间` 只能是猜测；没有 token 数，`ITL` 会被网络 flush 和客户端渲染时间混淆；没有 `stop_reason`，取消和模型自然停止无法区分。

### 35.1.2 边界：本章解决什么、不解决什么

本章解决单个服务副本内的 admission、队列、批处理、流式管道、SLO 和租户隔离。跨副本的路由、模型副本扩缩容、GPU 拓扑和权重加载在第16、19、28章已有背景；这里给出与 scheduler 的接口。不会给出“batch=8 一定最快”的硬件结论，不把 `prefill/decode` 解耦自动等同于低延迟，也不把网络流控当成模型取消。训练 serving、embedding 专用 ANN 服务、RL 在线 rollout 只有在它们暴露相同阶段指标时才可复用本章方法。

边界还包括安全和隐私：prompt、输出和租户标识属于数据治理范围，日志只能保留经过授权的 hash/长度/采样；`request_id` 必须避免把用户内容直接拼进 URL 或指标 label。SLO 预算是服务契约，不是绕过认证、内容安全或法律保留要求的理由。

### 35.1.3 先把三个“时间”分开

对一次流式生成，至少有三种时间：

1. **TTFT（time to first token）**：从服务接受请求到第一个可交付 token。通常包含 admission、排队、prefill、首个 decode 迭代、序列化和 flush；团队必须明确是否在 GPU token ready 还是客户端收到时取点。
2. **ITL（inter-token latency）**：相邻 token 的时间间隔。可报告均值、p50、p95、p99 和最大间隔；首 token 之前的空窗不应被误算为 ITL。
3. **E2E（end-to-end latency）**：从接受到完整结束、取消或 deadline。对流式请求，应明确是最后 token flush、FIN/RST、还是服务端 stop event。

一个“TTFT p95 很好”的服务可能在长答案上 ITL p99 很差；一个“E2E p99 很好”的服务可能大量拒绝请求。SLO 必须把可用性（接受率/正确响应率）、TTFT、ITL 和 E2E 放在同一份分母定义里。

### 35.1.4 队列理论的最低工具箱

设到达率为 `λ`（requests/s），平均服务时间为 `E[S]`，单副本有效并行度为 `c`。利用率近似是 `ρ=λE[S]/c`。当 `ρ` 接近 1，等待时间的尾部会非线性增长；不是把平均服务时间减半就能保持 p99 不变。Little 定律给出稳定系统的 `L=λW`：队列中的平均请求数 `Lq=λWq`，其中 `Wq` 仅是等待时间，不含服务。

对服务时间方差大的生成任务，`M/G/1` 的 Pollaczek–Khinchine 直觉很有用：

\[
E[W_q] = \frac{\lambda E[S^2]}{2(1-\rho)}.
\]

`E[S^2]` 包含长 prompt、长输出造成的平方放大。于是截断超长请求、按 token 预算而非 request 数 batching、或把长输出隔离到专用池，往往比单纯提高平均 FLOPS 更能改善 p99。公式假设稳态单服务器，不能直接替代真实 scheduler 仿真，但能解释为什么流量一抖尾延迟就爆。

### 35.1.5 SLO 的分母和预算

若团队承诺“TTFT p95 < 800 ms，ITL p99 < 80 ms，E2E p95 < 8 s”，还必须写：

- 统计窗口（例如 28 天）和请求样本最小数；
- 只计入已接受请求还是包括 429/503；
- streaming 取消、客户端断开、内部重试算成功还是失败；
- 每个模型版本、区域、租户的分片方式；
- 以服务端单调时钟还是客户观测时钟为准；
- error budget 消耗后的降级和发布门禁。

Prometheus 直方图建议保留固定 bucket，不把每个 `request_id`、prompt hash 或高基数租户名称当 label；参考 Prometheus 的 histogram 说明（https://prometheus.io/docs/practices/histograms/）。OpenTelemetry GenAI 语义约定（https://opentelemetry.io/docs/specs/semconv/gen-ai/）可作为字段起点，但必须审计隐私和采样。

## 35.2 心智模型：两条流水线、一个有界状态机

### 35.2.1 请求状态机

把请求画成有界状态机，而不是一个函数调用：

```text
NEW -> ADMITTED -> WAIT_PREFILL -> PREFILLING -> READY_DECODE
  |        |            |               |              |
  |        +--reject    +--deadline     +--error       +--cancel
  v                                                    v
REJECTED                                             CANCELED
                                                       |
READY_DECODE -> DECODING --stop/max_tokens--> COMPLETED
      |              |
      +--backpressure-+--client_disconnect--> ABORTED
```

每条边都应有计数器和原因。`ADMITTED` 只代表占用了配额，不代表 GPU 已开始；`CANCELED` 需要知道取消是在 prefill 前、prefill 中还是 decode 间隙生效；`ABORTED` 可能是客户端断开，也可能是服务端主动熔断。重试只允许在尚未向客户端交付不可逆 token 且请求幂等契约成立时进行。

### 35.2.2 prefill 与 decode 的资源形状

Prefill 对输入 prompt 做一次并行计算，算力和显存带宽压力较大，工作量近似随 `prompt_tokens × layers` 增长；decode 每次只追加一个 token，但要读写整个活动 batch 的 KV，工作量随活动序列数和上下文长度增长。于是它们的最佳 batch、GPU occupancy 和尾延迟目标不同：把长 prefill 插入 decode loop 会造成 head-of-line blocking；只服务 decode 又会让新请求 TTFT 飙升。

用两个逻辑队列表示：`Qp` 是 prompt 等待队列，`Qd` 是正在逐 token 迭代的活动集合。Continuous batching 每个 decode 迭代都允许完成的请求离开、候选请求加入，而不是等待整个静态 batch。调度器仍要限制 `Σ prompt_tokens`、`Σ active_sequences`、KV 页数和每次迭代的显存峰值。

### 35.2.3 共享预算而不是共享幻想

每个请求在进入 `Qp` 前冻结一份预算：输入 token、最大输出 token、估计 KV bytes、预计服务 ms、deadline 和租户权重。预算是 admission 的近似上界，不是模型一定会用完的事实；模型提前 stop 后应返还未用的输出预算。一个简单的 KV 估算是：

\[
B_{KV} \approx 2 \times L \times H_{kv} \times d \times bytes(dtype) \times (T_{prompt}+T_{generated}).
\]

实际系统还会有 page/block 对齐、分组查询注意力和压缩格式，必须以引擎的 block manager 为准。若只用 request 数做并发限制，短 prompt 和 128k prompt 会抢同一把锁，容易 OOM 或制造不可解释的 p99。

### 35.2.4 调度器是控制回路

服务不是“队列加 GPU”，而是反馈控制：测量 queue depth、active tokens、KV free pages、decode step time、TTFT p95、reject rate 和 tenant debt；依据信号调整 admission、batch token cap、prefill/decode 配比和扩缩容。控制周期过短会抖动，过长会让 error budget 被突发流量吃掉。所有自动调参都要记录配置版本和原因，能在事故中回放。

## 35.3 机制：从 batching 到 SLO 的可执行设计

### 35.3.1 静态、dynamic 与 continuous batching

**静态 batching** 在固定窗口收齐请求，形状一致，执行路径简单，等待却不可控；一个慢请求会把 batch 中的短请求绑住。它适合离线或严格固定 shape 的工作负载。

**Dynamic batching** 在时间窗内收集请求，达到最大 batch 或超时就发起一次推理。Triton 的 dynamic batcher 文档（https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher）把 queue delay、preferred batch size 和 priority 作为可配置机制；Ray Serve 也提供 request batching（https://docs.ray.io/en/latest/serve/advanced-guides/dynamic-request-batching.html）。dynamic batching 对一次调用完成的模型有效，但生成模型仍会被最长序列拖住。

**Continuous batching / iteration-level scheduling** 把 decode 看成不断变化的集合：

```text
for each decode iteration:
    retire finished/cancelled sequences
    admit prefill-complete sequences under token + KV budget
    choose a fair subset of waiting prompts
    run one decode token for active set
    flush stream events and expose metrics
```

Orca 论文把 iteration-level scheduling 与 selective batching 作为 LLM serving 的核心（https://www.usenix.org/conference/osdi22/presentation/yu）；vLLM 论文说明 PagedAttention 与 continuous batching 的系统组合（https://arxiv.org/abs/2309.06180）。这些论文的硬件和版本不同，不能直接当成今日吞吐数字，但机制证据足以说明“每轮重排”为什么减少空洞。

实现时要记住三个边界：

1. batch 形状变化可能触发编译或 kernel 退化，必须有 shape bucket 或 CUDA graph 约束；
2. 新 prefill 不能无限插队，否则旧 decode 的 ITL 违反 SLO；
3. retokenize、stop string、grammar constraint 等 CPU 工作也可能成为每轮瓶颈。

### 35.3.2 队列策略：FIFO、优先级和公平

FIFO 最容易解释，但长 prompt 可能产生 convoy：排在前面的一个 16k prompt 把 20 个短请求推迟。优先级队列让付费或交互请求更快，但没有 aging 会饿死低优先级。推荐把“优先级”和“服务债务”分开：有效分数可以写成

\[
score = base\_priority + age\_ms/age\_scale - tenant\_debt/weight.
\]

每租户债务按已消耗 prompt/decode token 或 GPU ms 累计，完成请求后衰减；权重大的租户获得更多份额，但仍受每租户并发与 KV cap 限制。Weighted Fair Queuing 的理想模型假设可切分包，decode token 是离散工作，实际只能近似。实验中应报告 Jain fairness index：

\[
J(x)=\frac{(\sum_i x_i)^2}{n\sum_i x_i^2},\quad 1/n\le J\le1.
\]

这里的 `x_i` 应是每租户获得的归一化服务量（例如 token 或 GPU ms / 权重），不能把原始请求数硬套进去。

### 35.3.3 Admission control：在队列前拒绝

Admission 是 SLO 的第一道闸。一个可审计的决策顺序：

1. 校验鉴权、模型版本、prompt 上限、输出上限和请求 deadline；
2. 估计 KV bytes、prefill ms 和最大 decode token，检查全局余量；
3. 检查租户并发、速率、token bucket 与 daily budget；
4. 根据当前 TTFT/ITL error budget 选择接受、排队、429（retry-after）或 503（无容量）；
5. 写入 `admission_epoch、quota_snapshot、rejection_reason`。

拒绝必须是可重试语义：429 表示客户端可依据 `Retry-After` 退避，503 表示服务暂时不可用；不要为了“看起来成功”接受后再在 GPU OOM。HTTP 语义参考 RFC 9110（https://www.rfc-editor.org/rfc/rfc9110）与 KServe data plane（https://kserve.github.io/website/latest/modelserving/data_plane/）。

Token bucket 的容量 `B` 和补充率 `r` 决定突发：请求消耗 `prompt_tokens + α·max_new_tokens` 个 token，只有 bucket 有足够余额才 admission。α 可按历史 stop ratio 调整，但改变它必须记录配置版本，避免租户通过声明很小的 max output 绕过预算。

### 35.3.4 deadline、超时和重试

服务端 deadline 应从请求头/协议映射成单调时间点，不要每层都重新加“60 秒”。gRPC deadlines/cancellation 指南（https://grpc.io/docs/guides/deadlines/、https://grpc.io/docs/guides/cancellation/）强调跨进程传播和及时取消；HTTP 客户端断开则要映射为内部 cancellation token。deadline 到期时：

- 尚未 prefill：从 `Qp` 删除并返还 admission 预算；
- prefill 中：是否能中断 kernel 取决于引擎，不能假装立即释放；至少标记“取消待生效”；
- decode 中：在安全迭代边界停止，释放 KV 页并发送一次终止事件；
- 已发出 token：客户端不得把重试结果与旧 stream 自动拼接，除非协议带 request id 和去重序号。

对生成请求默认采用 at-most-once 交付更安全；若业务需要重试，使用幂等 key、模型 revision、采样 seed 和已发 token 序号，明确重复费用和语义。

### 35.3.5 prefill/decode disaggregation

Disaggregation 将 prefill worker 与 decode worker 分离。prefill 计算后要把 KV cache 传到 decode worker；DistServe（https://arxiv.org/abs/2401.09670）和 Splitwise（https://arxiv.org/abs/2311.18677）讨论了利用两类阶段差异做容量规划，Sarathi-Serve（https://arxiv.org/abs/2403.02310）讨论 chunked prefill 与调度。设计时至少明确：

- KV layout、dtype、page size、layer/TP rank 映射和版本 digest；
- 传输协议是 NVLink、RDMA、共享内存还是网络，是否可校验和重传；
- prefill 成功但 KV 传输失败时，请求回退到本地 worker 还是取消；
- decode worker 的接收队列是否参与 admission，避免 KV in-flight 爆炸；
- 两池之间的容量比如何由 prompt/decode token 比例和 SLO 反馈调整。

如果 KV 传输时间 `T_kv` 大于本地 prefill 节省，解耦反而增加 TTFT。把 `T_prefill、T_transfer、T_decode_ready` 分开打点；不要把 transfer 隐藏在“prefill 完成”事件里。Llumnix 的迁移与负载均衡研究（https://arxiv.org/abs/2406.04837）也说明 KV 搬迁必须考虑中断和一致性，不是无成本的调度动作。

### 35.3.6 chunked prefill 与 decode 保护

长 prompt 可以切成多个 chunk，在每个 chunk 间让 decode 迭代运行，降低 head-of-line。chunk 大小过小会增加 kernel launch 和调度开销，过大又阻塞 decode。一个可操作的控制规则是：若最近 `ITL_p99 > budget`，缩小 prefill chunk 或暂停新 prefill；若 `TTFT_p95 > budget` 且 decode slack 足够，扩大 chunk 或增加 prefill 槽位。配置必须有上下界和 hysteresis，避免两个指标互相拉扯。

### 35.3.7 stream protocol：SSE、HTTP/2 与 gRPC

SSE 用 `text/event-stream` 和事件帧发送单向 token，浏览器标准说明见 MDN（https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events）；HTTP/2/3 的流控和取消仍适用，协议可参考 RFC 9113（https://www.rfc-editor.org/rfc/rfc9113）和 RFC 9114（https://www.rfc-editor.org/rfc/rfc9114）。一个稳健的生成 stream 至少包含：

```text
event: token\ndata: {"request_id":"r-123","seq":7,"text":"...","usage":null}

event: done\ndata: {"request_id":"r-123","seq":8,"finish_reason":"stop","usage":{"input_tokens":42,"output_tokens":8}}
```

`seq` 让客户端检测重复/缺口；`done` 必须幂等，连接断开后服务端保留短暂终态以回答重连查询。禁止把每个 token 拼成无限大的单条 data；按 token 或字节上限 flush，加入 heartbeat 以区分空闲与断链。完整输出模式仍应返回 usage、model revision 和 stop reason，流式与非流式的账单口径一致。

### 35.3.8 取消、断链和背压

生成速度可能快于客户端消费或网络发送。发送队列必须有界：当 buffer 到达 `N` tokens/bytes 时，生产者阻塞、降低 decode 优先级，或按策略丢弃尚未交付的 token 并终止请求；不能无限占用 Python 内存。对模型 token，通常选择“不丢中间 token，直接 cancel”而不是静默丢字。

取消路径要从 socket 到 scheduler 贯通：

```text
client FIN/RST or explicit cancel
 -> gateway cancel token
 -> router removes queued retry
 -> engine marks sequence cancelled
 -> next safe decode boundary frees KV
 -> stream emits one terminal event (if writable)
 -> metrics record cancel_requested/applied latency
```

若客户端已断开，写 terminal event 可能失败，但内部仍要释放资源。取消延迟（request 到 KV free）应单独设 SLO，因为它直接影响其他租户的排队。

### 35.3.9 多租户隔离：配额、优先级和 noisy neighbor

隔离至少有四层：

1. **入口层：** API key、并发、QPS、prompt/output token bucket；
2. **调度层：** per-tenant active sequences、weighted fair、最大等待时间；
3. **资源层：** KV page quota、GPU/副本池、prefill/decode 配比；
4. **观测层：** 每租户 SLO、错误预算、成本和审计事件，避免只有全局平均。

租户可以拥有不同 SLO，但不能让“金牌”无限吃掉所有容量。建议为保底份额 `min_share`、突发上限 `burst_share` 和全局保护线写成配置，发生压力时按降级阶梯执行：先拒绝超额新请求，再降低低优先级最大输出，最后才暂停可抢占租户；已经交付的 stream 不应被无提示地截断。

Kubernetes API Priority and Fairness（https://kubernetes.io/docs/concepts/cluster-administration/flow-control/）提供了控制面请求公平的参考；Envoy circuit breaking（https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/circuit_breaking）展示了并发/连接/重试预算的隔离思路。它们不是 LLM scheduler 的直接实现，但有助于把“公平”落成可配置的并发和债务。

### 35.3.10 观测：从阶段 span 到 SLO burn rate

每个请求至少有一个 root span 和四个阶段 span：`admission、prefill、kv_transfer、decode`，stream 则加 `serialize/flush`。指标分三类：

- **饱和度：** Qp/Qd depth、active sequences、KV used/free、prefill/decode queue、GPU busy；
- **延迟：** TTFT/ITL/E2E p50/p95/p99/max、cancel-to-free、stream flush gap；
- **结果：** accepted/rejected/cancelled/completed、stop reason、tokens、retries、429/503。

SLO 查询要按模型 revision、区域和租户切片，但控制高基数。日志记录决策（为何选该 batch、为何拒绝、债务和预算快照），不要记录完整 prompt。错误预算 burn rate 可采用短窗/长窗双阈值，避免单个尖峰触发扩容风暴；报警必须关联可执行动作（停发布、调整 batch cap、切换回退池）。

## 35.4 实验：CPU-only 调度 toy 的协议证据

### 35.4.1 实验目标和边界

`labs/ch35_serving_scheduling_lab.py` 只使用 Python 标准库，刻意把时间代理写成可读的小数毫秒。它验证：

- arrival、prompt token 和 max output 约束；
- FIFO/priority/weighted fair 的 batch 选择；
- 最大请求数和 prompt token 预算；
- coupled 与 disaggregated prefill 的阶段时间差；
- TTFT、ITL、E2E 的定义和 p50/p95 汇总；
- admission 全局/租户上限、明确拒绝原因；
- cancellation、stream bounded buffer 和 producer blocked 证据；
- 所有请求最终 `completed` 或 `cancelled`，不产生负延迟。

代理数字不能回答“某张 GPU 每秒多少 token”。要做生产 benchmark，需固定引擎 commit、CUDA/driver、模型权重、TP/PP、KV dtype/page、请求 trace、冷/热 cache 和网络观测。

### 35.4.2 运行命令

```bash
python3 -m py_compile labs/ch35_serving_scheduling_lab.py
python3 labs/ch35_serving_scheduling_lab.py --output reports/ch35-serving-scheduling-default.json
python3 labs/ch35_serving_scheduling_lab.py --burst --disaggregated --policy weighted_fair \
  --output reports/ch35-serving-scheduling-disagg.json
python3 tests/test_ch35_serving_scheduling_lab.py
```

默认 workload 有 gold/bronze 两个租户、不同 prompt/output、一次在 decode 前取消的请求；burst workload 让队列和公平策略更明显。输出 JSON 保存每个请求的 `prefill_start/finish、first_token、finish、ttft/itl/e2e、status`，同时保存租户 p95 和不变量。

### 35.4.3 结果如何读

默认 coupled 运行中，prefill 的 `2.7 ms` 代理成本先于 decode；同一批请求共享 decode 轮次，因此短输出请求完成后，活动集合缩小、后续 ITL 代理下降。`r2` 在取消时间到达时标为 `cancelled`，它不应被计入 completed TTFT p95。disaggregated 模式把新 prefill 的 `ready_at` 与已有 decode 时间线重叠，通常减少等待，但不是所有 workload 都更好：当 prefill 持续很长或 decode worker 已满，ready queue 可能把 transfer/排队推迟。

比较策略时不要只看全局 p95：读取 `tenant_stats`，检查 bronze 是否长期无完成；再改变 `tenant_weights` 和 `max_batch_prompt_tokens`，观察公平与 TTFT 的 trade-off。若把 token cap 从 32 降到 16，首轮更小，短 prompt 可能更快，但长 prompt 需要多个批次；这对应真实系统中的 chunking 和 scheduler overhead。

### 35.4.4 一个可重放的队列账本

每次实验至少保存：

```json
{
  "config_digest": "...",
  "traffic_seed": 35,
  "engine_revision": "toy-cpu-only",
  "slo": {"ttft_p95_ms": 8.0, "itl_p99_ms": 2.0},
  "admission": {"max_inflight": 8, "per_tenant": {"gold": 4, "bronze": 4}},
  "requests_sha256": "...",
  "decision_log": ["r0 admitted", "r2 cancelled at 4.0ms"]
}
```

`requests_sha256` 只证明 trace 身份，不应泄漏 prompt。生产回放另需记录采样率、脱敏规则和原始数据的访问权限。

## 35.5 故障诊所/失败：把症状还原成机制

### 35.5.1 症状一：TTFT p99 暴涨，GPU 利用率却不高

**可能链路：** admission 放行了太多长 prompt，Qp 等待；tokenizer/HTTP worker 成为瓶颈；prefill chunk 受 CPU 调度影响；GPU kernel 在小 batch 上 launch 过碎。

**诊断顺序：** 先画 `queue_wait、tokenize、prefill、kv_transfer` 的分位数，再看 active token 与 KV free。若 queue_wait 占大头，先收紧 admission 或按 prompt bucket 分流，不要先把 GPU 型号换大。若 tokenize 占大头，隔离 CPU 线程并设置每租户 prompt 上限；若 KV transfer 占大头，检查 page/layout 和网络重传。

**安全修复：** 临时把 max prompt token cap 降低、拒绝超额 429 并带 retry-after；保留旧配置以便回滚。只调大 batch size 可能让 TTFT 更差。

### 35.5.2 症状二：TTFT 正常，ITL p99 周期性尖峰

**常见原因：** 长 prefill 插入 decode；CPU stop/grammar 处理阻塞主循环；KV page eviction/迁移；下游 stream buffer 满触发 backpressure；某租户 burst 占满活动序列。

**验证：** 对尖峰记录 `decode_iteration_id、active_seq、prefill_tokens、kv_free_pages、flush_gap_ms、tenant_debt`。如果每次尖峰都和 prefill chunk 对齐，缩小 chunk 或启用 decode 保护；如果只在某租户出现，检查 per-tenant active cap；如果服务端 token 时间正常而客户端 gap 大，问题在网关/网络背压而不是模型。

### 35.5.3 症状三：取消率上升但 KV 没释放

**常见错误：** 取消只停了 HTTP response，没有把 token 传到 engine；engine 在 kernel 中途不能抢占，metrics 却把请求标成 cancelled；stream 关闭后重试仍保留旧 request。修复应包括 `cancel_requested` 与 `cancel_applied` 两个时间，增加 `cancel_to_kv_free` 指标，并在 active set 移除时断言 page 归还。若只能在迭代边界取消，SLO 和文档要写清最坏等待。

### 35.5.4 症状四：某租户 SLO 好，全局公平却差

**原因：** 只按优先级排序；weighted fair 的债务单位混用 request 数与 token；重试流量未计入 quota；取消前已消耗的 prefill 没算成本。

**修复：** 选择一个资源单位（prompt token、decode token 或 GPU ms），全链路使用；对 rejected/retried 请求记录原始 tenant；加 aging 防止低权重饿死；按租户报告 Jain index 和保底份额。公平不是每租户相同延迟，而是在权重、SLO 和可用容量下可解释。

### 35.5.5 失败实验：无界队列和“先接收再想办法”

一个常见反模式是网关返回 200，服务内部把请求放入无界 Python queue，等 GPU 有空。短时间看起来拒绝率为零，突发后进程 OOM，所有租户一起失败。第二个反模式是接受后才发现 KV 不足，靠 `torch.cuda.OutOfMemoryError` 作为 admission。实验中应注入：队列上限、KV 预估误差、prefill 失败、stream consumer 变慢、取消风暴；验证每种失败都产生明确的 429/503/terminal event，而不是沉默等待。

### 35.5.6 失败实验：把 p99 当平均值过滤掉

若只导出一个 `latency_avg_ms`，长 prompt 和取消请求会被掩盖。采用 fixed histogram buckets，检查样本数和 bucket 覆盖；对少量租户不要在指标中直接暴露个人标识。每次变更前后固定同一 trace 和 warm-up，报告 bootstrap 置信区间或至少重复次数。p99 只有在窗口内样本足够时才有意义，样本不足应显示 `insufficient_data`。

## 35.6 理解检查

1. 为什么 `TTFT p95` 低并不意味着 `E2E p95` 低？请指出至少两个阶段。
2. 在 Little 定律中，`Lq`、`λ`、`Wq` 各自是什么？当输出长度方差变大时，哪个二阶项会放大尾延迟？
3. continuous batching 与 dynamic batching 的边界在哪里？为什么生成模型需要 iteration-level retire/admit？
4. 若 prefill/decode 解耦后 TTFT 变差，应该先检查 `T_kv`、接收队列还是 decode kernel？请说明证据。
5. 一个租户权重为 2，另一个为 1，为什么不能简单承诺前者请求数一定是后者两倍？
6. 客户端断开后，服务端为什么仍要执行 cancel-to-KV-free？不这样做会影响哪个 SLO？
7. SSE 的 `seq` 和 `done` 事件如何帮助客户端处理重复、缺口和重连？
8. 429、503 和服务端 deadline 的语义各是什么？重试何时会产生重复计费或重复生成？
9. 为什么 prompt token cap 是 admission 的必要但不充分条件？还要预算哪些资源？
10. 你会为 ITL p99 设置哪个阶段的 trace 字段，来区分模型慢与网络 flush 慢？

## 35.7 练习

### 练习 A：从 SLO 反推容量

给定 30 req/s、平均 prompt 500、平均输出 120、`E[S]=180 ms`、`c=8`，先计算近似利用率和 Little 定律下的平均在途数。再假设输出长度的 p99 是均值的 4 倍，讨论为什么只按平均值配置并发会失效。写出至少一个 admission 规则和一个隔离长请求的方案。

### 练习 B：设计 coupled/disaggregated A/B

固定同一脱敏 trace，比较 coupled、chunked prefill、disaggregated 三种配置。要求报告 TTFT/ITL/E2E p50/p95/p99、KV transfer bytes、cancel-to-free、每租户公平指数和 GPU/CPU 饱和度。定义停止条件：若任一租户 rejection > 5% 或 ITL p99 超预算两次窗口，实验自动回滚。

### 练习 C：流式协议审查

为一个 SSE endpoint 写状态机：token、heartbeat、done、error、cancel、reconnect。指定序号、最大 frame bytes、buffer 上限、重连窗口和幂等 key。列出客户端已收到 seq=7、服务端只持久 seq=6 时应该返回什么，而不是重复 token。

### 练习 D：公平调度器

实现 weighted fair + aging 的 `_pick_batch`，资源单位使用 decode token-ms。构造一个 gold 长输出租户和三个 bronze 短输出租户，证明无 aging 时 bronze 会饥饿，加入 aging 后 Jain index 提高但 gold TTFT 变化。把每次决策写入可重放 JSON。

### 练习 E：故障注入 runbook

注入以下故障：KV transfer 丢包、prefill worker 崩溃、decode worker OOM、客户端半关闭连接、Prometheus scrape 延迟。对每个故障写“检测指标—安全动作—用户可见结果—回滚条件—需要保留的证据”，并标注哪些动作可自动化，哪些必须人工确认。

## 35.8 小结：SLO 是调度器的外部契约

在线推理的核心不是找到一个神奇 batch size，而是让到达、排队、prefill、decode、流式传输和取消都成为有界、可观测、可回放的状态机。TTFT 描述首 token 的等待，ITL 描述连续输出的节奏，E2E 描述完整生命周期；三者必须以统一分母和阶段时间戳进入 SLO。

Continuous batching 能减少静态 batch 的空洞，但它把每轮选择、KV 预算、公平、取消和 stream 背压的责任交给 scheduler。Prefill/decode disaggregation 可能提高利用率，也可能因 KV 传输和容量错配增加 TTFT；只有把 transfer 和失败回退写进协议，才算完成设计。Admission control 要在 GPU OOM 之前拒绝，流式协议要能表达序号与终止，多租户要同时保护保底份额和全局容量。

CPU toy lab 的价值在于把这些约束变成测试合同：所有请求终态可解释，预算超限会拒绝，取消会落账，公平策略有可比输出。生产系统还要补上真实硬件、网络、编译 kernel、模型安全和数据治理证据；没有这些边界声明，任何“快了 20%”都不应进入发布结论。

## 35.9 来源

### 论文与系统

- Orca: A Distributed Serving System for Transformer-Based Generative Models：https://www.usenix.org/conference/osdi22/presentation/yu
- vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention：https://arxiv.org/abs/2309.06180
- DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving：https://arxiv.org/abs/2401.09670
- Splitwise: Efficient Generative LLM Inference Using Phase Splitting：https://arxiv.org/abs/2311.18677
- Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve：https://arxiv.org/abs/2403.02310
- Llumnix: Dynamic Scheduling for LLM Inference with SLO Guarantees：https://arxiv.org/abs/2406.04837
- FastServe: Unlocking the Potential of GPU Acceleration in LLM Serving：https://arxiv.org/abs/2305.05920
- Clockwork: Predictable and Efficient Cloud Resource Provisioning for Deep Learning Inference：https://www.usenix.org/conference/osdi20/presentation/gujarati

### 引擎、协议与官方文档

- vLLM documentation：https://docs.vllm.ai/
- vLLM continuous batching source：https://github.com/vllm-project/vllm
- SGLang documentation：https://docs.sglang.ai/
- NVIDIA Triton dynamic batching：https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher
- Ray Serve dynamic request batching：https://docs.ray.io/en/latest/serve/advanced-guides/dynamic-request-batching.html
- KServe inference data plane：https://kserve.github.io/website/latest/modelserving/data_plane/
- gRPC deadlines：https://grpc.io/docs/guides/deadlines/
- gRPC cancellation：https://grpc.io/docs/guides/cancellation/
- Server-sent events (MDN)：https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- HTTP Semantics RFC 9110：https://www.rfc-editor.org/rfc/rfc9110
- HTTP/2 RFC 9113：https://www.rfc-editor.org/rfc/rfc9113
- HTTP/3 RFC 9114：https://www.rfc-editor.org/rfc/rfc9114
- Kubernetes API Priority and Fairness：https://kubernetes.io/docs/concepts/cluster-administration/flow-control/
- Envoy circuit breaking：https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/circuit_breaking
- Prometheus histograms：https://prometheus.io/docs/practices/histograms/
- OpenTelemetry GenAI semantic conventions：https://opentelemetry.io/docs/specs/semconv/gen-ai/

### 队列论与测量边界

- Little's Law overview (MIT OCW)：https://ocw.mit.edu/courses/15-072-queuing-theory-and-applications-spring-2006/
- Pollaczek–Khinchine queueing reference (Wikipedia for notation only)：https://en.wikipedia.org/wiki/Pollaczek%E2%80%93Khinchine_formula
- Jain fairness index：R. Jain, D.-M. Chiu, W. Hawe, https://www.cse.wustl.edu/~jain/papers/ftp/fairness.pdf
- OpenTelemetry tracing concepts：https://opentelemetry.io/docs/concepts/signals/traces/

### 本章本地证据

- CPU-only lab：https://github.com/openai/Beneath-the-Tokens/blob/main/labs/ch35_serving_scheduling_lab.py
- Toy contract tests：https://github.com/openai/Beneath-the-Tokens/blob/main/tests/test_ch35_serving_scheduling_lab.py
- Experiment report：https://github.com/openai/Beneath-the-Tokens/blob/main/reports/ch35-serving-scheduling-report.md

所有外部 URL 都需要在目标版本、目标引擎和目标硬件上重新核验；论文中的吞吐/延迟数字不作为本章 CPU toy 或生产 SLO 的承诺。
