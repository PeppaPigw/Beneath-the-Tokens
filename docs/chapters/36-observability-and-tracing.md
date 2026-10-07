---
id: ch36-observability-and-tracing
title: AI Infra 可观测性与性能诊断：Tracing、Profiling、eBPF 与 GPU 遥测
slug: /chapters/36-observability-and-tracing
description: 用 OpenTelemetry、Prometheus、DCGM、Nsight、PyTorch Profiler 与 eBPF 建立跨服务、跨 GPU、跨网络和存储的可审计诊断闭环
sidebar_position: 36
level: advanced
prerequisites:
  - ch20-observability-debugging-incident-response
  - ch21-ai-reliability-engineering
  - ch32-gpu-cluster-topology
  - ch33-storage-and-data-plane
  - ch35-serving-scheduling-and-slo
learning_objectives:
  - 能区分 traces、metrics、logs、profiles 和事件证据的时间语义与采集代价
  - 能用 OpenTelemetry context propagation 将网关、队列、GPU kernel、网络 RPC 和存储 I/O 关联到同一条 trace
  - 能设计 Prometheus 指标、histogram bucket、采样策略和 cardinality 预算，避免 telemetry 本身拖垮控制面
  - 能解释 DCGM、Nsight Systems/Compute、PyTorch Profiler、eBPF 各自观察的层次、盲区和合规边界
  - 能使用 tail latency、排队分解、GPU 利用率和网络/存储信号定位跨层瓶颈
  - 能运行 CPU-only toy lab，生成确定性的 span、metric、log、profile 和故障注入证据
  - 能在事故复盘中给出时间线、假设、反事实实验、修复和证据保留期限
estimated_hours: 46
hardware: CPU-only toy lab; production GPU evidence requires pinned driver, firmware, CUDA, model and traffic replay
risk_level: L3
last_verified: 2026-10-07
---

# 第36章　AI Infra 可观测性与性能诊断：Tracing、Profiling、eBPF 与 GPU 遥测

> 一个 p99 变坏的推理服务，未必是“GPU 不够快”。请求可能在网关重试，队列排队可能占掉 TTFT，GPU kernel 可能被 CPU launch 间隙打断，NCCL collective 可能等待最慢的 rank，NVMe 可能在 checkpoint flush 时抖动，最后客户端看到的只是一个超时。可观测性不是把所有东西都打进日志，而是为每个假设准备低成本、可关联、可重放的证据。

本章把可观测性视为一个受预算约束的测量系统。我们先定义问题和边界，再建立 traces、metrics、logs、profiles、eBPF 和 GPU telemetry 的心智模型；随后给出 OpenTelemetry（OTel）语义、Prometheus 指标设计、DCGM/Nsight/PyTorch Profiler 的取舍，最后在 CPU-only toy lab 中复现 trace correlation、tail latency、采样与 cardinality、GPU/网络/存储联合诊断以及一次完整事故复盘。实验结果只是协议和算法的局部证据，不是对任何型号 GPU、驱动或生产集群的性能承诺。

## 36.1 问题/边界：先回答“要证明什么”

### 36.1.1 可观测性问题不是“看更多图”

一个诊断请求应写成可证伪的命题：

- “p99 TTFT 上升是队列等待，而不是 prefill FLOPS 下降”；
- “只有跨可用区的 RPC 触发尾延迟，GPU kernel 时间保持不变”；
- “rank 3 的 all-reduce 等待由某个 NVLink 重试引起”；
- “checkpoint 时段 NVMe fsync 与 page cache 回写造成数据加载抖动”；
- “trace 丢失是 tail sampling collector 丢弃，还是 context propagation 断链”。

每个命题都要有一条主证据和至少一条反证。只展示平均 GPU 利用率，不能证明没有 memory throttle；只展示一条成功 trace，也不能证明错误路径不会泄漏 prompt。

### 36.1.2 本章的边界

本章覆盖在线推理、训练作业和数据面服务中的测量与诊断接口：OTel traces/metrics/logs，Prometheus 拉取和远端写入，DCGM GPU 计数器，Nsight Systems/Compute 时间线，PyTorch Profiler 算子与内存视图，eBPF 内核/网络/IO 观测，以及跨层 correlation。它不替代安全审计、模型质量评测、成本核算或容量规划章节；也不声称 eBPF 能读取 CUDA kernel 内部每个线程的寄存器状态，或 DCGM 能解释应用级 token 语义。

GPU 计数器会受驱动、MIG、权限和固件影响；Nsight 采样可能改变时序；PyTorch Profiler 的 `record_shapes`、内存和 stack trace 会放大开销；eBPF 程序受 verifier、内核版本、map 大小与 attach 点限制。生产启用前必须 pin 版本、评估 overhead、定义采样窗口，并保留停止开关。

### 36.1.3 信号的时间语义

- Trace span 是有开始/结束和父子关系的区间，适合回答“谁等待了谁”。它天然是请求级或操作级样本，不能直接等价于所有请求的分布。
- Metric 是按时间序列聚合的数值，适合告警和趋势。Counter 只能递增，Gauge 可上下波动，Histogram 需要 bucket 设计；每个 label 组合都会产生一条时序。
- Log 是离散事件和结构化上下文，适合异常细节与审计，但如果每个 token 都打一行，会制造 I/O 和 cardinality 风暴。
- Profile 是采样或插桩得到的栈/算子/硬件计数器快照，适合回答“CPU/GPU 时间花在哪里”，通常只在短窗口启用。
- Event 是部署、扩缩容、驱动重置、OOM、网络链路变化等边界标记；它们让多条 signal 在同一时间线对齐。

时间戳必须来自可比较的时钟。跨主机用 NTP/PTP 校准 wall clock 仅用于展示，span duration 和队列时延计算应使用 monotonic clock。记录 `clock_offset_estimate`，否则跨节点 2 ms 的“负时延”会被误判为软件 bug。

## 36.2 心智模型：五层测量平面

把 AI Infra 画成五层，每层既有自己的真相，也有与邻层的接口：

1. **请求层（intent）**：用户请求、租户、模型版本、deadline、token 数和错误码。主信号是 root span、SLO histogram 和结构化 access log。
2. **编排层（orchestration）**：队列、重试、批处理、路由、限流和 worker lease。主信号是 queue depth、admission reason、retry counter 和 child spans。
3. **执行层（execution）**：CPU 调度、CUDA launch、kernel、NCCL collective、allocator、线程池。主信号是 profiles、GPU activity trace、CPU run queue 和 runtime event。
4. **资源层（resource）**：GPU SM/显存/温度/功耗、NVLink/PCIe、NIC、磁盘、文件系统、页缓存。主信号是 DCGM、node exporter、eBPF、SMART 和 cgroup metrics。
5. **控制与证据层（control/evidence）**：collector、Prometheus、对象存储、采样策略、保留期、访问控制和脱敏。主信号是 telemetry pipeline 自身的 dropped spans、queue overflow、remote-write lag、权限拒绝。

诊断沿着“因果边”而非“组件列表”走：请求 span 的 `queue.wait_ms` 应能链接到 worker span；worker span 的 `gpu.batch_id` 应能链接到 Nsight range；batch 的 `device_uuid` 应能链接到 DCGM 时间序列；如果写入存储，则链接到 eBPF block I/O 区间。任何断边都要被视为观测缺口，而不是默认为“没有问题”。

### 36.2.1 四种 ID 的职责

- `trace_id`：一次端到端请求或作业内逻辑事务的关联 ID，长度和格式按 W3C Trace Context。
- `span_id`：一个操作区间的本地 ID，不应被当作业务主键。
- `request_id`/`job_id`：业务重试、幂等和审计的稳定标识，可在多个 trace 之间保持不变。
- `exemplar_id`：从 metric histogram 指向一个代表性 trace 的短引用。它让告警图能跳到样本，而不把所有 trace ID 变成 label。

不要把 `prompt_text`、完整 URL query、用户 email 或每个 token 序号当成 metric label。若需要排查特定请求，把经过授权的 hash 放入 log 或 span event，并设置 TTL。

## 36.3 机制：OpenTelemetry traces、metrics、logs

### 36.3.1 OTel 的最小协议

OTel API/SDK 负责在应用内创建 span、meter 和 logger；OTLP exporter 负责把信号送到 Collector；Collector 通过 receiver、processor、exporter 组成管道，可执行批处理、尾采样、属性删除、限速和多目的地路由。官方概念说明见 [OTel traces](https://opentelemetry.io/docs/concepts/signals/traces/)、[OTel metrics](https://opentelemetry.io/docs/concepts/signals/metrics/) 和 [OTLP specification](https://opentelemetry.io/docs/specs/otlp/)。

一个推理请求的 span 树可如下：

```text
server /v1/generate
├── auth.verify
├── scheduler.queue_wait
├── tokenizer.encode
├── model.prefill
│   └── cuda.launch (多个 event，不必每个 kernel 一个 span)
├── model.decode_loop
│   ├── nccl.all_reduce
│   └── kv_cache.append
├── stream.flush
└── response.serialize
```

并不是每个函数都应成为 span。span 的目标是解释跨边界等待和失败；纯 CPU 内联函数用 profile 更便宜。对于 decode loop，可把每 N 个 token 聚合成一个 span，并用 events 记录 `token_count`、`gpu_kernel_ms`、`queue_depth`，同时保留一个 token-level histogram。

### 36.3.2 Context propagation

HTTP 使用 `traceparent`/`tracestate`；gRPC 通过 metadata；Kafka、NATS 等消息系统把 context 放入 headers。生产代码必须在异步任务、线程池、GPU callback 和 actor mailbox 间显式传递 context，避免在全局变量中保存当前 span。跨服务边界要规定：

1. 接收端提取 context，创建 server/consumer span；
2. 出站端注入 context，创建 client/producer span；
3. 重试是否复用同一个 `request_id` 但新建一个 child span；
4. fan-out 使用 links 还是单一 parent；
5. 跨队列等待如何记录 enqueue/dequeue 时间。

一个常见错误是仅在 HTTP handler 建 root span，worker 从队列取任务时重新生成新的 trace。这样网关显示 20 ms，worker 显示 300 ms，却无法证明二者属于同一请求。正确做法是把序列化后的 context 和 `request_id` 一起放入队列消息，并在出队处恢复。

### 36.3.3 语义约定与隐私

优先采用 OTel semantic conventions，例如 `server.address`、`rpc.system`、`db.system`、`http.response.status_code`；生成式 AI 相关字段可参考 [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)，但在稳定前要锁定版本和自定义前缀。将模型版本、量化配置、并行度作为低基数 resource attributes；将 prompt 长度、输出 token 数作为 span attributes 或 histogram observation；不要将 prompt 本体作为 attribute。

脱敏处理放在 Collector processor 和应用 SDK 两层：应用层避免产生秘密，Collector 层删除未知高风险字段。设置 `attribute_limit`、`event_limit`、`span_limit`，超过上限时计数并采样，而不是无声截断。对日志启用结构化 JSON，保证 `trace_id`、`span_id`、`severity`、`timestamp` 和 `component` 字段固定。

## 36.4 Prometheus metrics：聚合、histogram 与 cardinality

### 36.4.1 指标命名和类型

Prometheus 建议以 `_total` 表示 Counter，以 `_seconds`、`_bytes` 表示单位。AI 服务的最低指标集：

- `inference_requests_total{model,route,status_class}`：请求计数，禁止 request_id label；
- `inference_latency_seconds{model,phase}`：Histogram，phase 为 admission/queue/prefill/decode/e2e；
- `inference_tokens_total{model,direction}`：生成与输入 token；
- `inference_queue_depth{queue,priority}`：Gauge；
- `inference_batch_size{model}`：Histogram；
- `telemetry_dropped_spans_total{pipeline,reason}`：Collector Counter；
- `gpu_duty_cycle_ratio{device,partition}`、`gpu_memory_bytes{device,kind}`：来自 DCGM adapter；
- `node_network_receive_bytes_total{device}`、`node_disk_io_time_seconds_total{device}`：资源层 Counter。

`model`、`route`、`device` 往往是可接受的有限集合；`pod_uid`、`kernel_name`、`user_id`、`trace_id` 可能导致高基数。把 kernel 细节放在 profile 或 span event，通过 exemplars 从 histogram 跳转，而不是为每个 kernel 建时间序列。

### 36.4.2 Histogram、Summary 和尾部

Histogram 以 bucket counter 记录分布，聚合后可计算近似 quantile；Summary 在客户端计算 quantile，跨实例聚合困难。参考 [Prometheus histograms](https://prometheus.io/docs/practices/histograms/) 选择与 SLO 相关的 bucket，例如 TTFT 以 10、25、50、100、250、500、1000、2000 ms 为主；ITL 可能需要 5、10、20、40、80、160 ms。Bucket 不是越多越好：每个 label 组合会复制 bucket、sum、count，远端写入成本近似线性增加。

用 `histogram_quantile()` 读取 p95 时要注明窗口和聚合顺序：先按 `le` 聚合所有实例，再计算 quantile。窗口太短会让 p99 只有几个样本；窗口太长会掩盖部署后的变化。对极端尾部，保留 raw exemplars 或离线日志样本，避免把 bucket 当作精确真相。

### 36.4.3 Cardinality 预算

给每个团队和指标设置预算。例如每个服务最多 2,000 条 active series，单个 histogram 不超过 20 个 bucket×5 个稳定 labels；上线前用 `promtool tsdb analyze` 或 Prometheus API 估算。高基数失控的症状包括：Prometheus head block 内存持续增长、scrape 超时、remote-write queue backlog、查询 p99 变慢。处理顺序应是：删除不必要 labels，聚合为 route/model class，降低 scrape 频率，最后才增加资源。不要用 drop metrics 作为第一反应，因为它可能丢掉故障证据。

## 36.5 采样策略：head、tail、事件触发和自适应

### 36.5.1 为什么 100% trace 不现实

每个 span 含时间戳、属性和 links；在高 QPS 推理网关，100% 采集会放大 CPU、网络和存储。概率 head sampling 在 root 创建时决定是否采样，成本低但无法知道后续是否出错；tail sampling 需要 Collector 暂存整条 trace，再按错误、延迟、租户或 route 决定保留，成本高但更适合抓 p99 和失败。

建议分层：

- 100% metrics 和低成本 access log；
- 成功且低延迟 trace 以 1% head sample；
- 错误、超时、`latency > SLO`、GPU reset、NCCL error 由 tail sampler 强制保留；
- 发布、扩缩容、驱动升级后设短时高采样窗口；
- 对隐私敏感租户采用更低采样或仅保留聚合指标。

Tail sampler 要有内存上限和超时。Collector 崩溃时“保留全部慢 trace”并不可行，必须记录 `sampling_decision` 和 `dropped_reason`，让缺口可见。参考 [OTel tail sampling processor](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/processor/tailsamplingprocessor)。

### 36.5.2 Sampling 与 cardinality 的耦合

采样只减少样本量，不会自动减少 metrics cardinality。把 `trace_id` 放进 metric label，即使 trace 采样 1%，每个请求仍会生成一条新时序。相反，exemplar 把少量 trace ID 附在 histogram bucket 上，可以在不增加 series 的情况下导航到样本。高基数 span attributes 也会增加 Collector 内存，需通过 `probabilistic_sampler`、`filter` 和 `transform` processor 限制。

## 36.6 Profiling 机制：CPU、GPU 与算子时间

### 36.6.1 PyTorch Profiler

[PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) 能记录 CPU operator、CUDA kernel、内存活动和通信。推荐使用 `schedule(wait, warmup, active, repeat)` 与 `on_trace_ready`，仅在稳定窗口采集；将 `record_shapes`、`with_stack`、`profile_memory` 作为逐步开启的开关。导出 Chrome trace 或 TensorBoard 后，首先检查：

1. CPU launch gap：GPU stream 是否有大片空洞；
2. kernel fusion：小 kernel 是否被调度开销淹没；
3. allocator/`cudaMalloc`：是否出现同步和碎片；
4. dataloader：CPU worker、page fault 和 pinned memory 是否跟不上；
5. NCCL：collective 是否在等待最慢 rank。

Profiler 中的 `self_cpu_time_total`、`cuda_time_total` 是算子视角；不要与 DCGM 的设备利用率直接相加。异步 CUDA 调用若未同步，端到端计时会低估，需使用 CUDA event 或适当的 profiler correlation。

### 36.6.2 Nsight Systems 与 Nsight Compute

[Nsight Systems](https://docs.nvidia.com/nsight-systems/) 侧重系统时间线：CPU thread、CUDA API、kernel、NVTX range、NIC 和存储事件；[Nsight Compute](https://docs.nvidia.com/nsight-compute/) 深入单个 kernel 的 occupancy、memory throughput、warp stall 和 roofline。两者都应先用窄窗口和过滤器，否则报告过大且扰动时序。

在应用中用 NVTX range 标记 `prefill`、`decode_step`、`all_reduce`、`checkpoint_write`，将 `trace_id` 的短 hash 或 batch id 作为 message（不要写 prompt）。然后在 OTel span 的 event 中保存 NVTX range 名称和时间，形成“span→系统时间线→kernel counter”的导航。由于 Nsight 的时间戳和 OTel wall/monotonic clock 不同，校准方法要写入报告。

### 36.6.3 eBPF：内核和网络的旁路证据

[eBPF](https://ebpf.io/what-is-ebpf/) 程序在内核安全执行，可 attach 到 tracepoint、kprobe、uprobes、tc、XDP 和 LSM。常见工具链 [bcc](https://github.com/iovisor/bcc) 与 [bpftrace](https://github.com/bpftrace/bpftrace) 可快速回答：

- `runqlat`：CPU run queue 等待是否增加；
- `biolatency`/`biosnoop`：块 I/O 延迟和慢设备；
- `tcplife`/`tcpconnect`：连接建立、重传和生命周期；
- `offcputime`：线程在何处阻塞；
- 自定义 uprobes：用户态 RPC 库的开始/结束。

eBPF 观测应通过 ring buffer/perf buffer 发送聚合数据，避免逐包日志；map key 必须有生命周期和上限。内核版本、BTF、CAP_BPF/CAP_PERFMON 和容器隔离会影响可用性。它看得到 socket、syscall 和调度，不代表看得到 GPU SM 内部；GPU 事件需要 CUDA/Nsight/DCGM 的专门接口。

## 36.7 GPU 遥测：DCGM、MIG 与应用语义

[NVIDIA DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/) 提供 GPU 健康、利用率、显存、温度、功耗、ECC、NVLink 和 MIG 相关 field；[dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) 可把 field 暴露给 Prometheus。生产配置应记录：驱动和 DCGM 版本、采样周期、device UUID、MIG instance、权限和 exporter endpoint。

关键字段可分成四组：

- **利用率/吞吐**：SM active、tensor active、DRAM active；区分“忙”与“有效工作”。
- **容量/压力**：显存 used/free、BAR1、allocator reserved；显存高不一定是泄漏，需结合分配 profile。
- **健康/节流**：温度、功率、clocks、thermal/power throttle、XID、ECC；节流事件要与部署和机架风道对齐。
- **互联**：NVLink throughput、replay、PCIe TX/RX；跨 NUMA 或 PCIe downgrade 时，通信尾部可能先于 SM 利用率恶化。

DCGM 的 `device` label 不能裸用短序号，因为重启后编号可能改变；优先使用 UUID 和机架/节点的稳定 resource attribute。MIG 场景同时记录 parent GPU 与 instance ID，避免把多个实例聚成一条“平均 GPU”。

## 36.8 分布式 trace correlation：把请求、批次、GPU、网络和存储串起来

### 36.8.1 correlation graph

建议维护一张显式图：

```text
trace_id
  ├─ request_id / retry_id
  ├─ scheduler_batch_id
  │    ├─ worker_pod / rank / device_uuid
  │    ├─ nccl_comm_id / collective_seq
  │    └─ nvtx_range_id
  ├─ rpc span ↔ peer trace_id
  ├─ storage span ↔ file/op hash
  └─ exemplar ↔ Prometheus histogram sample
```

每条边都应有 TTL、访问控制和脱敏策略。`collective_seq`、`batch_id` 等运行时 ID 适合 span attribute，不适合长期 metric label。跨进程传递时，使用 W3C baggage 谨慎：baggage 会随每个请求传播，不能放大字符串或秘密。

### 36.8.2 fan-out/fan-in 和重试

网关把一个请求 fan-out 到 tokenizer、retriever、model shards，最终 fan-in。使用 links 连接并行子任务，避免伪造单一 parent 导致树深度失真。每次 retry 新建 span，属性包括 `retry.attempt`、`retry.reason`、`backoff_ms`；业务 `request_id` 保持不变。若 retry 触发第二次计费或 token 生成，必须在 metrics 中单独计数，否则成功率看起来正常而成本翻倍。

### 36.8.3 时钟和采集延迟

Collector 的 batch processor 会延迟导出，Prometheus scrape 也有时间偏移。告警中区分 `event_time` 和 `ingest_time`；对于事故复盘，先用源端 monotonic duration 排序，再用 NTP-corrected wall time 对齐节点。记录 telemetry pipeline lag：`collector_export_latency_seconds`、`prometheus_scrape_timestamp_seconds`、`remote_write_samples_pending`。如果 pipeline lag 超过 SLO 窗口，不能把“无告警”当成“无故障”。

## 36.9 Tail latency：从 p99 追到原因

### 36.9.1 分解总延迟

对在线生成，可写：


`T_e2e = T_admission + T_queue + T_prefill + T_decode + T_network + T_serialize`。


其中 `T_decode` 可能包含 batch 等待、NCCL、KV cache miss 和 stream flush。将每一项分别做 histogram，才能知道 p99 总和是否由单一阶段主导。注意分位数不可线性相加：p99(A)+p99(B) 是上界式粗估，不等于 p99(A+B)。保留同一 trace 的阶段样本，才能研究相关性。

### 36.9.2 负载分桶和 tail amplification

按 prompt token、max_new_tokens、batch size、tenant、route、GPU partition、可用区分桶。若只看全局 p99，长请求比例变化会伪装成回归。定义 tail amplification：`p99/p50`，并随并发、队列深度和资源利用率绘制。高 amplification 通常意味着服务时间方差、锁竞争、重试或 head-of-line blocking；平均值不变也可能发生。

### 36.9.3 RED、USE 和四个黄金信号

- RED（Rate、Errors、Duration）适合请求服务；
- USE（Utilization、Saturation、Errors）适合资源；
- 四个黄金信号（延迟、流量、错误、饱和度）适合高层仪表板。

AI 服务额外补充 token throughput、queue age、GPU duty cycle、KV hit/miss、NCCL wait、NIC retransmit、disk latency。仪表板按“症状→证据”布局，而不是按团队所有权分栏。

## 36.10 GPU/网络/存储联合诊断

### 36.10.1 场景一：GPU 利用率下降但 TTFT 上升

先看 queue span：若 `T_queue` 上升而 `T_prefill` 稳定，是 admission/worker 数不足；若 CPU launch gap 增大，结合 eBPF runqlat 和 PyTorch profiler 检查 CPU oversubscription；若 NVLink throughput 和 NCCL wait 同时升高，查看 DCGM replay、PCIe link width 和 rank skew。只有当 kernel duration 本身增大且 clocks 被 throttle，才把根因指向 GPU。

### 36.10.2 场景二：网络重传放大 decode ITL

通过 OTel RPC spans 识别跨节点 shard；eBPF `tcpretrans` 或 NIC exporter 观测 retransmit；Nsight Systems 的 NCCL range 显示 collective gap；DCGM NVLink 仅覆盖 GPU 互联，不能替代 NIC counters。关联相同 `collective_seq` 和 rank，可区分“网络慢”与“某个 rank CPU 抢占导致 collective 未发出”。

### 36.10.3 场景三：checkpoint 后首 token 变慢

存储 span 显示 `fsync`/`write` 时间，eBPF `biolatency` 显示设备尾部，node exporter 显示 page cache 与 IO wait，GPU profile 显示 dataloader/weight page fault。若磁盘只有单个设备抖动，检查队列深度和 NVMe firmware；若所有节点同时抖动，检查对象存储限流或网络共享。修复可能是分离 checkpoint 设备、异步 flush、限速或错峰，而不是提高 GPU 数量。

### 36.10.4 证据时间线模板

| 时间（UTC） | span/metric/event | 观察 | 假设 | 下一步 |
| --- | --- | --- | --- | --- |
| 10:00:12 | deploy event | 新版本开始 | 可能回归 | 对比前后样本 |
| 10:02:31 | TTFT p99 | 800→1400 ms | 队列拥塞 | 查 queue_age |
| 10:02:35 | DCGM SM active | 42%→28% | GPU 等待输入 | 查 runqlat/NIC |
| 10:02:40 | eBPF biolatency | p99 3→90 ms | NVMe 抖动 | 对照 checkpoint |
| 10:03:05 | span error | deadline exceeded | 影响用户 | 降级/回滚 |

时间线只记录证据，不在“观察”栏写结论。每个假设附一个能使它失败的实验。

## 36.11 实验：CPU-only observability toy lab

### 36.11.1 实验目标和边界

`labs/ch36_observability_tracing_lab.py` 不调用 CUDA、DCGM、eBPF 或网络；它用确定性的整数/浮点算术模拟 12 个请求、跨服务 span、Prometheus 风格 histogram、日志、profile sample、GPU/网络/存储遥测快照和两个故障注入（network_tail、storage_tail）。目标是测试协议：context 是否贯通、tail 采样是否保留慢/错 trace、cardinality 是否受预算约束、联合诊断是否输出正确根因提示。它不是 GPU 性能基准，也不能证明 OTel Collector 或 Prometheus 的生产吞吐。

运行：

```bash
python3 labs/ch36_observability_tracing_lab.py --fault network_tail --output reports/ch36-observability-default.json
python3 tests/test_ch36_observability_tracing_lab.py
```

输出 JSON 含 `traces`、`metrics`、`logs`、`profiles`、`telemetry`、`sampling`、`diagnosis`、`invariants`。所有 ID 和时间在固定 seed 下可重放。

### 36.11.2 观测实验步骤

1. 先运行无故障 workload，检查每条 accepted request 都有 root/server、queue、prefill、decode、response spans，且 parent-child 关系无环。
2. 将 `--fault network_tail` 打开，比较 `itl_p99_ms`、`network.retransmits_total` 和 NCCL wait；tail sampler 应保留受影响 trace。
3. 将 `--fault storage_tail` 打开，比较 `checkpoint.fsync_ms`、`disk.io_latency_p99_ms` 和 queue wait；诊断应指向 storage，而非 GPU throttle。
4. 提高 `--cardinality-budget` 到 2 与 100，观察 dropped label series；确认 trace_id 不出现在 metric label。
5. 重复运行两次并比较 JSON 哈希，验证 deterministic；再用 `py_compile` 检查语法。

### 36.11.3 失败注入的预期

network_tail 只增加网络和 collective 等待，不改变 prefill CPU 时间；storage_tail 只增加 fsync/IO wait，并通过队列把影响传播到后续请求。若诊断器在 network_tail 报“GPU thermal throttle”，说明证据优先级或阈值设计错误；若 storage_tail 没有保留慢 trace，说明 tail sampler 的 predicate 不完整。

## 36.12 故障诊所/失败：常见观测反模式

### 36.12.1 “所有请求都采样”导致控制面雪崩

症状是 Collector memory limit hit、export queue full、应用 exporter 阻塞，最终业务 p99 变坏。修复是分离同步/异步 exporter、设置 batch 和 memory limiter、tail sampling 只保留错误/慢请求，并监控 dropped spans。不要直接删除 trace exporter 而不留下 `telemetry_dropped_total`。

### 36.12.2 把 trace_id 当 Prometheus label

症状是每个请求一条新时序，Prometheus OOM。删除该 label，改用 exemplar 或日志索引；对需要追踪的少量请求写入采样表。测试应检查指标 label 集合不含 request_id、trace_id、prompt_hash。

### 36.12.3 只有 GPU utilization，没有队列和网络

40% 的 SM active 可能是 CPU starve、NCCL 等待、数据读取或功率 throttle。补充 queue age、CPU runqlat、NCCL duration、DCGM clocks/throttle、NIC retransmit 和 disk latency，再做 trace correlation。单一利用率图不能作为容量扩容依据。

### 36.12.4 Profile 开得太久

Nsight/PyTorch 插桩改变调度，长时间收集会放大内存和 IO。用短 active window、采样率和发布标记；在报告中记录“启用 profiler 的版本”和 baseline overhead。出现“修复后 profile 更快”时，先确认两次采样开关一致。

### 36.12.5 context 在异步边界丢失

症状是网关 trace 结束后出现孤立 worker trace，日志没有 trace_id。修复队列 envelope，加入 context schema 版本和无 context 计数；对消息重投递用 links 或新 retry span。不要用线程局部变量跨 async task 传递。

### 36.12.6 时钟漂移制造负延迟

跨节点 wall-clock 未校准会出现 child span 在 parent 之前。保留源端 monotonic duration，Collector 做 clock skew 估计但不修改原始证据；复盘时展示 uncertainty interval。若 NTP offset 超过阈值，触发 node health 告警。

## 36.13 事故复盘：从告警到可执行修复

### 36.13.1 复盘模板

1. **影响**：受影响 route、租户、时间窗、请求数、SLO error budget 消耗。
2. **检测**：哪个告警先触发，检测延迟和 telemetry lag 多大。
3. **时间线**：部署、配置、流量、资源和用户症状，全部带 UTC 时间和来源链接。
4. **证据**：至少一条 root trace、阶段 histogram、资源 profile、控制面日志；标注采样和缺失。
5. **假设树**：列出 CPU、GPU、网络、存储、调度、外部依赖，每个给支持/反证。
6. **缓解**：回滚、限流、切换模型、迁移租户或关闭 profiler；记录谁执行、何时完成。
7. **根因和促成因素**：区分直接触发器、容量余量不足、观测缺口和流程问题。
8. **修复验证**：回放相同流量，比较 p50/p95/p99、错误率、成本和 telemetry overhead。
9. **行动项**：负责人、截止日期、验收指标和回滚开关。

### 36.13.2 反事实实验

好的复盘不只说“回滚后恢复”。例如：把 network_tail 的 replay 放到旧版本，若 tail 仍存在则版本不是根因；把 storage concurrency 降半，若 p99 恢复则验证 IO 饱和；禁用 tail sampler，若业务延迟下降则 telemetry overhead 是促成因素。反事实要在隔离环境或受控流量下运行，不能直接在生产制造第二次事故。

### 36.13.3 证据保留和访问控制

Trace/log/profile 的保留期应按事故窗口、合规要求和成本确定。原始 prompt、用户标识和网络地址分级加密；默认只保留长度、hash、模型版本和错误类别。复盘文档链接到受控对象存储，不在公共 issue 中粘贴敏感 payload。删除证据也要记录 tombstone 和理由，避免“无数据”被误读为“无故障”。

## 36.14 理解检查

1. 为什么 head sampling 不能可靠抓住 p99 错误请求？tail sampling 需要承担什么内存风险？
2. `trace_id` 放进 Prometheus label 会发生什么？exemplar 如何提供替代导航？
3. GPU SM active 下降、NCCL wait 上升、NIC retransmit 不变时，你会优先检查哪两层？
4. 为什么 histogram 的 p99 不能与另一个阶段的 p99 直接相加？如何用同一 trace 估算总尾部？
5. eBPF 能回答哪些 CPU/网络/存储问题，为什么不能替代 DCGM 或 Nsight？
6. 在跨节点 profile 中，如何处理 NTP offset 和 monotonic duration 的冲突？
7. 如果 Collector dropped spans 增加但业务 p99 未变，你会如何判断是采样策略还是 exporter 故障？
8. storage_tail 注入只改变 fsync 时间，为什么后续请求的 queue wait 也可能变长？

参考答案要点：head 在 root 决策时看不到后续错误；tail 需缓存并有内存/超时上限。trace_id 是高基数，exemplar/日志索引更合适。NCCL wait 优先看通信和 rank skew，再看 GPU clocks。分位数不可线性相加，需联合样本。eBPF 观察内核/用户态边界，GPU 内部需专用工具。排序用 monotonic，展示用校准 wall clock。检查 drop reason、pipeline lag、业务采样覆盖。IO 队列和共享设备会把 fsync 的阻塞传播给请求队列。

## 36.15 练习

1. 为一个多租户推理服务写 telemetry contract：列出 10 个 span attributes、8 个 metrics、6 个日志字段，并给每个字段基数和隐私等级。
2. 设计两个 Prometheus histogram bucket：一个用于 TTFT，一个用于 ITL。说明 bucket 如何映射 SLO，如何估算 series 数量。
3. 用 OTel Collector 配置草图实现“错误或超过 1 s 的 trace 100% 保留，成功 trace 1% 保留”，并给出内存上限和 dropped 指标。
4. 在 toy lab 增加一个 `cpu_starvation` 故障：只提高 run queue 和 launch gap。写测试确保诊断不会误报 network_tail。
5. 画出请求到 GPU kernel 的 correlation graph，标出 context 在 HTTP、队列、RPC、NVTX 和 DCGM 之间的传递方式。
6. 设计一次 15 分钟 profile window：何时开启 PyTorch Profiler，如何避免把用户 prompt 写进 NVTX，如何计算 overhead。
7. 选择一个真实事故（可匿名），按 36.13 模板写时间线、证据、反事实和行动项；附上缺失 telemetry 的风险。
8. 阅读 [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)、[Prometheus exemplars](https://prometheus.io/docs/prometheus/latest/feature_flags/#exemplars-storage) 和 [NVIDIA DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/)，比较它们的版本固定与升级回滚策略。

## 36.16 来源

以下来源均为官方文档、标准、论文或开源仓库；URL 用于核查概念和接口，生产部署仍需锁定版本并在目标硬件上重放：

1. OpenTelemetry traces、metrics、logs、OTLP specification：https://opentelemetry.io/docs/concepts/signals/traces/ 、https://opentelemetry.io/docs/concepts/signals/metrics/ 、https://opentelemetry.io/docs/concepts/signals/logs/ 、https://opentelemetry.io/docs/specs/otlp/
2. OpenTelemetry semantic conventions 与 GenAI conventions：https://opentelemetry.io/docs/specs/semconv/ 、https://opentelemetry.io/docs/specs/semconv/gen-ai/
3. OpenTelemetry Collector 与 tail sampling processor：https://opentelemetry.io/docs/collector/ 、https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/processor/tailsamplingprocessor
4. W3C Trace Context 与 Baggage：https://www.w3.org/TR/trace-context/ 、https://www.w3.org/TR/baggage/
5. Prometheus data model、histogram、exemplar、promtool：https://prometheus.io/docs/concepts/data_model/ 、https://prometheus.io/docs/practices/histograms/ 、https://prometheus.io/docs/prometheus/latest/feature_flags/#exemplars-storage 、https://prometheus.io/docs/prometheus/latest/command-line/promtool/
6. NVIDIA Data Center GPU Manager 与 exporter：https://docs.nvidia.com/datacenter/dcgm/latest/ 、https://github.com/NVIDIA/dcgm-exporter
7. NVIDIA Nsight Systems 与 Nsight Compute：https://docs.nvidia.com/nsight-systems/ 、https://docs.nvidia.com/nsight-compute/
8. PyTorch Profiler 文档：https://pytorch.org/docs/stable/profiler.html
9. eBPF 文档、BCC、bpftrace：https://ebpf.io/what-is-ebpf/ 、https://github.com/iovisor/bcc 、https://github.com/bpftrace/bpftrace
10. Linux kernel BPF ring buffer、tracepoints 与 perf events：https://www.kernel.org/doc/html/latest/bpf/ringbuf.html 、https://www.kernel.org/doc/html/latest/trace/events.html
11. NVIDIA NVTX 与 CUPTI：https://nvidia.github.io/NVTX/ 、https://docs.nvidia.com/cupti/
12. gRPC metadata/deadline/cancellation（用于 context 与 span 边界）：https://grpc.io/docs/guides/metadata/ 、https://grpc.io/docs/guides/deadlines/ 、https://grpc.io/docs/guides/cancellation/
13. Kubernetes metrics pipeline 与 cAdvisor：https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/ 、https://github.com/google/cadvisor
14. OpenMetrics exposition format：https://openmetrics.io/
15. Google SRE workbook 的 SLO、错误预算与监控章节：https://sre.google/workbook/monitoring/
16. 论文：Dapper 分布式追踪：https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/
17. 论文：The Tail at Scale：https://research.google/pubs/the-tail-at-scale/
18. 论文：PerfBench/性能诊断方法可参考 Linux perf 文档：https://www.kernel.org/doc/html/latest/tools/perf/index.html
19. 本章实验与证据：`labs/ch36_observability_tracing_lab.py`、`tests/test_ch36_observability_tracing_lab.py`、`reports/ch36-observability-tracing-report.md`、`evidence/ch36-observability-tracing-manifest.json`。这些是 CPU-only、确定性、局部协议测试，不构成生产性能保证。

## 36.17 小结

可观测性系统的价值不在于收集了多少 signal，而在于能否在 SLO 预算内，把一个用户症状映射到可证伪的跨层假设。OTel 提供 trace/metric/log 的统一上下文，Prometheus 提供可聚合的时间序列，DCGM 解释 GPU 健康和互联，Nsight 与 PyTorch Profiler 解释执行时间线，eBPF 补足内核、网络和存储边界。通过稳定 ID、低基数指标、尾部采样、短窗口 profile 和联合诊断，可以在不把 telemetry 变成新瓶颈的前提下复盘事故。最后，用 CPU-only toy lab 验证协议和失败边界，再在固定硬件、驱动、模型与流量上做生产验收，才能把“看起来合理”变成可审计的性能工程。
