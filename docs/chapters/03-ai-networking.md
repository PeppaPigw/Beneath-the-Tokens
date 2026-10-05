---
id: ch03-networking
title: AI 系统网络：TCP/IP、RPC、RDMA 与尾延迟
description: 从字节流到多机 collective，建立可测量的 AI 网络性能与故障直觉
slug: /chapters/03-networking
sidebar_position: 3
level: core
prerequisites:
  - ch01-ai-infrastructure
  - ch02-linux-process-files-observability
learning_objectives:
  - 能解释一次 AI 请求经过 TCP/IP、RPC、序列化和拷贝的时间线
  - 能区分带宽、延迟、并发、拥塞和尾延迟，并用模型做数量级估算
  - 能描述 RDMA、collective 通信与拓扑的适用条件和失败模式
  - 能运行最小 TCP/背压实验，采集 p50/p95/p99 并定位排队来源
estimated_hours: 16
hardware: CPU-only baseline; two hosts or loopback; RDMA optional
risk_level: L1
last_verified: 2026-10-05
---

# 第3章　AI 系统网络：TCP/IP、RPC、RDMA 与尾延迟

> AI 系统的“网络”不只是交换机。一个请求会在用户态和内核之间拷贝，在协议栈中排队，在 RPC 框架中等待连接和线程，在序列化器里分配内存；多卡训练又会把网络变成同步屏障。本章把这些路径拆开，用实验和公式回答三个问题：数据在哪里、等待为什么发生、尾部为什么比平均值更重要。

## 3.1 学习目标与统一视角

[设计判断] 读完本章，读者应能画出如下因果链：应用对象 → 序列化缓冲区 → socket/RPC → TCP/IP 或 RDMA → NIC 队列 → 交换机端口 → 对端 NIC → 内核/用户态 → 解码与计算。链上每个箭头都可能复制字节、增加排队或引入重试。优化前先标注每个时间点，否则“把网络换快”很容易变成没有证据的采购。

[定义] **吞吐**是单位时间传送的字节、请求或 token 数；**延迟**是一个操作从起点到终点的时长；**带宽**是链路可提供的最大传送速率；**并发**是同时处于飞行状态的操作数。四者不是同一个指标：带宽提升不一定降低小消息延迟，增加并发可能提高吞吐却恶化 p99。

[事实] 互联网协议分层把应用、传输、网络、链路和物理细节分开。分层降低了组合复杂度，但每层都有头部、状态和缓冲区。真实路径还包括 TLS、代理、服务网格、容器虚拟网卡和可观测性采样，这些并不出现在教科书的五层图里。

[机制] 对于 AI 推理，请至少区分三种网络动作：1）控制请求，如鉴权和路由，消息小但对首 token 延迟敏感；2）数据请求，如图像、长提示和 KV cache，消息大且受带宽、拷贝影响；3）同步请求，如 all-reduce、all-gather，所有参与者必须协同完成，最慢参与者决定整体进度。

[测量] 每次基准都记录消息大小、并发、连接复用、是否加密、是否跨可用区、CPU 频率、丢包/重传和时间窗口。没有这些元数据，两个“1000 QPS”结果可能并不具备可比性。

### 3.1.1 三条时间线

[机制] 端到端延迟可分解为

`T_e2e = T_queue + T_app + T_serialize + T_copy + T_transport + T_remote_queue + T_compute + T_return`。

这些项可能重叠，所以简单相加是上界式近似，而不是精确追踪。`T_queue` 可能包括客户端连接池、服务器 accept 队列、RPC 线程池和 NIC ring；`T_copy` 可能在用户态缓冲区、内核 socket 缓冲区、TLS 缓冲区和设备内存之间重复发生。

[测量] 用分布式追踪为每一段打时间戳，并携带单调时钟的 span id。跨机器直接比较 wall clock 需要时钟同步；即使 NTP 已同步，微秒级测量仍可能受到时钟偏差影响，因此可使用 TCP 时间戳、硬件时间戳或在同一主机上先做分段实验。

[推断] 当端到端 p99 上升而模型执行 p50 不变时，优先怀疑排队、连接池、重传和后处理，而不是立即怀疑模型内核。反过来，只有在网络分段和应用分段都稳定后，才有资格把问题归因于算力。

## 3.2 TCP/IP：可靠字节流如何产生等待

### 3.2.1 IP 与路由的边界

[事实] IP 提供尽力而为的数据报转发：它不保证到达、顺序或不重复。IPv4 头部含源/目的地址和 TTL 等字段；IPv6 扩展头提供更大地址空间和可选功能。路由器根据前缀和策略选择下一跳，数据包可能走不同路径。

[机制] TCP 在 IP 之上提供有序、可靠、面向连接的字节流。应用看到的是连续字节，而不是消息边界；一次 `send()` 不对应对端一次 `recv()`。因此应用必须自己定义 framing（长度前缀、分隔符或固定长度），否则会出现半包、粘包和解析阻塞。

[事实] TCP 三次握手交换初始序列号并协商能力；四次挥手允许双方独立关闭发送方向。连接建立和慢启动使短连接请求付出固定成本，连接复用通常更适合高 QPS 的 RPC，但长期连接也需要处理空闲超时、NAT 状态和服务端重启。

[测量] `ss -ti`、`nstat -az`、`/proc/net/snmp`、eBPF 工具和抓包可以看到拥塞窗口、往返时间（RTT）、重传和接收窗口。抓包本身可能改变时序；生产诊断优先使用低开销统计，再用短时抓包验证假设。

### 3.2.2 流量控制与拥塞控制

[机制] **接收窗口（rwnd）**限制发送方不要超过接收端缓冲能力；**拥塞窗口（cwnd）**限制发送方不要超过网络当前可承受的在途数据量。有效发送窗口近似为 `min(rwnd, cwnd)`。应用读取缓慢会缩小 rwnd；链路拥塞会让算法缩小 cwnd。

[事实] TCP 拥塞控制通常经历慢启动、拥塞避免，并根据丢包、RTT 或 ECN 信号调节窗口。Linux 可配置 CUBIC、BBR 等算法，具体行为受内核版本、参数和网络设备影响；不能只凭算法名称推断 p99。

[机制] 在带宽时延积（BDP）为 `带宽 × RTT` 的路径上，发送窗口需要足够大才可填满链路。例如 100 Gb/s、80 µs RTT 的 BDP 约为 1 MB。窗口太小，吞吐受窗口限制；窗口过大，在突发流量下会制造交换机队列，导致 bufferbloat 和尾延迟。

[推断] 看到“链路利用率只有 40%”并不意味着可以安全地把并发翻倍。剩余带宽可能被 RTT、窗口、突发和共享端口的队列消耗；应同时检查 cwnd、重传率和 p99。

### 3.2.3 队列与尾延迟

[机制] 一个 TCP 请求可能排在：客户端连接池、发送 socket、NIC TX ring、交换机输入/输出队列、对端 NIC RX ring、内核 backlog、RPC 线程池和模型批处理队列。任一队列接近容量，就会从“几微秒等待”变成毫秒级长尾。

[事实] 排队理论常用 `ρ = λ/μ` 表示利用率；当 ρ 接近 1，等待时间的方差会急剧上升。实际服务并非理想 M/M/1，但“接近饱和会放大尾部”的方向稳定成立。

[测量] 绘制每个队列的长度和等待时长，而不只绘制 CPU 利用率。交换机可看端口队列、ECN 标记和丢包计数；主机可看 listen backlog、SO_RCVBUF/SO_SNDBUF、软中断和队列丢弃。把 p99 与队列长度叠加，通常比单看吞吐更快定位瓶颈。

### 3.2.4 MTU、分片与拥塞信号

[事实] 链路有最大传输单元（MTU）。超过路径 MTU 的包需要分片或被丢弃并触发 PMTU 发现；大多数数据中心会避免跨设备分片。启用巨型帧（如 9000 字节）可减少每字节头部和中断，但要求路径上所有设备一致，错误配置会产生黑洞。

[机制] ECN 允许交换机在不丢包时标记拥塞，端点据此降低发送速率。RoCEv2 数据中心常依赖 ECN 与优先级流控（PFC），但 PFC 可能造成 head-of-line blocking 和拥塞扩散，必须结合队列监控和故障演练评估。

[设计判断] MTU、ECN、PFC 不是“打开就更快”的开关。先在隔离环境测量小包/大包、独占/共享端口、无拥塞/热点流四组场景，并记录丢包、标记、p99 和恢复时间。

## 3.3 RPC：把网络调用伪装成函数，别忘了它仍会失败

### 3.3.1 RPC 的组成

[定义] RPC 框架通常提供接口描述（IDL）、代码生成、序列化、连接管理、超时、重试、负载均衡和状态码。gRPC 常用 Protocol Buffers 与 HTTP/2；JSON over HTTP 具有可读性和生态优势；自定义二进制协议可以更贴近固定工作负载。选择协议应基于测量和治理需求，而非“二进制一定快”。

[机制] 一次 RPC 通常经过：客户端拦截器 → 名称解析 → 连接池/HTTP2 stream → 请求序列化 → TLS → TCP → 服务端解密/解析 → handler → 响应序列化 → 客户端回调。每一层都可能有超时和重试，叠加后容易产生“重试风暴”。

[事实] HTTP/2 在一个 TCP 连接上复用多路 stream，并用 HPACK 压缩头部；TCP 丢包会阻塞同一连接上尚未交付的后续字节（队头阻塞）。HTTP/3 使用 QUIC，基于 UDP 实现加密和独立 stream，减少连接级队头阻塞，但会增加协议与运维复杂度。

### 3.3.2 超时、重试与幂等

[机制] 端到端 deadline 应由入口设置并向下游传播：`deadline_child = deadline_parent - budget_overhead`。若每一层都重新设置固定 1 秒超时，请求可能在最外层已取消后仍继续占用资源。

[事实] 只有幂等操作才能安全自动重试。创建任务、扣费或追加日志等操作需要请求 id、去重表或事务语义，否则重试会产生重复副作用。指数退避加抖动可降低同步重试，但不能修复容量不足。

[推断] 当 p99 与错误率同时上升、下游 QPS 反而增加时，优先检查重试乘数：若第 i 层重试次数为 `r_i`，最坏放大近似为各层乘积，而不是简单相加。

### 3.3.3 流式 RPC 与 backpressure

[机制] 生成式模型常用双向流或服务端流：服务端逐 token 发送，客户端逐块消费。若客户端读取慢，HTTP/2 flow-control window 会收缩，最终阻塞模型生成线程或发送队列。正确实现应把网络写入与模型计算解耦，用有界缓冲并在达到高水位时暂停生产。

[测量] 记录 TTFT、token 间隔（TPOT）、流中断率和未消费缓冲长度。只统计完整响应会掩盖“首 token 很快、后续 token 被网络拖慢”的用户体验问题。

## 3.4 序列化与拷贝：字节搬运经常比算子更贵

### 3.4.1 消息格式的选择

[事实] JSON 文本可读、跨语言方便，但数字和结构需要解析，体积通常大于二进制格式。Protocol Buffers 通过 schema 和 varint 编码减少体积并支持向后兼容；FlatBuffers、Cap'n Proto 等格式允许在缓冲区上直接访问部分字段，减少反序列化拷贝，但要求更严格的数据布局和生命周期管理。

[机制] 序列化成本约为 `O(字段数 + 字节数)`，但常数项受分配次数、缓存局部性、压缩和语言运行时影响。对 GPU 推理，请区分“把图片从磁盘读入主机”“解码 JPEG”“转换成 tensor”“复制到设备”四个阶段，不能把它们统称为网络开销。

[测量] 基准至少包括：payload 大小、编码/解码 CPU 时间、分配次数、峰值 RSS、压缩比、网络字节、错误率。使用相同对象和随机种子，预热 JIT/缓存，并报告 p50/p95/p99，而不是只给平均 MB/s。

### 3.4.2 常见拷贝路径

[机制] 典型 TCP 路径是：应用缓冲区 → 内核 socket 缓冲区 → NIC DMA ring → 对端 NIC → 内核缓冲区 → 应用缓冲区。`sendfile`、`splice`、`MSG_ZEROCOPY`、io_uring 和 DPDK 等技术可减少某些 CPU 拷贝，但并不意味着“零成本”：仍需页固定、完成通知、缓存失效或设备限制。

[事实] DMA 允许 NIC 直接读写主机内存；IOMMU 提供地址转换与隔离。用户态零拷贝通常要求缓冲区生命周期覆盖异步发送，提前复用会导致数据竞争。对小消息，管理零拷贝的开销可能高于一次 memcpy。

[设计判断] 先通过火焰图和 `perf stat` 证明 memcpy、序列化或分配占比，再引入零拷贝。优化目标是端到端 p99 和 CPU 成本，而不是“拷贝次数为零”这一口号。

### 3.4.3 压缩的两面

[机制] 压缩减少网络字节，但增加 CPU 延迟和峰值内存。设未压缩大小为 S、压缩后为 rS、网络带宽 B、压缩/解压速率为 C，则压缩有利的粗略条件是 `S/B > S/C_encode + rS/B + S/C_decode`。真实系统还要考虑并发、NUMA 和尾部请求。

[推断] 在低带宽跨地域链路上，压缩常常值得；在 400 Gb/s 的机架内网络上，压缩可能把瓶颈从网络转移到 CPU。应按链路类型分层配置，而不是全局开启或关闭。

## 3.5 背压：让快生产者感知慢消费者

### 3.5.1 为什么必须有界

[机制] 无界队列会把瞬时突发变成内存增长和长尾。若生产速率 λ 长期大于消费速率 μ，队列长度按 `(λ-μ)t` 增长，最终触发 OOM 或级联超时。有界队列在高水位时阻塞、降采样、丢弃低优先级请求或返回过载错误，从而保护系统。

[事实] TCP 的接收窗口是底层背压；RPC stream 的 flow-control 是协议级背压；应用队列、批处理器和消息队列还需要业务级背压。只靠 TCP 并不能阻止应用在用户态先堆积几 GB 数据。

### 3.5.2 高低水位策略

[设计判断] 可为每个流设置低水位 L 和高水位 H：缓冲量超过 H 时暂停读取上游或暂停生成，降到 L 以下再恢复。H-L 留出滞回区，避免在阈值附近频繁开关。策略要定义超时、取消和优先级：交互请求通常优先于离线批处理。

[测量] 监控 `queue_depth`、`time_in_queue`、生产阻塞时间、丢弃数、取消数和恢复时间。仅监控总内存无法知道是哪个租户或哪条流造成背压。

### 3.5.3 一个 Python asyncio 示例

下面代码展示有界队列和慢消费者。它不追求生产级性能，目的是让队列长度与 p99 的关系可观察。

```python
# backpressure_demo.py
import asyncio, statistics, time

async def producer(q, n=200, interval=0.001):
    for i in range(n):
        t0 = time.perf_counter()
        await q.put((i, t0))      # 队列满时这里阻塞，形成背压
        await asyncio.sleep(interval)
    await q.put(None)

async def consumer(q, service=0.01):
    waits = []
    while True:
        item = await q.get()
        if item is None:
            q.task_done(); break
        _, t0 = item
        waits.append(time.perf_counter() - t0)
        await asyncio.sleep(service)
        q.task_done()
    return waits

async def main():
    q = asyncio.Queue(maxsize=8)
    p = asyncio.create_task(producer(q))
    waits = await consumer(q)
    await p
    q50 = statistics.quantiles(waits, n=100)[49]
    q99 = statistics.quantiles(waits, n=100)[98]
    print(f"samples={len(waits)} p50={q50*1e3:.2f}ms p99={q99*1e3:.2f}ms")

asyncio.run(main())
```

[实验] 将 `maxsize` 改为 0（无界）并把 `service` 改为 0.02 秒，比较内存、生产者阻塞时间和等待分位数。实验结果受调度器和机器负载影响，不应当当作普适常数；应保存运行环境和原始样本。

## 3.6 RDMA：绕过内核的代价与边界

### 3.6.1 基本术语

[事实] RDMA（Remote Direct Memory Access）允许一台主机直接读写另一台主机注册的内存，常见传输包括 InfiniBand、RoCE（RDMA over Converged Ethernet）和 iWARP。应用通过 verbs 创建保护域、内存区域（MR）、队列对（QP）、完成队列（CQ）并提交 work request。

[机制] 内存注册会固定页并建立 NIC 可用的地址映射；注册和撤销不是免费操作，因此高性能服务通常缓存 MR。发送方提交 WR 后由 NIC DMA，完成事件写入 CQ。轮询 CQ 延迟低但消耗 CPU；中断节能但引入抖动。

[事实] **两端操作**（send/receive）需要接收方预先张贴 buffer；**单边操作**（RDMA read/write）由发起方直接访问远端 MR，远端 CPU 可不参与数据搬运，但仍需权限、rkey 和内存生命周期管理。单边并不自动提供应用层一致性或事务语义。

### 3.6.2 RoCE 网络与拥塞

[机制] RoCEv2 把 RDMA 封装在 UDP/IP 中，通常依赖数据中心交换机的 ECN 和 PFC。PFC 在拥塞时暂停某优先级流量，能降低丢包却可能把一个热点扩散到整条链路；配置错误会出现 pause storm。必须监控 PFC pause、ECN mark、重传和无损队列长度。

[设计判断] 若工作负载是小消息 RPC、跨租户共享网络或经常变更拓扑，TCP 可能更稳健；若是高带宽、可控机架内通信（如训练梯度和参数交换），RDMA 的低 CPU 开销和可预测延迟更有价值。决定前先做故障注入：拔链路、降速、制造热点、重启 QP，并验证是否能超时恢复。

### 3.6.3 RDMA 的安全与可观测性

[事实] MR 的访问权限决定远端能读/写哪些地址；错误地暴露 rkey 可能泄露数据或破坏内存。使用最小权限、短生命周期 key、租户隔离和审计。与 TCP 不同，很多网络设备对 RDMA 负载的传统抓包能力有限，需要读取 NIC/交换机计数器和 verbs 错误码。

[测量] 记录 post/send 到 CQ completion 的时间、重试计数、CQ 深度、QP 状态、MR 缓存命中率、CPU 核绑定和 NUMA 节点。不要只报告“RDMA 端到端 5 µs”，说明消息大小、是否轮询、是否跨交换机和尾部样本数。

## 3.7 Collective 与拓扑：多卡同步的乘法效应

### 3.7.1 集体通信语义

[定义] 常见 collective 包括 broadcast（一个到多个）、scatter/gather（拆分/汇聚）、all-reduce（所有参与者得到归约结果）、all-gather（收集所有分片）和 reduce-scatter（先归约再分片）。collective 隐含参与者集合和同步点，任何 rank 迟到或异常都可能让全体等待。

[事实] NCCL 等库会根据 GPU、NIC、PCIe、NVLink 和交换机拓扑选择 ring、tree 或分层算法。算法选择受消息大小、rank 数和链路带宽影响，不应把某个算法当作绝对最优。

### 3.7.2 粗略通信模型

[机制] 常用 `α-β` 模型把一次消息成本写成 `T ≈ α + nβ`，其中 α 是启动/往返延迟，β 是每字节传输时间。ring all-reduce 在 p 个参与者、消息大小 n 下，理想化成本约为 `2(p-1)α + 2(p-1)nβ/p`；tree 的步数约为 `log₂p`，但每步可能增加带宽竞争。真实实现还受分片、重叠、协议阈值和拓扑映射影响。

[推断] 当 n 很小，α 主导，减少启动次数和融合小消息比提高链路带宽更有效；当 n 很大，β 主导，选择带宽高且拥塞少的路径更重要。把多个梯度桶融合可以摊薄 α，但会增加等待桶填满的延迟。

### 3.7.3 拓扑感知与 rank 映射

[事实] GPU 之间可能通过同一 PCIe root complex、NVLink、PCIe switch 或跨 NUMA socket 连接；NIC 可能只靠近某些 GPU。错误的 rank 映射会让本可走 NVLink 的流量绕行 PCIe 或跨 CPU socket，吞吐下降且尾部不稳定。

[测量] 使用 `nvidia-smi topo -m`、`hwloc-ls`、NCCL debug 日志和链路计数器确认实际路径。记录进程 CPU affinity、GPU affinity、NIC affinity 和 NUMA 内存策略。不要从机器型号推断拓扑，因为 BIOS、插槽和虚拟化配置会改变结果。

### 3.7.4 通信与计算重叠

[机制] 训练框架可在反向计算同时发起上一层梯度的 all-reduce。重叠要求独立 CUDA stream、足够的缓冲区和正确的同步事件；过度并发会争用 PCIe、SM 或 NIC。用时间线工具验证“看起来并行”是否真的重叠。

[设计判断] 优化 collective 时按顺序检查：rank 是否均衡 → 拓扑映射 → 消息融合阈值 → 算法与协议 → 计算通信重叠 → 拥塞与故障恢复。直接把 buffer 调大往往只会增加显存和等待。

## 3.8 延迟模型：从平均数走向尾部

### 3.8.1 基础估算

[机制] 传输一个大小为 S 的消息，理想化网络时间为 `T_net ≈ RTT + S/B`。例如 RTT=20 µs、S=1 MiB、B=25 GB/s（约 200 Gb/s），序列化和排队忽略时，传输时间约 42 µs；实际还会有协议头、PCIe、拥塞和软件开销。这个公式用于数量级检查，不能替代测量。

[事实] Little's Law 为 `L = λW`：稳定系统中平均在途请求数 L 等于到达率 λ 乘平均响应时间 W。它不直接给出 p99，但可用来检查容量声明是否自洽：若 QPS=10,000、平均端到端 50 ms，则平均在途约 500 个请求。

[机制] 百分位数不是可加的。两个串联系统各自 p99=10 ms，端到端 p99 可能接近 20 ms，也可能因相关性和排队远高于 20 ms。尾延迟分析需要请求级 trace 或联合分布，而不是把各服务的 p99 简单相加。

### 3.8.2 排队与批处理

[机制] 动态批处理把多个请求聚成一次 GPU kernel，摊薄启动和通信开销，但会让先到请求等待批窗口。批窗口越长，吞吐通常提高、TTFT 变差；应根据 SLA 设定最大等待时间和批大小。

[测量] 对每个请求记录到达时间、入批时间、出批时间、执行时长和返回时长。分别画“排队 p99”和“执行 p99”，否则批处理导致的等待会被误报为 GPU 变慢。

### 3.8.3 尾部放大的来源

[事实] 尾部常来自：请求大小长尾、共享队列、丢包重传、连接建立、GC/内存分配、CPU 频率变化、NUMA 远端访问、后台 checkpoint、交换机微突发和下游重试。

[推断] 若只在高并发时 p99 上升，而低并发 p50 稳定，优先检查队列和资源争用；若单请求大小增加就使 p50/p99 同步上升，优先检查带宽、序列化和内存带宽；若只有跨机架请求变慢，检查拓扑、路径 MTU、ECN 和端口拥塞。

## 3.9 可运行实验：从 TCP framing 到 netem

### 3.9.1 长度前缀 TCP 服务

以下服务演示应用层 framing、并发连接和请求级计时。服务故意保持简单，适合在同一主机上运行。

```python
# framed_echo.py
import asyncio, struct, time

MAX = 4 * 1024 * 1024

def pack(payload: bytes) -> bytes:
    if len(payload) > MAX:
        raise ValueError("too large")
    return struct.pack("!I", len(payload)) + payload

async def read_frame(reader):
    hdr = await reader.readexactly(4)
    (n,) = struct.unpack("!I", hdr)
    if n > MAX:
        raise ValueError("frame too large")
    return await reader.readexactly(n)

async def handle(reader, writer):
    peer = writer.get_extra_info("peername")
    try:
        while True:
            payload = await read_frame(reader)
            t0 = time.perf_counter_ns()
            writer.write(pack(payload))
            await writer.drain()   # 写缓冲达到阈值时产生背压
            dt = (time.perf_counter_ns() - t0) / 1e3
            print(peer, len(payload), f"write_us={dt:.1f}")
    except (asyncio.IncompleteReadError, ConnectionError, ValueError):
        pass
    finally:
        writer.close(); await writer.wait_closed()

async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", 9000,
                                        limit=MAX + 4)
    async with server:
        await server.serve_forever()

asyncio.run(main())
```

客户端可用以下片段发送固定数量消息并计算分位数：

```python
# framed_client.py
import asyncio, struct, time, statistics, sys

def pack(x): return struct.pack("!I", len(x)) + x

async def run(count=1000, size=1024):
    r, w = await asyncio.open_connection("127.0.0.1", 9000)
    samples=[]; payload=b"x"*size
    for _ in range(count):
        t=time.perf_counter_ns(); w.write(pack(payload)); await w.drain()
        n=struct.unpack("!I", await r.readexactly(4))[0]
        await r.readexactly(n); samples.append((time.perf_counter_ns()-t)/1e6)
    w.close(); await w.wait_closed()
    samples.sort()
    def q(p): return samples[min(len(samples)-1, int(p*len(samples)))]
    print({"n":len(samples),"p50_ms":q(.50),"p95_ms":q(.95),"p99_ms":q(.99),
           "mean_ms":statistics.mean(samples)})

asyncio.run(run(int(sys.argv[1]) if len(sys.argv)>1 else 1000,
                int(sys.argv[2]) if len(sys.argv)>2 else 1024))
```

[实验步骤]

1. 终端 A 运行 `python framed_echo.py`，终端 B 运行 `python framed_client.py 1000 1024`。
2. 分别测试 64 B、1 KiB、64 KiB、1 MiB，记录 p50/p95/p99 和 CPU。
3. 用多进程客户端增加并发，观察 `ss -tin` 中 cwnd、rtt 和重传。
4. 确认 framing 必须处理半包：把客户端每次 `write` 拆成两次并在中间 `sleep`，服务仍应正确解析。

[证据边界] 该实验测的是本机 loopback 和 Python 调度开销，不能代表生产网卡、TLS 或跨机架性能。把实验结果外推到 GPU 集群属于未经验证的推断。

### 3.9.2 用 tc netem 注入延迟和丢包

[实验] 在有权限的测试网络命名空间中执行（不要在生产接口执行）：

```bash
sudo tc qdisc add dev lo root netem delay 2ms 0.5ms loss 0.1%
python framed_client.py 2000 4096
sudo tc qdisc del dev lo root
```

[测量] 比较注入前后的 p50/p95/p99、重传和应用超时。将丢包从 0.1% 改为 1% 并观察是否出现重试级联。若命令失败，可能是内核未启用 netem 或接口已有 qdisc；记录失败原因，不要直接修改生产 qdisc。

### 3.9.3 序列化对比实验

```python
# serialize_bench.py（需要 pip install protobuf msgpack）
import json, msgpack, time, statistics
obj={"ids":list(range(1000)),"text":"hello"*1000,"flags":[True,False]*500}

def bench(name, enc, dec, n=2000):
    e=[]; d=[]; payload=enc(obj)
    for _ in range(n):
        t=time.perf_counter_ns(); b=enc(obj); e.append(time.perf_counter_ns()-t)
        t=time.perf_counter_ns(); dec(b); d.append(time.perf_counter_ns()-t)
    e.sort(); d.sort()
    print(name, "bytes",len(payload),"enc_p99_us",e[int(.99*n)]/1e3,
          "dec_p99_us",d[int(.99*n)]/1e3)

bench("json", lambda x: json.dumps(x).encode(), lambda b: json.loads(b))
bench("msgpack", lambda x: msgpack.packb(x), lambda b: msgpack.unpackb(b))
```

[测量] 该脚本没有使用 protobuf schema，故不能据此宣称“JSON 与 protobuf 的生产差距”。它展示的是测量方法：固定对象、报告字节数和尾部时间，并明确依赖版本。

## 3.10 失败案例：把“快”误当作“稳”

### 案例 A：连接池耗尽导致 p99 雪崩

[事实] 某推理网关把每个请求的连接池上限设为 32，下游模型平均执行 40 ms。高峰 2,000 QPS 时，Little's Law 估计需要约 80 个在途连接；额外请求在客户端队列等待并触发 100 ms 超时，随后重试。结果是下游 QPS 上升、p99 超过数秒。

[根因机制] 池大小、超时和重试没有按端到端预算设计；“平均执行 40 ms”掩盖了排队时间。修复包括：按并发预算调整池大小、传播 deadline、限制重试次数、对过载返回明确错误，并监控池等待时长。

[教训] 连接数不是越大越好。过大的池可能制造更多并发、挤压下游队列；需要用压测找出稳定区间。

### 案例 B：无界流式缓冲导致 OOM

[事实] 服务端按 token 生产，客户端因移动网络暂停读取。中间代理把每条流的输出放入无界内存队列，几十分钟后 RSS 达到上限，多个租户同时断流。

[根因机制] 没有把协议 flow-control 传到模型生产端，也没有单流内存上限和空闲超时。

[修复设计] 设置每流高水位；超过阈值暂停生成或取消低优先级请求；将缓冲计入租户配额；客户端断开时立即取消计算；对断流率和缓冲深度设告警。

### 案例 C：RoCE PFC pause storm

[事实] 训练集群在共享交换机上启用 PFC。一个端口的热点流填满无损队列，引发 pause 帧沿路径传播，其他租户的 RDMA 延迟从微秒级升到毫秒级。

[根因机制] 只验证了无丢包吞吐，没有验证拥塞隔离和 pause 传播；队列与优先级映射不当。

[修复设计] 分离训练与普通流量、限制 PFC 优先级、启用 ECN 提前标记、监控 pause duration 和队列长度，并进行热点/拔线演练。若无法保证网络运维能力，选择 TCP 或降低 RDMA 依赖可能更稳健。

### 案例 D：collective 死锁

[事实] 多卡训练中某个 rank 在数据加载异常后提前返回，其他 rank 继续进入 all-reduce，作业一直挂起，监控只看到 GPU 利用率为 0%。

[根因机制] collective 要求参与者以相同顺序调用；单个 rank 的异常没有传播到全局。

[修复设计] 为每轮 collective 设置 watchdog 和全局取消；记录 rank、序号、消息大小和拓扑；将数据加载错误转为可广播的失败状态；故障时保存最小复现信息而不是无限等待。

## 3.11 观测、容量与排障清单

[测量] 建议建立四层指标：

1. **应用层**：请求数、成功率、消息大小、序列化/反序列化时长、TTFT、TPOT、取消和重试。
2. **RPC 层**：连接池等待、stream 数、deadline 超时、状态码、flow-control window、每下游调用的重试次数。
3. **主机/协议层**：CPU softirq、socket backlog、cwnd、RTT、重传、ECN、丢包、NIC ring、DMA 错误。
4. **交换机/RDMA 层**：端口利用率、队列深度、PFC pause、ECN mark、链路错误、QP 状态、CQ overrun。

[设计判断] 指标要带上租户、模型版本、机架、NIC、GPU、消息大小桶和优先级等维度，但要控制高基数。对请求级诊断使用采样 trace，把完整 payload 留在受控的调试环境，避免把敏感数据写入日志。

[机制] 排障时按“先证据、后优化”顺序：

- 先确认问题范围：所有请求还是特定租户/机架/消息大小？
- 再拆分时间：排队、序列化、传输、远端排队、计算各占多少？
- 再检查资源：CPU、内存带宽、NIC、交换机队列、GPU/NIC 拓扑是否饱和？
- 最后做单变量实验：改变并发、窗口、批大小、压缩或协议，观察分布而不是单点。

[事实] “带宽测试跑满”只证明大流吞吐在某一场景可达；它不证明小消息延迟、混合流隔离、故障恢复或多租户公平性。容量评估应包含长尾大小、微突发、连接重建和依赖故障。

## 3.12 六个理解检查（含答案）

### 问题 1：为什么一次 `send()` 对应不到一次 `recv()`？

**答案**：[机制] TCP 提供字节流而非消息边界。内核可能把多次 send 合并，也可能把一次 send 拆成多个段；接收端必须使用长度前缀、分隔符或固定长度 framing，并循环读取直到完整消息。UDP 保留数据报边界，但不提供 TCP 的可靠有序语义。

### 问题 2：带宽翻倍后 p99 仍不变，可能是什么原因？

**答案**：[推断] 瓶颈可能在固定 RTT、连接/线程池排队、序列化、CPU、重传、下游模型队列或批处理等待。若消息很小，`RTT + α` 主导，增加 B 对 `S/B` 项几乎没有影响。应通过 trace 和队列指标验证，而不是继续升级链路。

### 问题 3：TCP 流量控制和拥塞控制有什么区别？

**答案**：[机制] 流量控制保护接收端，主要由 rwnd 表示；拥塞控制保护网络，主要由 cwnd 表示。发送窗口近似两者最小值。应用读取慢会缩小 rwnd；丢包、ECN 或 RTT 信号会让算法调整 cwnd。

### 问题 4：为什么无界队列会把短暂突发变成 OOM？

**答案**：[机制] 当长期生产速率 λ 大于消费速率 μ，队列按 `(λ-μ)t` 增长；即使突发最终结束，积压也会延迟很久。无界队列把过载隐藏在内存里，直到 OOM 或全局超时。有界队列通过阻塞、降级或拒绝让上游尽早感知。

### 问题 5：RDMA 单边写是否自动保证远端应用看到一致数据？

**答案**：[事实/机制] 不保证。RDMA write 绕过远端 CPU，但应用仍需定义写入完成、内存可见性、版本号和并发协议；错误使用 rkey 或在写入完成前复用 buffer 会造成数据损坏。RDMA 是搬运机制，不是事务系统。

### 问题 6：ring all-reduce 和 tree all-reduce 如何按消息大小选择？

**答案**：[机制] 小消息通常由启动延迟 α 主导，树的 `log p` 步数可能更有利；大消息由带宽项 β 主导，ring 能较好利用链路并均摊数据。真实选择还取决于拓扑、拥塞、协议和实现阈值，应通过 NCCL 日志和基准验证。

## 3.13 练习

1. **半包与粘包**：修改 `framed_client.py`，随机把一个 frame 拆成 1～5 段发送；再连续发送 10 个 frame 不等待响应。验证服务仍能按长度解析，并记录每段写入对 p99 的影响。
2. **窗口与 BDP**：在两个测试命名空间之间用 `tc netem` 注入 50 ms RTT，分别设置 socket buffer 为 64 KiB、1 MiB、16 MiB，测量大消息吞吐。解释窗口小于 BDP 时为何无法跑满。
3. **重试放大**：实现三层本地 RPC stub，每层在超时后最多重试两次。推导最坏调用次数，加入指数退避和抖动，比较下游 QPS 与恢复时间。
4. **序列化与内存**：使用 JSON、MessagePack 和一种 schema 二进制格式编码同一批图像元数据，测量字节数、CPU、峰值 RSS 和 p99；写出不能从本机结果外推到跨机架的原因。
5. **背压策略**：在 asyncio 示例中加入高/低水位、每租户配额和取消信号。让一个慢客户端与多个快客户端混合，验证慢客户端不会耗尽全局内存。
6. **拓扑实验**：在支持多 GPU 的节点上运行 `nvidia-smi topo -m` 和 NCCL tests，交换进程到 GPU/NIC 的绑定，比较 all-reduce p50/p99；把测量结果与拓扑图对应起来。
7. **故障注入**：在测试环境临时增加 1% 丢包、关闭一个服务副本或让一个 collective rank 延迟 5 秒。验证 deadline、watchdog、重试和告警是否按设计工作，记录恢复时间和未完成请求数。
8. **容量报告**：给定目标 5,000 QPS、p99<100 ms、平均消息 32 KiB，使用 Little's Law 和实测服务时间估算所需并发与连接池范围；列出至少三个会使估算失效的假设。

## 3.14 来源地图与证据类型

下表把本章关键结论映射到可复查的一手资料。章节中的 `[事实]` 主要来自规范或官方文档；`[机制]` 是基于这些规范的解释；`[测量]` 是实验方法；`[推断]` 和 `[设计判断]` 明确标注为需要在目标环境验证的判断。

| 主题 | 来源 | 用途与证据边界 |
|---|---|---|
| TCP 规范 | [RFC 9293](https://www.rfc-editor.org/rfc/rfc9293) | TCP 状态、可靠字节流、拥塞控制接口；不规定某个 Linux 算法的具体性能 |
| IPv4 | [RFC 791](https://www.rfc-editor.org/rfc/rfc791) | IP 尽力而为转发与分片语义 |
| IPv6 | [RFC 8200](https://www.rfc-editor.org/rfc/rfc8200) | IPv6 头部和扩展机制 |
| HTTP/2 | [RFC 9113](https://www.rfc-editor.org/rfc/rfc9113) | 多路复用、stream、flow-control、HPACK 相关语义 |
| QUIC/HTTP/3 | [RFC 9000](https://www.rfc-editor.org/rfc/rfc9000), [RFC 9114](https://www.rfc-editor.org/rfc/rfc9114) | QUIC 传输与 HTTP/3 映射；实现性能需实测 |
| ECN | [RFC 3168](https://www.rfc-editor.org/rfc/rfc3168) | IP/TCP 显式拥塞通知 |
| Linux TCP | [ip-sysctl 文档](https://docs.kernel.org/networking/ip-sysctl.html) | CUBIC、BBR、窗口与内核参数；参数效果依版本和硬件 |
| socket 统计 | [ss(8) man page](https://man7.org/linux/man-pages/man8/ss.8.html), [nstat(8)](https://man7.org/linux/man-pages/man8/nstat.8.html) | 观测 cwnd、RTT、重传和协议计数 |
| netem | [Linux tc-netem](https://man7.org/linux/man-pages/man8/tc-netem.8.html) | 测试环境延迟、丢包和抖动注入 |
| gRPC | [gRPC 官方文档](https://grpc.io/docs/what-is-grpc/core-concepts/) | RPC 生命周期、stream、deadline、状态码；具体语言实现有差异 |
| Protocol Buffers | [protobuf 编码指南](https://protobuf.dev/programming-guides/encoding/) | schema、wire format、兼容性和 varint |
| RDMA verbs | [Linux RDMA 文档](https://docs.kernel.org/infiniband/user_verbs.html) | PD、MR、QP、CQ 和用户态 verbs；厂商扩展另行验证 |
| RDMA 核心 | [RDMA-core](https://github.com/linux-rdma/rdma-core) | 用户空间库与示例代码；版本会改变 API 行为 |
| RoCE 拥塞 | [RoCEv2 规范（IBTA）](https://www.infinibandta.org/ibta-specification-download/) | RoCE 传输和拥塞相关规范；交换机配置需厂商文档 |
| NCCL | [NVIDIA NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html) | collective API、拓扑和调试变量；性能依 GPU/NIC/版本 |
| NCCL tests | [nccl-tests](https://github.com/NVIDIA/nccl-tests) | 可重复的 all-reduce/all-gather 基准工具 |
| 排队关系 | [Little 1961 论文](https://doi.org/10.1287/opre.9.3.383) | `L=λW` 的稳定系统关系；不直接预测 p99 |
| 分布式追踪 | [OpenTelemetry 规范](https://opentelemetry.io/docs/specs/otel/) | trace/span/属性定义；采样开销需实测 |

[设计判断] 读者应把来源地图当作“从概念走向实验”的索引，而不是保证目标环境行为的承诺。任何涉及具体内核版本、NIC 固件、交换机队列或 GPU 拓扑的结论，都应在部署前重跑对应基准并保存原始数据、配置和版本信息。

## 3.15 小结

[事实] TCP/IP 提供通用、可靠的连接语义；RPC 在其上增加接口、序列化、超时和重试；RDMA 通过注册内存和 NIC offload 降低 CPU 路径，但提高了网络与生命周期管理要求；collective 把网络变成多方同步问题。

[机制] 延迟不是一个数字，而是一串排队、拷贝、传输、计算和等待的组合。吞吐、平均延迟和尾延迟之间存在权衡：更高并发、批处理和压缩可能提高吞吐，却把等待转移到队列、CPU 或流控窗口。

[设计判断] AI 系统网络优化的可靠流程是：画因果链 → 定义指标与预算 → 运行最小实验 → 按队列和时间段定位 → 只改一个变量 → 在故障和长尾场景回归。能解释“为什么更快、在什么条件下失效、如何监测失效”，才算真正掌握网络，而不是记住某个带宽数字。

## 3.16 进阶：协议细节如何反映到 AI 工作负载

### 3.16.1 小消息与大消息不是同一种优化题

[机制] 对小消息，固定开销包括系统调用、锁、协议头、TLS 记录和一次往返；有效带宽几乎不起作用。可以把总成本写成 `T_small ≈ α_syscall + α_tls + α_rtt + S/B`，其中前三项远大于最后一项。批量发送、连接复用、减少 stream 创建和合并元数据，通常比更换 100 Gb/s 到 200 Gb/s 链路更有效。

[机制] 对大消息，`S/B` 和内存搬运占主导，发送窗口、DMA、NUMA 和压缩决定吞吐。把大消息切成太多小 frame 会增加每帧头部与完成事件；完全不切分又会增加队头阻塞和取消延迟。合理的 frame 大小要用目标链路和消息分布测得。

[测量] 建议按 1 KiB、4 KiB、64 KiB、1 MiB、16 MiB 五个桶分别报告吞吐和 p99，并把“同一连接顺序发送”和“多连接并行发送”分开。对推理系统，再加 token 数和图像分辨率两个维度，因为字节大小不能完全代表解码和计算量。

### 3.16.2 TLS、服务网格和可观测性的隐藏成本

[事实] TLS 提供机密性、完整性和端点认证，但握手、证书验证、记录加解密和密钥更新会消耗 CPU 与延迟。连接复用、会话恢复和硬件加速可降低成本，仍需在实际密码套件、CPU 型号和消息大小下测量。

[机制] 服务网格通常在应用旁边运行代理，形成应用 → sidecar → 内核 → 网络 → sidecar → 应用的额外跳数。它可能带来统一重试、熔断和指标，也可能因为默认重试、缓冲或 TLS 再加密放大尾延迟。每个代理都应明确超时预算、重试策略和缓冲上限。

[设计判断] 可观测性同样是数据路径的一部分。同步写日志、全量 payload 采集和高频 trace export 会占用 CPU、内存和网络，形成“监控导致变慢”的反馈。采用异步批量导出、采样、脱敏和丢弃策略，并把 exporter 阻塞计入容量模型。

### 3.16.3 多租户公平与优先级

[机制] 在共享链路上，先进先出并不保证公平。一个大文件流可以占满发送窗口，让小的交互请求等待；严格优先级又可能饿死低优先级批处理。可使用分层队列、令牌桶、加权公平队列和每租户并发上限，配合明确的丢弃或降级策略。

[测量] 为每个租户记录到达率、服务率、排队时间和丢弃数，并检查最差租户的 p99，而非只看全局平均。进行租户权重变更和突发流量演练，验证策略是否在故障时保持可预测。

[推断] 若全局 p99 看似良好，但某小租户 p99 极差，说明聚合指标掩盖了不公平。容量和 SLA 应按租户、优先级和消息桶分层，必要时为关键流量预留独立队列或链路。

### 3.16.4 取消、截止时间与资源回收

[机制] 客户端取消请求后，网络连接关闭只是第一步；服务端必须把取消传播到 RPC handler、批处理器、GPU kernel、RDMA work request 和下游调用。否则“用户看不到结果”的请求仍占用计算和缓冲，最终把过载放大。

[测量] 记录从取消信号到每一层资源释放的时间：socket、stream、队列项、GPU event、QP WR。统计取消后的“幽灵工作”比例和占用时长。若资源回收 p99 很长，优先修复清理路径，而不是一味加大容量。

### 3.16.5 可复现基准的最小记录集

[事实] 网络结果高度依赖版本和环境。每次基准至少保存：主机型号、CPU 微码、内核与 TCP 拥塞算法、NIC 固件和驱动、交换机端口速率与 MTU、GPU/驱动/NCCL 版本、NUMA 与 CPU affinity、容器限制、消息生成方式、并发模型、预热时长、样本数和原始分位数。

[设计判断] 将配置与结果一起版本化，使用脚本自动生成报告。报告中区分“观察到的数值”“基于公式的估算”和“待验证的推断”。同一实验至少重复三次，若分位数差异大，先寻找环境抖动原因，再取平均；不要通过挑选最好一次来制造回归。

## 3.17 实验报告模板（可直接复制）

```text
实验名称：
目标假设：[例如：连接复用可降低小消息 p99]
环境：主机/CPU/内核/NIC/交换机/容器限制
协议参数：TCP 算法、MTU、TLS、RPC 框架、压缩
工作负载：消息大小分布、QPS、并发、租户/优先级
步骤：预热、稳态时长、故障注入、恢复观察窗口
指标：p50/p95/p99、吞吐、CPU、内存、重传、队列深度、错误率
结果：原始样本链接与摘要
证据分类：[事实]/[机制]/[测量]/[推断]/[设计判断]
限制：哪些条件尚未覆盖，哪些结论不能外推
下一步：一个可证伪的改动或实验
```

[设计判断] 统一模板能防止“跑了一个 benchmark 就宣布优化成功”。每个结论都要回答：在什么输入、什么拓扑、什么负载和什么失败条件下成立？哪些观测能推翻它？

## 3.18 本章术语速查

- **RTT**：往返时延；从发送到收到对端响应或确认的时间
- **BDP**：带宽时延积；链路带宽乘 RTT，近似在途数据需求
- **cwnd/rwnd**：TCP 拥塞窗口/接收窗口
- **ECN/PFC**：显式拥塞通知/优先级流控，常见于数据中心无损队列
- **Framing**：应用层消息边界编码，解决 TCP 字节流无边界问题
- **Head-of-line blocking**：队头阻塞，前一项阻塞后续项交付
- **Backpressure**：背压，消费者变慢时让生产者减速或拒绝
- **RDMA**：远程直接内存访问；通过 NIC DMA 访问已注册远端内存
- **MR/QP/CQ**：内存区域、队列对、完成队列
- **Collective**：多参与者协同通信，如 all-reduce、all-gather
- **α-β 模型**：把通信成本拆成启动延迟 α 与每字节成本 β
- **TTFT/TPOT**：首 token 延迟/每 token 时间

[小结] 这些术语只有放在时间线、队列和拓扑中才有意义。能把“一个词”对应到可观测计数器、代码位置和故障动作，才是真正的工程知识。

## 3.19 复盘问题：把网络事故写成可学习的因果链

[设计判断] 事故复盘不应停在“网络抖动”四个字。请按五步记录：第一，列出用户可见症状，例如 p99、超时比例和受影响租户；第二，画出请求经过的队列、协议和拓扑；第三，标记第一个出现异常的计数器，而不是最晚爆炸的告警；第四，说明为什么现有背压、超时或重试没有阻止扩散；第五，提出一个可在实验环境证伪的修复假设。

[测量] 例如，若告警显示 p99 从 80 ms 升到 800 ms，同时交换机 ECN mark 增加、TCP 重传不变、模型执行时间稳定，则更像是队列拥塞而非链路断开。下一步可在隔离端口注入同等突发，验证 p99 是否重现；若重现，再比较限速、批窗口和优先级队列的恢复时间。

[推断] 复盘应区分触发条件与放大器。微突发可能是触发条件，无界缓冲、固定超时重试和缺少租户配额则是放大器。修复只消除触发条件而不修复放大器，下一次不同流量形状仍会导致同类事故。

[设计判断] 最有价值的网络知识不是一张“最佳配置”表，而是一套在新硬件、新内核、新拓扑和新负载下仍可复用的提问方法：数据从哪里来？在哪里排队？谁负责减速？哪一个计数器能证明假设？如果它失败，系统如何在预算内退化？

[小结] 以证据驱动网络设计，才能让速度、成本与可靠性同时可解释、可复现、可回滚。

## 3.20 内核路径与中断预算：从数据包到用户线程

[机制] 即使应用使用异步 API，数据仍要经过 NIC 接收队列、NAPI 轮询、软中断、socket 缓冲区和用户线程。中断合并把多个数据包批量交给 CPU，可降低每包中断开销，却会增加等待；关闭合并可能改善小消息延迟，却把 CPU 消耗推高。在高并发 AI 网关中，软中断若与模型线程争抢同一核心，会出现“网络 p99 上升、CPU 平均利用率并不高”的假象。

[测量] 用 `ethtool -c` 查看中断合并参数，用 `/proc/interrupts`、`mpstat -P ALL` 和 eBPF 工具观察 IRQ、softirq 与用户线程的核分布。将 IRQ affinity、RPS/XPS、线程绑核作为实验变量，每次只改变一项，并同时记录 p50/p99、每包 CPU 周期和上下文切换数。容器环境还要记录宿主机是否共享该核心，否则容器内的 CPU 百分比无法解释尾延迟。

[设计判断] 低延迟小消息服务可以选择较短的中断合并窗口和专用核心；大流吞吐服务则可接受更长批量以换取 CPU 效率。两类流量混在同一队列时，优先级和队列分离比全局调整参数更可靠。任何 sysctl 或 ethtool 改动都应先在影子流量上验证，并准备回滚命令。

## 3.21 版本升级与灰度：网络优化也需要回滚

[事实] 内核、NIC 驱动、RPC 库和 NCCL 版本可能改变默认拥塞算法、TLS 实现、队列阈值和 collective 选择。即使 API 不变，尾延迟和故障行为也可能变化。升级前应保存旧版本的配置、固件和基准原始样本，确保可以重放同一负载。

[机制] 灰度时把流量按租户、机架或请求哈希分组，比较新旧组的 p50/p95/p99、重传、ECN、PFC、取消率和错误码。若只比较平均吞吐，可能错过少数租户的长尾。灰度窗口应覆盖冷启动、连接重建、批处理高峰和至少一次故障注入。

[推断] 如果升级组的 p50 变好但 p99 变坏，通常意味着平均路径缩短、资源竞争或队列长尾变严重；如果 p50 与 p99 都变差，优先检查协议协商、MTU、拓扑映射和版本兼容。只有在回滚后指标恢复，才能把相关性提高为较可信的因果证据。

[设计判断] 将“可观测、可灰度、可回滚”作为网络特性的一部分，而不是发布流程的附加项。对 RDMA 和 GPU collective，尤其要验证异常退出后的 QP、通信组和显存缓冲是否被释放；否则一次失败升级可能留下只能通过重启节点清理的隐性状态。

## 3.22 端到端预算示例：给 100 ms SLA 留出余量

[机制] 假设在线生成请求的 SLA 为 p99 100 ms，可先制定预算：入口排队 8 ms、鉴权与路由 5 ms、序列化与拷贝 12 ms、网络往返 10 ms、模型首步计算 45 ms、返回与渲染 10 ms，预留 10 ms 作为抖动余量。预算不是事实，而是需要通过 trace 验证的设计假设。

[测量] 当某次发布把序列化从 8 ms 降到 4 ms，却使连接池等待从 6 ms 升到 20 ms，端到端 p99 可能反而变差。应比较预算各项的分位数和相关性，而不是只看单个优化点的平均耗时。若网络和模型计算具有相关拥塞，简单相加各自 p99 会低估联合尾部。

[设计判断] 预算应带有“触发动作”：入口排队超过 8 ms 时降低批窗口；连接池等待超过 5 ms 时限制新流；网络重传超过阈值时切换副本或降级；GPU/collective 超时则取消整条请求链。把动作写入 runbook，并在演练中确认真正生效，才能让预算成为控制回路而不是仪表盘上的数字。
