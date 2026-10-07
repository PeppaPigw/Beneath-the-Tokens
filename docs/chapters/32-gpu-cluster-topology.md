---
id: ch32-gpu-cluster-topology
title: GPU 集群拓扑、网络与大规模推理/训练的故障边界
slug: /chapters/32-gpu-cluster-topology
description: 从 GPU、PCIe、NVLink 到节点间网络和 collective，建立训练与推理的拓扑、容量和故障边界
sidebar_position: 32
level: advanced
prerequisites:
  - ch03-ai-networking
  - ch05-gpu-cuda
  - ch08-distributed-collectives
  - ch09-distributed-training
  - ch10-training-ops
  - ch16-model-serving-system
  - ch17-kubernetes-gpu-orchestration
  - ch20-observability-debugging-incident-response
  - ch21-ai-reliability-engineering
learning_objectives:
  - 能把 GPU、NVLink/NVSwitch、PCIe、NIC、交换机、机架和故障域画成可审计的拓扑图
  - 能从 rank mapping、collective 算法和消息大小推导节点内/节点间字节、带宽瓶颈与尾延迟
  - 能解释训练同步、推理副本、TP/PP/EP 和 elastic recovery 在 GPU/节点/网络故障下的不同边界
  - 能用 NCCL、Slurm、Kubernetes device plugin 和 Prometheus/NVML 指标设计证据链，而不是只看平均吞吐
  - 能运行 CPU-only toy lab，复现 all-to-all、ring all-reduce 和故障注入结果，并明确 toy 与生产测量的差异
  - 能写出上线前的拓扑契约、容量账本、回滚条件和演练清单
estimated_hours: 48
hardware: CPU-only lab required; real GPU/NCCL measurements optional and cost-bounded
risk_level: L4
last_verified: 2026-10-07
---

# 第32章　GPU 集群拓扑、网络与大规模推理/训练的故障边界

> 一台机器里有八张 GPU，不代表你拥有“一台八卡 GPU”。真正决定训练和推理行为的，是 GPU 到 GPU 的路径：是否通过 NVLink/NVSwitch，是否绕过 PCIe root complex，是否穿过同一个 NIC、同一个 ToR 交换机或一对 spine，是否和别的租户共享队列。拓扑还决定故障的形状：一张卡坏了，可能只影响一个推理副本；一个 NIC 坏了，可能让跨节点 collective 全部卡住；一个交换机或路由策略出错，可能让看似健康的 GPU 在同步屏障上永久等待。本章把“拓扑”当成协议，而不是一张静态机器照片。

本章和第3章的边界是：第3章介绍网络协议、拥塞与连接；这里把这些概念映射到 GPU collective 的字节和 rank。和第8章的边界是：第8章建立 all-reduce、all-to-all 等集体通信的语义；这里进一步讨论它们在 NVLink、PCIe、RDMA 和多层交换机上的路径与故障域。和第17章的边界是：第17章讲 Kubernetes 如何分配 GPU；这里强调调度器必须把拓扑、NUMA、NIC 和故障域一起作为可验证约束。和第28章的边界是：第28章讨论 vLLM/SGLang 的 serving；这里讨论为什么同一个 serving worker 在不同 GPU 拓扑上会有不同的 tail latency 与降级方式。

## 32.1 先画“路径”，再谈带宽

### 32.1.1 五层拓扑模型

新人常把拓扑理解成“GPU 数量 + 节点数量”。工程上至少要画五层：

1. **设备层（GPU）：** 每个 GPU 有 device id、PCI BDF、显存容量、时钟状态和 NVML 健康状态。GPU 的编号不等于物理顺序；`CUDA_VISIBLE_DEVICES` 可以重新编号，容器内的 `cuda:0` 不应直接当作宿主机的物理 GPU 0。
2. **节点内互连层：** NVLink、NVSwitch、PCIe switch 和 PCIe root complex 连接 GPU、NIC、NVMe。两张 GPU 都在同一节点，可能有高带宽 NVLink，也可能只能经 PCIe host memory 绕行。
3. **节点出口层：** NIC 数量、端口速率、NUMA 归属、GPUDirect RDMA 支持和 PCIe root 位置决定 GPU 到网络的实际路径。GPU 与 NIC 不在同一 NUMA 节点时，CPU socket 可能成为隐含跳点。
4. **网络层：** ToR、leaf/spine、rail、链路 oversubscription、PFC/ECN、路由哈希和队列调度影响跨节点 collective。所谓“200 Gb/s NIC”是链路速率，不是每个 rank 都能稳定得到的有效 payload 带宽。
5. **机架和故障域层：** 电源、风扇、机架顶交换机、spine、机房区域和维护窗口构成不同粒度的 blast radius。副本若分布在同一个 ToR 下，机架级容错只是幻觉。

可审计的拓扑记录至少包含：`host_id`、`gpu_uuid`、`pci_bdf`、`numa_node`、`nvlink_peers`、`nic_name`、`nic_pci_bdf`、`switch_id`、`rack_id`、`failure_domain`、驱动/CUDA/NCCL 版本和生成时间。只记录“节点 A 有 8 卡”无法解释 NCCL 为什么选择某条路径，也无法在换机后判断 placement 是否改变。

### 32.1.2 带宽、延迟和消息大小

对一个 payload 大小为 (M) bytes 的单向传输，最简单的时间模型是：

\[
T(M)=L+\frac{8M}{B_{bit}},
\]

其中 (L) 是启动延迟，(B_{bit}) 是有效 bit/s。这个式子只适合做数量级和边界判断。真实 GPU 通信还有协议分片、credit、DMA 注册、PCIe replay、链路共享、集体算法和 kernel launch 开销。小消息通常被延迟主导，大消息更容易暴露带宽和拥塞；把两者混成一个“GB/s”会误导容量规划。

如果 rank i 发给多个 peer，关键是最大完成时间而不是平均时间：

[
T_{step}≥max_ileft(L_i+rac{8S_i}{B_i}
ight),
]

其中 (S_i) 是 rank 或链路的发送字节。一个 hot rank 发送了平均值的两倍，其他 rank 即使空闲也要在 collective fence 等它。报告应同时保存 `mean`, `max`, `p95`, `p99`, `bytes_by_path` 和 `nonzero_peer_count`。

### 32.1.3 Toy 与生产边界

本章的 CPU lab 使用整数 rank、均匀 payload 和透明的带宽/延迟参数。它能验证守恒（发出的字节等于计划的字节）、路径分类、rank mapping 和故障范围；它**不能**证明 NVIDIA GPU、AMD GPU、NCCL、RCCL、UCC、InfiniBand 或 RoCE 的实际性能。生产测量必须锁定 GPU 型号、驱动、CUDA、NCCL、交换机配置、MTU、PFC/ECN、进程绑定、消息大小和并发租户，并用 `nccl-tests`、应用 trace 和 NVML/NIC 计数器互证。任何报告把 toy 的 `200 Gb/s` 或 `latency_proxy_ms` 写成硬件保证，均应拒绝发布。

## 32.2 GPU、NVLink、PCIe 与 NIC：同一节点也有多个故障边界

### 32.2.1 NVLink/NVSwitch 不是“无限共享总线”

NVLink 提供 GPU 之间的高速点到点链路，NVSwitch 把多个 GPU 连接成更规则的交换结构。即便是全连接逻辑拓扑，物理链路、交换芯片和注入带宽仍有限。需要区分：

- **GPU 对等路径：** `nvidia-smi topo -m` 显示的 `NV#`, `PIX`, `PHB`, `SYS` 等类别描述路径层级，不应被直接当成精确带宽。
- **注入/汇聚点：** 某个 GPU 或 NVSwitch 的端口故障可能只影响部分 pair，也可能把一个 collective 的所有流量压到较慢路径。
- **链路健康：** replay、CRC、Xid、温度和降频会让链路“可用但变慢”。只监控进程退出无法发现这类性能故障。

NVIDIA 的 [NVLink 文档](https://docs.nvidia.com/networking/display/nvlink5) 描述了代际能力；[NVIDIA System Management Interface 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html) 提供拓扑、Xid 和设备健康的查询入口。具体平台要以机型和固件为准，不能把某一代 DGX 的图复制到任意服务器。

### 32.2.2 PCIe root complex、NUMA 和 GPUDirect RDMA

当 GPU 与 NIC 位于不同 PCIe root complex，数据可能经过 CPU socket 或 QPI/UPI。GPUDirect RDMA 可以让 NIC 直接读写 GPU 显存，但前提是驱动、IOMMU、ACS、BAR、NIC 固件和拓扑均满足要求。常见反例是：容器看到了 GPU 和 NIC，却因为权限或 IOMMU 配置退化为 host-staging；吞吐降低、CPU 占用升高，应用仍然“功能正常”。

审计方法应结合：

- `nvidia-smi topo -m`：GPU/NIC 的路径类别和 CPU affinity；
- `lspci -tv`、`numactl -H`：PCIe 树和 NUMA 节点；
- `ibdev2netdev`、`ethtool -i`：RDMA 设备、端口和驱动；
- NCCL 初始化日志：实际选择的 interface、channel、P2P/SHM/NET transport；
- 应用 trace：通信 kernel、CUDA event 时间和 host-side wait。

一个 rank 显式绑定 GPU 但没有绑定 NIC/CPU 亲和性，可能在节点升级后悄悄改变路径。启动合同应把 rank、GPU UUID、CPU core、NUMA 和 NIC 记录为同一个不可分割的 placement 证据。

### 32.2.3 多 NIC、多 rail 与 rail-aware placement

多 NIC 节点常采用 rail-aware 设计：GPU 0-3 靠近 NIC 0，GPU 4-7 靠近 NIC 1，跨 rail 流量会经过额外交换。要避免把所有 rank 的出口哈希到一个端口，应让并行组和 NIC rail 对齐。训练时可以按 DP/TP/EP group 选择不同 rail；推理时还需考虑请求路由和 KV cache 迁移是否制造热点。

“启用两张 NIC”不是充分条件。应验证每个 rank 的 bytes-by-nic、端口计数器、拥塞标记和失败切换。一个 NIC 断开后，系统可能继续运行但所有流量落到剩余端口；这类降级必须进入 SLO，而不是等到带宽不足时才发现。

## 32.3 Rank mapping：把并行维度放到正确的物理位置

### 32.3.1 从 global rank 到并行坐标

大模型训练常把 world size 分解为：

[
W = D 	imes P 	imes T 	imes E,
]

其中 D 是 data parallel，P 是 pipeline parallel，T 是 tensor parallel，E 是 expert parallel。不同框架的坐标顺序不同；不要假设 global rank 连续就代表同一组。一个明确的映射示例是：

```text
rank = (((dp * P) + pp) * T + tp) * E + ep
```

这只是本章的示例，不是 Megatron、DeepSpeed 或任何框架的默认保证。部署记录必须输出每个 process group 的 rank 列表、leader、通信后端和目标拓扑。

### 32.3.2 哪些维度应该局部化

一般原则（仍需以测量验证）是：

- **TP：** 每层有高频 all-reduce/all-gather，优先放在 NVLink/NVSwitch 内，避免跨节点；如果模型大到必须跨节点，需计算每层通信是否淹没 GEMM。
- **PP：** stage 之间以 microbatch 激活传递，跨节点可以接受，但要控制 microbatch 数和链路抖动。stage 不均衡会产生 pipeline bubble。
- **EP：** token dispatch 的 all-to-all 对跨节点 bytes 很敏感，专家 owner 和数据 rank 的映射应考虑 rail/机架；hot expert 会把某条链路推向瓶颈。
- **DP：** 梯度 all-reduce 的频率较低但 payload 大，适合跨节点；分层 all-reduce 可先在节点内聚合再过网络。

当多个维度竞争同一 NVLink 或 NIC 时，应画出“每一步的通信时序”，而不是只列并行度。TP reduce、EP all-to-all、DP gradient reduce 叠加在同一窗口时，峰值带宽远高于任何单项基准。

### 32.3.3 Placement 的不变量

可把 placement 合同写成四个不变量：

1. **可达性：** 同一 process group 的 rank 必须能在规定 timeout 内互相通信；不能存在只在某个 job 上出现的隐藏防火墙或 RDMA ACL。
2. **局部性：** 要求本地的 group（例如 TP）不得跨故障域；若不可避免，必须显式标注和重新评估 SLO。
3. **独立性：** 推理副本不得共享同一个单点 NIC、ToR、电源或 MIG parent（除非接受共享故障）。
4. **可重现：** 同一版本的 scheduler、节点标签和拓扑发现应生成同样的 mapping；重启后 rank 顺序改变要进入 checkpoint/日志。

## 32.4 Collective 算法与拓扑：ring、tree、hierarchical

### 32.4.1 Ring all-reduce

Ring all-reduce 把 payload 分块，经过 reduce-scatter 和 all-gather 两个阶段。每个 rank 大致发送 (2(N-1)/Ncdot M) bytes，但完成时间受 ring 上最慢边限制。若逻辑 rank 顺序把两个节点交错排列，几乎每条边都跨节点；若先放完节点内 rank，再跨节点，跨节点边数会少很多。算法本身没变，拓扑映射却改变了瓶颈。

Ring 的优点是带宽利用率稳定、实现成熟；缺点是单个链路或 rank 故障可能阻断整环。NCCL 的 channel、chunk 和算法选择会依据拓扑和消息大小变化，不能从一条日志推断所有 workload 的行为。[NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)和 [nccl-tests](https://github.com/NVIDIA/nccl-tests) 是验证实际路径的起点。

### 32.4.2 Tree、double tree 与分层 collective

Tree 算法的消息沿树向上聚合、再向下广播，延迟更适合小消息；树根或高层链路成为故障和拥塞敏感点。多树可以分散根，但会增加调度复杂度。分层 all-reduce 常按三阶段执行：节点内 reduce、节点间 reduce、节点内 broadcast。它把昂贵的跨节点流量压缩为每个节点一次或少数几次 payload，前提是节点内拓扑足够快且节点间连接均衡。

对 GPU 集群，不能只比较“ring vs tree”的论文复杂度。要记录：算法、协议（LL/LL128/Simple 等）、channel 数、节点内/节点间路径、消息大小、并发 stream、链路计数和重试。NCCL 环境变量是排障工具而不是永久配置；临时设置 `NCCL_DEBUG=INFO` 可以收集证据，长期固定 `NCCL_ALGO` 或 `NCCL_PROTO` 可能阻止版本升级后的更优选择。

### 32.4.3 All-to-all 与 MoE/交换层

All-to-all 的矩阵 (M_{ij}) 直接暴露数据倾斜：rank i 发给 rank j 的 token、KV block 或专家激活越多，链路 j 越热。rank i 的发送字节为 (S_i=Bsum_j M_{ij})，接收字节为 (R_i=Bsum_j M_{ji})，step 的通信时间近似由 `max(send, recv)` 和最慢物理路径决定。均匀 token 在逻辑上不保证均匀物理路径；若专家集中在一个节点，跨节点字节仍会热。

第31章的 EP 路由讨论 token 语义，本章关注 owner/rank 到 NIC/switch 的物理路径。出现 all-to-all 尾延迟时，先区分：router skew、专家 owner skew、rank mapping、NIC rail 失衡、交换机拥塞，还是 GPU kernel 本身变慢。把所有问题都归咎于“网络慢”会让修复方向错误。

## 32.5 节点间网络：InfiniBand、RoCE、Ethernet 与拥塞

### 32.5.1 链路速率不等于应用带宽

InfiniBand 和 RoCE 都可以提供 RDMA，但运维合同不同。IB 依赖 fabric manager、子网管理和端口状态；RoCE 在以太网上运行，通常依赖 PFC/ECN、无损队列和交换机缓冲配置。普通 TCP fallback 可能让作业“还能跑”，却把 step 时间放大数倍。应用必须记录 transport，而不是只记录 hostname 和端口。

官方资料入口包括 [NVIDIA GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)、[NVIDIA DOCA RDMA 文档](https://docs.nvidia.com/doca/sdk/rdma-programming-guide/index.html)、[Linux rdma-core](https://github.com/linux-rdma/rdma-core) 和 [OpenFabrics 企业指南](https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17)。这些资料描述能力和接口，不能替代目标集群的 burn-in。

### 32.5.2 PFC、ECN 与拥塞传播

PFC 可以在优先级队列上暂停发送，避免丢包，但配置错误会产生 pause storm：一个拥塞端口把暂停扩散到不相关租户。ECN 则通过标记提示发送端降速；应用和驱动必须正确响应。调试时应同时采集：交换机队列深度、PFC pause 帧、ECN mark、端口丢包/重传、RoCE CNP 和 NCCL collective 时间。

一个典型故障是：训练作业平均带宽下降不多，但 p99 step 时间偶尔飙升。原因可能是交换机 buffer 在多个 job 同时进入 all-reduce 峰值时耗尽；均值吞吐看不出，collective barrier 却把尾部放大。容量测试必须并发真实作业或至少加入背景流，不能只跑单 job 的空载带宽。

### 32.5.3 MTU、路由和隔离

MTU 不一致会导致分片、丢包或连接建立失败；路由 ECMP 哈希会让大流集中在某条 spine 链路。训练 job 的 source port、rank 顺序和消息分片模式若固定，可能长期撞上同一哈希。网络变更需做 before/after evidence：端到端路径、MTU、接口、VLAN/VRF、ACL、ECN/PFC、NCCL transport 和应用 p99。

多租户环境要把 QoS 和配额纳入 job 合同。单个大规模 all-to-all 作业不应默认占用整个 fabric；如果必须独占，应由调度器声明并在拓扑标签中体现。否则另一个团队的延迟回归会被错误归因给模型或 GPU。

## 32.6 训练的故障边界：同步、检查点与恢复

### 32.6.1 为什么一张卡会停掉全局 step

数据并行和张量并行通常包含同步 collective。一个 rank OOM、Xid、ECC 错误、Python 异常或网络超时，会让其他 rank 在 collective 上等待。NCCL 的异步错误可能不会在发生点抛出，而是在后续 CUDA API 或同步处才暴露。应用应设置合理的 `NCCL_ASYNC_ERROR_HANDLING`、process-group timeout，并把 rank、collective、step、tensor bytes 和最近一次健康 heartbeat 写入日志。

“把 timeout 调大”不是修复。它只能把故障从快速失败变成长时间占用 GPU。重试前需确认：是否所有 rank 都退出/清理了 communicator，是否有僵尸进程持有 GPU，是否 checkpoint 版本一致，是否网络路径已经恢复。没有这些证据，重试可能让第二次作业撞上同一个坏 rank。

### 32.6.2 GPU、节点、NIC、交换机的恢复策略

- **GPU 级：** 只缺一个 rank 时，默认中止同步 job；若框架支持 elastic world size，必须重新构建 optimizer、数据 sampler 和并行组，不能只删掉一个进程。
- **节点级：** 节点上的多个 rank 同时消失；先隔离节点并保留 NVML/Xid、dmesg、BMC 和交换机日志，再决定替换或回收。不要把坏节点立即放回池中。
- **NIC 级：** 可能不影响本地推理，却让跨节点 all-reduce 退化或失联。验证另一张 NIC 的 rail、路由、PFC/ECN 和 GPUDirect 状态；只看 ping 不足以证明 RDMA 正常。
- **交换机级：** 影响多个作业和故障域。优先切换到备用路径或停止受影响作业，保护 checkpoint 和租户隔离，避免同时重启大批作业造成 thundering herd。

### 32.6.3 Checkpoint 与 topology drift

Checkpoint 除了模型和 optimizer，还应记录：world size、DP/TP/PP/EP、rank mapping、expert owner、数据 sampler offset、dtype、通信 backend、拓扑摘要和代码/容器 digest。恢复时要检查 topology drift：GPU 数量变了、节点内 NVLink 变了、NIC rail 变了、rank 顺序变了，都可能让性能和数值路径变化。

如果框架支持从不同 world size 恢复，需验证参数切片、optimizer shard、RNG、数据顺序和 loss 曲线；“能加载 checkpoint”不等于“可继续训练”。恢复验收至少跑固定小 batch golden step，再跑短程吞吐和通信 histogram。第10章的训练运维合同与 [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html) 可作为实现入口。

## 32.7 推理的故障边界：副本、并行和降级

### 32.7.1 单 GPU、副本和模型并行

单 GPU worker 的故障通常只影响它承载的请求；副本服务可以把新请求路由到健康 worker。TP/PP/EP worker 则是一个协同组：一张卡坏了，通常整个模型副本不可用。调度器要把“GPU 健康”提升为“协同组健康”，并为每个请求保存 group id、模型 revision 和 KV cache owner。

副本必须跨故障域放置。两副本在同一节点不同 GPU，可抵御单 GPU 故障，抵御不了节点/电源/NIC 故障；两副本在同一 ToR，不足以抵御 ToR 级故障。健康检查要包含模型前向、KV 分配、collective（若有）和网络路径，而不是只检查 HTTP 端口。

### 32.7.2 TP/PP 的请求级中止与重试

在 decode 阶段，TP group 每个 token 都可能需要 all-reduce。某个 rank timeout 时，重试请求到另一个完整副本通常比在半坏 group 上修补更安全。若请求带工具调用或外部副作用，重试必须携带幂等键；流式响应已发送部分 token 时，不能无条件从头拼接。SLO 应分别记录：首 token 延迟、每 token 延迟、流中断率、重试次数、切换时间和丢弃的 KV bytes。

### 32.7.3 EP 和动态 batch 的热点

EP serving 的 all-to-all 受请求内容影响：代码、JSON、长上下文或特定语言可能把 token 集中到少数专家。调度器不能只按请求数均衡，应观察每个 expert/rank 的 token、字节、队列和 p99。过载时的降级可以是限制 batch、把请求移到另一个副本、降低 top-k（若模型语义允许）或拒绝新请求；不能静默丢 token。

KV cache 迁移会把网络流量与模型通信叠加。若一个副本重启，缓存 warm-up、权重加载和请求转移可能同时打满 NIC。迁移策略要设带宽预算、优先级、取消和回滚，区分可重算的 prompt cache 与不可丢失的会话状态。

## 32.8 调度器与拓扑感知：Kubernetes、Slurm 和自定义平台

### 32.8.1 Kubernetes 资源不是完整拓扑

Kubernetes device plugin 可以分配 GPU，Topology Manager 可以协助 NUMA 对齐，但 GPU、NIC、NVLink、ToR 和故障域标签仍需平台提供。Kubernetes 的 [device plugin 文档](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)、[Topology Manager 文档](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/) 和 [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/) 描述机制；实际集群要验证 webhook、插件版本、MIG、容器权限和升级顺序。

建议把 placement 分成三层：

- **资源层：** GPU 型号、显存、MIG profile、驱动和 CUDA compatibility；
- **亲和层：** NUMA、GPU-NIC、NVLink island、hostNetwork/RDMA；
- **故障层：** node、rack、ToR、zone、电源和维护域。

只用 `nvidia.com/gpu: 8` 的 Pod 可能拿到八张可用但跨多个 NUMA/rail 的卡。大规模 job 应使用 gang scheduling，确保整个并行组一起获批或一起等待；部分分配会造成长时间占用和资源碎片。

### 32.8.2 Slurm 的 GRES、拓扑和隔离

Slurm 的 GRES 可以分配 GPU，`CUDA_VISIBLE_DEVICES` 负责进程可见性；但 GRES 分配并不自动保证 NCCL 的最佳 rank mapping。需要把 `gres.conf`、节点特性、cons_tres、CPU/内存绑定和自定义 topology plugin 一起审计。参考 [Slurm GRES 文档](https://slurm.schedmd.com/gres.html)、[cons_tres 文档](https://slurm.schedmd.com/cons_tres.html) 和 [topology 文档](https://slurm.schedmd.com/topology.html)。

训练脚本启动时应打印 scheduler job id、node list、local rank、global rank、GPU UUID、NIC 和 process group。job 重排或节点替换后，比较这些字段能快速确认性能回归是代码还是 placement。

### 32.8.3 Gang、配额和可抢占

大 job 需要同时拿到所有 rank。可抢占策略要定义：checkpoint 频率、通知窗口、正在进行的 collective 如何终止、KV cache 是否丢弃、以及租户错误预算。抢占一个 rank 而让其他 rank继续等待，会把 GPU 变成不可回收的泄漏资源。平台应在 cgroup/进程树、NCCL communicator 和网络连接上实现一致的取消。

## 32.9 可观测性：从“吞吐下降”定位到路径

### 32.9.1 四个时间轴

一次训练 step 或推理请求至少有四个时间轴：

1. **应用时间：** forward、backward、decode、queue、checkpoint；
2. **通信时间：** collective kernel、host wait、communicator init；
3. **设备时间：** GPU 利用率、显存、温度、时钟、ECC/Xid、NVLink counter；
4. **网络时间：** NIC bytes、packet/ECN/PFC、重传、端口错误、队列深度。

将这些时间轴用 `job_id`, `step_id`, `rank`, `group_id`, `collective_id`, `gpu_uuid`, `nic`, `switch_port` 关联。一个 p99 step 只有在这些字段齐全时才可追溯。

### 32.9.2 最小指标集合

**GPU/NVML：** utilization、memory used/free、power、temperature、clock、ECC、Xid、NVLink tx/rx。**NCCL/应用：** collective name、algorithm/protocol、message bytes、channel、duration、timeout、error code。**NIC/RDMA：** port state、link speed、tx/rx bytes、ECN/CNP、PFC pause、retransmit、completion error。**交换机：** queue occupancy、drops、ECN mark、PFC、CRC、link flap。**调度器：** placement、preemption、node condition、device plugin health。指标应保存标签基数上限，避免把每个 token id 直接打进高频 Prometheus。

### 32.9.3 Trace 与日志脱敏

通信 trace 可以包含请求 id、序列长度和租户信息。多租户环境要使用不可逆 request hash 或采样后的 bucket，避免把 prompt 或个人数据写进网络日志。第22章的供应链和隐私原则适用于 trace：保留必要字段、设置 TTL、限制访问、记录导出审计。

## 32.10 CPU-only 实验：把拓扑和故障变成可执行合同

`labs/ch32_gpu_topology_lab.py` 是一个标准库实现的协议模型，默认构造 2 节点 × 4 GPU 的逻辑集群。它提供：

- `rank_to_node`、`topology_matrix`：检查 rank 是否按节点连续、路径是 `intra_node` 还是 `inter_node`；
- `plan_alltoall`：将每个 rank 的 aggregate payload 平分给非 self peer，报告每条路径的字节、时间和 p50/p95；
- `plan_ring_allreduce`：显式生成 reduce-scatter + all-gather 的环边，统计跨节点字节和最慢边；
- `failure_injection`：注入 GPU、node、switch 级故障，分别计算同步训练和副本推理的影响；
- `simulate`：把上述结果序列化为可审计 JSON，保留 seed、placement、collective、failure 和 summary。

运行：

```bash
python3 labs/ch32_gpu_topology_lab.py \
  --nodes 2 --gpus-per-node 4 --payload-mib 64 --requests 128 \
  --replicas 2 --output reports/ch32-gpu-topology-default.json

python3 labs/ch32_gpu_topology_lab.py \
  --nodes 2 --gpus-per-node 4 --payload-mib 64 --requests 128 \
  --replicas 2 --failure-scope node --failure-target 0 \
  --output reports/ch32-gpu-topology-node-failure.json

python3 tests/test_ch32_gpu_topology_lab.py
```

实验中的 payload 是每个 rank 的逻辑 aggregate bytes；`intra_bw_gbps=900` 和 `inter_bw_gbps=200` 只是可修改的数量级参数。实验不会启动 CUDA、不会打开 RDMA、不会调用 NCCL，也不会模拟交换机 buffer、GPU kernel、DMA registration、PFC/ECN 或真实故障恢复。它的价值在于把“谁给谁发多少、经过哪类路径、一个故障影响谁”写成可测试的合同。

### 32.10.1 实验问题一：rank 顺序改变会发生什么

把 `gpus_per_node=4` 改成 2，或在代码中交换 rank 到 node 的映射，比较 ring 的 `inter_node_bytes` 和 `bottleneck_ms`。如果 ring 顺序把节点交错，跨节点边增多；如果按节点连续，跨节点边减少。这个结论是拓扑逻辑的测量，不是对 NCCL 最终算法的保证。生产验证要用 `NCCL_DEBUG=INFO` 和 `all_reduce_perf` 对照。

### 32.10.2 实验问题二：GPU 故障和节点故障为何不同

注入 `--failure-scope gpu --failure-target 0`，实验显示同步训练不能继续，但两个副本节点仍可服务；注入 `--failure-scope node --failure-target 0`，一个副本节点消失，但另一节点仍可处理请求；注入 `switch`，所有节点都不可达，副本也无法服务。这里的 replica 数按节点计数，真实 serving 还要考虑副本是否共享模型并行组、NIC、ToR 和电源。

### 32.10.3 实验问题三：all-to-all 的总字节守恒

测试断言每个 rank 的发送字节等于 payload，所有 link bytes 等于 `world_size * payload`（不含 self）。这是协议层守恒。真实 all-to-all 可能有 padding、header、压缩、重传、梯度和双向阶段，因此不能把 toy 的 bytes 直接换算成账单或带宽。

## 32.11 生产验收与故障演练清单

### 32.11.1 上线前拓扑证据

- 保存 `nvidia-smi topo -m`、GPU UUID、PCI BDF、NUMA、NIC、交换机端口和机架标签；
- 固定驱动、CUDA、NCCL/RCCL、OFED、固件和容器 digest；
- 用 nccl-tests 运行单节点、跨节点、不同消息大小和并发作业，保存 p50/p95/p99；
- 验证 GPUDirect RDMA 与 host-staging 的差异，并记录实际 transport；
- 验证 PFC/ECN、MTU、ECMP、端口错误和队列指标；
- 用目标框架打印 DP/TP/PP/EP process group 和 rank mapping；
- 运行固定 golden batch，保存 loss、logits、吞吐、显存峰值和通信 histogram。

### 32.11.2 故障注入矩阵

| 故障 | 训练默认结果 | 副本推理默认结果 | 必须保留的证据 |
| --- | --- | --- | --- |
| 单 GPU/Xid | 当前同步 job 中止，恢复到最近 checkpoint | 只摘除受影响 worker | Xid、NVML、rank、communicator、checkpoint |
| 单 NIC | 跨节点 collective 超时或降级 | 单卡副本可能继续，TP/PP 组通常摘除 | NIC port、RDMA、ECN/PFC、NCCL transport |
| 单节点 | 所有本节点 rank 消失，需重建 world | 其他故障域副本继续 | BMC、dmesg、调度器、placement、切换时间 |
| ToR/交换机 | 多 job 受影响，禁止盲目重试 | 同一 ToR 副本全部不可用 | 交换机队列、端口、路由、受影响租户 |
| checkpoint 损坏 | 停止恢复，转入人工/备份路径 | 只影响加载该版本的副本 | checksum、manifest、版本和权限审计 |

演练要测的不只是“能否重启”，还要测故障发现时间、停止错误写入时间、清理僵尸进程时间、恢复首 token/首 step 时间、质量回归、重复请求和租户隔离。一次只注入一个故障，明确停止条件；否则无法知道修复是否有效。

### 32.11.3 变更和回滚

拓扑变更（换 NIC、交换机、GPU 固件、驱动、容器、调度器插件）应有 canary 节点和固定通信基准。若 p99 step 或首 token 超过预算，先回滚 placement/版本，而不是同时调整模型 batch、NCCL 参数和网络 QoS。所有调参都必须写入 manifest，包含生效时间、责任人、影响 job 和回滚命令。

## 32.12 常见错误与纠正

1. **把 `nvidia-smi -L` 当作拓扑。** 它只列设备，不列完整路径。补充 `nvidia-smi topo -m`、PCIe/NUMA/NIC 和交换机证据。
2. **只看平均带宽。** collective 由最慢 rank 或链路决定。改报 max/p95/p99、热点和队列指标。
3. **假设 global rank 连续就局部。** 读取框架实际 process group 和 scheduler mapping，写入 checkpoint。
4. **把 NCCL timeout 调大当作修复。** 先定位故障、清理 communicator、隔离坏节点，再决定重试。
5. **副本都在同一机架。** 重新按 node/ToR/rack/zone 分层放置，并演练对应故障。
6. **认为 device plugin 自动完成 NUMA/NIC 对齐。** 验证 Topology Manager、RDMA、GPU Operator 和容器权限的实际效果。
7. **把 CPU toy 的 latency 当成生产数字。** toy 只验证逻辑守恒和边界，生产要用目标硬件和真实负载测量。

## 32.13 资料与源码索引

以下链接是可核验的论文、官方文档或成熟开源实现。链接只支持所列主题，不替代目标版本的运行验证：

- NVIDIA [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)：collective、环境变量、拓扑和错误处理入口。
- NVIDIA [nccl-tests](https://github.com/NVIDIA/nccl-tests)：all-reduce、all-to-all 等基准源码。
- NVIDIA [NVLink 文档](https://docs.nvidia.com/networking/display/nvlink5)：NVLink 代际与互连资料。
- NVIDIA [nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)：拓扑、Xid、健康和计数器。
- NVIDIA [CUDA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)：GPU 显存与 RDMA 的接口约束。
- Linux [rdma-core](https://github.com/linux-rdma/rdma-core)：用户态 RDMA 栈源码。
- NVIDIA [DOCA RDMA Programming Guide](https://docs.nvidia.com/doca/sdk/rdma-programming-guide/index.html)：RDMA 编程与设备能力。
- PyTorch [Distributed overview](https://pytorch.org/docs/stable/distributed.html)：process group、backend 和 collective API。
- PyTorch [Elastic run](https://pytorch.org/docs/stable/elastic/run.html)：弹性训练启动和故障恢复边界。
- PyTorch [Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)：分布式 checkpoint API。
- MPI Forum [MPI 4.1 文档](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf)：集体通信语义的标准背景。
- Kubernetes [Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)：设备分配接口。
- Kubernetes [Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/)：NUMA 感知调度。
- NVIDIA [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)：Kubernetes GPU 驱动、device plugin 和监控组件。
- Slurm [GRES](https://slurm.schedmd.com/gres.html)：GPU 资源分配与 `CUDA_VISIBLE_DEVICES`。
- Slurm [cons_tres](https://slurm.schedmd.com/cons_tres.html)：可消费资源调度。
- Slurm [Topology](https://slurm.schedmd.com/topology.html)：调度拓扑插件。
- NVIDIA [DGX SuperPOD 网络参考](https://docs.nvidia.com/dgx-superpod/reference-architecture/latest/)：多节点 GPU 集群参考架构。
- NVIDIA [InfiniBand Fabric Manager](https://docs.nvidia.com/networking/display/ibfabricmanager) ：IB fabric 管理入口。
- OpenFabrics [企业 RDMA 指南](https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17)：RDMA verbs、队列和错误语义。
- NCCL [GitHub 源码](https://github.com/NVIDIA/nccl)：拓扑发现、transport 和 collective 实现。
- UCX [GitHub 源码](https://github.com/openucx/ucx)：高性能传输层和 RDMA 支持。
- UCC [GitHub 源码](https://github.com/openucx/ucc)：统一 collective 通信库。
- NVIDIA [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)：MIG 分区、实例和隔离边界。
- NVIDIA [DCGM 文档](https://docs.nvidia.com/datacenter/dcgm/latest/)：GPU 监控、诊断和健康检查。
- Prometheus [Metric types](https://prometheus.io/docs/concepts/metric_types/)：指标类型和标签基数基础。
- OpenTelemetry [Tracing specification](https://opentelemetry.io/docs/specs/otel/trace/)：跨应用/通信 trace 关联。

这些资料分别回答“能力是什么”“接口在哪里”“源码如何实现”；只有把它们与目标集群的 placement、版本、指标和故障演练绑定，才能形成可运营的结论。

## 小结

GPU 集群不是一组可互换的加速器，而是由 GPU、互连、PCIe/NUMA、NIC、交换机和故障域组成的分布式系统。训练的同步边界通常比推理副本更脆弱：一个 rank 的异常即可阻断 collective；推理可以通过跨故障域副本降低影响，但 TP/PP/EP 组仍然要整体摘除。拓扑感知的 rank mapping、分层 collective、NIC/交换机可观测性和明确的 checkpoint/retry 合同，是把“偶发慢”和“集群挂住”变成可定位事件的基础。CPU-only lab 只负责验证逻辑；生产结论必须由目标硬件、真实通信库、版本锁定和故障演练共同证明。
