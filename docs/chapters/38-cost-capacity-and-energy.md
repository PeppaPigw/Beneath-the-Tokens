---
id: ch38-cost-capacity-and-energy
title: AI Infra 成本、容量与能源：从 GPU 利用率到 TCO、碳与弹性
slug: /chapters/38-cost-capacity-and-energy
description: 以容量规划、排队、TCO、弹性伸缩、功耗热设计、碳核算和 FinOps 把 GPU 资源转成可验证的 SLO 与预算决策
sidebar_position: 38
level: advanced
prerequisites:
  - ch04-performance-math
  - ch09-distributed-training
  - ch10-training-ops
  - ch16-model-serving-system
  - ch17-kubernetes-gpu-orchestration
  - ch21-ai-reliability-engineering
  - ch35-serving-scheduling-and-slo
  - ch36-observability-and-tracing
learning_objectives:
  - 能把令牌、样本、请求、GPU 小时、kWh、kgCO2e 和货币统一成可审计的容量与成本单位
  - 能识别平均 GPU utilization、显存占用和节点功率读数的误区，并用队列、尾延迟和有效吞吐补全证据
  - 能比较 on-demand、reserved/committed、spot/preemptible 与混合池在风险、机会成本和恢复时间上的差异
  - 能为训练、离线批推理、在线推理和数据处理建立容量计划、排队模型、弹性伸缩和 SLO 成本权衡
  - 能计算 IT 功率、PUE、冷却/网络/存储开销、区域碳强度和时间移峰对 TCO 与碳的影响
  - 能设计 FinOps 标签、预算闸门、成本异常检测、showback/chargeback 和因果回溯流程
  - 能运行 CPU-only toy lab，复现实例中的过载、spot 中断、热/功率上限、碳强度和 SLO 失败
  - 能区分 toy 模拟的教学证据、供应商计费数据与生产级容量/碳审计边界
estimated_hours: 48
hardware: CPU-only toy lab; GPU 功率、NVML/DCGM、云价目表和机房 PUE/碳强度必须在固定版本环境中复验
risk_level: L2
last_verified: 2026-10-07
---

# 第38章　AI Infra 成本、容量与能源：从 GPU 利用率到 TCO、碳与弹性

> 一张“GPU 利用率 72%、本月账单 10 万美元”的截图不足以回答容量是否健康。利用率可能是一个采样窗口里的 SM busy，账单可能漏掉存储、网络、控制面和失败重试，功耗可能是芯片瞬时读数而不是机架和冷却的总量。成本与能源工程要把需求、排队、资源、合同、故障、功率、热、碳和 SLO 串成同一条可审计的证据链。本章不把“把 GPU 填满”当成目标，而是寻找在可用性、尾延迟、交付时间、预算和碳约束下的可解释工作点。

本章的 CPU-only toy lab 使用固定 fixture 和纯 Python 标准库，计算容量、排队近似、租赁组合、弹性动作、功率/碳和 TCO。它不会连接云厂商、读取真实 NVML/DCGM、改变 Kubernetes、购买实例或证明任何生产数字。生产落地必须把 toy 中的抽象字段替换为有版本、有账单发票、有计量校准和有 owner 的真实证据。

## 38.1 问题/边界：为什么“更高利用率”不是完整答案

### 38.1.1 资源账单是一张多维量表

AI Infra 至少有八种彼此可换算但不能混淆的单位：

1. **需求单位**：训练样本/令牌、推理请求、输入/输出 token、并发会话、embedding 文档或数据扫描字节；
2. **服务单位**：有效 tokens/s、请求/s、batch/s、完成的训练 step、成功任务数，而不是所有尝试过的 step；
3. **资源单位**：GPU 秒、GPU 小时、CPU 核时、内存 GiB·小时、临时盘 GiB·小时、对象存储和出口 GiB；
4. **时间单位**：wall-clock、排队等待、执行时间、重试时间、恢复时间和保留实例的闲置时间；
5. **货币单位**：按实例、按 GPU、按 token、按承诺折扣、按流量和附加服务费计价的美元或本地货币；
6. **能量单位**：IT 设备 kWh、机架输入 kWh、冷却和配电损耗、峰值 kW 与持续平均 kW；
7. **碳单位**：kgCO2e，依赖区域、时段、市场/位置法、可再生能源合同和排放因子版本；
8. **可靠性单位**：SLO 达成率、deadline miss、重试率、spot interruption、MTTR 和容量错误预算。

例如，1M 输入 token 可能只需 0.02 GPU·小时，却因为队列等待 30 分钟导致用户超时；一项训练跑完 1000 GPU·小时，若其中 25% 是故障重试，单位有效 step 成本就比账单显示的 GPU·小时高三分之一。容量计划必须先锁定“产出是什么”，再把资源映射过去。

### 38.1.2 误区一：把平均 GPU utilization 当成吞吐

SM busy、tensor core active、显存带宽和 PCIe/NVLink 传输是不同计数器。一个 kernel 可能让 SM 看起来很忙，却被内存、通信或 CPU 输入阻塞；反过来，短 kernel 间的空洞在 1 秒采样中被抹平。平均利用率还会掩盖：

- prefill 和 decode 的工作形状不同，decode 受 KV cache 和同步影响；
- 多租户时一张卡的平均值很高，但某个租户的 p99 latency 已超 SLO；
- batch 合并提高平均吞吐，却增加排队和尾延迟；
- 训练 checkpoint、数据加载、梯度同步让 GPU idle，瓶颈在存储/网络而不在 GPU 数量；
- spot 抢占后重试使“忙”增加，实际有效产出下降。

因此至少同时记录有效吞吐、排队时间、执行时间、p50/p95/p99、显存水位、通信等待、失败重试和每个租户的服务量。NVIDIA [DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/) 与 OpenTelemetry [metrics 规范](https://opentelemetry.io/docs/specs/otel/metrics/)可作为采集语义的起点，但版本、采样粒度和标签基数仍需锁定。

### 38.1.3 误区二：按峰值乘平均值做容量

常见公式是“峰值 QPS × 单请求 GPU 用量 × 目标利用率倒数”。它忽略了 burst、相关性、冷启动、批量窗口、故障域和共享缓存。对于在线服务，应把到达率按时间片建模，把服务率按模型版本、序列长度和 batch 策略分层；对训练，应把 checkpoint、扩容、数据重放、验证和恢复纳入 wall-clock。一个 10 分钟尖峰不能用一天平均 QPS 抹平，也不能简单用最大值乘 24 小时。

容量计划的边界包括：

- 工作负载边界：哪些队列、租户、模型、地区和优先级计入；
- 资源边界：GPU 型号/显存、CPU、内存、存储、网络、控制面和许可证；
- 时间边界：日内、周内、发布窗口、季度增长和生命周期；
- 故障边界：节点故障、区域故障、spot 中断、冷却限功率和供应延迟；
- 经济边界：承诺折扣、按需价格、spot 价格/中断概率、迁移成本和机会成本；
- 环境边界：PUE、碳强度、电力合同、热回收和水资源指标。

把这些写进容量假设表，避免“看起来保守”却偷偷把风险转移给用户或机房。

### 38.1.4 误区三：把账单总额等同于 GPU 成本

GPU SKU 的小时费只是 direct compute。TCO 还包括 CPU/内存配比、系统盘和缓存、对象存储、跨区/出口、日志和 tracing、容器 registry、控制面、许可证、SRE 值班、冷却/机架、电力、预留闲置、失败重试以及工程折旧。训练要把成功 step、失败 step 和恢复时间分开；推理要把峰值冗余、灰度版本、shadow 流量和低峰闲置分开。FinOps 的任务不是把所有成本平均分摊，而是让产品、平台和基础设施 owner 对可控驱动因素负责。

### 38.1.5 能源与碳的边界

GPU 的功率遥测通常是设备侧读数，不能直接代表电表；机架还包括 CPU、内存、网络、风扇、存储、UPS 和制冷。PUE（Power Usage Effectiveness）是总设施能耗/IT 能耗，不能被当作一个固定全球常数。碳强度可能随小时和地区变化，位置法与市场法结果也不同。结论应记录：计量点、时间窗、PUE/碳因子版本、是否含 embodied carbon、是否有绿电凭证以及不确定区间。

本章 toy 只计算 operational electricity 与给定 PUE、location-based carbon factor 的 kgCO2e，不宣称生命周期碳、用水、硬件制造或抵消项目已核算。可参阅 [ISO 50001](https://www.iso.org/iso-50001-energy-management.html)、[GHG Protocol Scope 2](https://ghgprotocol.org/scope-2-guidance) 和 [SCI for AI](https://greensoftware.foundation/articles/software-carbon-intensity-for-ai) 的定义边界。

## 38.2 心智模型：需求、排队、资源、经济与环境的闭环

### 38.2.1 五层容量心智模型

把容量问题画成五层，任何一层的假设都必须可追溯：

1. **需求层（Demand）**：每个时间桶的请求、token、样本、deadline、优先级和区域；
2. **服务层（Service）**：模型/数据 pipeline 在给定 batch、序列长度、并行策略下的服务率与失败率；
3. **排队层（Queue）**：到达率 λ、服务率 μ、并发上限、调度权重、抢占、backlog 与尾延迟；
4. **资源层（Fleet）**：GPU/CPU/存储/网络池、机型、故障域、on-demand/reserved/spot 组合、预热与冷启动；
5. **约束层（Budget/SLO/Carbon）**：成本上限、deadline、可用性、p99、功率/热上限、碳预算与合规。

反馈环是：观测到达与服务→更新预测→计划容量→执行弹性→观察 SLO/成本/功率→修正模型。不要直接从账单反推需求，也不要只根据 GPU utilization 反推服务率。

### 38.2.2 三种有效率

本章用三个分数避免“一个 utilization 统治一切”：

- **设备忙碌率 U_device**：设备时间里执行 kernel 或通信的比例；回答“硬件是否有工作”；
- **服务有效率 U_service**：成功交付的 token/step/请求与理论可服务量之比；扣除排队超时、失败重试和空转；
- **经济效率 E_cost**：有效产出/总 TCO，可表示为 $/1M output tokens、$/successful training step 或 $/SLO-compliant request。

当 U_device 上升而 U_service 下降时，通常是过载、争用、重试或工作无效。把三者放在同一 dashboard，并为每个变化提供因果链接（版本、队列、机型、spot、温度、限频）。

### 38.2.3 排队的直觉与稳定性

最简模型是 M/M/1：ρ=λ/μ 必须小于 1；当 ρ 接近 1，平均等待时间会非线性增加。真实 AI 服务不是指数分布的单服务器：有 batch 窗口、优先级、分片、KV cache、不同序列长度和多 GPU 并行。但这个近似提醒我们，80% 平均利用率不是永远安全，尾延迟需要 headroom。容量计划可用“目标 ρ + 尾延迟校准 + 故障冗余”三项，而不是一个魔法百分比。

对于训练队列，deadline miss 常由资源供给和长任务阻塞造成；对在线推理，p99 通常由 burst、长上下文、batch 形成和下游依赖共同决定。记录等待、排队前取消、执行、重试和降级的时间分解，比只记录端到端 p99 更能指导成本动作。

### 38.2.4 机会成本心智模型

reserved/committed 资源把未来需求换成价格折扣，但也锁定容量和现金；spot/preemptible 把中断风险换成低价；按需实例提供灵活性却可能在峰值供应不足。比较方案时加入：

- 预期有效小时 = 租赁小时 × (1−中断损失率)；
- 中断恢复时间、checkpoint 频率、重排/迁移工程成本；
- 保留资源低峰闲置的机会成本；
- 为 SLO 留出的跨区冗余与备用池；
- 供应延迟和释放条款；
- 预算现金流和承诺违约风险。

一个 spot 小时价格只有按需的 30%，若中断导致 40% 重试和 20 分钟恢复，单位成功 step 可能更贵。反之，容错训练、有细粒度 checkpoint 的低优先级批任务可能非常适合 spot。

### 38.2.5 能源闭环

功率 P(t) 是速率，能量 E=∫P(t)dt；在离散采样下 E≈Σ P_i·Δt。IT 能耗乘 PUE 才是设施能耗，再乘时空碳因子 g(t,region) 得到 kgCO2e。峰值 kW 影响配电和热设计，即使总 kWh 不变；削峰可能需要限频、错峰、迁移或批处理。碳优化不能损害 SLO：把任务移到低碳区域若增加跨区网络、延迟和失败重试，净碳可能上升。

## 38.3 机制：从单位经济学到弹性与 FinOps

### 38.3.1 需求分层与单位成本

先按 workload type 分层，再选成本分母：

| 类型 | 主要产出 | 关键分母 | 常见隐藏项 |
| --- | --- | --- | --- |
| 预训练/微调 | 有效 token、成功 step | $/1M token、$/step | checkpoint、失败重试、评测、数据读带宽 |
| 离线批推理 | 完成样本/文档 | $/1M token、$/千样本 | queue wait、shuffle、输出存储、重跑 |
| 在线推理 | SLO 内请求、输出 token | $/请求、$/1M token | 峰值冗余、低峰闲置、灰度、缓存失效 |
| 检索/embedding | 向量、索引更新 | $/百万向量 | CPU/存储/网络、重建索引 |
| 平台共享 | namespace/service | showback、$ / team | 控制面、监控、值班、未归属资源 |

分母必须是“成功且符合 SLO 的产出”。例如，$0.20/1M token 如果 p99 超过承诺，不能在 FinOps 报表里当成优异表现；应同时展示 SLO 合格率和单位成本。

### 38.3.2 容量计划的三种时间尺度

- **战略（季度/年度）**：模型路线、GPU 采购/承诺、区域与机房、PUE、预算、供应风险；使用情景树而非单点预测；
- **战术（周/月）**：版本发布、训练窗口、reserved 覆盖率、spot 池、网络/存储扩容、灰度配额；
- **运行时（秒/分钟）**：HPA/KEDA/队列长度、batch、限流、降级、迁移和功率上限；动作要幂等、有 cooldown 和回滚。

战略计划留 15–30% 的情景余量并标注证据；战术计划按项目 deadline 预留；运行时弹性不应偷偷扩大长期承诺。每次变更记录“容量假设→动作→结果→模型更新”。

### 38.3.3 保留、按需和 spot 的混合策略

将每个 workload 分到三类：

1. **基线容量**：稳定且高利用的在线服务或关键训练，使用 reserved/committed 或自有集群；覆盖保守 P50/P75 而非绝对峰值；
2. **弹性容量**：可延迟、可重试或有 checkpoint 的任务，使用 spot/preemptible；设置最大可接受中断、迁移和 checkpoint 成本；
3. **保险容量**：SLO/故障期间的 on-demand 或跨区备用；平时可空闲或承载低优先级任务。

覆盖率指标应同时看 committed utilization、spot interruption、按需峰值、未满足请求和释放率。不要为了提高 reserved utilization 把低价值工作硬塞进去，也不要把关键在线 SLO 依赖单一 spot 池。云厂商定价与中断行为版本化，官方入口如 [AWS EC2 Spot](https://aws.amazon.com/ec2/spot/)、[Google Cloud Spot VMs](https://cloud.google.com/compute/docs/instances/spot) 和 [Azure Spot VMs](https://learn.microsoft.com/azure/virtual-machines/spot-vms) 只能作为当前合同的来源之一。

### 38.3.4 弹性伸缩机制

弹性控制器需要输入：队列长度、等待时间、到达率预测、服务率、GPU/CPU 水位、显存、冷启动时间、spot 可用性、功率/碳信号和预算 guardrail。输出是副本数、机型/池、优先级、batch/限流和迁移动作。

一个可解释的控制环：

1. 计算未来控制窗口内的排队需求 `backlog + λ·horizon`；
2. 估计每副本有效服务率 `μ_eff = μ·(1−failure−interruption)`；
3. 计算目标副本 `ceil(demand/μ_eff + headroom)`；
4. 应用 min/max、cooldown、启动延迟和预算/功率上限；
5. 先扩容 warm/spot，再启用按需保险；缩容反向执行并保护 active request；
6. 记录决策输入、策略版本、预计成本和撤销条件。

不要让 queue length 与 GPU utilization 同时作为无权重“最大值”触发扩容；那会在 batch 形成时振荡。使用单一主信号加辅助证据，并对扩缩容设置 hysteresis。Kubernetes HPA 的资源指标只是机制起点，队列感知可结合 KEDA [ScaledObject](https://keda.sh/docs/latest/concepts/scaling-deployments/)；GPU 供给和拓扑仍需在调度层验证。

### 38.3.5 功率、热与限频

功率上限的动作顺序通常是：先移除无效工作/重试，再做 batch 和并发整形，随后在可接受性能损失内限频或调整功率目标，最后迁移到有余量的节点/区域。只看平均功率可能错过瞬态峰值导致机架 breaker 或热保护触发；只追求低功率又可能让 wall-clock 变长、总能耗增加。

热设计要把传感器位置、采样、滞后和冷却响应纳入模型。GPU 温度不是机房入口温度，风扇功率和 CPU/网络热源也会变化。记录 throttle reason、时钟、功率、温度、ECC/XID、任务阶段和 PUE 计量点。生产可用 NVML/DCGM，但必须验证驱动/固件版本和传感器校准；不要把 toy 的 `power_w` 当作电表数据。

### 38.3.6 训练 TCO 与推理 TCO

训练 TCO 公式可以从成功目标出发：

`TCO_train = (compute + storage + network + platform + labor + failure_recovery + commitment_opportunity) / successful_steps`

其中 `compute` 是实际实例费用而非理论 GPU 小时；`failure_recovery` 包括 checkpoint 写入、恢复和重算；`labor` 可用标准化 fully-loaded rate，但要注明假设。比较优化前后时固定数据、模型、质量目标和时间窗口。

推理 TCO 更适合按 SLO 合格请求或输出 token：

`TCO_infer = (fleet_idle + active_compute + cache/storage + ingress/egress + observability + support) / SLO_compliant_requests`

量化、批处理、KV cache 和 speculative decoding 可能降低单位 token 成本，却增加显存、尾延迟或质量风险。任何优化都要报告质量、p99、错误率、能耗和回滚条件。

### 38.3.7 SLO 成本权衡

为每个服务写一张 frontier：x 轴是 $/SLO 合格请求，y 轴是 p99/可用性/碳。候选点包括更多副本、更大 batch、更短上下文、量化、缓存、降级模型、spot 混合和错峰。选择 Pareto 前沿上的点，并把不可接受的约束先过滤：例如 p99>800ms 的点即使便宜也不能进入高优先级 API。

错误预算允许在不破坏用户承诺的范围内用低成本策略承载流量，但不能把错误预算当成无限财务预算。一次 spot 中断若消耗了可用性预算，应暂时提高保险容量；当预算恢复，再回到成本较低的配置。SLO、成本和碳应在同一变更评审中出现，避免平台只优化某一个数字。

### 38.3.8 FinOps 数据契约

每笔资源使用事件至少带：`service`、`team`、`model`、`workload_type`、`environment`、`region`、`gpu_sku`、`capacity_class`（reserved/on-demand/spot）、`request_or_job_id`、`unit_output`、`slo_status`、`energy_estimate_version`。标签缺失要进入 unallocated 队列而不是静默摊平。高基数 request ID 不应直接写入账单标签，可在仓内使用映射表。

FinOps 看板分四层：

- **预算层**：计划、承诺使用、预测区间、异常和审批；
- **归因层**：团队/模型/环境/区域/容量类型的 direct 与 shared；
- **单位经济层**：$/1M token、$/成功 step、$/SLO 请求、kgCO2e/同一分母；
- **动作层**：释放闲置、调参、改 batch、迁移区域、修复标签、更新承诺和验证结果。

[FinOps Foundation](https://www.finops.org/framework/) 的“inform、optimize、operate”可作流程骨架；不要把其通用框架误读成某云厂商账单字段的保证。

## 38.4 实验：CPU-only 成本、容量与能源 toy

### 38.4.1 实验目标和边界

实验脚本 `labs/ch38_cost_energy_lab.py` 固定输入：需求 token、每副本服务率、GPU 小时单价、reserved 折扣、spot 中断率、每卡功率、PUE、区域碳因子、SLO、冷启动与 headroom。它输出基线和可注入故障的 JSON，演示：

- ρ 接近 1 时排队时间和容量建议的非线性；
- 设备忙碌率高但有效服务率因过载/重试下降的利用率陷阱；
- reserved/spot/on-demand 的 expected cost 与中断恢复影响；
- 弹性伸缩在 min/max、cooldown、预算和功率上限下的动作；
- IT kWh、PUE、kgCO2e 与碳强度移峰；
- 训练与推理 TCO、SLO 成本是否达标；
- underprovisioned、spot_interruption、thermal_limit、carbon_miss、slo_violation 的 fail-closed 结果。

脚本不会执行 sleep、网络请求、云 API 或 GPU 调用；相同参数必须得到字节级稳定的 JSON（排序键、固定小数位）。`power_w` 是合成 IT 功率，`carbon_factor_kg_per_kwh` 是输入假设，不代表电表或官方实时碳强度。

### 38.4.2 可重复命令

```bash
python3 -m py_compile labs/ch38_cost_energy_lab.py
python3 labs/ch38_cost_energy_lab.py --fault none --output reports/ch38-cost-energy-baseline.json
python3 labs/ch38_cost_energy_lab.py --fault spot_interruption --output reports/ch38-cost-energy-spot.json
python3 labs/ch38_cost_energy_lab.py --fault thermal_limit --output reports/ch38-cost-energy-thermal.json
python3 tests/test_ch38_cost_energy_lab.py
```

`simulate()` 返回 `schema_version=1`、`decision`、`capacity`、`queue`、`procurement`、`energy`、`tco`、`slo`、`finops` 和 `invariants`。故障结果不是生产告警级别，只是检查策略是否把明显违反约束的输入拒绝或升级。

### 38.4.3 预期观察

基线应在 headroom、SLO、功率和碳预算内 `admit`；underprovisioned 应报告 `capacity_shortfall` 与较高 p95 wait；spot_interruption 应将成功产出折损计入成本并建议 on-demand insurance；thermal_limit 应拒绝超过节点功率上限的扩容动作；carbon_miss 应建议错峰或低碳区域但保留 SLO 约束；slo_violation 应优先增加容量/降级，而不是为了成本继续压缩。

实验结果中同时比较 `device_utilization` 与 `effective_utilization`。如果把失败重试或排队时间算进“忙碌”，设备利用率会看似变好，`effective_tokens` 却下降。这个反例应成为 dashboard 的单元测试，而不是运营人员凭经验解释。

## 38.5 故障诊所/失败：为什么省钱动作会让系统更贵

### 38.5.1 诊所一：把 utilization 目标设成 95%

**症状**：平均 GPU utilization 95%，账单/GPU·小时下降，但 p99 翻倍、超时增加，团队申请更多 GPU。

**机制**：ρ 接近 1，batch 窗口和长序列形成排队；故障或发布时没有 headroom。看板的 utilization 把等待和重试隐藏了。

**修复**：先分解 queue wait、service、retry；设定每优先级的目标 ρ 和 p99；用 warm capacity、限流和长度分层；只把可抢占的低优先级任务填入余量。验证单位应是 `$ / SLO-compliant request` 与 deadline miss，而不是单一忙碌率。

### 38.5.2 诊所二：spot 价格低，训练却超预算

**症状**：spot 单价是按需的 35%，但项目超出 deadline，最终花费高于按需基线。

**机制**：checkpoint 粒度过粗；中断恢复需要重新拉取数据和编译 kernel；多次重试还占用 reserved 基线，机会成本未计入。

**修复**：为可重放阶段缩短 checkpoint 间隔，保存数据/optimizer 状态，准备按需保险池；把 interruption rate、recovery wall-clock、失败 step 和成功 step 写进成本分母。若 deadline 硬约束，采用 spot+on-demand 混合而非全 spot。

### 38.5.3 诊所三：功率上限降低，碳排反而增加

**症状**：把 GPU power cap 从 400W 降到 250W，瞬时 kW 下降，但总 kWh 和 kgCO2e 上升。

**机制**：性能下降使 wall-clock 延长，静态 CPU/内存/冷却功率持续消耗；频率变化还让通信和同步等待比例上升。

**修复**：比较同一有效产出下的能量，而不是每秒功率；对阶段（prefill、decode、checkpoint、通信）做功率/时间剖析；只在碳强度高时段限频，或把可延迟批任务迁移到低碳窗口。保留温度和 throttle 证据，避免把热问题误判为能源优化。

### 38.5.4 诊所四：把 PUE=1.1 写进所有报表

**症状**：不同地区、季节和租赁机房都使用同一个 PUE，碳预算看起来精确到小数点。

**机制**：PUE 依赖设施、负载和季节；云租户未必能拿到同一计量边界。固定常数制造了虚假精度。

**修复**：保存 PUE 来源、计量边界和时间粒度；使用区间（例如 1.15–1.35）做敏感性分析；将云供应商披露与自有电表分开；报告不确定性，不用抵消额抵掉运营事实。

### 38.5.5 诊所五：弹性扩容触发成本风暴

**症状**：队列告警后扩容，冷启动期间旧副本仍在服务；新副本又触发更长排队，最终超过预算。

**机制**：控制器没有 hysteresis/cooldown，扩容信号同时使用 queue、GPU 和 utilization 的最大值；预算 guardrail 与 SLO 没有优先级。

**修复**：采用单主信号、预热队列和分阶段扩容；模拟启动延迟与副本有效服务率；设置每小时成本上限和保险额度；超限时选择限流/降级并记录 SLO 影响。每次动作写出预计成本、回滚条件和 owner。

### 38.5.6 诊所六：成本归因缺标签

**症状**：30% GPU 小时进入 `unallocated`，团队互相争论却无法优化。

**机制**：服务账号、批任务和临时 notebook 没有统一标签；shared platform 成本被平均摊平，驱动因素消失。

**修复**：在 admission/提交时强制 `team/service/model/workload_type/capacity_class`；缺失标签进入隔离队列或低优先级；shared 成本采用可解释规则并展示分配前后的差异。把标签完整率和单位成本一起纳入平台 SLO。

## 38.6 机制落地：SLO、预算、能源与变更评审

### 38.6.1 成本/SLO 变更模板

每次调度、模型、批量或电力策略变更都填：

- 目标产出和分母（成功 step、SLO 请求、token 等）；
- 基线版本、工作负载样本、时间窗和需求情景；
- 预期 GPU/CPU/存储/网络、reserved/spot/on-demand 比例；
- p50/p95/p99、错误率、deadline miss、恢复时间；
- IT 功率、PUE、碳因子、峰值 kW 和热/功率上限；
- 预算上限、异常回滚阈值和成本 owner；
- 观测指标、实验随机种子和停止条件；
- 不确定性、未计入项和生产验收计划。

模板的目的不是增加审批，而是让“省 20% 成本”能回答省了哪个分母、承担什么尾延迟和谁负责回滚。

### 38.6.2 账单与遥测对账

用三角对账发现漏项：

1. **账单**：供应商实例、存储、网络、服务费和承诺抵扣；
2. **资源计量**：节点/GPU 秒、作业、请求、输出 token、队列与失败；
3. **设施/能源**：电表或供应商能耗、PUE、功率峰值、碳因子。

按日或按小时比较，允许明确的时区、计费粒度、免费层和延迟。差异超过阈值时冻结“精确单位成本”发布，进入异常调查；不要在数据不完整时自动摊平。对账结果要保留原始快照、查询版本和人工解释。

### 38.6.3 预测与情景树

预测至少包含 base、high-growth、supply-constrained、carbon-constrained 四个情景。每个情景改变需求增长、服务率、spot 中断、GPU 价格、承诺折扣、PUE、碳因子和质量目标。输出容量、预算、deadline、kgCO2e 和风险，而不是一个“预计 GPU 数”。当实际偏离时，更新哪个假设，避免每月只调一个增长百分比。

### 38.6.4 生产验收门

toy lab 通过后，生产验收还需：

- 用固定版本云价目表、合同折扣和发票对账；
- 用真实 workload replay、GPU 型号、驱动、批策略、序列长度和故障注入校准服务率；
- 在节点、机架、区域故障与 spot 中断下测恢复、SLO 和成本；
- 以电表/设施报告核对 NVML/DCGM 的功率读数和 PUE；
- 采用版本化区域/时段碳因子并记录位置/市场法；
- 验证 FinOps 标签从提交到账单、showback、预算告警和回滚闭环；
- 对每个自动动作设置 dry-run、审批/阈值、幂等键和审计保留。


### 38.6.5 资源碎片、拓扑与不可合并容量

容量表里的“16 张 GPU”并不等于任何任务都能使用的 16 张 GPU。并行训练可能需要同一节点的 NVLink、相同显存、相同 MIG profile 或可达的 RDMA；在线服务可能需要每个副本至少一张完整卡；embedding 作业则可以使用碎片化的 CPU/GPU。把容量拆成 `raw`、`allocatable`、`schedulable`、`healthy` 和 `SLO-safe` 五层：

- `raw` 是物理设备数或租赁小时；
- `allocatable` 扣除系统、驱动、MIG 和平台守护进程；
- `schedulable` 扣除拓扑、亲和性、污点、许可证和配额；
- `healthy` 扣除 ECC/XID、温度、网络或存储异常节点；
- `SLO-safe` 再扣除故障域冗余、升级窗口和峰值 headroom。

资源碎片率可定义为 `1 - schedulable_gpu_hours / allocatable_gpu_hours`，但要按 workload class 分桶，否则小任务的空闲碎片会被大任务的不可用拓扑掩盖。调度器应在拒绝时记录“显存不足、拓扑不匹配、配额、故障域或预算”中的具体原因。把节点简单合并成一个池会掩盖这些结构性容量损失。

### 38.6.6 训练 checkpoint 与恢复经济学

checkpoint 间隔是成本、恢复时间和存储之间的三角权衡。设训练每小时计算成本为 `C_gpu`，中断率为 `p`，checkpoint/恢复开销为 `t_ckpt`，则过长间隔会增加期望重算，过短间隔会增加写入、网络和停顿。一个简化的期望重算时间可用 `p · interval / 2` 估计，再加恢复固定开销；该式只用于比较方案，真实中断可能与长任务年龄、区域和 spot 池相关。

实验时固定：模型状态大小、对象存储吞吐、校验 hash、压缩比、恢复并行度和可接受的丢失 step。报告“每成功 step 的 checkpoint 字节”和“checkpoint 引起的尾延迟”，不要只报告写入速度。对于混合池，checkpoint 还应记录实例类型、capacity class 和代码/数据版本，避免恢复到不兼容的 kernel 或 driver 后静默改变数值结果。

### 38.6.7 低碳调度的约束优先级

碳感知调度常见三个动作：时间移峰（把可延迟任务放到低碳时段）、地域迁移（选择低碳区域）、功率整形（限制并发或频率）。优先级应是：安全与数据驻留 > 关键 SLO/截止时间 > 预算与可靠性 > 碳优化。迁移前估算额外跨区传输、对象复制、冷启动、spot 风险和审计成本；如果这些使 deadline miss 或失败重试上升，净 kgCO2e 可能更差。

将碳信号视作带不确定性的输入。电网实时因子可能有延迟或预测误差，市场法还涉及合同凭证；在控制器中使用带上下界的 `carbon_intensity`，并给出“保持当前区域”“延迟”“迁移”三种动作的敏感性。不要把可再生能源证书直接抵消设备实际消耗而隐藏负载增长；运营排放、市场法声明和抵消项目分栏披露。

### 38.6.8 预算闸门和自动修复的安全性

预算自动化容易从“提醒”滑向“破坏性动作”。把动作分级：

- L0 只读：标注、报告、生成 dry-run 建议；
- L1 可逆：暂停低优先级队列、缩短 batch 窗口、释放空闲 spot；
- L2 受控：降低副本上限、迁移非关键任务、调整承诺购买建议；
- L3 高影响：删除资源、终止训练、改变数据保留或跨区复制，必须有显式审批和回滚证据。

每个自动动作带预算阈值、SLO 保护、幂等键、最短冷却时间和 owner。异常检测要区分新模型上线、流量增长、价格变化和标签丢失；只依据环比百分比会在低基数服务上产生噪声。对于共享集群，预算动作不得跨租户终止别人的高优先级作业；应先应用 namespace 配额和优先级策略。

### 38.6.9 成本模型的单位与舍入

货币和能源报告容易因舍入产生“账单对不上”。内部计算保留足够小数，最终展示明确币种、税费、折扣、时间区间和舍入规则。对 token、请求和 step 使用整数或固定精度，避免浮点累计误差；对 GPU 小时按供应商计费粒度向上取整时，记录原始运行时长和计费时长两列。对 reserved/committed 折扣，区分现金支付、摊销后单价和未使用承诺损失，不要把折扣当成无条件节省。

成本模型应能回答反事实问题：如果服务率提高 20%，需要多少更少的 GPU；如果 p99 目标从 800ms 改为 500ms，保险容量和碳有什么变化；如果 spot 中断率翻倍，checkpoint 和按需保险的临界点在哪里。把这些作为可运行情景，而不是在文档里只写一组静态数字。

### 38.6.10 质量、隐私与成本的交叉约束

量化、蒸馏、缓存和降级都可能改变质量或隐私风险。单位成本下降而数据脱敏、租户隔离或拒答策略变弱，不是成功的 FinOps 优化。为每个优化保留质量 canary、隐私/安全测试和数据驻留约束；将它们作为硬门而非“以后再测”。共享缓存命中率提升可能降低能耗，却有跨租户 key、prompt 或 embedding 泄漏面；缓存的容量节省必须与隔离成本一起核算。

在模型路由中，便宜的小模型承担多数请求可能降低 $/request，但复杂请求转发和重复推理会增加 token、网络和追踪成本。按请求类别建立质量/SLO/能耗分桶，避免平均值把少量高价值或高风险请求隐藏。成本优化的终点是可验证的服务价值，而不是最低的设备账单。

### 38.6.11 观测数据的保留与隐私

FinOps 需要 request/job 维度证据，但日志和 trace 可能包含 prompt、客户标识、对象 URL 或 token。使用不可逆的内部 workload ID、聚合窗口和最小必要标签；将原始事件与账单事实分开保留。保留期限与法规、合同和事件响应需求对齐，删除或降采样前确认不会破坏结算、SLO 争议或碳审计。任何把高基数标签写入共享账单的设计，都应先做成本、查询性能和隐私评审。

### 38.6.12 生产迁移的分阶段路线

从 toy 到生产不要一步跳到自动买实例：

1. **观测阶段**：只读采集需求、服务率、队列、账单、功率和标签完整率；
2. **影子阶段**：离线计算容量/成本/碳建议，与人工决策比较偏差；
3. **受限动作**：只对低优先级、可回滚 workload 应用弹性和 spot；
4. **服务级动作**：为一个 SLO 明确的模型启用扩缩容、预算和碳 guardrail；
5. **组合治理**：把多租户、跨区、承诺采购和设施能耗纳入季度容量评审。

每阶段都有停止条件：账单对账差异、标签缺失、SLO 回归、异常功率、碳因子不可得或审计证据不足时回到上一阶段。把自动化当作一个需要验收的产品，而不是脚本上线就算完成。

## 38.7 理解检查

1. 为什么 90% GPU utilization 可能比 70% 更昂贵？请分别从 ρ、重试、尾延迟和功率静态项解释。
2. 给定 λ=80 req/s、每副本 μ=50 req/s，两个副本是否足够？还需要哪些 headroom 和故障假设？
3. spot 单价下降 60%，中断率 15%，每次中断恢复 20 分钟。哪些 workload 仍可能适合 spot？
4. PUE 从 1.2 变为 1.4，IT 能耗 10,000 kWh、碳因子 0.35 kg/kWh，设施能耗与碳增加多少？
5. “每 1M output tokens 成本下降”但 p99 和拒答率上升时，如何判断是否仍在 Pareto 前沿？
6. 为什么把失败重试算入 GPU busy 会误导 FinOps？如何定义有效服务率？
7. 什么时候应增加 reserved 基线，什么时候应保留按需保险或 spot？
8. 如何证明 carbon-aware routing 没有把更多网络和失败成本转嫁给系统？
9. 成本标签缺失时，为什么平均摊销比 unallocated 队列更危险？
10. toy lab 的功率和碳数字为什么不能直接写进 ESG 报告？

## 38.8 练习

### 练习 A：队列与 headroom

构造 15 分钟时间桶，给出请求到达率、序列长度和副本服务率。比较目标 ρ=0.6、0.75、0.85 的副本数、p95 wait 和成本。再注入一个 10% 节点故障，说明哪种 headroom 在 deadline 下更稳。

### 练习 B：训练采购组合

有 256 GPU·小时基线需求和 20% 可能的额外峰值。给出 reserved 折扣、on-demand 单价、spot 单价、中断率和 checkpoint 恢复时间。计算全按需、全 spot、reserved+spot+保险三种方案的成功 step 成本和 deadline miss 风险，写出你会提交给 FinOps 的建议。

### 练习 C：功率与碳敏感性

固定 1000 个成功 step，比较三种配置的 GPU 功率、wall-clock、PUE 和区域碳因子。分别报告峰值 kW、IT kWh、设施 kWh 和 kgCO2e，并说明为什么最低功率配置未必最低碳。

### 练习 D：在线推理 Pareto 前沿

用不同 batch、量化和副本数跑固定请求 trace。记录 `$ / SLO request`、p99、错误率、输出 token/s、显存和 kgCO2e。删除被另一配置全面支配的点，写出高优先级 API 与离线批任务各自的选择。

### 练习 E：FinOps 数据契约

为训练、在线推理、共享平台和 notebook 设计统一标签，处理 shared 成本、跨区流量和 unallocated。定义标签完整率、预算偏差、成本异常、单位经济和 SLO 的告警阈值，并说明谁拥有修复责任。

## 38.9 来源

以下链接是本章写作和实验边界的主要一手资料，访问日期为 2026-10-07；云价格、碳因子和驱动指标会变化，应在生产验收时重新固定版本：

- NVIDIA DCGM 文档：https://docs.nvidia.com/datacenter/dcgm/latest/
- NVIDIA NVML API：https://docs.nvidia.com/deploy/nvml-api/
- Kubernetes HPA：https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- KEDA ScaledObject：https://keda.sh/docs/latest/concepts/scaling-deployments/
- AWS EC2 Spot：https://aws.amazon.com/ec2/spot/
- Google Cloud Spot VM：https://cloud.google.com/compute/docs/instances/spot
- Azure Spot VM：https://learn.microsoft.com/azure/virtual-machines/spot-vms
- FinOps Framework：https://www.finops.org/framework/
- FinOps FOCUS 1.0：https://focus.finops.org/focus-specification/v1-0/
- OpenTelemetry Metrics：https://opentelemetry.io/docs/specs/otel/metrics/
- OpenTelemetry Semantic Conventions：https://opentelemetry.io/docs/specs/semconv/
- Green Software Foundation SCI for AI：https://greensoftware.foundation/articles/software-carbon-intensity-for-ai
- Green Software Foundation SCI Specification：https://sci.greensoftware.foundation/
- GHG Protocol Scope 2 Guidance：https://ghgprotocol.org/scope-2-guidance
- ISO 50001 Energy Management：https://www.iso.org/iso-50001-energy-management.html
- The Green Grid PUE：https://www.thegreengrid.org/en/resources/library-and-tools/1-puetm
- MLPerf Training benchmarks：https://mlcommons.org/benchmarks/training/
- MLPerf Inference benchmarks：https://mlcommons.org/benchmarks/inference/
- Google TPU power/energy best practices：https://cloud.google.com/tpu/docs/performance-guide
- AWS Well-Architected Cost Optimization：https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/welcome.html
- Azure Well-Architected Cost Optimization：https://learn.microsoft.com/azure/well-architected/cost-optimization/
- Google Cloud Architecture Framework cost optimization：https://cloud.google.com/architecture/framework/cost-optimization
- queueing theory overview (NIST): https://www.nist.gov/itl/iad/queueing-theory
- Carbon Aware SDK：https://github.com/Green-Software-Foundation/carbon-aware-sdk
- Kepler energy exporter：https://github.com/sustainable-computing-io/kepler
- CarbonTracker：https://github.com/lfwa/carbontracker
- CodeCarbon：https://github.com/mlco2/codecarbon
- PaLM scaling law paper：https://arxiv.org/abs/2204.02311
- Efficient Large-Scale Language Model Training on GPU Clusters：https://arxiv.org/abs/2104.04473
- MLPerf Power measurement overview：https://github.com/mlcommons/power
- NVIDIA Data Center GPU Manager licensing and metrics notes：https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/index.html
- Open Compute Project datacenter efficiency：https://www.opencompute.org/wiki/Main_Page
- ISO 14064 greenhouse gas quantification：https://www.iso.org/standard/66453.html

本章的本地证据索引见 `evidence/ch38-cost-energy-manifest.json`，实验报告见 `reports/ch38-cost-energy-report.md`。Toy 结果只证明输入、公式、边界和测试可重复，不能替代供应商账单、设施电表、正式碳盘查、GPU 隔离或 SLO 生产演练。
