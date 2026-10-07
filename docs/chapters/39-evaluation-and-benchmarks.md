---
id: ch39-evaluation-and-benchmarks
title: AI Infra 评测与基准：从微基准、系统基准到端到端质量与回归门禁
slug: /chapters/39-evaluation-and-benchmarks
description: 建立可重复、可审计的 AI Infra benchmark，把性能、成本、能耗、质量和线上回归门禁放在同一证据链
sidebar_position: 39
level: advanced
prerequisites:
  - ch04-performance-math
  - ch06-pytorch-execution
  - ch12-evaluation
  - ch16-model-serving-system
  - ch20-observability-and-tracing
  - ch21-ai-reliability-engineering
  - ch35-serving-scheduling-and-slo
  - ch36-observability-and-tracing
learning_objectives:
  - 能从问题、假设、指标、样本、版本、硬件和停止规则设计可复现 benchmark
  - 能区分 micro、meso、macro 三层测量，以及吞吐、延迟、成本、能耗和质量指标的因果边界
  - 能使用分位数、置信区间、效应量和误差预算解释 benchmark 结果，而不是只报一个均值
  - 能发现数据集污染、缓存热身、负载混合、硬件漂移和版本变化造成的伪回归
  - 能把离线质量与在线系统指标联合成 shadow、canary、回滚和回归门禁策略
  - 能运行 CPU-only toy lab，理解矩阵、CI95、污染检查、联合门禁和失败注入的证据格式
  - 能为生产 benchmark 留下版本、硬件、命令、原始样本、报告、审计清单和 owner
estimated_hours: 52
hardware: CPU-only toy lab; 生产复验需要锁定加速器型号、驱动、运行时、数据和负载回放版本
risk_level: L2
last_verified: 2026-10-07
---

# 第39章　AI Infra 评测与基准：从微基准、系统基准到端到端质量与回归门禁

> benchmark 不是一张排行榜，而是一种受约束的测量协议。协议要回答：谁在什么硬件上，用哪个版本、哪组数据、什么负载，观察到哪一个指标，重复多少次，结论能推到哪里，不能推到哪里。本章将性能、成本、能耗、质量和可靠性放入一条证据链，并把离线报告连接到线上 shadow、canary 与自动回归门禁。所有示例都明确区分事实、toy 实验、推断和设计判断。

## 39.1 问题/边界：为什么“跑一个 benchmark”远远不够

### 39.1.1 排行榜是一种投影，不是系统

同一个模型在不同 token 长度、batch、并发、编译选项、量化、缓存状态、网络拓扑和温度下，可能得到完全不同的吞吐与尾延迟。若只公布“每秒 token”，读者无法知道分母是输入 token、输出 token，还是合并后的 token；也无法知道失败请求、重试、warm-up 与排队等待是否被排除。一个数字只是在多维系统空间上的投影。评测工程的第一要务是保存生成这个投影所需的坐标。

本章的边界如下：

1. **测什么**：模型推理、训练 step、数据处理、检索和平台控制面都可以 benchmark，但每个 case 必须写清产出单位和 SLO。
2. **不测什么**：toy lab 不下载模型，不调用云 API，不读取 GPU 计数器，不代表任何厂商价格、功率或模型智能。
3. **谁能使用结论**：只对协议覆盖的 workload、版本、硬件和统计不确定性负责；跨协议比较要先证明语义等价。
4. **何时停止**：达到预设样本量、置信区间宽度或最小可检测效应（MDE）就停止，不能因为结果不漂亮而追加样本直到显著。
5. **怎样改变生产**：离线 benchmark 只能产生候选版本；线上改变流量、写入状态、接受合同或消耗预算仍需单独的审批与回滚机制。

一个常见反例是：A 版本在 batch=32、输入长度=128 时比 B 快 20%，于是把 B 全量替换。生产流量包含 5% 的 8k 长上下文，B 的 p99 反而更稳定。实验并没有“测错”，而是把一个局部结论误当成全局因果。

### 39.1.2 单一指标的五种误导

- **均值误导**：少量长尾被平均吞掉；用户等待的是 p95/p99，而不是 mean。
- **吞吐误导**：提高 batch 可能提高 requests/s，却增加排队、内存和超时；吞吐还可能包含失败重试。
- **利用率误导**：GPU busy 说明设备在工作，不说明工作产生了合格答案；数据加载、通信、编译和重试都可能让 busy 上升。
- **成本误导**：只除以 GPU 小时会漏掉 CPU、网络、存储、日志、闲置、失败和人工运维；低价不等于低单位成功成本。
- **质量误导**：离线准确率上升可能来自数据污染、提示泄漏、评测集过拟合或拒答策略改变；质量提升必须与数据、版本和安全约束一起审计。

因此每个 benchmark case 至少要有“产出、系统、质量、成本、能耗、可靠性”六类字段。缺字段不一定阻止探索，但必须在报告里标注“不可判定”，禁止自动晋级。

### 39.1.3 评测对象的层级边界

我们把评测拆成三层：

| 层级 | 观察对象 | 典型问题 | 不能证明 |
| --- | --- | --- | --- |
| micro | kernel、算子、tokenizer、通信原语 | 一个算子在固定形状下的延迟和带宽是多少？ | 真实服务 p99、质量、故障恢复 |
| meso | 单副本、batch 调度、流水线、KV cache、数据 loader | 组合组件在给定并发和长度分布下怎样排队？ | 全地域可用性、产品转化、长期成本 |
| macro | 端到端 API、训练作业、检索链路、线上流量 | SLO、单位成本、能耗和质量是否同时达标？ | 未覆盖流量与未知故障域 |

层级之间不是“低层更真实、高层更重要”的排序。micro 为机制提供诊断，meso 揭示组合效应，macro 验证用户可感知结果。任何层级的结果都要带 workload 语义，否则跨层相加会制造假精度。

## 39.2 前置条件与学习结果

读者应熟悉 p50/p95/p99、平均值与方差、基本排队关系 ρ=λ/μ、GPU/CPU/网络/存储边界、训练/推理服务路径，以及第12章对离线质量集的讨论。建议先运行本章 lab：

```bash
python3 -m py_compile labs/ch39_evaluation_benchmark_lab.py
python3 labs/ch39_evaluation_benchmark_lab.py --fault none
python3 tests/test_ch39_evaluation_benchmark_lab.py
```

完成后，读者应能从一份结果反查命令、输入、版本、硬件、原始样本、汇总统计、门禁决策和失败原因；也应能指出 toy 结果不能替代什么生产证据。

## 39.3 心智模型：benchmark 是一条带版本的证据图

### 39.3.1 一句话心智模型

**把“请求/样本”沿着固定版本与硬件送过系统，记录原始事件，再用统计、质量和约束把事件压缩成可回滚的决策。**

事件不是孤立数字，而是图上的节点：数据集快照→请求生成器→调度与执行→遥测→质量判定→成本/能耗估算→报告→门禁。任何节点换版本，都应使图的哈希或 manifest 发生变化。

### 39.3.2 从请求到决策的控制流

1. **定义问题**：写清候选改动、基线、用户场景、目标指标、MDE、停止规则和不适用范围。
2. **固定输入**：冻结数据集 ID、抽样种子、提示模板、token 长度桶、请求间隔、并发曲线与超时。
3. **准备环境**：记录硬件 SKU、核心数、内存、驱动、运行时、编译器、容器 digest、频率/功率策略和温度范围。
4. **预热并校准**：预热不计入业务指标，但要记录时间和缓存状态；校准计时器、功率采样与时钟同步。
5. **采集原始事件**：每个请求包含开始、排队、首 token、结束、状态码、重试、输入/输出 token、质量标签和 trace ID。
6. **聚合与不确定性**：计算分位数、均值、CI、效应量和样本量；不要把 p95 的点估计写成确定真相。
7. **联合判定**：质量、安全、延迟、吞吐、成本和能耗共同过门禁；任何关键指标缺失都进入人工复核。
8. **发布与回滚**：离线通过后先 shadow，再小流量 canary；监控窗口满足后才扩大，否则按版本和 reason code 回滚。

### 39.3.3 可重复性四层

可重复性不只有“同一台机器再跑一次”：

- **计算重复**（repeatability）：同一环境、同一输入、同一 runner，结果能否重现。
- **环境重复**（reproducibility）：另一个环境按 manifest 安装同样版本，结论方向是否一致。
- **结果稳健**（robustness）：换合理的输入抽样、时间窗、负载混合，结论是否仍在误差带内。
- **外推有效**（generalizability）：生产流量、硬件和用户任务是否被 benchmark 覆盖，未覆盖部分如何标注。

每层都应有验收证据。只做到第一层不能声称“可复现生产性能”。

## 39.4 机制：三层 benchmark 如何拼成系统证据

### 39.4.1 Micro：测量机制而不是漂亮数字

micro case 需要固定 shape、数据布局、精度、线程数、编译开关和同步语义。对于异步 GPU，计时必须使用设备事件或显式同步；CPU 计时器包住提交调用只能测到 enqueue。矩阵乘法、attention、归一化、tokenizer 和通信原语应分别记录：

- 计算时间、内存读写量、估算带宽和算术强度；
- 启动开销、同步点、编译/缓存命中；
- 输入 shape、padding、精度和数值误差；
- 单次延迟分布与 batch/并发扩展曲线。

一个 kernel 快 5% 但数值误差导致后续拒答率上升，就不是“免费收益”。micro 报告要将性能和正确性并列，至少做边界 shape、空输入、极长输入和异常值测试。

### 39.4.2 Meso：组合效应通常在这里暴露

meso 把多个 micro 连接成可运行组件。例如单副本推理要包括 tokenizer、prefill、decode、KV cache、batch formation、队列、后处理和网络序列化。输入不是一个固定长度，而是长度与到达间隔的分布。报告要拆出：

`端到端延迟 = 排队 + 预处理 + 调度等待 + prefill + decode + 后处理 + 网络 + 重试`

不要把排队等待归因给 kernel。若 batch 窗口从 2ms 调到 10ms，设备利用率和吞吐上升，但 p99 可能超 SLO；门禁应按服务等级选择权衡，而不是选择看起来最大的柱子。

meso 的版本矩阵应包含 batch 策略、缓存命中/未命中、单租户/多租户、冷/热副本、不同优先级和故障重试。每组至少有稳定状态与扰动状态：突发到达、下游慢、节点重启、模型加载和网络抖动。

### 39.4.3 Macro：以用户可感知的合格产出为分母

macro 测量完整任务或 API。输入包括真实或脱敏流量回放、用户优先级、区域、依赖服务和发布策略。分母应是“成功并满足 SLO 的请求/样本/step”，不是尝试次数。建议展示三张图：

1. **质量曲线**：准确率、事实性、工具成功、拒答安全性、人评一致性；按场景与长度分层。
2. **系统曲线**：p50/p95/p99、吞吐、排队、错误/重试、可用性、冷启动和恢复。
3. **经济环境曲线**：$/1k 合格请求、J/请求、kgCO2e/任务、峰值功率、资源闲置。

三张图要共享同一 trace ID 或 workload ID，才能解释“质量提升是否换来尾延迟和成本回归”。

### 39.4.4 吞吐、延迟、成本与能耗的统一单位

令 `N_ok` 为在 SLO 内成功完成的请求数，`T_wall` 为从实验开始到结束的墙钟时间，`C_total` 为所有可归因货币成本，`E_total` 为 IT 与设施边界内的能量，`Q` 为质量合格率。常用派生量：

- 有效吞吐 `R_ok = N_ok / T_wall`；
- SLO 合格吞吐 `R_slo = N_ok · Q_slo / T_wall`；
- 单位成本 `$ / 1k_ok = C_total / N_ok · 1000`；
- 单位能耗 `J / ok = E_total · 3.6e6 / N_ok`（若 E_total 用 kWh）；
- 质量加权产出 `Y = N_ok · Q`，但只有在质量标注可靠且权重公开时才可用；
- 成本效率 `E_cost = Y / C_total`，不要与设备利用率混称 efficiency。

成本与能耗测量边界必须写在 manifest：是否含控制面、存储、网络出口、预热、失败重试、冷却/PUE、碳因子和 embodied carbon。若只有 GPU 设备功率，应命名为“设备侧估计”，不能写成“机房能耗”。

### 39.4.5 质量与系统联合评测

一个候选版本可能质量更高但 p99、成本或安全性更差。联合门禁把每个目标变成布尔证据：

```text
quality_ok = accuracy >= A_min and safety >= S_min
system_ok  = p95 <= L_max and cost_per_1k <= C_max
joint_ok   = quality_ok and system_ok
```

这不是把所有指标硬加成一个分数；加权总分会掩盖安全或 SLO 的致命失败。若产品确实有多目标权衡，应先定义 Pareto 前沿、不可违反的安全/合规硬约束，再把可优化目标交给决策者。权重、阈值和 reason code 都应版本化。

质量集要隔离训练、调参、开发和最终评测；提示、检索文档、工具响应和 system message 也属于输入。对生成任务，自动指标只是代理，必须配合人评抽样、事实性检查、拒答安全集和任务成功率。质量 CI 需要报告区间和标注者一致性，而非只报百分比到小数点后三位。

### 39.4.6 统计置信区间与最小可检测效应

设样本观测为 `x_1...x_n`，均值 `x̄`，样本标准差 `s`。在近似独立且分布不太偏的教学场景，均值的 95% CI 可写成 `x̄ ± 1.96·s/√n`；小样本或重尾分布应使用 t 分布、bootstrap 或分位数方法。p95 的 CI 通常比均值宽，不能从均值 CI 推断 p95 CI。

报告至少包含：样本量、中心趋势、分位数、CI 方法、随机种子/重采样次数、效应量和停止规则。若基线 p95=100ms、候选=103ms，但 CI 重叠且 MDE=5%，不能自动拒绝；若候选=115ms 即使均值差异显著，也要解释是否由某一长度桶或冷启动驱动。

重复测量存在相关性：同一请求重试、同一机器连续样本、同一用户会话都不是独立观测。应按请求、会话、主机或时间块聚合，避免伪造 n。多指标、多版本、多数据集会增加多重比较风险，门禁可以预先冻结主指标，其余作为诊断。

### 39.4.7 数据集污染与泄漏

污染不只是训练集与评测集有完全相同的字符串。还包括：

- 评测问题出现在预训练语料或调参日志；
- 检索索引包含答案文档的未来版本；
- prompt 模板暴露标签、工具返回或评分规则；
- 合成数据由同一个模型生成并被同一模型评判；
- 线上 shadow 回流到训练集后，下一轮离线评测看似提升。

最低协议是保存数据来源、快照日期、去重规范、近似匹配阈值、访问权限、训练截止时间和评测冻结时间。发现 overlap 后默认 deny，不能用“只占 1%”作为自动放行理由；需要重新切分、重新标注或将结果降级为探索性。

## 39.5 版本与硬件矩阵：让结果可以被追溯

### 39.5.1 Manifest 字段

一个可审计 manifest 至少包括：

```yaml
run_id: 2026-10-07T09-00Z-abc
commit: sha256:...
runner: benchmark-runner@1.4.2
model: model-repo@digest
dataset: holdout-v7@sha256:...
prompt: prompt-set@2026-09-12
hardware: H100-SXM-80GB x8
driver: 550.54.15
runtime: cuda-12.4 / pytorch-2.5.1
workload: p95_input_tokens=2048, concurrency=32
seed: 17
warmup: 20 requests
repetitions: 10 blocks
primary_metrics: [p99_ms, ok_throughput, quality, cost_per_1k]
```

字段值可以变化，但字段本身不应被悄悄删除。manifest 与原始事件、汇总报告、代码版本、硬件照片/计量器件 ID 之间要能互相链接。生产中把秘密、用户内容和凭据排除在 manifest 外，只记录不可逆 ID 或脱敏摘要。

### 39.5.2 矩阵与交互项

版本矩阵通常有模型版本×运行时版本×硬件型号×精度×并发×长度桶。完整笛卡尔积很昂贵，可以用正交设计或分层采样，但要先覆盖风险最大的交互项：例如某量化只在特定 GPU 架构上触发 kernel fallback；某编译器版本只在长上下文下增加 p99；某 driver 与 power cap 联合导致频率抖动。报告中标注未测组合，不能用相邻格子插值替代证据。

硬件不是一串 SKU：NUMA、PCIe/NVLink 拓扑、MIG、频率锁定、温度、风扇策略、容器 cgroup 和邻居干扰都会改变结果。至少做冷机、热机和重复时段的对照；发现漂移时把它当作信号，而不是在报告里取最好的那次。

### 39.5.3 可重复性检查清单

运行前：确认 git 工作树干净、镜像 digest、依赖 lockfile、数据快照、时区、NTP、CPU governor、GPU persistence、频率/功率策略和随机种子。运行中：记录温度、频率、错误计数、重启、OOM、网络重传、缓存命中和队列深度。运行后：保存原始样本、聚合脚本版本、命令 stdout/stderr、校验和、失败 run 及 reason code。

“同一个命令”不等于“同一个实验”：环境变量、云实例邻居、spot 迁移、数据服务热度都可能变。若无法锁定，给结果附上变异区间和适用条件。

## 39.6 实验：CPU-only toy lab

### 39.6.1 实验目标与边界

`labs/ch39_evaluation_benchmark_lab.py` 用标准库构造固定的 micro/meso/macro case。它生成确定性的延迟样本，计算 p50/p95/p99、均值 CI95、吞吐、每千请求成本和每请求焦耳；再执行污染检查、质量+系统联合门禁、回归阈值和 shadow/canary 策略。它没有真实模型、GPU、网络、云账单或用户数据，因此只能验证字段和决策逻辑。

### 39.6.2 基线运行

```bash
python3 labs/ch39_evaluation_benchmark_lab.py --fault none > /tmp/ch39-baseline.json
```

输出包含 `schema_version`、`versions`、`hardware.accelerator=none`、按层级的原始延迟样本、CI、吞吐、成本、能耗、质量、污染结果、联合门禁和 shadow/canary 结果。固定 fixture 意味着两次运行 JSON 完全相同；这只证明 toy 的计算确定性，不证明生产硬件确定性。

在结果中，`mean_ci95` 是均值的教学近似；不能把它当成 p95 的置信区间。`cost_per_1k_requests_usd` 把一个合成的 GPU 小时价格和服务时间相乘；没有包含平台、存储、网络或失败重试。`energy_joules_per_request` 使用合成的功率值；没有 PUE、碳因子和计量误差。

### 39.6.3 故障运行

```bash
for fault in regression contamination canary low_quality hardware_drift; do
  python3 labs/ch39_evaluation_benchmark_lab.py --fault "$fault" > "/tmp/ch39-$fault.json"
done
```

预期全部 deny，但原因不同：`regression` 与 `hardware_drift` 触发延迟/成本回归，`contamination` 触发数据 overlap，`low_quality` 触发准确率，`canary` 展示候选在小流量阶段仍受同一回归门禁约束。程序不会为了生成漂亮的基线而吞掉错误；未知 fault、非法 tier、空样本和错误阈值应抛出 `ValueError`。

### 39.6.4 直接测试与可重复记录

```bash
python3 -m py_compile labs/ch39_evaluation_benchmark_lab.py
python3 tests/test_ch39_evaluation_benchmark_lab.py
```

测试覆盖确定性、CPU-only、三层矩阵、CI 顺序、联合门禁、shadow 不修改状态、污染 deny、回归 deny、输入校验以及 CLI stdout 与输出文件一致。测试通过不是生产 benchmark 通过；它只保证本章 toy 的协议没有被后续改动破坏。

## 39.7 故障诊所/失败：看起来健康的 benchmark 为什么会坏

### 症状一：吞吐提升，用户却说变慢

**可能机制**：batch 窗口变长、长上下文比例下降、排队时间未计入，或者只统计成功请求。**诊断**：分解排队/执行/重试；按长度桶和租户绘制 p95/p99；检查 timeout 是否被当作“未采样”。**修复**：以 SLO 合格请求为分母，冻结请求混合，给 batch 提升设尾延迟预算。

### 症状二：质量提升但线上投诉增加

**可能机制**：评测集污染、评判器与候选共享错误、拒答率变化、工具链失败被质量脚本忽略。**诊断**：做近似去重、时间切分、人工盲评、安全集和工具成功率；检查是否只报告平均分。**修复**：污染默认 deny，增加独立 holdout 与对抗集，质量与系统联合门禁。

### 症状三：p99 在同一版本间漂移

**可能机制**：GPU 温度/频率、NUMA、邻居噪声、缓存冷热、云实例族差异或采样块相关性。**诊断**：按主机和时间块聚合，记录温度、频率、错误计数，重复冷/热机。**修复**：锁定硬件与功率策略，报告中位数和区间，不能只取最好的 run。

### 症状四：回归门禁反复红绿

**可能机制**：阈值太接近测量噪声，多重比较，基线被不断更新，或不同 workload mix 被误当同一 case。**诊断**：冻结基线版本、主指标和 MDE，查看 CI 与效应量；验证 baseline/candidate 的 workload ID 相同。**修复**：设最小样本和区间宽度，使用 reason code，必要时人工复核而非无限重跑。

### 症状五：成本低了，碳和可靠性变坏

**可能机制**：迁移到廉价但高碳区域，spot 中断重试，或者只计设备功率不计设施。**诊断**：加入 PUE、时空碳因子、恢复时间、有效产出和峰值功率；比较成功请求的 $/1k 与 kgCO2e。**修复**：把碳、成本和 SLO 设成明确的 Pareto 约束；记录测量边界，不把估算写成电表事实。

## 39.8 online shadow/canary 与回归门禁

### 39.8.1 Shadow 是观察，不是批准

shadow 将生产请求复制给候选，但不让候选写用户状态、发送外部副作用或改变主响应。应脱敏、限速、去除凭据和高风险工具；记录候选输出的延迟、质量代理、错误、成本估算和资源影响。shadow 结果仍可能有偏：复制流量改变缓存、网络和容量，不能直接等价于真实全量。

### 39.8.2 Canary 是受控暴露

canary 先给有界流量和明确用户/区域，定义观察窗口、最小样本、自动回滚指标和人工 owner。门禁至少包括：

- 质量：任务成功、事实性、安全违规、拒答率；
- 系统：p50/p95/p99、错误率、排队、重试、可用性；
- 资源：有效吞吐、GPU/CPU/内存、峰值功率；
- 经济：$/1k 合格请求、spot/按需比例、额外存储/网络；
- 变化：与基线同 workload ID、同数据分层和同版本 manifest。

回滚要使用不可变 artifact digest，而不是“重新构建一个相同标签”。reason code 写入事件流，便于后续把一次误报和一次真实回归区分开。

### 39.8.3 回归门禁的三态结果

推荐 `admit / deny / review` 三态：

- **admit**：硬约束通过，CI 足够窄，效应方向稳定；允许进入下一阶段。
- **deny**：安全、污染、SLO 或成本硬约束失败；自动阻止并回滚。
- **review**：CI 太宽、数据缺失、硬件漂移或指标冲突；保留证据交人工，而不是把不确定性当通过。

门禁配置、阈值和基线要与代码同版本审查。不能在失败后临时放宽阈值，也不能把候选作为新基线以消除红灯。紧急 bypass 必须有过期时间、审批人、风险说明和事后复盘。

## 39.9 取舍与被拒绝的替代方案

1. **只报平均吞吐**被拒绝：无法看到尾延迟和失败；保留平均值作诊断，同时把 p95/p99 与有效吞吐设为主指标。
2. **一个总分融合所有指标**被拒绝：会掩盖安全和 SLO 失败；使用硬约束+Pareto/多目标说明。
3. **无限增加样本直到显著**被拒绝：制造 p-hacking；提前冻结 MDE、停止规则和主指标。
4. **直接用生产流量训练评判器**被拒绝：污染与反馈回路不可审计；使用独立、时间切分的 holdout，并记录回流隔离。
5. **跨硬件直接比较 SKU**被拒绝：功率、频率、拓扑和软件栈不同；比较应锁定矩阵或报告归一化边界。
6. **只读 GPU 功率当作能耗**被拒绝：遗漏 CPU、网络、冷却和采样误差；命名为设备侧估算并提供设施边界。
7. **canary 只看错误率**被拒绝：质量下降或成本爆炸可能没有 5xx；使用联合门禁。

## 39.10 论文、官方文档与仓库综合

- MLCommons 的 [MLPerf Benchmarks](https://mlcommons.org/benchmarks/) 展示了固定场景、规则和可复核结果的重要性；本章借鉴其“规则先于数字”的原则，但 toy 不声称符合 MLPerf 提交规范。
- Google 的 [TPU performance guide](https://cloud.google.com/tpu/docs/performance-guide) 说明 shape、输入管线、编译和利用率语义如何影响加速器性能；这对应本章的 micro/meso 边界。
- NVIDIA [DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/) 和 [NVML API](https://docs.nvidia.com/deploy/nvml-api/) 是硬件遥测入口，但计数器含义、权限和采样版本必须锁定。
- Kubernetes [Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/) 与 KEDA [ScaledObject](https://keda.sh/docs/latest/concepts/scaling-deployments/) 可实现按指标/队列伸缩；它们不会自动定义质量门禁或统计停止规则。
- OpenTelemetry [metrics specification](https://opentelemetry.io/docs/specs/otel/metrics/) 与 [traces specification](https://opentelemetry.io/docs/specs/otel/trace/) 提供语义基础；标签基数、隐私和 cardinality 预算仍需平台设计。
- 数据污染与数据卡片可以参考 [Datasheets for Datasets](https://dl.acm.org/doi/10.1145/3458723) 和 [Model Cards](https://arxiv.org/abs/1810.03993)；它们强调来源、适用范围和已知限制，而非只给一个分数。
- 能耗与碳边界可结合 [GHG Protocol Scope 2](https://ghgprotocol.org/scope-2-guidance)、[SCI for AI](https://greensoftware.foundation/articles/software-carbon-intensity-for-ai) 和 [PUE](https://www.thegreengrid.org/en/resources/library-and-tools/1-puetm)。本章 toy 未实现完整生命周期碳。
- 生产实现可研究 [Carbon Aware SDK](https://github.com/Green-Software-Foundation/carbon-aware-sdk)、[Kepler](https://github.com/sustainable-computing-io/kepler)、[CarbonTracker](https://github.com/lfwa/carbontracker)、[CodeCarbon](https://github.com/mlco2/codecarbon) 和 [MLCommons Power](https://github.com/mlcommons/power)，但接入前要审计测量误差、权限、版本和部署边界。

这些来源支持机制和术语，不自动证明任何本地结果。来源 URL、访问日期、claim 和本地验证命令记录在 `evidence/ch39-evaluation-benchmark-manifest.json`。

## 39.11 理解检查（含答案）

1. **为什么平均 GPU 利用率高不能推出 SLO 合格？**
   - 答：利用率只描述设备忙碌，可能来自重试、通信或无效工作；必须结合有效吞吐、排队、尾延迟和质量。
2. **micro、meso、macro 各回答哪类问题？**
   - 答：micro 诊断算子/原语，meso 观察组件组合和排队，macro 验证端到端用户可感知产出；层级不可互相替代。
3. **为什么 p95 的 CI 不能直接用均值 CI？**
   - 答：分位数是分布位置统计，尾部方差和样本量影响方式不同；应使用分位数 bootstrap 或合适的区间方法。
4. **发现训练集与评测集 overlap 后，为什么默认 deny？**
   - 答：评测可能泄漏，分数失去独立性；必须重切分/重标注或降级结论，不能用小比例自动放行。
5. **shadow 和 canary 的关键差别是什么？**
   - 答：shadow 复制请求但不改变主响应/用户状态，主要观察；canary 有界暴露真实响应路径，需要自动回滚与用户影响控制。
6. **联合门禁为什么不用简单加权总分？**
   - 答：加权总分可能用质量或成本掩盖安全、污染或 SLO 硬失败；先执行硬约束，再在可行集合内做多目标权衡。

## 39.12 练习

### 练习 A：回忆与标注

在一个 benchmark manifest 中补齐硬件、驱动、运行时、数据快照、提示模板、warm-up、重复次数、主指标、MDE 和停止规则。标注哪些字段会导致跨环境不可比。

### 练习 B：推导

基线有 10,000 个请求，其中 9,600 个在 p99 SLO 内成功；总成本 4.8 美元、总 IT 能耗 0.9kWh。候选有 10,000 个请求，9,700 个成功，但成本 5.4 美元、能耗 0.82kWh。分别计算 `$ / 1k 合格请求` 与 `J / 合格请求`，再说明需要哪些质量指标才可决定。

### 练习 C：实现

扩展 toy lab：增加一个“缓存命中”字段和命中/未命中两条延迟分布；报告整体与分层 p95，并确保回归门禁不会因为命中率变化而把 workload mix 误当版本回归。

### 练习 D：诊断

某候选版本 mean latency 降低 8%，p99 增加 20%，质量 CI 与基线重叠，成本下降 3%。设计 `review` 而不是 `admit/deny` 的判定规则，并列出需要的分桶和硬件遥测。

### 练习 E：设计

为一个 RAG 服务设计 10% shadow、5% canary 和全量发布的门禁：规定脱敏、工具副作用、最小样本、窗口长度、质量/系统/成本阈值、回滚 artifact 和 bypass 过期时间。画出事件流和 owner。

## 39.13 小结与下一依赖

本章的核心不是记住某个 runner 或某个排行榜，而是建立一条可审计链：问题与边界→固定 workload→版本/硬件矩阵→micro/meso/macro→原始事件→统计区间→质量+系统联合→污染检查→shadow/canary→回归门禁。吞吐、延迟、成本、能耗和质量必须共享同一分母与 trace 语义；缺数据时进入 review 而不是猜测。toy lab 验证协议和失败闭环，生产部署还需真实计量、脱敏、权限、容量、隐私与发布治理。

下一章可把本章的门禁接入平台工程：将 manifest、事件、报告和审批整合到 CI/CD、模型 registry、feature store、服务编排和 incident response；也可在第38章的成本/能源模型上增加 benchmark 运行的碳预算与容量 forecast。

## 39.14 来源与可复现性记录

### 39.14.1 主要来源

完整来源索引见 `evidence/ch39-evaluation-benchmark-manifest.json`，包括 MLPerf、NVIDIA DCGM/NVML、Google TPU 性能指南、Kubernetes HPA、KEDA、OpenTelemetry、Datasheets/Model Cards、GHG Protocol、SCI、PUE、Carbon Aware SDK、Kepler、CarbonTracker、CodeCarbon 与 MLCommons Power。链接为真实官方文档、论文或 GitHub 仓库，访问日期为 2026-10-07；生产使用前应重新核验版本。

### 39.14.2 本地复现实验

```bash
cd Beneath-the-Tokens
python3 -m py_compile labs/ch39_evaluation_benchmark_lab.py
python3 labs/ch39_evaluation_benchmark_lab.py --fault none
python3 labs/ch39_evaluation_benchmark_lab.py --fault regression
python3 tests/test_ch39_evaluation_benchmark_lab.py
```

报告 `reports/ch39-evaluation-benchmark-report.md` 记录环境、命令、输出摘要、故障观察和生产边界；`reports/ch39-phase2-audit.json` 记录结构、URL、前置条件、脚本语法、CPU-only、确定性、CLI JSON 和测试契约。数字是合成 fixture，不能作为云账单、GPU 性能、模型质量、机房能耗或线上 SLO 认证。

## 39.15 进阶：把 benchmark 做成长期制度

### 39.15.1 先写测量合同，再写脚本

团队经常从脚本开始：读取一批 prompt、调用接口、打印均值。脚本能跑不等于实验有合同。建议在代码之前写一页 measurement contract，包含六个问题：

1. **决策是谁做的**：发布 owner、平台 owner、质量 owner、财务/能源 owner 各自批准什么；没有 owner 的指标只会变成装饰。
2. **最小证据是什么**：哪些字段缺失就只能 review；哪些硬约束失败立即 deny；哪些观察指标只用于诊断。
3. **分母和边界是什么**：是否包括预热、失败、重试、超时、取消、排队、冷启动、模型加载与后台任务。
4. **扰动怎样注入**：突发、长上下文、下游慢、节点丢失、spot 中断、功率上限和数据服务故障是否在协议内。
5. **停止和重跑规则是什么**：CI 宽度、MDE、最大 wall-clock、硬件漂移阈值，以及何种错误允许重跑；重跑不能只保留最好一次。
6. **结论可以推到哪里**：对哪一版模型、哪类用户、哪组硬件和哪一时间窗有效；所有外推都写成假设。

合同应作为代码评审的输入，脚本只实现合同，不在脚本中偷偷改变实验目的。若需求中出现“让数字看起来稳定”或“挑一组最快的 prompt”，应把它改写为可审计的问题，例如“估计在输入长度分布 X 下 p99 的变异区间”。

### 39.15.2 原始事件优先于聚合图表

只保存 dashboard 截图会丢失重算能力。每个事件至少保留：`run_id`、`case_id`、`request_id`、时间戳、版本/硬件 ID、输入输出 token、排队与执行时长、状态码、重试次数、质量标签、功率采样序号、异常和 trace ID。用户内容可以哈希或脱敏，但不能删除使分层失效的长度桶、场景标签和错误类型。

聚合脚本应是纯函数：输入原始事件与 manifest，输出汇总与 reason code。这样可以在发现 percentile 实现错误、过滤条件错误或污染后重新计算，而无需重放生产请求。报告中同时保存脚本 commit、命令、依赖 lockfile 和输入校验和。任何手工编辑的数字都要标成“人工注释”，不能伪装成原始测量。

### 39.15.3 抽样与代表性

真实流量通常长尾且有季节性。简单随机抽样会丢掉稀有但高风险的长上下文、工具调用和安全场景；只做分层抽样又可能夸大极端比例。实务上可以分两套集：

- **自然比例集**：按生产分布估计整体负载和单位经济；
- **风险覆盖集**：过采样长上下文、拒答、安全、工具失败和资源边界，用于发现回归。

两套集不能混为一个“总分”。自然比例集回答“通常用户怎样”，风险覆盖集回答“已知危险是否可控”。若要合并，应公开权重、置信区间和每层样本量。训练、调参、分析和最终报告要使用不同数据时间窗，线上回流要有冻结日和隔离区。

### 39.15.4 质量标注的测量误差

人评不是绝对真值。应记录标注指南版本、标注者数量、盲法、随机化顺序、冲突解决和一致性统计。自动 judge 需要做位置偏差、长度偏差、引用核验和对抗抽样；一个 judge 同时生成训练偏好和最终评测会产生循环。对生成任务，报告点估计之外的评测者/样本不确定性，避免把 0.2 个百分点写成真实提升。

安全质量要把“模型拒答”与“用户任务成功”分层：过度拒答可能提高安全集分数却降低业务完成率；过度迎合可能提高主观满意度却增加政策风险。联合门禁应保留这些冲突，由产品与安全 owner 选择可接受前沿，而不是让一个黑箱权重替他们决定。

### 39.15.5 成本与能耗的计量三角校验

成本、资源和能耗应相互校验但不能互相替代。可以做三角检查：

1. **资源账单**：实例秒、存储字节、网络出口、日志量和控制面标签，与供应商账单/内部 showback 对账；
2. **资源遥测**：GPU/CPU 时间、显存、功率、节点温度、网络和存储吞吐，与请求 trace 对齐；
3. **设施或估算**：机架电表、PUE、区域碳因子或经审计的估算，记录采样与误差。

若三者差异超过阈值，不要平均三份数字，而是标记计量缺口。例如 GPU 侧报告 350W、节点电表报告 700W，可能有 CPU、内存、网络或采样窗口差异；结论应写“设备侧 350W，节点侧 700W，未完成归因”，而不是挑较小的一个算碳。

### 39.15.6 线上门禁的时间窗与告警疲劳

短 canary 可能错过低频长尾；长窗口则让坏版本影响更多用户。窗口设计应依据请求量、日夜周期、长任务完成时间和最小检测效应，而非固定“15 分钟”。告警要去重并按 reason code 聚合：质量下降、硬件漂移、数据污染、成本异常和流量变化走不同 owner。若同一指标反复在阈值附近抖动，优先检查测量噪声、基线漂移与阈值设计，而不是把告警静音。

回滚后要保留候选的事件与证据，禁止删除失败 run。这样才能进行事后复盘：是候选本身回归、流量混合变化、计量错误，还是门禁阈值不当。复盘结论应回写 measurement contract、测试 fixture、数据分层或阈值版本，形成制度改进而非一次性修补。

### 39.15.7 最小可行 benchmark 平台

小团队不必先建巨型平台。一个可用的最小闭环包含：版本化 manifest、固定 runner、原始事件对象存储、纯函数聚合、Markdown 报告、CI 门禁、shadow 复制器和 canary 回滚钩子。先为一个高价值服务建立主指标与两个失败场景，再扩展到更多硬件和数据。平台的成功标准是“能在失败时解释并回滚”，而不是“有最多的图表”。

## 39.16 章节交付物地图

本章交付物彼此有明确边界：正文解释概念、机制、取舍和生产边界；`labs/ch39_evaluation_benchmark_lab.py` 负责可运行的 CPU-only 合成测量；`tests/test_ch39_evaluation_benchmark_lab.py` 固定行为契约；`reports/ch39-evaluation-benchmark-report.md` 记录一次执行的观察与局限；`reports/ch39-phase2-audit.json` 记录结构和验证状态；`evidence/ch39-evaluation-benchmark-manifest.json` 将主张映射到真实来源与本地命令。修改其中任一项都应更新版本、命令或审计字段，避免“正文说过但代码已经变了”。

## 39.17 实战推演：从一次性能争议到可回滚结论

假设平台团队提交了一个 fused attention 改动，宣称“吞吐提高 12%”。按照本章协议，不先接受百分比，而是建立 baseline/candidate pair。两者必须使用同一模型 digest、同一 tokenizer、同一输入长度分层、同一并发曲线、同一硬件拓扑和同一 warm-up。先跑 micro：固定 shape 下记录 kernel latency、带宽、数值误差与编译缓存命中；若数值误差超出容差，候选直接 deny，后续的“快”没有意义。

若 micro 通过，再跑 meso：将 tokenizer、prefill、decode、batch scheduler 和 KV cache 接起来。此时把吞吐拆成成功且按时完成的 requests/s，记录 p50/p95/p99、排队比例、OOM、cache miss 和重试。若吞吐提高 12%，但 p99 从 180ms 变成 250ms，而业务 SLO 是 220ms，联合门禁应 deny；可以保留候选作为低优先级离线批任务的实验分支，而不是将其称为全局提升。

若 meso 通过，才进入 macro shadow。shadow 复制自然比例流量与风险覆盖集，候选输出不写用户状态。质量 owner 盲评事实性、引用完整性和拒答安全性；平台 owner 观察资源争用、网络和日志成本；能源 owner 观察节点功率与设施估算的计量三角。任何一类数据缺失都产生 review，而不是自动晋级。shadow 期间还要检查复制本身是否改变缓存和容量，必要时使用独立 shadow 池。

canary 设定 5% 的流量上限、至少一个高峰与一个低峰窗口，并要求最小样本量。自动回滚条件包括：安全违规率超过硬阈值，p99 相对基线超过 5%，单位合格请求成本超过 5%，质量主指标下降超过 MDE，或数据污染/版本 manifest 不完整。回滚动作引用候选 artifact digest 的反向版本，保留失败 run 的事件与报告。若所有条件通过，逐步扩大到 25%、50%、100%，每一步都重新检查冷启动、缓存热度和区域分布。

这次推演说明“benchmark 证明了什么”必须随阶段变化：micro 证明一个机制在固定 shape 下可行；meso 证明组合组件在给定负载下满足资源与尾延迟；macro shadow 证明在不改变用户状态的情况下，候选与生产流量的差异可解释；canary 证明在有界真实暴露下仍满足质量、系统、经济和安全约束。没有任何一个阶段可以替代另一个阶段。

### 39.17.1 反事实检查

每次声称“改动带来提升”时，问三个反事实问题：

- 如果不改变代码，只改变输入长度、batch 或硬件，提升是否仍存在？若不存在，结论是 workload interaction，不是普遍优化。
- 如果保留代码但打乱数据顺序、清空缓存或改变时间块，提升是否仍存在？若不存在，可能是 cache/warm-up 或相关样本。
- 如果把候选的额外资源、功率或重试成本算回分母，质量和系统联合指标是否仍通过？若不通过，提升只是把成本转移到别处。

反事实不是要穷尽所有组合，而是挑选能推翻当前解释的最小实验。把反事实结果写入报告可显著减少下次争论，也避免把偶然优势固化成平台默认。

### 39.17.2 从失败中更新先验

benchmark 的价值不止是通过/失败。一次污染失败可能暴露数据生命周期没有冻结日；一次硬件漂移可能暴露节点池标签不完整；一次成本异常可能暴露控制面与日志没有分摊；一次质量与系统冲突可能暴露产品目标没有明确优先级。复盘时更新三个对象：measurement contract（测量什么）、manifest schema（记录什么）、门禁策略（如何决策）。不要只在脚本里加一个 if，使同类问题在下一版本继续出现。

建议保留一个可搜索的 benchmark registry：每个 run 绑定 commit、artifact、数据快照、硬件、主指标、reason code、owner、结论和后续动作。registry 不必暴露用户内容，但必须能回答“这个数字何时、在哪台机器、用哪版代码、对哪种流量得到”。当基线更新时，旧基线仍可查询，用于解释长期趋势和回归来源。

### 39.17.3 报告写作模板

一份可供评审的报告可以按以下顺序：

1. **决策摘要**：候选是什么，admit/deny/review，主原因是什么；
2. **协议**：workload、数据切分、硬件/软件矩阵、预热、重复和停止规则；
3. **原始样本摘要**：样本量、缺失、错误、重试、异常 run；
4. **结果**：micro/meso/macro 表格与区间，质量和系统联合图；
5. **成本/能耗边界**：计量点、公式、因子版本、不确定性和未覆盖项；
6. **污染与安全**：去重、时间切分、工具副作用、脱敏和权限；
7. **shadow/canary**：流量比例、窗口、回滚条件、观察 owner；
8. **限制与下一步**：未测矩阵、外推假设、需要补的实验和到期时间；
9. **证据索引**：manifest、原始事件、代码 commit、命令、测试和审计 JSON。

这种结构让读者先看到决策，再看到证据和边界。若结论是 review，报告应明确列出“补什么数据会改变决定”，避免 review 变成无限期搁置。

## 39.18 评测运营的日常节奏

benchmark 进入生产后，最容易被忽视的是日常维护。建议把运行分成三个节奏。每日只检查轻量 smoke：版本是否能启动、主路径是否有错误、核心 p95 是否在明显异常范围、manifest 是否完整。每周运行固定的 micro/meso 矩阵，检查编译器、驱动、节点池和依赖更新带来的漂移；每月或每次模型发布运行 macro 自然比例集与风险覆盖集，更新质量抽样、成本归因和能源边界。每个节奏都有自己的停止规则和 owner，不能用每日 smoke 的绿色替代月度质量评测。

运行结果要与变更事件关联。依赖升级、硬件替换、功率策略、数据快照、提示模板、检索索引和流量路由都应生成 change ID；没有 change ID 的性能变化进入 review。长期趋势图要同时展示基线版本、候选版本、样本量、CI 宽度和重大事件标记，避免把自然季节性、流量变化或容量扩缩当作代码回归。

当 CI 反复变宽时，优先调查样本相关性、主机异质性、缓存状态和流量混合，而不是简单增加 n。增加样本只能减少随机误差，无法修复系统性偏差。若真实 workload 的分布发生变化，应创建新 case 或新 baseline，并在报告中说明不可与旧 baseline 直接比较。旧数据保留用于趋势分析，不能被悄悄重写成新分布。

最终，评测平台应让工程师在提交改动时就看到“需要补哪些证据”，让发布 owner 在 canary 阶段看到“哪条约束失败、谁负责回滚”，让复盘人员在数月后仍能回答“这个结论的边界是什么”。这比追求一个看似精确的单点数字更能降低 AI Infra 的长期风险。
