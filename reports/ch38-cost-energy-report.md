# 第38章 AI Infra 成本、容量与能源实验报告

- 日期：2026-10-07
- 实验：`labs/ch38_cost_energy_lab.py`
- 级别：L0，CPU-only，Python 3.10+ 标准库
- 目的：用固定 fixture 模拟容量规划、排队、GPU 利用率陷阱、reserved/spot/on-demand 混合、弹性/热功率、PUE/碳、训练与推理 TCO、SLO 成本权衡和 FinOps 数据契约
- 边界：不访问云 API、价格接口、Kubernetes、NVML/DCGM、GPU、电表、registry 或真实账单；功率、PUE、碳因子、价格、到达率与服务率都是合成输入，结果不是生产 SLO、能源或碳盘查证明

## 可复现命令

```bash
python3 -m py_compile labs/ch38_cost_energy_lab.py
python3 labs/ch38_cost_energy_lab.py --fault none --output reports/ch38-cost-energy-baseline.json
python3 labs/ch38_cost_energy_lab.py --fault underprovisioned --output reports/ch38-cost-energy-underprovisioned.json
python3 labs/ch38_cost_energy_lab.py --fault spot_interruption --output reports/ch38-cost-energy-spot.json
python3 labs/ch38_cost_energy_lab.py --fault thermal_limit --output reports/ch38-cost-energy-thermal.json
python3 labs/ch38_cost_energy_lab.py --fault carbon_miss --output reports/ch38-cost-energy-carbon.json
python3 labs/ch38_cost_energy_lab.py --fault slo_violation --output reports/ch38-cost-energy-slo.json
python3 tests/test_ch38_cost_energy_lab.py
```

脚本无随机数、无网络和无时间读取；stdout 与 `--output` 文件的 JSON 相同。结果 schema 为 `schema_version=1`，数字在边界处固定舍入。toy 中的 `gpu_hours`、`power_w`、PUE 和 `carbon_factor_kg_per_kwh` 是输入假设，生产必须替换成带版本的计费、遥测和设施证据。

## 结果摘要

| 场景 | 决定 | 关键证据 | 处置语义 |
| --- | --- | --- | --- |
| `none` | `admit` | 计划 3 副本，p95 wait 195ms，SLO/热/碳预算通过 | 保留成本、能耗和利用率证据 |
| `underprovisioned` | `deny` | 1 副本、ρ>1、overflow、p95 wait 4162.857ms | 扩容或降级，不能用高设备利用率掩盖排队 |
| `spot_interruption` | `deny` | 35% synthetic failure，成功服务率下降，恢复成本计入 | 启用按需保险、缩短 checkpoint、冻结 deadline 风险 |
| `thermal_limit` | `deny` | 计划峰值功率超过 600W 上限，active replicas 被限制 | 迁移/限频/排队，先检查热与配电边界 |
| `carbon_miss` | `deny` | 2.82kg CO2e 超过 0.5kg 预算 | 错峰或低碳区域候选，仍需验证 SLO/网络成本 |
| `slo_violation` | `deny` | 到达率 140 rps，p95 2820ms 且计划功率不足 | 增加安全容量或降级，不能只追求低 $/GPU·h |

基线中 `device_utilization` 与 `effective_utilization` 同时输出；后者扣除了重试/失败造成的无效工作。基线计划容量的目标利用率为 0.75 并留 10% headroom，toy 采用平滑的 M/M/1 风格近似，不是多服务器精确排队解。

## Toy 实现与合同

- `Workload.validate()`、`Pricing.validate()` 和 `EnergyModel.validate()` 对到达率、服务率、利用率、折扣、PUE、碳预算和功率上限做显式边界检查；畸形输入返回 `ValueError`，不静默生成账单。
- `plan_capacity()` 用 `ceil(arrival / (service_rate × target_utilization × (1−headroom)))` 产生可解释副本建议，并保留 raw、cap 和 capacity_shortfall。
- `queue_model()` 输出 λ、μ、ρ、overflow、p95 wait、effective tokens/s、device/effective utilization、deadline miss 和 availability。ρ≥1 时使用保守有限公式，避免 toy 除零，但生产应使用校准的多类/多服务器模型。
- `procurement_cost()` 比较 reserved、spot、on-demand 单价，显式计算 interruption/recovery 和成功 GPU 小时；它不代表任何云合同、税费或承诺摊销规则。
- `energy_carbon()` 计算 IT kWh、PUE 后设施 kWh、kgCO2e、峰值功率和碳预算余量。功率是合成设备侧输入，不等于电表读数；没有 embodied carbon 或抵消额。
- `tco_summary()` 给出 training `$/successful_step` 与 inference `$/SLO request`/kg per million requests，包含 toy storage/network/platform shared cost。
- `finops_contract()` 演示标签完整/缺失和异常预算动作；它不连接发票或执行终止。
- `simulate()` 将 underprovisioned、spot、thermal、carbon 和 SLO 场景 fail-closed，返回理由、假设和不变量；无外部副作用。

## 失败边界与解释

1. 到达率、服务率、request token、价格、PUE、碳因子和中断率都是小 fixture；它们只验证单调关系与 JSON 合同，不预测真实吞吐、账单、热或碳。
2. 排队是近似公式，没有 batch、优先级、序列长度、KV cache、网络、拓扑、冷启动和多 GPU 并行；生产需 replay trace、分层服务率和故障注入校准。
3. reserved/spot/on-demand 的折扣和 interruption 只用于教学；云厂商价格、计费粒度、承诺、税费、区域供应和中断历史必须按日期/合同固定。
4. `gpu_power_w` 和 PUE 不来自 NVML/DCGM 或设施电表；生产需核对采样点、驱动/固件版本、机架峰值、冷却、UPS、季节和供应商披露。
5. carbon factor 是单一位置法输入，未包含市场法、绿电凭证、生命周期/embodied carbon、水、迁移网络或不确定区间；不能直接写 ESG 声明。
6. TCO shared cost 是固定金额；没有工资、折旧、许可证、对象存储层级、出口、失败任务全量、日志保留和机会成本。
7. toy 只将 `spot_interruption` 作为 synthetic failure；真正恢复还涉及 checkpoint 内容、对象存储、驱动、拓扑、调度和数据一致性。
8. FinOps 标签只返回报告动作，不修改资源；生产自动动作必须有预算、SLO、审批、幂等、冷却和回滚。

## 生产验收计划

1. 在固定日期冻结价格表、合同折扣、计费粒度、税费和区域供应；用发票、资源清单和作业事件进行日级对账，报告差异而非平均摊平。
2. 用真实 request/训练 trace 分层校准服务率、序列长度、batch、KV cache、通信、冷启动、优先级和 p95/p99；记录成功产出、失败重试与 deadline miss。
3. 在 staging 注入节点/区域故障、spot 抢占、checkpoint 损坏、存储延迟、网络拥塞和热限频，验证 SLO、恢复 wall-clock、预算和单位成本。
4. 使用 NVML/DCGM、机架电表或云能耗披露核对 GPU/主机/网络/UPS/冷却边界；版本化 PUE、功率采样和传感器校准。
5. 选择位置/市场法碳因子、时间粒度和供应商来源，做灵敏度区间；验证 carbon-aware routing 的跨区流量、失败重试、数据驻留与用户延迟。
6. 让 `service/team/model/workload_type/environment/region/gpu_sku/capacity_class` 从提交、调度、遥测到账单闭环；缺失标签进入 unallocated 队列并有 owner。
7. 只读 shadow 建议稳定后，再对低优先级和可回滚作业启用弹性；每个自动动作保留 dry-run、预算阈值、SLO guardrail 和回滚证据。
8. 将单位经济（$/SLO request、$/successful step、kgCO2e/同一分母）、利用率、队列和错误预算放入同一变更评审，避免只优化 GPU 忙碌率。

## 对应测试

`tests/test_ch38_cost_energy_lab.py` 覆盖：

- 基线确定性、CPU-only/no-side-effect、SLO 和碳公式不变量；
- capacity planning、ρ>1 排队以及 device/effective utilization 陷阱；
- reserved/spot/on-demand 预期成本与中断恢复；
- IT kWh、PUE、设施 kWh、kgCO2e 关系和 thermal/carbon fault；
- underprovisioned、spot interruption、SLO violation 的 fail-closed reason；
- 输入校验与 CLI stdout/文件 JSON 一致性。

测试通过表示 toy 公式、边界和可重复输出满足合同；不表示云计费、GPU 功率、碳排或生产 SLO 已验证。

## 主要来源

- NVIDIA DCGM：https://docs.nvidia.com/datacenter/dcgm/latest/
- NVIDIA NVML：https://docs.nvidia.com/deploy/nvml-api/
- Kubernetes HPA：https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- KEDA：https://keda.sh/docs/latest/concepts/scaling-deployments/
- AWS EC2 Spot：https://aws.amazon.com/ec2/spot/
- Google Cloud Spot：https://cloud.google.com/compute/docs/instances/spot
- Azure Spot VM：https://learn.microsoft.com/azure/virtual-machines/spot-vms
- FinOps Framework：https://www.finops.org/framework/
- FinOps FOCUS：https://focus.finops.org/focus-specification/v1-0/
- OpenTelemetry Metrics：https://opentelemetry.io/docs/specs/otel/metrics/
- GHG Protocol Scope 2：https://ghgprotocol.org/scope-2-guidance
- Green Software SCI for AI：https://greensoftware.foundation/articles/software-carbon-intensity-for-ai
- Carbon Aware SDK：https://github.com/Green-Software-Foundation/carbon-aware-sdk
- Kepler：https://github.com/sustainable-computing-io/kepler
- CodeCarbon：https://github.com/mlco2/codecarbon
- MLPerf Training/Inference：https://mlcommons.org/benchmarks/
- NIST Queueing Theory：https://www.nist.gov/itl/iad/queueing-theory

本地证据索引见 `evidence/ch38-cost-energy-manifest.json`。数字和公式应在生产验收时使用固定版本的供应商账单、遥测、设施计量、碳因子和 workload replay 重新测量。
