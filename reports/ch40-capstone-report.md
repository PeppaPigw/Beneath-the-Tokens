# 第40章生产级 Capstone 实验报告

日期：2026-10-07  
范围：`labs/ch40_capstone_lab.py` CPU-only deterministic toy  
目的：验证 Atlas Summarize 场景的容量计划、SLO/错误预算、成本与能耗、灰度门禁、证据 manifest 以及故障 fail-closed 契约

## 环境与边界

- Python 3 标准库；无模型下载、无 GPU/NVML/DCGM、无网络请求、无云账单、无集群或用户数据副作用
- 固定 fixture：峰值 60 QPS，单节点有效 24 QPS，30% headroom，4 节点，p95 300 ms，可用性 99.5%，月预算 12,000 USD
- 节点月费、每请求能耗、质量、流量和 SLO 数字为合成教学输入，不代表供应商价格、设备功率、碳排、模型能力或生产承诺
- 结果中的 `admit/deny` 只证明代码路径和契约；生产必须用锁定硬件、版本、真实/脱敏流量、账单和质量集复验

## 执行命令

```bash
python3 -m py_compile labs/ch40_capstone_lab.py tests/test_ch40_capstone_lab.py
python3 labs/ch40_capstone_lab.py --fault none > /tmp/ch40-none.json
for fault in traffic_spike gpu_loss network_partition schema_drift budget_breach rollback; do
  python3 labs/ch40_capstone_lab.py --fault "$fault" > "/tmp/ch40-$fault.json"
done
python3 tests/test_ch40_capstone_lab.py
```

## 基线结果（fault=none）

- `decision=admit`，`required_nodes=4`、`provisioned_nodes=4`、`capacity_ok=true`
- `p95_ms=240`、`availability=0.999`，SLO 两项通过；月请求 2,000,000 时错误预算为 10,000 请求
- 节点月成本为 8,400 USD，低于 12,000 USD 预算；能耗按 1.8 Wh/请求换算为 3,600 kWh/月（合成值）
- baseline/canary rollout 为 `promote`；CPU-only、deterministic、no_external_side_effects 三个不变量为真
- evidence manifest 对规范化结果计算 SHA-256，并声明 request contract、capacity plan、SLO snapshot、rollout gate、fault record、owner ack 六类必备 artifact

## 故障观察

| fault | decision | 关键 reason code | 机制观察 |
| --- | --- | --- | --- |
| traffic_spike | deny | `capacity_insufficient`, `slo_latency` | 峰值放大到 108 QPS，需要 6 节点；先限流/扩容，不静默超 SLO |
| gpu_loss | deny | `capacity_insufficient`, `node_loss`, `slo_latency` | 丢一节点后仅 3 节点；N+1 余量不足，需降级或补容量 |
| network_partition | deny | `rollout_latency`, `rollout_errors`, `slo_availability` | canary 回滚，SLO 不可用；重试不能无限放大故障 |
| schema_drift | deny | `rollout_quality`, `rollout_errors`, `slo_availability` | schema/质量契约失败，禁止仅凭 HTTP 200 晋级 |
| budget_breach | deny | `budget_breach`, `rollout_latency` | 成本改为 14,400 USD，预算控制和回滚同时生效 |
| rollback | deny | `rollout_latency`, `rollout_errors` | 触发可审计的 rollback reason code |

所有故障都保留结构化 `deny_reasons`，没有吞错或把失败改写为成功。实验故障是确定性注入，不是故障概率估计。

## 测试覆盖

直接测试覆盖：

1. 基线确定性、CPU-only、无外部副作用和 evidence digest；
2. 容量公式、headroom、节点数、利用率和错误预算单位；
3. rollout p95/error/quality 联合门禁及 rollback reasons；
4. 节点成本、每请求能耗和预算字段；
5. traffic spike、node loss、network partition、schema drift、budget breach、rollback 六类故障 fail-closed；
6. 输入校验与 CLI stdout/`--output` JSON round trip。

## 生产接受条件与未覆盖项

生产采用前必须替换合成 fixture，并提供：模型/运行时/驱动/硬件和容器 digest；真实或脱敏流量与长度桶；真实队列、cache、网络、存储、GPU 和依赖遥测；按块分位数置信区间；账单、功率、PUE/碳因子和估算误差；质量/安全/污染/人评协议；租户 ACL、删除传播和审计证据；shadow 副作用隔离、canary 最小样本/观察窗口、自动回滚 artifact、break-glass 过期和 owner 签字。

本实验没有测真实 GPU kernel、模型智能、网络包丢失、租户侧信道、法务合规、跨区故障、供应商账单或用户转化。toy 通过只说明控制路径和教学公式稳定，不能作为性能、成本、可用性或安全认证。
