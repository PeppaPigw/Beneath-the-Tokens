# 第39章评测与基准实验报告

日期：2026-10-07  
范围：`labs/ch39_evaluation_benchmark_lab.py` CPU-only deterministic toy  
目的：验证 micro/meso/macro 结果结构、吞吐/尾延迟/成本/能耗单位、均值 CI95、数据污染检查、质量+系统联合门禁、回归门禁以及 online shadow/canary 的 fail-closed 契约

## 环境与边界

- Python 3 标准库；未安装模型、未访问网络、未使用 GPU/NVML/DCGM、未修改集群或用户数据
- fixture：6 个固定 case（2 micro、2 meso、2 macro），`runner=toy-1.0`、`runtime=cpu-stdlib`、`dataset=synthetic-v1`
- 合成成本、功率和质量仅用于验证公式与决策路径，不代表云价格、设备功率、碳排、模型能力或线上 SLO
- CI95 使用确定性的均值正态近似；真实长尾 p95 应使用按块 bootstrap/合适的分位数区间

## 执行命令

```bash
python3 -m py_compile labs/ch39_evaluation_benchmark_lab.py
python3 labs/ch39_evaluation_benchmark_lab.py --fault none > /tmp/ch39-none.json
for fault in regression contamination canary low_quality hardware_drift; do
  python3 labs/ch39_evaluation_benchmark_lab.py --fault "$fault" > "/tmp/ch39-$fault.json"
done
python3 tests/test_ch39_evaluation_benchmark_lab.py
```

## 基线结果（fault=none）

| tier/case | p95 ms | throughput rps | USD/1k requests | J/request |
| --- | ---: | ---: | ---: | ---: |
| micro/kernel_matmul | 3.825 | 439.637739 | 0.00113730 | 0.545904 |
| micro/tokenizer_prefill | 6.300 | 266.922913 | 0.00187320 | 0.674352 |
| meso/single_replica | 76.500 | 86.764367 | 0.00704334 | 3.57289531 |
| meso/batch_scheduler | 108.000 | 125.667609 | 0.00486292 | 2.307675 |
| macro/online_mix | 193.500 | 68.799943 | 0.00968993 | 4.79651562 |
| macro/rag_quality | 252.000 | 26.528717 | 0.02513000 | 12.250875 |

基线 `decision=admit`，联合质量/系统门禁四项均通过（accuracy、safety、p95、cost）。回归 baseline 与 candidate 使用同一 `online_mix` workload，避免把不同 macro case 的差异误判成版本回归。基线 JSON 的硬件字段明确 `accelerator=none`，两次运行字节级相同。

## 故障观察

| fault | 预期 | deny reason |
| --- | --- | --- |
| regression | deny | `regression_latency`, `regression_cost` |
| hardware_drift | deny | `regression_latency`, `regression_cost` |
| canary | deny | 同一回归门禁失败；shadow 仍是 observe 且不修改状态 |
| low_quality | deny | `quality_or_system_accuracy` |
| contamination | deny | `dataset_contamination` |

故障模拟通过合成 drift 或 train/eval overlap 注入，不代表真实硬件、数据集或线上 canary 的概率。所有失败均保留 reason code，未吞错或改写成通过。

## 测试覆盖

直接测试覆盖：

1. baseline 确定性、CPU-only、无外部副作用和联合门禁；
2. micro/meso/macro 三层矩阵、p50/p95/p99、均值 CI95、吞吐、成本和能耗单位；
3. 质量与系统联合失败；
4. 回归阈值、shadow observe、canary deny；
5. 训练/评测污染 overlap 与 regression/low_quality 故障闭环；
6. percentile、CI、非法 tier 输入校验；
7. CLI stdout 与 `--output` 文件 JSON 相等。

## 生产接受条件与未覆盖项

生产采用前必须替换合成 fixture，并额外提供：模型/运行时/驱动/硬件 digest，真实或脱敏数据快照，输入长度和并发分层，原始事件与 trace，按块分位数 CI，失败重试与冷启动边界，账单和节点/设施能耗对账，PUE/碳因子版本，污染审计，人评一致性，shadow 脱敏与副作用隔离，canary 最小样本/观察窗口、自动回滚 artifact 和 bypass 过期时间。

本报告未测真实 GPU kernel、网络拓扑、KV cache、模型准确率、供应商价格、设施 PUE/碳、跨区域故障、隐私合规或用户转化。toy 通过只说明代码契约和教学公式稳定，不能作为性能承诺或发布批准。
