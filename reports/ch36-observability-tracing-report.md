# 第36章可观测性与追踪实验报告

## 范围与边界

本报告记录 `labs/ch36_observability_tracing_lab.py` 的 CPU-only、确定性 toy 实验。它模拟 OpenTelemetry 风格 span/context、Prometheus histogram、结构化日志、短 profile 样本，以及 GPU/网络/存储遥测快照；不调用 CUDA、DCGM、Nsight、eBPF、Prometheus server 或真实网络。结果验证协议和失败边界，不是 GPU 型号、驱动或生产 QPS 的性能证明。

## 可复现命令

```bash
python3 -m py_compile labs/ch36_observability_tracing_lab.py
python3 tests/test_ch36_observability_tracing_lab.py
python3 labs/ch36_observability_tracing_lab.py --fault none --output reports/ch36-observability-baseline.json
python3 labs/ch36_observability_tracing_lab.py --fault network_tail --output reports/ch36-observability-default.json
python3 labs/ch36_observability_tracing_lab.py --fault storage_tail --output reports/ch36-observability-storage.json
```

固定 `seed=7`，默认生成 12 个请求。两次相同命令的 JSON 应完全相同。

## 结果摘要

| 场景 | 主要注入 | 诊断结果 | 证据 |
| --- | --- | --- | --- |
| none | 无 | `no_injected_fault` | phase histograms、GPU telemetry、trace topology |
| network_tail | 3 个请求增加 RPC/collective 尾部，含 retransmit | `network_or_collective_tail` | `network.rpc` spans、`network_retransmits_total`、decode/ITL tail |
| storage_tail | 3 个请求增加 NVMe fsync/IO 延迟 | `storage_io_tail` | `storage.checkpoint_fsync` spans、`disk_io_latency_ms`、queue tail |

所有场景应满足：

- `all_requests_terminal=true`；
- span parent ID 可解析、duration 非负、每个请求有完整 trace context；
- metrics 的 forbidden labels 明确包含 `trace_id` 和 `request_id`；
- tail sampler 对慢/错误或注入 fault 的 trace 保留至少一个样本；
- diagnosis 输出 trace→metric→resource→profile 的关联边。

## 失败边界与解释

1. toy 中 `sm_active_ratio`、重传数和磁盘延迟是合成值，不能据此推断真实 GPU clocks、NVLink 或 NVMe 行为。
2. histogram quantile 由有限样本插值，生产应记录 bucket 配置、窗口、样本数和 exemplar。
3. tail sampler 只演示错误/慢/fault predicate 与确定性 head sample；Collector 内存上限、OTLP 重试、权限和数据脱敏需在集成环境验证。
4. trace ID 没有进入 metric labels；`cardinality_budget` 故意保留 4 个 dropped series 以验证预算告警。
5. storage/network 故障只改变 toy phase，真实系统还需考虑 NUMA、NCCL rank skew、网络拥塞和对象存储限流。

## 生产验收清单

- 固定 OTel SDK/Collector、Prometheus、DCGM、驱动、CUDA、Nsight 和 PyTorch 版本；记录 schema 与采样配置。
- 压测 telemetry overhead：CPU、内存、出口带宽、Collector queue、remote-write lag 及故障时降级行为。
- 在 staging 回放真实 prompt 长度/输出长度分布，核对 TTFT/ITL/E2E 的时间点定义和 SLO 分母。
- 验证 context 在 HTTP/gRPC、队列、actor、NCCL/NVTX、GPU exporter 和存储 I/O 边界不丢失；统计无 context 事件。
- 设 cardinality budget、敏感字段脱敏、访问控制、保留期和删除审计；禁止把 prompt、token、trace_id 作为无限 labels。
- 用 network、storage、CPU starvation、GPU throttle 四类受控故障做反事实演练，要求诊断器给出支持与反证。

## 对应测试

`tests/test_ch36_observability_tracing_lab.py` 覆盖 deterministic replay、span tree、network/storage fault diagnosis、tail sampling、cardinality guard、percentile 输入校验和 CLI JSON 输出。测试通过只说明 toy 契约成立；生产上线仍需硬件和流量回放证据。
