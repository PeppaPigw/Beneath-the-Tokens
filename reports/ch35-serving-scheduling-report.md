# 第35章在线推理调度与服务 SLO 实验报告

- 日期：2026-10-07
- 实验：`labs/ch35_serving_scheduling_lab.py`
- 级别：L0，CPU-only，Python 3.10+ 标准库
- 目的：验证 arrival/queue、prompt token 预算、连续 batching、prefill/decode 耦合与解耦、TTFT/ITL/E2E 分位数、取消、stream 背压、admission 与多租户调度合同
- 边界：这是 deterministic protocol toy，不是 vLLM、SGLang、Triton、CUDA 或任何 GPU 的吞吐/延迟保证

## 可复现命令

```bash
python3 -m py_compile labs/ch35_serving_scheduling_lab.py
python3 labs/ch35_serving_scheduling_lab.py \
  --output reports/ch35-serving-scheduling-default.json
python3 labs/ch35_serving_scheduling_lab.py --burst --disaggregated \
  --policy weighted_fair --output reports/ch35-serving-scheduling-disagg.json
python3 tests/test_ch35_serving_scheduling_lab.py
```

## Toy 配置

- coupled 默认：`max_batch_requests=4`、`max_batch_prompt_tokens=32`
- prefill 代理：`0.30 ms + 0.08 ms/token`
- decode 代理：`0.25 ms + 0.12 ms/active sequence/iteration`
- workload：gold/bronze 两租户，包含不同 prompt/max output，以及一个 `cancel_ms=4.0` 的请求
- disaggregated workload：12-request burst，`weighted_fair`，gold 权重 2、bronze 权重 1
- 所有时间是固定浮点代理，用来验证阶段顺序和指标定义；没有 CUDA、GPU、网络或真实 tokenizer

## 默认 coupled 结果

| 指标 | 结果 |
| --- | ---: |
| completed | 4 |
| cancelled | 1 |
| TTFT p50 (ms) | 3.005 |
| TTFT p95 (ms) | 3.370 |
| ITL p50 (ms) | 0.7975 |
| ITL p95 (ms) | 0.945583 |
| E2E p50 (ms) | 6.000 |
| E2E p95 (ms) | 6.7005 |
| gold completed/cancelled | 2 / 1 |
| bronze completed/cancelled | 2 / 0 |

`r2` 的取消在 decode 边界生效，记录为 terminal `cancelled`，不进入 completed 的 TTFT/ITL/E2E 分位数。所有 request 的状态都是 `completed` 或 `cancelled`，并且 `tokens_accounted=true`。

## burst + disaggregated + weighted fair 结果

| 指标 | 结果 |
| --- | ---: |
| completed | 12 |
| cancelled | 0 |
| TTFT p50 (ms) | 6.485 |
| TTFT p95 (ms) | 8.300 |
| ITL p50 (ms) | 1.060 |
| ITL p95 (ms) | 1.210 |
| E2E p50 (ms) | 10.600 |
| E2E p95 (ms) | 12.910 |
| gold TTFT p95 (ms) | 6.000 |
| bronze TTFT p95 (ms) | 8.300 |

这个结果只说明在 toy 时间线中，prefill `ready_at` 可以与已有 decode 时间线重叠；它不证明解耦在目标 GPU 上一定更快。真实比较还要测 KV transfer bytes、传输重试、接收队列、page layout、GPU 利用率和冷启动。

## 测试合同

`tests/test_ch35_serving_scheduling_lab.py` 覆盖：

1. 默认 workload 的终态、不变量和 SLO 分位数；
2. batch request/prompt cap，六个请求被拆成至少三个 prefill batch；
3. coupled/disaggregated 的确定性、配置回显和 metrics；
4. admission 的全局/租户拒绝原因；
5. cancellation、bounded stream buffer 和 producer blocked 证据；
6. percentile 的插值、空输入和边界校验；
7. CLI stdout 与 `--output` JSON 完全一致。

## failure-boundary JSON

`reports/ch35-serving-scheduling-failure.json` 记录了容量不足时的 admission 拒绝（全局与租户原因）、一个取消终态和有界 stream buffer。它用于审计“在 GPU OOM 之前拒绝、取消最终释放、背压有界”的失败合同，不是异常吞吐 benchmark。

## 失败/边界观察

- 无界 queue、只按 request 数做 admission 或用 GPU OOM 作为准入信号，无法保护 p99；toy 要求显式 batch token cap 和终态。
- cancellation 只关闭 HTTP 连接不会自动释放 KV；生产系统需分别记录 `cancel_requested`、`cancel_applied` 和 `cancel_to_kv_free`。
- disaggregation 的 `T_kv` 可能超过本地 prefill 节省；必须把 `prefill、transfer、decode_ready` 三段独立打点。
- P95/P99 在样本不足时不可靠；生产报告需保留窗口、分母、bucket 和脱敏 trace digest。
- toy 没有模拟真实 kernel、CUDA graph shape 约束、KV eviction、网络丢包、客户端渲染、认证或内容安全，因此只能作为协议和测试合同证据。

## 生产验收清单

固定服务引擎 commit、模型 revision、量化、GPU 型号/驱动、TP/PP、KV dtype/page、请求 trace、并发和 warm-up；记录 TTFT/ITL/E2E p50/p95/p99/max、accepted/rejected/cancelled、KV free、prefill/decode queue、stream flush gap、tenant fairness 和 error-budget burn rate。所有自动降级需带配置版本、decision log、回滚条件和保留期限。

## 主要来源

- Orca：https://www.usenix.org/conference/osdi22/presentation/yu
- vLLM/PagedAttention：https://arxiv.org/abs/2309.06180
- DistServe：https://arxiv.org/abs/2401.09670
- Splitwise：https://arxiv.org/abs/2311.18677
- Sarathi-Serve：https://arxiv.org/abs/2403.02310
- Llumnix：https://arxiv.org/abs/2406.04837
- Triton dynamic batching：https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher
- Ray Serve batching：https://docs.ray.io/en/latest/serve/advanced-guides/dynamic-request-batching.html
- gRPC deadlines/cancellation：https://grpc.io/docs/guides/deadlines/、https://grpc.io/docs/guides/cancellation/
- SSE：https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- HTTP semantics：https://www.rfc-editor.org/rfc/rfc9110
- Prometheus histograms：https://prometheus.io/docs/practices/histograms/

外部来源需在目标版本和硬件上重新核验，论文结果和本地代理数字不能替代生产压测。
