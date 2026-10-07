# 第 29 章 Speculative + Structured Decoding CPU toy 实验报告

- 实验日期：2026-10-07
- 实验等级：L0，CPU-only、Python 标准库；不是 vLLM/SGLang、GPU、CUDA、真实模型或质量 benchmark
- 脚本：`labs/ch29_speculative_lab.py`
- 测试：`tests/test_ch29_speculative_lab.py`
- Python：以运行命令的 `python3 --version` 为准；本次使用 Python 3.12
- 随机种子：29；`logical_time_ms` 是 toy 成本函数的离散时钟，不是 wall-clock

## 实验问题和协议

实验回答四个可复现问题：

1. Leviathan accept-reject 在拒绝时是否提交一个 residual correction，并保持输出长度合同？
2. draft 长度和 acceptance 参数是否减少 target 逻辑推进次数，还是被 draft/grammar 成本抵消？
3. grammar mask 是否让非法 token 的概率为零，且 rejection 计数可观察？
4. baseline、speculative、lookahead 是否都在相同并发和 token budget 下完成请求？

`exact_verify` 接收目标分布 p、草稿分布 q 和候选 token；以 `min(1,p/q)` 接受，拒绝时从正部 residual `max(p-q,0)` 归一化采样一个修正 token。`constrained_speculative` 在一个小型 JSON-like 有限状态 grammar 上对 draft 和 target 使用同一 mask，然后生成 k 个候选并验证。`simulate` 用 baseline/speculative/lookahead 三种模式模拟 scheduler；它统计 target/draft token、accepted token、grammar reject、逻辑时间和 TTFT。所有随机数由 `random.Random(seed)` 驱动。

## 命令

```bash
python3 labs/ch29_speculative_lab.py \
  --seed 29 --requests 24 --concurrency 6 --output-len 32 \
  --draft-len 4 --accept-rate 0.72 --mode baseline \
  --output reports/ch29-target-only.json

python3 labs/ch29_speculative_lab.py \
  --seed 29 --requests 24 --concurrency 6 --output-len 32 \
  --draft-len 4 --accept-rate 0.72 --mode speculative \
  --grammar --grammar-reject-rate 0.10 \
  --output reports/ch29-speculative-json.json

python3 labs/ch29_speculative_lab.py \
  --seed 29 --requests 24 --concurrency 6 --output-len 32 \
  --draft-len 4 --accept-rate 0.72 --mode lookahead \
  --grammar --grammar-reject-rate 0.10 \
  --output reports/ch29-lookahead-json.json

python3 tests/test_ch29_speculative_lab.py
```

## 结果

| 场景 | 完成请求 | scheduler steps | logical time ms | TTFT p50/p95 ms | target tokens | draft tokens | accepted tokens | acceptance | grammar rejects | 最终输出 tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target-only baseline | 24 | 4 | 63.640000 | 15.01 / 43.58 | 768 | 0 | 0 | 0.000000 | 0 | 768 |
| speculative + grammar | 24 | 4 | 22.912500 | 3.773333 / 11.988333 | 172 | 641 | 620 | 0.967239 | 34 | 768 |
| lookahead + grammar | 24 | 4 | 23.272500 | 3.863333 / 12.258333 | 172 | 641 | 620 | 0.967239 | 34 | 768 |

这些结果只反映脚本中的成本常数：speculative 每轮最多 4 个候选，目标推进按 accepted/correction 估算；lookahead 额外加一个 tree bookkeeping 常数。`target tokens` 不是 GPU forward FLOPs，`logical time` 不是硬件时间。speculative 场景最终 output token 数与 baseline 一致，说明 toy 的 request-level completion 合同成立；不说明 token 内容分布与真实 target 一致。

## 观察和机制解释

1. **候选减少目标推进。** 在相同 768 个最终输出 token 下，speculative toy 统计 172 个 target logical tokens、641 个 draft tokens、620 个 accepted tokens。逻辑时间从 63.64 ms 降到 22.91 ms，是人为设置的 target/draft 成本结果，不能转译成 GPU 加速比例。
2. **grammar 成本可见。** 34 次 grammar reject 被单独记录，最终输出 token 数仍是 768。该指标只说明有限状态 mask/retry 账本工作；脚本 grammar 不是 JSON Schema parser，不能证明 JSON 语义合法。
3. **lookahead 不是免费。** lookahead 与 speculative 使用相同候选参数，但 tree bookkeeping 使逻辑时间略高（23.27 ms vs 22.91 ms）。真实树验证还会受到候选分支、attention mask、KV scatter 和 kernel occupancy 的影响。
4. **acceptance 不是质量指标。** acceptance=0.967239 表示 toy 的 q/p 提案在 mask 和随机种子下被接受的比例；高 acceptance 不代表事实性、安全或 schema 业务正确。
5. **TTFT 需要定义边界。** 本模拟将第一个 committed token 的逻辑事件作为 TTFT；draft token 从未写入输出流。真实 serving 必须确认 stream writer 只发送 commit queue，避免把可能 rollback 的 token 泄漏给客户端。

## 合同测试

`python3 tests/test_ch29_speculative_lab.py` 输出 `ch29 tests: PASS`，覆盖：

- q 把所有质量放在错误 token 时，accept-reject 拒绝并提交一个 correction；
- grammar mask 后非法 token 概率严格为零；
- constrained output 的每个 toy token 都满足对应 grammar state；
- baseline/speculative 请求都完成，最终输出 token 数等于请求合同；
- 相同 seed 的 scheduler payload 完全可复现；
- CLI stdout 与 `--output` 文件拥有相同 JSON（`schema_version=1`）。

测试是协议/状态机合同，不是 vLLM、SGLang、CUDA event、数值稳定性、吞吐、隐私或生产 readiness 证明。真实系统还需 property-based 分布测试、top-p/temperature、quantization、tree sibling mask、page boundary、cancel/stream race 和多租户 cache isolation。

## 受控扫描建议

- `--draft-len 1,2,4,8`：记录 accepted/token、target/draft/grammar 时间和 p99 ITL；低 acceptance 时应自动减小 k 或回退 target-only。
- `--accept-rate 0.2,0.5,0.8,1.0`：区分候选质量与 grammar 成本；不要只看 target token 数。
- `--grammar-reject-rate 0,0.1,0.5`：观察 reject、逻辑时间和 dead-end 行为；真实 grammar 应替换为 JSON Schema/regex backend。
- `--concurrency 1,6,24` 与 `--max-batch-tokens 8,32,128`：观察 admission、TTFT p95/p99 和 batch fairness。
- `--mode lookahead` 对比 speculative：固定候选分布，单独改变 tree bookkeeping 常数，避免把多个变量的效果混在一起。

## 可复现边界与下一步真实实验

toy 没有真实模型 logits、tokenizer/chat template、GPU kernels、PagedAttention/RadixAttention、CUDA graph、TP/DP/NCCL、PD disaggregation、HTTP/TLS、schema compile、网络 flush 或客户端取消。`json_allowed` 只是 7 状态 toy grammar；`grammar_reject_rate` 是 Bernoulli 故障注入，不是 parser 统计。`target_tokens`/`draft_tokens`/`logical_time_ms` 不可与论文 headline、vLLM benchmark 或 SGLang benchmark 直接比较。

下一步应 pin vLLM/SGLang commit、模型/tokenizer revision、driver/GPU、TP/DP、grammar backend 和 speculative backend；准备 target-only、cold/warm shared prefix、strict JSON/regex、长上下文和 cancel workload。保存 request-level JSONL（候选树、accepted/rejected、grammar compile/mask、rollback bytes、KV pages、preempt/swap、TTFT/ITL/TPOT/E2E p50/p95/p99），并运行 distribution/parse/schema/业务四层验收。只有在 exactness、schema、SLO 和容量合同都通过后，才可把某个 workload 的 speculative 开关设为默认。
