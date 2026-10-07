# 第 28 章 vLLM 与 SGLang serving CPU toy 实验报告

- 实验日期：2026-10-07
- 实验等级：L0，CPU-only、Python 标准库；不是 vLLM/SGLang、GPU、CUDA、网络或模型质量 benchmark
- 代码：`labs/ch28_vllm_sglang_lab.py`
- 测试：`tests/test_ch28_vllm_sglang_lab.py`
- 运行时：以 `python3 --version` 的实际输出为准；本次在 Python 3.12 环境执行
- 随机种子：7；脚本输出的 `elapsed_ms` 是离散逻辑时间，不是 wall-clock 硬件时间

## 实验问题和模型

实验回答四个可复现的小问题：

1. 同样的输入输出长度下，固定页（paged）和共享前缀 radix cache 对 prefill token、TTFT、页数有什么影响？
2. `max_batch_tokens` 和并发如何影响调度步数与尾延迟？
3. speculative draft/verify 在接受率为 0.75 时是否减少逻辑步骤？
4. 增加 grammar rejection 是否在输出 token 数不变时增加逻辑耗时？

每个请求有 input token、output token、arrival time、KV page 引用和状态 `WAITING → PREFILL → DECODING → FINISHED`。每一轮先为运行请求保留 decode 预算，再将剩余 token budget 分配给 prefill。`engine=paged` 不做前缀命中；`engine=radix` 在第一个请求提交公共前缀后，对后续请求做 synthetic longest-prefix hit。页数按 `ceil((input+output)/page_tokens)` 估算，省略真实 allocator、权重、logits、activation、CUDA graph 与 staging buffer。

speculative 每轮生成最多 `draft_tokens` 个候选，按 `accept_rate` 采样接受数，再由 target 提交接受前缀加一个校正 token。grammar rejection 只增加检查/重试逻辑时间，不改变最终输出 token 数。所有随机数由一个 `random.Random(seed)` 驱动，便于比较相同 seed 的基线和干预。

## 命令

```bash
python3 labs/ch28_vllm_sglang_lab.py \
  --seed 7 --requests 40 --concurrency 8 \
  --input-len 256 --output-len 64 --shared-prefix 0 \
  --engine paged --page-tokens 16 --max-batch-tokens 128 \
  --output reports/ch28-paged-baseline.json

python3 labs/ch28_vllm_sglang_lab.py \
  --seed 7 --requests 40 --concurrency 8 \
  --input-len 256 --output-len 64 --shared-prefix 128 \
  --engine radix --page-tokens 16 --max-batch-tokens 128 \
  --output reports/ch28-radix-shared-prefix.json

python3 labs/ch28_vllm_sglang_lab.py \
  --seed 7 --requests 40 --concurrency 8 \
  --input-len 256 --output-len 64 --shared-prefix 128 \
  --engine radix --page-tokens 16 --max-batch-tokens 128 \
  --speculative --draft-tokens 4 --accept-rate 0.75 \
  --grammar-reject-rate 0.10 --output reports/ch28-combined.json

python3 tests/test_ch28_vllm_sglang_lab.py
```

## 结果

| 场景 | prefill（计算）token | cache hit token | TTFT p50/p95/p99 ms | E2E p50/p95/p99 ms | logical output tok/s | speculative accepted | grammar rejects |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| paged cold | 10,240 | 0 | 199.128 / 394.756 / 401.410 | 278.016 / 459.050 / 461.170 | 5,324.90 | 0 | 0 |
| radix shared 128 | 5,248 | 4,992 | 187.304 / 370.832 / 373.922 | 263.496 / 435.552 / 436.202 | 5,616.60 | 0 | 0 |
| radix + spec + grammar | 5,248 | 4,992 | 173.882 / 358.628 / 380.342 | 233.084 / 399.212 / 407.812 | 5,985.48 | 1,921 | 257 |

三个 JSON 文件保留了 40 条请求明细、每请求 output、pages、draft、accept、grammar 和状态。radix 场景的 4,992 个命中 token 来自合成的 128-token shared prefix，不是模型真实 tokenizer 命中。组合场景 accepted tokens 小于 draft tokens，且 grammar rejects 增加；output token 总数仍为 2,560，表示 toy 中 grammar 检查不改变目标输出长度。

## 观察与机制解释

1. **Radix 命中减少 prefill。** shared-prefix 场景将计算的 prefill token 从 10,240 降到 5,248，TTFT p50 从 199.128 降到 187.304。逻辑时间模型把前缀查找成本设为每个 prefill item 的固定开销，因此收益没有按 token 数线性放大。真实 radix cache 还要承担节点分裂、锁、引用、淘汰和版本/租户校验。
2. **组合策略减少逻辑步骤，但不证明绝对加速。** speculative + accept 0.75 让 steps 从 338 降到 106；toy output tok/s 上升到 5,985.48。该值由人为设置的 draft、verify 和 grammar 常数产生，不能解释成 GPU throughput。若把 accept-rate 降到 0.2，额外 draft/verify 可能使 E2E 变差，真实引擎应按 acceptance 与尾延迟自适应回退。
3. **Grammar rejection 会抬高逻辑耗时。** 组合场景有 257 次 rejection，p99 TTFT 与 p95 的差距相对 radix baseline 变大；它模拟 token mask/retry 的控制面成本。toy 不实现 JSON parser，因此不能证明 schema 合法性。
4. **页数不等于总显存。** 三个场景 allocated_pages 都约为 800（组合场景 max live 154），因为请求长度和 page size 相同。真实服务还要加入权重、临时 logits、activation、allocator reserved、draft buffer、grammar cache、通信 staging 和碎片；KV usage 低不能排除 OOM。

## 合同测试

测试覆盖：

- 所有请求最终完成，调度 step 不会因为 `max_batch_tokens` 不收敛；
- radix synthetic prefix 的 prefill 计算 token 少于 cold baseline；
- paged engine 不伪造 radix hit；
- accepted token 不超过 draft token；
- grammar rejection 在相同 seed 下可复现且非零；
- CLI stdout 与 `--output` 文件拥有相同 `schema_version=1` JSON。

运行 `python3 tests/test_ch28_vllm_sglang_lab.py` 输出 `ch28 tests: PASS`。若 CI 安装 pytest，可用 `pytest -q tests/test_ch28_vllm_sglang_lab.py`。测试只证明 toy contract，不证明框架 API、CUDA event、数值一致性、吞吐、SLO、隐私或生产 readiness。

## 故障注入和进一步实验

- 把 `--shared-prefix 0` 与 `128` 配对，观察命中 token 和 TTFT；再把 `--engine paged` 作为对照，避免把 page allocator 的效果误归因于 radix。
- 把 `--max-batch-tokens` 从 64、128、256 扫描，分别记录短请求和长请求 p95/p99；用 `--concurrency` 1、8、32 观察 admission 和队列，不只看总 tok/s。
- 把 `--accept-rate` 设为 0.2、0.5、0.9，比较 draft token、accepted token、steps 和 E2E；低接受率时应考虑关闭 speculative。
- 把 `--grammar-reject-rate` 设为 0、0.1、0.5，观察 rejects 与逻辑时间；真实验证需要 grammar engine、tokenizer 和完成时 schema validator。
- 把 `--max-pages` 设为很小的值，观察 preemption 计数；toy 的回收策略只为说明水位与抢占关系，不等价于 vLLM swap/recompute。

## 可复现边界

这是 L0 离散模拟：没有真实模型 logits、tokenizer/chat template、GPU kernel、PagedAttention memory visibility、RadixAttention 节点锁、CUDA graph、TP/DP/NCCL、RDMA/NVLink、HTTP/TLS、客户端重试、autoscaling 或工具调用。随机接受和 grammar rejection 是 Bernoulli toy，不是语言分布。`output_tokens_per_s` 使用逻辑时钟，不能与 vLLM/SGLang benchmark 或论文 GB/s 比较。

下一步真实实验应 pin vLLM/SGLang tag、模型/tokenizer revision、GPU/驱动、TP/DP、page/radix 配置和 sampling；分别跑 cold random、warm shared-prefix、strict JSON 三类 workload；保存 request-level JSONL、TTFT/ITL/E2E p50/p95/p99、错误/取消、KV eviction、preempt/swap、speculative acceptance、grammar compile 和 GPU reserved/allocated。结论必须写明硬件和负载，禁止把 toy 或论文 headline 当作生产容量承诺。
