# 第31章 MoE Routing / Expert Parallel Toy Report

- 日期：2026-10-07
- 实验：`labs/ch31_moe_parallelism_lab.py`
- 目的：在无 GPU 的条件下检查 top-k、capacity/overflow、aux load-balance proxy、EP send/recv conservation 和 hot-expert rank skew
- 边界：标准库 CPU toy；不代表 CUDA/NCCL、DeepSpeed、Megatron-Core、vLLM 或 SGLang 的吞吐/延迟

## 可复现命令

```bash
python3 labs/ch31_moe_parallelism_lab.py --seed 31 --tokens 256 --experts 8 --ranks 4 --top-k 2 --capacity-factor 1.25 --overflow-policy drop --hidden-bytes 4096 --output reports/ch31-moe-routing-default.json
python3 labs/ch31_moe_parallelism_lab.py --seed 31 --tokens 256 --experts 8 --ranks 4 --top-k 2 --capacity-factor 1.25 --overflow-policy drop --hidden-bytes 4096 --hot-expert 0 --hot-bias 8 --output reports/ch31-moe-routing.json
python3 tests/test_ch31_moe_parallelism_lab.py
```

## 结果摘要

| 场景 | capacity | accepted assignments | overflow assignments | dropped tokens | aux proxy | all-to-all bytes | latency proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed-31 均匀输入 | 80 | 512 | 0 | 0 | 2.008816 | 2,097,152 | 1.327500 ms |
| expert-0 hot bias=8 | 80 | 336 | 176 | 176 | 2.495661 | 1,376,256 | 0.895402 ms |

均匀场景的最大 expert load 为 76，最小为 49；hot 场景的 expert-0 load 达到容量 80，而 expert-6 只有 21。hot 场景的 drop 下降了通信字节和 toy latency proxy，这不是性能收益：它代表大量 assignment 未进入 expert，质量必须由 residual/fallback 合同解释。将 overflow policy 改为 `second` 时，在同一 hot 输入上 token-level `drop_rate` 可为 0，但仍有 176 个 assignment overflow，accepted assignment 只有 336；因此报告必须同时记录 assignment 和 token 两个分母。

## 证据与解释

- `routing.expert_prob` 是 router softmax 概率均值，`selected_fraction` 是实际 accepted fraction；`aux_loss = E * sum(f_e * P_e)`，用于方向性比较。
- `all_to_all.send_matrix` 和 `recv_matrix` 的元素总和均等于 accepted assignments；bytes 按 4096 hidden bytes 计算。
- `latency_proxy_ms` 是 `compute + max(bytes) / 50 MiB/s + skew penalty` 的透明 toy 公式，不能用于硬件容量规划。
- hot bias 只改变 synthetic logits，不模拟真实模型语义；没有质量指标或 GPU kernel。

## 测试合同

`tests/test_ch31_moe_parallelism_lab.py` 覆盖 softmax 稳定性、capacity 上界和 overflow、second-choice 不重复 expert、send/recv conservation、seed 重现和 CLI JSON 输出。真实部署还需补充单步 GPU golden、跨节点 collective、专家权重 checksum、取消/timeout 清理、量化和端到端质量测试。
