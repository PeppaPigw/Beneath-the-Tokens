# 第30章 KV compression + tiering CPU toy 实验报告

- 实验日期：2026-10-07 UTC
- 实验等级：L0，CPU-only、Python 标准库；不是 GPU/CUDA、真实模型、SSD、CXL 或生产 benchmark
- 脚本：`labs/ch30_kv_compression_lab.py`
- 测试：`tests/test_ch30_kv_compression_lab.py`
- 输出：`reports/ch30-kv-compression.json`
- 随机种子：30；模拟 latency 是 toy 成本函数，不是 wall-clock

## 实验问题和协议

本实验回答四个可复现问题：

1. per-group quantization 的误差是否可计算且 deterministic？
2. scale、zero-point、低秩和稀疏 index 加入后，实际 page bytes 是否偏离简单 bit ratio？
3. COLD SSD INT4 → WARM DRAM INT8 → HOT HBM FP16 的 online promotion/demotion/admission 是否产生可观察事件？
4. 在固定访问流中，hit、miss、eviction、quality loss 与 p50/p95 是否可由同一 JSON 重建？

`quantize` 使用有限向量做对称 per-group round/dequant，返回 MSE 和 max error。`compressed_bytes` 计算 K/V payload、scale/zero-point metadata、rank 系数和 sparse index。`simulate` 用 Zipf-like page 选择，首次 miss 按 admission 放入 SSD；二次、三次命中分别触发 DRAM/HBM promotion，容量不足时向下 demote 或 eviction。参数是教学模拟，不是硬件声明。

## 命令

```bash
python3 labs/ch30_kv_compression_lab.py \
  --seed 30 --requests 64 --pages 48 --accesses 600 --page-tokens 32 \
  --hbm-bytes 12000000 --dram-bytes 6000000 --ssd-bytes 40000000 \
  --admission-threshold 0.20 \
  --output reports/ch30-kv-compression.json
python3 tests/test_ch30_kv_compression_lab.py
```

## 结果（toy）

下表来自 `reports/ch30-kv-compression.json`；由于实验代码和参数固定，重新运行应得到相同 JSON。

| 指标 | 结果 |
| --- | ---: |
| access 次数 | 600 |
| page 数 / page tokens | 48 / 32 |
| baseline FP16 all-pages bytes | 201,326,592 |
| resident bytes（各 tier 合计） | 51,838,976 |
| hit rate | 0.846667 |
| HBM/DRAM/SSD hits | {'dram': 36, 'hbm': 46, 'ssd': 426} |
| admissions / promotions / demotions / evictions | 92 / 462 / 912 / 53 |
| latency p50 / p95 | 3.628 / 6.0 ms |
| mean / p95 quality loss | 0.0675 / 0.09 |

`resident_bytes`、命中和 eviction 数量受 page bytes 与容量参数约束，报告故意把逐次运行值保存在 JSON，而不把 toy 常数写成生产结论。若修改 seed、容量或访问数，必须同时更新命令、JSON 和本表。

## 观察和解释

1. **metadata 改变压缩比。** HBM FP16 页包含 2-byte scale metadata；DRAM INT8 和 SSD INT4 的 bytes/page 不是理论的 1/2 或 1/4。rank/sparse 还会增加 basis/index 成本。
2. **冷层 admission 是一次决策。** 第一次访问的 miss 计入 source latency；只有 score 达到阈值才进入 SSD。调高 threshold 可以减少污染，却可能让重复页每次重算。
3. **晋升会支付双写窗口。** 从 SSD 到 DRAM/HBM 时，旧页在目标页可读前仍占空间；toy 的 `place` 在容量检查后才提交 tier，真实系统还需 generation、checksum 和引用计数。
4. **质量和延迟各有 tier 曲线。** toy 用 `plan_quality_loss` 让 FP16/INT8/INT4 有可观察差异，SSD 读取延迟较高；这不是模型 perplexity 或真实 SSD I/O。
5. **hit rate 不能单独决策。** 应按 `hits_by_tier`、latency 分位数、quality loss、evictions、recompute 和 bytes 一起画 Pareto 前沿。

## 合同测试

`python3 tests/test_ch30_kv_compression_lab.py` 覆盖：

- quantize 在相同输入/参数下可复现，误差非负；
- sparse reconstruction 的非零数与 keep ratio、rank tail error 的边界；
- INT4+rank page bytes 小于 FP16，合成质量损失随 bit 降低；
- 固定 seed 的 tier simulation 完全相同，包含命中、miss、admission 和 p95；
- 极小 tier 容量触发 eviction/无法晋升路径；
- CLI stdout 与 `--output` JSON 相同，`schema_version=1`。

测试是压缩账本和状态机合同，不是 CUDA 数值稳定性、kernel correctness、SSD durability、真实模型质量、租户隔离或生产 readiness 证明。

## 建议扫描

- page tokens 8/16/32/64/128：比较 metadata、padding 和 promotion 粒度；
- HBM 容量 0.25×/0.5×/1× 工作集：记录 hit、p95、eviction、recompute；
- admission threshold 0/0.2/0.5/1：观察扫描流量污染与重复访问收益；
- Zipf 与近似均匀访问：测试 policy 是否依赖热点；
- bits 4/8/16、group 32/64/128、rank 2/4、sparsity 0.25：计算 metadata break-even；
- 将 `quality_loss` 替换为真实 shadow logits/KL，并按 JSON/工具调用单独验收。

## 复现边界和下一步

toy 不包含真实 K/V tensors、attention、RoPE、tokenizer、CUDA/FP8/INT4 kernel、NUMA、CXL、SSD queue、远端 connector、租户加密或取消竞态。`plan_quality_loss` 是合成函数，不能解释为 perplexity、准确率或安全指标。下一步应 pin 模型/量化库/硬件/driver，保存逐页 format、scale、basis、checksum、generation、tier、migration 和质量 trace；运行 FP16 shadow、长上下文、JSON/代码、混合格式 batch、远端故障和回滚演练。只有在质量、容量、p99 和故障合同都通过后，才可以把某一 tier policy 设为默认。
