# 第 27 章 KV connector CPU toy 实验报告

- 实验日期：2026-10-07
- 代码：`labs/ch27_kv_connector_lab.py`
- 测试：`tests/test_ch27_kv_connector_lab.py`
- 实验等级：L0（CPU-only，标准库）；不是网络、GPU、RDMA 或模型质量 benchmark
- Python：3.11（运行时以 `python3 --version` 的实际输出为准）
- 随机种子：7、11（报告中的 p95/重试受 seed 影响）

## 命令

```bash
python3 labs/ch27_kv_connector_lab.py \
  --seed 7 --requests 20 --tokens 1024 --page-tokens 128 \
  --connectors memcpy,tcp,nixl,mooncake,lmcache \
  --lmcache-hit-rate 0.25 --output reports/ch27-kv-baseline.json
python3 labs/ch27_kv_connector_lab.py \
  --seed 11 --requests 50 --tokens 2048 --page-tokens 128 \
  --lmcache-hit-rate 0 --connectors tcp,nixl,mooncake
python3 tests/test_ch27_kv_connector_lab.py
```

## 实验模型

默认 `KVShape` 为 32 layers、8 KV heads、head dimension 128、2 bytes/element。理想单请求字节为：

`2 × 32 × tokens × 8 × 128 × 2`

脚本把每 128 token 切成一个 Page，生成稳定的 SHA-256 摘要而不分配完整 payload。profile 只设置教学用 bandwidth、setup、per-page overhead 和 loss rate；`bytes_sent` 计逻辑 payload，不等价于真实重传 wire bytes。命中请求直接进入 `CACHE_HIT → COMMITTED`，表示页已在本地，不模拟真实索引、租户 ACL、磁盘或 GPU。

## 观察与解释

1. token 数翻倍时理想 KV bytes 与 page payload 总和翻倍；这是公式与分页实现的测量，不是某个模型布局的生产保证。
2. page size 变小会增加页数和固定每页开销；page size 变大减少通知次数，却让最后一页浪费和取消粒度变粗。
3. toy profile 的相对 p50/p95 受固定 setup、带宽和随机丢包影响。它可用于验证“重试上界、命中减少字节、状态按顺序提交”，不能用于声称 NIXL/Mooncake/LMCache 的真实 GB/s。
4. 提高 `--lmcache-hit-rate` 会减少 bytes_sent 和 toy elapsed；请求数和逻辑页数不变。生产命中还需要模型、tokenizer、layout、租户和 lease 检查。
5. 把 TCP profile 的 loss rate 设高或把 max_retries 设低会得到 `ABORTED`/`checksum_mismatch`，用于演示失败状态。脚本的随机丢包是随机数，不模拟 TCP 拥塞控制或 RDMA 重传。

## 契约测试

当前测试覆盖：

- KV formula 对 token 线性；
- Page token_count 覆盖准确且 payload 总和等于公式；
- 成功传输状态为 INIT → NEGOTIATED → MEMORY_REGISTERED → TRANSFERRING → COMMITTED；
- 重试不超过 `max_retries × pages`；
- 100% cache hit 不发送字节；
- CLI 输出可解析 schema_version=1 的 JSON，且 `--output` 文件与 stdout 相同。

在本环境未安装 pytest 时，直接执行 `python3 tests/test_ch27_kv_connector_lab.py` 会运行内置断言并打印 `ch27 tests: PASS`。若 CI 安装 pytest，再执行 `pytest -q tests/test_ch27_kv_connector_lab.py`。

## 限制和下一步

- 没有真实 GPU、CUDA stream/event、NUMA、NIC、RDMA、NVLink、SSD 或对象存储；不能证明内存可见性、驱动兼容和硬件带宽。
- 没有真实 tokenization、模型质量、prefix hash 冲突、租户 ACL、加密、删除传播或 side-channel 认证。
- 单线程顺序模拟没有真实并发、batch、拥塞、优先级、公平和取消竞争。
- `bytes_sent` 未计算重传 wire bytes，p95 是 toy 的排序分位数，不是服务 SLO。

下一步应在目标版本和硬件上做最小真实页 read/write、descriptor 注册、端到端 post→ready、agent 掉线、checksum 错误、客户端取消和 lease 过期演练；用目标请求长度、缓存命中、租户策略和成本边界重建报告。论文或官方文档中的 headline 吞吐只应作为实验设计线索，并随硬件、驱动、拓扑和负载报告条件。
