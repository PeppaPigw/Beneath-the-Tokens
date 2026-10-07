# 第33章 存储与数据平面实验报告

- 日期：2026-10-07
- 实验：`labs/ch33_storage_data_plane_lab.py`
- 级别：L0，CPU-only，Python 3.10+ 标准库
- 目的：验证 shard 唯一覆盖、对象/POSIX/NVMe 延迟代理的方向，以及 checkpoint 在 commit marker 前失败时不可恢复
- 边界：这是透明的协议 toy，不是 S3、Lustre、CephFS、BeeGFS、NVMe、SPDK 或任何云厂商的性能/耐久性保证

## 可复现命令

```bash
python3 labs/ch33_storage_data_plane_lab.py \
  --shards 12 --workers 3 --shard-mib 4 --checkpoint-step 100 \
  --output reports/ch33-storage-data-plane-default.json

python3 labs/ch33_storage_data_plane_lab.py \
  --shards 12 --workers 3 --shard-mib 4 --checkpoint-step 100 \
  --fail-after 8 --output reports/ch33-storage-data-plane-failure.json

python3 tests/test_ch33_storage_data_plane_lab.py
```

## 环境与 workload

- seed：33（round-robin toy 不使用随机打乱，但保留 seed 作为审计字段）
- shard 数：12；worker 数：3；每 shard 4 MiB；总 payload 48 MiB（50,331,648 bytes）
- 读模型：`elapsed_ms = objects × latency_ms + objects × shard_mib / bandwidth_mb_s × 1000`
- checkpoint：step 100；完整 run 写 12/12 shard；故障 run 用 `--fail-after 8`，只写 8/12
- backend proxy：object 180 MB/s + 8 ms/request；posix 420 MB/s + 1.4 ms/request；nvme 1200 MB/s + 0.08 ms/request

## 默认测量结果

| 后端 | objects | bytes | elapsed_ms | throughput_mb_s |
| --- | ---: | ---: | ---: | ---: |
| object | 12 | 50,331,648 | 362.666667 | 132.352941 |
| posix | 12 | 50,331,648 | 131.085714 | 366.172624 |
| nvme | 12 | 50,331,648 | 40.960000 | 1171.875000 |

Round-robin 分配为 worker-0 `[0,3,6,9]`、worker-1 `[1,4,7,10]`、worker-2 `[2,5,8,11]`；`coverage=12` 且 `all_shards_unique=true`。完整 checkpoint 写入 12 个 shard，manifest hash 为 `aefe819c5a97915381a77a84b98296ad585cab904ea6694e716055bed39d7d6f`，`committed=true`、`recoverable=true`。

代理方向为 NVMe 最快、POSIX 次之、object 最慢。这只反映本实验设置的启动延迟和带宽，不能外推真实设备；真实测量还需固定缓存冷热、并发、块大小、队列深度、NUMA、网络、服务区域和版本。

## commit 前故障结果

`--fail-after 8` 产生 `written_shards=8`、`committed=false`、`recoverable=false`，manifest hash 为 `393e297498fbf957049239d88bb7dd26568690354583b248c74d48f11dc3ac37`。脚本明确输出 `uncommitted_write_is_visible=true`，含义是 toy reader 可以观察到有写入痕迹，但协议要求忽略这个前缀并回滚到上一个已提交版本；它不表示任意对象服务的 list/read-after-write 语义。

## 测试合同

`tests/test_ch33_storage_data_plane_lab.py` 覆盖：

1. shard round-robin 的确定性、无重叠和覆盖；
2. 固定 payload 下 object 与 NVMe 的延迟/吞吐方向；
3. 完整 checkpoint 的 manifest commit 与失败 checkpoint 的拒绝；
4. `simulate()` 的 seed 重现、失败字段和最快后端；
5. CLI stdout 与 `--output` JSON 完全一致。

## 机制解释

1. `N·L + S/B` 把小对象启动成本与 payload 传输成本分开；当对象数增加时，降低 metadata 请求数量往往比换更快介质有效。
2. shard assignment 只证明 toy 的唯一覆盖；真实 distributed sampler 还要记录 epoch、world size、shuffle seed、drop_last、cursor 和重复率。
3. checkpoint 先写 shard hash，再生成 manifest，最后才算 committed；文件/对象存在不等于可恢复版本。
4. manifest hash 是内容身份代理。生产系统应哈希实际 payload，并验证 size、dtype、shape、版本 pointer、权限和跨故障域复制。

## 失败解释与生产边界

- 脚本没有创建大文件、访问云、调用 fsync、模拟掉电、模拟 MDS/OST/NVMe 控制器或真实网络拥塞。
- `list_delay_ticks` 在当前默认路径仅是配置证据，实验没有伪造真实 eventual consistency；生产必须按目标对象服务的官方一致性契约与现场测试验收。
- toy 的 `recoverable=false` 只表示缺少全部 shard/commit marker；真实恢复还需检查 optimizer、RNG、sampler cursor、world size、代码 digest 和跨区域副本。
- throughput proxy 是十进制 MB/s 计算；不要把 1171.875 MB/s 写成目标 NVMe 的保证。

## 参考证据

- [Amazon S3 API](https://docs.aws.amazon.com/AmazonS3/latest/API/Welcome.html)
- [Amazon S3 数据一致性](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)
- [Linux fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html)
- [Linux rename(2)](https://man7.org/linux/man-pages/man2/rename.2.html)
- [Linux block layer](https://docs.kernel.org/block/index.html)
- [NVM Express specifications](https://nvmexpress.org/specifications/)
- [fio](https://github.com/axboe/fio)
- [Lustre manual](https://doc.lustre.org/lustre_manual.xhtml)
- [BeeGFS documentation](https://doc.beegfs.io/latest/)
- [CephFS documentation](https://docs.ceph.com/en/latest/cephfs/)
- [NFS RFC 8881](https://www.rfc-editor.org/rfc/rfc8881)
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
- [DeepSpeed model checkpointing](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)
- [Megatron-LM paper](https://arxiv.org/abs/2104.04473)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [MegaScale paper](https://www.usenix.org/system/files/nsdi24-jiang-ziheng.pdf)
- [vLLM documentation](https://docs.vllm.ai/en/stable/)
- [TensorRT-LLM disaggregated serving](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html)
- [OpenTelemetry traces](https://opentelemetry.io/docs/concepts/signals/traces/)
- [Prometheus histograms](https://prometheus.io/docs/practices/histograms/)

## 复现限制

需要真实生产结论时，锁定对象区域/API、文件系统和 kernel/client 版本、NVMe 型号、CPU/NUMA、队列深度、数据大小、冷热缓存、并发租户、失败注入时间和清理策略；同时保留 request id、object version/inode、size/hash、fsync/commit 时刻、lease epoch、worker/rank、重试和 p50/p95/p99。toy 通过不代表生产可用。
