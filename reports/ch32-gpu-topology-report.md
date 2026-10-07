# 第32章 GPU 集群拓扑实验报告

- 日期：2026-10-07
- 实验：`labs/ch32_gpu_topology_lab.py`
- 目的：在无 GPU 的条件下验证 rank→node 映射、all-to-all 字节守恒、ring all-reduce 跨节点边界，以及 GPU/节点/交换机故障对同步训练和副本推理的不同影响
- 边界：Python 标准库 CPU toy；不代表任何 GPU、NVLink/NVSwitch、PCIe、InfiniBand、RoCE、NCCL、RCCL 或交换机的吞吐、延迟和恢复保证

## 可复现命令

```bash
python3 labs/ch32_gpu_topology_lab.py \
  --nodes 2 --gpus-per-node 4 --payload-mib 64 --requests 128 \
  --replicas 2 --output reports/ch32-gpu-topology-default.json

python3 labs/ch32_gpu_topology_lab.py \
  --nodes 2 --gpus-per-node 4 --payload-mib 64 --requests 128 \
  --replicas 2 --failure-scope node --failure-target 0 \
  --output reports/ch32-gpu-topology-node-failure.json

python3 tests/test_ch32_gpu_topology_lab.py
```

## 默认拓扑结果

| 指标 | 结果 |
| --- | ---: |
| 节点 × 每节点 GPU | 2 × 4 |
| world size | 8 |
| 每 rank aggregate payload | 64 MiB |
| all-to-all inter-node bytes（全部 rank 合计） | 306,783,380 |
| all-to-all bottleneck proxy | 0.391479 ms |
| ring all-reduce inter-node bytes | 234,881,024 |
| ring bottleneck proxy | 0.343544 ms |
| 配置副本数 / 可用副本数 | 2 / 2 |
| 推理丢弃率 | 0 |

整数除法产生的 peer remainder 会按确定性顺序分配，所有 rank 的发送字节仍等于 64 MiB。ring 计划显式展开 reduce-scatter 和 all-gather 两次遍历；跨节点边的数量取决于 rank 顺序。proxy 使用配置中的 900 Gbit/s 节点内和 200 Gbit/s 节点间链路，并加上 2/8 微秒启动延迟。

## 节点故障结果

| 指标 | 结果 |
| --- | ---: |
| 注入 | `scope=node`, `target=0` |
| 受影响 rank | 0,1,2,3 |
| training collective survives | false |
| 可用推理副本 | 1（配置 2） |
| requests=128 的 dropped | 0 |
| inference drop rate | 0 |

同步训练要求每个 rank 都参加 collective，因而节点内任一 rank 消失都应中止或重建作业；副本推理只要另一故障域仍有完整副本即可继续。`switch` 故障会把所有节点标为不可用并使 128 个请求全部 dropped；这只是本 toy 的全局 fabric 语义，生产系统要按 ToR/spine/zone 的真实连接图建模。

## 测试合同

`tests/test_ch32_gpu_topology_lab.py` 覆盖：

- rank 到 node 的确定性映射、self/intra-node/inter-node 路径分类；
- all-to-all 每 rank payload 守恒、总字节和 bottleneck 计算；
- ring all-reduce 两阶段边数、跨节点字节和非零延迟；
- GPU、node、switch 故障范围和副本可用性；
- seed/配置重现、failure JSON 字段和 CLI 输出文件一致性。

## 解释与生产边界

1. `intra_bw_gbps`、`inter_bw_gbps` 和 latency 是可修改的透明参数，不是硬件规格。实验没有 CUDA kernel、DMA、NCCL communicator、RDMA 注册、PCIe replay、NVLink counter、交换机队列、PFC/ECN、重传或真实故障恢复。
2. all-to-all 只发送均匀 aggregate payload；真实 MoE、KV 迁移或梯度交换会有 padding、header、压缩、重试和负载倾斜。
3. ring 计划按逻辑 rank 顺序连接相邻 rank；NCCL/RCCL 会基于实际拓扑、算法、协议和消息大小选择不同方案，必须用目标版本的 `nccl-tests`、应用 trace 和 NIC/NVML 计数器复核。
4. 节点故障下“副本可用”不等于请求一定成功。生产 serving 还要验证模型并行组完整性、KV cache、幂等重试、健康探针和跨 ToR/机架放置。
5. 真实训练恢复需保留 rank mapping、world size、optimizer/data sampler、版本 digest 和 checkpoint checksum；不能只重启缺失进程。

## 参考证据

- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NCCL networking troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html)
- [NCCL RAS and observability](https://developer.nvidia.com/blog/networking-reliability-and-observability-at-scale-with-nccl-2-24/)
- [nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [PyTorch distributed](https://pytorch.org/docs/stable/distributed.html)
- [Kubernetes Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [Slurm GRES](https://slurm.schedmd.com/gres.html)
- [MegaScale NSDI 2024](https://www.usenix.org/system/files/nsdi24-jiang-ziheng.pdf)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
