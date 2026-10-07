# 第34章 编译器与 Kernel 实验报告

- 日期：2026-10-07
- 实验：`labs/ch34_compiler_kernel_lab.py`
- 级别：L0，CPU-only，Python 3.10+ 标准库
- 目的：验证图捕获守卫、graph break、pointwise 融合、编译缓存命中、autotune 预算、reference/fused 数值一致性与安全回退
- 边界：这是可解释的协议 toy，不是 PyTorch Inductor、Triton、CUDA、XLA、MLIR 或任何 GPU/TPU 的性能保证

## 可复现命令

```bash
python3 -m py_compile labs/ch34_compiler_kernel_lab.py
python3 labs/ch34_compiler_kernel_lab.py \
  --output reports/ch34-compiler-kernel-default.json
python3 labs/ch34_compiler_kernel_lab.py \
  --dynamic-shape --unsupported-op --fail-after 1 \
  --output reports/ch34-compiler-kernel-failure.json
python3 tests/test_ch34_compiler_kernel_lab.py
```

## 默认实验配置

- backend：`inductor`（仅作为协议标签；脚本不导入 PyTorch）
- 图：`add -> mul -> relu`，16 个 `fp32` 元素，contiguous layout
- 代理资源：单节点每个 64 bytes，分别 16 FLOPs；融合前后使用固定带宽/算力模型
- 编译 cache：key 包含 toy compiler digest、backend、节点 metadata 和 config；同一进程第二次编译命中
- autotune 候选：block 64/128/256（第 4 个候选因 8 ms budget 被截断），warps/stages 一并写入证据

## 默认结果

| 项目 | 结果 |
| --- | ---: |
| graph break | 0 |
| 融合组 | 1（3 个 pointwise 节点） |
| cold launch 数 | 1（分离图为 3） |
| cold compile ms（代理） | 4.6 |
| hot compile ms | 0 |
| hot steady ms（代理） | 0.012001 |
| cold bytes（代理） | 128 |
| FLOPs | 48 |
| 第二次 cache hit | true |
| max absolute error | 0.0 |
| autotune best | block=64, warps=1, stages=1 |

`N·L + B/M + F/P` 的代理模型只验证方向：pointwise 融合减少中间写回和 launch，热路径消除了编译成本。数字故意很小，不能外推 GPU HBM、SM occupancy、Triton 编译时间或真实吞吐。

## 失败/边界实验

命令加入 `--dynamic-shape --unsupported-op --fail-after 1`：

- guard 增加 `dynamic_shape=true`；
- `custom` side-effect 节点产生 `custom:unsupported_or_side_effect` graph break；
- cache 仍在同一进程第二次命中，但 break 不会自动消失；
- 第一个 autotune 候选被标为失败，选择 `reference_fallback`，`safe_fallback=true`；
- pointwise 子集仍做 reference/fused 比较，误差为 0；reduction/unsupported op 在 toy reference 中显式拒绝，避免假装已实现。

“有 graph break”与“程序不能运行”不同：生产编译器可能保留多个子图并回到 eager。toy 只给出 break 证据与安全路径，不模拟 Python resume、真实设备同步或二进制加载。

## 测试合同

`tests/test_ch34_compiler_kernel_lab.py` 覆盖：

1. 静态/动态 shape guard 与 unsupported side-effect break；
2. 仅相邻 pointwise 节点融合，reduction 保持边界；
3. compile key 确定性与第二次 cache hit；
4. 融合减少 launch、cold compile 只在首轮出现；
5. reference 与 fused 的数值误差；
6. autotune budget、确定性及失败候选安全回退；
7. `simulate()` summary 与 CLI stdout/`--output` JSON 一致。

## 生产测量边界

真实 PyTorch 2.x、Triton/CUDA、XLA/MLIR 验收还需要：固定 release/commit、driver、GPU/TPU 型号、compute capability、batch/sequence shape 分布、dtype/layout、warm-up、CUDA synchronize、并发、allocator、编译 cache、p50/p95/p99、寄存器/shared-memory/occupancy、dram bytes、NaN/Inf、梯度和回滚。CPU toy 通过只表明协议与测试合同自洽，不能声称某个 kernel 更快或跨设备可移植。

## 来源

- PyTorch compiler：https://pytorch.org/docs/stable/torch.compiler.html
- TorchInductor：https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html
- Triton：https://triton-lang.org/main/index.html
- CUDA C Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- StableHLO：https://github.com/openxla/stablehlo
- MLIR：https://mlir.llvm.org/docs/
- XLA：https://openxla.org/xla
- FlashAttention：https://arxiv.org/abs/2205.14135
- Ansor autotuning：https://arxiv.org/abs/2006.06762

