---
id: ch07-numerical-precision
title: 数值计算与精度：FP32/BF16/FP16/FP8/INT8/INT4
slug: /chapters/07-numerical-precision
description: 解释低精度格式、舍入、稳定性、loss scaling、量化校准、误差传播以及内存吞吐权衡，并用 CPU 实验验证误差与性能
sidebar_position: 7
level: systems
prerequisites:
  - ch04-performance-math
  - ch06-pytorch-execution
learning_objectives:
  - 能读懂 FP32、BF16、FP16、FP8、INT8、INT4 的位布局、范围和精度差异
  - 能区分表示误差、舍入误差、溢出/下溢、截断误差与累积误差
  - 能解释 autocast、混合精度、loss scaling 以及 FP8 的 amax/scaling 流程
  - 能设计静态、动态、逐通道和逐组量化，并选择校准数据和误差指标
  - 能估算低精度对内存、带宽、吞吐、转换开销和端到端延迟的影响
  - 能运行 CPU-only 量化实验，报告误差分布、饱和率、校准敏感性和速度
  - 能诊断 NaN、inf、loss 不下降、精度回退和量化后长尾输出等故障
  - 能为低精度上线定义版本边界、回滚开关和安全边界
estimated_hours: 18
hardware: CPU-only baseline; CUDA/FP8 GPU optional
risk_level: L2
last_verified: 2026-10-05
---

# 第7章　数值计算与精度：FP32/BF16/FP16/FP8/INT8/INT4

> 低精度不是“把 float32 换成更小的类型”这么简单。格式决定可表示的范围和间隔，舍入规则决定每次运算的误差，内核决定累加在哪里完成，量化参数决定长尾是否被裁掉，而测量边界决定所谓“提速”是否只是把转换和校准成本藏起来。本章把数值格式、训练稳定性和量化部署放在同一张工程地图上。

## 7.1 先把“精度”拆成四个契约

### 7.1.1 存储精度、计算精度、累加精度、指标精度

一次矩阵乘法至少有四个相关但不同的精度：

1. **存储精度**：权重、激活或梯度在内存中使用多少位。它决定内存占用和带宽需求，也决定从内存读出时已经损失多少信息。
2. **计算精度**：乘法或逐元素运算的输入和中间结果使用什么格式。例如输入是 FP16，乘法可以在 FP16 或 Tensor Core 的内部格式完成。
3. **累加精度**：点积和归约把许多项相加时使用什么格式。FP16 乘法配 FP32 累加，通常比全 FP16 更稳定。
4. **指标精度**：最终 loss、准确率、BLEU、召回率或业务得分用什么精度和容差判断。指标是 FP32 并不保证前面的低精度误差已经可接受。

[事实] 一个框架的 `dtype` 往往描述张量的存储类型，不一定精确描述底层 kernel 的乘法和累加路径。要确认后者，需要看算子文档、硬件指令或 profiler 的 kernel 名称。

[机制] 将存储从 FP32 改成 BF16 可以减半带宽，但若每层都在 FP32 与 BF16 之间来回转换，转换 kernel 和额外读写可能抵消收益。应把转换成本纳入端到端测量。

[设计判断] 低精度改造的第一问不是“哪种 dtype 最快”，而是“哪一层的存储、计算、累加和指标契约可以放宽，放宽后谁负责检测错误”。

### 7.1.2 三种误差不要混为一谈

- **表示误差**：真实值无法在目标格式精确表示，例如十进制 `0.1` 在二进制浮点中通常是近似值。
- **运算舍入误差**：每次加、乘、除或指数运算后把无限精度结果舍入回有限格式。
- **量化误差**：把连续值映射到离散整数格点，通常还伴随裁剪（clipping）和零点（zero-point）偏移。

此外还有**建模误差**（例如近似激活函数）和**测量误差**（例如没有同步 GPU 就计时）。本章的实验尽量把数值误差和计时误差分开报告。

### 7.1.3 一份可追踪的精度报告

任何低精度实验都应记录：

```text
模型/算子与提交号：
输入 shape、分布、随机种子：
存储 dtype、计算 dtype、累加 dtype：
量化方案（对称/非对称、粒度、scale、zero-point）：
舍入模式和是否随机舍入：
校准数据集及覆盖范围：
误差指标（max、MAE、RMSE、相对误差、饱和率）：
速度边界（预热、同步、转换和校准是否计入）：
硬件、驱动、框架/编译器版本：
已知失效输入、回滚开关与安全阈值：
```

没有这份上下文，“INT8 提速 2 倍”或“FP8 精度下降 0.2%”都不可迁移。

## 7.2 浮点格式的位布局与可表示范围

### 7.2.1 统一公式

IEEE 风格二进制浮点可以写成：

\[
x = (-1)^s \times (1.f)_2 \times 2^{e-bias}
\]

其中 `s` 是符号位，`e` 是指数域，`f` 是尾数域。指数全 0 通常用于零和次正规数（subnormal），指数全 1 用于无穷和 NaN。格式的位数并不直接等于有效十进制位数；指数位多，范围大，尾数位多，间隔更细。

### 7.2.2 FP32（单精度）

FP32 使用 1 位符号、8 位指数、23 位显式尾数（含隐藏的最高位后约 24 位有效二进制精度）。最大有限值约为 `3.4e38`，最小正常正数约为 `1.18e-38`，机器 epsilon 在 1 附近约为 `1.19e-7`。

FP32 的优点是范围和精度均衡，长期作为优化器状态、loss、归约和参考结果的基线。代价是每元素 4 字节、带宽压力高，并不总能充分利用新一代矩阵单元。

### 7.2.3 FP16（IEEE half）

FP16 使用 1/5/10 位布局，约 11 位有效二进制精度；最大有限值约 `65504`，最小正常正数约 `6.10e-5`。这意味着大激活容易溢出，小梯度容易下溢。次正规数在部分硬件上会被 flush-to-zero，进一步缩小有效范围。

FP16 的 10 位尾数在 1 附近的间隔约 `2^-10≈9.77e-4`，相邻大数之间的间隔还会随指数变大。它适合矩阵乘法和带 FP32 累加的路径，但对 softmax、归一化、指数和梯度尤其敏感。

### 7.2.4 BF16（bfloat16）

BF16 使用 1/8/7 位布局，指数位与 FP32 相同，最大有限值同量级（约 `3.39e38`），但只有约 8 位有效二进制精度。它通常不容易因范围不足而溢出，却更容易因尾数短而产生舍入误差。

BF16 对训练很有吸引力：梯度和激活的动态范围接近 FP32，很多模型可以不使用 FP16 那样激进的 loss scaling；代价是矩阵乘法累加和归一化仍需要适当保留 FP32，且在需要高相对精度的小差值计算中可能不如 FP16。

### 7.2.5 FP8：E4M3 与 E5M2

常见 FP8 不是唯一格式。E4M3 使用 1 位符号、4 位指数、3 位尾数，动态范围较窄但相对精度较好，常用于前向激活和权重。E5M2 使用 5 位指数、2 位尾数，范围更大但精度更低，常用于反向梯度。不同硬件和库可能对 NaN、无穷和最大有限值的编码略有差异，必须以目标实现的文档为准。

FP8 几乎总要配合 scale：先以 FP32/BF16 计算张量的绝对最大值 `amax`，选择缩放因子 `s`，把 `x*s` 转成 FP8，kernel 计算后再乘 `1/s`。如果一整层共用一个 scale，少数 outlier 会迫使大多数值落在很小的有效区间；逐通道或逐块 scale 可减轻此问题，但会增加 scale 读写和元数据。

### 7.2.6 用代码观察范围与间隔

下面的 CPU 代码无需 GPU，可直接打印不同 numpy 浮点类型的最大值、最小正常值和在 1 附近的下一个可表示数：

```python
import numpy as np

for dt in [np.float32, np.float16, np.float64]:
    info = np.finfo(dt)
    one = dt(1)
    next_up = np.nextafter(one, dt(2), dtype=dt)
    print(dt.__name__,
          "max=", info.max,
          "tiny=", info.tiny,
          "eps=", info.eps,
          "ulp@1=", float(next_up - one))

# numpy 支持 bfloat16 的版本可能不同；没有该 dtype 时跳过
if hasattr(np, "bfloat16"):
    info = np.finfo(np.bfloat16)
    print("bfloat16", info.max, info.tiny, info.eps)
```

[故障模式] 只比较 `itemsize` 就推断数值质量，会忽略 exponent/mantissa 分配。BF16 和 FP16 都是 2 字节，但一个主要扩大范围，一个主要保留尾数精度；面对不同分布时结论相反。

## 7.3 舍入、稳定性与误差累积

### 7.3.1 round-to-nearest-even 与 tie

最常见的浮点舍入是“就近取整，半数取偶”（round-to-nearest-even）。当无限精度结果正好位于两个可表示数中间时，选择低位为偶数的那个，长期统计偏差通常小于总是向上或向零截断。

量化库还可能支持 toward-zero、floor、ceil 或 stochastic rounding。随机舍入按距离概率选择相邻两个格点，期望值更接近原数，在训练小梯度时可能有帮助，但会引入随机性，必须记录随机种子和并行实现，否则复现实验会困难。

### 7.3.2 灾难性消去与相对误差

当两个近似相等的大数相减，前面的有效位相互抵消，剩下的小差值只由低位误差决定。例如在低精度中计算 `sqrt(x+1)-sqrt(x)`，直接相减比有理化形式 `1/(sqrt(x+1)+sqrt(x))` 更不稳定。

相对误差 \(|\hat{x}-x|/|x|\) 在真实值接近零时会爆炸，因此同时报告绝对误差和相对误差。对于 logits、梯度和残差，建议给出分位数而不是只给 max，避免一个极端 outlier 掩盖主体行为。

### 7.3.3 归约与加法顺序

对 `n` 个同号数做朴素浮点求和，误差上界常与 `n·u` 成正比，其中 `u` 是单位舍入误差；树形归约和 Kahan/Neumaier 补偿求和可降低误差，但会增加操作和同步。

矩阵乘法通常在更高精度中累加。例如 FP16 输入、FP32 累加相当于把每个乘积先低精度化，再在较大范围的累加器中求和。若改成 FP16 累加，长向量和大 batch 更容易出现溢出或累加误差。GPU kernel 可能在 tile 内使用 FP32，再在写回时转回低精度，不能仅凭输出 dtype 断言累加精度。

### 7.3.4 softmax、logsumexp 与归一化

直接计算 `exp(logits)` 可能溢出。稳定 softmax 先减去最大值：

\[
\operatorname{softmax}(x_i)=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\quad m=\max_j x_j
\]

`logsumexp` 也使用同样技巧。减最大值不会改变理论结果，却把指数输入限制在 `(-∞,0]`。在低精度下仍建议用 FP32 计算 `max`、指数和归一化，最后再按需要转换输出。

LayerNorm、RMSNorm、BatchNorm 的均值/方差归约是常见敏感点。小方差、长序列和极端 outlier 会放大舍入误差；保留 FP32 统计量或使用稳定的两遍算法通常更可靠。不要只看前向输出近似，还要检查反向梯度和多步训练后的漂移。

### 7.3.5 FMA、融合和可重复性

融合乘加（FMA）在一次舍入中完成 `a*b+c`，通常比先乘再加更准确，但改变了最后几位。编译器重排、并行归约和原子操作也会改变运算顺序。若业务需要 bitwise reproducibility，应关闭会重排的优化或采用确定性算法，同时接受速度下降；若只需任务指标稳定，则定义合理容差更现实。

## 7.4 混合精度训练：autocast、累加与 loss scaling

### 7.4.1 autocast 的职责边界

自动混合精度（AMP）根据算子白名单、输入 dtype 和设备，选择低精度或 FP32 执行。矩阵乘法和卷积通常适合 BF16/FP16，softmax、归一化、损失和比较操作常保留 FP32。autocast 改变的是算子执行策略，不负责把模型参数永久转换，也不保证所有自定义算子都有安全策略。

[事实] PyTorch 的 `torch.autocast` 与 `torch.amp.GradScaler` 是两个可独立使用的组件。推理可以只使用 autocast；FP16 训练通常还需要梯度缩放以降低下溢风险。具体设备和版本的调用签名应查看目标版本文档。

一个可迁移的训练骨架如下（CUDA 只是示例，CPU 也可把 device 改成 `"cpu"` 并选择支持的 dtype）：

```python
import torch

model = Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
use_fp16 = (device.type == "cuda")
amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type,
                        dtype=amp_dtype,
                        enabled=True):
        pred = model(x)
        loss = loss_fn(pred, y)
    scaler.scale(loss).backward()
    # 需要梯度裁剪时，先解除缩放再裁剪
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

[故障模式] 在 `scaler.scale(loss).backward()` 之后直接裁剪梯度，会把缩放后的梯度当成真实梯度，阈值失效。正确顺序是 `unscale_`、检查/裁剪、`step`。使用多个 optimizer 时，每个 optimizer 最多 `unscale_` 一次，但 `scaler.update()` 每个迭代只调用一次。

### 7.4.2 loss scaling 的数学动机

设真实梯度 `g` 小于 FP16 的最小可表示范围，直接转换会下溢为零。把 loss 乘以尺度 `S`，反向得到 `Sg`，再在 optimizer step 前除以 `S`，可以把梯度暂时搬到可表示区间：

\[
\tilde{L}=S L,\quad \tilde{g}=\nabla \tilde{L}=Sg,\quad g=\tilde{g}/S
\]

若 `S` 太大，某层可能溢出为 inf。动态 GradScaler 通常在检测到 inf/nan 时跳过这次更新并减小 `S`，连续若干安全步后再增大。尺度可能低于 1；不要假设它一定从 1 增长到很大的整数。

### 7.4.3 FP16 与 BF16 的选择

- **FP16**：尾数较长、范围较窄。适合硬件 Tensor Core 和推理，但训练常需要 loss scaling、FP32 master weights 或敏感算子禁用低精度。
- **BF16**：范围大、尾数短。训练中通常更容易稳定，GradScaler 可能不必启用；对于需要精细相对精度的归一化、量化参数和小更新要保留 FP32。
- **混合策略**：参数和 optimizer state 用 FP32，激活/权重存储用 BF16 或 FP16，关键归约和 loss 用 FP32。显存节省取决于哪些张量真正转换，不能仅凭 autocast 上下文估算。

### 7.4.4 梯度累积与缩放

在 `k` 个 micro-batch 上累积梯度时，应保持缩放和除法一致。常见做法是每个 micro-batch 计算 `loss/k`，再调用 `scaler.scale(loss).backward()`，在第 `k` 个 batch unscale、裁剪和 step。若中途检测到 inf，整个 optimizer step 应跳过，并重置累积状态；把部分坏梯度继续累积会污染下一步。

## 7.5 FP8：格式、scale 和校准窗口

### 7.5.1 为什么 FP8 不能直接替代 FP16

FP8 的指数和尾数都更短，必须将张量缩放到合适区间。一个简单的张量级方案是：

\[
 s=\frac{q_{max}}{\max |x|+\epsilon},\quad q=\operatorname{round}(s x),\quad \hat{x}=q/s
\]

这里 `q_max` 是 FP8 格式的最大有限值。实际 Transformer Engine 等库会使用 amax 历史、margin、幂次舍入、E4M3/E5M2 角色分工以及逐张量/逐通道/逐块 scale。scale 本身通常用 FP32 保存。

### 7.5.2 current scaling 与 delayed scaling

- **Current scaling**：用当前张量的 amax 立即计算 scale。适应分布变化快，但需要在当前 kernel 前完成归约，可能增加延迟。
- **Delayed scaling**：维护最近若干步的 amax 历史，用历史统计量为下一步选 scale。可把 amax 计算与主路径重叠，代价是分布突然变化时 scale 滞后，出现饱和或有效位浪费。

`amax_history_len`、选择 max 还是最近值、margin 和异常值处理都会影响训练。记录 scale、amax、饱和率和 inf/nan 计数，比只记录 loss 更容易发现 FP8 退化。

### 7.5.3 粒度的权衡

- **per-tensor**：元数据少、kernel 简单，受 outlier 影响大。
- **per-channel/per-row**：对线性层权重和输出通道更友好，scale 数量增加。
- **per-group/per-block**：将向量切成固定大小（例如 32 或 128）分别缩放，能处理局部 outlier，代价是 scale 读写、对齐和硬件支持要求。

当 batch、序列长度或专家路由改变时，通道/块的分布可能不同。校准时应覆盖真实 shape 与路由，而不是只用一个平均 batch。

### 7.5.4 FP8 训练的安全护栏

1. 先以 BF16/FP16 建立收敛和吞吐基线，再逐层开启 FP8。
2. 默认保留 embedding、softmax、归一化、loss、optimizer state 和稀疏路由统计为 BF16/FP32。
3. 设置饱和率阈值（例如超过 0.1% 触发告警，具体阈值需按任务标定），同时监控 scale 是否连续撞到上下限。
4. 对 E4M3 前向、E5M2 反向的组合做 A/B；不要把一种格式的 scale 复制给另一种格式。
5. 准备一键回退开关，能够在不重启整个服务的情况下关闭 FP8 kernel 或恢复 BF16 checkpoint。

## 7.6 量化的数学模型与粒度

### 7.6.1 对称量化

对称量化把零点固定为 0：

\[
q=\operatorname{clip}(\operatorname{round}(x/s), q_{min},q_{max}),\qquad \hat{x}=s q
\]

`int8` 常取 `q_min=-128,q_max=127`，也有实现使用 `[-127,127]` 以保持对称。scale 可以取 `amax/q_max`，其中 `amax=max|x|`。对权重接近零均值时简单、kernel 高效；若激活明显偏正，会浪费负半轴的量化格点。

### 7.6.2 非对称（仿射）量化

非对称量化引入 zero-point：

\[
 s=\frac{x_{max}-x_{min}}{q_{max}-q_{min}},\quad
 z=\operatorname{round}(q_{min}-x_{min}/s),\quad
 q=\operatorname{clip}(\operatorname{round}(x/s+z),q_{min},q_{max})
\]

反量化为 `x_hat=s(q-z)`。它能把任意区间映射到整数范围，更好利用激活动态范围，但矩阵乘法需要处理 zero-point 交叉项，kernel 复杂度和算术开销更高。

### 7.6.3 粒度：per-tensor、per-channel、per-group

- **per-tensor**：一个 scale/zero-point 覆盖整个张量，内存最省，误差通常最大。
- **per-channel**：权重按输出通道（或输入通道）各有一组参数，适合通道间尺度差异大且权重静态的线性/卷积层。
- **per-group**：每 `G` 个元素共享参数，常用于 INT4 权重。`G=32/64/128` 是常见候选，越小误差越低但 scale 元数据和 kernel 开销越高。

粒度必须和 kernel 的布局一致。把按行 scale 错当成按列 scale，数值不会必然报错，却会产生持续偏差，属于最危险的静默故障之一。

### 7.6.4 PTQ、动态量化与 QAT

- **PTQ（后训练量化）**：模型训练完成后，用校准数据估计 scale/zero-point，再转换权重和激活。成本低、易上线，但对敏感层和分布漂移较脆弱。
- **动态量化**：权重离线量化，激活在每次推理动态计算 scale。适合输入分布变化大、CPU 推理的线性层；运行时有 amax/归约开销。
- **QAT（量化感知训练）**：前向插入 fake quant，反向用 STE（straight-through estimator）近似梯度，让模型适应离散格点。训练成本高，但在 INT8/INT4 严格精度目标下通常更稳。

新版本 PyTorch 生态正在把量化开发集中到 torchao/PT2E 等路径，旧的 `torch.ao.quantization` FX API 可能处于迁移阶段。生产代码应锁定版本并测试实际 API，不要复制多年以前的教程片段。

## 7.7 校准：如何估计 scale 而不被 outlier 绑架

### 7.7.1 校准数据的覆盖

校准集应覆盖线上真实输入的长度、语言、类别、温度、用户分群和异常边界。仅用训练集随机样本可能遗漏线上长尾；仅用极端样本又会把 scale 拉大，让常见值只使用很少格点。

记录每层激活的 `min/max/amax`、分位数、均值、标准差、零值比例和饱和率。对 Transformer 还要区分 token 位置、注意力头、专家路由和 KV cache 的统计。校准代码不得把用户敏感数据写入日志；只保存聚合统计和不可逆摘要。

### 7.7.2 MinMax、percentile、MSE 与 KL

- **MinMax**：直接使用观测最小/最大值，简单但对单个 outlier 极其敏感。
- **Percentile**：把上下 0.01% 或 0.1% 裁剪掉，再估计 scale。需要验证被裁剪长尾对任务的影响。
- **MSE/均方误差**：在候选 clipping 阈值中选择重构误差最小者，适合非均匀分布，但计算更贵。
- **KL/熵匹配**：寻找量化分布与原分布的近似阈值，经典 PTQ 工具常用；实现细节和直方图分桶会影响结果。

校准目标可以和业务指标不一致。让平均 RMSE 最小不一定让 top-1 或生成质量最好，最终要在代表性验证集上评估任务指标。

### 7.7.3 逐层敏感性扫描

将每层单独量化，其余层保持 FP32/BF16，测量输出误差和任务指标。按敏感性排序后，可把前 1% 最敏感层保留高精度，把预算集中在收益最大的地方。这种扫描比盲目“全模型 INT8”更快找到可行配置。

### 7.7.4 校准代码审计清单

- 统计是否在 `eval()` 和 `no_grad()` 下运行。
- 统计前是否清空旧的 observer/histogram。
- 直方图 bin 边界是否固定，跨 batch 是否使用同一坐标系。
- 是否把 padding、mask 后的无效 token 混入统计。
- 分布式校准时是否合并了各 rank 的 amax/计数，而不是只使用 rank 0。
- 保存 scale 时是否记录 dtype、shape、版本和量化布局，避免加载时错位。

## 7.8 INT8 与 INT4：内核、打包和误差传播

### 7.8.1 INT8 线性层

典型 INT8 GEMM 将 `A_int8 @ W_int8` 的乘积累加到 INT32，再乘以输入和权重 scale：

\[
Y \approx s_A s_W (A_q-Z_A)(W_q-Z_W)
\]

INT32 累加器的范围约为 `±2.1e9`。对长内积、较大 batch 或非对称 zero-point，必须估算最坏情况下的累加上界，避免累加器溢出。高性能 kernel 常将输入和权重按对称方式量化，减少 zero-point 交叉项。

### 7.8.2 INT4 weight-only

INT4 通常只量化权重，激活保持 BF16/FP16，在 kernel 内把 4-bit packed weight 解包并与高精度激活相乘。这节省权重内存，对大语言模型的内存带宽和 KV/权重驻留很有帮助；激活未量化，算力和 scale 处理仍然重要。

每字节可打包两个 4-bit 值。实现需要明确 nibble 顺序、符号编码（有符号范围 `[-8,7]` 或无符号 `[0,15]`）、组大小、scale dtype 和是否有 zero-point。不同库的打包布局不可混用；加载错误布局时通常不会崩溃，只会让输出明显偏差。

### 7.8.3 outlier 与混合精度回退

如果少数权重或激活通道幅度远大于主体，统一 INT4 scale 会让大多数值挤在少数格点。常见策略包括：

- outlier 通道保留 FP16/BF16，主体通道 INT4；
- 使用更小 group size 或 per-channel scale；
- SmoothQuant 一类方法把激活尺度迁移到权重；
- 对敏感层（首尾层、lm_head、归一化前后）保持高精度；
- QAT 让训练适应裁剪和离散误差。

任何回退都要报告实际量化覆盖率、参数字节数和 kernel 路径。模型文件标注“4-bit”并不代表所有层、所有激活都是 4-bit。

## 7.9 误差传播：从单层重构到端到端指标

### 7.9.1 一阶敏感性近似

设层输出 `y=f(x;W)`，权重扰动为 `δW`，一阶近似：

\[
\delta y \approx J_W \delta W
\]

其中 `J_W` 是对权重的雅可比。大范数或高条件数的层会放大量化噪声；残差连接可能把误差绕过多个层叠加。实际网络还有非线性、归一化和注意力重加权，单纯的局部 RMSE 只能作为筛选信号，不能替代任务评估。

### 7.9.2 乘法误差与加法误差

浮点舍入常可近似为 `fl(a op b)=(a op b)(1+δ), |δ|≤u`；量化误差更像受限加性噪声 `x_hat=x+ε`，其中 `ε` 与 scale、分布和 clipping 相关。多层后误差可能相互抵消，也可能因注意力 softmax、激活门控或归一化而被放大。

对生成模型，logits 的微小偏差在高温度/低温度采样下影响不同：低温度会放大 top-k 边界附近的排序变化。评估时固定采样种子并报告多次采样的分布，而不是只比较一次文本。

### 7.9.3 误差指标的组合

建议同时记录：

- `max_abs`：发现严重爆点；
- `MAE/RMSE`：主体重构质量；
- `p50/p99_abs`：尾部分布；
- 相对误差（对非零值设置阈值）；
- cosine similarity：方向一致性；
- KL/JS：概率分布差异；
- 任务指标：准确率、召回率、困惑度、延迟、成本。

如果输出包含大量接近零值，单看相对误差会夸大噪声；如果有长尾，单看 RMSE 会掩盖大多数值的精度。指标应和下游决策匹配。

## 7.10 内存、吞吐与端到端权衡

### 7.10.1 字节数与理论带宽

单元素存储大小：FP32 4 字节，BF16/FP16 2 字节，FP8 1 字节，INT8 1 字节，INT4 平均 0.5 字节（还要加 scale/zero-point 元数据）。在内存带宽受限的 kernel 中，理想吞吐提升近似与字节数反比；但真实提升受以下因素限制：

- scale、zero-point、padding 和对齐元数据；
- 低精度转换和反量化 kernel；
- 矩阵单元是否支持目标格式；
- 计算是否从内存瓶颈转为算力瓶颈；
- kernel launch、同步和通信开销；
- batch/shape 是否足以填满硬件。

INT4 常节省权重带宽，但解包和反量化会消耗 ALU/寄存器；小矩阵或低 batch 下可能不如 BF16。

### 7.10.2 roofline 视角

算术强度 `I=F/B`（每字节浮点操作数）决定算子更接近带宽屋顶还是计算屋顶。降低 dtype 减少 `B`，提高 `I`，可能把算子推向计算瓶颈；如果目标硬件的低精度矩阵峰值更高，双重收益才会出现。测量时要区分：

1. 纯 kernel 时间；
2. 包含量化/反量化的模块时间；
3. 包含数据加载、同步和服务排队的端到端时间。

### 7.10.3 batch 与形状效应

大 batch 可以摊薄 scale 计算和 launch 开销，小 batch/逐 token 解码更受内存延迟和 kernel 启动限制。动态序列长度导致 padding、不同 tile 选择和重新编译；量化参数按固定 group size 时，尾部不足一个 group 的处理也会影响效率。

性能报告至少给出 batch、序列长度、并发请求数和 p50/p95/p99。只测一个“大矩阵”很容易高估生产收益。

## 7.11 CPU 可运行量化与误差实验

本节提供一个不依赖 GPU、只需 Python 3 和 NumPy 的完整实验。它比较 FP16/BF16（若 numpy 支持）、INT8 的对称/非对称量化，扫描 clipping 百分位，报告误差、饱和率和矩阵乘法结果差异。

### 7.11.1 保存脚本

```python
# ch07_cpu_quant_experiment.py
import math
import time
import numpy as np


def symmetric_quantize(x, bits=8, clip_percentile=100.0):
    x = np.asarray(x, dtype=np.float32)
    qmax = (1 << (bits - 1)) - 1
    if clip_percentile < 100:
        amax = np.percentile(np.abs(x), clip_percentile)
    else:
        amax = np.max(np.abs(x))
    amax = max(float(amax), 1e-12)
    scale = amax / qmax
    q = np.clip(np.rint(x / scale), -qmax - 1, qmax).astype(np.int8)
    return q, np.float32(scale)


def affine_quantize(x, bits=8, clip_percentile=None):
    x = np.asarray(x, dtype=np.float32)
    qmin, qmax = 0, (1 << bits) - 1
    if clip_percentile is None:
        lo, hi = float(x.min()), float(x.max())
    else:
        p = float(clip_percentile)
        lo, hi = np.percentile(x, [100 - p, p])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8), np.float32(1.0), 0
    scale = (hi - lo) / (qmax - qmin)
    zp = int(np.rint(qmin - lo / scale))
    zp = max(qmin, min(qmax, zp))
    q = np.clip(np.rint(x / scale + zp), qmin, qmax).astype(np.uint8)
    return q, np.float32(scale), zp


def dequant_sym(q, scale):
    return q.astype(np.float32) * np.float32(scale)


def dequant_affine(q, scale, zp):
    return (q.astype(np.float32) - np.float32(zp)) * np.float32(scale)


def report(name, ref, approx, q=None):
    err = approx - ref
    abs_err = np.abs(err)
    denom = np.maximum(np.abs(ref), 1e-6)
    rel = abs_err / denom
    sat = 0.0 if q is None else float(np.mean((q == q.min()) | (q == q.max())))
    print(f"{name:26s} MAE={abs_err.mean():.6g} "
          f"RMSE={np.sqrt(np.mean(err*err)):.6g} "
          f"p99={np.percentile(abs_err,99):.6g} "
          f"rel_p99={np.percentile(rel,99):.6g} sat={sat:.4%}")


def matmul_error(seed=0):
    rng = np.random.default_rng(seed)
    # 对数正态 + 少量 outlier，更接近量化最容易失败的分布
    a = rng.standard_normal((256, 384), dtype=np.float32)
    b = rng.standard_normal((384, 192), dtype=np.float32)
    a[rng.random(a.shape) < 2e-4] *= 25
    b[rng.random(b.shape) < 2e-4] *= 25
    ref = a @ b
    print("-- activation/weight reconstruction --")
    for p in [100.0, 99.99, 99.9, 99.0]:
        qa, sa = symmetric_quantize(a, 8, p)
        qb, sb = symmetric_quantize(b, 8, p)
        ah, bh = dequant_sym(qa, sa), dequant_sym(qb, sb)
        report(f"INT8 symmetric p={p}", a, ah, qa)
        # 反量化后 matmul；真实 kernel 会在整数域累加再乘 sa*sb
        out = ah @ bh
        report(f"matmul p={p}", ref, out)

    # 非对称量化示例：把激活平移为非负区间
    pos = np.abs(a)  # 偏正分布
    q, s, z = affine_quantize(pos, 8, 99.9)
    recon = dequant_affine(q, s, z)
    print("-- affine activation --")
    print("scale=", float(s), "zero_point=", z)
    report("UINT8 affine p=99.9", pos, recon, q)


def timing(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2048, 2048), dtype=np.float32)
    q, s = symmetric_quantize(x)
    # 这里只比较 NumPy 的转换和 FP32 matmul，不声称代表专用 INT8 kernel
    for _ in range(2):
        _ = x @ x.T
    t0 = time.perf_counter(); _ = x @ x.T; t1 = time.perf_counter()
    t2 = time.perf_counter(); _ = dequant_sym(q, s); t3 = time.perf_counter()
    print("fp32 matmul_ms=", (t1 - t0) * 1e3,
          "int8 dequant_ms=", (t3 - t2) * 1e3,
          "bytes fp32/int8=", x.nbytes, q.nbytes)
    print("注意：NumPy 的这段代码没有调用专用 INT8 GEMM，不能把它当作硬件加速结论")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    matmul_error()
    timing()
```

运行：

```bash
python ch07_cpu_quant_experiment.py
```

在一台普通 CPU 上，`p=100` 通常具有较低的主体裁剪误差但对 outlier 敏感；把 clipping 调到 99.9 或 99.0 可能降低 p99 以外区域的误差，却会提高饱和率。具体数字随随机种子、BLAS 和 CPU 变化，不能把示例输出当成硬件承诺。

### 7.11.2 如何解读实验

1. **重构误差**：比较 `a` 与 `dequant(q)`，确认 scale/zero-point 公式和 dtype 转换没有错误。
2. **饱和率**：`q` 落在最小/最大端点的比例；高饱和率提示 clipping 太紧或分布存在 outlier。
3. **矩阵误差**：反量化后再 matmul 只是便于教学。真实 INT8 kernel 在整数域累加，误差还会受累加器、零点交叉项和融合顺序影响。
4. **速度**：脚本只测 NumPy FP32 matmul 与反量化，不代表 CPU 的 VNNI/AMX、ARM dotprod 或 GPU Tensor Core。要测专用 kernel，应使用目标框架的量化算子，并把转换和线程设置写入报告。

### 7.11.3 可选扩展实验

- 将 per-tensor 改为 per-row/per-column，对比误差和 scale 数量。
- 对 `a` 的 outlier 通道单独保留 FP32，测混合精度输出。
- 用 `np.float16` 计算累加（分块求和避免 NumPy 自动升格），观察长向量误差。
- 固定一组校准样本，再用均值平移后的分布测试，量化校准漂移。
- 记录不同 clipping 百分位的任务代理指标，例如线性分类器 logits 的 top-1 是否改变。

## 7.12 GPU 可选实验与测量边界

若有 CUDA GPU，可用 PyTorch/torchao/Transformer Engine 复现更接近生产的路径；本节不假设具体 GPU 型号。

### 7.12.1 AMP 基线

建立 FP32、BF16 autocast、FP16 autocast+GradScaler 三组基线。每组固定模型、shape、batch、种子和预热次数，使用 CUDA event 测 kernel 时间，同时用端到端 wall time 记录数据拷贝和同步。报告：吞吐、p50/p95、峰值显存、loss 曲线、梯度 inf/nan 次数和最终指标。

### 7.12.2 FP8 逐层开启

在支持 FP8 的硬件和库上，从单个线性层开始，记录 E4M3/E5M2、scale 模式、amax history、饱和率和反向梯度。若任务指标回退，按层二分定位；不要一次把 embedding、attention、MLP、loss 全部切换，无法判断是哪类算子造成问题。

### 7.12.3 INT8/INT4 端到端

比较 FP16/BF16、INT8 activation+weight、INT8 weight-only 和 INT4 weight-only。将模型加载、量化参数准备、首 token 延迟、稳态 token/s 和内存峰值分开计时。对逐 token 解码，batch=1 与 batch=16 的相对收益可能完全不同。

## 7.13 低精度故障案例

### 案例 1：FP16 loss 变成 NaN

**症状**：训练前几十步正常，某个 batch 后 loss=nan，之后全部梯度为 nan。

**常见根因**：logits/attention score 超过 FP16 范围，`exp` 溢出；梯度缩放过大；归一化方差接近零；数据中已有 inf/nan。

**诊断顺序**：在 autocast 内外分别检查 logits、softmax 输入、loss、梯度的 `isfinite`；打印 GradScaler 当前 scale 和跳步计数；用 FP32 运行同一 batch 验证数据问题；降低学习率或对敏感算子禁用 autocast。

**修复与验证**：稳定 softmax/logsumexp、保留 FP32 累加、启用动态 loss scaling、裁剪异常输入，并用固定坏 batch 回归。不要只把 NaN 替换为 0，否则会掩盖真正的溢出位置。

### 案例 2：BF16 训练不溢出但准确率下降

**症状**：loss 曲线平滑、无 inf/nan，但验证准确率比 FP32 低几个百分点。

**根因候选**：尾数只有 7 位，归一化统计、small residual 或 logits 排序受舍入影响；学习率、warmup 或权重衰减对低精度敏感；某个自定义算子误用了 BF16 累加。

**修复**：把归一化、loss、softmax、optimizer state 和关键归约改为 FP32；对敏感层做 per-channel scale 或保留 FP16；检查 kernel 累加 dtype；重新调学习率和 warmup。比较层级输出余弦相似度，定位误差从哪一层开始放大。

### 案例 3：INT8 校准后线上长尾崩溃

**症状**：离线平均准确率正常，特定用户、长文本或罕见类别的输出明显错误。

**根因**：校准集未覆盖长尾；percentile clipping 截掉了关键 outlier；动态 shape 导致 padding/mask 统计错误；线上预处理改变了数值范围。

**修复**：按用户分群和长度分桶重做校准，记录每层饱和率；敏感层改为 per-channel 或保留 FP16；上线前做分布漂移监控与 canary；当漂移超过阈值自动回退高精度路径。

### 案例 4：INT4 模型更小但延迟更高

**症状**：权重文件减少约 4 倍，batch=1 延迟却比 BF16 高。

**根因**：CPU/GPU 没有适配的 INT4 GEMM；每层即时解包/反量化；group size 太小导致 scale 读取；kernel 对齐不佳或算子频繁切换。

**修复**：确认 profiler 中真正调用了 INT4 kernel；测试 group size 32/64/128；将权重预打包并缓存；对小矩阵保留 BF16；把首 token、稳态 token 和多 batch 分别优化。仅凭模型文件大小不能推断延迟。

### 案例 5：FP8 scale 持续撞上限

**症状**：amax 或 scale 长期位于最大/最小边界，饱和率周期性升高。

**根因**：delayed scaling history 太长导致滞后；margin 设置错误；E4M3 用在反向梯度；异常 batch 未隔离；分布式 rank 使用了不同 scale。

**修复**：缩短 history 或切换 current scaling，检查 E4M3/E5M2 角色，采用 per-channel/block scale，合并各 rank 统计，并对异常 amax 设告警和回退。

### 案例 6：量化结果每次略有不同

**症状**：同一 checkpoint 多次运行的 logits 或指标有小幅差异，难以复现。

**根因**：随机舍入、并行归约顺序、动态校准窗口或不同线程/硬件路径；未固定 Python/NumPy/框架随机种子。

**修复**：在诊断模式关闭随机舍入，固定校准顺序和种子，启用确定性 kernel（接受性能下降），保存 scale/zero-point 快照；验收使用容差和分布，而非强求所有位一致。

## 7.14 调试与观测工作流

### 7.14.1 先验证数值契约

1. 用 FP32 eager 在固定小 batch 上生成参考输出和梯度。
2. 检查输入、权重、激活、loss、梯度是否有限。
3. 只转换一个算子或一层，比较输出 max/MAE/RMSE/cosine。
4. 逐步扩大到整模型、训练多步和真实 shape 分布。
5. 最后才测吞吐、显存和服务尾延迟。

### 7.14.2 记录直方图而不是单个标量

对每个关键张量记录 amax、p50/p90/p99、零值率、饱和率、scale、inf/nan 计数。只记录均值可能看不见长尾，记录完整用户数据又有隐私风险；应使用聚合统计、哈希后的层名和访问控制。

### 7.14.3 二分定位敏感层

将模型分成若干 stage，每次只把一半 stage 切回 FP32/BF16，比较任务指标。如果指标恢复，继续在该半段二分。对注意力、MLP、归一化、embedding 和输出头分别做单层 ablation，可在几十次实验内定位主要误差源。

### 7.14.4 版本和硬件边界

记录 PyTorch/torchao/Transformer Engine 版本、CUDA/cuBLAS/cuDNN、驱动、CPU 指令集（AVX2/AVX-512/AMX/NEON）、GPU 架构和编译选项。FP8/INT4 kernel 可能只在特定架构可用；同一 dtype 在不同设备上的累加和舍入路径不同。升级后重新跑正确性矩阵和性能基线。

## 7.14.5 误差预算与容量预算：把选择变成可计算的约束

低精度设计可以先写成一个预算问题。设一层输入、权重和输出的最大允许绝对误差分别为 `E_x`、`E_w`、`E_y`，量化重构误差为 `ε_x`、`ε_w`，累加舍入误差为 `ε_acc`。对线性层 `y=Wx` 的粗略一阶估计是：

\[
\|\delta y\| \lesssim \|W\|\,\|\epsilon_x\| + \|x\|\,\|\epsilon_W\| + \epsilon_{acc}
\]

它不是严格的端到端上界，却能帮助我们先找主要项。如果 `||W||` 很大，优先降低激活误差；如果输入范数很大而权重很敏感，优先 per-channel 或保留权重精度；如果两者都小却出现大误差，检查累加 dtype、布局和归约顺序。对残差网络，还要把每个 block 的误差预算留出余量，避免在最后一层才发现已经超标。

内存预算也应显式写出。设参数数目为 `P`，权重存储字节为 `b_w`，scale/zero-point 元数据总字节为 `M`，工作区和高精度保留副本为 `A`，则权重相关显存近似为：

\[
C_{weight} = P\times b_w + M + A
\]

FP32 的 `b_w=4`，BF16/FP16 为 2，FP8/INT8 为 1，INT4 约为 0.5；但 per-group scale 可能让 `M` 达到参数量的数个百分点，optimizer master weight 又可能让训练时 `A` 大于压缩权重本身。部署规划若只看 checkpoint 文件大小，容易低估运行时显存。

可以把允许的误差、显存和延迟写成一个表，再逐层选择格式：

```text
层/模块       误差上限       显存上限       延迟目标       候选格式
embedding     cosine>0.999   可较宽松       首 token 敏感   BF16/FP16
attention qkv  p99<阈值      中等           带宽受限       FP8/INT8
norm/softmax   max_abs<...    小             稳定性优先     FP32/BF16
MLP 权重       任务指标不降   最大占比       稳态吞吐       INT4/INT8
lm_head        top-k 不变     可放大         质量优先       BF16/FP32
```

先标记“必须保持高精度”的算子，再把剩余预算分给收益最大的层，比从全模型统一切换某种 dtype 更稳。若多种方案都满足约束，再用真实流量的 p99 延迟和总成本决定，而不是只看离线平均吞吐。

### 7.14.6 一个最小的误差回归门禁

将量化模型与 FP32 参考固定在同一批输入上，门禁至少检查四类条件：

1. **有限性**：所有公开输出和梯度都满足 `isfinite`，scale 不为零且不超出允许边界。
2. **局部误差**：每个敏感层的 MAE、RMSE、p99 和余弦相似度不超过层级阈值。
3. **全局指标**：任务准确率、困惑度、召回率或校准误差在统计置信区间内；生成任务应固定多个种子并比较分布。
4. **系统指标**：峰值显存、首请求延迟、稳态吞吐和 p99 尾延迟不回退到基线以下；量化转换和加载时间单独报告。

门禁失败时，不要只提高容差让流水线变绿。先保存失败输入、层名、scale、版本和硬件，判断是随机性、分布漂移还是实现错误。若允许回退，回退应发生在同一请求的可控边界，而不是把部分层静默混用导致难以追踪的状态。

### 7.14.7 校准和量化参数的生命周期

量化参数不是一次性常量。模型版本、tokenizer、预处理、batch/序列长度、采样温度和硬件 kernel 改变后，旧 scale 可能不再适用。建议把 scale、zero-point、group size、clip 百分位、校准样本摘要和生成工具版本一起存储，给每个参数文件一个内容哈希。加载时验证张量名、shape、布局、dtype 和哈希；有一项不匹配就拒绝静默加载。

线上监控只需聚合信息：按层统计 amax 分位数、饱和率、零值率、inf/nan 计数、回退次数和任务代理指标。设定一个观察窗口，只有连续多个窗口超过阈值才触发自动再校准或回退，避免单个异常请求引发抖动。再校准应在隔离流程完成并进行离线回归，不能直接拿未经审核的新 scale 覆盖生产。

## 7.15 六个理解检查（含答案）

### 检查 1：BF16 和 FP16 都是 2 字节，为什么训练行为差异很大？

**答案**：BF16 继承 FP32 的 8 位指数，范围大、较不易溢出，但只有 7 位尾数，舍入更粗；FP16 只有 5 位指数、10 位尾数，精度较细但范围窄，梯度/激活更易下溢或溢出。前者常减少 loss scaling 需求，后者常依赖 FP32 累加和动态缩放。

### 检查 2：`autocast` 是否会自动修复所有低精度溢出？

**答案**：不会。autocast 只按算子策略选择 dtype；自定义算子、未覆盖的设备、异常输入和错误累加仍可能溢出。需要稳定公式、`isfinite` 监控、必要时 FP32 fallback 与回归测试。

### 检查 3：为什么 INT8 的 scale 不能只在训练集上估计一次？

**答案**：线上激活分布可能因用户、长度、路由或版本漂移而改变。固定 scale 在新分布上可能大量饱和，动态 scale 又可能引入运行时开销。应使用覆盖真实流量的校准集，并监控 amax/饱和率，必要时重新校准或回退。

### 检查 4：per-channel 量化一定比 per-tensor 好吗？

**答案**：误差通常更低，但 scale 元数据、读取和 kernel 复杂度更高；在小矩阵或硬件不支持时可能变慢。还必须保证粒度和内存布局/内核约定一致，否则会出现静默错位。

### 检查 5：为什么 INT4 权重减少 4 倍不等于延迟减少 4 倍？

**答案**：INT4 需要解包、反量化和 scale 读取，且硬件可能没有高效 INT4 GEMM。端到端延迟还受 batch、内存访问、kernel launch、未量化层和数据准备影响。只有在权重带宽是主瓶颈且目标硬件支持时，才可能接近理论收益。

### 检查 6：FP8 为什么通常需要每张量或每块的 scale？

**答案**：FP8 的范围和尾数都短，直接表示真实张量会产生大量溢出或量化到少数格点。scale 把数值搬到可表示区间；一个全局 scale 会被 outlier 牵制，逐张量/逐块 scale 能提高有效位利用率，但增加元数据和归约成本。

## 7.16 练习

1. **格式账本**：为 FP32、BF16、FP16、E4M3、E5M2、INT8、INT4 填写符号/指数/尾数或整数范围、字节数、典型用途和主要风险。
2. **稳定 softmax**：用 NumPy 实现直接 softmax 与减最大值 softmax，在 logits=[1000,999,998] 的 FP16/FP32 下比较结果和 `isfinite`。
3. **loss scaling 仿真**：生成对数均匀分布的小梯度，把它转成 FP16，统计零值比例；尝试不同 `S` 后除以 `S`，找出下溢与溢出的折中区间。
4. **校准扫描**：在脚本中加入 percentile=99、99.5、99.9、99.99，绘制 clipping 阈值、饱和率和 RMSE 曲线，解释 Pareto 前沿。
5. **粒度实验**：实现 per-row 和 per-group INT8，比较相同 scale 元数据预算下的重构误差。
6. **误差传播**：构造 20 层残差 MLP，分别在第 1、10、20 层加入高斯量化噪声，观察输出 RMSE 和梯度范数如何变化。
7. **敏感层二分**：给一个小 Transformer，把一半层恢复 FP32，使用固定验证集测量指标，重复二分找到最敏感层。
8. **端到端报告**：对 CPU 量化脚本加入线程设置、预热、p50/p95 计时，明确哪些时间被纳入吞吐，哪些只是辅助统计。
9. **回滚演练**：设计 FP8/INT8 开关和饱和率阈值，当阈值连续三分钟超标时回退 BF16；写出告警、状态保存和恢复步骤。

## 7.17 来源地图

以下来源用于核对格式、API 和版本边界；链接指向官方文档，阅读时应选择与实际环境匹配的版本：

- [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)：autocast、GradScaler、设备策略和算子分类。
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)：梯度裁剪、累积、多 optimizer、反向和 unscale 顺序。
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)：浮点误差、归约、TF32、批次形状和非确定性。
- [PyTorch quantization overview](https://docs.pytorch.org/docs/stable/quantization.html)：量化 API 迁移和 torchao/PT2E 方向。
- [torchao quantization overview](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html)：当前量化类型、QAT 和低比特实现状态。
- [torchao quantized inference](https://docs.pytorch.org/ao/stable/workflows/inference.html)：INT8/INT4/FP8 推理配置和硬件依赖。
- [NVIDIA Transformer Engine FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)：E4M3/E5M2、amax、delayed scaling 和 autocast。
- [NVIDIA Transformer Engine scaling recipes](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/scaling.html)：current/delayed/block scaling 的参数边界。
- [IEEE 754-2019 标准概览](https://ieeexplore.ieee.org/document/8766227)：浮点编码、舍入和特殊值的规范来源（可能需要订阅）。
- [ONNX quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)：静态/动态量化、校准和误差调试的框架无关参考。
- [Intel oneDNN quantization](https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html)：CPU INT8 scale、zero-point 和累加路径。

[版本边界] PyTorch/torchao 的量化 API 仍在演进，旧版 `torch.quantization` 和 FX graph mode 示例不一定适用于新版本。FP8 格式、最大有限值、block size 和硬件支持依赖 Transformer Engine、cuBLAS/cuDNN 与 GPU 架构。部署前固定版本，运行最小校准/正确性脚本，不能只依据网页标题判断可用性。

## 7.18 安全边界与上线清单

低精度优化通常是 L2 风险，但以下边界必须保持：

1. **不要把精度降级用于身份、财务、医疗或安全决策而不做独立验证**。量化误差可能改变排序或阈值，必须有人审查并保留高精度审计路径。
2. **校准数据最小化和脱敏**。只保存聚合统计；不要把用户原文、token、梯度或可逆 embedding 写入共享日志。
3. **未经审核不要加载不受信任的量化权重/自定义 kernel**。打包的 pickle、C++/CUDA 扩展和驱动代码拥有进程权限，先在隔离环境验证哈希和来源。
4. **保留可回滚 checkpoint 与高精度基线**。量化参数、scale、zero-point、版本、布局和校准集摘要应随模型版本保存。
5. **把监控阈值写成可执行策略**。对 inf/nan、饱和率、scale 边界、任务指标、p99 延迟和分布漂移设告警和自动回退条件。
6. **不要为了追求吞吐关闭所有检查**。禁用 finite check、确定性或审计日志前，应证明风险可接受并提供离线验证。
7. **报告真实内存与能耗边界**。低比特可能减少显存，却因额外重算/转换增加 CPU、功耗或总成本；不要只宣传模型文件大小。

## 7.19 章节完成标准

读者完成本章后，应能回答：

- 目标张量的存储、计算、累加和指标精度分别是什么？
- 选择 BF16、FP16、FP8、INT8 或 INT4 的主要收益与失效边界是什么？
- 发生 overflow、underflow、clipping 或长尾回退时，哪一个观测量能最快定位问题？
- 量化粒度、校准集和 scale 如何影响误差、元数据和吞吐？
- 端到端基准是否包含转换、校准、同步、编译和服务排队？
- 如何在任务指标退化或分布漂移时自动回滚到高精度？

如果这些问题还不能回答，先运行 CPU 实验并写一页精度报告，再尝试 GPU/FP8/INT4。把每一次改动都绑定到一个可证伪的假设，保留 FP32 参考和真实流量回归，避免把单一硬件上的偶然数字当成通用规律。

## 7.20 小结

数值精度是一条从位布局延伸到系统成本的链：指数位决定范围，尾数位决定局部间隔，舍入和归约决定误差累积，loss scaling 保护 FP16 梯度，FP8 依赖 amax 与 scale，INT8/INT4 依赖量化粒度和校准，最终还要由 kernel、内存带宽、转换开销和业务指标共同验收。

可靠的低精度流程是：先用 FP32 建立正确性基线；再固定输入分布和版本；选择适合的存储/计算/累加组合；用代表性数据校准并记录 scale、饱和率和尾部误差；分层开启低精度，测量 kernel 与端到端两种边界；为 NaN、漂移、性能回退和安全风险准备监控与回滚。这样做，低精度才是可解释、可验证、可持续的系统优化，而不是一次不可复现的数字游戏。
