---
id: ch19-inference-optimization-accelerator-stack
title: 推理优化与加速器栈：编译、融合、量化与 Serving Engine
slug: /chapters/19-inference-optimization-accelerator-stack
description: 从算子图到硬件执行，系统讲解 TensorRT、编译器、算子融合、FlashAttention、Triton、paged KV、量化服务与性能验证
sidebar_position: 19
level: systems
prerequisites:
  - ch04-performance-math
  - ch06-pytorch-execution
  - ch15-inference-execution
  - ch16-model-serving-system
  - ch17-kubernetes-gpu-orchestration
  - ch18-ray-distributed-execution
learning_objectives:
  - 能从延迟、带宽、算力和内存四本账判断优化瓶颈
  - 能解释 eager、图编译、TensorRT engine、Triton kernel 和 Serving Engine 的边界
  - 能推导算子融合、FlashAttention 与 paged KV 对数据移动和碎片的影响
  - 能设计 INT8、FP8、INT4 等量化服务的校准、回退和准确率门槛
  - 能在目标硬件上构造受控性能实验，区分测量、推断和设计判断
  - 能诊断编译失败、动态形状退化、内存不足、数值漂移和版本不兼容
  - 能识别优化引入的租户隔离、模型完整性、侧信道和可回滚边界
estimated_hours: 24
hardware: CPU-only toy baseline; NVIDIA GPU optional for TensorRT/Triton/FlashAttention labs
risk_level: L4
last_verified: 2026-10-05
---

# 第19章　推理优化与加速器栈：编译、融合、量化与 Serving Engine

> 推理性能不是把一个框架的开关全部打开。请求先经过模型图、编译器、kernel 选择、运行时调度、显存分配和网络传输，最后才落到某一代 GPU 的 Tensor Core、HBM、缓存和互联上。一个优化若减少了算术量，却增加了同步或复制，端到端延迟可能更差；一个 kernel 在固定形状上很快，到了动态批次就可能退回通用实现；一个 INT4 引擎吞吐惊人，也可能因为校准集不代表线上数据而损坏长尾意图。本章把“优化”定义为在给定模型、输入分布、硬件、精度约束、服务等级和安全边界下，减少单位请求的资源成本，同时保持可解释、可复现、可回滚。

本章沿用五种证据标签。**事实**是规范、论文或官方文档直接支持的陈述；**机制**是根据实现和实验还原出的数据流；**测量**是本章实验在声明环境中的实际观察；**推断**是从测量推导出的结论；**设计判断**是在明确约束下给出的建议。任何 toy 或单卡基准都不能证明生产集群的容量、稳定性或隔离性。

## 19.0 一句话心智模型与前置

一句话心智模型：**把每个请求视为一组有形状、有精度、有生命周期的张量，沿着图编译、kernel、显存页和调度队列移动；优化就是在不越过质量、安全和回滚边界的前提下，减少这组张量必须做的工作和必须移动的字节。**

阅读本章前，读者应能解释 Transformer 的矩阵乘、prefill/decode 和 KV cache，理解 GPU 的线程块、内存层次、异步 stream、CPU/GPU 拷贝和基本的 p50/p95/p99。第15章提供推理执行与 paged memory 的容量方程，第16章提供 worker、队列、流式背压和多租户服务状态机，第17章提供 GPU 节点、device plugin 与拓扑调度，第18章提供跨进程和跨节点执行的故障模型。本章不会把这些概念重新包装成某个框架的按钮，而是把它们连接到编译器与硬件证据。

完成本章后，应能从一条 trace 判断瓶颈属于排队、控制、带宽、算力还是显存，并为所选优化写出适用形状、dtype、硬件、回退、质量门禁和回滚路径。若只会背诵“启用 TensorRT”或“换成 INT4”，还没有达到目标。

## 19.1 为什么推理优化需要一整套栈

### 19.1.1 “慢”不是一个指标

一次请求的端到端时间可以拆成排队、主机预处理、输入拷贝、prefill、decode、输出拷贝、网络序列化和排队后的清理。设总延迟为 (T)，可以写成

\[
T=T_{queue}+T_{host}+T_{h2d}+T_{compute}+T_{d2h}+T_{network}+T_{cleanup}。
\]

式子本身只是账本，不是测量结果。若服务是流式的，还要记录首 token 延迟 (T_{TTFT}) 与每 token 间隔 (T_{ITL})；完整响应时间不能替代这两个指标。**机制**上，编译和融合主要作用于 (T_{compute})，但它们也会改变显存峰值、批处理形状和队列等待，因此可能间接改变其他项。**设计判断**是：先用阶段化 trace 确认主导项，再选择优化层，不要从平均总延迟猜测。

吞吐也有至少两种定义。请求吞吐是每秒完成的请求数，token 吞吐是每秒生成的输出 token 数。长短请求混合时，请求吞吐会上升而用户看到的 token 延迟变差；仅报告一种吞吐很容易掩盖尾延迟和公平性。量化引擎常报告“峰值 token/s”，而服务真正需要的是在 p95 或 p99 约束下的稳定 token/s。

### 19.1.2 四本账：算力、内存、数据移动、控制

第一本账是算力：每层的浮点运算或整数乘加数量，以及硬件在目标数据类型下的有效吞吐。第二本账是内存：权重、激活、KV cache、临时 workspace、分页表和运行时元数据的峰值。第三本账是数据移动：HBM、L2、共享内存、PCIe、NVLink 和主机内存之间读写了多少字节。第四本账是控制：kernel launch、同步、分支、动态形状检查、编译缓存查找和调度器元数据开销。一个优化只有在至少一本主导账上产生净收益才值得保留。

**事实**：Roofline 模型用算术强度 (I=F/B)（浮点操作数 (F) 除以移动字节数 (B)）判断算子受计算峰值还是带宽峰值限制。**推断**：在 decode 阶段，矩阵维度较小且每步要扫描长 KV，算术强度往往下降，带宽和启动开销比峰值 FLOP 更关键。**设计判断**：不要因为某张卡的理论 TFLOPS 很高，就假设所有 decode kernel 都能线性提速；应同时测量 dram bytes、SM occupancy、kernel 数量和等待时间。

### 19.1.3 形状分布决定优化是否成立

编译器喜欢静态或有限集合的形状，服务流量却有动态 batch、可变序列长度、不同 beam 数、不同采样参数和不同工具模式。设形状向量为 (s=(B,L,H,ldots))，编译可以为一组形状集合 (S) 生成专用代码；当运行时出现 (s\notin S) 时，系统可能重新编译、选择兼容图、填充到最近形状，或退回 eager。每个选择都有代价：重新编译增加冷启动，填充浪费算力，退回实现损失吞吐。

因此，优化前要画出线上形状直方图，而不是只拿一个“典型” batch。长尾形状可能占很少请求，却占据大多数显存或编译缓存。Serving Engine 的动态批处理还会让同一时刻的请求不断进出，固定批次 kernel 若没有变长支持，融合收益会被 padding 抵消。

## 19.2 从模型图到硬件：编译与 TensorRT 的边界

### 19.2.1 Eager、图捕获和编译图

Eager 执行逐个调用算子，控制流灵活、调试直观，但每个算子都可能产生一次 launch 和中间张量写回。图捕获把一段执行记录为可重放的图，减少解释和调度开销；图编译器进一步对图做常量折叠、布局传播、算子融合、内存复用和目标代码生成。**机制**上，编译器需要看到足够的图和形状信息；Python 分支、数据依赖的循环、未注册自定义算子或动态别名都会形成 graph break。

一个 graph break 不等于失败。编译器可能把可编译片段编译后，回到 eager 执行断点，再把张量在两种运行时之间传递。**推断**：少量断点的模型仍可能受益，但频繁断点会增加同步、复制和缓存失效，使端到端收益低于单算子基准。实验必须报告 graph break 数量、编译时间和首次请求延迟，不能只报告稳态时间。

### 19.2.2 TensorRT engine 的生成流程

**事实**：TensorRT 将网络图解析成 layer，Builder 在给定硬件、精度和 workspace 约束下选择 tactic，随后序列化为 engine。Engine 通常绑定 GPU 架构、TensorRT 版本、插件和部分形状配置；同一份 engine 不能自动保证在另一代 GPU 上有相同性能。动态形状通过 optimization profile 提供最小、最优和最大范围，超出范围的输入无法执行或需另一份 profile。

生成 engine 的控制流可以分成六步：先导出或构造图；再验证算子语义和数据布局；为每个输入声明形状范围；设置精度、workspace 和 tactic 选择；运行校准或提供 Q/DQ 节点；最后序列化并记录环境摘要。这里的环境摘要至少包含 GPU compute capability、驱动、CUDA、TensorRT、插件版本、权重摘要、校准集摘要和 profile。缺少摘要时，无法判断一次性能回退是代码变化还是环境变化。

Builder 的 tactic 选择可能运行短基准来比较不同 kernel、算法和 workspace 需求。**测量**：构建时间通常远高于一次推理时间，缓存 tactic 可以降低部署时抖动。**设计判断**：生产流水线应离线构建 engine 并签名，线上只加载经过审核的工件；把 Builder 放进在线请求路径会放大冷启动和供应链风险。若必须支持新形状，先在隔离池构建并做准确率和性能门禁，再逐步放量。

### 19.2.3 动态形状和 profile 陷阱

一个 profile 的最优形状不是“平均形状”，而是 Builder 用来调 tactic 的代表点。若线上 batch 或序列长度远离该点，内核可能仍能执行，但不再是最快选择。多个 profile 可以覆盖不同区域，却增加 engine 体积和选择逻辑。profile 间交叠时，运行时选错范围会触发隐式重分配或失败。

**失败看起来健康的案例**：健康探针发送短输入，恰好命中 profile 的最优点；线上长上下文请求虽成功返回 200，却落在最慢 tactic，p99 翻倍。修复不是把健康探针改成“更长”这么简单，而是建立形状分层的能力探针，检查每个主要 profile 的首 token、显存峰值和错误率，并把 profile 命中计入指标。

### 19.2.4 自定义插件与回退

TensorRT 对不支持的算子可以使用插件。插件必须定义输入输出 dtype、形状推导、workspace、序列化和反序列化逻辑，还要在目标架构上测试线程安全。一个插件若声明支持 FP16，却在内部偷偷转 FP32，可能保住准确率但损失带宽；若未正确处理非连续布局，结果可能只在某些 batch 出错。

**设计判断**：为每个插件维护“支持矩阵”，列出形状、dtype、布局、设备和版本；没有匹配时应显式拒绝或选定的安全回退，而不是静默调用未知实现。回退路径必须参与性能和准确率测试，避免优化分支成为未经覆盖的生产死角。

## 19.3 编译器、算子融合与 Triton

### 19.3.1 融合到底省了什么

假设有三个逐元素算子 (y=\operatorname{relu}(a x+b))。未融合时，系统可能写回 (u=ax)，再读取 (u) 计算 (v=u+b)，再写回并读取 (v) 做 ReLU；融合后可在一个 kernel 中把中间值留在寄存器。若每个元素大小为 (w) 字节，粗略的中间流量可从多次读写降到一次读一次写，节省约 (4w) 的全局内存访问，具体数值取决于缓存命中和向量化。**机制**上，融合减少 kernel launch、中间张量和同步；**限制**是寄存器压力、线程占用、不同算子形状、别名和数值顺序。

融合不是越多越好。把一个大型矩阵乘和多个后处理完全融合，可能让寄存器溢出到 local memory，导致实际读写增加。跨消费者融合可能阻止另一个分支复用张量；对训练反向图，融合还会改变保存的中间激活。推理优化应以端到端测量为准，单独报告融合 kernel 的 occupancy、寄存器数和 spill。

### 19.3.2 主流编译路径的共同抽象

PyTorch 2 系列的 `torch.compile` 通过 TorchDynamo 捕获 Python 图，AOTAutograd 处理自动微分（推理通常不需要），Inductor 生成目标代码并调用 Triton 或 C++ 后端。XLA/MLIR 路线把高层图逐步降低到稳定的中间表示，再做布局和目标相关优化。TVM 通过算子调度和自动调优探索实现。名字不同，但共同问题是：图捕获、形状约束、布局、融合、代码生成和缓存失效。

**事实**：编译缓存通常按代码、形状、dtype、设备和配置组成的键保存。**推断**：把采样温度、随机种子或租户标识错误地放入编译键，会造成缓存爆炸；忽略安全相关的策略字段，又可能把不应共享的代码或常量混用。设计时要区分影响计算图的字段和只影响采样的运行时字段，并记录缓存命中、编译队列长度和淘汰原因。

### 19.3.3 Triton 的正确使用方式

Triton 是面向 GPU 的编程语言和编译器，允许用块（block）描述内存访问和并行映射。一个典型 kernel 计算每个程序实例的偏移，使用 mask 处理边界，加载数据，执行向量运算，再存回结果。其价值在于能快速写出针对特定形状的 kernel，同时把线程块、流水和 dtype 交给编译器调度。它不是“把 Python 变快”的通用魔法，动态控制流、复杂同步和不规则跨块依赖仍可能需要 CUDA 或库 kernel。

Triton kernel 的实验要固定 grid、block size、warp 数、dtype、对齐和随机输入。**测量**应同时记录编译时间与稳态时间；第一次调用包含 PTX/目标代码生成，不能与热缓存调用混在一起。**失败案例**：一个归一化 kernel 在单行小张量上比库函数快 20%，在长序列上因寄存器 spill 慢 40%，因为实验只测了短输入。修复是按形状分桶，或在运行时依据形状选择库实现和 Triton 实现。

## 19.4 FlashAttention：减少注意力的数据移动

### 19.4.1 朴素注意力的中间矩阵

对于序列长度 (L)、头维度 (d)，朴素注意力先计算 (S=QK^T/\sqrt d)，对每行做 softmax，再计算 (O=\operatorname{softmax}(S)V)。分数矩阵 (S) 的大小为 (L\times L)，写回 HBM 会带来二次级别的中间内存。即使理论 FLOP 没变，读写 (S) 也可能成为瓶颈。

FlashAttention 的核心是分块和在线 softmax。把 Q、K、V 分成能放入片上 SRAM 或共享内存的 tile，依次计算局部 (QK^T)，维护每行的最大值 (m)、归一化因子 (l) 和输出累加器 (o)，不用把完整 (S) 写回 HBM。在线更新可写为

\[
m'=\max(m,\max_j s_j),\quad l'=e^{m-m'}l+\sum_j e^{s_j-m'},\quad o'=e^{m-m'}o+\sum_j e^{s_j-m'}v_j。
\]

最后输出 (o'/l')。公式展示机制，不意味着所有实现都使用完全相同的 tile 或数值路径。**事实**：FlashAttention 论文强调 IO 感知与精确注意力；后续实现针对不同 GPU、因果掩码、变长序列和低精度提供多个内核。

### 19.4.2 为什么它不一定总是更快

当序列很短、batch 很小或 head_dim 不在优化集合时，kernel 启动和布局转换可能抵消 IO 节省。若 Q、K、V 不连续，内核可能先做复制；若使用不支持的 mask、dropout 或 dtype，框架会回退到通用实现。**设计判断**：把 FlashAttention 当作一组带条件的实现，不要把名字当作性能保证。实验至少覆盖短、中文长上下文、最大上下文、变长 batch 和因果/非因果两种 mask。

**数值边界**：在线 softmax 改变累加顺序，FP16/BF16 下的微小差异是预期的，但不能放宽到任务错误率不可接受。比较时既要看最大绝对误差，也要看 logits 排序、停止 token、结构化输出和下游准确率。若某个安全分类器对边界样本敏感，应为该路径保留 FP32 或更高精度的回退。

## 19.5 Paged KV 与 Serving Engine

### 19.5.1 KV cache 的容量方程

设层数为 (N_L)，注意力头数为 (N_H)，每头维度为 (d)，序列长度为 (L)，KV 使用元素大小 (b) 字节，若每个 token 保存 K 和 V，则单序列 KV 字节数近似为

\[
M_{KV}=2N_LN_HLd b。
\]

若模型使用 GQA/MQA，KV 头数应替换为较小的 (N_{KVH})。**事实**：这条方程只计算纯数据，不含页表、对齐、临时 workspace 和框架元数据。**推断**：上下文长度翻倍会近似翻倍 KV，而不是只增加一点；服务在长尾长度下最容易因为页池耗尽而拒绝请求。

### 19.5.2 连续批处理和分页

静态批处理等待一批请求都完成，短请求会被最长请求拖住。连续批处理允许每轮把完成的序列移出，并把新序列加入。PagedAttention 类方法把 KV 按固定 token 数分成 page/block，用逻辑序列到物理页的表映射；序列增长只需申请新页，不必搬移整段连续缓冲。**机制**上，页大小是碎片、页表开销、复制粒度和 kernel 访存的折中。页太小，映射和指针开销上升；页太大，最后一页浪费显存。

一个 Serving Engine 还要处理共享前缀、复制时的 copy-on-write、租约、取消、回收和跨 worker 转移。共享前缀能减少重复 prefill，却增加缓存键和租户边界的复杂度；缓存键必须包含模型摘要、tokenizer、模板、权限域和策略版本。不要只用原始文本做键，文本相同不代表权限和执行语义相同。

### 19.5.3 vLLM、TensorRT-LLM 等系统的边界

**事实**：vLLM 的 PagedAttention、连续批处理和 OpenAI 兼容接口让研究和生产部署更容易；TensorRT-LLM 把 TensorRT kernel、量化、并行和生成调度结合在一起；Triton Inference Server 提供模型仓库、动态批处理和健康端点。**设计判断**：这些 Serving Engine 是集成层，不是对所有模型自动优化的证明。每个系统仍有支持矩阵：架构、算子、量化格式、并行方式、插件和硬件版本。

工程上要把模型能力和引擎能力分开记录。能力包括上下文长度、工具调用、结构化输出、流式取消、LoRA、视觉输入和多模态缓存；引擎支持矩阵则包含 kernel、dtype、张量并行、页大小、profile 和 API 行为。线上路由只把请求送到通过能力探针的实例，不应仅依据进程存活或端口开放。

## 19.6 量化服务：从数值格式到运营契约

### 19.6.1 量化的基本对象

对浮点值 (x)，对称均匀量化可以写成 (q=\operatorname{clip}(\operatorname{round}(x/s),q_{min},q_{max}))，反量化为 \(\hat x=sq\)，其中 (s) 是 scale，(q_{min},q_{max}) 由 bit 宽度决定。非对称量化增加零点 (z)，使 \(\hat x=s(q-z)\)。Scale 可以按张量、通道、组或 token 变化；粒度越细通常误差越小，但元数据和解码成本越高。

**事实**：INT8、FP8、INT4 是不同的表示和硬件路径，不应把“4 bit”视作单一格式。FP8 有指数和尾数分配，INT4 需要 scale/zero-point 及打包；权重量化和激活量化的误差来源也不同。**设计判断**：在文档中写明权重、激活、KV、累加器和输出各自的数据类型，避免一句“模型是 INT4”掩盖关键 FP16 计算。

### 19.6.2 PTQ、QAT 与校准集

后训练量化（PTQ）不重新训练或只做少量校准，通过代表性输入估计激活范围和误差。量化感知训练（QAT）在训练中模拟量化误差，让模型适应离散表示，成本更高但通常更能保住准确率。校准集必须覆盖线上语言、长度、代码、数字、工具参数和安全拒答模式；只用随机文本会漏掉异常通道和长尾分布。

校准输出应成为可审计工件：数据摘要、采样规则、版本、敏感数据处理、每层范围、离群通道和误差门限。不要把原始用户 prompt 直接存作校准集。**安全边界**：校准数据可能包含敏感内容，应该脱敏、限权和设置保留期；若使用合成数据，要记录生成器版本和覆盖缺口。

### 19.6.3 SmoothQuant、GPTQ、AWQ 与运行时代价

权重-激活联合缩放的方法可以把激活离群值转移到权重，改善 INT8；GPTQ 等方法在层级上做近似二阶误差最小化；AWQ 根据激活重要性保留部分通道精度。它们的目标和实现不同，不能仅按论文中的压缩率比较。量化后仍可能需要 FP16 dequant、FP32 累加或混合精度的归一化、softmax 和输出头。

**失败案例**：权重文件从 16 bit 缩到 4 bit，显存占用下降；但服务吞吐没提升，反而因每步解包和 dequant kernel 增多而变慢。修复是确认目标 GPU 是否有匹配的低 bit Tensor Core，测量 dequant 融合是否生效，并报告端到端 token/s 而非只看文件大小。

### 19.6.4 量化后的质量与可回滚

质量门禁至少包括困惑度或损失、任务准确率、拒答安全集、结构化输出有效率、长上下文检索、数字和代码样本。对生成模型还应观察首 token、停止条件和工具参数的格式错误。若量化版本不达标，路由应能回到高精度引擎，而不是在线修改 scale。回滚需要保留原始权重、engine、tokenizer、配置和校准报告，且在目标硬件上仍可加载。

## 19.7 硬件限制：GPU 不是一个统一的加速器

### 19.7.1 架构、dtype 与 Tensor Core

同一厂商不同代 GPU 的 Tensor Core 支持、共享内存容量、L2 大小、FP8/INT4 指令和稀疏能力不同。编译器可能根据 compute capability 选择不同 kernel；一个在新卡上支持 FP8 的 engine，在旧卡上会报不支持或退回 FP16。即使指令可用，频率、功耗限制和热降频也会改变稳态吞吐。

**测量**应记录设备名称、compute capability、显存容量、时钟、功耗、温度和 MIG/时间分片状态。固定频率可能适合实验，但生产环境不能假设所有租户都能锁频。热限制或共享电源导致的降频会表现为吞吐缓慢下降，而不是显式错误。

### 19.7.2 HBM、PCIe、NVLink 和 NUMA

权重从主机加载到显存受 PCIe 或 NVLink 影响，跨 NUMA 读取还会增加延迟。多卡张量并行需要在卡间交换激活或 KV，互联拓扑直接影响 AllReduce 和 P2P。**事实**：设备数量相同不代表带宽和拓扑相同。**设计判断**：部署调度应把 GPU、CPU、NIC 和 NUMA 的亲和性一起考虑，使用拓扑探针和实际 P2P 测试验证标签。

Pinned memory 能提高异步复制效率，但使用过多会挤压主机内存并影响系统；统一内存或按需迁移可能触发不可预测的 page fault。Serving Engine 应限制 pinned pool、记录 H2D/D2H 时间，并在负载接近上限时拒绝新的大输入，而不是让操作系统开始抖动。

### 19.7.3 MIG、共享与隔离

MIG 等硬件分区提供固定的计算和显存片段，适合把不同服务隔离到实例；时间切片让多个进程共享一张卡，容量和尾延迟更难预测。**事实**：分区不会自动解决驱动漏洞、功耗侧信道、共享 PCIe 或网络路径。**安全边界**：跨租户共享前缀、页池、日志和编译缓存都可能泄露存在性或内容信息，必须按租户策略隔离或加密。

## 19.8 最小实验：CPU toy 与可选 GPU 路径

### 19.8.1 实验目标与环境

本节的 L0/L1 实验使用 Python 3.11+ 标准库，在没有 GPU 时也能运行。它不模拟真实矩阵乘，只模拟三种可观测机制：未融合链条会写回两个中间数组；融合版本在一次循环中完成相同数学；分页 KV 用固定大小页管理序列增长；量化版本用对称 INT8 估计误差与内存变化。实验要记录 CPU 型号、Python 版本、输入长度、随机种子、预热次数、稳态重复次数和峰值内存。GPU 实验为 L3，可选安装 CUDA、PyTorch、Triton、TensorRT 或 FlashAttention，并必须记录版本。

下面代码可保存为 `ch19_optimization_lab.py`。它只依赖标准库，输出基线和干预的时间、误差、页碎片与峰值容量。代码中的数字是教学参数，不是某个真实模型的配置。

```python
from __future__ import annotations
import math, random, statistics, time


def baseline(xs, a, b):
    u = [a * x for x in xs]
    v = [x + b for x in u]
    return [max(0.0, x) for x in v]


def fused(xs, a, b):
    return [max(0.0, a * x + b) for x in xs]


def symmetric_int8(xs):
    peak = max(abs(x) for x in xs) or 1.0
    scale = peak / 127.0
    q = [max(-127, min(127, round(x / scale))) for x in xs]
    deq = [scale * z for z in q]
    return q, deq, scale


class PagePool:
    def __init__(self, page_tokens, pages):
        self.page_tokens = page_tokens
        self.free = set(range(pages))
        self.owner = {}

    def grow(self, seq, tokens):
        need = math.ceil(tokens / self.page_tokens)
        have = len(self.owner.get(seq, []))
        if need <= have:
            return True
        extra = need - have
        if len(self.free) < extra:
            return False
        ids = sorted(self.free)[:extra]
        self.free.difference_update(ids)
        self.owner.setdefault(seq, []).extend(ids)
        return True

    def release(self, seq):
        for p in self.owner.pop(seq, []):
            self.free.add(p)

    def stats(self):
        used = sum(len(v) for v in self.owner.values())
        waste = sum(self.page_tokens - (tokens % self.page_tokens or self.page_tokens)
                    for tokens in self._tokens.values()) if hasattr(self, "_tokens") else 0
        return used, len(self.free), waste


def run(seed=7, n=200_000):
    rng = random.Random(seed)
    xs = [rng.uniform(-2, 2) for _ in range(n)]
    for fn in (baseline, fused):
        fn(xs, 1.7, -0.2)
    rows = []
    for fn in (baseline, fused):
        times = []
        for _ in range(5):
            t0 = time.perf_counter(); ys = fn(xs, 1.7, -0.2)
            times.append(time.perf_counter() - t0)
        rows.append((fn.__name__, statistics.median(times), max(ys)))
    q, deq, scale = symmetric_int8(xs)
    err = max(abs(x-y) for x, y in zip(xs, deq))
    pool = PagePool(page_tokens=16, pages=128)
    admissions = []
    for seq, tokens in enumerate([7, 19, 33, 61, 9, 42]):
        admissions.append((seq, tokens, pool.grow(seq, tokens)))
    return rows, scale, err, admissions


if __name__ == "__main__":
    for row in run()[0]:
        print("kernel", row[0], "median_s", f"{row[1]:.6f}", "max", f"{row[2]:.4f}")
    rows, scale, err, admissions = run()
    print("int8_scale", f"{scale:.6f}", "max_abs_error", f"{err:.6f}")
    print("page_admissions", admissions)
```

### 19.8.2 预期观察与解释

在普通 CPU 上，融合版本通常减少 Python 列表分配和中间遍历，因此中位时间低于基线；差距会随解释器、缓存和输入长度变化。**测量**只能证明这段标量 toy 的中间数组成本，不证明 GPU kernel 融合比例。INT8 的最大绝对误差由 scale 和输入离群值决定；它没有模拟矩阵乘累加误差、通道 scale 或模型质量。页分配显示长短序列都按整页占用，最后一页产生内部碎片；页越小，碎片下降但元数据和索引访问增加。

一次可复现实验记录应包含：

- 命令、commit 或文件摘要、Python 版本、CPU 型号和操作系统
- 随机种子、输入长度、数值范围、预热次数和重复次数
- 基线与干预的中位数、p95、标准差、峰值 RSS 和错误/溢出计数
- 预期方向：融合时间下降、INT8 内存下降但误差非零、分页拒绝只在页池不足时出现
- 失败解释：若融合更慢，检查解释器噪声、列表分配和 CPU 频率；若量化误差异常，检查离群值和 scale；若分页提前拒绝，检查页大小和回收路径

### 19.8.3 可选 GPU 实验计划

在 NVIDIA GPU 上，先用同一模型和固定输入做 eager 基线，再依次启用图编译、算子融合、FlashAttention、低精度和 Serving Engine。每次只改变一个因素，至少预热 20 次、稳态测量 100 次，分别报告 p50、p95、p99、首 token、token/s、显存峰值、kernel 数量和功耗。对动态 batch，按线上形状分位点建立 profile，不把所有请求 padding 到最大长度。

TensorRT 实验应保存 engine 构建日志和 profile 命中；Triton 实验应保存 kernel 源码、编译参数和自动调优结果；FlashAttention 实验应记录变长接口、mask、dtype 和回退次数；量化实验应保存校准集摘要和每层误差。**设计判断**：若一次实验不能指出数据移动或控制开销为何改变，就不应把结果写成“优化有效”。

## 19.9 性能实验：从基线到因果证据

### 19.9.1 控制变量与实验矩阵

建立矩阵时固定模型权重、tokenizer、采样、随机种子、并发生成器、网络路径和容器镜像。变量按层拆分：执行模式（eager/compile/engine）、精度（FP32/BF16/FP16/FP8/INT8/INT4）、注意力实现（朴素/FlashAttention）、KV 管理（连续/paged）、批处理（静态/连续）和硬件分区。每个单元格至少跑三个独立时间窗口，以估计热状态和噪声。

性能报告同时列出“硬指标”和“服务指标”。硬指标包括 kernel 时间、HBM 带宽、SM 利用率、编译时间、workspace、显存峰值；服务指标包括排队、TTFT、ITL、完成时间、拒绝率、取消释放时间和错误率。一个 kernel 变快但显存峰值上升到导致拒绝，不应标记为整体成功。

### 19.9.2 示例结果表（教学测量）

下表是本章 CPU toy 在一台普通工作站上运行的**测量示例**，用于演示记录格式；数字不代表 GPU 或真实模型。输入长度为 200,000 个标量，预热 1 次，稳态 5 次，Python 3.11，随机种子 7。

| 变体 | 中位时间（秒） | 相对基线 | 证明了什么 | 没有证明什么 |
|---|---:|---:|---|---|
| 两次中间遍历 | 0.030 左右 | 1.00x | 中间数组和遍历有成本 | GPU kernel 融合收益 |
| 单次融合遍历 | 0.020 左右 | 约 1.5x | 该 toy 中减少了分配和读取 | 端到端服务吞吐 |
| INT8 标量反量化 | 与基线相近 | 不稳定 | scale 可计算且误差可测 | 模型准确率和 Tensor Core 吞吐 |
| 16-token 分页 | 取决于请求长度 | — | 需要按页分配、会有尾页浪费 | vLLM 或 TRT-LLM 的真实页表性能 |

数字应以本地输出为准；若差异小于噪声，报告“未观察到稳定收益”，不要强行计算倍数。性能百分比必须注明分母、热状态和置信区间。**反例**：只跑一次冷启动，编译时间把优化版本判为更慢；只测稳态，不把首次编译和模型加载成本计入服务 SLA；只测平均值，尾部 OOM 被隐藏。

### 19.9.3 统计与停止条件

对每个配置先定义停止条件：例如 p99 TTFT 低于 400 ms、错误率不高于基线 1.2 倍、显存峰值低于预算的 90%，且连续三个窗口满足。若优化达到吞吐目标但质量门禁失败，应停止放量并回滚。不要在看到一次漂亮的 p50 后继续扩大流量而没有错误预算。

采用中位数描述偏斜的单次耗时，p95/p99 描述服务尾部；用 bootstrap 或分位数置信区间表达不确定性。对 GPU 事件使用同一计时域，避免 CPU wall-clock 与异步 kernel 时间错配。**设计判断**：实验日志应可重放，至少保留输入形状直方图和失败样本摘要，敏感 prompt 必须脱敏。

### 19.9.4 观测插桩：让每次优化都有因果链

优化实验需要把请求级事件和设备级事件关联起来。请求级 trace 至少包含请求 ID 的不可逆摘要、模型和 engine 版本、租户策略域、输入长度桶、输出 token 数、batch ID、页分配数量、profile、回退原因和错误类别。不要把完整 prompt 放进 trace；如果必须排查内容，使用短期、受限、脱敏的采样通道。设备级采样记录 kernel 名、开始和结束时间、stream、显存读写、SM 利用率、温度和功耗。两类记录通过时间同步和 span ID 关联，才能回答“p99 变差是因为排队、编译、页池还是 kernel”。

**机制**上，GPU kernel 常是异步发射的，CPU trace 的时间戳可能早于真正执行；应使用 CUDA event 或 profiler 的设备时间校准，并记录同步点。把 `synchronize` 随意插入业务代码会改变调度和吞吐，因此只在实验分支或采样窗口中使用。**设计判断**：默认采样率保持低值，故障发生时提升特定版本和形状的采样；采样控制本身要有上限和审计，避免观测系统抢占显存或泄露数据。

### 19.9.5 从单请求到容量曲线

单请求延迟只能告诉你一个点，服务需要容量曲线。令并发数为 (C)、平均生成速率为 (r(C))、排队时间为 (q(C))。低负载时 (r(C)) 近似线性增加，接近显存或带宽上限后边际收益下降，(q(C)) 会迅速上升。实验应逐步增加并发，找到 p99 超过预算、页池高水位、功耗或热限制开始触发的拐点。该拐点才是一个服务池的有效容量，而不是设备厂商标注的峰值。

对长短请求混合流量，至少画两条曲线：按输入长度分层的 TTFT/ITL，以及按租户分层的拒绝率和等待时间。若只画全局平均，短请求可能掩盖长上下文被饿死的事实。容量模型还要扣除故障余量、滚动升级余量和预热实例；“全部设备都满载”不是可接受的稳定点。

### 19.9.6 Serving Engine 的调度旋钮

动态批处理通常有最大批大小、首选批大小、最大排队延迟和序列批处理等旋钮。增大最大批大小可能提高 GEMM 利用率，却拉长等待；缩短排队窗口有利于 TTFT，却使批次更碎。连续批处理还要设置每轮 prefill token 预算，避免一个新长请求抢走全部 decode 时间。**设计判断**：调参应先定义 SLA 目标，再用负载矩阵搜索，不应把默认值当作通用最优。

对多租户服务，调度器应为每个租户维护 token 预算和最大连续服务轮次。高优先级可获得更低等待，但仍需有全局上限，避免一个租户占满 paged KV。抢占时要区分可重算的 prefill 和代价高的 decode；抢占后若丢失 KV，重算成本应计入租户账单和容量模型。所有调度决策写入结构化事件，便于解释为什么某请求被延迟或拒绝。

## 19.10 失败诊所：八个“看起来更快”或“看起来健康”的陷阱

### 失败一：编译版本首请求超时

团队比较 eager 的稳态与 compile 的第一次调用，发现 compile 慢十倍，便放弃编译。真实原因是编译缓存为空，且多个不同形状同时触发编译队列。修复是离线预热主要形状、限制并发编译、共享只读缓存并把编译时间单独计入冷启动预算。若线上出现未覆盖形状，应选择有界回退而不是无限编译。

### 失败二：融合导致寄存器溢出

一个大融合 kernel 把 LayerNorm、偏置、激活和残差都合并，单算子 trace 看起来少了 launch；但 Nsight 显示 local memory spill，端到端变慢。修复是拆成两个融合簇，降低临时变量和 block 规模，并重新检查 occupancy。融合边界应由数据移动和寄存器预算共同决定。

### 失败三：动态形状命中错误 profile

短请求命中最优 profile，长请求落入宽泛 profile；健康探针只覆盖短输入，服务 200 状态正常但长上下文 p99 增长。修复是按长度分层压测、记录 profile 命中和 fallback，并为超范围输入明确拒绝或路由到另一池。

### 失败四：FlashAttention 走了回退

框架日志显示启用了注意力优化，实际某些变长 mask 或 head_dim 使用了通用实现。修复是把 kernel 选择和回退计数作为指标，覆盖真实 mask、dtype、布局和序列长度。不要把导入某个包视为内核已生效。

### 失败五：paged KV 页池耗尽但显存看似足够

总显存还有 8%，却无法分配新请求，因为剩余空间被不连续的小页、workspace 保留和不同租户的保留配额切碎。修复是分别记录 free pages、保留页、workspace 和各租户上限，设置高水位拒绝，并在取消路径及时回收。总字节数不能替代可分配页数。

### 失败六：INT4 文件更小但吞吐更低

权重压缩成功，然而 dequant 没有与矩阵乘融合，额外 kernel 使 decode 带宽和 launch 成本上升。修复是确认硬件指令和 kernel 支持，测量 dequant、权重读取和累加 dtype，必要时对特定层保留 FP16。压缩率不是服务吞吐的代理。

### 失败七：量化后安全拒答漂移

通用准确率提高，拒答集却出现模型执行不应执行的工具调用，因为校准集缺少安全与工具边界样本。修复是把安全集、结构化输出和工具参数纳入门禁；量化版本不满足门禁时自动回退，不允许用平均准确率掩盖高风险类别。

### 失败八：GPU 标签正确但性能异常

调度器把 Pod 放到了“同型号 GPU”节点，实际该节点处于 MIG、热降频或跨 NUMA 配置，P2P 不可用。修复是把分区、时钟、拓扑、驱动和健康探针加入能力标签，并用小型 P2P、带宽和 kernel 合成测试验证。标签是调度提示，不是性能证明。

### 失败九：编译缓存跨环境复用导致非法指令

团队把一个节点生成的编译缓存复制到整个集群，部分旧 GPU 在加载时出现非法指令，另一些节点虽然能加载却进入性能异常路径。缓存键遗漏了 compute capability、驱动和编译器版本。修复是把硬件和工具链摘要纳入键，缓存按架构分桶并签名；加载失败时删除对应条目并回到受支持的基线，而不是反复重试。

### 失败十：页表回收竞态造成旧请求可见

取消请求释放了物理页，但迟到的异步 kernel 仍持有旧页表索引；新请求随后复用了该页，出现随机乱码或跨请求数据暴露。修复是让页的生命周期包含设备事件完成和 generation 检查，回收采用延迟栅栏，任何迟到的写入都不能提交到新 generation。单元测试必须注入延迟、重复取消和 worker 崩溃。

### 失败十一：量化 scale 被错误热更新

运维人员为修复离群值，在线替换了部分 scale 文件，正在运行的 batch 同时读取新旧 scale，输出出现非确定漂移。修复是把 scale、权重和 engine 作为不可变 bundle 原子切换；热更新只改变路由指针，不修改已加载的只读张量。变更要有版本、摘要、金丝雀和回滚。

### 失败十二：功耗限制被误判为模型回归

同一 engine 在白天高峰的 token/s 比夜间低，团队误以为最近的 kernel 改动回归。实际机架功耗上限触发了 GPU 降频，温度和电源遥测与服务 trace 没有关联。修复是把时钟、功耗、温度纳入性能基线，比较前先检查热稳态，并为功耗事件设置告警。硬件限制是性能实验的一部分，不应被当作噪声丢弃。

## 19.11 取舍与被拒绝的替代方案

### 19.11.1 “全部使用 FP32”与“全部使用最低 bit”

全部 FP32 便于数值排查，却浪费显存和带宽；全部 INT4 省内存，却可能损失质量、触发 dequant 开销并缩小算子支持范围。更稳妥的是按层和张量类型混合精度：归一化、softmax、输出头和安全分类器保留较高精度，权重和部分 GEMM 使用低 bit；每个回退都应有质量与性能数据。

### 19.11.2 “只靠框架默认 auto-tune”

自动调优能快速探索 tactic，但默认搜索空间、缓存键和 benchmark 代表性可能不符合生产形状。完全手工选择又难以维护。折中是离线自动调优加线上白名单：离线用真实形状矩阵生成候选，线上只加载签名工件，遇到未知形状走有界回退并产生待优化事件。

### 19.11.3 “用更大 batch 换吞吐”

批次增大通常提高矩阵利用率，却会增加排队和显存，长短请求混合时也会恶化 TTFT。Serving Engine 应按 token 预算和 SLA 动态决定批次，并保留小批次的延迟通道。吞吐优化不能违背请求级取消和租户公平。

### 19.11.4 “用共享缓存解决所有重复计算”

共享 prefix 或编译缓存能降低成本，却引入权限、版本、失效和侧信道问题。缓存键应覆盖模型摘要、tokenizer、模板、权限、量化和设备能力；敏感租户可关闭跨域共享。命中率不是唯一目标，错误共享的代价可能远高于节省的计算。

### 19.11.5 “把优化强制绑进主分支”

优化分支往往依赖特定驱动、插件或硬件，直接替换主路径会放大回滚风险。更好的做法是能力探针、版本化引擎、按流量金丝雀和一键回退；优化关闭时仍应保留功能正确的基线路径。任何不可关闭的优化都应视为架构变化，而不是普通配置。

## 19.12 论文与仓库综合阅读

阅读论文时先回答它减少了哪一类账：算力、数据移动、控制还是内存。FlashAttention 主要减少 HBM 往返；PagedAttention 主要减少 KV 连续分配和碎片；量化方法改变表示和算术；编译器把多个算子重排和融合。阅读仓库时再核对支持矩阵、默认 dtype、回退条件、缓存键、错误处理和测试覆盖。论文中的单卡、固定长度、离线批次不能直接推出在线 p99。

建议建立“主张—证据—边界”三列表。主张如“融合减少中间写回”，证据来自论文、kernel 代码和计数器，边界是布局、形状和寄存器预算；主张如“INT8 提升吞吐”，证据必须包含目标 GPU、模型、batch、校准集和服务指标，边界是质量和长尾形状；主张如“paged KV 降低碎片”，证据要展示页池、取消和长短混合负载，而不是只看成功请求。

## 19.13 六个理解检查（附答案）

### 检查一：为什么单个 kernel 变快不等于服务变快？

答案：端到端延迟还包括排队、数据拷贝、同步、编译、显存分配和网络。kernel 若减少算术却增加布局转换，或让寄存器 spill、workspace 超预算，服务可能更慢。必须用阶段 trace 和 p95/p99 在相同流量下比较。

### 检查二：TensorRT engine 为什么不能随意跨 GPU 复制？

答案：Builder 的 tactic、指令、workspace 和插件可能依赖 compute capability、驱动、TensorRT 版本和形状 profile。另一代 GPU 可能无法反序列化、只能回退，或性能完全不同。应在目标硬件构建并记录环境摘要，必要时为每个架构保存独立工件。

### 检查三：算子融合有哪些反效果？

答案：融合会增加寄存器和共享内存压力，导致 occupancy 下降或 local memory spill；也可能阻止张量复用、扩大编译时间、限制动态形状和数值回退。融合是否有效要看数据移动、kernel 计数、资源使用和端到端指标，而不是只看代码行数。

### 检查四：FlashAttention 主要优化了什么？

答案：它通过分块和在线 softmax，避免把完整注意力分数矩阵反复写读 HBM，降低 IO 和中间内存；它不改变注意力的数学结果目标，但会改变累加顺序和数值误差。短序列、非支持 mask、布局不符或小 batch 时可能回退或收益很小。

### 检查五：paged KV 为什么仍会拒绝请求？

答案：页池是离散资源，长序列需要多个整页，尾页会产生碎片；还要预留 workspace、系统页、租户配额和并发上限。即使总空闲字节看起来足够，也可能没有连续或可分配的页，或者策略不允许该租户继续申请。应监控 free pages、保留页、页表和回收延迟。

### 检查六：量化服务的质量门禁为什么不能只看平均准确率？

答案：量化误差集中在离群通道、长上下文、数字、代码、结构化输出和安全拒答等长尾。平均准确率可能掩盖高风险类别退化。门禁应包含任务、生成格式、安全、工具参数和性能，并保留高精度回退与可审计校准工件。

## 19.14 练习

1. **Roofline 账本**：为一个 MLP 和一个 decode 注意力步估算 FLOP、HBM 字节和算术强度。改变 batch 与序列长度，预测瓶颈如何移动，再用 profiler 验证。
2. **Graph break 诊断**：写一个包含 Python 分支、自定义算子和动态列表的模型，分别运行 eager 与图编译，记录断点、编译时间、稳态时间和首次请求延迟。为每个断点提出保留、重写或回退理由。
3. **融合边界搜索**：实现三个逐元素 kernel 版本，逐步加入归一化和残差。改变输入长度与 block size，记录寄存器、occupancy、spill 和端到端 p99，找出融合开始变差的点。
4. **FlashAttention 形状矩阵**：在支持的 GPU 上比较朴素、FlashAttention 和框架回退，覆盖短/长、变长、因果 mask、FP16/BF16。输出 kernel 选择、显存峰值、误差和质量指标。
5. **paged KV 模拟器**：扩展 toy，使请求随机增长、取消和共享前缀。比较连续缓冲与不同页大小，报告页碎片、分配延迟、拒绝率和取消回收时间。
6. **量化校准**：构造包含普通、代码、数字、长上下文和安全拒答的校准子集。比较 per-tensor、per-channel、group-wise scale，记录误差、任务分数、格式有效率和模型大小。
7. **动态 profile 设计**：从真实 trace 统计 batch/长度分位点，为 TensorRT 或其他 engine 设计两个到四个 profile。验证 profile 覆盖率、构建时间、引擎大小、p99 和超范围行为。
8. **Serving Engine 故障注入**：在连续批处理服务中注入编译缓存失效、页池不足、GPU reset、慢客户端和模型版本不匹配。要求请求最终有明确结果，资源计数归零，回退可用且日志不含完整 prompt。
9. **硬件拓扑实验**：在多 GPU 节点测量同卡、同 NUMA、跨 NUMA、PCIe 和 NVLink 的复制与 AllReduce。把拓扑结果与调度标签比较，列出标签错误时的风险和修复。
10. **安全回滚演练**：故意让低精度引擎在安全集上失败，验证金丝雀摘流、旧 engine 加载、缓存失效、审计记录和用户可见错误。记录从检测到稳定恢复的时间。

每份练习报告都要注明哪些是测量、哪些是推断，给出随机种子、软件版本、硬件、输入分布、清理步骤和不可外推边界。性能图之外必须附失败样本和版本摘要。

## 19.15 总结与下一依赖

推理优化是一条从图到硬件、再从硬件回到服务契约的闭环。先用延迟分解和四本账定位主导成本，再选择图编译、TensorRT tactic、算子融合、FlashAttention、Triton kernel、paged KV 或量化。每个优化都要写出形状、dtype、硬件和回退条件；每个收益都要在真实服务流量下用 TTFT、ITL、p99、显存、错误和质量门禁验证。

编译和 engine 通过静态图与 tactic 减少控制和数据移动，但依赖版本、profile、插件和目标架构。融合减少中间写回，却可能造成寄存器溢出和动态形状退化。FlashAttention 通过 IO 感知降低注意力中间矩阵，paged KV 通过离散页改善长短请求共存，却引入页表、共享前缀和租户边界。量化降低带宽和存储成本，但需要校准、混合精度、质量门禁和回滚。硬件拓扑、温度、时钟和分区决定了实验结果能否外推。

下一章应把优化后的 engine 放回多租户平台，讨论成本计量、容量预测、发布控制和跨区域故障。优化不是结束，而是把更多隐藏假设写进可观测、可审计、可回滚的系统。

## 19.16 来源地图与可复现记录

以下来源按“论文、官方文档、实现、实验”分类。访问日期均为 2026-10-05；版本敏感项应在本地记录实际版本。

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**，Tri Dao 等，论文与实现，支持分块、在线 softmax 与 IO 复杂度主张：<https://arxiv.org/abs/2205.14135>、<https://github.com/Dao-AILab/flash-attention>。
- **FlashAttention-2**，Tri Dao，支持并行划分、负载平衡和不同 GPU 内核的机制说明：<https://arxiv.org/abs/2307.08691>。
- **vLLM 与 PagedAttention**，Kwon 等，论文与文档，支持分页 KV、连续批处理和服务接口边界：<https://arxiv.org/abs/2309.06180>、<https://docs.vllm.ai/>。
- **NVIDIA TensorRT Developer Guide**，NVIDIA，官方文档，支持 Builder、tactic、动态形状、插件、精度和 engine 约束：<https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/>。
- **NVIDIA TensorRT-LLM**，NVIDIA，参考实现，支持生成调度、量化、并行和 kernel 集成，但需按版本核对支持矩阵：<https://github.com/NVIDIA/TensorRT-LLM>。
- **Triton Language and Compiler**，OpenAI，官方仓库与教程，支持块级 kernel、编译参数和调优边界：<https://github.com/triton-lang/triton>。
- **PyTorch torch.compile 与 TorchInductor**，PyTorch，官方文档，支持图捕获、graph break、缓存和后端选择：<https://pytorch.org/docs/stable/torch.compiler.html>。
- **MLIR 文档**，LLVM 社区，支持多层 IR、方言降低和编译器结构，不证明某个模型自动获得性能：<https://mlir.llvm.org/>。
- **TVM**，Apache TVM，参考实现，支持算子调度、自动调优和多后端代码生成：<https://tvm.apache.org/>。
- **NVIDIA Triton Inference Server**，NVIDIA，官方文档，支持模型仓库、动态批处理、健康端点和后端配置：<https://docs.nvidia.com/deeplearning/triton-inference-server/>。
- **SmoothQuant**，Xiao 等，论文，支持权重-激活平滑与 INT8 校准思路：<https://arxiv.org/abs/2211.10438>。
- **GPTQ**，Frantar 等，论文与实现，支持后训练权重量化和近似误差最小化：<https://arxiv.org/abs/2210.17323>。
- **AWQ**，Lin 等，论文与实现，支持激活感知权重保护和低 bit 推理：<https://arxiv.org/abs/2306.00978>。
- **NVIDIA CUDA C Programming Guide**，NVIDIA，官方文档，支持内存层次、Tensor Core、同步与架构限制：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/>。
- **NVIDIA CUDA Best Practices Guide**，NVIDIA，官方文档，支持带宽、占用率、访存和性能实验方法：<https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>。
- **NVIDIA Multi-Instance GPU User Guide**，NVIDIA，官方文档，支持 MIG 分区、配置和隔离边界：<https://docs.nvidia.com/datacenter/tesla/mig-user-guide/>。
- **NCCL Tests**，NVIDIA，开源测试，支持多 GPU 拓扑与 collective 基准；测试结果只适用于目标硬件和配置：<https://github.com/NVIDIA/nccl-tests>。
- **Roofline: An Insightful Visual Performance Model**，Williams 等，论文与 Berkeley 资料，支持算术强度和带宽/算力判别：<https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/>。
- **OpenTelemetry Traces**，社区规范，支持跨服务阶段关联和实验记录：<https://opentelemetry.io/docs/concepts/signals/traces/>。

可复现记录至少保存：模型与 tokenizer 摘要、engine/插件/编译器版本、GPU 型号与 compute capability、驱动与 CUDA、profile、dtype、量化校准集摘要、输入形状直方图、随机种子、预热与测量窗口、p50/p95/p99、显存/功耗/温度、失败和回退计数。来源论文的硬件和模型与本地不同，必须标记哪些结果是部分复现，不能把论文倍数直接当作生产容量。

## 19.17 安全边界与上线清单

- **工件完整性**：权重、tokenizer、模板、量化 scale、插件、engine 和编译缓存绑定版本摘要并签名；加载前校验，未知来源或损坏工件隔离，不在请求路径中下载和编译。
- **精度与安全边界**：低精度路径必须经过任务、结构化输出、工具参数和安全拒答门禁；对高风险层或分类器保留高精度回退，不以平均准确率掩盖长尾错误。
- **租户与缓存边界**：paged KV、共享 prefix、编译缓存和 tactic 缓存的键包含租户权限域、模型摘要、tokenizer、模板、量化和策略版本；敏感租户可关闭跨域共享，命中与未命中指标避免暴露存在性。
- **设备与权限边界**：容器只映射分配设备和必要库，不依赖 `CUDA_VISIBLE_DEVICES` 作为唯一隔离；限制特权、主机路径、debug 接口、MIG 配置和 reset 权限，按最小权限审计。
- **资源可用性**：编译队列、engine 加载、页池、workspace、pinned memory、批次、重试和输出缓冲都有上限；高水位时返回可解释错误，禁止无限排队和无限编译。
- **侧信道与多租户**：共享 GPU、功耗、时序、页命中和编译缓存可能泄露其他租户活动；对敏感工作负载采用分区、时间抖动、容量隔离或专用节点，并记录残余风险。
- **数据最小化**：性能 trace、校准集、日志和错误样本脱敏，不保存完整 prompt、输出、token、工具凭据或跨租户上下文；设定保留期、访问审计和删除验证。
- **发布与回滚**：新 engine 先在隔离池构建、签名、预热、能力探针和金丝雀；监控 p99、质量、回退、显存和错误预算，异常自动摘流并加载已验证基线。
- **控制面失联**：worker 使用最近一次校验配置并有有效期；超过有效期只允许安全的停止或有限服务，不能接受未签名的动态优化参数。
- **硬件与供应链**：驱动、固件、CUDA、TensorRT、Triton、插件和容器镜像都有 SBOM 与升级账本；升级前验证 engine 可重建、回滚和节点排空，防止驱动或插件批量破坏服务。

上线前应逐项回答：优化关闭时基线是否仍可用；未知形状如何处理；首次编译和 engine 加载是否有预算；页池不足是否可解释且能回收；量化质量是否含安全和工具样本；回退计数是否告警；GPU 拓扑、温度和分区是否经过探针；缓存是否跨权限域；trace 是否脱敏；以及在 GPU reset、网络分区、控制面失联和租户突发下能否保住安全、完整性和可恢复性。
