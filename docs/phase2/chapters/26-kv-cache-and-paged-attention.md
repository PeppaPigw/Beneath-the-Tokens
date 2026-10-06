---
id: phase2-26-kv-cache-and-paged-attention
title: KV Cache 与 PagedAttention：把注意力历史变成可管理的工作集
description: 从两个请求和一张容量账本出发，推导 KV cache、分页块表、前缀复用与引用计数，并用 CPU toy 实验验证正确性和碎片边界
slug: /phase2/chapters/26-kv-cache-and-paged-attention
sidebar_position: 126
phase: 2
chapter_number: 26
level: foundation
prerequisites:
  - ch06-pytorch-execution
  - ch15-inference-execution
learning_objectives:
  - 能从层数、KV 头数、头维度、dtype 和上下文长度计算 KV cache 容量，并检查单位
  - 能用逻辑 token 到物理 block 的映射解释 PagedAttention，区分尾部碎片与连续张量
  - 能实现并测试 block 分配、前缀哈希、引用计数、copy-on-write 和淘汰的最小状态机
  - 能沿固定版本 vLLM 源码描述 block manager、prefix cache 和 attention kernel 的数据路径
  - 能运行 CPU 模拟实验，报告命中率、占用、碎片、p50/p95/p99 延迟和故障恢复限制
  - 能区分论文测量、源码观察、toy 测量和生产推断，写出不可迁移的边界
paper_count: 4
source_commits:
  - vllm@v0.6.3
  - vllm@v0.8.5
  - vllm@v0.10.1
  - btt-phase2@330cbfbf7f3e8d5873ddb67969e6e67e4724d0a0
  - btt-phase2@feddc3aabacc9218a8c666eb0fa5b7001303ce21
  - btt-phase2@21ae54d46f8f4e18bca1e5896fdd0b31c5bf5c65
lab_paths:
  - labs/phase2/kv_block_pool.py
last_verified: 2026-10-06
source_commit: vllm@v0.6.3
lab_path: labs/phase2/kv_block_pool.py
estimated_hours: 12
---

# KV Cache 与 PagedAttention：把注意力历史变成可管理的工作集

> **本章地图**：先看两个同时生成的请求为什么会把显存吃满，再把“每个 token 留下两份向量”画成一张可分页的工作集地图。我们手算容量、尾部浪费和一次前缀命中，随后沿 vLLM 固定版本源码追踪 block table、引用计数和 copy-on-write，最后运行一个纯 Python CPU 模拟。这里先解决本地 KV cache 与 PagedAttention 的正确性；跨机器 KV 传输、量化压缩和 connector 协议留到后续章节，不把 toy 的延迟当成 GPU 服务承诺。

## 问题边界：两个请求为什么会突然把显存用完

晚上八点，服务刚切到一个 8B 模型。请求 A 带着 28K token 的长文档进入 prefill，请求 B 带着同一份系统提示和一个很短的问题紧随其后。模型权重没有变化，batch 也只有两个请求，但监控显示：A 的首 token 还没返回，B 在队列里等待；显存从 70% 迅速涨到 98%，随后调度器暂停了新请求。工程师直觉上想“把上下文张量压紧一点”，却还没有回答三个更小的问题：

1. **每个已经生成的 token 究竟留下什么？** 只保存最后一个 hidden state 是否足够？如果不够，缺的是什么？
2. **为什么两个请求的序列长度只差一个 token，分配器却可能浪费一整页？** 浪费来自数据本身、连续内存约束，还是调度策略？
3. **A 和 B 的前缀相同时，能否共享已经算过的历史？** 如果 B 继续生成并修改最后一页，怎样避免把 A 的结果写坏？

本章只研究 decoder-only Transformer 的自回归生成路径。我们假设权重和 tokenizer 已经加载，输入 token id 已经确定；关注从 prefill 产生 K/V、到 decode 每一步读取历史、再到请求结束释放物理存储的生命周期。训练时的反向激活、检索系统的向量索引、跨节点传输、KV 量化的质量误差不在本章主线中，但会在失效边界中说明它们如何改变假设。

先把可观察现象写成一个小契约。请求 `r` 有 token 序列 `x_0 ... x_{T-1}`，模型有 `L` 层。对每层、每个历史位置，注意力需要一个 key 向量和一个 value 向量；decode 第 `t` 步的 query 要与所有 `0..t-1` 的 key 做点积，再用权重加权 value。**事实**：只保留最后一个 token 的 K/V 会改变注意力结果，因为旧 token 仍可能获得非零权重。**设计判断**：因此 serving 系统必须把历史 K/V 当作可增长的工作集管理，而不能把它当成一次性中间结果。

另一个边界是“缓存命中”的含义。本章的命中指：同一模型语义、同一位置编码约定、同一 KV 布局和允许共享的租户域内，某段 token 前缀的 K/V 已经存在，可以直接读取。命中不等于模型回答正确，也不等于跨租户授权通过；哈希只帮助定位内容，不能替代 ACL。后文会把“计算结果可复用”和“数据可以被这个请求读取”分开。
## 直觉模型：把 GPU 显存看成一座有页表的仓库

先不要记住 PagedAttention 这个名字。想象仓库里有许多同样大小的货架格子，每个格子能放 16 个 token 在一层中的 K/V。请求手上只有一张逻辑清单：第 0–15 个 token 在“第 0 格”，第 16–31 个 token 在“第 1 格”。仓库管理员把这些逻辑格子放到任意空闲的物理货架上，并在一张小表里记录“逻辑格子 0 → 物理格子 17”。模型读取历史时，先查表，再按物理格子顺序 gather；请求看起来连续，显存不需要连续。

```
逻辑序列（请求 A）:  token 0 ... 15 | 16 ... 31 | 32 ... 47
                              │             │            │
block table:                 [  7 ]       [  2 ]        [ 19 ]
                              │             │            │
物理池:        block 0  block 1  block 2 ... block 7 ... block 19 ...
```

货架固定大小会产生一个可计算的尾部浪费：若请求只有 17 个 token，第二个格子只用 1 个位置，其余 15 个位置暂时空着。连续大张量分配则会产生另一类浪费：即使总空闲字节足够，只要空闲区域被切成几段，就无法满足一个更大的连续请求。分页把“逻辑连续”和“物理连续”拆开，代价是查表、块边界处理和 kernel 的 gather 逻辑。

现在加入共享。A 的系统提示占满前两页，B 的前缀完全相同；仓库管理员让两张逻辑清单都指向物理块 7 和 2，并把引用计数从 1 加到 2。A 继续生成时要写第三页，若第三页也被共享，管理员不能直接覆盖；他先复制出一份新的物理块，再让 A 的清单指向副本，这就是 copy-on-write（写时复制）。只读共享减少计算和存储，写时复制保护了已经完成的前缀。

这个比喻里有三个必须落地的状态：

- **逻辑 block table**：按请求记录 block 顺序、有效 token 数和最后一页是否可写。
- **物理 block pool**：固定大小的字节区域、空闲队列、引用计数、内容哈希和可淘汰标记。
- **生命周期事件**：allocate、append、share、cow、release、evict。任何漏掉的 release 都会表现为“缓存命中率很好但可用块越来越少”。

第一次出现的几个术语现在可以和运行时对象对应：KV cache 是物理块中的 K/V 张量；PagedAttention 是使用 block table 访问这些非连续块的注意力计算；prefix cache 是以 token 前缀哈希定位可共享块的索引；refcount 是物理块的生命周期保护。术语不是额外的魔法，它们只是仓库清单、页表和借阅记录的名字。
## 最小例子：先手算一页，再算一个共享前缀

### 最小例子续算：3.1 容量的数字账本

取一个简化模型：`L=2` 层，每层 `n_kv_heads=2` 个 KV 头，每头维度 `d=4`，dtype 为 FP16（每个元素 2 字节）。一个 token 在一层需要 K 和 V 两个向量，因此字节数是：

```
每层每 token = 2（K、V） × 2（KV 头） × 4（head_dim） × 2（bytes）
             = 32 bytes
全模型每 token = 2（层数） × 32 = 64 bytes
```

如果请求历史长度为 10 token，理想的 K/V 数据是 `10 × 64 = 640 bytes`。设置 `block_size=4` 后需要 `ceil(10/4)=3` 个 block，每个 block 容纳 `4 × 64=256 bytes`，物理占用 `3 × 256=768 bytes`，尾部未使用 `768-640=128 bytes`，占物理容量约 16.7%；最后一个 block 有 2/4 的 token 槽位未用（50% 的尾部块容量）。这里的百分比是本 toy 的内部碎片，不是 GPU allocator 的真实碎片率。

写成一般形式。设 `b` 是每个元素的字节数，`h_kv` 是每层 KV 头数，`d_h` 是每头维度，`L` 是层数，`T` 是 token 数：

```
bytes_per_token = 2 × h_kv × d_h × b
bytes_total      = L × T × bytes_per_token
blocks           = ceil(T / B)
bytes_paged      = blocks × B × bytes_per_token × L
internal_waste   = bytes_paged - bytes_total
```

`B` 是每个 block 的 token 容量。式子假设每层 K/V 都使用相同 dtype、没有额外 scale、对齐和元数据；量化、混合精度、不同层的 head_dim、对齐 padding 都会让真实值偏离。最容易出错的是把 query head 数写成 KV head 数。GQA/MQA 中 query 头可以很多，但 cache 只存 `h_kv` 组；把 32 个 query heads 当成 8 个 KV heads 会把容量估算放大四倍。

### 最小例子续算：3.2 Llama 风格的数量级检查

再取一个接近真实部署、但只用于心算的配置：`L=32`、`h_kv=8`、`d_h=128`、bf16 2 字节。每 token：

```
2 × 8 × 128 × 2 × 32 = 131,072 bytes ≈ 128 KiB
```

32K token 的单请求理想 KV 是 `131,072 × 32,768 = 4,294,967,296 bytes`，按二进制约等于 4 GiB；若 `B=16`，最后一页平均会浪费约 8 token 的容量（随机长度、块对齐且忽略调度相关性时的期望），单请求额外约 `8 × 128 KiB = 1 MiB`。**推断**：在这个配置下，多个长上下文请求会很快把“剩余显存”转化为并发上限；提高 batch 而不计算 KV 工作集，可能只是在增加排队和驱逐。

### 最小例子续算：3.3 一次前缀共享

A 的 token 序列是 `[11, 22, 33, 44, 55, 66]`，B 是 `[11, 22, 33, 44, 55, 66, 77]`，`B=4`。A 的逻辑页是 `[11,22,33,44]` 和 `[55,66]`；B 查找前缀时，第一页完整匹配，第二页只有前 2 个 token 与 A 相同，是否能共享取决于实现的粒度：

- **整块前缀缓存**：只共享完整块 `[11,22,33,44]`，B 的第二页重新计算并在其中追加 77；逻辑简单，命中边界清楚。
- **部分块共享**：需要记录有效 token 数和写入权限，B 不能覆盖 A 的 55、66，通常先 COW 出一个可写块；节省计算，但状态更多。
- **按 token 的连续张量缓存**：可以共享 6 个 token，但要维护更细粒度的引用和地址，元数据、kernel 和淘汰成本增加。

本章的 toy 默认采用完整块作为 prefix cache 单位，同时显式模拟最后一块的有效长度和 COW。这样能先验证最重要的不变量：共享块只读，写入前必须复制；前缀命中只能从左到右连续查找，遇到第一个不匹配就停止。
## 3.4 最小例子再走一遍：同一个请求在三个时刻的内存账本

为了把“逻辑长度”和“物理占用”分开，重新跟踪请求 C。模型配置沿用前面的简化值，每个 token 理想占 64 bytes，块大小为 4，因此每块槽位是 256 bytes。

**时刻一：输入 3 token。** C 的页表只有 `[5]`，有效长度为 3。理想数据是 `3×64=192 bytes`，物理分配是 256 bytes，内部浪费 64 bytes。若这时把块发布到完整块前缀索引，答案应为“不发布”，因为块尚未填满；如果产品要求缓存短 system prompt，也可以发布，但必须把“部分块可共享、写入前 COW”作为明确协议。

**时刻二：追加第 4 token。** C 仍指向块 5，有效长度变为 4。理想数据 256 bytes，物理数据也 256 bytes，尾部浪费为零。此时 toy 可以计算 whole-prefix hash 并原子发布；生产链式索引会在完整块边界发布。注意发布前要确认四个 token 都来自同一模型语义和位置起点；只对前三个 token 的 hash 做索引，会让后来追加的 token 覆盖一个本应 immutable 的条目。

**时刻三：追加第 5 token。** C 申请物理块 9，页表变为 `[5,9]`，第一个块仍是完整共享块，第二个块有效长度为 1。理想数据为 320 bytes，物理分配 512 bytes，内部浪费 192 bytes。若 C 此时结束，块 5 可以继续留在 prefix cache，块 9 是否发布取决于实现是否缓存部分块；若不发布，释放 C 后块 9 回 free，块 5 仍由 cache owner 保留。

同一个请求在三个时刻的物理占用依次为 256、256、512 bytes。若监控只记录“每 token 平均字节”，就会在第五个 token 突然看到 102.4 bytes/token，误以为模型 dtype 变了；实际上变化来自块对齐。容量仪表应同时展示 logical tokens、allocated slots 和 waste slots。
## 3.5 最小例子续算：分页前后注意力输出为什么应该相同

容量表证明了分页如何存数据，还没有证明读取后模型仍算同一个结果。现在只取一个注意力头、二维向量、三个历史位置，完全在纸上计算。令 query 为 `[1,0]`，三个 key 分别是 `[1,0]`、`[0,1]`、`[1,1]`，value 分别是 `[10,0]`、`[0,20]`、`[30,30]`。按缩放点积注意力，分数为 `[1,0,1]/sqrt(2)`，约为 `[0.707107,0,0.707107]`。

为稳定计算 softmax，先减最大分数，得到 `[0,-0.707107,0]`，取指数约为 `[1,0.493069,1]`，总和约为 `2.493069`。三个权重约为 `[0.401112,0.197776,0.401112]`。输出第一维为 `0.401112×10+0.197776×0+0.401112×30≈16.04448`；第二维为 `0.401112×0+0.197776×20+0.401112×30≈15.98888`。这是连续布局的参考输出，具体末位以运行中的浮点结果为准。

现在设 `B=2`，前三个位置分别放到物理块 9 的槽 0、槽 1，以及物理块 3 的槽 0，请求页表是 `[9,3]`。读第零个逻辑位置时查 `table[0]=9`，页内偏移为零；读第二个逻辑位置时查 `table[1]=3`，页内偏移仍为零。若 gather 的次序仍是原来的三个位置，分数、softmax 和加权和就完全相同。**推导**：分页改变的是存储地址，数学算子没有改变；只有浮点归约顺序或具体 kernel 实现可能引入数值误差。

这个例子可以构造三种故障。第一，把页表误写成 `[3,9]`，并且没有同步调换位置、mask 或 RoPE，模型读取的逻辑上下文就错了。第二，把最后一个物理块未写入的槽也算进去，多出第四个 key/value，softmax 分母改变，即使该 value 为零，原来三个位置的权重也会被稀释。第三，只重排 key 不重排 value，注意力分数仍然可能“看起来合理”，但权重乘到了错误的内容上。因而数值正确性测试必须验证 K/V 配对、顺序和有效长度三个维度，单看输出是否为有限数远远不够。

再看一个更隐蔽的点：如果没有位置编码，而且同时用相同的置换重排全部 key/value，单个 query 的集合注意力和可能不变。这不意味着页表顺序无关，因为真实 decoder 还依赖因果 mask、位置编码、窗口边界和当前 token 的写入位置。测试不能只用对置换不敏感的玩具输入；至少加入位置相关 key 或按逻辑位置生成的 mask，才能抓到表顺序错误。这里的验收标准应写成“在相同逻辑位置语义下，分页与连续输出在容差内一致”，而不是“任意重排都无影响”。
## 正式定义与推导：从注意力公式到页表访问

### 正式定义补充：4.1 自回归注意力需要什么历史

对第 `l` 层、时间步 `t`，设 query 为 `q_t^l ∈ R^{h_q×d_h}`，历史 key/value 为 `K_{0:t}^l`、`V_{0:t}^l`。忽略多头 reshape、RoPE 和 mask 的细节，注意力输出可以写为：

```
A_t^l = softmax((q_t^l (K_{0:t}^l)^T) / sqrt(d_h) + mask) V_{0:t}^l
```

`K` 的行按 token 位置排列，`V` 同样按位置排列；`mask` 保证位置 `t` 不读取未来 token。decode 时 `q_t` 只有一个新位置，但矩阵乘的另一侧长度随 `t` 增长，所以每一步都需要遍历历史 K/V。prefill 可以把多个 query 一起算，decode 通常更受内存带宽和 kernel launch 影响。**事实**：KV cache 通过保存每层历史 K/V，避免每个 decode 步重新计算旧 token 的投影；它没有消除读取历史的成本。

对 GQA/MQA，`h_q` 与 `h_kv` 不同。多个 query head 映射到一组 K/V head，容量公式使用 `h_kv`；注意力 kernel 在读取时做 head mapping 或广播。若 checkpoint、张量并行切分或 RoPE 版本不匹配，数值上“形状相同”仍不代表语义相同，这在跨进程或持久化缓存时尤其危险。

### 正式定义补充：4.2 逻辑位置到物理块

设 block 容量为 `B`，逻辑 token 位置 `i` 的页号和页内偏移：

```
logical_block = floor(i / B)
offset        = i mod B
physical_block = block_table[logical_block]
```

在连续 cache 中，`physical_block` 隐含为 `base + logical_block`；在分页 cache 中，它是一个查表结果。每层、每个 KV head、每个 offset 的地址还要乘上 dtype stride、head_dim stride 和层/批次布局。一个教学级布局可以写成 `[num_physical_blocks, num_layers, 2, B, h_kv, d_h]`，其中 `2` 表示 K/V；生产 kernel 常把维度重排成更适合 coalesced load 的顺序，也可能按 layer 分页。**定义**：block table 只描述逻辑→物理的映射，不承诺物理张量的具体 stride；metadata 必须同时记录 layout 才能安全解释一块字节。

注意力 gather 的伪代码如下：

```python
for logical_block, physical_block in enumerate(block_table):
    valid = min(B, seq_len - logical_block * B)
    kv = pool[physical_block, layer, :, :valid, :, :]
    scores, values = dot_and_weight(query, kv)
    accumulate(scores, values)
```

真实 kernel 不会逐块启动 Python 循环，而是把 block table、sequence length、head mapping 传给 CUDA/Triton kernel，在一个 CTA 或 warp group 中循环。伪代码的价值是暴露两个边界：最后一块只有 `valid` 个 token，不能读取未写入槽位；block table 中任意物理编号都必须经过范围和生命周期检查。

### 正式定义补充：4.3 前缀哈希链

完整块前缀缓存需要把 token 内容映射为稳定 key。生产实现常见的一种构造是链式哈希（本节先用它解释为什么可以逐块停在第一个 miss）：

```
h_0 = H(model_fingerprint || tokenizer_fingerprint || token_ids[0:B] || metadata)
h_i = H(h_{i-1} || token_ids[iB:(i+1)B] || metadata)
```

`metadata` 至少包含 RoPE/位置约定、LoRA 或 adapter 版本、dtype/layout、允许共享的 tenant 域和 block size。链式设计让生产索引可以逐块匹配：第 `i` 块的 hash 依赖前一块，前一块 miss 后面的 hash 即使相同也不能直接当成同一上下文。**本章 toy 的实现边界**：`KVBlockPool.hash_prefix` 对整个 token tuple 做带 namespace 的规范化 SHA-256，`longest_prefix` 线性扫描缓存条目并比较 token slice，语义上仍返回最长前缀但没有 O(1) 链式索引。**事实**：哈希碰撞在密码学哈希下概率很低，但“低概率”不是授权机制；读取前仍要检查租户和模型 fingerprint，并可用 token ids 或 checksum 做二次校验。

对生产链式索引，命中算法可以从 `i=0` 开始计算块 key，遇到第一个 miss 就停止，返回命中的 token 数和需要 prefill 的剩余区间；这个算法不自动支持中间片段拼接。对本章 toy，`longest_prefix` 会枚举已缓存的完整 token tuple、比较请求前缀 slice，再选最长条目，复杂度是线性扫描而不是生产索引的性能承诺。要复用非前缀内容，需要额外的 CacheBlend 或检索式机制，并重新计算连接处的状态。

### 正式定义补充：4.4 引用计数、写时复制和不变量

把物理块 `p` 的状态写成 `(refcount, immutable, valid_tokens, hash, tier)`。当一个请求获得共享引用时，`refcount += 1`；请求释放或驱逐其引用时，`refcount -= 1`。以下不变量是 toy 和生产实现都应该能单测的：

1. `refcount == 0` 的块不得出现在任何活跃请求的 block table 中。
2. `refcount > 1` 的块标记为只读；对它 append 必须先 COW，旧块的内容和 hash 不变。
3. `valid_tokens` 在 `[0, B]` 内；最后一块以外的共享块通常必须恰好为 `B`，否则 hash 与位置语义要显式记录。
4. 只有完整写入并校验后，块才进入 prefix index；正在写的块不能被另一个请求命中。
5. 驱逐只允许选择 `refcount == 0` 且未被 in-flight DMA/kernel 使用的块。

COW 的成本是一个可测的权衡。若 beam search 的多个分支共享长前缀，读共享几乎免费；一旦每个分支都在最后一页写入不同 token，系统需要复制块并增加显存。复制粒度越大，带宽和尾部浪费越高；复制粒度越小，元数据和 kernel 复杂度越高。**设计判断**：教学实现先以 block 为 COW 单位，生产实现应基于实际 block size、写频率和 kernel 支持评估更细粒度方案。
## 4.5 正式定义补充：缓存为何通常保存 K/V，而不是每轮的注意力矩阵

初学者经常问：既然每一步都要读历史，为什么不把上一步的注意力权重一起缓存？关键在 query 变了。第十个 token 的 query 与第十一个 token 的 query 是不同向量；即使历史 key 不变，它们与历史的点积也不同。上一步的 softmax 权重不能直接用于下一步。K/V 则是每个旧 token 在各层的投影，在通常的因果 decoder 前向语义下，未来 token 不会反过来改变已经生成的旧位置表示，因此旧 K/V 可以复用。

用最小向量看：历史 key 是 `[1,0]` 与 `[0,1]`。query `[1,0]` 偏向第一个 key，query `[0,1]` 偏向第二个 key；同一套缓存支持两次不同的打分，但旧打分不能替代新打分。缓存 query 同样没有太大价值，因为每一步只需要当前 query，除非在训练、批量验证或特殊重算路径中需要旧 query。**推导**：可缓存性来自依赖图，不能由“这个张量很大”决定。任何改变依赖图的结构，例如非因果注意力、修改前缀 token 或改变位置约定，都需要重新判断旧 K/V 是否仍有效。

还要区分 KV cache 与训练激活检查点。训练保存或重算激活是为了反向传播，需要梯度、随机数状态和算子上下文；推理 KV cache 是为了未来自回归步骤读取。两者都叫 cache 或 checkpoint，却有不同的生命周期、数值要求和回收时点。把训练的 activation checkpointing 开关用于解释推理 KV 容量，通常是对象混淆。
## 4.6 正式定义补充：多头、分组查询和多查询的三本账

假设模型隐藏维度为 4,096，query head 数为 32，每头 128 维。常规多头注意力使用 32 个 K/V heads；多查询注意力让 32 个 query heads 共享一个 K/V head；分组查询注意力介于两者之间，例如八个 K/V heads、每组四个 query heads。三种结构不是同一个 checkpoint 的任意运行开关：投影权重形状和训练过程都参与定义模型。

以 32 层、bf16、8,192 token 为例，三本理想 KV 账本如下：

- 32 个 KV heads：每 token 全模型 512 KiB，8K token 约 4 GiB。
- 8 个 KV heads：每 token 全模型 128 KiB，8K token 约 1 GiB。
- 1 个 KV head：每 token 全模型 16 KiB，8K token 约 128 MiB。

这是容量推导，不是质量排名。MQA 的共享方式减少了保存和读取的 K/V 字节，但也改变模型表达能力；GQA 通过中间组数取得折中。服务团队不能简单把配置中的 KV heads 从八改成一并期待同一模型“无损省八倍”，因为原来的投影权重、head mapping 和训练假设会被破坏。

Shazeer 的 MQA 论文把增量解码反复读取 K/V 的带宽成本作为问题，提出跨 query heads 共享 K/V；Ainslie 等人的 GQA 论文研究从多头 checkpoint 进行 uptraining，并用中间数量的 KV heads 折中速度与质量。两者支持“head 数是容量公式的重要结构参数”这个因果解释；它们的质量测量不能替代你的模型评估。本文不搬运它们的速度倍数，避免把不同硬件和训练配方混到 PagedAttention 对照中。
## 4.7 正式定义补充：从容量上限推到安全并发

一张卡的显存不能全部分给 KV。假设可用于教学预算的总量是 24 GiB，权重占 15 GiB，运行时 workspace、激活峰值与图捕获等预留 3 GiB，保留 1 GiB 余量，那么 KV 池预算只剩 5 GiB。对于每 token 128 KiB 的配置，理想上限约为 `5×2^30/2^17=40,960` token。若每个请求当前长度都是 8,192 token，理想上限是五个请求；但若它们还要各生成 1,024 token，最终总量是 46,080 token，已经超过预算。

于是准入至少有两种策略。保守预留按 `输入长度+最大输出长度` 预留全部块，易于保证不会中途耗尽，但用户经常提前停止时会浪费可用并发。增量分配只按已写 token 加少量 lookahead 预留，利用率高，却必须有可靠的抢占、重算、排队或拒绝机制处理增长。PagedAttention 让增量分配更灵活，并没有替你决定哪种风险可接受。

假设五个请求都声明最多生成一千 token，实际只有一个生成九百、其它各生成一百。保守方案始终为五千输出 token 留空间；增量方案只逐步占用一千三百 token，可接纳更多短请求。但若五个请求同时触及上限，增量方案必须在安全边界前停止接纳，或把一部分请求移出运行集。**设计判断**：面向交互服务时，应记录抢占次数、重算 token、排队尾延迟和输出取消率，而不是只表扬“更高显存利用率”。容量使用率高但持续重算，可能在消耗同一 GPU 的计算资源来补内存不足。

并发公式还要用当前长度分布，而不是平均数。十个 1K 请求与一个 10K 请求的理想 token 总量相同，但其尾部浪费、kernel 并行度和每步调度开销不同。长请求可能占住块很久，短请求来得快去得快。生产容量模型应保留长度与生命周期的联合分布；不能把平均输入长度乘平均输出时间当成可靠峰值预算。
## 4.8 正式推导：前缀共享节省的上限

设有 `N` 个请求，每个请求前 `P` token 完全相同，之后的后缀互不相同，块大小为 `B`。若只计算可完整共享的块，公共部分的完整块数为 `floor(P/B)`；无共享时每个请求仍各自占用这些完整块，完整块共享后的理想节省为 `(N-1)×floor(P/B)` 个块。最后一个部分块、后缀和 COW 另行计算。若每块有 `S_block` bytes，则完整块节省字节是：

```
Saved_bytes_ideal = (N - 1) × floor(P / B) × S_block
```

这里使用 `floor(P/B)` 是因为只计完整可共享块；如果实现支持部分块 cache，需要单独算 COW 和有效长度。这个上限不含 hash、元数据、复制、最后 logits 重算和 decode 读取成本，也不意味着请求可以无限增加。公共前缀若跨租户不能共享，`N` 应按每个授权域分别计数，而不是把全局请求数代入。

举例：`N=8、P=1,000、B=16`，完整公共块数为 62。无共享需要 `8×ceil(1000/16)=504` 块；只共享完整公共块时，62 个公共块之外仍需每个请求自己的部分尾块，共 `62+8=70` 块，理想节省 434 块。若每块是 2 MiB，数字看起来很大；但实际模型可能每层独立分片、某些块因滑动窗口不再需要、或共享最后一个完整块要付出额外调度成本。研究报告应把“公式上限”与“实验观测”分成两行，并解释差额来自哪些边界。

当后缀长度很短时，共享可显著降低物理占用，却未必降低 wall-clock。所有请求仍要做 tokenization、调度和至少一次 logits 计算；block table gather 也可能让 kernel 变得更复杂。反过来，公共前缀很长时，节省的 prefill 计算和块容量可能同时提高有效吞吐。**推断**：共享收益应以 saved prefill tokens、saved physical blocks 和 TTFT/TPOT 的关联来验证，而不是由 hit ratio 单独推断。
## 4.9 正式定义补充：从一层扩展到整张卡：布局、分片和生命周期

前面的公式把所有层叠加成一个数字，但实现时每层的地址不能混成一段没有边界的字节。考虑一个两层、四个物理块的教学布局。每块有 `B=4` 个位置，每个位置有 `h_kv=2` 个头、每头 `d_h=4` 个元素。可以把一块看作：

```
block p
 ├─ layer 0
 │   ├─ K: [offset 0..3][kv_head 0..1][dim 0..3]
 │   └─ V: [offset 0..3][kv_head 0..1][dim 0..3]
 └─ layer 1
     ├─ K: ...
     └─ V: ...
```

这种布局便于演示“一个物理块同时承载所有层”，但真实实现常按层分组或为不同 attention 类型使用不同 page size。原因是 GPU kernel 在处理某一层时只需读取本层 K/V，跨层交错会增加 stride；另一方面，按层分配又会让 allocator 维护更多 pool。**设计判断**：布局选择要和访问粒度一起评估，不能仅按总字节数比较。若每层的 block 都单独编号，`block_table` 可能需要一个 layer offset；若整组层共享一个 block id，释放和复制必须原子地覆盖整组。

张量并行进一步改变“一个请求拥有一块 K/V”这句直觉。假设四张 GPU 做 tensor parallel，每张卡只保存部分 query/KV head，那么一个逻辑 block 对应四个 rank-local 物理块。请求的 block table 可以是全局 block id 加 rank 映射，也可以每个 rank 维护独立表；两者都要求 rank 间的 token 边界、层顺序和 epoch 一致。某个 rank 先释放而另一个 rank 仍在 kernel 中读取，会产生只在高并发下出现的错误。因此，跨 rank 的块回收需要同步协议或每个 rank 的完成票据，不能把 Python 端引用计数当成全局完成。

当 pipeline parallel 把不同层放在不同 stage，KV cache 的所有权也随 stage 分裂。prefill 的 token 经过 stage 0、1、2，最后每个 stage 写自己的 K/V。decode 时每个 stage 读取自己层的历史，stage 间只传 hidden state 和控制信息。一个全局容量公式仍然有用来估算总量，但 admission 时要看每个 stage 的最小剩余块数：任一 stage 没有足够块，整个请求都不能安全启动。**失败边界**：只监控总显存而不监控 per-rank KV 使用率，会让一个 rank 先 OOM，随后整个 batch 失败。

生命周期还要区分三种“完成”：

1. **写入完成**：新 token 的 K/V 已写到目标地址，kernel 或 DMA event 已记录。
2. **可读完成**：后续 attention stream 与写入 stream 建立了依赖，不会读到半写数据。
3. **可共享完成**：checksum、metadata 和 hash index 已原子提交，其他请求可以命中。

这三者的时间可能不同。把块在第一个 event 后就放进 prefix index，另一个请求可能读到尚未完成的最后一层；把块在第三个 event 后才可读，则会牺牲一点命中延迟换正确性。生产系统应在接口上区分 `ready_for_local_read` 和 `published_for_share`，而不是一个布尔值覆盖所有状态。
## 4.10 正式定义补充：prefill 与 decode 的时间线：同一个 cache，不同的瓶颈

画一条简化时间线有助于理解为什么“减少 prefill 计算”不等于“每步 decode 更快”。请求输入有 2,048 个 token，输出 32 个 token：

```
时间 ───────────────────────────────────────────────────────────────>
Prefill:  [一次处理 2048 token，写入 2048/B 个 block]
Decode:                         [q+read KV][append]×32
```

prefill 可以在 batch 维度和序列维度上并行，矩阵乘的算术强度通常较高；decode 每次只有一个新 query，却要读取不断增长的 K/V，常常受 HBM 带宽、cache gather 和调度开销约束。prefix cache 命中把 prefill 的一段去掉，但 decode 仍需读取命中的历史块。若服务目标是 TTFT，命中可能带来明显改善；若目标是每 token 时间（TPOT），收益取决于读带宽、batch 和 kernel 是否能有效合并请求。

用一个不可冒充真实设备的数字例子：假设 toy 中前缀 prefill 操作每 token 计 1 个单位，decode 每 token 读取历史计 `0.02×历史长度` 个单位。一个 1,000 token 前缀、生成 20 token 的请求，冷 cache 的计算账本为 `1000 + Σ_{i=1}^{20}0.02(1000+i) ≈ 1,400` 单位；命中完整前缀后账本约为 `Σ 0.02(1000+i) ≈ 400` 单位，节省主要来自 prefill。若前缀只有 20 token，命中节省约 20 单位，而 decode 读历史仍约 8 单位，额外 hash/refcount 开销可能抵消收益。这个模型不是 GPU 性能模型，却提醒我们要按 TTFT、TPOT 和总成本分别测量。

**实验设计判断**：压测报告至少拆出 `prefill_tokens`、`decode_tokens`、`cache_hit_tokens`、`time_to_first_token` 和 `inter_token_latency`。只报告总 tokens/s 无法告诉读者收益来自哪一阶段；只报告 TTFT 又可能掩盖长输出的 TPOT 回归。
## 4.11 正式定义补充：block size 的推导：为什么不存在万能页大小

设请求长度随机变量为 `T`，块大小为 `B`。忽略共享时，内部浪费 token 数为 `W = B ceil(T/B) - T`。当 `T mod B` 在 `0..B-1` 近似均匀时，`E[W] ≈ (B-1)/2`；相对浪费约为 `E[W]/E[B ceil(T/B)]`。这个近似在长度分布均匀时有用，但在真实 trace 中长度常集中在系统 prompt、对话轮次或最大长度附近，余数并不均匀。

加入 prefix cache 后，块大小还影响命中粒度。公共前缀长度为 `P`，完整块命中 token 数是 `B floor(P/B)`；尾部不足一块的公共 token 不能直接共享，除非实现支持部分块 COW。`B` 大会减少 block table 条目和 hash 次数，却可能把 `P mod B` 的计算重新做一遍；`B` 小会提高命中粒度，却增加表查找、元数据和 kernel 边界分支。

可以把一个选择写成成本函数，而不是凭经验选参数：

```
C(B) = α × E[internal_waste_bytes(B)]
     + β × E[prefix_recompute_bytes(B)]
     + γ × metadata_entries(B)
     + δ × gather_overhead(B)
```

`α、β、γ、δ` 来自业务目标：显存紧张时 α 大，TTFT 敏感且公共前缀长时 β 大，CPU 调度成为瓶颈时 γ 大，kernel 对小块边界敏感时 δ 大。这个函数是设计判断，不是自然定律；实验需要用真实 trace 估计四项，再看哪个 B 在约束下可接受。若没有 trace，至少报告长度分布、prefix 分布和参数扫描，不要只给一个“推荐 16”。
## 4.12 正式定义补充：前缀命中算法的细节：完整块、部分块和模板 token

聊天服务通常把 system、developer、user 消息拼成模板，再交给 tokenizer。人眼看到“相同系统提示”，token id 不一定相同：模板可能插入不同的 role 标记、空格、BOS/EOS，Unicode 规范化也可能改变字节序列。prefix cache 必须以最终 token ids 为准；如果系统在 tokenizer 前就计算字符串 hash，命中语义可能错误。

完整块命中伪代码：

```text
lookup(request_tokens, metadata):
  previous = ZERO_HASH
  for start in range(0, len(tokens), B):
      chunk = tokens[start:start+B]
      if len(chunk) < B: break
      key = H(previous || chunk || metadata)
      block = index.get(key)
      if block is None or not compatible(block, metadata): break
      retain(block)
      table.append(block.id)
      previous = key
  return table, len(table)*B
```

两个边界值得单独测试。第一，最后一块不完整时通常不进入可共享索引，因为另一个请求若继续写入需要 COW；若实现允许部分块索引，key 必须包含有效 token 数且块必须 immutable。第二，metadata 的 `position_offset` 要和模型位置编码语义一致。对 RoPE，若相同 token ids 在不同起始位置会产生不同 K/V，那么只 hash token ids 会错误命中；对某些相对位置编码，前缀是否可平移复用又是另一个模型特性，不能一概而论。

命中失败要有可观测分类：`token_mismatch`、`metadata_mismatch`、`tenant_denied`、`hash_collision_suspect`、`block_not_published`、`capacity_miss`。把所有 miss 归成一个计数会让容量问题和安全拒绝混在一起。指标标签要控制基数，不要把完整 request id 或 token 文本放进 Prometheus label；详细原因可以在采样日志或受控 trace 中记录。
## 4.13 正式定义补充：淘汰和准入：缓存不是越大越好

prefix cache 的块即使 immutable，也会占住物理池。最简单的 LRU 以最后访问时间排序，但“最近访问”不总等于“重算代价高”。一个 2K token 公共前缀命中一次后可能很久不再出现；一个 128 token 前缀每秒命中百次，LRU 更偏好后者。可以用一个教学成本估计：

```
keep_score = reuse_probability × recompute_cost - transfer_or_eviction_cost
```

其中 `recompute_cost` 可按 token 数和层数估算，`reuse_probability` 可由历史命中频率平滑，`transfer_or_eviction_cost` 先用本地释放成本近似。这个 score 不是直接生产策略，而是帮助读者理解为什么“严格 LRU”只是一个 baseline。真实实现还要纳入租户配额、内存 tier、in-flight 引用和公平性。

准入策略也应有边界。若长请求一次就填满池子，直接让它进入 prefix cache 可能把短请求都驱逐。可设最小复用次数、最大单租户占用或按公共前缀长度分桶；若请求被取消，未完成的块不得发布。**设计判断**：缓存策略的目标不是最大化命中率，而是在正确性、安全、TTFT/TPOT 和内存预算下最大化可服务请求数。
## 机制与源码入口：找出 waiting 何时变成 running

现在把心智模型映射到一个固定版本。我们选择 vLLM `v0.6.3` 作为容易阅读的 V1 过渡版本，另外记录 v0.8.5 和 v0.10.1 只是为了提醒读者 API 会演进。源码阅读不是“看过目录就算理解”，而是带着一个问题：**当一个请求的前缀命中时，哪个对象把物理块加入请求；当请求要追加 token 时，谁保证写入不会覆盖共享块？**

### 5.1 源码阅读任务和固定路径

在 `vllm-project/vllm` 的 `v0.6.3` tag 中，先看以下路径（链接和版本见来源地图）：

- `vllm/core/block_manager_v1.py`：V1 的逻辑 block 与物理 block 分配、释放、`can_append_slots` 检查；阅读 `BlockSpaceManagerV1` 的状态变更，而不是只看类名。
- `vllm/core/block_manager_v2.py`：对照 V2 的 block manager 接口和调度器交界，关注 sequence group 何时申请新 block；不要把 V2 的类名倒灌到 V1。
- `vllm/worker/cache_engine.py`：物理 KV cache 的层级分配和 swap/copy 操作，关注 GPU/CPU slot 的布局约定。
- `vllm/attention/layer.py` 与对应 attention backend：把 `block_table`、`seq_lens` 传给 kernel 的位置；不同 tag 的 backend 文件名会变，必须以 tag 中的注册表为准。
- `vllm/v1/core/kv_cache_utils.py`（v0.10.1 tag）：检查 page size、layer 组和 hybrid manager 的容量计算，不要把新路径反推成旧版本已经存在。

源码链接应固定到 tag 或 commit，例如 `https://github.com/vllm-project/vllm/blob/v0.6.3/vllm/core/block_manager_v1.py` 或 `.../block_manager_v2.py`，而不是 `main`。如果你在当前 main 看到更清晰的 `kv_cache_manager.py`，可以把它作为“后续实现观察”，不能拿来证明 v0.6.3 的行为。

### 5.2 正常路径：prefill 到 decode

一次新请求通常经历以下抽象状态（具体枚举随版本变化）：

```
WAITING --admit--> RUNNING --append token--> RUNNING
   │                   │                         │
   │                   └--finish/abort----------┘
   └--prefix lookup--> attach cached blocks
```

1. **admit**：调度器检查 token 数、可用物理块和请求预算。若 prefix cache 索引返回完整块，block manager 为命中的块增加引用，并只为未命中的尾部预留块。
2. **prefill**：worker 对未命中 token 运行模型。每层生成 K/V，按 cache engine 规定的 stride 写入物理块；最后一个 block 记录 `valid_tokens`。
3. **decode**：每生成一个 token，先检查当前最后一块是否有空槽；若满则从 free queue 申请新块。attention backend 读取 block table，kernel 遍历历史块，并把新 token 的 K/V 写到可写位置。
4. **finish**：序列结束、取消或达到长度上限时，释放请求的逻辑引用。块若 refcount 变成零，才回到 free queue；若开启 prefix cache，完整且 immutable 的块可以进入 hash index，而不是马上归还。

一个实用的阅读方法是追踪四个变量：`block_table`（逻辑到物理）、`num_full_blocks` 或有效长度、`free_blocks`/free queue，以及请求状态。每读到一个函数，回答“它改变了谁的所有权？它是否可能在 kernel 尚未完成时释放？”。不要只记录调用层级；真正容易出错的是异常路径和异步事件的先后。

### 5.3 源码观察：为什么不直接拼一个大张量

**源码观察（v0.6.3）**：block manager 暴露的接口以 block 数和 sequence group 为中心，而不是让每个请求自行持有一个连续 `torch.Tensor`。这使调度器能在请求间复用物理块，避免因为一条长序列扩容而搬移整段历史。cache engine 负责把逻辑 slot 转成具体层、头和偏移的地址；attention backend 负责把 block table 传进设备 kernel。三者之间的接口边界很重要：调度器不应该猜 kernel 的 stride，kernel 也不应该直接修改 free queue。

**源码观察（较新 v0.10.1）**：V1 的 KV cache manager 和 block pool 将前缀查找、块分配、引用计数拆成更明确的对象；Hybrid KV manager 还要处理不同层组的 page size。这个变化说明“PagedAttention”不是一个单独 kernel 就能解决的功能，容量、调度、哈希索引和回收必须协同演进。版本升级时，应重新画调用链并更新证据 manifest；旧章节中写死的类名很容易失效。

### 5.4 设备 kernel 的最小不变量

即使暂时不编译 CUDA，也可以用伪 kernel 检查四件事：

- `block_table` 中的每个物理 id 在 pool 范围内，且该块 epoch 与请求 epoch 兼容。
- 最后一个 block 只读取 `valid_tokens`，不会把未初始化槽位当作 K/V。
- GQA 的 query head 到 KV head 映射与 checkpoint 的 `n_kv_heads` 一致。
- 释放事件发生在 kernel/异步 copy 完成之后，不能因为 Python 引用消失就立刻复用物理块。

生产实现通常用 CUDA event、stream-ordered free 或 worker 的完成队列保证最后一点。一个 CPU toy 很难模拟 GPU stream 的内存可见性，因此实验只能证明状态机在同步假设下正确，不能证明真实异步执行没有 use-after-free。把这条限制写进报告比假装“单元测试通过即无竞态”更可靠。
## 5.5 源码练习：沿一次 append 画状态变化

读 vLLM 固定 tag 时，可以把一次 `append_token` 练习写成表格。起始状态：请求 A 的 block table `[7, 2]`，`block 2` `refcount=1`，有效长度 15，`B=16`。追加第 16 个 token 前，最后一块还有一个槽位；追加后有效长度变成 16，块可以在完成校验后进入 prefix index。再次追加第 17 个 token 时，需要从 free queue 取新物理块 19，table 变成 `[7,2,19]`。

再改成 `block 2 refcount=2`，因为 B 也共享 A 的前缀。第 16 个 token 仍不能原地写 block 2：即使只有最后一个槽位未用，B 可能也要写自己的第 16 个 token。正确路径是 COW：申请 block 19，复制 block 2 的 15 个有效 token，把 A 或 B 的 table 指向副本（由哪个请求先写决定），旧 block 2 refcount 降为 1，副本有效长度从 15 变为 16。源码阅读时要找的是这个“写入权限检查 + 复制 + table 更新 + refcount 变更”的连续状态，而不是只找到一个叫 `copy` 的函数。

如果分支路径把 table 更新放在复制完成之前，另一个 stream 可能读到半复制数据；如果先减少旧 refcount 再完成 table 更新，驱逐线程可能误以为旧块无人引用。工程实现通常通过锁、actor 顺序或 stream event 把这几个动作串成一个原子逻辑事务。toy 可以用单线程顺序验证不变量，但要在报告中明确它没有覆盖真正的并发竞态。
## 5.6 源码观察补充：vLLM 如何暴露容量和统计

vLLM `v0.6.3` 的 `CacheEngine.get_cache_block_size` 计算逻辑与本章公式对应：block size 乘 KV heads、head size、K/V 两份和 attention layers，再乘 dtype size；pipeline parallel 会按 stage 划分 block 数。阅读这段函数时，先把每个乘数写在纸上，再对照模型配置，能快速发现把 query heads 或总层数重复乘进去的错误。

在 `v0.10.1` 的 `KVCacheManager.get_computed_blocks` 中，prefix cache 命中长度受 `request.num_tokens - 1` 限制，源码注释解释了最后一个 token 的 logits 和 block-size 对齐限制。`allocate_slots` 先删除滑动窗口之外的块，再检查 free blocks，touch 命中块，分配新块，最后缓存已完成 token。这个调用顺序是一个有用的阅读练习：先回收不可见块可以降低 eviction，先检查容量再提交 table 可以避免半分配状态，延迟缓存允许 P/D 传输中的块在完成前不进入共享索引。

这些是固定版本的源码观察，不是对所有 serving framework 的规范。不同版本可能把 coordinator、block pool 和 attention backend 拆成不同文件；不同框架可能使用 radix tree 或 token page，而不是同样的对象名。迁移时保持不变量和状态图，重新核对接口，不要机械搜索旧函数名。
## 5.7 机制补充：一张双引用账本逐步走到回收

下面不依赖框架名，用具体操作追踪所有权。池有四个块，每块四个 token，初始 free ids 为 `[0,1,2,3]`。创建 A，内容为 `[1,2,3,4,5]`，得到页表 `[0,1]`；块零有四个 token，块一只有一个。两个块各被 A 引用一次。把 A 的完整前缀 `[1,2,3,4]` 发布到缓存后，采用本 toy 的计数方式，块零多一个“缓存索引拥有的引用”，总计为二；块一仍为一。

B 以 `[1,2,3,4,9]` 到达。前四个 token 命中块零，增加一个请求引用，块零计数变三；B 的后缀 9 放在块二，B 的页表 `[0,2]`。此刻物理内容只有九个 token，但两条请求逻辑上合计十个 token，因为四个前缀 token 被共享。这个差额是共享节省，不能当成负的内部碎片。内部碎片必须按物理块中的实际有效槽位计算，逻辑 token 总和可以大于物理槽位。

A 结束时，块零减去一个请求引用，从三变二；块一从一变零并返回 free pool。B 结束时，块零从二变一，块二归零回收。此时没有活跃请求，但块零仍被缓存条目保留。这不是泄漏，前提是缓存拥有权在快照中可见。随后淘汰该前缀，释放缓存引用，块零归零回收。最终四个块全部空闲，整个生命周期闭合。

真实 vLLM 的计数口径与 toy 不应混称。某些实现的 `ref_cnt` 只统计活跃请求，零引用但仍带 hash 的块留在 free queue 中，只有再次分配时才清理 hash；本 toy 把缓存索引视为显式 owner，先移除条目再释放它持有的引用。两种设计都可正确，但审阅者必须知道 `refcount=0` 的含义。若你把生产实现的零引用块直接当成“内容已被删除”，会误读 hit/eviction 行为；若把 toy 的缓存引用误当成活跃请求，会误判内存泄漏。

因此，不变量应写成带口径的等式：`refcount = 请求表引用数 + 本实现计入的缓存拥有权`。还要说明“free”是可再分配、已清零，还是仅没有活跃请求；这三种状态不同。对敏感数据，释放可再分配并不等于安全擦除；若要求零化，需要单独的性能和验证路径，不应把它隐含在 free 一词中。
## 5.8 机制补充：COW 并不是任何共享都复制

考虑两个分支都共享一个恰好填满的块。下一次 append 会创建新块，不需要复制已满块，因为旧内容保持只读。只有写入地址落在已经共享的可见物理块中，才需要复制。因此，完整块前缀缓存通常能避免前缀部分的 COW，复制集中在最后的部分块或截断分支上。这也解释了为什么“只共享完整块”虽然少命中一些 token，却使实现和正确性证明简单很多。

另一个例子是截断 fork。父请求有 `[1,2,3,4]`，子请求只继承前两个 token，但其表仍指向同一个四槽块。子请求下一次写 8 时，逻辑偏移是二；不能把 8 附加在物理数组末尾形成 `[1,2,3,4,8]`，必须在 COW 后覆盖或截断隐藏尾部，得到对子请求可见的 `[1,2,8]`。若测试只覆盖“末尾部分块 append”，就可能漏掉这种隐藏尾部错误。

再考虑所有权退化：子请求 fork 后父请求立即释放，物理块只剩子请求一个 owner。此时可不复制，但仍必须按子请求的可见长度写入，不能把父请求的隐藏尾部当成有效内容。**推导**：唯一所有权解决的是写入隔离，不自动解决逻辑长度；refcount 和 token_count 必须共同决定写地址。

COW 的失败也应具有明确语义。池已满时，复制需要新块，若没有可淘汰块，就必须保持旧块和页表不变并返回容量错误。不能先减少旧引用、再尝试分配，因为分配失败后会丢失所有权。对整批 append，工程上可以提供全有或全无语义，也可以允许已成功的前缀提交，但必须文档化并让调用方知道实际写入多少 token。本 toy 的状态测试应覆盖失败后的可见内容和引用一致性，而不只检查异常类型。
## 5.9 机制补充：块哈希如何与请求哈希区别开

请求级 hash 是把整个前缀一次性编码后计算摘要；块级链式 hash 则让每个完整块都有可复用 key。二者在无碰撞、相同序列化和 namespace 下都能正确区分内容，但复杂度不同。本 toy 使用前者：`hash_prefix` 对完整 tuple 做一次 SHA-256，`longest_prefix` 线性枚举并比较 slice；生产链式块索引只需逐块推进并在第一次 miss 后停止。教学实现不能因此声称复现生产块索引的性能，链式索引列为后续实现对照。

序列化格式也参与 key 语义。简单地把整数转字符串后拼接会把 `[1,23]` 和 `[12,3]` 都变成“123”，造成确定性的编码碰撞；这与密码学哈希碰撞无关。应使用长度前缀、规范 JSON、CBOR 或固定宽度整数编码，并把格式版本纳入 namespace。对多模态输入，token id 之外还有图片或音频特征；如果这些影响 K/V，也必须成为语义 fingerprint 的一部分。

只把 LoRA id 字符串加入 key 还不够，若同一 id 后面的权重被热更新，旧 cache 依然会命中。可使用不可变的 adapter digest，或在更新时改变 epoch 并清空相关索引。相同问题也存在于模型别名、tokenizer 名称和 chat template 名称：名字是可变指针，摘要或明确版本才是内容身份。**设计判断**：缓存 key 的版本化应和模型发布 manifest 联动，不应靠运维人员记得手动点一次“清缓存”。
## 失效边界：分页解决了什么，又没有解决什么

### 6.1 尾部碎片不是全部内存浪费

固定块把每个请求的尾部浪费限制在最多 `B-1` 个 token 槽，但还会有三种其它成本：block table 元数据、对齐 padding、空闲块被长期 prefix cache 占用。若请求长度分布集中在 `B+1`，每条请求都会浪费接近一个 block；若长度服从重尾分布，大请求会消耗大量完整块，小请求的内部浪费反而成为主要问题。**推断**：不存在一个对所有 workload 都最优的 block size；需要用真实长度分布和命中率扫描，而不是只比较一个平均长度。

### 6.2 PagedAttention 不是免费加速

分页引入 block table 读取、边界判断和非连续 gather。对很短的序列，查表和 kernel 分支可能超过连续张量的收益；对极高命中率但很小的前缀，hash lookup 和 refcount 维护也可能增加 CPU 开销。论文中的吞吐提升通常以特定 GPU、batch 和请求分布为条件，不能直接当作你的 p99 SLO。本 toy 只把 contiguous baseline 作为解析容量下界，实测比较不同 block size 与 prefix_rate 的 paged 状态机；冷/热 GPU cache 需要后续硬件实验。

### 6.3 前缀命中需要语义完全兼容

以下任一项变化都可能使 K/V 不可复用：

- 模型权重或 tokenizer 版本不同，token id 序列甚至可能已变化；
- RoPE base、position offset、sliding-window 规则不同；
- LoRA/adapter、量化 scale、KV dtype/layout 不同；
- tensor/pipeline parallel 拓扑改变，某 rank 的 K/V 分片不再对应；
- 租户、用户授权或数据保留策略不允许共享；
- 请求使用了会影响前缀语义的特殊 system token 或 chat template。

**设计判断**：cache key 应包含这些影响语义的 fingerprint；宁可 miss 后重算，也不要在不匹配时“尽量解释”一块字节。命中率下降是性能问题，错误 K/V 是正确性和安全问题，处理优先级不同。

### 6.4 哈希碰撞和迟到写入

加密哈希碰撞几率很低，但缓存服务还会遇到非恶意的 key 错误：只用 token ids 忘记 model id，只用前缀 hash 忘记 LoRA，或在更新索引时迟到的写入覆盖了新版本。可用两阶段发布降低风险：先写物理块和校验信息，完成后以不可变 manifest 提交索引；读取时检查 checksum、shape、epoch 和 ACL。请求取消后，迟到的 DMA 仍可能完成，free 操作必须等待或用 generation guard 防止旧数据写入新分配的同一物理块。

### 6.5 共享带来的侧信道

即便 K/V 内容本身被正确隔离，命中与否的延迟、显存占用和 eviction 行为仍可能泄露某个公共前缀是否存在。多租户服务若允许共享 system prompt，应评估 timing side channel、block count side channel 和容量争用。可选缓解包括按租户分区、限制跨租户 prefix cache、对命中路径加抖动或只共享公开前缀。**设计判断**：不要为了一个更高 hit rate 默认打开跨租户共享；先把威胁模型写清楚。

### 6.6 与 sliding window、稀疏注意力的关系

PagedAttention 管理“仍然需要的”块，但并不决定模型要看多长的历史。若模型采用 sliding-window attention，超过窗口的旧块可以释放或转入冷 tier；若采用稀疏/检索注意力，block table 可能需要非连续选择，kernel 和 prefix 命中语义都不同。把“分页”误写成“无限上下文”是常见宣传错误：容量公式仍然约束可保留的工作集，质量也可能随着远距离信息丢失而变化。
## 6.7 错误语义：cache miss 可以慢，cache hit 不能错

缓存系统的错误分类要先问“能否安全重算”。

- **容量 miss**：没有足够空闲块。可以排队、拒绝或减少 batch；不会改变模型语义。
- **内容 miss**：token 前缀不同或 hash 不存在。重新 prefill 即可。
- **metadata mismatch**：模型/LoRA/RoPE/dtype 不兼容。必须拒绝复用并从边界重算，不能“尝试转换”。
- **checksum failure**：块损坏或迟到写入。丢弃该块、标记监控事件、重算；不要把错误 K/V 交给模型。
- **authorization denial**：租户不允许读取。即使 hash 命中，也必须返回 miss 或拒绝，不能暴露命中时间细节过多。
- **in-flight timeout**：发布前的写入没有完成。等待有限时间，超时清理临时块并重算。

把 miss 统一成“重算”并不意味着可以忽略原因。重算次数、丢弃字节、ACL 拒绝和 checksum 错误分别对应容量、性能、安全和可靠性指标。报告应至少给出每类计数和对 TTFT 的影响。
## 6.8 失效边界补充：位置编码和缓存可平移性

位置编码决定“同一 token 在不同绝对位置是否产生同一 K/V”。绝对位置 embedding 通常直接把位置加入 token 表示，前缀从位置零移动到位置十就会改变投影；RoPE 把位置旋转进 Q/K，也可能依赖起始 offset、base 和缩放策略；相对位置方法对平移的性质不同。因而 prefix cache key 中的 position metadata 不是可选装饰。

假设同一个系统提示在两个请求中出现，但一个请求前面多了 BOS，另一个使用不同 chat template。token ids 或位置起点变化后，简单字符串比较会误判“相同前缀”。更隐蔽的是两个请求 token ids 恰好相同，但模型服务在一个 batch 中使用了不同的 rope scaling 配置；K/V shape 相同、数值却不兼容。生产系统应把位置编码配置摘要放入模型 fingerprint，或者在运行时拒绝混合配置。

滑动窗口还会改变块生命周期。历史超过窗口后，某些层不再需要最早的块，但其它全注意力层仍可能需要它。统一按请求长度保留所有层会浪费内存，统一释放又会破坏全注意力层。vLLM 新版本引入 KV cache groups 和不同 spec，就是因为“每层同一 page size”并非永远成立。教学 toy 把所有层视为相同，只能说明最基本的 block table，不应被解释为 hybrid attention 的完整实现。
## 6.9 失效边界补充：一次错误复用如何逐层传播

错误 K/V 不一定立即产生 NaN。假设从另一个 adapter 取来形状完全相同的一块 K/V，attention kernel 可以正常完成，softmax 也可能数值稳定，HTTP 返回仍是成功。错误会通过加权和进入 hidden state，再经过残差、MLP 和后续层，最终改变 logits。它可能只改变一个不常见 token，在几十步后才表现为内容跑题。因而健康探针、有限数检查和吞吐基准无法单独检测语义错误。

一个最小对照是固定输入 token、模型、adapter 和位置，分别运行无缓存、冷缓存、热缓存三条路径，比较每个目标位置的 logits 或 hidden-state 摘要，给出 dtype 对应的数值容差。若只比较最终采样文本，随机性可能掩盖小的数值差异；若使用贪心文本，两个不同 logits 也可能产生同一 argmax。更强的检查是逐位置数值差异加任务级质量回归，二者测量对象不同。

但逐元素完全相等也不是通用要求。不同 kernel 的浮点归约顺序、混合精度和 fused operation 会引入可接受误差。因此容差要和参考实现、dtype、长度及评估目标一起声明。可以报告最大绝对误差、相对误差、logit 排名变化和任务输出变化，而不是一个笼统的“相同”。若引入 KV 量化，误差已不是纯布局变化，必须转入近似质量协议，不能继续沿用 exact paging 的结论。
## 可运行实验：CPU 模拟能证明哪些不变量

### 7.1 实验问题与实现范围

`labs/phase2/kv_block_pool.py` 是一个标准库 Python toy。它不实现 Transformer 数值前向，而是实现状态和容量：

- `BlockPool(block_size, capacity)`：维护固定物理块、free queue、refcount、有效 token 数、哈希和可淘汰标志。
- `RequestState`（承担 RequestTable 的教学角色）：保存逻辑 block table、token 数和 prefix hash；当前 toy 不保存 epoch/in-flight。
- `PrefixIndex`：用整个 token tuple 的 SHA-256 key 做线性最长前缀扫描；链式 block index 只作为生产对照，不由本 toy 实现。
- `create_request`、`append_token(s)`、`cache_prefix`、`fork_request`、`ensure_writable`（COW）、`release_request`、`evict`：分别对应正常和异常路径。
- `snapshot_metrics`：返回 active request tables 的 active blocks、free blocks、logical tokens、unique physical slots、active_used_slots、internal waste、hit tokens、cow copies 和 eviction count；release 后仍被 prefix cache pin 住的块要从 `stats()` 的 allocated/free/prefix_entries 查看。

toy 为每个物理块保存整数 token 作为“内容”，这样可以在不依赖 NumPy/PyTorch 的情况下检查共享和 COW。真正的 K/V 数值不会被计算；实验不能证明 CUDA kernel 的带宽、GPU 显存、TTFT、TPOT、量化误差或论文吞吐倍数。

### 7.2 基线、变量和控制变量

主实验是顺序工作集而不是并发 serving：每个请求创建、测量后立即 release，再进入下一个请求；固定 Python 3.12、Linux x86_64、单线程、`capacity=64` blocks、随机种子 7、请求长度来自 `[3, 7, 15, 16, 17, 31, 32, 33, 63]` 的可复现序列。`run_benchmark` 实际测量的是同一个 paged pool 状态机：`prefix_rate=0` 是不命中前缀的 no-share 对照，`prefix_rate=0.5/1.0` 是完整块 prefix cache + COW 的测量条件。连续内存 `contiguous_ideal` 只按容量公式计算，是解析下界，不是跑过的实现策略。

扫描 `block_size ∈ {4, 8, 16}` 和前缀命中率 `{0%, 50%, 100%}`。每个组合运行 30 个独立请求序列，先预热 5 次再记录 25 次；toy 的“延迟”是状态操作的纳秒/微秒级时间，只用于比较 Python 控制流，不代表 GPU serving。报告 p50/p95/p99（使用排序后线性插值的同一约定）、平均 hit tokens、内部浪费比例、峰值 active blocks、COW 次数和 eviction 恢复成功率。

运行命令：

```bash
python labs/phase2/kv_block_pool.py --seed 7 --requests 60 --block-size 8 --capacity 64 \
  --prefix-rate 0.5 --repeats 30 --report reports/kv-block-toy-seed7.json
python -m unittest labs/phase2/test_kv_block_pool.py -v
```

原始 JSON 必须保存参数、Python 版本、git commit、每次重复的计数和聚合方法。若运行环境没有 `git`，artifact 中将 commit 写成 `unknown`，但不能伪造版本。报告中的 p99 在 25 个样本上不稳定，应同时给出原始样本和样本数；不要把它写成服务尾延迟的置信保证。

### 7.3 预期观察与手算校验

先用 `block_size=4`、请求长度 `[3, 5, 8]` 手算：

- 理想 token 数 16；分页槽位 `ceil(3/4)+ceil(5/4)+ceil(8/4)=1+2+2=5` blocks，共 20 槽，内部浪费 4 槽（20%）。
- 如果第二个请求与第一个请求共享前缀 `[101,102,103,104]`，完整块命中会减少 1 个物理块；第二个请求的后缀仍需新块。
- 若两个请求都指向共享的最后一块，任一请求 append 前会 COW，物理块数增加 1，旧块 refcount 保持 1。

实验应验证这些计数，而不是只打印“运行成功”。在 `prefix_rate=0` case 中，hit tokens 应为 0（benchmark 仍会执行一个独立 COW probe，因此 COW 指标记录该 probe，而不是把它当作请求前缀命中）；在 100% 完整前缀命中时，hit tokens 应等于可整块匹配的前缀长度；block size 越大，平均尾部浪费通常上升，但 block table 条目数下降。若观察相反结果，先检查请求长度分布和 prefix key 是否把不同后缀错误地合并。

### 7.4 统计和不确定性

对每个 `block_size × prefix_rate` 组合报告（artifact 不伪造 `contiguous_ideal` 或额外 strategy 字段）：

```
metric                mean   p50   p95   p99   min   max   n
operation_us           ...    ...   ...   ...   ...   ...   25
internal_waste_ratio   ...    ...   ...   ...   ...   ...   25
hit_tokens             ...    ...   ...   ...   ...   ...   25
```

操作时间受 Python 调度、文件系统和共享 CPU 影响，重复间的方差可能比 block size 的效应大。可以报告 bootstrap 95% 区间或简单的 min/max，但要说清方法。`internal_waste_ratio` 是 active request tables 中 `(unique_physical_slots - active_used_slots)/unique_physical_slots`，共享物理块只计一次；它不包含 release 后 pinned prefix-cache blocks，也不是 CUDA allocator 的碎片。`operation_us` 不可外推为 TTFT。实验支持的最强结论是“在这个状态模型和参数下，COW 保持内容隔离、完整块命中减少物理块、块越大尾部浪费的方向符合公式”。
## 7.5 实验扩展：故障注入和不变量测试

除了主实验，建议按顺序设计四个低成本故障注入。当前回归直接覆盖重复 prefix 插入、部分块 COW、容量错误、eviction 和分页注意力等价；`double-release` 没有独立回归用例，hash 碰撞、epoch/in-flight 和故障序列 artifact 也尚未实现，下面的协议步骤不能报告为已经运行：

1. **双重释放（待补回归）**：同一请求调用两次 `release_request`。预期第二次是幂等 no-op 或显式错误，refcount 不得变负，free queue 不得重复出现同一 id；当前没有独立测试或 failure artifact。
2. **hash 碰撞模拟**：把 toy 的 hash 函数替换为只返回低 4 bit，插入不同 token 块。预期 index 在 checksum/token 比较失败时拒绝命中，不得静默共享。
3. **迟到完成（协议练习，未由本 toy 实现）**：在 release 后再提交一个旧 epoch 的完成事件。预期 generation guard 丢弃写入，新的 owner 内容保持不变；若要运行它，需要在后续 connector lab 中增加 epoch/in-flight 状态。
4. **容量枯竭（有单测、待 failure artifact）**：capacity 设置为 2，申请第三个请求。当前单元测试覆盖 `MemoryError` 与状态保持，但尚无独立 failure artifact；预期活跃请求的 block table 和 refcount 保持一致。

每个注入都要把“预期不变量”写成断言，例如 `sum(refcount) == number_of_active_owners`、`len(set(free_ids)) == len(free_ids)`、`published_hashes ⊆ complete_immutable_blocks`。失败时保存最小操作序列，形成一个可以回归的缩减案例。**研究方法**：先让实验稳定复现 bug，再修改实现；不要在随机负载中看到一次失败就直接改变多个参数。
## 7.6 实验结果解读：先看不变量，再看曲线

假设一次 toy run 输出：`block_size=4` 的 waste ratio 0.18、hit tokens 240、COW 31；`block_size=16` 的 waste ratio 0.36、hit tokens 208、COW 7。不能直接宣布小块更好或大块更好。小块命中粒度细、COW 复制少量数据，但 block table 和 hash 次数更多；大块元数据少、复制次数少，却浪费尾部槽位并可能重算更多前缀尾巴。

先检查三条不变量：

1. `active_used_slots <= physical_slots`，且 waste ratio 在 `[0,1)`；共享请求的 `logical_tokens` 可能重复，不能直接与物理槽位比较；
2. 每个命中块的 token 内容与请求前缀一致，COW 后父子请求内容隔离；
3. 释放所有请求并按报告的缓存策略淘汰后，allocated blocks 等于仍由缓存 owner 保留的块数。

若不变量失败，性能曲线没有解释价值。若不变量通过，再按目标 SLO 选择：TTFT 敏感且公共前缀长，关注 hit tokens 和 prefill 操作；显存敏感，关注 waste ratio 和 peak blocks；CPU 调度敏感，关注 block table 条目和 operation p95。把指标与目标绑定，才能避免“选平均值最好看的配置”。
## 7.7 可运行实验补充：怎样把 toy 的随机性与计时分开

状态计数应可确定复现，墙钟计时通常不能。固定 seed 会固定请求长度、前缀选择、分支和取消操作，但无法固定操作系统调度、CPU 频率、缓存热度或其它进程。因此 JSON artifact 应把“确定性结果”和“时间样本”分开：同一 seed 的 token 数、命中数、COW 数、最终 free blocks 应一致；每次 `operation_us` 可以不同。测试不应断言某个微秒值，否则 CI 会成为随机失败器。

比较两种策略时，可以对同一请求 trace 做配对测量：先生成 trace 保存下来，再分别重放，而不是让每个策略自己调用随机数生成器。否则某个策略多消耗一次随机数后，后续请求就不再相同，所谓性能差异可能来自输入差异。顺序也可能影响 CPU 热状态，正式实验可交替运行策略顺序或随机化顺序，并保留运行顺序字段。

预热的目的要具体。本 toy 没有模型权重加载和 GPU 图捕获，五次预热主要减少解释器路径和文件 I/O 的冷启动影响；不能把它类比成真实服务的完整预热。真实服务还应分别记录模型加载、kernel JIT、CUDA graph capture、allocator pool 和 prefix cache 的热度。把所有预热效果揉进“丢弃前五次”会丢失部署冷启动的重要成本。

p99 对样本数敏感。二十五个重复的 p99 基本靠近最大样本，可能只是一次 CPU 抢占。对教学模型，保留所有原始数据并报告 min/max 足够说明不稳定；对生产 SLO，需要更大请求样本、持续时间和负载条件。计算百分位前还要选择单位：按每次请求操作时间、每轮 trace 总时间还是每 token 时间排序，得到的数值含义不同。实验报告应给出字段定义，不能仅使用一个漂亮的 `p99` 列名。
## 失败诊所：命中率上升，为什么可用块反而下降

**观察**：一次压测中 prefix hit rate 从 42% 升到 78%，平均 prefill 操作减少；但 5 分钟后 free blocks 从 30 降到 2，新请求频繁触发 eviction 和 recompute，p95 反而恶化。

**假设 A：** 命中块的 refcount 没有在请求结束时递减，导致不可回收的“幽灵块”。

**假设 B：** hit rate 上升只是因为请求更短，命中 token 占比变大，但绝对的共享块数量没有变化；free blocks 下降来自长尾请求。

**假设 C：** eviction 正在驱逐 in-flight 块，随后迟到的 kernel 写回又把同一物理 id 标记为可用，产生隐藏数据损坏。

最小排查按成本从低到高：

1. 对每个块打印 `block_id、refcount、owners、immutable、valid_tokens、last_access、in_flight`，在请求完成事件后断言 owner 集合为空或与 refcount 相等。
2. 固定请求长度和前缀分布，只改变 `prefix_rate`；比较 hit tokens 的绝对值与 free block 曲线，区分 A 和 B。
3. 打开 toy 的故障注入：在 `release_request` 前插入一次 delayed completion，确认 `evict` 拒绝 in-flight 块；若拒绝后 free queue 仍增长，检查 generation guard。
4. 对每次 prefix index 插入保存 checksum，随机抽取命中块与原 token ids 比较；hash 命中但内容不等说明 key/epoch 错误，而不是容量问题。

**证据**：若释放后 refcount 仍为正且没有活跃 owner，支持 A；若 refcount 正确但长尾请求占满完整块，支持 B；若同一物理 id 在两个 epoch 出现不同内容而 checksum 不匹配，支持 C。不要看到 hit rate 高就直接修改 eviction 阈值。

**修复**：A 需要让 release 与取消/异常路径共用幂等函数，并在测试中重复释放不使 refcount 变负；B 需要按绝对块数和租户配额做 admission，而不是只看 hit ratio；C 需要把异步完成事件纳入可回收条件，必要时宁可重算也不复用未确认的块。修复后重新运行冷 cache、热 cache 和随机取消三组实验，并保留失败 artifact。**设计判断**：正确性故障优先于命中率，cache 丢失可以重算，错误 K/V 不可以悄悄返回。
## 8.1 失败诊所补充：命中了一整条提示，为什么仍有 prefill

假设请求长度恰好等于两个完整块，所有 K/V 都已在缓存中。新人可能期待“零 prefill，直接返回 token”。但 K/V 本身通常不包含最后位置的最终 logits；生成下一个 token 仍需要取得 logits 或保存额外结果。vLLM v0.10.1 的 `get_computed_blocks` 在源码中明确把最大可命中长度限制为请求 token 数减一，并说明由于块对齐，可能要重算最后一个完整块。这里的重算不是 prefix cache 失效，而是输出接口需要 logits。

**观察**：监控显示前缀查找成功，但 TTFT 中仍有一小段模型执行。**假设**：可能是需要最后位置 logits，也可能是 prompt logprobs 请求绕过缓存，或块未发布。**最小排查**：先检查 sampling 参数中是否请求 prompt logprobs，再记录最大命中长度、block size、已算 token 和新算 token；用长度 `B-1、B、B+1` 三组输入比较。若在长度恰好整块时出现更大的重算跳变，支持块对齐解释。

**修复或选择**：不要为了消灭这段重算直接跳过最后一步，那会使采样器缺少目标分布。可以研究缓存 logits、支持非整块已算长度或改变接口，但这些方案都需要额外内存、版本和正确性验证。**源码观察**只证明固定 tag 的路径，后续版本可能改变；报告应链接具体函数并说明没有测量 GPU 代价。
## 8.2 失败诊所补充：空闲块很多却仍然不能分配

**观察**：监控显示 pool 有 10 个 free block，但新请求申请 12 个块失败；团队认为 free counter 错了。**假设 A** 是请求需要的块数超过当前上限；**假设 B** 是一部分块在 CPU tier 或其它 rank，当前 GPU pool 不能直接使用；**假设 C** 是 allocator 的 free queue 与 refcount 统计脱节。

最小排查先打印请求的 `num_tokens_need_slot`、block size 和每个 KV cache group 的需求，再分别打印 GPU/CPU/rank 的 free 数。若请求属于一个需要每组各 6 块的 hybrid 配置，单一总数 10 并不代表每组都有 6；若多卡分片要求每个 rank 12 块，某个 rank 少一个就不能启动。最后运行 `assert_invariants` 检查 free set、allocated map 和 request table 的互斥性。

**修复**：容量检查按实际 block group 和 rank 维度进行，监控也按同一维度暴露；只有确认统计口径一致后，才去调大 pool。用一个总 free counter 掩盖分片瓶颈，会让 admission 在错误位置失败，也会诱导团队不必要地提高显存预留。
## 方案边界比较：连续 cache、PagedAttention 与前缀共享

| 方案 | 主要收益 | 代价与最坏情况 | 适用前提 | 本章如何验证 |
| --- | --- | --- | --- | --- |
| 每请求连续张量 | kernel 简单，短序列低开销 | 扩容搬移、外部碎片；无法自然共享前缀 | 并发小、长度接近、共享需求低 | ideal/contiguous 容量下界 |
| 分页但不共享 | 物理块可复用，尾部碎片受 `B` 控制 | block table/gather 开销；仍重复 prefill | 长度分布宽、需要弹性 admission | `run_benchmark(prefix_rate=0)` |
| 分页 + 完整块 prefix cache | 共享系统提示，减少重复计算和块数 | hash/refcount/COW；语义不兼容时必须 miss | key 可含模型/租户/位置 metadata，前缀重复 | `run_benchmark(prefix_rate=0.5/1.0)` |
| token 级共享或非前缀拼接 | 命中粒度细，可能节省更多 | 元数据和重算边界复杂；质量需重新验证 | 有 CacheBlend/检索式机制和质量预算 | 本章不实现，作为后续研究 |

比较必须使用同一请求序列、同一模型语义和同一 admission 上限。论文报告的速度提升不能直接填入表格的“收益”；本 toy 只测块状态，不测 attention kernel。若线上请求没有重复前缀，第三行的额外复杂度可能没有回报；若前缀高度重复但租户隔离严格，收益又被安全策略限制。
## 8.3 方案边界补充：为什么仿真不应制造服务延迟

一种常见的错误是给每个 hash 操作分配一微秒、每个块复制分配十微秒，再把总和命名为 TTFT。这个模型可以用于敏感性分析，但参数来自假设，因此输出必须叫“模拟成本单位”或“参数化时间”，不能叫测量。若参数来自真实硬件微基准，还需要说明是否包括并行重叠、排队、kernel launch、同步和数据路径，才能用于预测。

本章采用实际 CPU 控制流墙钟计时和确定性块计数，不构造虚假的 GPU TTFT。对系统决策，最有价值的是发现指标的方向和不变量，例如某个 block size 让尾部浪费从八槽变成三十二槽，或某个 partial-prefix 场景多产生一次 COW。等到有目标 GPU 后，再用 profiler 和服务负载验证这些状态变化是否转化成性能收益。如果计数改善但 GPU 延迟没有改善，应寻找新的瓶颈，而不是修改图表让它符合预期。
## 论文谱系、源码观察与证据地图

### 9.1 PagedAttention：从内存碎片到块表

**问题**：Kwon 等人在 SOSP 2023 的 PagedAttention 论文中观察到，当多个生成请求的长度动态增长时，连续 KV cache 预分配会带来内部和外部碎片，降低可服务 batch。**核心想法**：像虚拟内存一样，把逻辑序列分成固定 token 数的 blocks，物理 blocks 非连续；attention kernel 根据 block table gather。**论文结果**：论文在其 GPU、模型和请求 workload 上报告了 vLLM 相对当时系统的吞吐优势，并测量了 beam search、parallel sampling 的块共享节省。具体倍数和 37.6–55.2%/6.1–9.8% 的节省只适用于论文设置；原文没有证明所有硬件、版本和流量分布都得到相同数字。**今天保留/改变**：vLLM 仍保留块表和共享的基本思想，但 manager、kernel、hybrid page size 和 prefix hash 随版本演进，不能把论文中的类名当作当前 API。

### 9.2 vLLM prefix caching：共享的条件

**问题**：仅分页能减少碎片，却无法避免两个请求对同一 system prompt 重复计算。vLLM 的 automatic prefix caching 通过块哈希链和引用计数把完整前缀映射到物理块。**机制**：从左到右查找最长连续命中；共享块 immutable，写入或分支时 COW；free queue 和 refcount 决定何时回收。**证据边界**：官方文档说明了 key 链和命中流程，但具体 hash、对象名和 page size 在版本间变化；本章把文档和 tag 源码分别列为 evidence entries，不假装一个页面覆盖所有版本。

### 9.3 FlashAttention 的相邻启示

FlashAttention 论文解决的是注意力的 IO 访问顺序和分块，而不是 KV cache 的跨请求生命周期。它提醒我们：即使数学公式不变，HBM 读写、tile、在线 softmax 和 kernel 融合也会决定性能。把 FlashAttention 的 IO-aware kernel 与 PagedAttention 的物理块管理混为“一个优化”会误导读者：前者主要改变单次 attention 的数据移动，后者改变跨请求存储和地址映射；两者可以组合，也可以在特定 backend 中互相约束。

### 9.4 从源码到实验的证据链

本章每个重要结论应在 manifest 中有一条可独立检查的记录：容量公式是 `derivation`，PagedAttention 论文中的节省是 `paper_result`，vLLM tag 的 block manager 路径是 `source_observation`，toy 的 hit/waste/COW 是 `experiment_measurement`，跨版本迁移建议是 `design_judgment` 或 `inference`。若一条断言同时依赖源码和实验，就拆成两条，不用一个“参考文献列表”掩盖因果差异。
## 9.5 论文教学卡：如何避免把数字搬错

读 PagedAttention 论文时，先画出作者比较的系统、模型、GPU 和请求分布，再摘录数字。论文中的“吞吐”可能是 request/s、token/s 或 normalized throughput；“内存效率”可能按可容纳请求数、KV bytes 或碎片比例定义。把不同单位放在同一列会制造虚假的比较。

建议为每张教学卡写五行：

- **当时的痛点**：连续 KV cache 的动态增长导致什么碎片或排队现象？
- **提出的机制**：block table、非连续物理块、共享和 COW 分别解决哪一段因果链？
- **测量条件**：模型、GPU、batch/请求长度、对照系统、重复/统计是否报告？
- **没有证明的事**：是否没有覆盖多租户、跨节点、量化、不同 tokenizer 或新版本 kernel？
- **今天如何复现**：固定论文版本和 vLLM tag，先跑 toy 的状态不变量，再在 GPU 上做同工作负载对照。

FlashAttention 的卡片应明确“减少 HBM IO 的 kernel 技巧”与“块生命周期管理”是不同层次；prefix caching 的卡片应明确“前缀内容相同”不等于“允许共享”。教学卡不是参考文献堆砌，而是帮助读者把历史问题、机制和当前边界对齐。
## 9.6 论文与实现的迁移规则

从论文迁移到生产实现，至少问四个问题。第一，论文的 block 是否按每层、每请求或整组定义，和你的 cache layout 是否相同。第二，论文的共享是 beam/parallel sampling 的瞬时共享，还是跨请求持久前缀，生命周期和安全边界是否相同。第三，论文的评测是否包含 tokenizer、排队、kernel warm-up、日志和取消请求，若没有就不能直接承诺服务 p99。第四，论文比较的 baseline 是否已经具备同样的 continuous batching、量化和 attention kernel，否则 speedup 可能来自功能差异而非单一机制。

PagedAttention 论文把“near-zero waste”作为物理块管理目标；工程实现仍要选择 block size、eviction、prefix policy、adapter key、异步生命周期和多租户边界。FlashAttention 把“更少 HBM access”作为 IO 目标；工程实现还要验证 paged gather 是否改变 coalescing 和归约顺序。GQA/MQA 把“更少 KV heads”作为模型结构选择；工程实现要保留质量和 checkpoint 兼容。三条谱系可以组合，但不能把一个论文的指标拿来证明另一个机制的效果。
## 10.5 从本地 cache 到后续 connector：哪里开始需要新协议

本章把物理块放在同一进程或同一 GPU 内。下一步若把块搬到 CPU pinned memory、NVMe 或远程节点，block table 之外还需要 wire metadata：`request_id、model_fingerprint、tenant_id、block_hash、token_range、layer/rank、shape/stride、dtype/layout、source、checksum、epoch`。传输必须先准备目标 buffer，再发送数据，最后以完成事件提交 manifest；接收端不能在 checksum 通过前把块放进 prefix index。

这个扩展改变了错误边界：本地 cache miss 只增加计算，远端读可能遇到网络超时、部分块、旧 epoch 和权限失败。正确性仍然依赖本章的不变量，但需要 lease、幂等 put、重试和重算策略。**设计判断**：把远端 connector 写进同一个“cache 命中率”指标会掩盖传输时间和一致性风险；应分别报告 local hit、remote hit、bytes transferred、recompute tokens 和 failure class。后续 KV transfer 章节会实现一个 mock producer/consumer，不在这里提前假装已经证明 RDMA 性能。
## 10.6 研究问题补充：用反例检验“几乎免费”

1. **几乎零碎片**：构造长度总在 `B+1` 的 trace，测内部浪费是否接近一个完整块；如果不是，解释共享、部分块和请求释放如何改变统计。
2. **灵活共享**：让两个请求只共享中间四个 token、前后缀不同。PagedAttention 的前缀 hash 是否能直接复用？若不能，设计一个 CacheBlend 风格的选择性重算方案，并标出质量验证点。
3. **免费 COW**：让一百个分支共享同一块后同时 append。计算复制块数、复制字节和池容量；说明共享读取便宜不代表写入无成本。
4. **命中即加速**：固定命中 token 数，逐步增加 decode 输出长度，测 TTFT、TPOT、总 wall-clock 的变化方向；解释为什么总 wall-clock 的收益可能趋于平坦。

研究问题的验收不是某个预设数字，而是变量、对照、输出指标和限制都清楚。若结果与直觉相反，优先检查单位、请求分布和实现口径，而不是删掉反例。
## 研究问题与理解检查

### 理解检查 1：为什么 128 KiB/token 不是“每层 128 KiB”

给出 `L=32、h_kv=8、d_h=128、bf16`。先算每层每 token：`2×8×128×2=4,096 bytes`，再乘 32 层得到 131,072 bytes。若答案直接写 4,096 bytes，说明把层数漏掉；若写成 query heads=32 的四倍，说明没有区分 GQA 的 KV 头。进一步问：把 dtype 从 bf16 改成 FP8（1 byte）时，容量在理想模型中减半，但量化 scale、kernel 和质量并未由这个公式证明。

### 理解检查 2：共享块何时必须 COW

A 和 B 共享物理块 7，`refcount=2`。B 要在该块的有效尾部追加一个 token。直接写会让 A 看到 B 的 token，破坏内容隔离；正确步骤是分配新块、复制旧块的有效内容、将 B 的 table 指向副本、旧块 refcount 减 1、新块 refcount 设为 1，再写入新 token。若块已经 `refcount=1` 且 owner 唯一，可以原地 append，但仍需检查 kernel/in-flight 状态。

### 研究问题 1：如何选择 block size

在真实 trace 上扫描 4、8、16、32、64 token，记录内部浪费、block table 条目数、prefix hit tokens、COW 字节和 attention kernel p95。你预期哪个 workload 会偏好小块，哪个偏好大块？答案必须同时提到长度分布、前缀命中粒度和 kernel/metadata 开销，不能只说“块越小越省内存”。

### 研究问题 2：命中率和有效吞吐为何可能反向

构造两个请求集合：集合 A 每条都有 2K token 的公共前缀，集合 B 每条只有 16 token、但 100% 命中。集合 B 的 hit ratio（命中 token / 总 token）可能更高，但绝对节省的 prefill FLOPs 和物理块不一定更多。请分别报告 hit requests、hit tokens、saved bytes、recompute tokens 和 p95 admission latency，说明为什么只看一个 ratio 会误导容量决策。
## 11.5 研究问题：把公式变成可反驳的假设

1. **容量假设**：在你的 trace 中，`bytes_paged - bytes_total` 是否接近 `B/2` token 的均匀余数近似？若偏离，画出 `T mod B` 的直方图，说明长度模板如何改变浪费。
2. **共享假设**：prefix hit tokens 增加一倍时，prefill 操作是否近似减半？若没有，拆出 hash、COW、调度和 decode 读历史的成本。
3. **安全假设**：跨租户共享会不会让命中计时能区分公开前缀？在 toy 中加入固定抖动和租户分区，比较攻击者的分类准确率；如果没有真实威胁模型，至少写出攻击输入、观测和停止条件。
4. **版本假设**：v0.6.3 与 v0.10.1 的 block manager 名称不同，核心不变量是否仍能用 block table/refcount/immutable 描述？找出一个不再成立的细节，并更新章节边界。

这些问题的答案不应从章节直接复制，而应来自受控实验、源码 diff 或明确的未知。研究级写作的进步标准是能指出自己不知道什么，以及下一次测量如何减少未知。
## 练习：从复算到设计

1. **复算题（容量与单位）**：模型有 40 层、16 个 KV 头、每头 128 维、FP16，block size 32。计算每 token 字节数、8K token 的理想字节、完整块数量和最坏尾部浪费。验收标准：每个中间值写单位，明确没有计入 scale、对齐和元数据。
2. **故障诊断题（引用计数）**：压测结束后 free blocks 低于 5%，但活跃请求为空。给出至少三个可区分假设，设计最小日志字段和一条断言来区分“泄漏”“prefix index pin 住块”“in-flight 未完成”。验收标准：排查顺序先读状态再改参数，且不能通过强制清空池子掩盖错误。
3. **设计题（跨租户前缀）**：产品要求让所有租户共享相同的系统提示以提高命中率。写一份一页 ADR，包含 cache key 字段、ACL 检查、侧信道风险、eviction 配额、回滚条件和至少一个被拒绝的替代方案。验收标准：明确 hash 不是授权，命中失败只能影响性能而不能返回错误 K/V。
4. **实验题（冷热缓存）**：在 toy 中加入 `cold` 与 `hot` 两种访问序列，固定 token 总量，比较 prefix rate 0/0.5/1 和 block size 4/16。报告 p50/p95/p99 操作时间、命中 token、内部浪费、COW 和 eviction；解释为什么操作时间结果不能直接写成 GPU TTFT。
5. **源码题（版本迁移）**：分别在 v0.6.3 和 v0.10.1 tag 找到“申请新 block”和“释放共享 block”的入口，画出函数调用链并标注名称变化。验收标准：链接固定 tag，指出哪些结论是源码观察，哪些是你对新旧版本的推断。
## 12.5 读者自检清单

完成章节后，读者可以逐项打勾：

- 我能用自己的数字算出 bytes/token，并指出 GQA 使用 KV heads 而不是 query heads。
- 我能画出三层结构：逻辑 token、block table、物理 pool，并标记最后一块的有效长度。
- 我能区分生产链式 prefix hash 与 toy 的 whole-prefix SHA-256 + linear longest scan，并解释为什么生产实现会在第一个 miss 停止、metadata 为什么包含模型和租户信息。
- 我能在 refcount=2 的块上说明 append 的 COW 顺序，并指出异步完成事件的风险。
- 我能沿固定 tag 找到 block manager、cache engine 和 attention backend 的接口，而不是只贴一个 main 链接。
- 我能运行 toy 已覆盖的稳定 hash、COW、eviction、分页注意力和容量错误测试；double-release 目前只能按 §7.5 的协议手动设计，hash 碰撞/迟到完成仍未实现，且这些测试没有证明 GPU 竞态。
- 我能先用公式计算 contiguous analytical lower bound，再运行两种 paged 条件（`prefix_rate=0` 与 `prefix_rate=0.5/1.0`）比较 hit tokens、active physical slots、waste 和操作分位数；真实 TTFT/TPOT 仍是后续 GPU 实验。
- 我能在论文卡中写出硬件、模型、统计口径和限制，避免把一次论文 speedup 写成生产保证。

若有一项做不到，先回到对应的小例子或实验，不要通过背诵名词跳过。KV cache 的难点不是记住一个公式，而是把容量、地址、所有权、语义兼容和测量边界同时放进同一张图。
## 12.6 练习答案提示

容量题的步骤是先算 `2×16×128×2=8,192 bytes/层/token`，再乘 40 层得到 327,680 bytes/token；8K token 理想约 2.5 GiB，完整块数为 `ceil(8192/32)=256`，此例恰好无尾部浪费。若长度改为 8,193，块数变 257，新增整块槽位带来接近一个 block 的浪费。答案必须注明没有算 metadata、scale、对齐和其它运行时预留。

引用计数诊断题至少可提出：释放路径漏掉异常取消、prefix index 持有的缓存 owner 没被统计、in-flight 块尚未完成。日志要有 request id 摘要、block id、refcount 前后、owner 类型、epoch、hash 是否发布和 free queue 事件。双重释放应幂等或报错，不能让 refcount 负数；强制 reset 只能作为受控回滚，不能代替查明根因。

跨租户 ADR 应拒绝“只用 token hash 全球共享”的替代方案，说明 hash 不表达授权、命中计时可形成侧信道、模型更新会造成语义混淆。推荐把公开 system prompt 与私有 user 前缀分区，key 含模型/模板/位置/adapter digest，数据面和 metadata 面都检查 tenant ACL，异常时回退重算。验收标准是任何拒绝都不返回其它租户的 K/V，命中率降低只影响性能。

源码题的答案不要求旧新函数名相同，而要求调用链和变化说明。例如旧版本可能在 `block_manager_v1.py` 分配 block，新版本由 `KVCacheManager.allocate_slots → coordinator → block_pool` 完成；两者都可映射到“检查容量、附着命中块、分配新块、发布完整块、释放请求”这五个状态。若只贴 main 链接或只复制类名，不能算完成。
## 术语表

- **KV cache**：每层历史 token 的 key/value 投影集合，decode 读取它来避免重算旧 token；它不是模型权重，也不是任意 hidden state 缓存。
- **K/V head（KV 头）**：保存 key 和 value 的注意力头数量。GQA/MQA 中通常少于 query heads，容量公式必须使用它。
- **block / page**：固定 token 容量的 KV 存储单元。page 是内存虚拟化的类比，block 是常见实现叫法。
- **block table**：请求级逻辑 block 到物理 block 的映射，以及有效长度等元数据。
- **PagedAttention**：按 block table gather 非连续 KV 块的注意力实现和内存管理思路，最初由 vLLM 论文系统化。
- **prefix cache**：按 token 前缀和语义 metadata 索引已完成的完整块，命中后通过 refcount 共享。
- **refcount**：物理块被多少活跃 owner 引用的计数；不是租户授权，也不能单独表示 GPU kernel 是否完成。
- **copy-on-write（COW）**：共享块在写入前复制，保护只读前缀；复制粒度和异步完成条件会影响成本。
- **内部碎片**：已分配块中尚未使用的 token 槽位；本章公式不包含外部碎片和 allocator metadata。
- **prefill / decode**：prefill 一次处理输入上下文并写入大量 K/V；decode 每步追加 token、读取历史 K/V。
- **page size**：每个逻辑页覆盖的 token 数与每层 KV 字节数的组合；混合层模型可能有不同 page size。
- **immutable block**：已完成校验、可共享且不得原地修改的物理块；写入要走 COW 或唯一 owner 路径。
- **generation/epoch**：分配和异步完成事件的代数标记，用于阻止迟到写入污染新 owner。
## 13.5 研究实践：一份可以交给同事的审阅记录

完成一个 KV 改动后，审阅记录可以围绕四个问题组织。第一，**数据有没有变**：固定前缀时，冷/热 cache 的数值输出在容差内是否一致；metadata 不匹配时是否拒绝。第二，**所有权有没有闭合**：所有请求结束后，只剩文档化的缓存拥有权，重复释放和容量失败不会损坏池。第三，**资源有没有真正节省**：同时报告理想 token 字节、物理分配字节、内部浪费、峰值块数和重算。第四，**结果能否迁移**：哪些来自 toy、哪些来自 GPU、哪些来自论文，版本与 workload 是否一致。

审阅时先看最小反例，再看大规模曲线。一个十行 trace 能稳定复现错误引用，价值高于百万请求的平均吞吐。一个 metadata mismatch 测试能阻止跨模型错误复用，价值高于命中率提高几个百分点。性能优化不必放弃严谨性；它需要把正确性门放在基准前，避免把错误共享带来的“节省”算作优化。

如果要形成研究报告，至少保留基线选择理由、被拒绝的替代方案、失败样本和未验证假设。例如没有实现远端 KV 传输，就明确它仍是下一章任务；没有 GPU，就明确无法说明 kernel gather 是否比连续读取更快。这样的边界不会降低章节价值，反而让后来的工程师知道哪里可以直接复用、哪里需要重新测量。
## 来源地图与复现清单

### 一手论文与官方实现

1. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*（SOSP 2023，arXiv v1/v2 页面）：https://arxiv.org/abs/2309.06180。阅读摘要与第 3–5 节的碎片、block table、共享和评测；论文数字只在其 GPU、模型和 workload 条件内成立。
2. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（NeurIPS 2022，arXiv）：https://arxiv.org/abs/2205.14135。用于区分 attention kernel 的 IO 优化与跨请求 KV 生命周期；不是 PagedAttention 的替代证据。
3. vLLM `v0.6.3` 源码 tag：https://github.com/vllm-project/vllm/tree/v0.6.3。阅读 `vllm/core/block_manager_v1.py`、`vllm/core/block_manager_v2.py`、`vllm/worker/cache_engine.py` 和 attention backend 的 block table 接口；tag 是证据版本。
4. vLLM automatic prefix caching details（文档版本 v0.6.3）：https://docs.vllm.ai/en/v0.6.3/automatic_prefix_caching/details.html。核对 hash 链、最长连续命中、refcount/COW 的语义；文档不覆盖所有后续 API。
5. vLLM `v0.10.1` KV cache manager 源码：https://github.com/vllm-project/vllm/tree/v0.10.1/vllm/v1/core。用于版本对比，不把新 manager 名称倒灌到旧 tag。

### 复现实验清单

1. 从干净 checkout 进入仓库根目录，确认 `git rev-parse HEAD` 和 `python --version`；本章 toy 只依赖 Python 3.10+ 标准库。
2. 安装或激活环境后运行 `python -m unittest labs/phase2/test_kv_block_pool.py -v`；预期所有状态、不变量、COW、哈希、eviction 测试通过。
3. 运行 `python labs/phase2/kv_block_pool.py --seed 7 --requests 60 --block-size 8 --capacity 64 --prefix-rate 0.5 --repeats 30 --report reports/kv-block-toy-seed7.json`。检查 artifact 包含参数、原始重复、p50/p95/p99、命中 token、active physical slots、内部浪费、COW、state counters 和 eviction recovery；failure injections 仅作为未来/手动协议练习，不宣称已写入当前 artifact。
4. 改变 `--block-size 4/16` 和 `--prefix-rate 0/1`，保持 seed、请求长度和 capacity 其它不变；比较方向而非绝对微秒。
5. 若要运行真实 vLLM kernel，另建 GPU 环境，固定 CUDA、驱动、vLLM tag、模型权重和 tokenizer，先完成小 batch correctness 对照，再测冷/热 cache 的 TTFT、TPOT、显存和 p99。没有 GPU 时不要填“GPU 结果”。
6. 清理 `reports/kv-block-toy-*.json` 之外的临时文件；保留失败命令、环境摘要和 git diff 供审阅。

本章的最小可交付证据是：公式的数字代入、固定版本源码入口、CPU toy 的原始 JSON、测试输出和一份说明“它没有证明 GPU 带宽、真实模型质量、跨租户安全或跨节点一致性”。如果这些条件缺失，正确的结论是“证据不足”，而不是把 PagedAttention 写成万能的长上下文解决方案。
## 13.6 最终复现清单的证据格式

提交 artifact 时，建议目录如下：

```
reports/
  kv-block-toy-seed7.json
  kv-block-toy-matrix.json
  # kv-block-failures.json（未来故障协议 artifact，当前未生成）
labs/phase2/
  kv_block_pool.py
  test_kv_block_pool.py
docs/phase2/
  chapters/26-kv-cache-and-paged-attention.md
  evidence/26-kv-cache.json
```

`kv-block-toy-matrix.json` 至少包括环境摘要、每个组合的控制变量、原始重复列表、聚合公式、p50/p95/p99、命中 token、active physical slots、浪费、COW、state counters 和 eviction recovery。四种故障注入的最小操作序列属于未来/手动协议练习，当前没有 `kv-block-failures.json`；artifact 路径要与 manifest 的 `experiment_id` 对应，不能只在文末说“已经运行”。

验证顺序也要固定：先运行单元测试和不变量检查，再生成实验 JSON，随后执行 phase2 template/evidence validator、内容 audit、Python compile，最后在网站构建中确认 Markdown frontmatter 可解析。若某一步失败，报告确切命令和错误，不要把其它步骤通过写成全部通过。没有 GPU 时，复现清单应明确跳过真实 kernel；只有在目标环境完成 correctness 对照和负载测量后，才添加 GPU 结果。
## 14.6 本章结语：把 KV 当作有契约的工作集

KV cache 的核心不是“多存一些张量”，而是为一个会增长、会共享、会分叉、会淘汰的工作集定义契约。容量公式告诉你一张卡最多能留下多少历史；block table 把逻辑顺序与物理地址分离；refcount 和 COW 让共享不破坏写入隔离；prefix hash 把重复前缀变成可定位对象；eviction 和 admission 决定哪些计算值得保留；metadata、epoch 和 ACL 让错误复用不会被误写成性能。

PagedAttention 的贡献可以用一句可检验的话总结：在逻辑序列仍按 token 顺序解释的前提下，允许 KV blocks 在物理内存中非连续放置，并让 attention kernel 根据页表读取它们，从而降低动态长度带来的碎片，支持受控共享。它没有消除历史读取、位置编码、容量上限、异步生命周期、跨租户授权或质量评估。每一个“没有”都是后续实验和系统设计的入口。

如果读者能手算 128 KiB/token 的来源，能在一张纸上画出 `[logical block → physical block]`，能解释 refcount=2 时为什么必须 COW，能用固定 tag 找到源码中的 block allocation 和 prefix lookup，并能在 CPU toy 中复现已覆盖的稳定 hash、COW、eviction 和容量错误，能设计/手动检查 double-release 协议，同时知道 double-release、hash 碰撞与迟到完成尚未在本版 toy 中独立回归，那么下一章的 KV 压缩、分层存储和远端传输就有了可靠地基。若还只能说“vLLM 用了 PagedAttention 所以更快”，请回到两个请求的开场问题，重新写出你要测的单位和边界。
