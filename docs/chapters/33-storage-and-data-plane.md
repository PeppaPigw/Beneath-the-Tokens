---
id: ch33-storage-and-data-plane
title: AI Infra 存储与数据平面：对象存储、并行文件系统、检查点与数据加载
slug: /chapters/33-storage-and-data-plane
description: 从对象存储、POSIX/并行文件系统和 NVMe 到训练与推理的数据加载、检查点、一致性和故障边界
sidebar_position: 33
level: advanced
prerequisites:
  - ch02-linux-process-files-observability
  - ch03-networking
  - ch04-performance-math
  - ch09-distributed-training
  - ch10-training-ops
  - ch11-data-systems
  - ch13-artifact-management
  - ch15-inference-execution
  - ch16-model-serving-system
  - ch20-observability-debugging-incident-response
  - ch21-ai-reliability-engineering
learning_objectives:
  - 能区分对象存储、POSIX/并行文件系统、NVMe/本地盘和内存缓存的语义、性能与故障域
  - 能从请求、元数据、数据块、网络和介质推导训练/推理数据面的延迟、吞吐、放大和尾部
  - 能设计带版本、校验和、原子提交、租约和清理策略的可恢复 checkpoint 协议
  - 能把数据集分片、worker/rank 采样、预取、缓存和 backpressure 连接为可观测的数据加载流水线
  - 能解释一致性、可见性、重试、幂等和故障注入对训练数值与推理 SLO 的影响
  - 能运行 CPU-only 实验，审计 shard 分配、后端延迟代理和不完整 checkpoint 的恢复边界
estimated_hours: 44
hardware: CPU-only lab required; NVMe/object-store/parallel-filesystem measurements optional and cost-bounded
risk_level: L3
last_verified: 2026-10-07
---

# 第33章　AI Infra 存储与数据平面：对象存储、并行文件系统、检查点与数据加载

> 模型参数、训练样本、tokenized shard、optimizer state、checkpoint、权重文件、prompt 和 KV cache 都是“数据”。它们经过的路径不同，可靠性和性能也不同：一个 S3 object 需要经过 DNS、TLS、HTTP、网关、元数据索引和存储节点；一个本地 NVMe 读请求可能只需提交到 PCIe 控制器和 flash queue；一个并行文件系统读请求既包含元数据 RPC，也包含数据条带化和客户端缓存。若只说“磁盘慢”或“对象存储不一致”，就没有说明哪一种语义、哪一段路径和哪一种证据。

本章的目标是建立一张可以审计的数据平面地图。我们把路径写成 `producer -> namespace -> metadata -> data blocks -> network -> cache -> consumer`，再把训练和推理的控制流接上：数据集如何被 worker 分片、如何预取和限流，checkpoint 如何由临时对象变成可见版本，故障时哪些字节可以重试、哪些状态必须回滚。章节中的数值实验是 CPU-only toy，验证协议和守恒关系，而不是替代目标硬件的 benchmark。

## 33.1 问题、边界与学习路线

### 33.1.1 为什么存储问题会伪装成模型问题

一个训练 step 的时间通常写作：

\[
T_{step}=T_{compute}+T_{collective}+T_{input}+T_{checkpoint}+T_{wait}.
\]

`T_input` 不只是 read 系统调用；它包含对象列表、权限校验、连接建立、解压、反序列化、随机数和 Python worker 排队。`T_checkpoint` 也不只是写 bytes；它还包含 shard flush、校验和、远端可见性、manifest commit、目录重命名和旧版本清理。当数据面拥塞，GPU 利用率下降、step p99 上升，工程师很容易把锅甩给 kernel 或 NCCL。正确的做法是把每个时间段都关联到 `job_id、step、rank、shard、backend、request_id`。

一个健康但错误的指标是“平均读取带宽 2 GB/s”。如果 99% 的 shard 命中缓存，剩余 1% 在对象存储冷启动时耗时 30 秒，平均值仍然很好，训练却会在少数 step 上出现长尾。另一个反例是“目录里出现了 checkpoint-100 文件夹”，但其 manifest 尚未原子提交；恢复器看见目录就加载，会读到不同 step 的混合 shard。

### 33.1.2 本章不负责什么

第11章讲数据系统的表、分区和查询；这里不讨论 SQL 优化器。第13章讲 artifact 管理和供应链；这里关注 checkpoint 及训练数据在运行时的读写协议。第15、16章讲推理执行和 serving；这里聚焦权重、prompt、KV 和检索上下文如何进入这些执行引擎。对象存储、Lustre、BeeGFS、CephFS 和 NVMe 的具体部署参数以目标版本官方文档和现场测量为准，本章不把 toy 中的 MB/s 解释成硬件承诺。

### 33.1.3 先建立四个问题

读任何一个数据面实现时先问四句：

1. **谁拥有名字？** bucket/key、POSIX inode、并行文件系统的目录项还是本地文件名？
2. **谁拥有字节？** object server、OST/OSD、NVMe namespace、page cache 还是进程内 cache？
3. **什么时候可见？** 写返回、fsync 完成、manifest commit、跨区域复制还是列表索引刷新？
4. **坏了如何恢复？** 重试同一个 request、重新计算 shard、回滚到旧 manifest，还是放弃整个 job？

这四问能把“存储”从模糊名词变成协议边界。

## 33.2 一句话心智模型：字节有生命周期，名字有版本

本章的心智模型是：**数据平面是一个带版本的字节生命周期图；每个字节都有生产者、命名空间、缓存层、校验和、可见性时刻、所有者和恢复路径。**

用一个样本 shard 表示：

```text
raw records -> tokenizer -> shard-0042.tmp
           -> checksum/size -> object or POSIX write
           -> durable acknowledgement -> manifest-v17
           -> worker lease -> decode/prefetch -> tensor batch
           -> sample cursor / metric / retry evidence
```

`manifest-v17` 是控制点，不是装饰文件。没有它，目录里有多少 `.tmp` 都不等于一个可用数据集。对 checkpoint 同理：参数、optimizer、RNG、数据 cursor 和拓扑摘要必须由同一版本的 manifest 绑定。推理权重也需要版本指针，例如 `model/current -> model/sha256:abcd...`，服务只接受完整并校验过的 immutable revision。

定义三个变量：`S` 是字节数，`B` 是有效 payload 带宽（byte/s），`L` 是每个请求启动和元数据延迟（s），`N` 是对象或分片数。顺序读取的粗略代理是：

\[
T \approx N\cdot L + \frac{S}{B}.
\]

当 `N` 很大时，减少小文件和批量列表比提高介质带宽更有效；当 `S` 很大时，压缩、并行连接和条带化更重要。训练数据通常同时有大量小 metadata 请求和中等大小 shard，checkpoint 则是大文件/大对象的 burst。不要用单一“磁盘吞吐”覆盖两种形状。

## 33.3 对象存储：HTTP 名字空间与不可变对象

### 33.3.1 key、metadata、payload 三层

对象存储暴露 `bucket/key -> bytes + metadata`，而不是可随意修改的 POSIX 文件句柄。请求通常经历认证签名、路由、metadata lookup、数据节点读写、checksum 和响应。对象的 key 可以包含斜杠，但这不代表服务器真的有目录树；“列出目录”是带前缀的索引扫描，结果可能分页、限速或受到列表可见性规则影响。

把对象拆成三类证据：

- **标识证据：** key、版本 ID、ETag、内容 hash、生成工具和 schema version；
- **大小证据：** `Content-Length`、压缩前后 bytes、分片数和 multipart part map；
- **语义证据：** 数据集/模型版本、许可、tokenizer digest、dtype、shape、创建 step 和是否已 commit。

ETag 在某些 multipart 或加密设置下不等于 MD5；应用需要显式保存 SHA-256 或 CRC 校验和。上传成功只说明服务接受了请求，不说明下游列举一定马上看到 key，也不说明跨区域副本已经完成。

### 33.3.2 一致性：写后读、列表和版本指针

不同对象服务对新对象写后读、覆盖、删除、列表和跨区域复制有不同承诺。即使服务声称强一致，应用仍要处理网络超时后的“未知提交”：客户端可能在服务端已成功写入后丢失响应，重试会产生同 key 覆盖、重复版本或重复事件。稳妥协议是使用 immutable key（含内容 hash 或 UUID）加条件写，再以一个小的 manifest/pointer 作为提交点。

一个版本发布流程可以是：

1. 生成 `releases/r17/parts/part-0000` 等不可变对象；
2. 逐个读取并校验 size/hash，记录 `parts[]`；
3. 写入 `releases/r17/manifest.json.tmp.<uuid>`；
4. 通过条件写或版本化 copy 生成 `releases/r17/manifest.json`；
5. 最后更新 `models/current` 指针，并在读取端验证 manifest hash；
6. 保留至少一个旧版本，等待观察窗口后再清理临时对象。

“先更新 current，再补上传 shard”会制造半版本；“按字典序列出 key，然后加载所有结果”会把临时文件和旧版本混入训练。

### 33.3.3 Multipart、并行连接与放大

大对象通常分成 multipart parts 并行上传。设 part 大小为 `P`，对象大小为 `S`，则 part 数约为 `ceil(S/P)`。`P` 太小，连接、签名和 metadata 放大；`P` 太大，失败重试成本和并行度不足。上传端应限制并发，避免 checkpoint burst 把训练读流量挤出。下载端可用 range read，但要验证每个 range 与对象版本绑定，否则覆盖写可能让不同 range 来自不同版本。

压缩并不总是加速。若 CPU 解压吞吐 `C` 小于网络节省后的有效速度，端到端时间反而增加。定义压缩比 `r=S_compressed/S_raw`，压缩时间 `T_c`，则：

\[
T_{total}=T_c+\frac{rS}{B_{net}}+T_{decode}.
\]

测量应同时记录 CPU 利用率、网络 bytes、对象请求数、p50/p99 和失败重试；只记录网络带宽无法判断是否被解压拖慢。

### 33.3.4 对象存储的删除与生命周期

删除请求也需要版本和权限语义。带 versioning 的 bucket 里，删除可能只是写 delete marker，旧版本仍占空间。生命周期规则如果按前缀粗暴清理，可能删掉仍被旧训练 job 引用的 checkpoint。发布流程应写 `retention_until` 或 generation lease，并让清理器只删除没有活跃 lease、没有 manifest 引用且超过 grace period 的对象。审计日志至少保留删除请求的 actor、policy revision 和匹配的对象版本。

## 33.4 POSIX 与并行文件系统：名字、锁与条带

### 33.4.1 POSIX 语义不是一个性能数字

POSIX 给出 open/read/write/rename/fsync 等接口及可观察语义。`write()` 返回并不代表数据已经落到稳定介质；`fsync()` 的具体成本与文件系统、挂载选项和设备 cache 有关。`rename()` 通常提供同一文件系统内的原子名字替换，但跨挂载点不是原子操作。目录项、inode、数据块和 page cache 的一致性边界必须分别考虑。

常见模式是“写临时文件、fsync 文件、rename、fsync 目录”。如果应用省略目录 fsync，机器掉电后可能出现文件已写而目录项未持久化。教学 toy 不模拟掉电，但生产 checkpoint 协议应明确哪些 fsync 或远端 commit 是必需的。

### 33.4.2 NFS、CephFS、Lustre、BeeGFS 的数据/元数据路径

NFS 把客户端通过网络访问远端 server；属性缓存和失效策略会影响多个客户端观察到的时间。CephFS 分离 MDS 元数据和 RADOS 数据对象，目录热点可能先压垮 MDS。Lustre 通过 MDS/MDT 管理名字和 OST 管理大块数据，条带数、条带大小和 OST 选择会决定并行读写；BeeGFS 也把 metadata 与 storage target 分离，并支持多目标条带。它们都可以提供 POSIX API，但锁、缓存、故障切换和运维工具不同。

并行文件系统的关键问题是 **metadata fan-in**：数千 worker 在同一目录 `stat()`、`open()`、`readdir()` 小文件，会让 MDS 成为瓶颈，即使 OST 带宽仍很高。将样本打包成较大 shard、提前生成索引、让每个 worker 读取自己的分区，可降低目录操作数量。不要把 `ls` 的速度当成训练读带宽。

### 33.4.3 条带化与顺序读

对一个大小为 `S` 的文件，条带大小 `W`、条带目标数 `K` 决定每个目标收到的块。`K` 太小，无法利用并行目标；`K` 太大，小文件的元数据和网络连接放大。顺序大文件常受益于较大 `W`，随机小读则可能在多个目标间产生 seek/queue 开销。测量要固定客户端数量、文件大小、预读策略和目标选择，否则不同 run 的差异无法解释。

跨节点训练还要考虑客户端网络：多个 worker 从同一共享文件系统读取，网络和 MDS/OSS 队列可能同时饱和。实践中常见两级缓存：共享文件系统保存 canonical shard，本地 NVMe 保存 job 期间的热 shard；作业结束清理缓存，canonical 版本仍由 manifest 绑定。

### 33.4.4 锁、租约和 split-brain

POSIX advisory lock、文件锁和分布式 lease 的故障行为不同。一个进程崩溃时，内核可能释放本地锁；一个网络分区时，客户端可能持有过期 lease 或被服务端 fencing。checkpoint writer 不能只依赖“目录不存在”来抢锁；需要带 owner、epoch、TTL 和 fencing token 的 lease。读取端拒绝 epoch 较旧的 writer，避免两个作业同时更新 `latest`。

## 33.5 NVMe 与本地盘：低延迟不等于无限容量

### 33.5.1 队列深度、块大小和尾延迟

NVMe 通过 PCIe 和多队列提交 I/O。小块随机读关注 IOPS 和 p99，顺序大块读关注带宽；队列深度增大通常提高吞吐，却可能增加 tail latency。`fio` 的 `iodepth`、`numjobs`、block size 和 direct I/O 设置都会改变结果。页缓存命中时，测到的是 DRAM 而不是 flash；想测介质需要明确 `O_DIRECT` 或预热/冷启动协议。

本地盘的故障域通常是单节点。它适合临时 shard cache、spill、shuffle 和快速 checkpoint staging，不适合作为唯一 canonical checkpoint。节点被回收、磁盘磨损或实例迁移时，本地 bytes 可能消失。每个本地缓存条目都应带来源 URI、版本 hash、大小和生成时间；cache miss 可以回源，cache hit 也要验证 hash。

### 33.5.2 写放大、TRIM 与空间水位

文件系统、日志、压缩和 flash FTL 会产生写放大。设应用写入 `A` bytes，设备实际写入 `W` bytes，则写放大 `WA=W/A`。checkpoint 频繁覆盖同一文件、产生许多临时副本时，`WA` 和 GC 可能让延迟慢慢恶化。监控应包括 bytes written、media errors、available spare、温度、以及 filesystem free/inode。只看剩余 GB 可能漏掉 inode 或 write endurance 问题。

空间策略至少有三道水位：

- **保护线：** 低于此值禁止新的大 checkpoint，保证恢复和清理有余量；
- **回收线：** 清理无 lease 的临时 shard 和过期 cache；
- **紧急线：** 进入只读/降级，停止可重算的预取，保护 canonical 数据。

把 `rm -rf` 作为自动化第一反应很危险；先根据 manifest 引用和 lease 判断哪些文件可删。

## 33.6 数据加载：从 shard 到 batch 的流水线

### 33.6.1 分片、样本顺序和 worker contract

设数据集有 `N` 个样本、`W` 个 data workers、`R` 个训练 rank。一个样本分配协议应定义：分片列表版本、shuffle seed、epoch、rank/world size、worker id、drop_last、重复样本规则和 cursor。简单 round-robin 可以保证不重叠，但若 shard 大小不均会导致 worker 负载倾斜；按字节或 token 数加权更接近真实工作量。

训练可复现不仅需要保存 random seed，还要保存“已经消费到哪个样本”。如果 worker 在预取队列中已有 batch，进程崩溃后重放可能重复或跳过样本。exactly-once 数据消费很昂贵；许多训练系统选择 at-least-once 加 sampler checkpoint，并把重复率作为可观测指标。对对比学习、RL 或有状态增强，重复样本可能改变优化轨迹，需要在实验记录里标注。

### 33.6.2 预取、背压和 pinned memory

数据加载通常有 producer（读/解压）、transform、collate、consumer（GPU copy）四段。队列太小，GPU 等 I/O；队列太大，占满内存并掩盖后端故障。设 batch 消费率为 `λ_c`，生产率为 `λ_p`，当 `λ_p<λ_c` 时队列最终为空；当 `λ_p>>λ_c` 时，backpressure 应限制 producer，防止无限缓存。监控 `queue_depth、wait_ms、decode_ms、copy_ms、cache_hit`。

Pinned memory 能加速 CPU 到 GPU 的异步复制，但会占用不可分页内存；本章 CPU toy 不模拟它。生产系统应把 pin budget、NUMA 亲和性和 worker 数写入配置，并用固定 batch 做 A/B。仅仅增加 `num_workers` 可能把共享文件系统和 CPU 解压打爆。

### 33.6.3 mmap、顺序读取和小文件

`mmap` 让页错误按需加载，适合可随机访问的本地大文件；它不自动解决远程文件一致性和 page-cache 抖动。对对象存储通常先下载到本地再 mmap。大量小文件会放大 open/stat/close 和 metadata RPC，打包成 tar、WebDataset、TFRecord、Parquet 或自定义 shard 可以减少请求数，但会牺牲随机重读和单样本更新。选择格式必须结合训练访问模式和故障重试粒度。

### 33.6.4 数据校验和“看起来能读”的坏样本

文件存在且能解压不等于内容正确。压缩流可能在尾部损坏，模型直到后几个 batch 才报错。读取端应在 shard 级验证 size/hash，在样本级验证 schema、token 范围和长度上限；坏样本策略要明确：隔离并计数、重取另一副本、还是中止实验。静默替换坏样本会改变数据分布，至少要记录替换列表和原因。

## 33.7 Checkpoint：把瞬时状态变成可恢复版本

### 33.7.1 Checkpoint 的状态集合

一个训练 checkpoint 不只是参数。最小集合包括：

- 模型参数和精度/量化配置；
- optimizer state、scheduler state、梯度 scaler；
- DP/TP/PP/EP 的 world size、rank mapping 与 shard layout；
- RNG（Python、NumPy、框架、每个 data worker）；
- 数据集 manifest、shuffle seed、epoch、sampler cursor；
- 代码/容器/依赖 digest、配置和 git commit；
- 最近一次完整 step、loss/吞吐摘要、拓扑和设备健康快照。

缺失 optimizer 或 sampler state 的“checkpoint”只能用于推理或近似恢复，不能宣称无缝继续训练。保存这些字段会增加 metadata，但它们是解释恢复结果的证据。

### 33.7.2 两阶段提交与 manifest

推荐的 checkpoint 协议：

```text
step-00100/_staging/<writer-epoch>/part-*.tmp
  -> fsync/upload + size/hash verification
  -> manifest.json (lists every part, dtype, shape, source step)
  -> COMMIT marker / conditional pointer
  -> latest -> step-00100
```

任何 reader 只解析带 `COMMIT` 且 manifest hash 正确的版本。`latest` 指针可以滞后，但不能指向未完成版本。提交后再异步复制到第二故障域；恢复器根据复制状态选择“本地完整版本”或“上一个跨域完整版本”，不得混拼不同 step。

manifest 需要防止路径穿越、重复 part、声明大小与实际大小不符和未知 schema。解析器应使用白名单字段、严格类型和最大 entries，避免一个损坏对象耗尽内存。manifest 本身也要版本化并带签名/校验和。

### 33.7.3 增量与分层 checkpoint

全量 checkpoint 的成本近似为模型和 optimizer 总 bytes；增量 checkpoint 只写自上次快照变化的块，但恢复需依赖完整链。可采用 `base + delta_1 + ... + delta_k`，每 `k` 个 delta 生成新的 base。清理策略要保留一条可解析链，不能删除仍被 delta 引用的 base。若 optimizer shard 每步都变化，增量节省可能比参数-only 预期小。

异步 checkpoint 会把写入从训练 critical path 移出，但需要 copy-on-write 或冻结一致的 state snapshot。直接在训练张量上并发序列化会得到前后混合的 tensor。框架提供的 distributed checkpoint API 负责分片与协调，但仍需验证 backend 的原子提交和重试语义。

### 33.7.4 Checkpoint 恢复验收

恢复验收分三层：

1. **结构层：** 所有 part、大小、hash、manifest、版本指针完整；
2. **语义层：** 固定小 batch 的 loss、梯度、参数 hash 或容差与 golden run 对齐；
3. **运行层：** 短程吞吐、通信 histogram、数据 cursor、checkpoint 再次提交成功。

“能够 load_state_dict”只通过结构层的一部分；如果 world size 或分片布局改变，必须明确转换逻辑和数值偏差预算。

## 33.8 一致性、故障与恢复边界

### 33.8.1 故障分类

把故障按数据平面位置分类比按产品名分类更有用：

- **命名故障：** DNS、bucket policy、权限、MDS unavailable；
- **可见性故障：** 写成功但 list 看不到、旧缓存未失效、跨域复制落后；
- **字节故障：** partial write、校验和不匹配、坏 block、压缩尾损坏；
- **容量故障：** quota、inode、对象请求限流、NVMe 空间/队列耗尽；
- **协调故障：** 两个 writer、过期 lease、rank cursor 不一致；
- **恢复故障：** checkpoint 链缺 base、版本指针回滚失败、重试不幂等。

每类故障都要有证据：request id、object version、inode/path、checksum、fsync/commit 时间、lease epoch、worker/rank 和重试次数。

### 33.8.2 Retry、幂等与未知提交

可重试错误包括连接超时、429、部分读、临时节点不可用；不可盲重试的是权限拒绝、schema 错误和校验和失败。对 PUT 使用 immutable key 或条件写，客户端超时后先查询 idempotency key/版本，再决定是否重发。对 range GET 要绑定版本和 range checksum；对 checkpoint manifest 更新使用 compare-and-swap，避免后写覆盖先写。

指数退避应加入随机抖动，且有总预算。所有 worker 同时重试会造成 thundering herd，把一个短暂故障放大为全局雪崩。训练端可暂停新 batch、降低预取并让已有 GPU 工作完成；推理端可对可重试 prompt 重新路由，对已发送 token 的流式请求执行幂等恢复或明确中止。

### 33.8.3 故障注入与“健康但错误”

至少演练：写入 shard 后在 commit 前杀进程；删除一个 part；篡改一个字节；让 list 延迟一个 tick；让对象 GET 返回 timeout；让本地 cache 与 canonical hash 不同；让两个 writer 使用不同 epoch。成功标准不是“程序没有崩溃”，而是：reader 不加载未提交版本、校验失败可定位、旧版本仍可恢复、训练 cursor 是否重放有明确记录、推理请求是否按 SLO 降级。

“健康但错误”的典型案例是：所有 HTTP 请求 200，训练 loss 也在下降，但一个 worker 重复消费了 3% 数据；或者所有 checkpoint part 都有 200 响应，但 manifest 列的是旧 tokenizer。必须把样本计数、schema digest、tokenizer hash、commit epoch 纳入指标，而不是只看状态码。

## 33.9 训练数据平面：吞吐、可复现和恢复

### 33.9.1 训练前的容量账本

训练规划至少列出：数据集 canonical bytes、压缩后 bytes、每 epoch 读取量、shuffle buffer、worker cache、checkpoint 全量/增量 bytes、保留版本数、跨域复制倍数和清理 grace period。若每 step 消费 `b` bytes、step rate 为 `r`，则读取速率下界为 `b*r`；加入解压和重试放大 `α` 后，目标 backend 带宽应大于 `α*b*r`，并为 checkpoint burst 留出余量。

不要把 GPU 进程数直接当作数据并发。一个 rank 可能有多个 loader worker、prefetch queue 和异步上传线程。容量账本应按“请求者”统计：对象 GET/PUT 数、文件打开数、NIC bytes、CPU 解压核数和 page cache bytes。

### 33.9.2 训练 step 与数据 cursor 的耦合

训练 step 提交前，数据 cursor 是否前移要有定义。若先前移再写 checkpoint，崩溃可能跳过样本；若先写 checkpoint 再前移，重启可能重复样本。常见折中是记录“已消费 batch 的上界”并允许少量重复，或为每个 shard 写消费日志再压缩。关键是把策略写进实验协议，不能在故障后猜测。

分布式 sampler 还要处理 world size 改变。若从 64 rank 恢复到 32 rank，旧 cursor 不能简单除二；应根据全局样本 ID、epoch 和 sampler schema 重新计算，验证没有遗漏和重复超出预算。

### 33.9.3 训练期间的 checkpoint 节流

checkpoint 频率由可接受重做时间、写带宽和故障概率决定。若 step 时间为 `t`，checkpoint 写入耗时为 `c`，每 `k` steps 一次，粗略开销是 `c/(k*t)`；异步写入可以降低前台阻塞，却会消耗后台带宽和 CPU。系统应在高 I/O 时降低预取并延后非关键复制，而不是让训练和 checkpoint 互相争抢全部链路。

## 33.10 推理数据平面：权重、prompt、KV 与检索

### 33.10.1 权重加载与启动风暴

多副本同时启动会从对象存储下载同一权重，造成 N 倍流量。可采用节点级 cache、分层预热、镜像层或 peer-to-peer 分发，但每层都要校验内容 hash 和版本。启动探针不能在权重尚未完整加载时报告 ready；ready 应包含模型 revision、tokenizer digest、dtype、KV 配置和依赖的检索索引版本。

权重文件是 immutable artifact；“覆盖同名文件然后 reload”容易让 mmap reader 看到混合版本。更安全的是写入新目录，校验完成后原子更新 revision pointer，旧 revision 在所有 worker drain 后再清理。

### 33.10.2 Prompt、检索上下文和输入缓存

在线请求的输入数据通常来自 API、对象文档、向量数据库和工具结果。prompt cache 命中减少 tokenization 和 prefill，但 cache key 必须包含模型 revision、tokenizer、system prompt、工具 schema 和安全策略版本。只用原始 prompt 字符串会在模型升级后返回旧 token 序列。

检索文档的读取延迟会影响 TTFT；应记录 `retrieval_ms、object_get_ms、decode_ms、queue_ms、prefill_ms`，区分后端慢和模型慢。对可复用文档可把规范化 chunks 放在本地 NVMe，canonical 仍在对象存储；cache miss 或校验失败时回源。

### 33.10.3 KV cache 与数据平面故障

KV cache 通常驻留 GPU HBM，也可能溢出到 CPU/NVMe 或通过 disaggregated serving 迁移。迁移字节与权重加载、prompt 读取共享 NIC 时，会造成尾延迟。cache entry 必须带请求/会话 ID、模型 revision、序列位置和格式版本。网络重试不能把同一 token 追加两次；迁移中断时宁可丢弃并重算，也不能静默拼接错误的 KV。

流式推理的恢复比离线 batch 更严格：已发送 token 不能被无提示替换。服务应定义客户端可见的 request id、重试策略和 partial-output 语义；对有外部副作用的工具调用使用幂等键。

## 33.11 从请求到硬件：一条可审计的机制链

下面按对象读取举例：

1. **应用层：** loader 根据 manifest 选择 shard，生成 request id 和版本约束；
2. **SDK 层：** 签名、连接池、重试预算、range 和 checksum header；
3. **网络层：** DNS、TLS、负载均衡、NIC、交换机队列和 MTU；
4. **服务层：** bucket policy、metadata index、object version、part/replica 选择；
5. **介质层：** storage node 的 page cache、SSD queue、纠删码/副本读；
6. **返回层：** checksum、解压、反序列化、worker queue、CPU/GPU copy；
7. **观测层：** 每层写入 trace/span 和 bytes/latency/error，统一 request id。

POSIX/NVMe 路径把第 2–5 步换成系统调用、VFS、page cache、block layer、NVMe submission queue 和 flash。并行文件系统增加 MDS/metadata RPC、客户端缓存、OST/target 选择。相同的应用接口并不意味着相同的故障语义；实验和 runbook 必须记录 backend 类型和 mount/SDK 版本。

## 33.12 CPU-only measured laboratory

实验脚本 `labs/ch33_storage_data_plane_lab.py` 提供三组可验证的代理：

- **后端读模型：** object、posix、nvme 使用透明的延迟和带宽参数，验证 `N·L + S/B` 的方向；
- **分片分配：** round-robin 将 shard 唯一分给 worker，输出 coverage 和不重叠证据；
- **checkpoint 协议：** 写入 shard hash 后才提交 manifest；`--fail-after` 模拟 commit 前故障，验证 reader 拒绝不完整版本。

它不会创建大文件，也不会接触 GPU、真实 S3、Lustre、NVMe 或 fsync。命令：

```bash
python3 labs/ch33_storage_data_plane_lab.py \
  --shards 12 --workers 3 --shard-mib 4 --checkpoint-step 100 \
  --output reports/ch33-storage-data-plane-default.json

python3 labs/ch33_storage_data_plane_lab.py \
  --shards 12 --workers 3 --shard-mib 4 --checkpoint-step 100 \
  --fail-after 8 --output reports/ch33-storage-data-plane-failure.json

python3 tests/test_ch33_storage_data_plane_lab.py
```

默认参数生成 12 个 shard、3 个 worker，所有 shard 唯一覆盖；failure 场景只写 8 个 shard，`committed=false`、`recoverable=false`。后端 proxy 的排序应为 NVMe 最快、POSIX 次之、object 最慢，但这只是配置中的机制演示。要测真实系统，固定对象版本、文件系统挂载、队列深度、CPU 亲和性、并发、缓存冷热状态和数据大小，然后记录 p50/p95/p99。

实验的最小证据表：

| 证据 | 含义 | 不能证明 |
| --- | --- | --- |
| `worker_shards`、`coverage` | 分片唯一性和覆盖 | 真实 sampler 在 world size 变化时无重复 |
| `elapsed_ms`、`throughput_mb_s` | 透明公式的方向 | 目标云/并行文件系统实际带宽 |
| `manifest_hash`、`committed` | 两阶段提交状态 | 掉电后硬件是否真的持久化 |
| `uncommitted_write_is_visible` | toy reader 不加载半版本 | 任意 SDK 的 list/read-after-write 语义 |

## 33.13 Failure clinic：症状、证据和修复顺序

### 症状 A：GPU 利用率每隔几分钟归零

先查 loader queue depth、对象 GET p99、重试次数、解压 CPU 和 page faults，再查 GPU kernel。若 checkpoint 正好每 100 step 写入且对象 PUT 与 GET 共享限额，可能是数据面自我拥塞。修复顺序是限并发/设置预算、错峰 checkpoint、增加本地 cache，再评估扩容；盲目加 GPU 只会增加读请求。

### 症状 B：训练恢复后 loss 曲线偏移

验证 checkpoint manifest、optimizer/scaler、RNG、sampler cursor、tokenizer/dataset digest、world size 和代码版本。若结构完整但样本重复率上升，偏移可能来自 cursor 重放；若 rank mapping 变化而 optimizer shard 未转换，数值也可能漂移。保留 golden batch 和短程对照，区分加载错误、数据顺序改变和非确定性 kernel。

### 症状 C：对象目录看见了 checkpoint，但恢复器报缺 shard

检查 list 分页、前缀过滤、版本 ID、删除 marker、跨域复制状态和 manifest 指针。恢复器应只读 manifest 声明的 immutable keys，而不是重新 list 目录。若 manifest 已 commit 但 part hash 不匹配，停止自动重试，进入隔离和回滚。

### 症状 D：推理冷启动 p99 超过 SLO

拆分权重下载、校验、解压、mmap、GPU copy、编译和 ready probe 时间。比较节点 cache hit/miss 与对象请求并发。建立预热池或分层 cache 前，先确保 revision pointer 和 hash 校验；否则启动更快但可能加载错误版本。

### 症状 E：本地 NVMe 快取命中率高但结果错误

查 cache key 是否含模型/数据版本、是否在节点回收后复用旧目录、是否验证 hash/size、是否存在并发 writer。清理并重建 cache 前先保留坏条目证据；修复应包括 generation-aware key、原子 rename 和后台 scrub，而不只是调高 TTL。

## 33.14 权衡与被拒绝的替代方案

### 33.14.1 全部放对象存储

优点是容量和跨区域耐久性，缺点是小请求/列表/启动延迟、网络费用和请求限流。适合作为 canonical 数据和跨故障域 checkpoint，不适合直接承载每个样本的随机小读。必须配合打包 shard、manifest 和本地 cache。

### 33.14.2 全部放共享 POSIX

优点是应用改动小、rename/权限语义熟悉；缺点是 metadata fan-in、MDS/OST 故障域、跨租户争抢和容量扩展。适合高带宽训练集和团队共享 scratch，但仍应保留对象存储 canonical 副本，不能把单一挂载点当备份。

### 33.14.3 全部放本地 NVMe

优点是低延迟，缺点是节点故障丢失、容量有限、复制成本和生命周期管理。适合 cache、spill、shuffle、权重预热；不适合作为唯一 checkpoint 或数据集来源。

### 33.14.4 只做全量 checkpoint

实现简单但 I/O burst 大、恢复时间长。增量/分层增加协议复杂度和垃圾回收风险。若选择增量，必须有链完整性测试、定期 base、跨域复制和删除保护；资源不足的小实验可以全量，但要明确恢复窗口。

### 33.14.5 依赖“最终一致性下多重重试”

重试不能修复错误版本、错误 manifest 或数据损坏，反而会放大流量。应采用 immutable object、显式版本、条件提交、checksum 和有界退避；对未知提交先查询状态，再决定是否重发。

## 33.15 论文、官方文档与仓库综合

- **S3**：Amazon 的 [S3 API 文档](https://docs.aws.amazon.com/AmazonS3/latest/API/Welcome.html) 和 [数据一致性模型](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html) 是对象语义入口；具体云服务的限额和跨区复制需按区域验证。
- **POSIX/Linux**：[`open(2)`](https://man7.org/linux/man-pages/man2/open.2.html)、[`fsync(2)`](https://man7.org/linux/man-pages/man2/fsync.2.html)、[`rename(2)`](https://man7.org/linux/man-pages/man2/rename.2.html) 解释系统调用边界；[Linux block layer 文档](https://docs.kernel.org/block/index.html) 解释 I/O 路径。
- **NVMe**：[NVM Express Base Specification](https://nvmexpress.org/specifications/) 与 [SPDK](https://github.com/spdk/spdk) 展示队列和用户态 I/O；[fio](https://github.com/axboe/fio) 用于固定 workload 的测量。
- **并行文件系统**：[Lustre 文档](https://doc.lustre.org/lustre_manual.xhtml)、[BeeGFS 文档](https://doc.beegfs.io/latest/)、[CephFS 文档](https://docs.ceph.com/en/latest/cephfs/) 和 [NFS RFC 8881](https://www.rfc-editor.org/rfc/rfc8881) 描述 metadata/data path、条带、缓存与故障语义。
- **数据集与加载**：[WebDataset](https://github.com/webdataset/webdataset)、[NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)、[PyTorch DataLoader](https://pytorch.org/docs/stable/data.html) 提供 shard、worker、prefetch 和解码参考；框架默认值不能替代目标数据的测量。
- **Checkpoint**：[PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)、[torch.distributed.checkpoint 源码](https://github.com/pytorch/pytorch/tree/main/torch/distributed/checkpoint) 和 [TorchElastic](https://pytorch.org/docs/stable/elastic/run.html) 说明分片、重启和 world-size 变化的接口；[DeepSpeed ZeRO checkpoint](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html) 说明 optimizer/parameter 分片边界。
- **训练系统论文**：[Megatron-LM](https://arxiv.org/abs/2104.04473) 讨论大规模并行训练；[ZeRO](https://arxiv.org/abs/1910.02054) 讨论状态分片；[BytePS](https://www.usenix.org/system/files/osdi20-jiang.pdf) 讨论异构层级聚合；[MegaScale](https://www.usenix.org/system/files/nsdi24-jiang-ziheng.pdf) 展示大规模训练中的 checkpoint、诊断和故障恢复。
- **推理数据面**：[vLLM 文档](https://docs.vllm.ai/en/stable/)、[TensorRT-LLM disaggregated serving](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html)、[SGLang 文档](https://docs.sglang.ai/) 说明权重、KV 和 prefill/decode 数据路径；真实 SLO 需记录版本、模型和网络。
- **可观测性**：[OpenTelemetry tracing](https://opentelemetry.io/docs/concepts/signals/traces/)、[Prometheus histogram](https://prometheus.io/docs/practices/histograms/) 和 [eBPF block I/O](https://github.com/iovisor/bcc/tree/master/tools) 可把请求、设备和尾延迟关联起来。

这些来源分别支撑接口或机制事实；本章关于 cache 层次、checkpoint 两阶段提交、容量水位和故障演练的具体阈值属于设计建议，必须在目标环境用实验验证。

## 33.16 理解检查（含答案）

1. **为什么“对象 PUT 返回 200”不足以证明训练可以恢复？**  
   答：可能仍缺少其他 shard、manifest、checksum、跨域复制或 commit pointer；恢复器应验证完整版本而不是单个请求状态。
2. **何时减少文件数比提高存储带宽更有效？**  
   答：当请求启动/metadata 延迟 `N·L` 占主导、存在大量小文件或 MDS/list fan-in 时，打包 shard 能减少请求和目录操作。
3. **`fsync(file)` 和原子发布之间还缺什么？**  
   答：同一文件系统内的名字替换及目录项持久化；还需 fsync 目录或使用后端明确的 durable commit，并由 manifest 绑定所有 part。
4. **为什么 checkpoint reader 不能按目录列表加载所有文件？**  
   答：列表可能包含临时、旧版本、delete marker 或分页遗漏；manifest 是声明的版本边界，应按 immutable key 和 hash 读取。
5. **world size 改变时只把 sampler cursor 除以二有什么风险？**  
   答：旧分片布局、shuffle 和 drop_last 规则不同，可能重复或跳过样本；需按全局样本 ID 和 sampler schema 转换并测量。
6. **本地 NVMe cache 命中率 99% 仍可能返回错误数据吗？**  
   答：可能。若 key 未包含模型/数据版本、cache 未校验 hash 或并发 writer 产生混合文件，命中率高只说明读到了某些 bytes，不证明语义正确。

## 33.17 练习：从守恒到设计

- **回忆题：** 列出 object、POSIX/并行文件系统、NVMe 三种后端各自的 canonical、cache 和故障域角色。
- **推导题：** 给定 64 个 worker、每个 worker 每秒读取 80 MiB、对象请求启动延迟 6 ms，比较 1 MiB 小文件与 256 MiB shard 的理论请求开销。说明为什么只比较 GB/s 不够。
- **实现题：** 扩展 toy lab，加入 `list_delay_ticks` 和带版本 ID 的 object pointer；测试 reader 在 list 延迟和未知提交时仍只加载已提交 manifest。
- **诊断题：** 训练 p99 每 500 step 飙升，日志只显示 GPU idle。设计一条从 loader queue、object p99、解压 CPU、checkpoint PUT 到交换机队列的证据链，并写出停止条件。
- **设计题：** 为 1 PB 数据集、8,000 GPU、跨三个可用区的训练任务设计 canonical/shard/cache/checkpoint 分层。给出 shard 大小、并发预算、保留策略、租约和区域故障恢复顺序；明确哪些数字必须通过 burn-in 测量。

## 33.18 小结与下一依赖

本章把存储从“一个磁盘”提升为数据平面协议：对象存储提供带版本的 key 和跨域耐久性，POSIX/并行文件系统提供名字、锁和高带宽共享路径，NVMe 提供低延迟但节点级易失的 cache。训练数据加载需要 shard、sampler、预取、backpressure、校验和 cursor 共同构成；checkpoint 需要所有状态、两阶段提交、manifest、版本指针和恢复验收。推理则要把权重、prompt、检索和 KV cache 的版本与 SLO 连接起来。

下一步可以回到第20、21章，把本章的 request/span、checksum、lease、queue depth 和 checkpoint commit 接入可观测性与可靠性 runbook；也可以结合第27、30章，研究 KV 迁移和压缩如何改变数据面字节与故障边界。无论选择哪条路径，先保留版本、大小、校验和、时间、所有者和恢复证据，再谈“更快”。

## 33.19 来源地图与可复现记录

- 本章实验脚本：`labs/ch33_storage_data_plane_lab.py`（Python 3.10+，标准库，CPU-only）。
- 合同测试：`tests/test_ch33_storage_data_plane_lab.py`；覆盖分片唯一性、后端延迟方向、checkpoint 原子提交、故障和 CLI JSON。
- 默认与故障输出：`reports/ch33-storage-data-plane-default.json`、`reports/ch33-storage-data-plane-failure.json`；报告见 `reports/ch33-storage-data-plane-report.md`。
- 证据清单：`evidence/ch33-storage-data-plane-manifest.json`；记录官方文档、论文、成熟仓库和本地测量的用途与验证方法。
- 环境边界：实验不访问云账户、不上传数据、不创建大文件；生产复现需锁定对象服务区域/API、文件系统和 kernel/客户端版本、NVMe 型号、CPU/NUMA、缓存冷热状态、并发、数据大小、失败注入时间和清理策略。
- 结果解释：toy 的 `elapsed_ms`、`throughput_mb_s`、`committed` 与 `recoverable` 只证明脚本中的透明合同；不能证明任何厂商的带宽、耐久性、一致性或生产恢复时间。
