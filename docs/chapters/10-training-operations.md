---
id: ch10-training-ops
title: 大规模训练运维：调度、弹性、检查点与实验可复现
slug: /chapters/10-training-ops
description: 从队列与资源调度、数据局部性和抢占恢复，到检查点、实验追踪、集群利用率与发布回滚，建立可解释、可恢复、可复现的训练运维体系
sidebar_position: 10
level: systems
prerequisites:
  - ch02-linux-process-files-observability
  - ch04-performance-math
  - ch06-pytorch-execution
  - ch08-distributed-collectives
learning_objectives:
  - 能把训练作业拆成提交、排队、放置、运行、抢占、恢复和发布等可观测状态，并为每个状态定义超时与证据
  - 能解释 gang scheduling、配额、公平性、优先级、碎片化和拓扑感知放置的取舍
  - 能计算数据局部性、缓存命中、网络流量和 straggler 对端到端吞吐的影响
  - 能设计包含模型、优化器、数据游标、随机数、配置和代码版本的可恢复 checkpoint
  - 能区分静态 world size、弹性 world size 与抢占后的语义变化，避免重复样本或错误梯度累积
  - 能用 CPU-only 模拟验证调度、抢占、checkpoint 原子提交与恢复时间线
  - 能建立实验追踪、指标、制品和环境锁定，使结果可复现并能审计
  - 能用利用率、队列等待、p95 恢复时延和发布回滚指标进行容量与风险决策
estimated_hours: 22
hardware: CPU-only baseline; GPU/高速网络/批处理调度器 optional
risk_level: L2
last_verified: 2026-10-05
---

# 第10章　大规模训练运维：调度、弹性、检查点与实验可复现

> 模型训练的最后 10% 往往消耗前面 90% 的工程时间：作业排不上队、某个节点被抢占、数据缓存失效、检查点损坏、恢复后指标悄悄漂移，或者同一个配置下再也跑不出相同结果。本章把训练看成一个由调度、数据、状态和发布共同构成的系统，而不是“提交一个脚本，等它跑完”。核心问题有四个：资源是否被放在正确的位置，作业在被中断后能否继续，结果是否能解释和重现，平台是否能在失败时安全回滚。文中的公式和模拟都以 CPU 为基线，迁移到 GPU 集群时必须重新测量。

## 10.1 训练运维是算法的一部分

### 10.1.1 从单个进程到作业状态机

一个训练作业至少经历以下状态：

```text
SUBMITTED -> ADMITTED -> QUEUED -> PLACED -> RUNNING
                               |          |
                               |          +-> PREEMPTING -> CHECKPOINTING
                               |                            -> SUSPENDED
                               +-> REJECTED
RUNNING -> SUCCEEDED
RUNNING -> FAILED -> RETRYING -> QUEUED
SUSPENDED -> RESUMING -> PLACED
```

状态不是 UI 上的颜色，而是有进入条件、离开条件和证据的契约。`PLACED` 表示调度器已经选择了节点，但容器尚未完成初始化；`RUNNING` 应至少意味着 rank 进程组已建立、数据迭代器可读且首个训练 step 已开始。若只看 Pod 为 `Running`，可能把镜像拉取、挂载失败或 NCCL 初始化等待误判成有效训练时间。

每个状态都要记录单调时钟和墙钟：`state_enter_ts`、`state_exit_ts`、`scheduler_attempt`、`allocation_id`。墙钟便于跨系统关联，单调时钟避免 NTP 调整使时长为负。状态转移必须幂等：重复收到节点释放事件时不能生成第二个 checkpoint 提交，也不能把已成功的作业标记为失败。

### 10.1.2 三种时间：队列、计算与恢复

端到端完成时间可写成：

\[
T_{e2e}=T_{queue}+T_{provision}+T_{startup}+T_{compute}+T_{preempt}+T_{checkpoint}+T_{recovery}+T_{teardown}.
\]

很多团队只优化 `T_compute`，但在高峰期 `T_queue` 可能占 70%，抢占频繁时 `T_checkpoint+T_recovery` 又会吞掉有效算力。报告应同时给出有效训练时间 `T_useful` 和墙钟吞吐：

\[
\text{有效吞吐} = \frac{\text{已提交且可验证的样本数}}{T_{e2e}},\qquad
\text{算力利用率} = \frac{T_{useful}}{T_{allocated}}.
\]

如果恢复后重放了 30 分钟数据，GPU 可能显示 90% 利用率，但有效吞吐低于没有重放的基线。运维指标必须围绕“完成了多少可信工作”，不能只围绕设备忙碌比例。

## 10.2 作业调度：谁先跑、放在哪里、何时让路

### 10.2.1 资源向量与请求合同

训练作业的资源请求应是向量而非单一的 GPU 数：

```text
R = (cpu_cores, memory, gpu_count, gpu_mem, nic_bw,
     local_nvme, shared_fs_bw, network_class, topology_hint)
```

`gpu_count=8` 不能表达每卡显存、GPU-NIC 亲和、NVLink 互联或是否允许跨节点。调度器需要知道硬性约束（必须有 8 张同型号 GPU）与软性偏好（尽量同 NUMA、尽量本地缓存）。把偏好误写成硬约束会造成队列饥饿；把硬约束写成偏好会在运行时失败。

请求合同还应包括生命周期：预计运行时长、可接受抢占次数、优雅终止预算（例如 120 秒）、检查点目标间隔、最大重试次数和失败后是否保留日志。调度器不能从“训练脚本可能跑几天”猜出这些信息，因为错误的运行时长估计会让短作业等待，也会让长作业频繁被挤出。

### 10.2.2 优先级、公平性与配额

常见策略包括先进先出（FIFO）、静态优先级、加权公平队列（fair-share）、最短作业优先（SJF）和基于截止期的调度。没有单一策略能同时最小化平均等待、尾部等待和资源碎片化。

- FIFO 容易理解，但一个大作业会挡住大量小作业，队列头阻塞明显。
- 优先级适合生产作业，但若优先级可随意提升，低优先级研究作业会永久饥饿。
- Fair-share 按项目历史消耗调整权重，可缓解长期不公平，但短时突发作业可能感觉“排不上”。
- SJF 降低平均周转时间，却会让长时间训练反复被推迟。
- 截止期调度需要可信的剩余时长估计，估错会造成级联抢占。

配额（quota）是租户在时间窗口内可占用的上限，不等于当前分配。`quota=100 GPU-hours/day` 允许本日最多消耗 100 GPU 小时，但如果集群只剩 2 张 GPU，作业仍要排队。要同时展示“配额不足”“集群无空闲”“拓扑不匹配”“镜像/数据未就绪”四种原因，避免用户把所有等待归因于一个黑盒优先级。

### 10.2.3 gang scheduling 与弹性准入

数据并行作业通常要求一组 rank 同时就绪。gang scheduling 的准入条件是“一次性拿到最小所需资源”，而不是先启动一半 rank 再等待其余 rank。前者避免部分启动占用节点并阻塞其他作业；后者在弹性训练中可能有意义，但必须由训练框架明确支持。

一个实用的准入协议：

1. 调度器预留候选节点与 GPU，并锁定拓扑。
2. 节点代理确认驱动、容器、挂载和端口可用。
3. rendezvous 服务返回完整成员列表和代次 `membership_epoch`。
4. 训练进程验证 world size、dtype、数据分片和 checkpoint 代次。
5. 首个全局 step 完成后，作业才从 `STARTING` 标为 `RUNNING`。

如果第 2 步失败，应释放全部预留，而不是留下半套 GPU；如果第 4 步发现 checkpoint 来自不同 world size，要转入恢复流程而不是强行继续。

### 10.2.4 bin packing、碎片化与回填

GPU 资源的碎片化有两种：数量碎片（总 GPU 足够但分散在不同节点）和形状碎片（GPU 型号、显存或互联不匹配）。调度器常用 best-fit、first-fit decreasing 或拓扑感知的图匹配。回填（backfilling）允许短作业使用被大作业预留的空隙，但前提是不能推迟已承诺的启动时间。

设节点剩余资源为 `c_i`，作业请求为 `r`。即使 \(\sum_i c_i \ge r\)，也不代表存在某个节点集合满足拓扑和网络约束。可以用可行性检查：

\[
\exists S: \sum_{i\in S} c_i\succeq r \land \text{topology}(S)\in H,
\]

其中 `H` 是作业允许的拓扑集合。调度日志应解释候选被拒的第一条硬约束，而不是只给“insufficient resources”。

## 10.3 放置与数据局部性

### 10.3.1 局部性层次

训练样本可能位于多层存储：

```text
GPU HBM -> host RAM -> 本地 NVMe -> 节点共享盘 -> 分布式文件系统 -> 对象存储
```

访问延迟和带宽沿路径变化数个数量级。把所有数据放进“可访问”集合并不等价于吞吐可接受。一个 epoch 的数据读取时间可近似为：

\[
T_{io}=\sum_{l} \frac{B_l}{BW_l}+\sum_l N_l\cdot L_l,
\]

其中 `B_l` 是从第 `l` 层读取的字节数，`BW_l` 是有效带宽，`N_l` 是请求次数，`L_l` 是每次请求延迟。小文件过多时，`N_l·L_l` 主导；把文件复制到本地并不会解决元数据服务瓶颈，除非同时合并请求或使用索引。

### 10.3.2 分片、缓存与一致性

数据并行通常把样本分片给 rank。`DistributedSampler` 之类的实现需要在每个 epoch 设置相同的种子和长度规则；世界大小变化后，旧的 `rank/world_size` 不再是有效分片身份。弹性恢复时应保存逻辑样本游标或可重放的 shard+offset，而不是只保存“当前 epoch=3”。

缓存策略可以分为：

- **预热缓存**：作业启动前把热数据复制到本地 NVMe，启动慢但首个 epoch 稳定。
- **按需缓存**：首次读取时下载并写入缓存，启动快但多个节点可能同时放大流量。
- **共享缓存**：节点或机架共享，复用率高但增加网络和一致性复杂度。
- **内容寻址缓存**：按数据块哈希命名，能去重和校验，但索引和垃圾回收要纳入资源预算。

缓存条目必须带数据集版本和校验和。只按文件名命中会把旧预处理结果误当成新版本，造成“代码和配置相同但指标不同”的隐蔽问题。缓存清理要有租户边界，不能让一个项目写满所有节点的本地盘。

### 10.3.3 straggler 与同步边界

同步训练的 step 时间接近最慢 rank：

\[
T_{step}=\max_r(T_{compute,r}+T_{io,r})+T_{collective}.
\]

平均读取带宽很高并不能说明尾部好；应记录每 rank 的 p50、p95、p99 数据等待时间。单个远端对象存储请求超时可能让所有 rank 在下一次 collective 等待。可采用预取、分片重排、慢节点剔除或异步数据读取，但每种优化都会改变顺序和恢复语义，必须配合 checkpoint 中的数据游标验证。

## 10.4 抢占与弹性训练

### 10.4.1 抢占类型

抢占可按通知窗口分为三类：

1. **优雅抢占**：收到 SIGTERM 或调度器事件，有足够时间写 checkpoint，再由 SIGKILL 结束。
2. **短窗口抢占**：只有几秒，需要写增量或轻量元数据，不能假设完整 checkpoint 可完成。
3. **硬抢占**：节点立即回收，进程和本地临时数据消失，只能依赖远端最近一致版本。

训练框架应把信号处理和 checkpoint 写入放在独立线程或协程，但必须限制并发写。收到多个信号时只启动一次保存，并让主训练循环在安全点停下；如果在任意 CUDA kernel 中强行复制模型，可能产生半写状态。

### 10.4.2 静态、可伸缩和弹性 world size

- **静态 world size**：作业一旦启动，rank 数不变。最容易复现；节点丢失通常整作业失败后重启。
- **可伸缩 world size**：框架允许在检查点边界增加或减少 rank，但全局 batch、学习率和数据分片需要重新计算。
- **弹性 world size**：成员可在训练过程中变化，rendezvous 以代次管理成员。必须定义旧 rank 的梯度、未消费样本和优化器状态如何处理。

若每个 rank 的本地 batch 为 `b`，world size 为 `P`，全局 batch 为 `B=P·b·accumulation`。从 8 缩到 4 而不调整学习率，优化器看到的梯度统计就会改变。线性缩放学习率只是启发式，动量、梯度裁剪、混合精度损失缩放和 scheduler 进度也要同步改变。恢复日志应记录 `old_membership_epoch -> new_membership_epoch`、全局 batch、有效样本数和优化器超参。

### 10.4.3 抢占协议的时间线

一个可测试的协议如下：

```text
T0 scheduler sends PREEMPT_NOTICE(deadline=T0+120s)
T1 trainer stops admitting new batch; marks step as draining
T2 all ranks reach a safe point and freeze optimizer updates
T3 rank 0 writes manifest.intent; workers upload shard files
T4 every shard checksum verified; rank 0 commits manifest.done
T5 trainer records membership_epoch and exits 0 (graceful)
T6 scheduler releases nodes
T7 new allocation validates manifest.done and resumes
```

如果在 `T3` 后断电，远端可能只有部分 shard。恢复逻辑只能选择上一个 `manifest.done`，不能根据目录中“看起来最大的 step”猜测。退出码也要区分：优雅抢占通常是可重试的非错误；数据损坏或契约不一致则不可盲目重试。

## 10.5 检查点：保存什么、何时提交、如何证明可信

### 10.5.1 状态清单

可恢复训练至少需要：

- 模型参数与缓冲区（包括 EMA、量化 scale、稀疏 mask 等非显式权重）；
- 优化器状态（动量、二阶矩、step、loss scale）；
- 学习率 scheduler、梯度累积计数、梯度裁剪统计；
- 数据集版本、分片列表、epoch、样本游标或可重放的随机种子；
- Python、NumPy、框架、CUDA（若有）随机数状态；
- world size、rank 映射、membership epoch、有效全局 batch；
- 配置、代码提交、容器镜像摘要、数据处理流水线版本；
- 指标快照、最佳模型指针和评估集版本。

只保存 `model.state_dict()` 不能保证恢复。优化器缺失会让动量归零；数据游标缺失会重复或跳过样本；随机状态缺失会让 dropout 和增强序列改变。是否要求 bitwise 重现取决于任务，但必须显式写出容差和验收指标。

### 10.5.2 两阶段提交与原子可见性

推荐把 checkpoint 写成不可变 shard 加小型 manifest：

```text
run_id/
  ckpt-000120/
    rank-0000.bin
    rank-0001.bin
    ...
    manifest.intent.json
    manifest.done.json
    checksums.sha256
  LATEST -> ckpt-000120
```

流程是“先写数据，后写意图，再校验，最后写 done”。`manifest.done.json` 中包含 step、world size、数据版本、每个 shard 的长度与哈希、提交时间和写入工具版本。`LATEST` 只能在 done 存在后更新。对象存储的目录列表不一定是强一致的，不能用“列出目录后看到文件”作为提交证明。

跨节点上传应设置超时、重试上限和幂等键；重试相同 shard 时使用同一对象名或内容寻址，避免生成多个无法区分的副本。清理旧 checkpoint 时先保留最近两个已验证版本，再删除孤儿 shard；不要在提交线程仍运行时并发删除。

### 10.5.3 频率、成本与恢复点目标

检查点间隔 `I` 越短，丢失工作越少但写入开销越大。可用简化的 Young/Daly 模型估算：若平均故障间隔为 `M`，一次 checkpoint 时间为 `C`，最佳间隔数量级接近 \(\sqrt{2MC}\)，但分布式训练还要考虑全局同步和对象存储峰值。生产上应从实测 p95 写入时延和抢占概率出发，分别给出：

- **RPO（恢复点目标）**：最多允许丢失多少有效 step 或样本；
- **RTO（恢复时间目标）**：从节点释放到首个有效 step 需要多久；
- **写入预算**：每小时可用多少带宽和存储费用；
- **保留策略**：最近 N 个、每小时一个、最佳指标和发布前锚点。

如果 checkpoint 写入与训练并行，必须证明异步写不会读取被后续 optimizer 修改的 buffer。常见做法是双缓冲、copy-on-write 或在安全点短暂停顿；只依赖 Python 引用计数是不够的。

## 10.6 恢复语义：重试不等于继续

### 10.6.1 至少一次、至多一次与有效一次

训练样本消费和 checkpoint 提交共同决定语义：

- **至少一次**：恢复可能重放样本。实现简单，但需要确认重复样本对指标可接受。
- **至多一次**：宁可跳过未确认样本，也不重复；需要可靠游标提交，可能损失数据。
- **有效一次**：希望每个样本对优化器只生效一次。通常需要把数据游标、梯度更新和 checkpoint 做成同一事务，代价最高。

大多数大模型训练采用“step 边界至少一次”，并通过固定随机种子和样本哈希监测重复比例。不要在文档中笼统写“支持断点续训”，要写出恢复点前后样本是否可能重复、学习率是否回退、指标是否重算。

### 10.6.2 恢复验证清单

恢复前检查：

1. manifest 签名/哈希和每个 shard 校验通过；
2. 模型结构、参数 dtype、优化器类型与当前代码兼容；
3. 数据集版本、预处理哈希和 schema 匹配；
4. world size 变化有明确迁移函数；
5. step、epoch、scheduler 与日志中的最后提交一致；
6. 随机状态可加载，或已声明非确定性；
7. 新节点的驱动、框架和通信后端在允许的版本范围内；
8. 先运行一个小的恢复探针，验证 loss、梯度范数和样本哈希，再放量。

探针应在隔离队列执行，避免恢复失败污染生产作业。若只测试“能够加载 checkpoint”，却不执行一个 optimizer step，可能遗漏 lazy state、dtype 转换或数据迭代器错误。

## 10.7 实验追踪与可复现性

### 10.7.1 一次实验的最小可识别集合

一个 run 的主键可以写成：

```text
run_id = hash(code_commit, config_canonical, data_snapshot,
              image_digest, launcher_version, seed_policy)
```

`run_id` 不应只取随机 UUID；UUID 仍可保留作显示名，但内容哈希让两个系统能够判断是否逻辑同一实验。记录内容至少包括：

- 代码提交和未提交 diff 的摘要；
- 规范化配置（排序键、显式默认值、单位）；
- 数据集快照、样本计数、过滤规则和预处理镜像；
- 容器镜像 digest、主机内核、驱动、框架与编译器版本；
- 启动命令、环境变量白名单、调度器 allocation_id；
- 随机种子策略、world size、全局 batch 和梯度累积；
- step/epoch 指标、吞吐、显存、CPU、网络、checkpoint 状态；
- 评估脚本版本、评估集哈希、阈值和人工审核备注。

密钥、访问令牌、用户原始样本和完整环境变量不能进入追踪系统。应使用引用（secret version、数据权限标识）而不是明文。追踪系统本身也需要租户隔离和保留策略。

### 10.7.2 指标的时间语义

每个指标要标记发生时间和聚合窗口：`step_time` 可是最近 100 step 的平均，`loss` 可是当前 micro-batch，`gpu_util` 通常是采样窗口均值。恢复后如果把旧曲线直接追加到新 run，仪表盘可能显示重复 step。建议使用单调的 `global_step`，并为每个 `membership_epoch` 增加维度；重放数据时记录 `replayed=true`。

实验追踪的“可复现”不等于每个浮点值完全相同。应同时定义：

- 数值复现：loss/accuracy 在给定容差内；
- 轨迹复现：关键 step 的样本哈希和 scheduler 状态一致；
- 环境复现：镜像和依赖锁定；
- 过程复现：调度、抢占、恢复事件可重放；
- 语义复现：改变 world size 后结果仍符合声明的缩放规则。

## 10.8 集群利用率、容量与成本

### 10.8.1 不只看 GPU busy

设备利用率常由采样器统计，不等于有效训练。建议每个作业报告：

\[
U_{effective}=U_{device}\times (1-f_{input\_stall})\times(1-f_{preempt})\times(1-f_{replay}).
\]

这只是诊断近似，不能替代端到端吞吐。还要监控：队列等待 p50/p95、资源碎片率、启动失败率、抢占率、checkpoint 成功率、恢复 RTO、每有效样本成本和电力/散热约束。将 GPU 小时与有效 tokens 绑定，才能比较不同并行策略。

碎片率可以粗略定义为：

\[
F=1-\frac{\text{可满足的最大作业请求容量}}{\text{空闲容量总和}}.
\]

如果空闲 GPU 分散到四种型号，`F` 可能很高。容量规划应按拓扑和显存分桶，而不是只做一个总数。

### 10.8.2 SLO 与告警

可定义如下 SLO（示例值需按组织调整）：

- 95% 交互式作业在 10 分钟内获得首个可用节点；
- 99% 优雅抢占在截止期前完成 checkpoint 提交；
- 恢复后 15 分钟内完成首个有效 step；
- 每月 checkpoint 损坏率低于 0.1%；
- 关键发布能在 5 分钟内回滚到上一个已验证模板。

告警应包含可执行上下文：哪个租户、哪个队列、哪种资源、最近一次状态转移和建议动作。不要只设“GPU 利用率低于 50%”告警；低利用率可能是作业在等待数据或 checkpoint，应该导向不同排查路径。

## 10.9 发布与回滚

### 10.9.1 训练模板和运行时版本

将 launcher、容器、调度器模板、数据管道和 checkpoint schema 一起版本化。一个生产训练模板的 digest 至少包含：

```text
launcher_commit
container_digest
framework+driver range
scheduler_api_version
checkpoint_schema_version
data_pipeline_version
```

升级框架或通信库时，先在 CPU 小规模运行语义探针，再在单节点目标硬件跑吞吐和恢复测试，最后进入 canary 队列。canary 的输入、样本顺序和指标阈值应固定，不能只看“作业成功退出”。

### 10.9.2 蓝绿、金丝雀与回滚

- **蓝绿**：保留旧模板（蓝）和新模板（绿），先在少量作业验证，失败则把流量切回蓝。
- **金丝雀**：选择代表性数据和短时长作业，比较吞吐、数值和恢复指标。
- **影子运行**：复制配置和输入但不提交生产模型更新，验证读取、调度和监控。

回滚需要三件可定位的东西：旧模板 digest、兼容的 checkpoint schema、以及明确的写入/读取方向。新版本如果已经写入不可逆格式，旧版本可能无法加载；此时应先双写或提供迁移工具，并在发布前验证回滚。不要把“镜像可拉取”当成“可回滚”。

## 10.10 CPU 可运行模拟：调度、抢占、checkpoint 与恢复

下面的脚本只使用 Python 标准库，模拟两个队列、有限 CPU 资源、优雅抢占和原子 checkpoint。它不模拟真实 GPU 或网络速度，目的在于让状态机、时间线和恢复语义可观察。将代码保存为 `ch10_cpu_ops_sim.py` 后运行：

```python
from dataclasses import dataclass, field
from pathlib import Path
import hashlib, json, heapq, shutil, tempfile

@dataclass(order=True)
class Job:
    sort_key: tuple = field(init=False, repr=False)
    priority: int
    submit: int
    name: str = field(compare=False)
    need_cpu: int = field(compare=False, default=2)
    total_steps: int = field(compare=False, default=12)
    step_cost: int = field(compare=False, default=1)
    checkpoint_every: int = field(compare=False, default=4)
    step: int = field(compare=False, default=0)
    state: str = field(compare=False, default="QUEUED")
    preempt_at: int | None = field(compare=False, default=None)
    resume_count: int = field(compare=False, default=0)
    last_ckpt: int = field(compare=False, default=-1)

    def __post_init__(self):
        # 负 priority 使高优先级排在前面；submit 保证同优先级先进先出
        self.sort_key = (-self.priority, self.submit, self.name)


def atomic_checkpoint(job: Job, root: Path, now: int) -> Path:
    """两阶段写入：先写临时目录，再校验并原子改名。"""
    tmp = root / f".{job.name}.tmp"
    final = root / f"{job.name}-step{job.step:04d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    payload = {
        "job": job.name, "step": job.step, "resume_count": job.resume_count,
        "data_cursor": job.step * 10, "membership_epoch": job.resume_count,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    (tmp / "state.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (tmp / "checksums.sha256").write_text(digest + "  state.json\n")
    (tmp / "manifest.done").write_text(json.dumps({"step": job.step, "sha256": digest}))
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    job.last_ckpt = job.step
    print(f"t={now:02d} CHECKPOINT job={job.name} step={job.step} hash={digest[:8]}")
    return final


def simulate():
    root = Path(tempfile.mkdtemp(prefix="ch10-ckpt-"))
    jobs = [
        Job(priority=10, submit=0, name="prod", need_cpu=2, total_steps=10,
            step_cost=1, checkpoint_every=2),
        Job(priority=5, submit=0, name="research", need_cpu=2, total_steps=8,
            step_cost=1, checkpoint_every=2),
    ]
    waiting = jobs[:]
    running: list[Job] = []
    free_cpu = 4
    preempt_requested = False
    print(f"checkpoint_root={root}")
    for now in range(30):
        # t=3 模拟更高优先级的抢占请求；先让低优先级作业优雅保存
        if now == 3 and not preempt_requested:
            preempt_requested = True
            for j in running:
                if j.name == "research":
                    j.preempt_at = now + 2
                    print(f"t={now:02d} PREEMPT_NOTICE job={j.name} deadline={j.preempt_at}")

        # 处理到期抢占，必须先有最近可验证 checkpoint
        for j in list(running):
            if j.preempt_at is not None and now >= j.preempt_at:
                if j.last_ckpt != j.step:
                    atomic_checkpoint(j, root, now)
                j.state = "SUSPENDED"
                j.resume_count += 1
                running.remove(j); free_cpu += j.need_cpu
                waiting.append(j)
                print(f"t={now:02d} SUSPEND job={j.name} step={j.step}")

        # 简单 best-fit：按优先级和提交次序准入
        waiting.sort(key=lambda x: x.sort_key)
        for j in list(waiting):
            if j.state in {"SUCCEEDED", "SUSPENDED"} and j.resume_count == 0:
                continue
            if j.need_cpu <= free_cpu and j.state != "SUCCEEDED":
                waiting.remove(j); running.append(j); free_cpu -= j.need_cpu
                old = j.state; j.state = "RUNNING"
                # 抢占后的恢复需要重新申请下一次通知窗口
                j.preempt_at = None
                print(f"t={now:02d} START job={j.name} from={old} step={j.step} cpu={j.need_cpu}")

        # 每个 tick 执行一个 step；抢占通知后不接收新的 batch，只完成安全点
        for j in list(running):
            j.step += 1
            if j.step % j.checkpoint_every == 0:
                atomic_checkpoint(j, root, now)
            if j.step >= j.total_steps:
                j.state = "SUCCEEDED"
                running.remove(j); free_cpu += j.need_cpu
                print(f"t={now:02d} SUCCEEDED job={j.name} step={j.step}")

        if not running and not waiting:
            break

    print("final:", [(j.name, j.state, j.step, j.last_ckpt, j.resume_count)
                      for j in jobs])
    print("manifests:", sorted(p.name for p in root.glob("*/manifest.done")))
    shutil.rmtree(root)

if __name__ == "__main__":
    simulate()
```

在 Python 3.10+ 上运行时，输出的时间点可能因实现细节略有不同；一个典型结果如下（哈希前缀是运行时计算值）：

```text
checkpoint_root=/tmp/ch10-ckpt-xxxx
 t=00 START job=prod from=QUEUED step=0 cpu=2
 t=00 START job=research from=QUEUED step=0 cpu=2
 t=01 CHECKPOINT job=prod step=2 hash=...
 t=01 CHECKPOINT job=research step=2 hash=...
 t=03 PREEMPT_NOTICE job=research deadline=5
 t=05 SUSPEND job=research step=5
 t=05 START job=research from=SUSPENDED step=5 cpu=2
 ...
 t=09 SUCCEEDED job=prod step=10
 t=09 SUCCEEDED job=research step=8
 final: [('prod', 'SUCCEEDED', 10, 10, 0), ('research', 'SUCCEEDED', 8, 8, 1)]
```


这个简化模拟有意暴露几个工程问题：`research` 被通知抢占后仍可能执行到下一个安全点；checkpoint 只在步边界写入；恢复时 `resume_count` 变成新的 membership epoch；`manifest.done` 缺失时不会将临时目录视为有效版本。要模拟硬抢占，可在 deadline 前删除临时目录并把 `last_ckpt` 回退到上一个提交版本，再验证重放 step 和数据游标是否符合声明。

### 10.10.1 模拟的可扩展实验

1. 把 `need_cpu` 改成向量（CPU、内存、NVMe），观察数量足够但形状不匹配的情况。
2. 在 `atomic_checkpoint` 中随机抛出异常，确认旧 manifest 仍可恢复，临时目录被清理。
3. 让 world size 从 4 变为 2，给每个 job 增加全局 batch 和学习率，记录迁移前后有效样本。
4. 增加远端上传延迟和有限带宽，计算 checkpoint 写入是否超过抢占窗口。
5. 将 `waiting.sort` 换成 fair-share 分数，比较长作业的 p95 队列等待和短作业平均周转。

## 10.11 失败案例：指标变好但训练其实坏了

某团队把 64 卡训练从静态作业改成弹性作业，希望在高峰期让出节点。上线后一周，仪表盘显示 GPU 利用率从 72% 提升到 88%，验证集准确率也略高。两周后他们发现同一代码和配置无法重现，且最终模型在长尾集上变差。

复盘时间线：

1. 作业从 64 卡缩到 48 卡时，launcher 更新了 world size，但没有更新数据分片的 `epoch_seed`。
2. checkpoint 只保存了模型和优化器，没有保存每个 rank 的 shard+offset；恢复后 16 个新 rank 从 shard 开头重新读取。
3. 为了缩短抢占窗口，checkpoint 在后台线程直接读取正在被 optimizer 写入的 bucket，偶发产生混合版本；校验和对混合文件仍然有效。
4. 实验追踪把恢复后的指标追加到旧 run，没有记录 `membership_epoch`，曲线看起来连续。
5. 调度器按“GPU busy”计算利用率，把数据重放和 checkpoint 上传期间的忙等待算作有效工作。

修复步骤不是把 timeout 调大，而是先恢复语义：

- 让每个 checkpoint 保存全局样本游标、数据集快照和 membership epoch；
- 在安全点冻结 optimizer 或采用 copy-on-write，禁止读取可变 bucket；
- 恢复时创建新 run_id，并通过 lineage 字段指向父 run；
- 用样本哈希和有效 tokens 重新计算吞吐；
- 在缩放前运行 100 step 的小型探针，比较 loss、梯度范数和重复样本率；
- 将“可接受重复比例”和“数值容差”写入发布门槛。

根因是把“进程重启成功”当成“训练语义保持”。弹性、checkpoint 和追踪必须一起升级；单独改 launcher 或单独增加监控都会留下盲区。

## 10.12 六个理解检查（含答案）

### 检查 1：为什么 gang scheduling 对同步训练重要？

**答案**：同步训练需要所有 rank 进入同一个通信域并按顺序调用 collective。只启动部分 rank 会占用节点却无法推进 step，还可能让已启动 rank 在 rendezvous 或 collective 中无限等待。弹性框架可以允许成员变化，但必须在明确定义的 membership epoch 和 checkpoint 边界切换，而不是让半套 rank 长期运行。

### 检查 2：为什么“总空闲 GPU 足够”仍可能无法放置作业？

**答案**：GPU 可能分散在不同节点、型号或显存桶，且 GPU-NIC、NVLink、NUMA 和网络拓扑不满足硬约束。调度器需要在资源容量之上检查集合可行性与拓扑；否则作业会在启动时失败或通信性能不可接受。

### 检查 3：checkpoint 只有模型参数时，恢复会缺什么？

**答案**：至少缺优化器动量/二阶矩、scheduler 和 loss scale、数据游标、随机数状态、world size/批量语义、代码与数据版本。恢复可能加载成功却重复或跳过样本、改变学习率轨迹、产生不同 dropout，最终指标漂移。应按任务要求声明允许的非确定性，并保存可验证的状态集合。

### 检查 4：为什么目录里有最高 step 的文件不代表它可恢复？

**答案**：并行上传可能只完成部分 shard，临时文件或校验未提交；对象存储列表也可能有延迟。只有包含完整 shard 哈希、版本、元数据并写入 `manifest.done` 的原子提交才是可见 checkpoint。恢复器应选择最近一个已验证 manifest，而不是猜目录名。

### 检查 5：world size 变化为什么会改变优化语义？

**答案**：全局 batch、梯度平均/求和、学习率缩放、梯度累积、scheduler 进度和数据分片都可能变化。若只更新 rank 数而不迁移这些状态，单个 step 表示的样本数和梯度统计会改变。弹性恢复必须记录旧/新 membership epoch，并定义 batch、学习率和样本游标的迁移规则。

### 检查 6：GPU 利用率很高，为什么有效吞吐仍可能下降？

**答案**：设备可能在等待数据、重放样本、上传 checkpoint 或反复恢复；这些时间会被采样器算作 busy，却没有增加可验证样本数。应结合输入 stall、抢占、重放比例、有效 tokens 和端到端墙钟计算有效吞吐，并查看 rank 尾部等待。

## 10.13 练习

1. **状态机**：为一个 16 卡训练作业画出提交、排队、启动失败、优雅抢占、硬失败和恢复的状态机；给每条边写事件、超时和幂等键。
2. **队列公平性**：用 4 个租户、两种作业时长和 16 个 GPU，分别模拟 FIFO、静态优先级和 fair-share，报告平均等待、p95 等待和饥饿作业。
3. **拓扑放置**：给出两节点各 8 GPU、每节点 2 NIC 的图，设计一个 8 卡和一个 16 卡作业的硬约束与软偏好，说明回填会如何影响承诺启动时间。
4. **局部性模型**：测量小文件数量、缓存命中率和对象存储带宽，使用 \(T_{io}\) 公式估算元数据延迟与带宽项的交叉点。
5. **抢占窗口**：在 CPU 模拟中加入随机 checkpoint 写入时间和 30 秒/120 秒抢占窗口，估算未完成 checkpoint 的概率与 RPO。
6. **checkpoint 迁移**：设计从 world size=8 到 4 的状态迁移函数，列出参数分片、optimizer state、数据游标和随机数如何合并。
7. **可复现矩阵**：固定代码和数据，改变线程数、CPU 指令集、通信后端和随机种子；区分 bitwise、数值、轨迹和语义四种复现层级。
8. **失败注入**：在 checkpoint 上传中途杀死一个 worker、删除一个 shard、篡改 manifest；验证恢复器拒绝损坏版本并回退到上一个 done。
9. **利用率解释**：构造 GPU busy=90% 但有效吞吐下降的时间线，分别加入数据 stall、重放和 checkpoint 上传，计算修正后的指标。
10. **发布回滚**：为新 launcher 和 checkpoint schema 写 canary 门槛、双写周期、回滚命令和审计证据；证明旧模板能读取至少两个新版本 checkpoint。

## 10.14 安全边界与运维清单

### 10.14.1 租户隔离与最小权限

调度器、节点代理、对象存储和实验追踪系统应使用最小权限身份。训练容器不应拥有修改其他租户队列、读取全局缓存或删除共享 checkpoint 的权限。共享文件系统按项目目录和 ACL 隔离；日志中不写入云访问令牌、用户样本和完整环境变量。

### 10.14.2 数据与 checkpoint 机密性

模型权重、训练数据和优化器状态可能是敏感资产。传输和静态存储使用组织批准的加密与密钥轮换；manifest 中只记录对象引用和哈希，不把密钥放在 checkpoint。需要供应商排障时先用合成数据、脱敏拓扑和最小日志复现。

### 10.14.3 抢占和删除的安全阈值

自动删除旧 checkpoint、释放节点和清理缓存都是有破坏性的动作。默认先保留最近两个已验证版本，并在删除前确认新 manifest 已提交且恢复探针通过。调度器收到“强制清理”信号时要有截止期、操作者、作业和审计记录，不能由任意容器直接发出。

### 10.14.4 供应链与镜像

锁定容器 digest、依赖 lockfile、launcher 和 checkpoint 工具版本。不要从未验证的公共镜像加载特权节点代理或通信插件。升级驱动、框架和调度器时在隔离队列做语义探针、性能基线和故障恢复，确认回滚路径后再扩大范围。

### 10.14.5 资源拒绝服务

限制单作业 GPU/CPU、并发数、文件描述符、对象存储请求率和 checkpoint 大小。对用户可控的 batch、shape、数据路径和重试次数设上限，避免意外或恶意输入耗尽共享资源。监控单租户流量和本地盘增长，达到阈值先暂停新作业而不是让系统整体失稳。

## 10.15 版本边界与迁移注意

1. Kubernetes Volcano、Kueue、Slurm、YARN 和云厂商批处理器的 gang、抢占和队列语义不同；以目标版本 CRD/配置和事件日志为准，不把一个调度器的状态名直接映射到另一个。
2. PyTorch Elastic、torchelastic、torchrun 和各版本 rendezvous 后端对成员变化、超时与环境变量的要求会变化。升级前运行 world size 变化和 rank 失败探针。
3. 对象存储的列举、重命名和条件写入一致性取决于服务和客户端。优先使用不可变对象名、内容哈希、条件写和显式 manifest，不依赖目录 rename 的原子性假设。
4. 文件系统的 fsync、O_DIRECT、网络缓存和容器挂载可能改变 checkpoint 持久化边界。若需要崩溃一致性，在目标介质上做断电或进程杀死测试。
5. 混合精度、TF32、确定性 kernel、CPU 指令集和线程调度会影响数值复现。升级驱动或编译器后重新测量容差，而不是要求不现实的 bitwise 相等。
6. 训练追踪系统的 step、run、artifact 和 lineage API 可能在版本中改变。导出一份与平台无关的 manifest，确保迁移或离线审计时仍可解释。
7. 调度器和 launcher 的超时单位、信号转发、退出码映射必须在真实节点验证。把 SIGTERM 当成普通失败会错过 checkpoint 窗口；把所有非零退出都自动重试会掩盖数据损坏。

## 10.16 来源地图

以下来源优先选择官方文档、规范和维护者指南，用于核对 API、调度和恢复边界；不同版本应使用对应文档。

- [Kubernetes Scheduling 文档](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/)：调度框架、过滤/打分、节点资源和扩展点。
- [Kubernetes Pod 生命周期](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)：状态、终止信号、重启策略和容器退出语义。
- [Kubernetes Job 文档](https://kubernetes.io/docs/concepts/workloads/controllers/job/)：批处理 Job、重试、完成和失败策略。
- [Kueue 文档](https://kueue.sigs.k8s.io/)：队列、配额、准入、借用和抢占的 Kubernetes 实现。
- [Volcano 文档](https://volcano.sh/en/docs/)：gang scheduling、队列与批量作业插件。
- [Slurm 作业调度指南](https://slurm.schedmd.com/overview.html)：分区、优先级、抢占、资源分配和作业步骤。
- [Slurm checkpoint/restart](https://slurm.schedmd.com/checkpoint_blcr.html)：检查点接口和调度器集成边界（具体实现依版本）。
- [PyTorch Elastic 文档](https://pytorch.org/docs/stable/elastic/run.html)：torchrun、rendezvous、成员变化和环境变量。
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)：分布式 state dict、保存/加载、规划器与存储抽象。
- [PyTorch 随机性与确定性](https://pytorch.org/docs/stable/notes/randomness.html)：随机种子、确定性算法和跨版本限制。
- [MLflow Tracking 文档](https://mlflow.org/docs/latest/ml/tracking/)：run、参数、指标、artifact 与 lineage 记录。
- [Weights & Biases Experiment Tracking](https://docs.wandb.ai/guides/track)：配置、指标、artifact、run resume 与权限边界。
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/trainer/overview/)：分布式训练 CRD、作业生命周期和 Kubernetes 集成。
- [OpenTelemetry Metrics](https://opentelemetry.io/docs/concepts/signals/metrics/)：指标、聚合、标签和观测边界。
- [POSIX `fsync` 说明](https://man7.org/linux/man-pages/man2/fsync.2.html)：文件与目录持久化语义；具体存储系统仍需实测。
- [Daly, “A higher order estimate of the optimum checkpoint interval”](https://dl.acm.org/doi/10.1145/261649.261654)：检查点间隔成本模型的经典参考。

[方法说明] 本章的状态机、公式和 CPU 模拟是教学模型，不承诺任何特定调度器、对象存储或训练框架的性能。请在目标硬件、版本、拓扑和租户策略上复现测量，并记录未验证假设。

## 10.17 章节完成标准

读者完成本章后，应能对一次训练作业回答以下问题：

1. 作业当前处于哪个状态，进入/退出证据和超时是什么？
2. 请求的 CPU、内存、GPU 显存、NIC、NVMe 和拓扑硬约束是什么？
3. 队列策略如何处理优先级、公平性、配额、饥饿和回填？
4. 数据从哪一层读取，缓存命中和 rank 尾部等待是多少？
5. 抢占是优雅、短窗口还是硬回收，RPO/RTO 和退出码如何定义？
6. checkpoint 保存了哪些模型、优化器、数据、随机数、代码和环境状态？
7. manifest 如何原子提交，恢复器如何证明它完整且与当前 schema 兼容？
8. world size 变化如何迁移全局 batch、学习率、数据游标和 membership epoch？
9. run_id、配置、数据快照、镜像、指标和 lineage 是否足以重建实验？
10. 利用率是否扣除了输入 stall、重放和恢复，发布是否有 canary 和回滚证据？

如果其中任何一项答不出来，先在 CPU 两作业模拟中写出状态转移和日志，再到单节点目标硬件验证 checkpoint 和恢复，最后才扩大到多节点。不要用无限重试、无限 timeout 或删除旧日志来掩盖未定义的语义。

## 10.18 事故时间线与定量诊断手册

训练运维事故最容易犯的错误是只看最后一条错误消息。调度、节点、数据、训练进程和对象存储各自都有时钟，必须先把事件归并到一条单调时间线，再讨论责任归属。每个作业至少记录提交时间、进入队列时间、准入时间、节点分配时间、容器启动时间、首个 batch、稳态开始、最近一次成功 checkpoint、收到终止信号、进程退出和资源释放时间。对每个时间点附上作业状态、控制器观察到的原因、用户进程的退出码和节点健康摘要。这样可以区分“排队很久但运行正常”“运行很快却频繁恢复”以及“调度器以为完成、训练进程仍在写文件”等完全不同的问题。

队列等待过长时，先拆分为资源不足、配额不足、准入策略等待和拓扑约束不满足四类。资源不足看同一时段的可用 GPU、CPU、内存和 NIC；配额不足看租户或项目的已用量与借用上限；准入等待看优先级、gang 条件、借用和抢占事件；拓扑约束不满足则检查“总空闲 GPU 足够但无法组成同一节点或同一机架”的碎片化。不能用平均等待时间掩盖 p95 和最老作业年龄，后者更能暴露饥饿。若调度器声称“无资源”，应同时给出过滤器拒绝计数，例如显存、节点标签、拓扑、污点和卷挂载各自拒绝了多少候选节点。

作业启动后吞吐低，先判断是输入尾部、同步尾部还是计算退化。输入尾部可由每个 rank 的 batch 读取完成时间和缓存命中率确认；同步尾部可由 barrier 或 collective 的进入时间差确认；计算退化则看同一算子在不同节点上的持续时间和频率限制。GPU busy 很高并不等于有效吞吐高：重放、无效 token、等待 CPU 发起的短 kernel 和 checkpoint 上传都可能让设备处于忙碌状态。建议同时报告每秒有效样本、每秒处理 token、重放样本、丢弃样本、输入等待占比和恢复占比，并把这些指标按 step 关联，而不是用整个作业的单一平均值。

抢占事故应从终止信号开始倒推。优雅抢占通常先发出可捕获信号，留出一个窗口完成梯度落盘和 checkpoint 提交；硬回收可能直接杀死容器，任何未提交目录都不能被当作可恢复版本。记录信号发送者、信号到达进程的延迟、训练器是否收到、最后成功写入的 shard、manifest 是否已提交以及节点何时被重新分配。若同一个作业连续因“抢占”重试，但每次都在 checkpoint 上传阶段失败，根因可能是存储带宽或权限，而不是队列策略。重试计数和恢复步数要分开，否则会把数据重放成本误算成训练进度。

checkpoint 损坏诊断应遵循从目录到内容的顺序。先检查提交标志和 manifest 版本，再检查 shard 数量、大小、内容哈希和写入时间，最后在独立进程中加载参数、优化器、数据游标和 RNG。目录中存在更大的 step 不代表它更可信，未提交或哈希不符的版本应立即隔离。恢复器遇到缺 shard 时要返回结构化错误，指出缺失对象、期望长度和可回退的最近提交点；不要自动忽略缺键继续训练。若只能恢复模型参数，应把作业标记为重新开始或近似继续，避免下游把结果与精确续训混在同一实验中。

## 10.19 调度策略的回放实验与容量推理

调度策略的优劣不能仅凭少量成功作业判断，应使用脱敏的历史轨迹或合成事件做回放。每条轨迹包含作业提交时间、资源向量、持续时间、优先级、截止期、拓扑要求、抢占容忍度和实际完成状态。回放器先复现基线策略，再只替换一个决策，例如优先级 aging、最短作业优先、队列借用或拓扑打分。比较时同时报告平均和 p95 排队时延、吞吐量、公平性、抢占次数、GPU 碎片率、队列饥饿时长和失败重试成本。若只提高总体 GPU 利用率却让小作业 p95 无限增长，策略不能直接上线。

容量规划需要把资源向量视为多维，而非只算 GPU 数。一个训练作业可能受显存、主机内存、NVMe、NIC 带宽、对象存储请求率和文件描述符共同限制。对每个资源记录峰值、稳态平均和恢复阶段峰值；checkpoint 上传往往使网络和本地盘在短时间内高于训练稳态。利用率应采用“按可调度容量加权”的定义，并保留被拓扑、配额和维护窗口排除的资源。若节点长期 GPU 空闲但 NIC 饱和，继续增加 GPU 只会扩大排队；若 GPU 满载但数据缓存命中低，则应优先扩展缓存或预取带宽。

碎片化是容量推理中经常被忽略的损耗。设集群有许多小块空闲 GPU，总数足够却不能放置一个要求同节点八卡的作业，此时可用容量与可调度容量不同。记录每次调度尝试的候选节点集合、过滤原因和剩余碎片，并计算“可放置作业占理论空闲作业”的比例。回填策略可以提高短期利用率，但要模拟长作业被延迟、抢占和 checkpoint 开销后的真实完成时间。对有截止期的作业，应该使用最坏情况下的可用窗口做容量承诺，而不是使用历史平均空闲量。

公平性指标也要和业务语义一致。简单的队列份额只反映提交量，不能说明作业等待是否与消耗成比例。可以为每个租户计算归一化 GPU 小时、等待时间、获得的拓扑质量和被抢占次数，再比较其与配额的偏差。对交互式小作业，延迟比吞吐重要；对超大训练，稳定的连续窗口比瞬时公平重要。策略评估应把这些目标写成可审查的门槛，例如 p95 等待不能超过某值、任何租户不得连续多个窗口没有准入、抢占必须保持最小恢复点目标。门槛未定义时，调度器的“优化”无法被验证。

## 10.20 可复现实验的证据链与迁移检查

一次实验可复现的最低条件不是保存一个配置文件，而是保存从数据到结果的证据链。manifest 应包含代码提交标识、容器镜像摘要、依赖锁定文件、训练配置、数据快照或不可变引用、词表与预处理版本、并行网格、world size、硬件拓扑、随机数种子、环境变量白名单、起始 checkpoint、每个阶段的 step 范围和指标聚合方式。对于可能泄露秘密的环境变量，只记录变量名和经过脱敏的摘要；凭证本身绝不写入日志或 artifact。每个 artifact 都要有长度和哈希，避免同名对象被覆盖后无法追溯。

指标的时间语义必须固定。训练 loss 是按样本、按 token 还是按 optimizer step 聚合，吞吐是否扣除输入等待和恢复时间，GPU 利用率采样窗口多长，p95 是按 step 还是按作业，均应在 manifest 中声明。迁移到另一套追踪系统时，先导出平台无关的事件和指标，再做映射；不能只复制一张最终曲线。对于恢复后的 run，应保留原始 run 标识、恢复次数、恢复点、重复样本数量和 membership epoch，并把续训曲线与原始曲线关联而不是覆盖。

确定性需要区分可重复和可接受的非确定性。固定种子不能消除异步通信、并行归约顺序、GPU kernel 和数据加载线程造成的微小差异。实验报告应给出允许的 loss、梯度范数和评估指标容差，并说明跨硬件或跨版本时哪些算子会改变结果。若要求逐位一致，应锁定硬件、驱动、编译器、线程数和确定性算法开关，并用小数据集逐步比对；逐位一致失败不等于实验不可复现，但必须证明差异在预设容差内且不改变结论。

迁移前先做影子恢复。把生产 manifest 和最近一个已验证 checkpoint 复制到隔离队列，使用新 launcher、调度器或追踪版本加载，执行少量 forward、一个 optimizer step 和一次再次保存。比较模型、优化器、数据游标、RNG、step 和指标的结构化摘要，确认新版本可以回到旧版本生成的状态。影子恢复成功后再做小规模 canary，限制资源、运行时长和写入范围；只有 canary 达到预设门槛，才允许扩大并发。任何无法解释的字段缺失或版本降级都应阻止自动迁移。

发布回滚必须有可执行的证据，而不是文档中的口号。保留旧版 launcher、镜像、checkpoint schema 和调度模板，在双写期间让新旧读取器都验证同一份 manifest。回滚条件包括错误率、恢复失败、吞吐回退、队列 p95 恶化和安全审计告警，并定义谁有权触发以及触发后如何冻结新提交。回滚时不要删除新版本 artifact，先把失败样本和日志标记为隔离，以便复盘。完成迁移后仍保留一段观察窗口，确认没有延迟出现的资源泄漏、磁盘增长或追踪数据丢失。

将证据链和事故手册结合起来，团队才能回答“这次结果为何可信”。可信不等于没有失败，而是失败有明确边界、恢复点可验证、指标含义可追溯、策略变化可回放。任何只保存最终模型而没有数据版本、优化器状态或调度事件的实验，都不具备足够证据支持生产决策。


## 10.21 小结

大规模训练运维的对象不是一段脚本，而是一条可观测、可恢复、可复现的流水线。调度器决定资源何时以及以何种拓扑交给作业；数据局部性决定这些资源是否真正忙于有效样本；抢占和弹性决定 world size、梯度和数据游标怎样变化；checkpoint 和 manifest 决定失败后能否回到一个可信状态；实验追踪与发布系统则决定结果能否解释、比较和回滚。

实用的工程顺序是：先写状态机和资源合同，再测队列/放置/数据局部性的基线；定义 checkpoint 的完整状态和原子提交；用 CPU 模拟注入抢占、部分上传和恢复；在目标硬件上做小规模语义探针；最后用有效吞吐、RPO、RTO、p95 等指标推动容量和发布决策。这样，当作业被抢占、节点消失或版本升级时，团队知道丢失了什么、恢复到哪里以及下一步如何安全继续，而不是只看到一条“作业失败”消息。
