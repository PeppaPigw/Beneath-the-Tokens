---
id: ch11-data-systems
title: AI 数据系统：对象存储、文件系统、湖仓、格式与数据质量
slug: /chapters/11-data-systems
description: 从对象存储和本地文件系统，到 SSD、RocksDB、LSM、Parquet、Arrow、湖仓、流批处理与数据质量，建立可复现、可审计、可恢复的数据系统
sidebar_position: 11
level: systems
prerequisites:
  - ch02-linux-process-files-observability
  - ch04-performance-math
  - ch06-pytorch-execution
  - ch08-distributed-collectives
  - ch10-training-ops
learning_objectives:
  - 能比较对象存储、分布式文件系统、本地文件系统和 SSD 的一致性、延迟、吞吐及故障边界
  - 能解释 LSM-tree、RocksDB 的 WAL、memtable、SSTable、compaction 和 tombstone，并据此估算写放大
  - 能区分 Parquet 的列式持久化与 Arrow 的内存列式布局，设计零拷贝或低拷贝的数据路径
  - 能用 manifest、不可变对象、条件写和校验和实现原子发布与可恢复读取
  - 能设计数据分片、重试、缓存、去重和 lineage，使流批管道在扩缩容后仍有明确语义
  - 能为 schema、缺失值、重复样本、时间戳、PII 和删除请求建立数据质量与隐私门禁
  - 能用 watermark、checkpoint、幂等 sink 和迟到数据策略统一流处理与批处理结果
  - 能在 CPU-only 环境运行分片实验，验证确定性、覆盖率、无重复和 manifest 完整性
  - 能诊断小文件、热点分区、compaction、缓存抖动和对象存储列举延迟等常见故障
  - 能为数据集、特征、训练样本和模型建立可追踪的血缘、版本、质量证据和安全边界
estimated_hours: 24
hardware: CPU-only baseline; local SSD/object store/distributed filesystem optional
risk_level: L2
last_verified: 2026-10-05
---

# 第11章　AI 数据系统：对象存储、文件系统、湖仓、格式与数据质量

> 模型只会看到数据系统交给它的字节。字节来自哪里、按什么顺序排列、是否重复、是否被截断、何时可见、谁可以读取，都会改变训练结果，却不一定在 loss 曲线上留下明显的痕迹。本章把数据当成一条有状态的系统：原始对象被摄取、分片、编码、缓存、查询、去重、审计，最后以一个可验证的快照交给训练或在线服务。我们先建立存储层和数据格式的心智模型，再讨论 manifest、湖仓、流批一致性、质量门禁与隐私，最后用 CPU 实验把分片和可复现性落到可运行的代码。

“数据管道成功”至少要回答四个问题：第一，读到的到底是哪个版本；第二，所有消费者看到的集合是否一致；第三，数据是否满足质量和隐私约束；第四，失败或重试后能否得到同样的结果。只要其中一个问题没有证据，所谓的数据集版本就只是一个目录名。

## 11.1 数据系统是模型语义的一部分

### 11.1.1 从字节到样本的多层契约

一次训练读取通常经过下面的链路：

```text
采集事件 -> 原始对象 -> 解码/规范化 -> 分区文件 -> manifest
       -> 分片器 -> 缓存 -> 迭代器 -> batch -> 梯度/指标
```

每一层都有不同的不变量：原始对象要能校验完整性；规范化要固定字符集、时区和数值单位；分区文件要声明 schema 和统计信息；manifest 要定义快照边界；分片器要保证覆盖率和去重；缓存要验证内容哈希；迭代器要报告游标；训练循环要把游标和随机状态写进 checkpoint。任何一层只凭“文件存在”判断成功，都会把局部成功伪装成端到端成功。

可以把训练数据快照抽象成：

\[
D_v=(O_v,S_v,M_v,Q_v,P_v),
\]

其中 `O_v` 是对象集合及内容哈希，`S_v` 是 schema 与编码，`M_v` 是 manifest 和分片规则，`Q_v` 是质量报告，`P_v` 是隐私与访问策略。模型的 lineage 应引用 `D_v` 的不可变标识，而不是可变的 `latest` 路径。若只记录路径，后来覆盖同名对象后无法解释旧模型的来源。

### 11.1.2 新鲜度、完整性与可重复性不是同一件事

- **新鲜度（freshness）**：数据距现在有多近，例如事件延迟 p95 为 5 分钟。
- **完整性（completeness）**：应到达的分区或事件是否都到达，是否存在空洞。
- **一致性（consistency）**：不同消费者是否看到同一快照，是否有跨表事务边界。
- **可重复性（reproducibility）**：在相同输入和代码下，能否重建相同样本集合与顺序。

实时推荐可能优先新鲜度，财务报表优先完整性与可审计性，预训练数据优先去重和覆盖率。把一个指标当作全部目标，会导致错误优化，例如为了“今天数据不落后”而放宽迟到事件处理，最终标签泄漏到特征中。

### 11.1.3 快照而非目录

目录是命名空间，不是版本控制。一个稳健的快照至少包含：

```json
{
  "dataset_id": "reviews",
  "snapshot_id": "2026-10-05T00:00Z-7f2c",
  "schema_id": "reviews.v4",
  "objects": [
    {"uri": "s3://bucket/reviews/date=2026-10-04/part-000.parquet",
     "bytes": 18327492, "sha256": "...", "rows": 98231}
  ],
  "partition_spec": "date/hour",
  "created_by": "normalize@commit:abc123",
  "quality_report": "sha256:...",
  "privacy_policy": "policy:v3",
  "parent_snapshot": "2026-10-04T00:00Z-a1b8"
}
```

`objects` 列表是冻结的；消费者只读列出的 URI，不扫描目录猜测新增文件。创建新快照是追加动作，修复数据也产生新版本并保留 parent。这样可以用内容哈希和 lineage 解释差异，而不依赖对象存储的列表顺序或目录重命名语义。

## 11.2 存储层：对象、文件与 SSD 的边界

### 11.2.1 对象存储：高耐久与不同的命名空间语义

对象存储把数据放在 `(bucket, key, version)` 或类似三元组中，通常提供高耐久、水平扩展、按字节计费和跨区域复制。它擅长大对象的顺序读写、归档和跨节点共享，但小对象和高频随机更新代价高。常见限制包括：

1. **命名空间要看产品类型**：传统平面对象存储中，`a/b/c` 只是 key 前缀；支持分层命名空间或目录 bucket 的产品可能提供不同能力。无论哪种，列举接口都不能自动代替应用的多对象事务。
2. **重命名可能是复制加删除**：大对象重命名会放大流量和失败窗口，不能假设 POSIX `rename` 的原子性。
3. **一致性以服务契约为准**：Amazon S3 与 Google Cloud Storage 的常规对象读写和列表已有强一致性保证，不能沿用“对象存储一概最终一致”的旧结论。跨区域异步复制、配置传播和多对象事务是另一组边界；强一致的单操作不意味着多次分页读取构成固定快照。
4. **请求粒度很重要**：把 1 亿条 1 KB 样本存成小对象会让每次 GET 的 TLS、鉴权和延迟成为瓶颈。
5. **删除通常是逻辑操作**：版本化、保留策略和生命周期规则可能使“删除”只增加 tombstone，仍产生存储和合规责任。

S3 的单 key 更新具有原子性，但不提供任意多个 key 的原子更新；这正是应用仍需要 manifest 的原因。[Amazon S3 一致性模型](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel)、[Cloud Storage 一致性](https://cloud.google.com/storage/docs/consistency)

对象存储的实用写模式是不可变大对象加 manifest。上传先写带随机后缀的临时 key，完成后写 checksum 和元数据，再以条件写提交 manifest。读取端只跟随 manifest；不要在训练时循环列举前缀。

### 11.2.2 分布式文件系统：命名空间与并发元数据

HDFS、CephFS、Lustre、BeeGFS、NFS 等文件系统提供 POSIX 或接近 POSIX 的路径、目录和文件句柄。它们适合需要大量并发顺序读、共享 scratch 或高频小文件的场景，但元数据服务、锁、网络和故障恢复都可能成为瓶颈。

文件系统的“写入完成”要区分三层：应用写入页缓存、文件系统提交日志、底层介质持久化。`fsync(fd)` 只对当前文件提出持久化请求，目录项还可能需要 `fsync(dirfd)`；网络文件系统和容器卷还要查明服务器端语义。训练数据不是金融账本时可以接受较弱持久化，但 checkpoint、manifest 和审计日志必须明确 RPO。

小文件问题具有两个维度：存储占用浪费和元数据压力。即使总字节数不大，百万个文件也会让 `stat`、目录列举和 open 系统调用耗尽 CPU。合并成 Parquet row group 或 tar shard 可降低元数据请求，但要保留内部索引，否则随机读取会退化成整 shard 扫描。

### 11.2.3 本地 SSD/NVMe：低延迟但易丢失

本地 SSD 提供微秒级到毫秒级访问和高 IOPS，适合作为解压 scratch、热缓存、RocksDB WAL/SSTable 或预取缓冲。代价是容量有限、节点释放时可能丢失、磨损和温度会影响尾延迟。需要记录：

- 设备型号、剩余寿命和写放大；
- 文件系统挂载选项、块大小和队列深度；
- 加密与节点回收时的清理策略；
- 缓存对象的内容哈希和失效版本；
- 达到高水位时的回收顺序与保护租户。

把本地盘当作真源会让作业在重调度后无数据可读。正确做法是远端不可变对象为真源，本地 SSD 只保存可重建的缓存或明确复制的中间结果。若本地盘保存唯一 WAL，必须先把 WAL 复制到有持久性的介质，再确认提交。

### 11.2.4 延迟、带宽、并发与成本模型

一次读取的墙钟时间不只是字节除以带宽：

\[
T = T_{setup} + N_{req}L + \frac{B}{BW_{eff}} + T_{queue} + T_{retry}.
\]

`T_setup` 包含连接和鉴权；`N_req L` 对小文件尤其关键；`BW_eff` 受并发、压缩和 CPU 解码限制；`T_queue` 来自网卡、磁盘或元数据服务排队；`T_retry` 则是超时和重试的尾部成本。成本模型还应加入 GET/PUT 请求费用、跨区流量、临时 SSD、压缩 CPU 和数据保留年限。通过合并小对象减少请求数，可能以增加并行解码和单对象重试代价为代价，必须在目标硬件实测。

## 11.3 SSD 上的 RocksDB 与 LSM-tree

### 11.3.1 为什么训练管道会需要 KV 存储

数据管道常要维护去重指纹、URL 到内容哈希的映射、增量游标、特征物化状态和在线缓存。这些访问通常是键值读写，不适合每次扫描 Parquet。RocksDB 是嵌入式 KV 引擎，核心结构是 LSM-tree（Log-Structured Merge-tree）：写入先追加 WAL 并进入内存表，内存表满后冻结为 immutable memtable，刷成磁盘上的 SSTable；后台 compaction 合并重叠文件并丢弃已删除版本。

```text
put(k,v)
  -> WAL (顺序写，崩溃恢复)
  -> memtable (有序内存结构)
  -> flush
  -> L0 SSTable (可能重叠)
  -> compaction
  -> L1/L2/... SSTable (分层、较少重叠)
```

LSM 用顺序写换取随机写性能，但读路径可能检查多个层级。Bloom filter、block cache 和索引减少无效磁盘访问；热点 key 仍可能把 cache 打满。RocksDB 适合单节点本地状态或分片后的独立实例，不等于一个自动分布式数据库。跨节点复制、主从、租户隔离和备份要由上层实现。

### 11.3.2 WAL、flush、checkpoint 的三种“已写入”

- **进入 memtable**：进程内可读，断电会丢失，除非同步 WAL。
- **WAL `sync` 成功**：可在进程重启时恢复，介质和文件系统仍需满足持久化契约。
- **SSTable/快照已上传远端**：节点丢失后可恢复，但上传期间仍可能有缺口。

数据管道若需要 exactly-once 的效果，不能只调用 `put`；要把处理输入的游标、输出对象和 KV 状态放进同一个可验证提交协议，或设计幂等键使重复处理不改变最终状态。

### 11.3.3 Compaction 与写放大

LSM 的写放大近似为：

\[
WA = \frac{\text{写入介质的总字节}}{\text{用户逻辑写入字节}}.
\]

压缩、重写和索引会让 `WA>1`；层级 compaction、leveled/size-tiered 策略和 tombstone 数量决定实际值。一个去重服务持续写入新版本、但很少删除旧值时，SSTable 会膨胀；频繁 TTL 删除会产生 tombstone，直到 compaction 才释放空间。应监控 pending compaction bytes、flush stall、L0 文件数、读放大和 block cache 命中率。

参数调整要结合设备：增大 memtable 可减少 flush，但占用 RAM；提高并发 compaction 可缩短积压，却与训练读取争抢 SSD；压缩节省字节但增加 CPU。不要在训练高峰期盲目执行全库 compaction，先在副本上测量 P99 写延迟和读尾延迟。

### 11.3.4 RocksDB 与对象/列式存储的分工

一个常见的分层设计是：

- Parquet/对象存储：长期、批量、可扫描的真源数据；
- Arrow：进程内批量传递和向量化计算；
- RocksDB：小型索引、去重状态、增量游标和本地缓存；
- manifest：跨对象和状态的发布边界；
- 远端备份：恢复和审计。

不要把整个训练语料塞进 RocksDB 只因为随机读取方便。这样会失去列式压缩和大规模顺序扫描优势，也让 compaction 成为数据平台的单点瓶颈。相反，也不要把每次去重查询都实现成扫描 Parquet；KV 索引和列式真源应各司其职。

## 11.4 Parquet、Arrow 与湖仓格式

### 11.4.1 Parquet 的列式物理布局

Parquet 文件由多个 row group 组成，每个 row group 按列存储 page，并带有 schema、统计信息和编码。查询只读所需列和满足过滤条件的 row group，可获得列裁剪和谓词下推收益。压缩效果依赖值的局部相似性；把随机字符串、巨型嵌套数组和高基数列放在同一 row group，往往压缩差且解码成本高。

设计 Parquet 时要显式选择：

1. **row group 大小**：太小导致元数据和请求放大，太大导致并行度低、过滤后仍读很多字节。先按目标扫描并发和对象大小做基准。
2. **排序/分区键**：按时间或租户排序可让 min/max 统计过滤，但热点键会造成单分区倾斜。
3. **编码与压缩**：字典编码适合低/中基数，RLE/bit-pack 适合重复整数；ZSTD、Snappy、GZIP 在压缩率、CPU 与兼容性上取舍不同。
4. **嵌套类型**：LIST/STRUCT/MAP 需统一逻辑 schema，避免不同生产者把空数组、null 和缺失字段编码成不同语义。
5. **时间与精度**：明确 UTC、单位（秒/毫秒/微秒）和溢出策略；不同语言默认类型可能不一致。

Parquet 的统计信息是优化提示，不是质量证明。若生产者写错了 min/max，查询可能漏数据；发布前应抽样验证统计和真实值范围。

### 11.4.2 Arrow 的内存布局与零拷贝边界

Apache Arrow 定义列式内存格式，数组通常由 validity bitmap、offset buffer、values buffer 组成。相同进程或兼容语言之间可以直接共享这些 buffer，减少序列化和拷贝。Arrow IPC/Flight 可在进程间传输批量列数据；但“零拷贝”不是魔法：跨设备、压缩、类型转换、非连续切片和 Python 对象列都会触发复制。

在训练输入管道中，推荐把解码、过滤和拼 batch 尽量保持 Arrow/NumPy 连续缓冲，再在最后一步转换为框架张量。记录每次转换的字节数与耗时；若发现 CPU 主要花在 Python 对象分配，说明没有获得列式路径的收益。生命周期也很重要：张量引用了 Arrow buffer 时，不能提前释放或复用底层内存。

### 11.4.3 湖、湖仓与表格式

“数据湖”通常指以对象存储为底座、用开放文件格式保存原始和派生数据的架构；“湖仓”在湖上加上表级元数据、事务、schema 演进、时间旅行、优化和权限。Delta Lake、Apache Iceberg、Apache Hudi 等表格式都通过 metadata/manifest 文件记录数据文件集合，但协议、并发提交、删除向量和分区演进不同。

表格式解决的是表快照和并发写入问题，不会自动解决原始数据质量、PII 清理或错误业务逻辑。读取某个表版本时仍要验证：

- metadata 版本与客户端兼容；
- 快照引用的文件均存在且哈希/大小匹配；
- schema 演进没有把 decimal、timestamp 或 nullability 悄悄改变；
- 删除和重写是否满足保留、审计和用户删除请求；
- compaction、vacuum、rewrite 是否会破坏旧模型的 lineage。

把“表可读”当成“数据正确”是常见的反模式。表格式提供提交边界，质量系统提供内容证据，两者应在发布门禁中同时出现。

## 11.5 Manifest：把不可见的集合变成可验证的快照

### 11.5.1 Manifest 的最小字段

一个适合训练和批处理的 manifest 至少记录：

```yaml
manifest_version: 3
snapshot_id: reviews-20261005-7f2c
created_at: 2026-10-05T00:00:00Z
producer:
  pipeline: normalize_reviews
  commit: abc123
  image_digest: sha256:...
schema:
  id: reviews.v4
  fingerprint: sha256:...
partition_spec: [event_date, shard]
files:
  - uri: s3://bucket/reviews/date=2026-10-04/part-000.parquet
    rows: 98231
    bytes: 18327492
    sha256: ...
    min_event_ts: 2026-10-04T00:00:01Z
    max_event_ts: 2026-10-04T23:59:59Z
quality:
  null_rate_text: 0.0003
  duplicate_rate: 0.0011
  pii_scan: pass
privacy:
  classification: internal
  retention_until: 2027-10-05
lineage:
  parents: [raw-20261004-a1b8]
  code: git:abc123
```

字段可按场景扩展，但要避免把秘密写进 manifest。访问令牌、加密密钥和原始 PII 不应出现在路径、标签或日志中。`schema.fingerprint` 应由规范化的字段名、类型、nullability 和元数据计算，避免 JSON 字段顺序导致虚假变化。

### 11.5.2 两阶段发布

推荐的发布协议：

1. 生成不可变数据文件到临时前缀；每个文件在上传完成后记录长度和 SHA-256。
2. 运行行数、schema、范围、重复率、PII 和可读性检查，写 `quality.json`。
3. 写 `manifest.intent`，包含文件清单、质量报告和预期父快照。
4. 独立验证器读取所有文件和哈希，检查权限和配额。
5. 以条件写或对象版本写入 `manifest.done`，内容包含 `intent` 的哈希。
6. 更新指向最新快照的轻量指针（如 `CURRENT`），更新失败时不影响旧版本。
7. 清理未提交临时对象，清理必须可重试且不删除其他快照。

消费者只接受 `manifest.done`，并验证其内部文件数量、哈希和 schema。`CURRENT` 丢失时可以通过受信任的索引恢复，但不能扫描“看起来最新”的目录名。并发发布要带 parent 版本和条件写，检测冲突后重试合并或失败，而不是静默覆盖。

### 11.5.3 分页读取与部分发布

即使每次 LIST 都强一致，连续多次分页调用也不一定共享同一个事务时间点；分页期间仍有人新增或删除 key，得到的集合就不能代表一个事先定义的业务批次。更直接的问题是：某个对象的 PUT 已成功，不代表这一批其余对象都成功。基于列表计数推断“整批上传完成”是不安全的；应由上传者维护完整清单，由验证器逐项 HEAD/GET 检查。独立复制目的地还要遵守复制服务的进度契约。网络重试可能产生未提交的 multipart 部件或孤立对象，需要生命周期规则和审计指标。单对象原子性与多对象完整性必须分别检查。

## 11.6 分片、分区与样本顺序

### 11.6.1 分区键与分片键的差异

**分区（partition）**决定物理目录或文件集合，通常用于时间、租户、地域等查询过滤；**分片（shard）**决定并发消费者如何分工。两者可以相同，也可以不同：按日期分区、按哈希用户 ID 分片能兼顾时间裁剪和负载均衡。

分区设计要避免两个极端：

- 分区太粗：每次查询扫描大量无关数据，单个分区成为热点。
- 分区太细：产生海量小文件和元数据，提交与列举成本上升。

分片数量应考虑未来并发和扩容。固定 `N` 个 shard 在 world size 变化时可以用 `shard_id % world_size` 再分配，但会导致重分配大；一致性哈希减少移动，却需要处理热点和空 shard。训练语料一般优先稳定、可审计的逻辑 shard 列表，再由运行时把逻辑 shard 映射到 rank。

### 11.6.2 确定性分片函数

给每个样本定义稳定键 `k`（原始 URL、业务主键或内容哈希），用：

\[
shard(k)=H(namespace\parallel k)\bmod N.
\]

必须固定哈希算法、编码、namespace 和 `N`。Python 内置 `hash()` 默认带随机盐，不能用于跨进程或跨运行复现；应使用 SHA-256、xxHash 等明确算法。若需要扩容，直接把 `N` 改大通常会重映射大部分样本；可以采用虚拟节点、两级 shard 或在 manifest 中冻结旧映射。

覆盖率与去重检查：

\[
coverage=\frac{|\bigcup_i S_i|}{|D|},\qquad
duplicate\_rate=1-\frac{|\bigcup_i S_i|}{\sum_i |S_i|}.
\]

`coverage=1` 只表示每个输入至少出现一次；仍要检查是否有输入被错误过滤、是否有同内容不同键的重复，以及分片大小的 Gini 或 p99/p50 比例。训练通常允许按 epoch 改变顺序，但不能在恢复后无意跳过或重复大量样本。

### 11.6.3 游标与恢复

不要只保存“rank=3，step=100”。至少保存：

```json
{
  "dataset_snapshot": "reviews-20261005-7f2c",
  "shard_plan": {"num_shards": 128, "hash": "sha256"},
  "rank": 3,
  "world_size": 8,
  "shards": [12, 44, 77],
  "cursor": {"shard": 44, "row_group": 5, "row": 812},
  "samples_seen": 163840,
  "epoch_seed": 99117
}
```

恢复时验证快照和 shard plan 指纹；若 world size 变化，明确是从已提交全局样本游标重分配，还是从上一个 epoch 边界重放。二者都可接受，但必须记录重放数量和对指标的影响。

## 11.7 缓存、去重与 lineage

### 11.7.1 分层缓存

常见缓存层：内存对象缓存、进程共享 Arrow batch、节点 NVMe、机架共享缓存和远端对象存储。缓存命中率 (h) 不能单独代表收益，还要考虑命中与未命中的不同大小和尾延迟：

\[
T_{avg}=h\,T_{hit}+(1-h)\,T_{miss}+T_{evict}+T_{refresh}.
\]

热点数据可能让平均命中率很高，但长尾租户在 miss 风暴中超时。缓存键应包含 snapshot/schema/变换参数，例如：

```text
sha256(snapshot_id || object_hash || transform_code || tokenizer_version)
```

只以路径为键会把新版本误命中到旧结果。缓存写入应先写临时文件、校验后原子 rename；跨节点共享缓存要限制权限，并把用户可控字段做路径转义，防止目录穿越。

### 11.7.2 内容去重与近似去重

**精确去重**用规范化内容的哈希（如 SHA-256）判断完全相同字节；**近似去重**用 SimHash、MinHash、n-gram 等判断相似文本或图像。去重键的选择决定语义：保留 URL 会让同一内容的镜像重复，保留内容哈希可去掉镜像却可能合并合法的多语言或不同上下文样本。

去重索引本身也是数据集的一部分，应记录算法、规范化版本、阈值和时间窗口。近似去重可能误删，质量报告必须给出抽样审计和误合并风险。在线增量去重可用 RocksDB 记录 `fingerprint -> first_seen_snapshot`，定期把状态快照上传远端；状态过期或 compaction 失败时要能从 manifest 重建，而不是把本地库当唯一真源。

### 11.7.3 Lineage 图

Lineage 至少包含节点（原始对象、表快照、变换、质量报告、特征、训练 run）和边（读取、生成、过滤、合并、发布）。边上记录代码提交、容器 digest、参数和时间窗口。一个可查询的 lineage 让你回答：

- 某模型包含哪些原始对象和标签版本？
- 某对象被删除或纠正后，哪些下游模型需要重建？
- 两个数据集差异来自内容、schema、分片还是变换代码？
- 训练指标变化是否与质量门禁或去重阈值变化同步？

不要把 lineage 全部塞进一个巨大 JSON；按快照追加不可变事件，并定期物化索引。索引丢失时，manifest 和变换日志仍应足以重建关键路径。

## 11.8 数据质量、schema 演进与隐私

### 11.8.1 质量维度和门禁

建议至少检查以下维度：

1. **存在性**：必需分区、文件和字段均存在。
2. **类型与范围**：数值、时间、枚举和单位符合 schema。
3. **完整性**：null、空字符串、截断和无效 UTF-8 比例在阈值内。
4. **唯一性**：业务主键、内容哈希和样本 ID 的重复率可解释。
5. **一致性**：跨列关系成立，例如结束时间不早于开始时间。
6. **分布漂移**：类别、长度、语言、地域和标签分布与基线比较。
7. **泄漏风险**：未来字段、评估集重叠、用户级跨集合泄漏。
8. **可追溯性**：每行能回到源对象、抓取时间和变换版本。

门禁分为阻断（block）、警告（warn）和记录（log）。阈值应随数据集版本保存，不能在脚本里悄悄修改。质量报告本身也要有哈希并被 manifest 引用。抽样检查不能替代全量的行数、哈希、schema 和隐私扫描；但对昂贵的语义模型检查可以采用分层抽样并报告置信区间。

### 11.8.2 Schema 演进

兼容性分类：

- **向后兼容**：新消费者可读旧数据，例如新增有默认值的可选列。
- **向前兼容**：旧消费者可忽略新字段。
- **破坏性变化**：改名、改类型、改变单位/nullability、删除字段或重定义枚举。

schema registry 或表格式能保存版本，但不能自动判断业务语义。`int64` 到 `double` 可能技术可读却损失精度；`timestamp_ms` 到 `timestamp_us` 若未更新元数据会产生千倍偏移。发布时应运行旧消费者和新消费者的兼容性测试，并在 manifest 中写明迁移或双写周期。

### 11.8.3 隐私、删除与最小化

数据分类决定加密、访问和保留策略。处理 PII 时采用最小化原则：只采集任务需要的字段，尽早脱敏或令牌化；把映射表与训练数据隔离；日志和样本预览默认不包含原始邮箱、电话、精确位置、生物特征或未成年人信息。

删除请求在不可变对象系统里并非简单 `DELETE key`。需要：

1. 定位受影响的原始对象、派生对象、缓存、索引和模型 lineage；
2. 生成新的不含该主体的快照或按表格式执行受控删除；
3. 使旧快照和备份在保留策略允许后不可再被普通读取；
4. 重新运行质量、去重和训练集重叠检查；
5. 留下删除证明，但证明本身不暴露被删除内容。

加密密钥轮换不能替代逻辑删除；“把对象改名”也不能让备份和缓存消失。高敏感字段进入第三方服务前需有明确授权、最小范围和审计记录。

## 11.9 流处理与批处理：同一语义的两种时间尺度

### 11.9.1 事件时间、处理时间和摄入时间

- **事件时间**：业务事件实际发生的时间。
- **摄入时间**：平台接收到事件的时间。
- **处理时间**：算子处理事件的本地时钟。

标签、窗口和去重通常应基于事件时间；SLA 和资源监控基于处理时间。若把迟到事件按处理时间归入窗口，会出现跨窗口标签错配。所有时间字段都要包含时区或统一为 UTC，并定义时钟回拨和无时间事件的处理。

### 11.9.2 Watermark、迟到与状态

watermark 是系统对“某事件时间之前的数据基本已到达”的声明，不是全局真理。以事件时间窗口 `[a,b)` 为例，当 watermark 越过窗口结束边界 `b` 时可发出首个结果；如果配置允许迟到 `L`，通常继续保留该窗口状态，直到 watermark 越过 `b+L` 才回收。此后到达、原本属于该窗口的事件进入旁路、补偿任务或被丢弃。不同引擎对边界时刻的严格/非严格比较和触发器语义不同，需按版本测试。选择 `L` 需要平衡结果及时性、状态大小和修正成本。

流处理状态包含窗口聚合、去重指纹和连接缓存。checkpoint 要保存算子状态、输入 offset、watermark 和 schema 版本；只保存输出文件会在恢复后重复处理或错过事件。状态后端可以使用 RocksDB，但 compaction、TTL 和 checkpoint 上传应纳入吞吐预算。

### 11.9.3 幂等 sink 与近似 exactly-once

端到端 exactly-once 往往依赖每个边界的幂等：

```text
输入 offset 提交 + 状态快照 + 输出事务提交
```

如果 sink 支持以 `(job_id, epoch, partition, sequence)` 为幂等键，重试写入不会产生重复；如果只支持追加文件，就要在 manifest 层按版本选择一次，并在消费端去重。不要把“消息系统 exactly-once”直接等同于“业务结果 exactly-once”，中间的外部 API、对象存储和模型推理仍可能重复调用。

### 11.9.4 流批统一

同一逻辑变换若分别实现流版和批版，容易出现边界和 null 语义差异。推荐共享规范化函数、schema 和 golden test：用一批固定事件同时跑微批和离线重放，比较按键聚合、迟到修正、去重和最终快照。流作业的周期性 checkpoint 可产生批作业消费的 manifest；批作业重放同一 manifest 时，结果应在允许的浮点容差内一致。

## 11.10 可复现数据管道

### 11.10.1 锁定输入、代码、环境和随机性

可复现不只锁定 Python 包。建议在 manifest 或 run 记录中保存：

- 输入快照 ID、每个对象哈希和分区计划；
- 变换代码提交、依赖 lockfile、容器 digest 和系统架构；
- schema、词表、tokenizer、图像解码器和时区数据库版本；
- 并发度、线程数、排序稳定性、随机种子和哈希算法；
- 参数、环境变量白名单、CPU 指令集和浮点模式；
- 输出 manifest、质量报告、日志和执行时间线。

随机性要分层：采样、数据增强、分片顺序、并行归约和第三方库各自可能有 RNG。只调用一个全局 `seed(42)` 不足以保证。若业务只要求语义复现，可定义样本集合、统计量和模型指标容差，而不追求逐字节或 bitwise 相等。

### 11.10.2 幂等与增量

每个变换应尽量满足：相同输入 snapshot 和参数，重复运行产生相同输出内容或可证明等价的输出。输出对象名可包含输入哈希和变换版本；先检查已存在且校验通过的对象，再决定是否重算。增量处理把“上次成功边界”写成 manifest 或 offset，而不是读取目录里最新文件。

失败重试时要区分：

- 网络超时：可重试同一请求，使用指数退避和抖动；
- 部分上传：清理临时对象后重试，不能复用未知完整性的 multipart；
- schema/质量失败：不可盲目重试，应阻断并告警；
- 代码 bug：固定输入重放，修复后生成新 snapshot；
- 资源不足：调整分片/并发或排队，不通过无限重试放大流量。

### 11.10.3 观察和成本

数据系统要暴露每个 snapshot 的行数、字节、压缩率、文件数、平均/最大文件大小、质量失败数、缓存命中、对象请求数、重试、延迟分位数、compaction 积压和 lineage 事件。指标标签不能直接使用高基数用户 ID；按租户、管道、分区和错误类型聚合。成本按快照、管道和租户归因，避免把跨区流量和临时 SSD 隐藏在平台总账里。

## 11.11 CPU 可运行实验：确定性分片、去重与 manifest

下面的实验不依赖云服务，使用本地 JSONL 模拟原始对象，验证四个性质：

1. 同一 `namespace + key` 每次映射到相同 shard；
2. 重新运行不会产生重复输出；
3. 所有输入 key 恰好覆盖一次；
4. 只有通过哈希和行数验证的 manifest 才被消费。

保存为 `ch11_cpu_shard_experiment.py` 后运行（Python 3.10+）：

```python
from __future__ import annotations
import hashlib, json, shutil, tempfile
from pathlib import Path

N_SHARDS = 4
N_ROWS = 37
NAMESPACE = "reviews.v4"


def stable_shard(key: str, n: int = N_SHARDS) -> int:
    raw = (NAMESPACE + "\0" + key).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % n


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def make_input(root: Path) -> Path:
    src = root / "raw.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for i in range(N_ROWS):
            # 故意让文本有重复，但 key 不重复，观察“键去重”和“内容去重”的差异。
            row = {"id": f"r-{i:03d}", "text": "same" if i % 9 == 0 else f"review-{i}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return src


def build_snapshot(src: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError("immutable snapshot already exists")
    tmp = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
    handles = [ (tmp / f"shard-{i:03d}.jsonl").open("w", encoding="utf-8")
                for i in range(N_SHARDS) ]
    seen = set()
    counts = [0] * N_SHARDS
    try:
        for line in src.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = row["id"]
            if key in seen:
                raise ValueError(f"duplicate key: {key}")
            seen.add(key)
            shard = stable_shard(key)
            handles[shard].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[shard] += 1
    finally:
        for h in handles:
            h.close()

    files = []
    for p in sorted(tmp.glob("shard-*.jsonl")):
        files.append({"name": p.name, "bytes": p.stat().st_size,
                      "sha256": sha256(p),
                      "rows": len(p.read_text(encoding="utf-8").splitlines())})
    manifest = {"snapshot": "demo-001", "namespace": NAMESPACE,
                "num_shards": N_SHARDS, "input_rows": len(seen),
                "counts": counts, "files": files,
                "keys_sha256": hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest()}
    (tmp / "manifest.intent").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # 验证后再提交；rename 在同一文件系统内是原子的，远端对象存储需用条件写替代。
    verify_manifest(tmp / "manifest.intent")
    (tmp / "manifest.done").write_text((tmp / "manifest.intent").read_text(encoding="utf-8"), encoding="utf-8")
    tmp.rename(out)
    return manifest


def verify_manifest(path: Path) -> None:
    m = json.loads(path.read_text(encoding="utf-8"))
    total = 0
    ids = []
    names = set()
    if m["namespace"] != NAMESPACE or m["num_shards"] != N_SHARDS:
        raise ValueError("incompatible shard plan")
    for item in m["files"]:
        if item["name"] in names or Path(item["name"]).name != item["name"]:
            raise ValueError("duplicate or unsafe file name")
        names.add(item["name"])
        p = path.parent / item["name"]
        if not p.exists() or p.stat().st_size != item["bytes"]:
            raise ValueError(f"size/missing: {p}")
        if sha256(p) != item["sha256"]:
            raise ValueError(f"hash mismatch: {p}")
        with p.open(encoding="utf-8") as f:
            part = [json.loads(line) for line in f]
        for row in part:
            if item["name"] != f"shard-{stable_shard(row['id']):03d}.jsonl":
                raise ValueError("wrong shard assignment")
            ids.append(row["id"])
        rows = len(part)
        if rows != item["rows"]:
            raise ValueError(f"row mismatch: {p}")
        total += rows
    if total != m["input_rows"]:
        raise ValueError(f"coverage {total} != {m['input_rows']}")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate keys")
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    if digest != m["keys_sha256"]:
        raise ValueError("key set mismatch")


def consume(snapshot: Path) -> list[str]:
    done = snapshot / "manifest.done"
    if not done.exists():
        raise RuntimeError("uncommitted snapshot")
    verify_manifest(done)
    m = json.loads(done.read_text(encoding="utf-8"))
    ids = []
    for item in m["files"]:
        for line in (snapshot / item["name"]).read_text(encoding="utf-8").splitlines():
            ids.append(json.loads(line)["id"])
    return ids


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="ch11-shard-"))
    try:
        src = make_input(root)
        snap = root / "snapshot"
        m1 = build_snapshot(src, snap)
        ids1 = consume(snap)
        # 第二次构建到不同目录，比较 manifest 和集合；输出顺序由文件名和输入顺序确定。
        snap2 = root / "snapshot2"
        m2 = build_snapshot(src, snap2)
        ids2 = consume(snap2)
        print("counts:", m1["counts"])
        print("coverage:", len(ids1), "unique:", len(set(ids1)))
        print("deterministic_manifest:", m1 == m2)
        print("deterministic_ids:", ids1 == ids2)
        # 注入损坏，验证消费者拒绝半写文件。
        bad = snap / "shard-000.jsonl"
        with bad.open("a", encoding="utf-8") as f:
            f.write("{bad json}\n")
        try:
            consume(snap)
        except ValueError as e:
            print("corruption_rejected:", type(e).__name__)
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
```

一个典型输出（临时目录名和哈希不影响结果）如下：

```text
counts: [12, 9, 6, 10]
coverage: 37 unique: 37
deterministic_manifest: True
deterministic_ids: True
corruption_rejected: ValueError
```

实验中故意让多个样本的 `text` 相同，但 `id` 不同，说明基于业务键的分片并不等于内容去重。可以把 `key=row["id"]` 改成规范化文本的 SHA-256，比较 coverage、重复率和合法多样性损失。还可以把 `N_SHARDS` 从 4 改成 8，观察大多数 key 被重新映射；若要支持在线扩容，需要在 manifest 中冻结映射或实现虚拟 shard。

### 11.11.1 实验扩展

1. 删除 `manifest.done`，确认消费端拒绝临时目录；
2. 修改一个 shard 的一行，确认哈希检查失败；
3. 打乱输入 JSONL 行顺序，比较集合相等但 `deterministic_ids` 是否仍为真，并思考是否需要按 key 排序；
4. 并发运行两个构建器，给 manifest 写入增加版本条件，模拟冲突；
5. 统计各 shard 数量的 p50、p99 和最大/最小比值，构造一个热点 key 集合；
6. 用 `pyarrow` 将 JSONL 写成 Parquet，改变 row group 大小，测量过滤查询的字节读取量；
7. 把去重索引改成 RocksDB 或 sqlite，注入进程崩溃，检查 WAL/事务后的状态；
8. 为每个样本增加事件时间和迟到标记，实现一个批式 watermark 报告。

## 11.12 失败案例：数据“成功”但模型结果失真

下面是综合常见失效模式构造的教学案例，数值用于解释诊断过程，不代表某个已公开的真实事故。

### 案例一：分页处理错误导致训练少了 3% 样本

某管道在每个 epoch 扫描对象存储前缀，只处理了第一页结果，且没有继续使用响应中的 continuation token。大多数小批次没有超出单页上限，缺陷长期未暴露；高峰期间对象数增加，部分 key 没有被读取，作业仍按“读取到的文件数 > 0”判定成功。模型在短期验证集上没有明显下降，但长尾语言覆盖率降低。

直接原因是分页实现错误，更深层的问题是把动态列表当作快照且缺少预期集合。修复方式是先完成并验证不可变 manifest，训练只读取 manifest；对每个对象记录哈希、行数和事件时间范围；在训练前比较 manifest 的预期总行数与质量报告。若必须在线摄取，使用表格式或版本化清单定义一致性边界，不在消费过程中列举可变前缀。

### 案例二：Parquet 小文件和高基数分区拖垮元数据服务

为了按用户和小时查询，团队生成了数千万个小 Parquet 文件。单文件只有几十 KB，压缩率很好，但训练启动要执行数百万次 `HEAD` 和 schema 读取，p99 延迟超过 20 分钟。增加 worker 只让元数据服务更快耗尽连接。

修复包括：把高基数用户键移到文件内排序和 bloom/统计过滤，分区只保留日期或区域；按目标 row group 大小合并文件；在 manifest 中缓存 schema、行数和哈希，训练无需逐文件探测；设置并发上限和预取窗口。合并后要重新测量随机采样成本，必要时建立轻量索引或抽样 shard，不能只追求文件数下降。

### 案例三：RocksDB compaction 与 SSD 写放大引起流处理延迟

在线去重状态使用 RocksDB，输入流量增加后 L0 文件积压，compaction 抢占 CPU 和 NVMe 带宽。延迟 p99 上升，watermark 停滞，迟到事件被错误地丢到旁路。团队尝试把内存表调大，短期减少 flush，却让节点 OOM 并触发重启。

正确排查顺序是同时看 `pending_compaction_bytes`、L0 文件数、写放大、block cache 命中、磁盘队列和 checkpoint 上传；降低 compaction 并发或把状态迁移到独立 SSD，限制输入并发，增加 TTL 和分层状态；为迟到事件保留可重放的原始日志；重启后从已提交 offset+状态 checkpoint 恢复。不能通过放宽 watermark 或无限重试掩盖状态后端饥饿。

### 案例四：schema “兼容”却改变了时间单位

生产者把 `event_time` 从毫秒改为微秒，字段名和整数类型都没变，因此 registry 判定兼容。下游按旧单位解释新值，把事件推到极远的未来，或因越界而解析失败，质量检查只看非空率，模型训练悄悄丢弃这些样本。

修复是在 schema 中编码单位和逻辑类型，加入时间范围与时区门禁；发布前用旧消费者、新消费者和固定 golden 数据比较窗口结果；把单位变更视为破坏性版本，双写期间同时保留旧列和新列。数值类型相同不等于语义相同。

### 案例五：删除请求只删除了真源，缓存和模型仍含敏感数据

团队收到用户删除请求后删除对象存储中的原始记录，却忘了 RocksDB 去重索引、节点 SSD、特征表快照和已训练模型。审计时无法指出哪些模型读取过该记录，也无法证明备份何时过期。

修复需要从 lineage 图反向遍历所有派生对象和 run，生成新的不含主体的快照，按保留策略清理缓存和备份，并记录删除证明。高敏感字段不进入普通日志；密钥轮换只解决访问控制，不替代数据重建和过期删除。

## 11.13 六个理解检查（含答案）

### 检查 1：为什么训练不应在对象存储上直接扫描可变前缀？

**答案**：前缀是命名空间，不是快照。即使服务对单次读取和列表提供强一致性，并发新增/删除与分页错误仍会改变消费集合，而且单个对象成功不代表整批文件已完成。S3 的原子 PUT 不会向读者暴露半写单对象，但读者仍可能看到只完成一部分的业务批次。不可变对象加已验证的 `manifest.done` 才能定义一致的文件集合；训练读取 manifest 中的 URI 和哈希，而不是猜目录状态。

### 检查 2：LSM-tree 为什么写入快，却可能让读延迟和 SSD 写放大上升？

**答案**：写入先顺序追加 WAL 和 memtable，避免随机更新；刷盘后多个 SSTable 需要在读时查询，后台 compaction 又会反复重写数据。SSTable 数、Bloom filter、cache 和 compaction 策略决定读放大与写放大。应监控 L0 积压、pending bytes、写放大和 p99，而不是只看平均吞吐。

### 检查 3：Parquet 和 Arrow 分别解决什么问题？

**答案**：Parquet 是面向持久化和扫描的列式文件格式，提供 row group、列裁剪、统计和压缩；Arrow 是进程/服务间交换的列式内存布局，减少序列化和拷贝。Parquet 文件读入 Arrow 仍可能因压缩、类型转换、非连续切片或 Python 对象产生拷贝，不能把“列式”自动等同于“零拷贝”。

### 检查 4：怎样证明分片既无遗漏又无重复？

**答案**：固定哈希算法、namespace、键编码和 shard 数，在 manifest 中保存 shard plan 指纹；汇总各 shard 的样本键或可验证计数，检查并集大小等于输入键集合、总计数等于并集、重复率为零或在声明阈值内。仅比较每个 shard 的行数不能发现同一键跨 shard 重复，也不能证明输入没有被过滤掉。

### 检查 5：流处理的 watermark 是否保证迟到事件不会再改变结果？

**答案**：不保证。watermark 是进度估计，系统还要定义允许迟到窗口、旁路、补偿更新或丢弃策略。checkpoint 应保存 offset、算子状态和 watermark；恢复时重放必须是幂等的。watermark 过早会丢迟到事件，过晚会增大状态和结果延迟。

### 检查 6：为什么只锁定代码提交和随机种子仍可能无法复现？

**答案**：输入 snapshot、对象内容、schema/单位、词表、解码器、依赖和容器、线程/并发、哈希算法、CPU 指令集、分片映射和数据游标都可能变化。可复现记录应包含输入对象哈希、manifest、变换参数、环境 digest、随机状态和输出质量报告；还要明确追求 bitwise、数值、轨迹还是语义层级的复现。

## 11.14 练习

1. **存储选型**：给出 10 TB 训练语料、100 KB/样本、每日增量和随机抽样需求，比较对象存储、分布式文件系统和本地 SSD 的布局、成本、RPO 与吞吐。
2. **请求放大**：用 (T=T_{setup}+N_{req}L+B/BW) 估算 1 亿个 1 KB 小对象与 1000 个 100 MB 对象的差异，加入重试概率和并发上限。
3. **LSM 参数**：在 RocksDB 模拟中改变 memtable 大小、compaction 并发和压缩算法，记录写放大、p99 写延迟、L0 积压和空间峰值。
4. **Parquet 基准**：用 Arrow 写三种 row group 大小，分别测试全列扫描、单列过滤和随机抽样，比较读字节、CPU、峰值内存和 p99。
5. **manifest 提交**：实现 intent/done 两阶段提交，随机杀死进程，证明消费者只读取最近一个完整 done，并能清理未提交临时对象。
6. **分片扩容**：比较 `hash % N`、一致性哈希和虚拟 shard 从 4 扩到 8 时的样本移动比例与热点分布。
7. **去重语义**：对 URL、规范化文本和内容哈希分别去重，报告误删多语言样本、镜像重复和近似去重阈值的影响。
8. **schema 演进**：设计从毫秒到微秒时间戳、从 nullable 到 required、从字符串到字典编码的兼容测试和双写迁移计划。
9. **流批一致**：用固定事件集实现批式和微批式窗口聚合，加入迟到事件，比较 watermark、allowed lateness 和最终快照。
10. **删除与 lineage**：给定一个包含 PII 的数据集和两个训练 run，画出受删除请求影响的对象、缓存、特征和模型图，写出验证删除完成的证据。
11. **缓存压力**：构造 90% 热点、10% 长尾的访问分布，比较 LRU、LFU、分层缓存和按租户配额的 p99 延迟与公平性。
12. **可复现矩阵**：固定 snapshot，改变线程数、排序稳定性、容器 digest 和 CPU 指令集，定义 bitwise、数值和语义三种验收指标并解释差异。

## 11.15 安全边界与数据系统清单

### 11.15.1 访问、密钥与租户隔离

对象存储、文件系统、RocksDB 备份、表格式元数据和 lineage 服务分别使用最小权限身份。训练容器只读所需 manifest 和对象前缀，不能列举或删除其他租户；写入端不能修改已发布 snapshot。密钥通过受控的密钥服务或短期凭证注入，绝不写入 manifest、日志、环境快照或样本内容。跨区域复制和供应商支持访问要有范围、时限和审计。

### 11.15.2 软删除、硬删除与备份

把对象标记删除、删除 manifest 引用、清理缓存、清理备份和真正不可恢复的硬删除分开记录。自动生命周期规则要避开仍被 lineage 引用的快照；硬删除前保留可验证的恢复点和操作者记录。若法规要求保留原始审计证据，需定义不可篡改存档与访问控制，不能把“方便训练”凌驾于适用的保留或删除义务。

### 11.15.3 输入不可信与解析器安全

JSON、CSV、Parquet、图像和压缩包都可能来自不可信来源。限制单对象大小、嵌套深度、解压比、行长度、列数量和 CPU 时间；禁用不必要的外部链接、宏或代码执行；对解析器和依赖做补丁和沙箱。文件名、partition 值和 URL 不能直接拼接本地路径或 shell 命令。遇到畸形输入要隔离到 quarantine，并保留哈希和错误类型以便审计。

### 11.15.4 数据泄漏与模型训练边界

训练、验证、测试和线上回放按主体或时间隔离，避免用户级跨集合泄漏。日志、指标和样本预览默认脱敏；调试导出使用合成数据。第三方标注、OCR、embedding 或安全扫描服务只接收最小必要字段，传输前确认授权和保留政策。若数据可能包含未成年人、健康、金融、精确位置或凭证信息，提升审批和审查级别，不以“只是训练”作为豁免。

### 11.15.5 可用性与资源拒绝服务

对单租户对象请求率、文件数、row group 大小、RocksDB 状态、compaction、缓存和 lineage 写入设置配额。限制重试次数和退避上限，防止上游故障造成 thundering herd。缓存 miss 风暴时先保护真源和其他租户；数据质量失败应阻断该快照而不是无限重试。监控本地 SSD 寿命、对象存储费用、跨区流量和元数据 CPU，达到阈值触发降级或排队。

### 11.15.6 事故响应证据

发生数据损坏、隐私暴露或 manifest 冲突时，保留：snapshot/manifest 哈希、操作人和服务身份、代码与镜像 digest、请求 ID、对象版本、质量报告、删除或恢复时间线。先冻结受影响快照和下游发布，再用已验证的上一个版本恢复；不要在事故中直接覆盖或清理证据。恢复后增加能重现根因的测试和门禁。

## 11.16 版本边界与迁移注意

1. S3、GCS、Azure Blob、Ceph RGW 等对象存储的一致性、条件写、版本化和列表语义不同；以目标服务和 SDK 版本的官方文档为准，不能假设所有实现都等同于 POSIX。
2. HDFS、CephFS、Lustre、BeeGFS 和 NFS 的锁、缓存、`fsync`、故障恢复与容器挂载不同；在目标介质上做崩溃和断电测试。
3. RocksDB 不同版本和配置的 compaction、blob DB、WAL、checkpoint 与统计字段会变化；升级前在副本上测量写放大、读尾延迟和恢复时间。
4. Parquet logical type、timestamp、加密和 page index 的支持取决于 writer/reader 版本；Arrow C Data Interface、Flight 和各语言绑定的内存所有权要按版本验证。
5. Delta Lake、Iceberg、Hudi 的事务、快照、删除向量、schema 演进和 vacuum 语义不同；表格式版本与引擎版本必须一起锁定。
6. Flink、Spark Structured Streaming、Beam 等引擎对 checkpoint、watermark、allowed lateness、状态后端和 exactly-once sink 的定义不同；以目标版本的执行器和连接器文档为准。
7. 数据隐私与删除要求受地域、行业和合同影响。来源地图只帮助理解技术机制，不替代组织法务、隐私和安全审查。

## 11.17 来源地图

以下来源优先选择规范、官方文档和维护者资料，用于核对协议与实现边界；阅读时应切换到部署所用的版本。

- [Amazon S3 数据一致性模型](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)：对象读写、版本化和条件请求的服务契约。
- [Google Cloud Storage 一致性](https://cloud.google.com/storage/docs/consistency)：对象、列表和元数据操作的一致性说明。
- [Azure Blob Storage 并发控制](https://learn.microsoft.com/en-us/azure/storage/blobs/concurrency-manage)：ETag、条件写与并发控制。
- [POSIX `fsync(2)`](https://man7.org/linux/man-pages/man2/fsync.2.html)：文件和目录持久化边界，具体文件系统需实测。
- [The Linux Storage Stack](https://www.kernel.org/doc/html/latest/block/index.html)：块层、I/O 调度和设备队列背景。
- [RocksDB Wiki](https://github.com/facebook/rocksdb/wiki)：WAL、memtable、SSTable、compaction、列族和性能调优。
- [RocksDB Tuning指南](https://github.com/facebook/rocksdb/wiki/RocksDB-Tuning-Guide)：写放大、读放大、缓存和后台线程指标。
- [Apache Parquet 文档](https://parquet.apache.org/docs/overview/)：文件布局、类型、编码和压缩。
- [Apache Arrow 格式规范](https://arrow.apache.org/docs/format/Columnar.html)：列式内存布局、buffer、null bitmap 和 IPC。
- [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)：跨语言零拷贝交换与内存所有权。
- [Apache Iceberg 规范](https://iceberg.apache.org/spec/)：表 metadata、manifest、快照和 schema 演进。
- [Delta Lake Protocol](https://github.com/delta-io/delta/blob/master/PROTOCOL.md)：事务日志、版本和动作模型。
- [Apache Hudi 文档](https://hudi.apache.org/docs/overview/)：增量摄取、表服务和时间线。
- [Apache Flink 状态与 checkpoint](https://nightlies.apache.org/flink/flink-docs-stable/docs/ops/state/checkpoints/)：状态、checkpoint、savepoint 和恢复语义。
- [Apache Beam Programming Guide](https://beam.apache.org/documentation/programming-guide/)：事件时间、watermark、窗口和触发器。
- [Spark Structured Streaming 指南](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)：微批、连续处理、watermark 和输出模式。
- [OpenLineage 规范](https://openlineage.io/docs/spec/)：作业、运行、数据集事件和 lineage 字段。
- [W3C PROV](https://www.w3.org/TR/prov-overview/)：通用 provenance 概念与交换模型。
- [Great Expectations 文档](https://docs.greatexpectations.io/)：数据质量断言、验证结果和数据文档。
- [TensorFlow Data Validation](https://www.tensorflow.org/tfx/guide/tfdv)：schema、统计、漂移和异常检测。
- [NIST Privacy Framework](https://www.nist.gov/privacy-framework)：隐私风险识别、治理与控制。
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：数据、评估和部署风险的治理框架。
- [GDPR 官方法规文本](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng)：第17条涉及适用场景下的删除与例外；具体义务需法务判断。

[方法说明] 本章中的公式、状态机和 CPU 实验是教学模型，不承诺任何云服务、文件系统、RocksDB 配置或流处理引擎的性能。请在目标版本、硬件、区域、权限和数据分类上复现测量；对合规、删除和高敏感数据处理，必须遵守组织政策与适用法律。

## 11.18 章节完成标准

读者完成本章后，应能对一条数据管道回答：

1. 真源是对象、文件系统、表格式还是 KV 状态，各自的持久化和故障边界是什么？
2. 数据快照由哪个 manifest 定义，文件哈希、schema、分区和质量报告在哪里？
3. 读取路径的请求数、带宽、缓存命中、尾延迟和成本是多少？
4. Parquet row group、Arrow buffer、压缩和解码如何影响 CPU、内存与网络？
5. 分片键、哈希算法、world size 变化和游标恢复规则是否冻结且可验证？
6. RocksDB WAL、memtable、SSTable、compaction 和备份如何与快照提交配合？
7. 流处理的事件时间、watermark、迟到、checkpoint、offset 和 sink 幂等语义是什么？
8. schema、单位、缺失、重复、漂移、泄漏和 PII 检查哪些阻断、哪些警告？
9. lineage 是否能从模型追到原始对象，也能从删除请求反向定位派生物？
10. 失败、重试、删除、回滚和事故响应是否有最小权限、证据和可恢复路径？

如果不能回答其中一项，先在 CPU 实验中写出 manifest、分片和损坏注入，再在目标存储和引擎上做小规模基准，最后才把快照提供给大规模训练。不要用“目录看起来完整”“列表返回了文件”或“表能被读取”代替数据系统证据。

## 11.19 小结

AI 数据系统不是把文件放在云上，而是定义一组可验证的集合、顺序、状态和责任边界。对象存储提供耐久和规模，文件系统提供共享命名空间，本地 SSD 提供低延迟缓存；RocksDB/LSM 处理小型增量状态，Parquet 负责高效持久化扫描，Arrow 负责进程内列式交换。Manifest 把这些对象组成不可变快照，分片器把快照映射给并发消费者，缓存和去重在不改变语义的前提下降低成本，lineage 让结果可以追溯和重建。

流和批并非两个互不相干的世界：事件时间、watermark、状态 checkpoint、幂等 sink 和 manifest 可以把二者连接起来。可复现管道还需要锁定输入哈希、schema、代码、环境、随机性和分片计划，质量门禁则要覆盖范围、类型、重复、漂移、泄漏和隐私。最终目标不是“永远不失败”，而是失败时能证明读到了什么、提交到了哪里、丢失了多少、谁有权访问，以及如何从上一个可信快照继续。
