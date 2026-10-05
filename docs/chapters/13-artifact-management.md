---
id: ch13-artifact-management
title: 实验与模型工件管理：配置、注册表、版本、血缘与可复现构建
slug: /chapters/13-experiment-artifacts
description: 将配置、数据、代码、环境、模型和评估结果组织成不可变、可寻址、可追溯且可回滚的工件链路
sidebar_position: 13
level: systems
prerequisites:
  - ch02-linux-process-files-observability
  - ch04-performance-math
  - ch06-pytorch-execution
  - ch10-training-ops
  - ch11-data-systems
  - ch12-evaluation
learning_objectives:
  - 能把配置、代码、数据、环境、检查点和评估报告区分为可寻址工件，并定义不可变边界
  - 能设计内容哈希、manifest、签名和注册表，使工件可以校验、去重、审计和回滚
  - 能为数据集、模型、tokenizer、特征和评估结果建立有向无环的 provenance/lineage 图
  - 能解释版本号、内容地址和人类别名（如 latest）各自的语义，避免把别名当版本
  - 能设计可复现构建流程，控制依赖锁定、时间戳、随机数、排序、并行和硬件差异
  - 能处理数据和模型的联合版本、兼容性矩阵、迁移脚本及 schema 演进
  - 能建立发布、回滚、保留、撤回和审计策略，区分可恢复删除与不可恢复擦除
  - 能在 CPU-only 环境运行标准库 artifact manifest 实验并解释篡改、缺失和漂移失败
  - 能识别工件管理中的安全、隐私、供应链和合规边界，知道何时升级人工审查
  - 能用理解检查和练习把工件证据接入训练、评估和服务运维
estimated_hours: 22
hardware: CPU-only baseline; GPU/remote registry optional
risk_level: L2
last_verified: 2026-10-05
---

# 第13章　实验与模型工件管理：配置、注册表、版本、血缘与可复现构建

> 一个模型并不等于一个 `.pt` 或 `.safetensors` 文件。模型的数值依赖于代码、配置、数据快照、tokenizer、随机状态、编译器、依赖、硬件和评估规则。若这些对象没有稳定的身份、完整的血缘和可验证的内容，所谓“同一模型”只是一种叙事。本章把实验与发布对象称为工件（artifact），讨论如何为工件分配内容地址，怎样在注册表中维护生命周期，如何记录 provenance/lineage，以及怎样把构建步骤变成可回放、可审计、可回滚的过程。

本章不要求读者先部署某个特定平台。我们从文件和标准库开始，建立最小的工件契约，再把它映射到对象存储、模型注册表、CI/CD、供应链证明和生产回滚。示例中的 manifest 是教学实现，生产环境应叠加访问控制、签名、不可篡改日志、密钥管理和组织政策。

## 13.1 工件是实验的语义边界

### 13.1.1 从“文件”改成“可寻址对象”

[机制] 文件路径是位置，不是身份。`/runs/exp17/model.pt` 可能被覆盖、复制到不同磁盘，或者在不同容器里解析到不同内容。工件身份应该至少包含四个维度：

1. **类型（type）**：config、source、dataset、tokenizer、checkpoint、evaluation、environment、release 等。
2. **内容摘要（digest）**：对规范化字节计算的 SHA-256 或组织认可的算法，格式通常是 `sha256:<hex>`。
3. **描述信息（metadata）**：媒体类型、字节数、创建者、生成时间、许可证、分类和保留策略。
4. **关系（relations）**：输入工件、生成步骤、父版本、兼容的 schema、测试和签名证明。

内容地址（content address）回答“这串字节是什么”，而注册表中的逻辑版本回答“团队把它当作哪一次发布”。两者要同时存在但不能混淆：逻辑版本可以指向一个 digest，digest 不应因为改了显示名称而改变。

可把工件写成二元组：

\[
A=(d,m),\quad d=H(\operatorname{canonicalize}(b)),
\]

其中 `b` 是原始字节，`canonicalize` 规定换行、编码、字段顺序和浮点格式，`m` 是元数据。若同一 JSON 以不同空白或键顺序保存，原始哈希会不同；因此 manifest 中应明确哈希的是原始字节还是规范化表示。对模型权重通常哈希原始文件，对配置和记录可先采用 JCS（JSON Canonicalization Scheme）或稳定排序。

### 13.1.2 工件类型和最小字段

建议为每类工件定义 schema，而不是在一个任意 JSON 里不断增加键。一个通用最小记录如下：

```json
{
  "artifact_id": "sha256:…",
  "kind": "checkpoint",
  "media_type": "application/vnd.model.checkpoint+zip",
  "size_bytes": 38192017,
  "created_at": "2026-10-05T12:00:00Z",
  "producer": {"name": "train.py", "revision": "git:9c2e1f7"},
  "inputs": [
    {"role": "dataset", "artifact_id": "sha256:…"},
    {"role": "tokenizer", "artifact_id": "sha256:…"},
    {"role": "config", "artifact_id": "sha256:…"}
  ],
  "properties": {
    "model_family": "decoder-small",
    "schema_version": 3,
    "step": 12000,
    "metrics": {"valid_loss": 1.93}
  },
  "integrity": {"sha256": "…", "signature": "sigstore:…"}
}
```

必填字段应尽量少且稳定；实验专属字段放进 `properties`，并声明 `schema_version`。把 `created_at` 直接放进被哈希内容会破坏可重复构建，因此要区分“构建输入”与“登记时间”：前者进入 digest，后者由注册表在写入时附加。

### 13.1.3 配置不是参数袋，而是输入工件

训练配置通常包含学习率、批大小、数据路径、混合权重、随机种子、精度、编译开关和评估规则。配置如果来自环境变量、命令行和默认值的隐式合并，就很难复现。推荐经过三步：

1. **收集原始配置**：保存命令行、环境变量白名单和默认值来源。
2. **解析和验证**：把字符串转换成带单位和范围的类型，拒绝未知键，检查互斥选项。
3. **冻结规范配置**：按稳定键顺序序列化成只读工件，记录 schema 和来源。

例如，`batch_size=8` 只说明每个设备的微批还是全局批？`lr=1e-4` 是否随有效批大小线性缩放？`precision=bf16` 是否在 CPU fallback 到 fp32？这些语义必须落在 schema 中。配置一旦用于训练就不应原地修改；修复拼写或默认值要创建新 digest，并在注册表记录 supersedes 关系。

## 13.2 不可变工件与可变别名

### 13.2.1 三层命名

为避免“latest 覆盖”造成追溯断裂，可以采用三层命名：

- **摘要名**：`sha256:ab12…`，内容地址，永不复用。
- **不可变版本**：`model/decoder-small/1.4.0` 或 `run/20261005-001`，指向一个摘要，发布后禁止重绑定。
- **可变别名**：`model/decoder-small/canary`、`production`、`latest`，通过受控操作指向某个不可变版本。

读取训练或评估输入时必须解析到摘要或不可变版本，并把解析结果写入 manifest；在线服务可以读取别名，但启动日志要记录别名解析到的 digest。别名更新应是原子的，包含操作者、理由、前后值和审批证据。

### 13.2.2 何时必须不可变

以下对象一旦被消费，就应不可变或由版本化快照表示：

- 训练和评估数据集、样本索引、过滤规则和 tokenizer 词表
- 代码提交、依赖锁文件、编译器和基础镜像摘要
- 模型权重、优化器状态、学习率调度器状态、随机数状态
- 评估输入、提示模板、评分脚本、人工 rubric 和标注导出
- 生产发布包、模型服务器配置、特征 schema 和安全策略

缓存、日志和临时分片可以是可变的，但只要它们影响结果，就必须在最终 manifest 中固化摘要。一个简单判断是：如果删除或替换对象会改变“同样输入是否得到同样输出”，它就不能只靠可变路径引用。

### 13.2.3 删除和保留

“不可变”不等于“永远保留”。保留策略需要把可恢复删除和不可恢复擦除分开：

- **逻辑撤回**：注册表将工件标记为 revoked，阻止新消费者，但保留字节以支持事故调查。
- **归档**：移动到低成本存储，仍可按 digest 恢复。
- **到期删除**：满足保留期、法律和业务条件后删除对象，并保留删除证明。
- **合规擦除**：删除含个人数据的原始和派生工件，记录范围、批准人和不可恢复证据；可能需要重建受影响模型。

不要把“对象存储返回 404”当成删除完成的全部证据。版本化、复制、缓存、备份和下游导出可能仍然存在。数据主体删除和模型记忆治理属于高风险流程，需专门的隐私与法务审查。

## 13.3 Manifest：把集合、顺序和规则写下来

### 13.3.1 文件哈希不够

单个文件哈希只能证明文件字节，不能证明它属于哪组文件、以什么顺序拼接、由何种规则筛选。数据集和模型发布需要一个顶层 manifest：

```json
{
  "manifest_version": 2,
  "snapshot_id": "dataset:reviews@2026-10-05.3",
  "objects": [
    {"path": "part-000.jsonl", "sha256": "…", "bytes": 2718, "rows": 42},
    {"path": "part-001.jsonl", "sha256": "…", "bytes": 2661, "rows": 41}
  ],
  "ordering": {"mode": "lexicographic", "locale": "C"},
  "schema": "reviews.v4",
  "transform": {"code": "git:9c2e1f7", "rules": "sha256:…"},
  "parent": "dataset:reviews@2026-10-04.9",
  "totals": {"objects": 2, "rows": 83, "bytes": 5379}
}
```

其中 `objects` 的顺序、重复项、总计和父快照都应由验证器检查。若把对象列表按哈希排序，可抵抗文件系统列举顺序变化；若业务要求时间顺序，则把排序键和时区写出。manifest 自身也要计算 digest，形成“manifest 的 manifest”或在注册表中以内容地址存储。

### 13.3.2 规范化与浮点陷阱

JSON 中 `1`, `1.0` 和 `1e0` 在语义上可能相同，在字节上却不同。浮点 NaN、负零、超大整数和 Unicode 转义还会造成跨语言差异。建议：

- 对控制面记录采用明确的 UTF-8、LF、无 BOM 和递归键排序
- 数值字段规定精度或改用十进制字符串；禁止把 NaN 放进必须跨语言解析的 manifest
- 路径统一使用 POSIX 分隔符，禁止依赖本地大小写和时区
- 对二进制模型哈希原始字节，不经压缩或解压重排
- 保存验证器版本，避免未来规则变化被误认为内容变化

### 13.3.3 大型 manifest 的分层

亿级样本不适合一个几百 GB 的 JSON。可以采用分层 Merkle 结构：每个分片有局部 manifest 和摘要，顶层只保存分片摘要、计数和排序规则。Merkle 父节点由子摘要按固定顺序拼接后哈希：

\[
 h_{parent}=H(h_0\Vert h_1\Vert\cdots\Vert h_{k-1}),
\]

其中 `||` 代表明确编码的连接，而不是字符串直接相连。验证器可以只下载一条路径来证明某个对象属于快照。分层结构还方便并行生成和局部重试，但要记录叶子顺序、空树表示和重复叶子的规则。

## 13.4 注册表：状态、兼容性与责任

### 13.4.1 注册表不是文件浏览器

注册表（registry）至少承担三种职责：索引工件、维护状态、执行策略。对象存储保存字节，注册表保存“哪个工件在什么状态、允许谁使用、由什么输入生成”。一个模型注册表条目应包含：

```text
name: assistant-small
version: 2.3.1
artifact_digest: sha256:...
stage: candidate -> canary -> production
compatibility: tokenizer=v7, schema=chat.v3, runtime>=2.1
quality: eval_report=sha256:..., gates=pass
provenance: run=sha256:..., source=git:...
owners: team-ml-platform
retention: 2 years
```

状态转换应由有限状态机约束，而不是让任意脚本改字符串。典型流程为 `draft -> validated -> candidate -> canary -> production -> retired/revoked`。每次转换产生审计事件，包含旧状态、新状态、操作者、时间、理由和证据摘要。

### 13.4.2 兼容性矩阵

模型版本自身不足以描述能否加载。至少建立以下矩阵：

| 工件 | 必须匹配 | 可迁移 | 不兼容信号 |
| --- | --- | --- | --- |
| checkpoint | 模型架构、参数命名、dtype、词表大小 | 位置编码、优化器状态可选 | missing/unexpected keys、shape mismatch |
| tokenizer | 词表、特殊 token、模板 | 映射 embedding 后的新词表 | tokenizer hash 不同 |
| dataset | schema、列类型、过滤规则 | 显式迁移脚本 | 缺列、单位变化、标签定义变化 |
| runtime | 算子、驱动、ABI、编译选项 | CPU/GPU fallback 经验证 | kernel 不存在、数值超差 |
| eval | 题集、评分器、聚合规则 | 重算报告 | 分母、阈值或 rubric 改变 |

“能加载”只证明字节形状大致相容，不代表语义相容。加载后要运行 smoke test：固定输入、特殊 token、空输入、长输入和拒答样例，并比较输出分布与容差。

### 13.4.3 注册表的并发与幂等

两个发布流水线可能同时把 `production` 指向不同版本。更新别名应使用条件写（例如比较当前 revision 的 CAS），失败后重新读取，不可盲目覆盖。注册新 digest 应幂等：重复提交同一 digest 返回同一条记录；同名不同 digest 必须显式创建新版本。网络重试不能产生两个相同语义却不同 ID 的发布事件。

## 13.5 Provenance 与 lineage：从结果反查输入

### 13.5.1 两个相近概念

- **Provenance（来源证明）**关注一次活动如何产生一个对象：谁、何时、用什么程序和输入生成了它。
- **Lineage（血缘）**关注对象之间的关系图：数据集到特征、特征到模型、模型到评估和发布的路径。

可以把活动图表示为有向无环图（DAG）：

```text
raw-2026-10-01 ──normalize@9c2e──> dataset-v4
                                   │
config-v12 ───────┐               │
tokenizer-v7 ─────┼─train@run-031──> checkpoint-step12000
code-9c2e1f7 ─────┘                       │
                                         ├─eval@report-88
                                         └─release assistant:2.3.1
```

节点是工件，边是活动；活动本身也应有 digest，包含命令摘要、环境、开始/结束时间、输入输出、日志位置和退出码。若允许循环（例如在线反馈更新），要把每次迭代切成时间版本，否则无法做确定性重放。

### 13.5.2 边的语义比“父子”更重要

同一工件可能有多个角色：数据集既是训练输入，也可能作为污染检查的对照；一个 tokenizer 可能被模型和评估器共同使用。推荐使用显式 `role`：`train_input`、`eval_input`、`code`、`policy`、`derived_from`、`tested_by`、`supersedes`。这样查询“哪些模型使用了某个包含 PII 的数据片段”时，可以区分直接输入和仅作对照的关系。

### 13.5.3 最小 provenance 事件

一次构建事件至少记录：

- `run_id` 和幂等键
- 规范命令及参数（敏感值脱敏）
- 输入工件 digest 列表和角色
- 输出工件 digest 列表
- 代码 revision、依赖锁文件、基础镜像和硬件摘要
- 随机种子、并行度、时区和 locale
- 开始/结束时间、退出码、日志和测试结果
- 运行者身份、工作区和审批引用

不要把 API token、完整环境变量或个人数据原样写入日志。秘密应引用安全存储中的版本 ID，且该引用本身也要受访问控制。审计可验证“使用了哪个秘密版本”，不需要暴露秘密值。

## 13.6 可复现构建：控制差异预算

### 13.6.1 可重复、可复现、可解释

三个词常被混用：

- **可重复（repeatable）**：同一环境、同一机器再次运行得到相同结果。
- **可复现（reproducible）**：另一台符合记录的环境也能得到等价结果。
- **可解释（traceable）**：即便数值有合理差异，也能说明差异来自哪一层。

深度学习中逐位相同往往很难，GPU 原子操作、库版本和浮点归约会引入差异。可复现契约应声明等级：`bitwise`、`numerically_equivalent`（指标在容差内）或 `behaviorally_equivalent`（关键测试与排序稳定）。不要把“loss 看起来差不多”写成逐位复现。

### 13.6.2 输入与工具链锁定

最小构建闭包包括：

1. 源码 commit 和未提交 diff 的摘要；
2. 依赖锁文件、包索引、编译器和链接器版本；
3. 基础镜像 digest、操作系统、CPU 指令集、GPU 驱动和 CUDA/cuDNN；
4. 配置、数据和 tokenizer manifest；
5. 随机种子、随机状态、采样顺序、并行度和线程环境变量；
6. 构建脚本、生成器版本和验证器版本。

安装依赖时锁住解析结果，而不是仅记录顶层包名。若依赖来自本地 wheel 或 git URL，记录文件摘要和来源；若包索引可变，使用带摘要的 lockfile 或内部镜像快照。不要在构建中执行 `pip install -U`、`apt-get update` 后不锁定版本，或从“最新”标签拉取基础镜像。

### 13.6.3 时间、排序和随机数

常见非确定性来源包括：

- 将当前时间写入模型文件或压缩包顺序；
- 遍历目录、字典、数据库查询没有显式排序；
- 多线程/多进程完成顺序影响聚合；
- Python、NumPy、框架和数据加载器使用不同 RNG；
- 哈希随机化、locale、时区、浮点舍入模式不同；
- GPU 非确定性 kernel、通信归约和混合精度缩放不同。

解决方法是把时间注入固定的 `SOURCE_DATE_EPOCH`，按字节或业务键排序，使用显式 tie-breaker，保存每个 worker 的 seed 和游标，设置 locale=`C`、时区=`UTC`，对非确定性算子做黑名单或容差测试。并行性能与逐位确定性往往有取舍，必须把取舍写入实验契约。

### 13.6.4 构建缓存与可证明命中

缓存不是魔法。每个缓存键应覆盖所有影响输出的输入：源码 digest、配置 digest、工具链 digest、输入 manifest、目标平台和生成器版本。若键遗漏了环境变量，缓存命中会提供错误的旧工件。缓存返回时仍需校验输出 digest，并记录 `cache_hit=true/false`。对有副作用的步骤（上传、发布、删除）不要把缓存命中当作已执行；把“生成工件”和“登记/发布”分成两个幂等阶段。

## 13.7 数据集与模型版本协同

### 13.7.1 独立版本会遮蔽兼容性

数据集 `v5`、模型 `v2` 和 tokenizer `v7` 各自递增并不代表组合可用。训练运行应有一个组合 manifest：

```yaml
run_id: run-031
code: git:9c2e1f7
config: sha256:cfg12
inputs:
  dataset: sha256:data55
  tokenizer: sha256:tok07
  init_checkpoint: sha256:ckpt00
outputs:
  final_checkpoint: sha256:ckpt31
  eval_report: sha256:eval88
compatibility:
  schema: reviews.v4
  tokenizer_template: chat.v3
  runtime: image@sha256:img2
```

组合 manifest 的 digest 是一次实验的稳定 ID。重新运行同一个组合但更换硬件，应创建新 run 记录并注明复现等级，而不是覆盖旧结果。

### 13.7.2 Schema 演进与迁移

数据 schema 变化可分为：

- **向后兼容**：增加可选列、放宽约束；旧消费者仍能读取，但要更新统计和 manifest。
- **需要迁移**：改列名、单位或标签定义；提供版本化迁移脚本，并保留迁移前后计数和抽样 diff。
- **破坏性变化**：删除列、改变语义、重标注；创建全新数据集 lineage，禁止伪装成小版本更新。

迁移脚本本身是代码工件，输入和输出都要哈希。迁移后的数据不能只记录“由 v4 升到 v5”，要记录规则、异常行处理、丢弃计数和人工批准。对于训练标签，哪怕列类型不变，只要标注指南变化也应视为语义破坏性变化。

### 13.7.3 检查点与恢复点

模型 checkpoint 不应只保存权重。为了从中断处继续，至少考虑：模型参数、优化器和 scheduler 状态、全局 step、数据游标、每 worker 的 RNG、梯度缩放器、配置 digest、代码和环境摘要。为了发布推理模型，优化器状态可以裁剪，但要生成新工件并标明 `derived_from`。恢复训练时先验证输入组合与原 run 兼容，再加载；禁止静默跳过缺失键。

## 13.8 发布、回滚与审计

### 13.8.1 发布前质量门

发布门禁至少包括：

1. manifest 完整性和所有对象 digest 校验；
2. provenance 图闭合，关键输入未使用可变别名；
3. 依赖和镜像有锁定摘要，漏洞/许可证扫描通过或有批准例外；
4. 单元、加载、推理 smoke test 和关键 benchmark 通过；
5. 与基线比较的质量、延迟、内存、成本和安全指标在阈值内；
6. 访问策略、数据分类、保留期和撤回路径已配置；
7. 发布别名更新采用条件写并产生审计事件。

门禁结果应作为评估工件保存，而不是只在 CI 控制台显示。门禁脚本升级后，旧报告仍能按验证器版本解释。

### 13.8.2 回滚不是“换回 latest”

正确的回滚流程：

1. 冻结当前别名更新，记录事件编号和影响范围；
2. 读取审计日志，确定上一个已验证的不可变版本；
3. 校验该版本的 digest、签名、兼容性和最小 smoke test；
4. 用 CAS 将别名从当前 digest 原子切换到目标 digest；
5. 监测错误率、延迟和业务指标，确认恢复；
6. 标记被撤回版本，保留其字节和证据；
7. 创建事故报告，说明触发器、受影响工件和防复发行动。

如果模型改变了数据写入 schema，单纯回滚模型可能仍无法读取新数据；应把模型、特征和服务配置作为发布单元，或提供向后兼容层。回滚训练不等于回滚已经写入数据库的预测结果，后者需要独立的数据修复计划。

### 13.8.3 审计日志的最小属性

审计事件要追加写、带单调序列号或链式摘要，并限制删除权限。每条事件包含：事件类型、目标工件、前后状态、操作者、时间、原因、审批、客户端/工作流版本和证据 digest。日志本身可按日封存并签名。注意“谁读取过模型输入”与“谁发布了模型”是两种不同事件，访问审计不能只记录写操作。

### 13.8.4 取舍与被拒绝的替代方案

### 13.8.5 审计查询、回放与影响范围

发布完成后，最常见的调查问题不是“文件还在吗”，而是“这个服务在某段时间到底使用了哪组输入”。因此 registry 需要同时支持正向和反向查询。正向查询从一次 run 出发，列出代码、配置、数据、tokenizer、环境、checkpoint、评估和别名；反向查询从一个数据对象或策略版本出发，列出所有下游模型、报告、服务和时间窗口。两类查询都必须基于不可变摘要和带角色的关系，不能只在名称字段上模糊匹配。

回放（replay）应分成三个层级。第一层是**元数据回放**：验证 manifest、provenance 和审计事件能否重建时间线，不需要重新执行训练。第二层是**构建回放**：在隔离环境重新生成模型或报告，比较输出 digest、指标和允许的差异预算。第三层是**行为回放**：把固定输入送入发布包，比较输出、延迟、拒答和安全门结果。调查先从低成本的元数据层开始，只有证据不足或质量影响重大时才升级到构建和行为层。这样可以避免为了确认一个别名错误而重新消耗整轮训练资源。

影响范围分析不能只看模型直接输入。一个含敏感字段的原始分片可能经过规范化、去重、采样、特征聚合和缓存，最终影响多个数据快照和 checkpoint。lineage 边应保存活动的时间范围、过滤规则和抽样比例，使查询可以给出保守上界与已确认集合。若某一步没有完整血缘，应把结果标成未知，而不是把未知当成零影响。对于线上服务，还要结合部署时间、实例启动日志和缓存预热时间，确定哪些请求窗口可能使用了受影响 digest。

审计回放也会遇到日志不完整、事件乱序和时钟漂移。事件应使用单调序列号或链式摘要排序，时间戳只作辅助；跨机器时间差应记录时区和时钟同步状态。发现缺口时，保留原始日志并创建更正事件，不要重写历史行。查询工具输出的每个结论都应带证据摘要、验证器版本和查询时间，方便另一位审查者独立复核。

将这些查询纳入日常演练：每次发布随机抽取一个旧版本，验证能否在只读权限下追到输入；每季度模拟撤回一个数据分片，检查是否能列出受影响模型并阻断新发布；每次 registry 或 schema 升级，先在副本上回放历史事件。演练失败本身就是质量门结果，应阻止把新的可变别名推进生产。

工件系统没有脱离上下文的“最佳”设计，以下选择应写进架构决策记录：

- **Git 大文件 vs 对象存储 + registry**：Git 提供熟悉的审查和分支，但不适合海量样本和高吞吐 checkpoint；对象存储适合大对象，却需要额外的权限、生命周期和审计。小规模教学可用 Git LFS，生产通常把 Git commit 作为代码身份、registry digest 作为数据和模型身份。
- **平面 manifest vs Merkle manifest**：平面 JSON 容易阅读、便于人工 diff；对象数达到百万级后，Merkle 分层可并行验证和局部证明，但实现复杂，必须固定叶子顺序和空树规则。
- **数据库注册表 vs 文件型注册表**：SQLite/关系数据库擅长 CAS、约束和查询；纯 JSONL 易于离线复制和灾备，却需要自己处理并发、唯一性和索引。不能因为 JSONL 简单就省略锁和审计。
- **逐位确定性 vs 吞吐**：关闭所有非确定性 kernel 可以增强复现，却可能降低 GPU 利用率。先确定结论所需的复现等级，再决定是否支付性能代价；不要为了漂亮的 hash 把生产吞吐砍掉而没有测量。
- **自动清理 vs 保守保留**：自动删除降低成本，却可能破坏事故调查和合规响应。先做逻辑撤回和归档，确认影响范围后再执行物理删除；涉及个人数据时由专门流程批准。

本节是设计判断而非某个平台的默认行为。目标是让团队能解释为何选择某个边界、测量了什么代价，以及失败时如何恢复。

## 13.9 CPU-only 实验：标准库构建并验证 artifact manifest

**实验卡（可复现记录）**：级别为 L1（单进程、单机、无网络）；基线是“直接读取目录并信任文件名”，干预是“对象哈希 + 规范 manifest + provenance + 审计事件”；软件为 Python 3.10+ 标准库，硬件为任意 x86/ARM CPU，预计运行时间小于 1 秒、成本为零；输入是三个小文本工件，输出是 manifest、模型占位文件、评估报告和 JSONL 审计；清理方式是删除临时目录，`--keep` 仅用于检查中间文件；若平台没有 Python 3.10，可在 3.9 上移除 `list[...]` 等新式注解，或把实验改成伪代码。验收标准是：健康快照验证无错误，等长篡改至少报告 digest 错误，未登记文件报告 unlisted，配置漂移产生不同摘要，任何验证失败都不改变已发布指针。以下代码是教学实现，不能替代带权限、签名、远端条件写和不可篡改日志的生产 registry。

[测量] 下面实验只用 Python 3.10+ 标准库，模拟一个小型训练流水线：创建不可变输入文件，计算对象和 manifest 摘要，写入 provenance，验证全部哈希，然后故意注入篡改、缺失和配置漂移。实验目标不是替代真实 registry，而是让读者看到工件契约中哪些字段应当被机器检查。

### 13.9.1 完整脚本

将以下内容保存为 `ch13_artifact_manifest.py` 后运行。脚本不会联网，也不会写入项目目录之外的临时目录。

```python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    """稳定 JSON：键排序、UTF-8、无多余空白。"""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 教学实现：先写临时文件，再用同文件系统替换，避免半文件。
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha256_bytes(data)


def write_json(path: Path, value: Any) -> str:
    return write_bytes(path, canonical_json(value))


@dataclass(frozen=True)
class ObjectRecord:
    path: str
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256,
                "bytes": self.bytes}


def scan_objects(root: Path, exclude: set[str]) -> list[ObjectRecord]:
    records: list[ObjectRecord] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in exclude or rel.endswith(".tmp"):
            continue
        records.append(ObjectRecord(rel, sha256_file(path), path.stat().st_size))
    return records


def make_manifest(root: Path, config_digest: str) -> dict[str, Any]:
    objects = scan_objects(root, {"manifest.json", "provenance.json"})
    body = {
        "manifest_version": 1,
        "snapshot": "toy-run-001",
        "ordering": {"mode": "lexicographic", "locale": "C"},
        "config_digest": "sha256:" + config_digest,
        "objects": [r.as_dict() for r in objects],
        "totals": {
            "objects": len(objects),
            "bytes": sum(r.bytes for r in objects),
        },
    }
    body_bytes = canonical_json(body)
    body["manifest_digest"] = "sha256:" + sha256_bytes(body_bytes)
    return body


def verify_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    listed = {item["path"]: item for item in manifest.get("objects", [])}
    actual = {r.path: r for r in scan_objects(root, {"manifest.json", "provenance.json"})}
    for rel, item in listed.items():
        path = root / rel
        if not path.exists():
            errors.append(f"missing:{rel}")
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != item["sha256"]:
            errors.append(f"digest:{rel}")
        if size != item["bytes"]:
            errors.append(f"size:{rel}")
    for rel in sorted(set(actual) - set(listed)):
        errors.append(f"unlisted:{rel}")
    expected_total = sum(x["bytes"] for x in listed.values()
                         if (root / x["path"]).exists())
    if expected_total != manifest["totals"]["bytes"]:
        errors.append("total_bytes")
    if len(listed) != manifest["totals"]["objects"]:
        errors.append("total_objects")
    # 验证 manifest_digest：去掉摘要字段后重新规范化。
    unsigned = dict(manifest)
    declared = unsigned.pop("manifest_digest", None)
    actual_digest = "sha256:" + sha256_bytes(canonical_json(unsigned))
    if declared != actual_digest:
        errors.append("manifest_digest")
    return errors


def append_event(path: Path, event: dict[str, Any]) -> None:
    """追加 JSONL 审计事件；事件不含秘密，只放摘要。"""
    with path.open("ab") as f:
        f.write(canonical_json(event))


def run_demo(keep: bool = False) -> None:
    temp = Path(tempfile.mkdtemp(prefix="ch13-artifact-"))
    print("root", temp)
    try:
        inputs = temp / "inputs"
        outputs = temp / "outputs"
        registry = temp / "registry"
        inputs.mkdir(); outputs.mkdir(); registry.mkdir()

        config = {
            "schema": "train-config.v1",
            "seed": 17,
            "learning_rate": "0.0001",
            "dataset": "dataset:toy@1",
        }
        config_digest = write_json(inputs / "config.json", config)
        write_bytes(inputs / "data.txt", "alpha\nbeta\ngamma\n".encode())
        write_bytes(inputs / "tokenizer.txt", b"<pad>\n<eos>\na\nb\n")

        manifest = make_manifest(inputs, config_digest)
        write_json(inputs / "manifest.json", manifest)
        # 把 manifest 作为输入快照的不可变记录复制到注册目录。
        shutil.copy2(inputs / "manifest.json", registry / "manifest.json")

        provenance = {
            "run_id": "run-001",
            "activity": "toy-build",
            "started_at": "2026-10-05T00:00:00Z",
            "code": "git:deadbeef",
            "inputs": [manifest["manifest_digest"]],
            "outputs": [],
            "status": "running",
        }
        write_json(registry / "provenance.json", provenance)

        model_bytes = b"toy-model\n" + manifest["manifest_digest"].encode()
        model_digest = write_bytes(outputs / "model.bin", model_bytes)
        report = {
            "model": "sha256:" + model_digest,
            "metric": {"toy_accuracy": 1.0},
            "tested_manifest": manifest["manifest_digest"],
        }
        report_digest = write_json(outputs / "report.json", report)
        provenance["outputs"] = ["sha256:" + model_digest,
                                  "sha256:" + report_digest]
        provenance["status"] = "succeeded"
        write_json(registry / "provenance.json", provenance)
        append_event(registry / "audit.jsonl", {
            "event": "publish", "alias": "toy/latest",
            "target": "sha256:" + model_digest,
            "actor": "demo", "reason": "smoke-test-pass",
        })

        loaded = json.loads((inputs / "manifest.json").read_text())
        print("initial_verify", verify_manifest(inputs, loaded))
        print("manifest_digest", loaded["manifest_digest"])
        print("model_digest", model_digest)

        # 失败案例 A：篡改内容，文件摘要不再匹配。
        (inputs / "data.txt").write_text("alpha\nBETA\ngamma\n")
        print("after_tamper", verify_manifest(inputs, loaded))
        (inputs / "data.txt").write_bytes(b"alpha\nbeta\ngamma\n")
        print("after_restore", verify_manifest(inputs, loaded))

        # 失败案例 B：新增文件但未更新 manifest，出现 unlisted。
        write_bytes(inputs / "debug.log", b"not-in-snapshot")
        print("after_unlisted", verify_manifest(inputs, loaded))
        (inputs / "debug.log").unlink()

        # 失败案例 C：配置漂移；同一 run 的 config digest 改变。
        drift = dict(config)
        drift["learning_rate"] = "0.0002"
        drift_digest = write_json(inputs / "config-drift.json", drift)
        print("config_drift", drift_digest != config_digest)
    finally:
        if keep:
            print("kept", temp)
        else:
            shutil.rmtree(temp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    run_demo(args.keep)
```

### 13.9.2 两阶段提交、CURRENT 指针与回滚

上面的脚本验证了内容，却还没有模拟发布时的原子指针。生产对象存储不能假设本地 `os.replace`，但可以先理解本地语义，再映射为带版本条件的对象写。下面片段把快照目录视为已完成的不可变工件：消费者只读取 `manifest.done` 存在的快照，`CURRENT` 是可变指针；本地片段用 expected revision 做冲突检查，真正多进程或远端 registry 必须使用数据库事务或条件写（CAS），回滚只移动指针而不删除新版本。

```python
import json, os, tempfile
from pathlib import Path

def canon(x):
    return (json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode()

def sha_file(path):
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def append_audit(root, action, snapshot, previous, reason):
    audit = root / "audit.jsonl"
    seq = sum(1 for _ in audit.open()) + 1 if audit.exists() else 1
    event = {"seq": seq, "action": action, "snapshot": snapshot,
             "previous": previous, "reason": reason}
    with audit.open("ab") as f:
        f.write(canon(event))

def read_current(root):
    current = root / "CURRENT"
    if not current.exists():
        return None
    pointer = json.loads(current.read_text())
    done = root / pointer["snapshot"] / "manifest.done"
    if not done.is_file() or sha_file(done) != pointer["manifest_sha256"]:
        raise RuntimeError("CURRENT digest mismatch")
    manifest = json.loads(done.read_text())
    model = root / pointer["snapshot"] / "model.bin"
    if not model.is_file() or sha_file(model) != manifest["sha256"]:
        raise RuntimeError("artifact hash mismatch")
    return pointer["snapshot"]

def publish(root, snapshot, reason, action="publish", expected_previous=None):
    done = root / snapshot / "manifest.done"
    if not done.is_file():
        raise RuntimeError("uncommitted snapshot")
    previous = read_current(root)
    if expected_previous is not None and previous != expected_previous:
        raise RuntimeError("revision conflict")
    pointer = {"snapshot": snapshot, "manifest_sha256": sha_file(done)}
    tmp = root / ".CURRENT.tmp"
    tmp.write_bytes(canon(pointer))
    os.replace(tmp, root / "CURRENT")  # 单进程教学；远端用条件写/CAS替代
    append_audit(root, action, snapshot, previous, reason)

# 运行示例：先创建两个已经校验通过的快照目录
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    for name, body in [("v1", b"model-v1"), ("v2", b"model-v2")]:
        snap = root / name; snap.mkdir()
        (snap / "model.bin").write_bytes(body)
        (snap / "manifest.done").write_bytes(canon({"snapshot": name,
                                                    "sha256": sha_file(snap / "model.bin")}))
    publish(root, "v1", "initial")
    publish(root, "v2", "quality gate pass", expected_previous="v1")
    print("current_before_rollback:", read_current(root))
    (root / "v2" / "model.bin").write_bytes(b"corrupt-v2")
    try:
        read_current(root)
    except RuntimeError as e:
        print("tamper_rejected:", e)
    # 修复仅用于继续演示；生产应撤回并重建新 digest，而不是原地修复
    (root / "v2" / "model.bin").write_bytes(b"model-v2")
    publish(root, "v1", "rollback after online error", action="rollback", expected_previous="v2")
    print("current_after_rollback:", read_current(root))
    print((root / "audit.jsonl").read_text(), end="")
```

预期输出包含 `current_before_rollback: v2`、`tamper_rejected: artifact hash mismatch`、`current_after_rollback: v1`，以及三条按序号排列的 publish、publish、rollback 事件。片段把 `manifest.done` 中的模型摘要与模型文件交叉验证；完整实现还应在验证器中检查对象列表、总计和 manifest 摘要，并在快照目录创建阶段使用 `manifest.intent -> manifest.done -> 原子 rename` 的两阶段提交。验证失败、崩溃或只留下 intent 时，`CURRENT` 必须保持旧值。本地 `os.replace` 只保证一次替换的原子可见性，不能阻止两个进程同时读到同一旧值；生产实现应把 expected revision 放进事务条件，并在条件失败时重新读取再决定是否重试。

### 13.9.3 运行、预期输出与解释

在 Python 3.10+ 下运行：

```text
$ python ch13_artifact_manifest.py
root /tmp/ch13-artifact-xxxx
initial_verify []
manifest_digest sha256:...
model_digest ...
after_tamper ['digest:data.txt']
after_restore []
after_unlisted ['unlisted:debug.log']
config_drift True
```

摘要尾部会因临时目录和实现细节而不同，但错误类别应一致。实验展示四个要点：

1. manifest 验证的是集合和总计，不是“目录存在”；
2. 内容被篡改时，即使文件名不变也会失败；
3. 新增未登记文件会暴露“扫描目录即数据集”的隐患；
4. 配置变化必须产生新摘要，不可用旧模型的 run_id 掩盖。

### 13.9.4 从 clean checkout 重放一个确定性 toy 结果

课程要求的“从干净检出重现模型结果”可以用一个极小的确定性计算来验证工件闭包。将脚本和输入 manifest 复制到新的空目录，只保留下列代码，禁止读取当前时间或未列出的文件：

```python
import hashlib, json
from pathlib import Path

root = Path(".")
manifest = json.loads((root / "inputs/manifest.json").read_text())
config = (root / "inputs/config.json").read_bytes()
# toy 模型结果：把输入摘要和配置摘要映射为 [0, 1) 的可重复分数
key = (manifest["manifest_digest"] + hashlib.sha256(config).hexdigest()).encode()
score = int(hashlib.sha256(key).hexdigest()[:8], 16) / 2**32
print("manifest", manifest["manifest_digest"])
print("toy_score", f"{score:.8f}")
```

先运行 `python -I reproduce.py`，再删除整个工作目录，从同一 commit 重新检出并重建输入，第二次的 `manifest` 和 `toy_score` 应逐字相同。故意把对象列表排序改为文件系统原顺序、改变配置键顺序但不做规范化，或偷偷读取 `time.time()`，即可观察差异。这个 toy 分数不是模型质量证据；它只证明“代码 + 配置 + 输入 manifest”能形成可回放的确定性闭包。真实训练还需把依赖、硬件、随机状态和容差报告加入 provenance。

### 13.9.5 自审与扩展

先把 `manifest.json` 的一个键顺序手工调整，再运行验证。由于读取后验证会重算摘要，应该出现 `manifest_digest` 错误；这说明摘要绑定了规范化表示。接着将 `scan_objects` 的排序改为 `os.scandir` 原顺序，重复运行并比较 manifest。若文件系统返回顺序变化，摘要就会漂移，证明显式排序是可复现构建的一部分。

可选扩展包括：

- 给每条对象增加 `media_type`、行数和 schema 摘要，并验证统计；
- 将审计事件改为链式哈希，检测日志中间删除或重排；
- 为模型创建 `aliases.json`，用文件锁或 CAS 模拟原子别名更新；
- 添加 `parent_manifest`，构建增量快照并计算新增、删除、修改集合；
- 用 `unittest` 写 6 个测试：空目录、缺失文件、篡改、未列文件、重复路径和错误总计；
- 把 `config.json` 复制到另一目录，改变换行或键顺序，比较原始哈希和规范化哈希的差异。

生产实现还需要防止 TOCTOU（检查后被替换）：先在临时目录构建，再对冻结目录做一次最终扫描和原子发布；读取对象时再次校验摘要，避免只验证上传前的文件。

安全强化清单：本实验为了教学简短，假定 manifest 由本地脚本生成，尚未把它当作恶意输入。生产验证器应拒绝绝对路径、包含 `..` 的路径、重复路径、符号链接和设备文件，只允许路径落在快照根目录内；应验证 manifest schema、版本、排序声明、总计、`config_digest` 对应的文件存在，并在读取后重新计算所有摘要。若对象存储支持版本 ID，摘要之外还要绑定版本和加密校验信息，防止同一 key 被替换。

验证和发布之间存在 TOCTOU 窗口：攻击者或错误任务可能在哈希后替换文件。稳健做法是先把输入复制到只读暂存区，对句柄或冻结版本做最终扫描，再生成 `manifest.done`；消费者打开文件后仍应校验摘要，不能相信上传阶段的结果。对于大对象，分块哈希和 Merkle 证明可以降低重复读取，但必须定义块大小、尾块编码和版本。

审计文件也不能只用“行数加一”当序列号。多进程追加可能重号或交错，崩溃可能留下半行；生产实现应使用带约束的数据库、单写者队列或支持条件追加的日志服务，记录链式摘要、请求 ID、操作者和验证器版本，并通过 `fsync`/WORM/远端封存保证重启后可验证。示例的 `audit.jsonl` 仅用于观察事件顺序，不能证明不可抵赖性。

## 13.10 失败案例：工件看似成功，证据却断裂

### 案例一：`latest` 被覆盖，无法解释线上回归

团队每天把新模型写到 `models/assistant/latest/`，评估报告只记录这个路径。一次 tokenizer 修复后，线上准确率下降；调查时路径已指向新模型，旧权重和旧 tokenizer 被垃圾回收，无法重跑。根因是把可变别名当作版本。修复方式是每次发布写入内容地址和不可变版本，`latest` 只做别名，并在服务启动日志记录解析后的 digest。别名更新采用 CAS，审计事件保留前后值。

### 案例二：manifest 漏列隐藏文件，训练结果依赖工作目录

数据准备脚本扫描 `*.jsonl`，却把同目录下的 `stopwords.txt`、过滤规则和本地缓存作为隐式输入。另一台机器没有这些文件，样本数和词频改变。修复方式是把所有影响输出的规则、默认值和环境白名单纳入输入 manifest；扫描器应拒绝未登记的可读文件，或在构建前生成干净沙箱。

### 案例三：依赖浮动导致“同一 commit”不同模型

代码 commit 未变，但 `pip install torch` 拉到不同小版本，基础镜像的系统库也更新。模型指标相差 0.4 个百分点，团队误以为随机种子失效。实际上依赖和 ABI 没有锁定。修复方式是 lockfile 加包摘要、镜像 digest、构建时间和仓库快照；构建器在解析到不同 digest 时拒绝复用缓存。

### 案例四：数据修复覆盖原始分区，训练血缘无法回溯

工程师为修复错误标签，直接覆盖对象存储中的 `date=2026-09-18/part-003`。旧模型的 manifest 只保留路径，没有文件哈希，导致同一路径的字节发生变化。修复方式是 append-only 快照和 parent 链，修复产生新分区和新 digest，旧快照只读保留。若必须物理删除，先评估受影响模型和合规要求，并保留删除证明。

### 案例五：只回滚模型，没有回滚特征和服务 schema

新模型要求 `feature_v5`，服务同时把输入 schema 升级并写入新格式。模型出现异常后只把权重切回旧版，旧模型无法解析新特征，错误率反而更高。修复方式是把模型、tokenizer、特征 schema、服务配置作为兼容性单元，或者提供双读/双写过渡。回滚演练要包含真实数据路径，而不只是加载权重。

### 案例六：审计日志写在同一可变磁盘，事故后证据消失

发布脚本把 audit.jsonl 和模型放在同一工作区。磁盘损坏和清理任务同时删除两者，团队无法证明谁在何时发布了哪个 digest。修复方式是把审计事件追加写到受保护的远端或 WORM 存储，使用链式摘要或签名定期封存；访问日志和发布日志分离，最小权限限制清理。

## 13.11 理解检查（含答案）

### 检查 1：为什么 `latest` 不能作为训练输入的唯一标识？

**答案：** `latest` 是可变别名，可能在训练中途解析到不同内容，也不能解释历史结果。正确做法是解析别名后把不可变版本和内容 digest 写入 run manifest；训练消费者应直接读取该摘要或冻结快照。

### 检查 2：文件哈希和 manifest 哈希分别证明什么？

**答案：** 文件哈希证明某个路径在某一时刻的字节摘要；manifest 哈希还绑定对象集合、顺序、总计、schema、规则和父快照。只有前者时，漏列、增文件或排序变化可能悄悄改变训练输入。

### 检查 3：为什么“同一代码 commit”不等于可复现构建？

**答案：** 依赖、编译器、镜像、硬件、环境变量、随机状态、输入数据和时间戳都可能变化。可复现需要记录并锁定完整构建闭包，并声明逐位、数值等价或行为等价的目标等级。

### 检查 4：数据 schema 增加一个可选列，是否一定是破坏性版本？

**答案：** 不一定。若旧消费者忽略该列且语义不变，可向后兼容，但仍应更新 schema、统计和 manifest，并验证读取路径。如果列名、单位、标签定义或默认值语义改变，则需要迁移或新的主版本。

### 检查 5：为什么回滚模型别名前必须做兼容性和 smoke test？

**答案：** 旧模型可能依赖旧 tokenizer、特征 schema、算子或服务配置。仅切换权重无法保证输入输出语义兼容，甚至会扩大事故。回滚前验证 digest、签名、依赖和固定样例，回滚后监测错误率与业务指标。

### 检查 6：审计日志应该记录秘密值以便“完整复现”吗？

**答案：** 不应该。秘密值属于高敏感凭据，写入日志会扩大泄露面。应记录秘密版本的不可逆引用、访问主体和时间，使用安全存储在受控环境重放；若任务涉及凭据变更或跨边界传输，需人工交接和专门审批。

## 13.12 练习

1. **配置冻结器**：设计一个只允许声明字段的配置 schema，拒绝未知键；实现默认值展开、单位检查和规范 JSON 摘要。测试命令行顺序、Unicode 键和 `1`/`1.0` 的行为。
2. **增量 manifest**：在实验脚本基础上实现 parent manifest。给定新旧快照，输出 added、removed、modified 三个集合，并验证总计差异与父摘要一致。
3. **别名 CAS**：用标准库 `sqlite3` 或文件锁模拟注册表。两个线程同时更新 `production`，只有期望 revision 匹配者成功；记录失败者的重试与审计事件。
4. **可复现压缩包**：创建包含源码的 zip/tar 工件。固定文件排序、mtime、权限和压缩参数，比较两次 digest；再故意写入当前时间，观察差异。
5. **血缘查询**：把 manifest 和 provenance 事件存成 JSONL，写一个反向查询：给定某个数据对象 digest，列出所有训练 run、checkpoint、评估报告和发布别名。
6. **兼容性门禁**：为一个 toy checkpoint 定义架构、词表和输入 schema 版本。实现加载前检查，分别制造缺键、shape 不匹配、tokenizer hash 不同和 schema 迁移缺失四种失败。
7. **回滚演练**：模拟两个模型版本、一个特征 schema 和一个别名。先发布 v2，再注入错误率升高，按“冻结—校验—CAS 切换—监测—封存”步骤生成审计文件。
8. **攻击面评审**：列出构建过程中的路径穿越、符号链接、manifest 注入、日志泄密、依赖投毒和重放攻击，给每项指定检测、阻断和升级责任。

### 13.12.1 术语小表（快速参考）

- **工件（artifact）**：可以独立寻址、校验和传递的输入或输出对象。权重、配置、数据快照、评估报告和构建证明都可以是工件。
- **内容地址（content address）**：由内容摘要命名对象，例如 `sha256:...`。它解决“这是什么字节”，不自动解决“谁生产、是否可信”。
- **Manifest**：描述一组对象、顺序、规则、总计和父快照的机器可读记录。它把“目录里有什么”变成可验证的集合契约。
- **Registry**：保存工件索引、状态、兼容性和发布责任的控制面。它可以引用对象存储，但不应把路径存在当成质量证明。
- **Provenance**：描述某次活动如何由输入产生输出的来源证据，通常含代码、环境、操作者和时间。
- **Lineage**：跨多个活动连接数据、模型、评估和发布的关系图，可用于正向重建和反向影响分析。
- **不可变版本**：发布后不再改变指向或字节的版本。若需要修复，创建新版本并通过 `supersedes`、`parent` 等关系连接。
- **可变别名**：如 `latest`、`canary`、`production`，可以原子移动，但每次解析结果必须写入运行记录。
- **CAS（compare-and-swap）**：仅当当前 revision 与预期一致时才更新，防止并发发布覆盖彼此。
- **复现等级**：逐位相同、数值在容差内相同、或关键行为相同。等级越强，通常需要更多性能和工程成本。
- **两阶段提交**：先写入并验证暂存快照，再写 `done` 标记和当前指针。中途失败只留下可清理的暂存物，不影响上一个可信版本。

术语之间的关系可以用一句话记忆：manifest 说明集合，digest 说明内容，registry 说明状态，provenance 说明生成，lineage 说明影响，policy 说明谁能读、写、发布和删除。

## 13.13 来源地图与延伸阅读

以下链接用于核对规范和工具边界（访问日期：2026-10-05）。它们不会替代组织的安全、隐私、法务和发布政策；实现细节应按部署版本重新验证。

- [SLSA v1.0](https://slsa.dev/spec/v1.0/)：软件供应链等级、来源证明和构建要求。
- [in-toto 规范](https://in-toto.io/)：以步骤和布局描述供应链完整性，适合把构建活动绑定到输入输出。
- [in-toto: Providing Farm-to-Table Guarantees for Bits and Bytes](https://www.usenix.org/conference/usenixsecurity19/presentation/torres-arias)：USENIX Security 2019 论文，解释布局、步骤和链接元数据如何抵御供应链篡改；论文中的威胁模型不能直接替代本组织的风险评估。
- [Sigstore](https://www.sigstore.dev/)：短期签名、透明日志和无密钥工作流的生态说明。
- [SPDX 规范](https://spdx.dev/specifications/)：软件包、许可证和组件清单（SBOM）格式。
- [CycloneDX](https://cyclonedx.org/specification/overview/)：SBOM 与依赖关系交换格式。
- [OCI Image/Distribution 规范](https://opencontainers.org/)：镜像内容寻址、清单、层和分发语义。
- [W3C PROV](https://www.w3.org/TR/prov-overview/)：通用 provenance 概念与实体—活动—代理模型。
- [OpenLineage](https://openlineage.io/docs/spec/)：作业、运行和数据集事件的 lineage 交换协议。
- [JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)：跨实现稳定 JSON 表示的规范。
- [NIST SP 800-57](https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final)：密钥管理生命周期与保护建议。
- [NIST SSDF SP 800-218](https://csrc.nist.gov/publications/detail/sp/800-218/final)：安全软件开发框架和供应链实践。
- [Reproducible Builds](https://reproducible-builds.org/docs/definition/)：构建可重复性、时间戳和环境差异的说明。
- [DVC 文档](https://dvc.org/doc) 与 [Git LFS](https://git-lfs.com/)：数据/模型大文件版本化的常见工作流；使用前核对远端和权限语义。
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)：模型版本、别名、阶段和注册表接口示例。
- [Hugging Face Hub 文档](https://huggingface.co/docs/hub/repositories-getting-started)：模型和数据仓库、revision 与大文件存储；生产使用需配置组织权限和审计。
- [OCI Artifacts](https://github.com/opencontainers/image-spec) 与 [ORAS](https://oras.land/)：把非容器工件放进 OCI registry 的清单和推送工具。
- [The Update Framework (TUF)](https://theupdateframework.io/docs/)：更新元数据、密钥轮换和回滚保护。

[设计判断] 来源地图的边界：上述规范说明格式和安全机制，不保证任何平台默认启用不可变存储、签名或审计，也不证明某个模型、数据集或本章示例的数值。请保存自己的原始 manifest、日志、签名和验证器版本。

## 13.14 安全边界与升级路径

1. **高敏感数据**：不要把密码、API token、完整个人资料或私钥写入配置、manifest、日志和 provenance。使用秘密管理器的版本引用；需要输入凭据时让用户在安全表单或人工交接中完成。
2. **供应链信任**：摘要只能证明“字节相同”，不能证明来源可信。对构建器、依赖、镜像和签名密钥建立信任根，验证签名和撤销状态，并防止重放旧但有效的签名。
3. **权限与隔离**：构建任务使用最小权限、短期身份和只读输入；发布、撤回、删除和密钥轮换分离角色。未授权的第三方指令不能改变 registry 状态。
4. **数据删除**：撤回别名不等于删除数据。涉及隐私请求、跨区域复制、备份或模型记忆时，升级到隐私/法务/安全负责人，记录范围和批准。
5. **可用性与灾备**：不可变对象需要多副本和定期恢复演练；若只在一个 registry 保存 digest，registry 故障会阻断恢复。备份本身也要有 manifest、加密和访问审计。
6. **数值差异**：不要把硬件差异造成的可接受浮点偏差误报成篡改，也不要用“容差通过”掩盖明显的语义回归。为指标、输出排序和安全测试分别定义阈值。
7. **审计隐私**：审计日志可能暴露项目名称、用户标识和数据路径。按用途做字段最小化、分层访问和保留期限；调查时提供经过脱敏的证据副本。
8. **升级路径**：当工件跨组织、跨云或进入高影响决策时，引入独立代码审查、双人批准、签名验证、SBOM、渗透测试和事故演练。标准库实验只能证明基本逻辑，不能替代这些控制。

## 13.15 章节完成标准

读者完成本章后，应能回答：

1. 每个训练输入、代码、配置、环境和输出的不可变标识是什么，哈希覆盖原始字节还是规范化表示？
2. manifest 是否定义对象集合、顺序、schema、规则、总计、父快照和验证器版本？
3. 注册表的版本、别名、状态、兼容性和审计事件如何区分，别名更新是否原子且可回放？
4. 从线上模型 digest 能否反查数据、tokenizer、代码、配置、运行环境和评估报告？
5. 构建中哪些差异允许数值容差，哪些差异必须阻断？随机状态、时间、排序和并行度是否有证据？
6. 数据 schema、标签语义、tokenizer 和模型 checkpoint 的版本变化是否有迁移脚本和兼容性测试？
7. 回滚是否覆盖模型、特征、服务配置和下游写入，还是只切换一个权重路径？
8. 逻辑撤回、归档、到期删除和合规擦除的权限、证据与影响范围是否不同？
9. 构建和发布中如何防止秘密泄露、依赖投毒、重放、路径穿越和审计日志篡改？
10. 当实验结论将用于高影响决策或处理高敏感数据时，谁负责升级审查，何时停止自动化？

## 13.16 小结：让每个结果都能被定位、验证和撤回

工件管理的核心不是多写几个 YAML 字段，而是为实验建立可验证的边界。配置、代码、数据、tokenizer、环境、checkpoint、评估和发布包都应有稳定身份；内容摘要、不可变版本和可变别名各司其职；manifest 把集合、顺序和规则冻结下来；registry 维护状态、兼容性和责任；provenance/lineage 让结果可以从线上反查到原始输入，也能从数据撤回请求定位受影响的模型。

可复现构建需要控制输入闭包、依赖、镜像、时间、排序、随机数、并行和硬件差异，并明确逐位、数值或行为等价的目标。数据和模型要用组合 manifest 协同版本，schema 变化通过迁移和兼容性矩阵表达。发布与回滚必须是有证据的状态转换，而不是覆盖 `latest`；审计日志和对象保留策略要能支撑事故调查、恢复和合规擦除。

最后，标准库实验提醒我们：一个文件被改动、漏列或配置漂移，就足以让“同一 run”失去意义。把这些检查接入训练、评估和部署的早期门禁，失败时保留原始证据，成功时记录可复现的摘要，才能让模型质量从一次性的分数变成可持续维护的工程事实。
