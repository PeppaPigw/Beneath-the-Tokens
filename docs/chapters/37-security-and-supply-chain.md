---
id: ch37-security-and-supply-chain
title: AI Infra 安全与供应链：模型、容器、依赖、机密与运行时隔离
slug: /chapters/37-security-and-supply-chain
description: 用威胁建模、SBOM、签名与 provenance、容器/Kubernetes 隔离、GPU 多租户、机密与网络策略建立可审计的 AI Infra 供应链防线
sidebar_position: 37
level: advanced
prerequisites:
  - ch13-artifact-management
  - ch17-kubernetes-gpu-orchestration
  - ch20-observability-debugging-incident-response
  - ch21-ai-reliability-engineering
  - ch22-ai-infra-security-privacy-supply-chain
  - ch33-storage-and-data-plane
  - ch36-observability-and-tracing
learning_objectives:
  - 能为模型、数据集、容器镜像、依赖、节点和租户边界绘制威胁模型与信任边界
  - 能解释模型/数据投毒、恶意权重、反序列化风险、依赖漏洞和构建流水线攻击的证据链
  - 能使用 digest、SBOM、签名、provenance、可复现构建和 admission policy 设计 release gate
  - 能在 Kubernetes/容器中落实 rootless、read-only rootfs、seccomp、capability drop、网络默认拒绝和 sandbox
  - 能区分独占 GPU、MIG、多进程共享与不隔离共享的泄漏面，并定义 GPU 多租户准入条件
  - 能安全处理 registry、对象存储、KMS、云 API 和推理服务机密，避免日志、镜像和 trace 泄漏
  - 能建立审计哈希链、事件保留、撤销、隔离、取证、回滚和复盘流程
  - 能运行 CPU-only toy lab，复现证据缺失、投毒、漏洞、运行时隔离、机密和网络策略故障
estimated_hours: 54
hardware: CPU-only toy lab; GPU/MIG、Kubernetes 和生产供应链结论必须在固定版本的隔离环境中复验
risk_level: L3
last_verified: 2026-10-07
---

# 第37章　AI Infra 安全与供应链：模型、容器、依赖、机密与运行时隔离

> AI 系统的攻击面不是一台“装着模型的 GPU 机器”。训练数据从对象存储进入预处理作业，权重经过转换、量化、打包、签名和镜像构建，随后由 Kubernetes 调度到共享节点，运行时还要访问 registry、KMS、向量库、队列、监控和用户请求。任何一个边界都可以改变最终执行的字节。安全工程的目标不是宣称“镜像安全”，而是让每一份被执行、被加载、被挂载和被上传的材料都有可验证的来源、完整性、最小权限、隔离边界和可追溯的处置动作。

本章使用“证据门 + 最小权限 + 可回滚响应”三条主线。证据门回答“这份模型、数据和依赖是不是我们批准的那一份”；最小权限回答“即使被利用，进程能够触及哪些资源”；响应回答“发现异常后如何冻结推广、撤销信任、保留证据并恢复服务”。CPU-only toy lab 只模拟策略与证据，不下载真实软件、不连接 Kubernetes、不使用 GPU，也不证明任何生产环境已经安全。

## 37.1 问题/边界：从“模型文件”扩展到完整供应链

### 37.1.1 AI 供应链的资产图

先列出必须保护和验证的资产，而不是直接写一份 Pod YAML：

- **数据资产**：原始样本、标注、过滤规则、去重索引、数据集 manifest、增量补丁、评测集和 canary 集；
- **模型资产**：训练 checkpoint、tokenizer、配置、量化表、LoRA/adapter、合并脚本、转换器和模型卡；
- **执行资产**：容器镜像、基础镜像、操作系统包、Python/Rust/Node 依赖、CUDA/NCCL/Triton 扩展、启动脚本；
- **基础设施资产**：节点内核、GPU 固件和驱动、Kubernetes API、控制器、registry、对象存储、构建 runner、缓存和签名服务；
- **身份与机密**：registry pull、对象存储、KMS 解密、云 API、数据库、队列、租户密钥和 webhook；
- **证据资产**：SBOM、provenance、签名、扫描结果、admission decision、运行时审计、trace、日志和取证快照。

资产图必须记录“谁可以写、谁可以读、谁可以批准、谁可以撤销”。例如，训练作业可以读取训练数据，却不应能覆盖生产 registry；构建 runner 可以推送候选 digest，却不应持有生产 KMS 解密权限；签名机器人可签署通过审核的 digest，却不应修改源码或扫描结果。

### 37.1.2 安全边界与非目标

本章覆盖离线训练/评测、模型注册与发布、镜像构建、Kubernetes admission、在线推理 worker、GPU 多租户、机密注入、网络 egress、依赖和事件响应。它不替代内容安全、隐私影响评估、出口管制、密码学实现审计或物理数据中心安全；也不把“使用某个扫描器”视为合规证明。扫描器的数据库、配置、忽略规则和版本同样属于供应链的一部分。

模型质量下降未必是安全事件，安全事件也未必立即造成质量下降。一个恶意依赖可能只窃取环境变量；一个被投毒的数据集可能让离线指标略微变好、线上触发特定后门；一个 GPU 侧信道可能不改模型输出，却泄露另一个租户的形状或时序。每种信号都要定义阈值、反证和升级路径，避免把单一分数当作真相。

### 37.1.3 威胁建模的最小字段

对每条数据流填写：资产、来源、消费者、信任级别、可写者、机密性、完整性、可用性、预期寿命、检测点和失效动作。使用 STRIDE、Kill Chain 或 MITRE ATLAS 作为词汇表均可，但必须落到可执行控制：

| 数据流 | 典型威胁 | 预防 | 检测/响应证据 |
| --- | --- | --- | --- |
| 数据集 manifest → 训练 | 增量投毒、标签翻转、时间窗污染 | 版本锁定、双人批准、不可变对象 | hash、抽样重放、canary loss、冻结发布 |
| checkpoint → 转换器 | 恶意序列化、任意代码执行、权重替换 | 使用安全格式、隔离转换、digest pin | 转换日志、沙箱审计、签名差异 |
| 源码/依赖 → 构建 | typosquat、依赖混淆、构建 runner 被劫持 | lockfile、私有镜像、构建隔离 | SBOM、provenance、重建比较 |
| 镜像 → 集群 | tag 劫持、基础镜像漏洞、恶意 entrypoint | digest、签名、admission policy | 准入拒绝、运行时 syscall 与镜像 ID |
| Pod → GPU/网络 | 逃逸、跨租户显存/时序泄漏、异常 egress | rootless、seccomp、MIG/独占、NetworkPolicy | XID、IOMMU、流量日志、终止隔离 |
| 服务 → 机密 | 环境变量/日志/trace 泄漏、过宽 token | 引用式 secret、短 TTL、最小 audience | secret access audit、轮换、吊销 |

威胁模型的边界要写出“攻击者已经获得什么”。供应链防守通常假定 registry 中可能存在一个合法格式但恶意的包，或构建过程被短暂控制；不能只假定“攻击者没有凭据”。

## 37.2 心智模型：四条链和三个信任根

### 37.2.1 四条链

把一次发布拆成四条可独立验证的链：

1. **内容链（data/model lineage）**：原始数据版本 → 清洗/过滤 → 训练配置 → checkpoint → 转换/量化 → 推理 artifact。每个节点记录输入 digest、代码 revision、参数和输出 digest。
2. **执行链（build/runtime）**：源码 commit → lockfile → builder image → 编译产物 → OCI manifest → Pod spec → node/GPU。这里要防止“源码签了但实际运行的是另一个 tag”。
3. **身份链（authorization）**：人或服务账号 → 签名 key → registry/KMS/集群角色 → 具体资源。授权要绑定 digest、namespace、租户和操作，而不是只绑定仓库名称。
4. **证据链（audit/response）**：检查结果 → admission decision → 部署/回滚 → 运行时事件 → 取证包 → 修复和撤销。证据必须能证明当时为什么允许或拒绝。

四条链相交处才是 release gate。例如，模型签名正确而 provenance 缺失，不能仅凭签名放行；镜像 SBOM 完整而运行时需要 privileged，仍应拒绝；数据 manifest 正确但 canary 失真，应进入 quarantine，而不是把风险转换成“低质量模型”。

### 37.2.2 三个信任根

- **密码学信任根**：签名验证 key、KMS/HSM、透明日志或不可变审计存储。它回答“字节由谁签过”；不回答签名者是否有权签这份内容。
- **构建信任根**：受控、可重建的 builder、基础镜像和工具链。它回答“产物怎样从源码得到”；不回答训练数据是否干净。
- **运行时信任根**：节点内核、容器 runtime、Kubernetes admission 和 GPU/IOMMU 边界。它回答“执行时能触及什么”；不回答 artifact 是否含后门。

信任根本身要轮换和审计。签名 key 泄漏时，必须有撤销/denylist 或重新发布策略；节点被攻破时，不能继续把该节点的日志当作唯一证据。把关键事件复制到隔离的写入端或不可变存储，并记录时间同步状态。

### 37.2.3 风险预算而非“零漏洞”

漏洞数量不能直接等价于风险。一个未修复的 critical 依赖如果只在离线、无网络、非 root 的转换沙箱中使用，与暴露管理端口的同一漏洞风险不同。不过“上下文降低风险”不等于忽略漏洞：应记录 CVSS/EPSS（若适用）、可达路径、是否可利用、补丁版本、临时缓解和到期时间。生产门禁可以规定 critical/high 且可达就阻断，medium 需要 owner 和截止日期，low 进入待处理队列；所有例外要有签名批准和自动过期。

## 37.3 机制：威胁建模到可执行控制

### 37.3.1 供应链攻击的典型路径

攻击者常利用“信任转换”而不是直接破解 GPU：

1. 在公共仓库发布名称相似的依赖（typosquat）或劫持过期 maintainer；
2. 在构建脚本、依赖安装 hook、模型转换器或 notebook 中植入代码；
3. 通过 tag、缓存 key 或 mutable latest 让构建取得未审查字节；
4. 生成看似正常的镜像，借助过宽推送权限进入 registry；
5. 准入只检查仓库名或漏洞数量，未检查 digest、签名与 provenance；
6. Pod 以 root/privileged 运行，读取节点 token、云凭据或其他租户设备；
7. 通过日志、trace、DNS、对象存储或出站 HTTPS 回传机密和模型信息；
8. 运营团队只回滚应用 tag，却没有撤销被盗的签名 key、缓存和长期 token。

防护点必须覆盖每一跳。只在最后一跳做镜像扫描，无法解释数据和源码从哪里来；只签署最终镜像，无法证明签名者检查了哪个训练数据版本。

### 37.3.2 digest、签名与 provenance

OCI tag 是人类可读指针，可能被移动；digest 是内容寻址标识，必须在部署和审计中记录。签名声明“某个主体认可 digest”，provenance 声明“该 digest 由哪个源码、构建器、参数和依赖产生”。两者都需要验证主体授权和时间窗口。

常见的实现组合包括 [Sigstore Cosign](https://docs.sigstore.dev/cosign/signing/signing_with_containers/)、[in-toto](https://in-toto.io/)、[SLSA](https://slsa.dev/spec/v1.0/)、[OCI Image Specification](https://github.com/opencontainers/image-spec) 和 Kubernetes admission webhook。策略示例：

```text
image.ref 必须是 registry.example/ai/*@sha256:<64 hex>
签名身份必须属于 release-bot@project.example
provenance.buildType 必须是受控 builder，sourceRevision 必须在主分支
SBOM 必须存在，critical/high 可达漏洞为 0，例外在 7 天内过期
模型 digest、tokenizer digest、镜像 digest 必须与 release manifest 相符
```

签名不应把秘密放进 annotation；签名 payload 和透明日志可能是长期证据，避免嵌入 prompt、用户 ID 或密钥。验证失败要输出具体 reason（digest、identity、provenance、SBOM、撤销），而不是只返回“unauthorized”。

### 37.3.3 SBOM 的边界和使用方式

[SPDX](https://spdx.dev/specifications/) 和 [CycloneDX](https://cyclonedx.org/specification/8.0/) 用于描述组件、版本、许可证、来源和关系。容器 SBOM 应覆盖 OS 包、语言依赖、编译出的共享库、CUDA/NCCL 绑定、模型转换工具和启动脚本。不要把“扫描器列出组件”误认为“所有组件可达性已分析”：SBOM 仍需要与运行时进程、import graph、动态加载器和网络权限结合。

SBOM 在三处生成和比对：源码/lockfile 阶段的预期清单，构建产物阶段的实际清单，运行时镜像解包后的独立重扫。三者不一致要解释：生成代码、系统库、静态链接、插件目录和缓存都可能导致差异。保留 SBOM 的 digest，并把其生成器版本、数据库快照和忽略规则写入 provenance。

### 37.3.4 模型格式和反序列化

权重下载、转换和加载是高风险边界。优先使用只表达张量和元数据的安全格式（例如 [safetensors](https://github.com/huggingface/safetensors)），对旧格式或自定义 pickle 转换放入无网络、无机密、只读输入、非 root、受 seccomp 约束的沙箱。不要为了“快速加载”打开任意对象反序列化；即使文件来自内部 bucket，也要先验证 digest 和来源。

转换工具要记录输入/输出 digest、工具版本、命令参数、环境和耗时；禁止工具写入源目录或通过动态 import 拉取网络包。转换后用张量 shape、dtype、范数、分片索引和固定 canary 做完整性检查。数值一致不证明没有隐蔽后门，但能捕获截断、分片错位和意外量化。

### 37.3.5 模型与数据投毒

投毒有三类常见信号：manifest/hash 漂移，样本或标签统计异常，canary 或触发器行为变化。检测器应同时看绝对阈值与相对基线：

- manifest hash 必须与批准版本匹配；增量数据要有父版本和签名；
- 标签分布、重复率、语言/来源比例和时间窗要与历史区间比较；
- 对受控 canary 运行固定评测，记录 loss、准确率、拒答率、触发器命中率和置信度；
- 将数据供应商、清洗代码和随机种子纳入 provenance；
- 异常先 quarantine，保留原始对象和检查日志，禁止覆盖“已批准”路径。

统计阈值不是证明无后门。攻击者可低于采样率或伪装为自然漂移；因此需要多来源抽样、人工复核、差分重训和线上 shadow 观察。模型输出的安全评估不能替代完整性验证。

### 37.3.6 依赖漏洞和构建隔离

锁文件只固定解析结果，不保证下载源可信，也不覆盖系统包和构建插件。构建流程至少分离：解析依赖、下载缓存、编译、测试、打包、签名。builder 默认无生产网络和云凭据；需要下载时通过允许列表代理，记录 URL、digest 和缓存命中。缓存 key 应包含 lockfile、平台、编译器和安全补丁版本，避免把一个项目的二进制喂给另一个项目。

扫描结果要分“存在”和“可达”：把 CVE 版本范围映射到实际组件和调用路径，避免只看数量。对 Python/Rust/Node 依赖同时扫描 wheels、源码包、native extension 和镜像层。供应商公告、OSV、NVD 等数据库有延迟和误报，需记录数据库更新时间；无法查询时不要把“扫描器无结果”写成“无漏洞”。

### 37.3.7 容器与 Kubernetes 隔离

最小 Pod 安全上下文应包括：非 root UID/GID、read-only root filesystem、drop all capabilities、`allowPrivilegeEscalation: false`、seccomp `RuntimeDefault` 或更严格配置、限制 CPU/内存/临时存储、禁止 hostPID/hostNetwork/hostIPC、禁止 hostPath，服务账号 token 只在确需时挂载。Namespace、ResourceQuota、LimitRange 和 Pod Security Admission 负责组织边界，但不能替代节点内核和容器 runtime 更新。

NetworkPolicy 默认拒绝 ingress/egress，再按 namespace、registry、DNS、KMS、模型存储和服务端口放行。策略要覆盖 IPv4/IPv6、DNS、sidecar 和 host network 例外；仅在应用层写 allowlist 不会阻断一个被攻破的进程访问云 metadata endpoint。生产中可结合 egress gateway、mTLS 和 DNS 日志，但日志同样要脱敏。

沙箱不是单一开关。gVisor、Kata Containers、microVM、seccomp、AppArmor/SELinux、no_new_privs 和用户命名空间各有成本与边界。需要 GPU 的 Pod 可能受设备插件、VFIO/IOMMU、MIG 和驱动 ioctl 限制；应在目标 workload 上验证功能、性能、崩溃回收和取证。禁用 seccomp 或给 privileged 只是“修复兼容性”的最后手段，必须有临时期限。

### 37.3.8 GPU 多租户

GPU 共享方式从隔离强到弱大致为：独占节点/设备、MIG 硬件分区、受控 time-slicing、MPS/多进程共享、无约束共享。隔离面包括显存地址空间、copy engine、L2/DRAM 争用、时钟和功耗、NVLink/PCIe、驱动 ioctl、错误重置和 side channel。MIG 提供硬件资源切片，但并不自动隔离网络、共享文件、日志或驱动漏洞；time-slicing 可能暴露时序和服务质量争用。

准入策略要把租户敏感级别和 GPU 模式绑定：高敏感租户使用独占或经验证的 MIG profile；禁止跨租户复用不清理的显存和 page cache；作业结束执行设备复位或安全清理；记录 parent GPU、instance UUID、Pod UID、租户和时间窗。遇到 XID、ECC、驱动异常或不可解释的性能干扰，隔离整张卡并保存现场，而不是只重启单个容器。

### 37.3.9 机密管理

机密的生命周期是创建、授权、注入、使用、轮换、吊销、取证和删除。Kubernetes Secret 只是 API 对象；需要 etcd 加密、RBAC、短 TTL、audience、namespace 限制和审计。优先在运行时通过 secrets manager 或 workload identity 获取短期 token，避免把长期 token 烘进镜像、环境变量、命令行、ConfigMap、模型 metadata、core dump、trace 或异常消息。

日志和遥测处理应有“不可逆脱敏”策略：只记录 secret 名称、版本和访问结果，不记录值；禁止把 HTTP header、Authorization、完整 URL、prompt 和对象存储签名 URL 写入 trace。发生泄漏时，先吊销和轮换，再判断日志/缓存/备份是否也含副本。不要只删除一条日志，需保留处置证据和受影响的 audience。

### 37.3.10 审计与事件响应

审计事件至少包含：时间（wall + monotonic/offset）、主体、动作、资源 digest/UID、策略版本、决定、理由、关联 request/job/trace ID、节点和来源。事件采用链式 hash 或写入不可变存储，禁止把秘密和原始 prompt 作为字段。hash 链能证明记录被改动过，不等于事件发生时节点未被攻破，因此关键事件要有独立收集端。

响应顺序可按“停、证、撤、修、复、学”：

1. **停**：冻结 release promotion、暂停受影响 job、阻断 egress，避免继续扩散；
2. **证**：保存镜像/模型/数据 digest、Pod spec、审计日志、SBOM、签名和时间同步状态；
3. **撤**：撤销签名 key、registry token、KMS grant、服务账号和受影响版本；
4. **修**：修补依赖/策略/数据管道，重建并重新签名，不在原 digest 上覆盖；
5. **复**：在隔离环境复现，执行 canary、回滚和增量放量，验证租户边界；
6. **学**：更新 threat model、检测规则、保留期限和演练剧本，给例外设置到期。

## 37.4 机制落地：从 release manifest 到 admission

### 37.4.1 一个可审计的 release manifest

建议每个模型版本拥有一份小而稳定的 manifest：

```json
{
  "model_digest": "sha256:...",
  "tokenizer_digest": "sha256:...",
  "dataset_manifest_digest": "sha256:...",
  "image_digest": "sha256:...",
  "source_revision": "git:...",
  "sbom_digest": "sha256:...",
  "provenance_digest": "sha256:...",
  "signing_identity": "release-bot@project.example",
  "policy_version": "ai-release-v4"
}
```

manifest 只引用摘要和身份，不携带秘密。部署控制器检查 manifest 与 OCI image、模型仓库、Kubernetes annotation 的一致性；发现差异时拒绝。上线后把 manifest ID 写入 metrics resource attribute、trace resource 和审计事件，避免只靠可变 tag 进行关联。

### 37.4.2 Policy as code 的拒绝原因

无论使用 Kyverno、OPA Gatekeeper、Sigstore policy-controller 还是自研 webhook，拒绝响应都应可行动：`digest_required`、`signature_identity_mismatch`、`provenance_missing`、`sbom_missing`、`critical_cve`、`privileged_pod`、`host_path`、`network_policy_missing`、`gpu_partition_unisolated`、`secret_value_embedded`。拒绝原因进入审计，但不要把用户提交的秘密原样回显到 API 错误。

策略测试要覆盖允许、拒绝和例外过期。每次策略发布记录规则版本、测试集合、评审者和回滚点。admission webhook 不可用时，选择 fail-closed 或受限 fail-open 必须是明确的风险决策；高敏感命名空间一般 fail-closed，低风险开发环境可短时隔离并告警。无论哪种模式，都要监控 webhook 延迟和拒绝率，避免运营人员以“紧急”名义永久绕过。

### 37.4.3 可复现构建和两人批准

可复现构建不是要求每个 GPU kernel 字节都相同，而是让关键输入、工具链、参数和输出摘要可重建、可解释。固定基础镜像 digest、时区、locale、编译器、依赖 lockfile、随机种子和时间戳；使用 SOURCE_DATE_EPOCH 等机制消除非确定性；比较 SBOM 和输出 digest。对不可复现的产物记录差异来源和接受理由，不要静默覆盖。

签名最好由受控服务在两人/两阶段条件下执行：构建机器人提交候选，扫描与测试服务出具证据，批准者或自动策略签发短期签名。签名服务不能接受任意 digest；应根据 manifest、CI run、分支保护和策略版本计算允许集合。紧急补丁也要进入同一证据链，事后补签会破坏时间语义。

## 37.5 实验：CPU-only 供应链证据门 toy lab

### 37.5.1 实验问题与边界

实验问四个可重放问题：

1. 完整 digest、签名、provenance、SBOM 是否允许一个候选 artifact 进入下一步？
2. manifest 漂移、标签异常和 canary loss 回归是否把数据版本置于 quarantine？
3. 未修复 high/critical 依赖、privileged/可写容器、未隔离 GPU、嵌入式 secret 和 allow-all egress 是否 fail-closed？
4. 处置决定能否写入不含秘密的链式审计事件，并给出冻结/吊销/阻断动作？

实验脚本为 `labs/ch37_security_supply_chain_lab.py`，只使用 Python 标准库。fixture 中的 digest、漏洞、GPU 分区和 secret 引用是合成值；没有 registry、Kubernetes API、GPU、网络访问或真实密码。实验通过不是“生产安全认证”，而是策略契约的可执行示例。

### 37.5.2 运行命令

```bash
python3 -m py_compile labs/ch37_security_supply_chain_lab.py
python3 labs/ch37_security_supply_chain_lab.py --fault none --output reports/ch37-security-supply-chain-baseline.json
python3 labs/ch37_security_supply_chain_lab.py --fault poisoned_data --output reports/ch37-security-supply-chain-poisoned.json
python3 labs/ch37_security_supply_chain_lab.py --fault unsafe_runtime --output reports/ch37-security-supply-chain-runtime.json
python3 tests/test_ch37_security_supply_chain_lab.py
```

`simulate()` 使用固定 fixture，每种 fault 只修改一个控制面因素。输出包含 artifact、model_data、dependencies、runtime、audit_events、incident 和 invariants；stdout 与 `--output` 文件应完全相同。脚本故意不会打印任何 secret 值，事件只保存 `secret_handling` 等理由。

### 37.5.3 基线和故障矩阵

| fault | 变化 | 预期决定 | 关键证据 |
| --- | --- | --- | --- |
| `none` | 所有证据完整，MIG-isolated，registry-only egress | `admit` | 四类 evidence、audit hash chain |
| `unsigned_model` | signed=false | `deny` | `artifact_evidence`、signature false |
| `poisoned_data` | manifest 漂移、poison/label anomaly、canary 回归 | `quarantine` | model-data reasons、冻结推广 |
| `vulnerable_dependency` | 未修复 critical CVE | `deny` | SBOM/依赖 findings、block release |
| `unsafe_runtime` | root、可写 rootfs、capabilities、privileged、hostPath、弱 sandbox | `deny` | runtime isolation findings |
| `gpu_cross_tenant` | shared-unisolated 分区 | `deny` | GPU tenancy finding |
| `secret_leak` | Pod spec 中出现 value: 引用 | `deny` | secret handling，事件无 secret 值 |
| `unexpected_egress` | allow-all 网络策略 | `deny` | network policy finding |

### 37.5.4 观察与解释

基线的 `all_required_evidence`、`audit_chain_valid`、`no_secret_values_in_events` 应为真。`poisoned_data` 的决定是 quarantine 而非普通 deny，表示 artifact 可以保留在隔离区供取证，但不能成为生产候选；响应应冻结 release promotion、保留原始数据和撤销已生成的 promotion token。其他控制面违反则直接 deny，阻断部署或构建。

toy 的 `verify_artifact()` 对 malformed digest 采用 deny result；`assess_model_data()` 只报告统计信号，不声称识别所有后门；`scan_dependencies()` 只处理 fixture 中的严重度和 fixed 字段；`evaluate_runtime()` 只检查策略字段，不执行 syscall 或 GPU 操作。工程师应把这些接口替换为真实 verifier、SBOM scanner、admission webhook、KMS 审计和节点遥测，并保留同样的理由结构。

### 37.5.5 从 toy 到 staging 的扩展

在 staging 做三层替换：第一层用真实镜像/模型 digest、Cosign 签名和 provenance；第二层用真实 SBOM、漏洞数据库快照、Kubernetes admission 和 NetworkPolicy；第三层在隔离 GPU 节点验证 MIG、驱动 ioctl、reset、显存清理、租户切换和故障取证。每次扩展都要保留合成 fixture 作为回归测试，避免真实系统依赖导致策略悄悄放宽。

## 37.6 故障诊所/失败：常见误解和诊断顺序

### 37.6.1 “tag 是最新的，所以一定是批准版本”

症状：Pod 显示 `model:prod`，事故后无法确定当时运行的权重。根因是 tag 可移动，审计没有 digest。诊断先取运行时 image ID、模型文件 hash、release manifest 和 admission decision；若任一缺失，进入证据缺口而不是猜测。修复是部署强制 digest，tag 只作人类导航，并在启动日志/trace resource 中记录摘要。

### 37.6.2 “镜像扫描 0 个 CVE，所以可以 privileged”

症状：扫描报告干净，但容器可读 `/proc`、hostPath 和云 metadata。根因是扫描证明组件状态，不证明运行时权限。诊断检查 effective UID、capabilities、seccomp、host namespaces、设备节点、NetworkPolicy 和服务账号。修复是 rootless、read-only、drop caps、no_new_privs、默认拒绝网络和最小 RBAC；必要的 GPU 设备访问写成明确例外并测试回收。

### 37.6.3 “MIG 就等于完整隔离”

症状：两个租户有不同 MIG instance，却共享日志、对象存储 token、driver bug 或高层网络。根因是只验证了 SM/显存切片。诊断把 parent GPU、instance UUID、Pod UID、namespace、驱动和 reset 事件对齐，检查跨租户 trace、共享 mount 和 egress。修复是按租户分 namespace/RBAC/secret/network，评估 side channel 风险，敏感工作负载使用独占设备；MIG 不能替代补丁和最小权限。

### 37.6.4 “把 secret 放进环境变量最方便”

症状：错误 trace、core dump 或 debug endpoint 出现 token。根因是 secret 进入进程可继承和可观测路径，日志过滤不完整。诊断先吊销 token，再搜索日志、trace、构建层、shell history、对象存储 URL 和备份；不要等待定位完才轮换。修复使用短期 workload identity、文件描述符或受控 sidecar，显式 allowlist 字段，建立 secret access audit 和轮换演练。

### 37.6.5 “数据指标变好，所以不是投毒”

症状：离线准确率上升，线上特定触发器输出异常。根因是攻击者针对评测集或目标行为优化。诊断比较多版本 manifest、来源时间窗、去重/标签分布、独立 canary 和差分重训；保留原始对象和随机种子。修复把 quarantine 和 shadow 发布作为默认路径，加入触发器/后门评测和人工抽样，不把一个总分当作安全结论。

### 37.6.6 “依赖没有 CVE，构建就是可信”

症状：镜像无已知漏洞，却在构建时上传环境变量。根因可能是恶意包、安装脚本、维护者账号被盗或数据库尚未收录。诊断比较 lockfile、SBOM、provenance、网络流量和构建 runner 文件；使用可复现重建和独立 builder 交叉比对。修复私有代理、允许列表、无生产凭据构建、脚本最小权限和行为监控。漏洞扫描是必要但不充分的控制。

### 37.6.7 “审计 hash 链能证明一切”

症状：事件链完整，但关键节点时间不可信或审计 agent 被同一节点篡改。根因是完整性和真实性被混为一谈。诊断核对独立 collector、NTP/PTP offset、写入确认、节点启动/驱动事件和云审计。修复关键事件双写到隔离存储，记录采集器身份和丢失计数；hash 链只承诺检测后续篡改，不能创造不存在的事件。

### 37.6.8 运行时隔离突然让模型加载失败

症状：启用 read-only/seccomp/sandbox 后，tokenizer 或 CUDA extension 报权限错误。根因可能是真实的写路径、隐式网络下载、缺失 ioctl 或不兼容的 sandbox，而不是“安全配置坏了”。诊断在隔离 staging 记录 syscall、文件写入、DNS、设备访问和失败版本；不要直接改回 privileged。修复预热并缓存依赖、把临时写入重定向到受限 emptyDir、增加精确 seccomp 例外，或换用 Kata/gVisor/独占节点；例外有 owner、期限和回滚。

## 37.7 理解检查

1. 为什么只验证容器签名不能证明训练数据没有投毒？请指出内容链与执行链的缺口。
2. tag、digest、签名和 provenance 各自承诺什么？哪一个能证明“签名者有权批准”?
3. 一个 critical CVE 在无网络、非 root、只读输入的转换沙箱里，为什么仍要记录但可以有条件延期？延期证据应包含哪些字段？
4. MIG 提供了哪些硬件边界，哪些网络、机密和驱动边界仍需单独控制？
5. 为什么事件 hash 链不能替代独立审计 collector？给出节点被攻破时的反例。
6. 发生 secret 泄漏时，为什么第一步通常是吊销/轮换而不是先搜索所有日志？
7. `quarantine` 与 `deny` 的运营语义有什么区别？哪种情况应保留原始对象供取证？
8. NetworkPolicy 默认拒绝仍可能被绕过的两个路径是什么？（提示：hostNetwork、云 metadata/允许的 egress gateway。）
9. 一个扫描器报告“0 漏洞”时，如何证明扫描覆盖了 native extension、系统库和实际运行层？
10. 解释为什么“只记录 request_id，不记录 prompt”仍可能泄露隐私，以及如何用 hash、TTL 和访问控制降低风险。

## 37.8 练习

### 练习 A：画出一个可撤销的供应链

选择一个开源模型，画出数据、源码、构建、签名、registry、Kubernetes、GPU 和用户请求的节点与边。为每条边标注写权限、读权限、digest、审计点和撤销动作。再设计一个签名 key 泄漏的演练：哪些版本进入 denylist，如何验证旧 Pod 是否仍在运行，如何在不泄露 prompt 的前提下通知租户？

### 练习 B：写一份 admission policy

使用 Rego、Kyverno 或伪代码实现以下门禁：digest pin；签名 identity；provenance builder/source；SBOM 存在；critical/high 可达漏洞为 0；rootless/read-only/drop caps；禁止 hostPath/privileged；默认拒绝网络；GPU 必须 exclusive 或 mig-isolated；Secret 只能是引用。为每条拒绝写测试和可行动 reason，并加一个 7 天后自动失效的临时例外。

### 练习 C：投毒检测的反事实

构造三个数据版本：随机标签污染 1%、特定触发器 0.2%、自然分布漂移 5%。选择 manifest、标签统计、去重、独立 canary、差分重训和人工抽样的组合，说明哪些能检测、哪些可能漏检。定义 quarantine 的最大保留时间和升级条件。

### 练习 D：GPU 多租户故障演练

在不使用真实攻击代码的前提下，模拟两个 tenant 的 batch 负载、MIG/time-slicing 配置和驱动 reset。要求输出 parent GPU/instance UUID、Pod UID、租户、时间线、性能异常和处置动作。比较独占、MIG、time-slicing 的成本、容量和侧信道证据，给敏感租户写准入矩阵。

### 练习 E：机密与日志审计

列出一个推理服务所有可能的机密出口：环境变量、命令行、core dump、trace attributes、HTTP access log、对象存储 URL、错误页面、缓存和备份。为每个出口指定禁止字段、脱敏方式、保留期限、访问角色和轮换后的验证命令。用 toy lab 的 `no_secret_values_in_events` 不变量写一个回归测试。

### 练习 F：从故障到复盘

选择 `vulnerable_dependency` 或 `unexpected_egress` 场景，写一份 30 分钟时间线：发现、停、证、撤、修、复、学。明确哪些证据是已验证事实，哪些只是待证假设；给出回滚条件、通知边界、恢复后的 shadow/canary 指标和 action owner。

## 37.9 生产验收清单

- [ ] 每个模型、tokenizer、数据 manifest、镜像和 SBOM 都有不可变 digest，并在 release manifest 中互相引用；
- [ ] 签名 key、provenance builder、透明日志和撤销路径经过轮换/泄漏演练；
- [ ] 构建 runner 无生产凭据，依赖解析、下载、编译、测试、签名分阶段隔离；缓存 key 包含 lockfile、平台和工具链；
- [ ] SBOM 覆盖系统包、语言依赖、native extension、CUDA/NCCL 绑定和启动脚本，扫描数据库版本和忽略规则可追溯；
- [ ] 模型加载使用安全格式或隔离转换沙箱，输入只读、无网络、非 root，记录工具和输入/输出 digest；
- [ ] 数据版本具备 manifest、来源、清洗代码、随机种子、独立 canary、毒性/触发器抽样和 quarantine 流程；
- [ ] Kubernetes 默认 rootless/read-only/drop caps/no_new_privs，禁止 privileged/hostPath/host namespaces，服务账号和 RBAC 最小化；
- [ ] NetworkPolicy 默认拒绝，显式限制 registry、KMS、模型存储、服务端口和 metadata 路径，IPv4/IPv6/sidecar/hostNetwork 例外有测试；
- [ ] GPU 使用独占或经验证的 MIG；parent/instance UUID、租户、Pod、驱动和 reset 事件可关联，异常时有隔离和清理；
- [ ] Secret 不进入镜像、manifest、命令行、日志、trace、core dump 或持久 URL；采用短 TTL、audience、轮换和吊销审计；
- [ ] admission decision、运行时事件和响应动作写入独立、不可变或链式审计，事件不含 prompt/secret 原文；
- [ ] 例外有 owner、理由、最小范围、到期时间、自动提醒和回滚；fail-open/fail-closed 经过风险评审；
- [ ] 每季度演练 unsigned artifact、投毒、critical CVE、Pod 逃逸、GPU reset、secret 泄漏和异常 egress，保存证据并更新 threat model。

## 37.10 来源

以下链接优先选择规范、官方文档、论文或维护者仓库；版本会变化，生产实施前应固定 revision 并重新核验：

- [NIST SP 800-161 Rev.1：Cybersecurity Supply Chain Risk Management](https://csrc.nist.gov/pubs/sp/800/161/r1/final)
- [NIST SSDF SP 800-218](https://csrc.nist.gov/pubs/sp/800/218/final)
- [SLSA v1.0 specification](https://slsa.dev/spec/v1.0/)
- [in-toto framework](https://in-toto.io/)
- [Sigstore Cosign container signing](https://docs.sigstore.dev/cosign/signing/signing_with_containers/)
- [OCI Image Specification](https://github.com/opencontainers/image-spec)
- [SPDX specifications](https://spdx.dev/specifications/)
- [CycloneDX specification](https://cyclonedx.org/specification/8.0/)
- [Kubernetes Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Kubernetes NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Kubernetes Secrets good practices](https://kubernetes.io/docs/concepts/configuration/secret/#good-practices)
- [Kubernetes audit policy](https://kubernetes.io/docs/tasks/debug/debug-cluster/audit/)
- [Kubernetes device plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [NVIDIA DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [NVIDIA Container Toolkit security](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/security.html)
- [gVisor architecture](https://gvisor.dev/docs/architecture_guide/)
- [Kata Containers](https://katacontainers.io/)
- [OCI Runtime Specification](https://github.com/opencontainers/runtime-spec)
- [seccomp Linux kernel documentation](https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html)
- [Landlock Linux security module](https://www.kernel.org/doc/html/latest/userspace-api/landlock.html)
- [safetensors](https://github.com/huggingface/safetensors)
- [OSV vulnerability database](https://osv.dev/docs/)
- [OpenSSF Scorecard](https://github.com/ossf/scorecard)
- [OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/)
- [MITRE ATLAS](https://atlas.mitre.org/)
- [MITRE ATT&CK Software Supply Chain Compromise](https://attack.mitre.org/techniques/T1195/)
- [OWASP CycloneDX SBOM guide](https://owasp.org/www-project-cyclonedx/)
- [Python packaging security guidance](https://packaging.python.org/en/latest/guides/analyzing-pypi-package-downloads/)
- [Sigstore Rekor transparency log](https://docs.sigstore.dev/logging/overview/)
- [TUF specification](https://theupdateframework.io/docs/)
- [D2X: Data Poisoning Attacks against Machine Learning](https://arxiv.org/abs/1708.06733)
- [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.06733)
- [Neural Cleanse](https://www.usenix.org/conference/usenixsecurity19/presentation/wang)
- [DNN backdoor survey](https://dl.acm.org/doi/10.1145/3464377)
- [Dapper distributed tracing](https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/)
- [OpenTelemetry security considerations](https://opentelemetry.io/docs/security/)

本章的本地证据索引见 `evidence/ch37-security-supply-chain-manifest.json`，实验报告见 `reports/ch37-security-supply-chain-report.md`。论文和官方文档说明威胁与机制；toy lab 只验证本章定义的字段、决定和失败闭环，不可替代目标环境的渗透测试、合规评审、GPU/驱动验收或真实流量演练。
