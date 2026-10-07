# 第37章 AI Infra 安全与供应链实验报告

- 日期：2026-10-07
- 实验：`labs/ch37_security_supply_chain_lab.py`
- 级别：L0，CPU-only，Python 3.10+ 标准库
- 目的：以可重放的证据门模拟 artifact 完整性、签名/provenance/SBOM、模型/数据投毒信号、依赖漏洞、容器/Kubernetes 隔离、GPU 多租户、机密、网络策略、审计 hash chain 与事件响应
- 边界：不下载包、不访问 registry/Kubernetes/GPU/KMS，不执行真实反序列化或攻击；数字是合成输入，不是生产安全、GPU 隔离或漏洞扫描保证

## 可复现命令

```bash
python3 -m py_compile labs/ch37_security_supply_chain_lab.py
python3 labs/ch37_security_supply_chain_lab.py --fault none --output reports/ch37-security-supply-chain-baseline.json
python3 labs/ch37_security_supply_chain_lab.py --fault poisoned_data --output reports/ch37-security-supply-chain-poisoned.json
python3 labs/ch37_security_supply_chain_lab.py --fault unsafe_runtime --output reports/ch37-security-supply-chain-runtime.json
python3 tests/test_ch37_security_supply_chain_lab.py
```

相同命令使用固定 fixture，stdout 与 `--output` 文件完全一致。`simulate()` 的输出版本为 `schema_version=1`，事件链仅含动作、主体、策略理由和前一事件摘要，不含任何 secret value。

## 结果摘要

| 场景 | 决定 | 主要证据 | 处置语义 |
| --- | --- | --- | --- |
| `none` | `admit` | digest、signature、provenance、SBOM、数据质量、依赖、MIG-isolated、registry-only policy 均通过 | 不采取 containment，保留审计 |
| `unsigned_model` | `deny` | artifact gate 的 `signature` 失败 | 阻断发布，不能用 tag 绕过 |
| `poisoned_data` | `quarantine` | manifest mismatch、poison rate、label anomaly、canary loss regression | 冻结推广、撤销 promotion、保留原始数据供取证 |
| `vulnerable_dependency` | `deny` | 未修复 critical CVE finding | 阻断构建/部署，修复后重建新 digest |
| `unsafe_runtime` | `deny` | root/privileged、可写 rootfs、capabilities、弱 sandbox、host path | 阻断部署，不能以“扫描干净”作为例外 |
| `gpu_cross_tenant` | `deny` | `shared-unisolated` GPU partition | 选择独占或经验证 MIG，隔离设备并审计 UUID |
| `secret_leak` | `deny` | Pod spec 嵌入 `value:`，secret handling reason | 吊销/轮换，事件不打印 secret 值 |
| `unexpected_egress` | `deny` | allow-all 网络策略 | 阻断出站，检查 metadata/registry/KMS 路径 |

所有场景的 `audit_chain_valid`、`no_secret_values_in_events` 和 `fail_closed` 不变量应为真。`all_required_evidence` 只表示 toy 的字段齐全，不表示 artifact、数据或运行时已经通过现实世界的渗透测试。

## Toy 实现与合同

- `Artifact` 强制非空 digest、模型 hash、来源和版本；`verify_artifact()` 要求 `sha256:` digest、签名、provenance、SBOM，并接受可配置的漏洞评分阈值。Malformed evidence 返回 deny，而不是抛出一个让调用方误以为“未检查”的异常。
- `assess_model_data()` 比较 manifest hash、poison/label anomaly rate 和独立 canary loss 相对变化。它只标记信号；对未知后门不作完备检测承诺。
- `scan_dependencies()` 对未修复 high/critical finding 生成 `block_release`，记录组件、版本、CVE 和 severity；实际环境仍需锁文件、native extension、镜像层和数据库快照。
- `RuntimeSpec` 检查 rootless、read-only rootfs、drop capabilities、restricted seccomp、no-new-privs sandbox、默认拒绝/registry-only network、secret 引用和 GPU partition。它不运行 syscall，也不验证真实设备插件。
- `_audit_events()` 使用 `GENESIS` 加链式 SHA-256，验证 `prev_hash` 与事件 body；链的完整性不替代独立 collector、可信时钟和节点取证。
- `simulate()` 将模型/数据问题区分为 quarantine，将执行边界、依赖、机密和网络策略问题 fail-closed deny，并返回可行动的 `deny_reasons`。

## 失败边界与解释

1. digest、签名和 provenance 是合成字段，toy 没有连接 Sigstore、透明日志、KMS 或真实身份；staging 必须验证签名主体、撤销与过期。
2. SBOM/漏洞清单是手工 fixture，未解析 OS 包、Python wheels、native extension、CUDA/NCCL 或可达性；生产扫描需锁数据库版本和忽略规则。
3. model/data poisoning 只演示 manifest、统计和 canary 信号，低于阈值的后门或自然漂移可能绕过，必须加入独立抽样、差分重训和人工复核。
4. rootless/seccomp/NetworkPolicy/GPU partition 仅作策略字段；没有 Kubernetes API、容器 runtime、IOMMU、MIG、驱动 ioctl、side-channel 或 reset 行为。
5. secret 只以字符串标记 `value:supersecret` 注入 fixture；事件不包含该值。生产发生泄漏时，先吊销/轮换，再调查日志、trace、缓存、备份和 URL 副本。
6. hash chain 能检测后续编辑，不保证采集器未被攻破、时钟准确或事件未丢失；关键事件应双写到隔离且不可变的审计端。

## 生产验收计划

1. 在隔离 registry 中用固定 digest 的真实镜像、Cosign 签名、SLSA/in-toto provenance 和 SBOM 重放 artifact gate；测试签名 key 撤销、透明日志不可用和 admission webhook 超时。
2. 将模型转换器放入无网络、无 secret、非 root、只读输入的 sandbox，记录输入/输出 digest、syscall、文件写入和工具版本；比较安全格式加载与旧格式转换的差异。
3. 用真实依赖数据库和独立 builder 生成 SBOM，核对 lockfile、镜像解包、运行时进程和 native library；为每个例外设 owner、理由、范围、期限和自动失效。
4. 在 Kubernetes staging 做 Pod Security、NetworkPolicy、RBAC、metadata endpoint、IPv4/IPv6、sidecar、hostNetwork 和节点升级测试；演练 fail-closed 与受限 fail-open 的影响。
5. 在隔离 GPU 节点比较独占、MIG 和 time-slicing：清理显存/page cache，记录 parent/instance UUID、Pod UID、租户、驱动/XID/ECC/reset，观察性能和时序泄漏证据。
6. 注入 unsigned artifact、manifest drift、poisoned canary、critical dependency、secret logging 和异常 egress，按“停、证、撤、修、复、学”执行并保存完整时间线。

## 对应测试

`tests/test_ch37_security_supply_chain_lab.py` 覆盖：

- 基线确定性、终态和审计/机密不变量；
- digest、signature、provenance、SBOM 四个 artifact gate；
- model/data poisoning quarantine 以及 dependency critical finding deny；
- unsafe runtime、GPU cross-tenant、secret embedding、unexpected egress 的 fail-closed reason；
- 组件输入校验和 CLI stdout/文件 JSON 一致性。

测试通过只说明 toy policy contract 成立。生产结论必须来自固定版本、隔离环境、真实身份与签名、GPU/驱动、Kubernetes、网络和流量回放证据。

## 主要来源

- NIST SP 800-161 Rev.1：https://csrc.nist.gov/pubs/sp/800/161/r1/final
- SLSA：https://slsa.dev/spec/v1.0/
- Sigstore Cosign：https://docs.sigstore.dev/cosign/signing/signing_with_containers/
- in-toto：https://in-toto.io/
- Kubernetes Pod Security Standards：https://kubernetes.io/docs/concepts/security/pod-security-standards/
- Kubernetes NetworkPolicy：https://kubernetes.io/docs/concepts/services-networking/network-policies/
- NVIDIA MIG User Guide：https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- SPDX：https://spdx.dev/specifications/
- CycloneDX：https://cyclonedx.org/specification/8.0/
- safetensors：https://github.com/huggingface/safetensors
- MITRE ATLAS：https://atlas.mitre.org/
