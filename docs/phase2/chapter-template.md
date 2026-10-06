---
id: phase2-XX-topic
# Keep the published一期 identity fields: Docusaurus and the一期 audit read these.
title: 章节标题：先写读者要解决的问题
description: 一句话说明本章边界、对象和读完后能做什么
slug: /phase2/chapters/XX-topic
sidebar_position: 100 # replace with a unique numeric position
# The phase-two contract begins here.
phase: 2
chapter_number: XX
level: foundation # foundation | core | systems | advanced | frontier | capstone
prerequisites: [] # stable chapter ids, for example [phase2-01-evidence-language]
learning_objectives:
  - 能从一个小例子说出本章要解释的现象和边界
  - 能用公式、单位和数字例子检查关键约束
  - 能沿固定版本源码入口描述一次请求或数据的路径
  - 能运行 CPU 或目标硬件实验，报告分位数和限制
paper_count: 0
source_commits:
  - REPLACE_WITH_PINNED_COMMIT
lab_paths:
  - labs/phase2/REPLACE_WITH_LAB.py
last_verified: 2026-10-06
# Optional compatibility aliases for the一期 metadata renderer. Keep them equal
# to the first plural entry while the phase-two renderer is being migrated.
source_commit: REPLACE_WITH_PINNED_COMMIT
lab_path: labs/phase2/REPLACE_WITH_LAB.py
estimated_hours: 8
---

# 章节标题：先写读者要解决的问题

> **本章地图**：用两三句话说明起点、终点，以及哪些问题本章暂时不回答。

## 问题边界：读者此刻卡在哪里

从一个真实的两请求（或两批数据）场景开场。给出可观察的现象、输入、输出和本章不讨论的相邻问题。不要先列名词。

## 直觉模型：先用生活化图景抓住约束

画一张能回答问题的图，或用一段时间线解释资源如何移动。第一次出现的术语放在直觉之后，并说明它在运行时对应什么。

## 最小例子：手算一次，再推广

构造尽量小的数字例子。把中间步骤写出来，检查单位；读者应能在纸上复算，而不是只看结论。

## 正式定义与推导：变量、公式和假设

先重述要解释的量，再写公式。逐一解释变量、单位、数据来源和假设；指出拿掉某个假设时哪一步失效。公式先用上节数字代入，再给符号形式。

## 机制与源码入口：从请求到硬件（或存储）

先给读者一个阅读任务，例如“找出 waiting 何时变成 running”。注明仓库、tag/commit、目录和 20–40 行关键代码；画出调用链，解释正常路径、异常路径、状态变化和指标。不要贴无法逐行讲解的大段代码。

## 失效边界：结论何时不再成立

用反例或失败现象检验模型。区分“实现没有做到”和“指标没有测到”；说明硬件、版本、数据分布、并发度或网络条件改变时会发生什么。

## 可运行实验：基线、变量和原始证据

写清 CPU/硬件、操作系统、软件版本、输入、基线、变量、控制变量、命令、随机种子、重复次数、原始输出位置和统计方法。至少报告 p50/p95/p99、吞吐、TTFT、ITL/TPOT、显存/KV 使用率中与本章相关的指标，并写出误差来源和不能推断的结论。不能运行目标硬件时，提供诚实的 CPU/模拟 fallback，并标出它不能证明什么。

## 失败诊所：一个看起来健康的结果

选一个容易误判的症状。按“观察 → 假设 → 最小排查 → 证据 → 修复/回滚”叙述，保留失败命令和异常路径。

## 方案边界比较：至少两个选择

用相同的工作负载和单位比较方案；同时列出收益、代价、适用前提、最坏情况和被拒绝的替代方案。不要把论文中的一次测量写成生产保证。

## 论文谱系、源码观察与证据地图

按“当时哪里慢/错/贵 → 关键瓶颈 → 核心想法 → 实验证明了什么/没有证明什么 → 今天保留与改变了什么”讲论文。每项事实、定义、推导、论文结果、源码观察、实验测量、推断、设计判断和未验证假设都能在 `docs/phase2/evidence/XX-topic.json` 找到对应条目。

## 研究问题与理解检查

提出至少两道由浅入深的问题并给答案或提示。问题应让读者解释机制、检查单位或设计对照实验，而不是复述标题。

## 练习：从复算到设计

至少包含一道复算/定义题、一道故障诊断题和一道设计或实验题。写出输入、验收标准和可能的错误方向；不要只给“思考题”。

## 术语表

用自己的话解释本章首次出现的术语，并链接到依赖章节。英文名只在需要精确检索时保留。

## 来源地图与复现清单

列出论文、规范、官方文档、仓库固定版本和实验 artifact。复现清单应能让另一位读者从干净 checkout 开始：准备环境、运行命令、预期观察、允许的误差、原始输出位置和清理步骤。

---

## Author contract

- Keep the frontmatter fields above. `source_commits` and `lab_paths` are the canonical plural fields; the singular aliases exist only for the一期 metadata renderer during migration.
- Use stable phase-two ids in `prerequisites`; do not make a chapter depend on a prose title or a `main` branch.
- Add one evidence manifest beside each chapter. The manifest is provenance, not a bibliography: every entry has a claim, classification, pinned version, and limitation.
- Treat this outline as a set of questions, not a heading quota. A chapter may merge or split sections when the learner’s path is clearer; do not inflate length with repeated summaries.
- Run the deterministic checks in `scripts/test_phase2_template.py` (and the repository phase-two audit) before review. A passing check does not replace a teacher’s read-through.
