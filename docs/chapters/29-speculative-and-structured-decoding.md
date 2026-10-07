---
id: ch29-speculative-and-structured-decoding
title: Speculative Decoding 与结构化解码：从精确性证明到 Serving 调度
slug: /chapters/29-speculative-and-structured-decoding
description: 深入 Leviathan、Medusa、EAGLE、SpecInfer、Lookahead speculative decoding，grammar/JSON/regex 约束，accept-reject 精确性、调度器集成、性能模型与 vLLM/SGLang 源码验收
sidebar_position: 29
level: advanced
prerequisites:
  - ch15-inference-execution
  - ch16-model-serving-system
  - ch19-inference-optimization-accelerator-stack
  - ch20-observability-debugging-incident-response
  - ch21-ai-reliability-engineering
  - ch28-vllm-and-sglang-serving
learning_objectives:
  - 能从目标分布 p 与草稿分布 q 推导 Leviathan accept-reject，证明输出分布不变并说明实现中的数值与停止边界
  - 能区分独立小模型、Medusa 多头、EAGLE 特征草稿、SpecInfer 树验证、Lookahead n-gram/多步草稿的候选生成与验证代价
  - 能把 draft、verify、rollback、KV page、grammar 状态和 continuous batching 放进同一个 serving 调度步模型
  - 能设计 JSON Schema、regex、CFG/有限状态 grammar 的 token mask、缓存键、编译、回退与安全边界
  - 能建立 acceptance、目标模型调用、候选分支、grammar reject、TTFT、ITL、TPOT、显存与 goodput 的可审计性能模型
  - 能使用 CPU-only toy lab 验证精确采样、约束状态机和调度方向，并写出不把 toy 或论文数字当成生产承诺的报告
estimated_hours: 42
hardware: CPU-only lab required; CUDA GPU optional and cost-bounded
risk_level: L4
last_verified: 2026-10-07
---

# 第29章　Speculative Decoding 与结构化解码：从精确性证明到 Serving 调度

> speculative decoding 不是“让一个小模型替大模型写答案”，而是让草稿提出一段候选，再由目标模型一次验证；只要接受—拒绝规则满足条件，最终 token 序列仍来自目标模型分布。structured decoding 也不是把不喜欢的 token 删掉就结束：grammar、JSON Schema、regex 和工具调用协议必须在 token 边界上维护状态、处理回退，并且与 KV、批调度、取消和流式传输共同构成正确性边界。本章把两者作为一个运行时问题来分析。

## 29.1 本章范围：从“生成得快”到“提交得对”

第 23 章已经介绍采样温度、top-k、top-p 和基础质量指标，本章不重复采样调参，而关注 serving 实现中的四个难点。

1. **候选如何产生。** 可以是独立 draft model、同一模型的 Medusa 多头、EAGLE 的隐藏状态预测器、SpecInfer 的树、Lookahead 的 n-gram 或多步自回归草稿。它们的 q 分布、候选数和 GPU 代价不同。
2. **目标如何验证。** 验证必须使用目标模型在同一上下文、同一位置编码、同一 logits 处理和同一约束状态下的 p 分布。批量验证能摊薄权重读取，却不能跳过拒绝后的 residual sampling。
3. **约束如何进入概率空间。** grammar mask 应作用于 p 和 q 的同一个允许集合并重新归一化；如果先按无约束 q 接受，再在事后删除非法 token，通常会改变分布并破坏 exactness。
4. **调度如何提交。** draft 产生的 KV 是暂存的；只有接受前缀和一个校正 token 被提交，剩余分支必须 rollback 或放回可复用 cache。scheduler 还要决定 speculative group 是否抢占其他 decode、是否跨请求共享 grammar 编译，以及结构化输出等待时如何背压客户端。

本章使用五类证据标记：**事实**来自论文/官方文档/源码；**机制**是根据指定版本路径重建的控制流；**测量**来自 `labs/ch29_speculative_lab.py` 的 CPU toy；**推断**是由公式或受控实验计算的结果；**设计判断**是服务 SLO、正确性和安全边界下的建议。任何数字没有模型、硬件、版本、输入输出分布和 warmup，就不应被解释为通用 benchmark。

### 29.1.1 先写验收合同

在改 scheduler 或启用 draft 前，先写一页合同：

- **分布合同：** 在固定随机种子、tokenizer、temperature/top-p、logits processor 和 grammar 下，speculative 输出的 token 序列分布与 target-only 相同（允许统计置信区间，而不是逐序列相同）。
- **状态合同：** 每个已提交 token 有唯一 position；被拒绝的候选、对应 KV page 和 grammar 状态不会泄漏到下一轮。
- **延迟合同：** 记录 TTFT、ITL、TPOT、E2E 的 p50/p95/p99；把 draft、verify、grammar 编译、排队和网络 flush 分段。
- **资源合同：** 记录 target forward 次数、draft forward 次数、候选 token 数、accepted token 数、临时 KV bytes、rollback bytes、grammar cache 命中率和 preemption。
- **安全合同：** schema/regex 的最大深度、最大分支、编译超时、递归引用和恶意超长字符串有明确上限；拒绝时返回可诊断错误而不是静默截断。

合同先于 flag。`--speculative-model`、`--num-speculative-tokens` 或 `--grammar-backend` 只是实现选择，不能取代这些可观测断言。

## 29.2 心智模型：自回归目标、草稿与验证的统一表示

给定上下文 (x_{<t})，目标模型定义下一个 token 的分布 (p_t(v)=P_M(vmid x_{<t}))。草稿模型或草稿头给出 (q_t(v)=P_D(vmid x_{<t}))。普通解码从 (p_t) 采样一个 token，追加后再计算下一位置。speculative 一轮让草稿连续提出 (k) 个 token (y_{t:t+k-1})，随后目标以一次批量 forward 得到每个位置的 (p_{t+j})。

候选不是“先写入输出再验证”，而是三层状态：

1. **逻辑 committed prefix：** 客户端可见、计入 position 和 stop 条件的 token。
2. **tentative branch：** 草稿提出的 token、临时 KV、logits 和 grammar 状态；可能整体回滚。
3. **verified suffix：** 目标模型已经计算但尚未全部提交的 logits；验证后最多提交接受前缀加一个 target correction。

设第一处拒绝位置为 (r)，则本轮提交 (y_t,ldots,y_{t+r-1}) 和一个从 residual target 分布采样的校正 token；若全部 (k) 个候选接受，再额外从 (p_{t+k}) 采样一个 token。后一 token 常称 bonus token。它保证每次 target verification 至少推进一个位置，避免草稿完全匹配时“没有新 token”的空转。

一次 target forward 的输入可以是连续序列，也可以是树状候选。连续序列易于使用 flash/paged attention；树验证需要共享祖先的 KV 和每个分支的 position/mask。无论实现，目标 logits 必须对应真实前缀，而不是包含尚未接受的兄弟分支。

### 29.2.1 目标模型的“同一性”比模型名字重要

exactness 依赖的 target 不是“同一个 checkpoint 名称”，而是同一概率程序：权重 revision、tokenizer vocab/normalizer、chat template、RoPE/位置缩放、量化舍入、temperature、top-k/top-p、repetition penalty、bad-word/stop processor、grammar mask 和随机数语义都必须一致。若 draft 用了不同 tokenization，候选 token 的 id 仍可送入 target，但 q 的概率需要在 target 的 token 空间中定义；若 tokenizer vocab 不同，直接比较 q[token_id] 是错误的。

在 vLLM/SGLang 这类 runtime 中，engine 还可能把 logits processor 插件、guided decoding backend、request-level seed 和 stop checker 放在不同进程。源码审计要追踪 request metadata 是否随 speculative batch 传到 worker；只在 API 层检查参数而没有把状态放入 worker，属于“看似支持、实际不约束”的故障。

## 29.3 Leviathan accept-reject：算法与 exactness 证明

### 29.3.1 单步接受规则

对草稿提出的 token (y)，目标概率为 (p=p_t(y))，草稿概率为 (q=q_t(y))。当 (q>0) 时，计算

[
alpha(y)=\min(1,p/q).
]

用独立 (u\sim U[0,1]) 接受 (y) 当且仅当 (u\lealpha(y))。若拒绝，定义 residual 分布

[
r(v)=\frac{[p_t(v)-q_t(v)]_+}{Z},\qquad Z=\sum_v[p_t(v)-q_t(v)]_+.
]

从 (r) 采样一个校正 token。若 (q) 是归一化分布，则 (Z=1-sum_v\min(p_t(v),q_t(v)))，也就是总变差距离 (TV(p_t,q_t))；目标和草稿越相近，拒绝概率越低。

### 29.3.2 一步分布证明

对任意 token (v)，接受路径贡献 (q(v)\min(1,p(v)/q(v))=\min(p(v),q(v)))。拒绝路径发生概率为 (1-Z) 的补集，即 (Z)，再乘 residual 概率 ([p(v)-q(v)]_+/Z)，贡献 ([p(v)-q(v)]_+)。相加得

[
\min(p(v),q(v))+[p(v)-q(v)]_+=p(v).
]

因此提交 token 的边缘分布是 p。对连续 k 个草稿 token，按从左到右在每个位置应用同一规则；一旦拒绝，后续草稿位置不提交，下一轮以上一轮的 target correction 为上下文。由条件概率链式法则，整个序列分布与 target-only 相同。这个证明不要求 q 是某个特定模型，但要求每个 q 条件于与 target 相同的已提交前缀，且随机变量独立且实现了 residual。

### 29.3.3 数值和实现陷阱

- **零概率：** 若 q(y)=0，候选在正确采样器里不应出现；若浮点下出现极小 q，必须定义上溢/下溢策略，不能直接产生 NaN。常见做法是用 logprob 比较 `log(u) <= min(0, logp-logq)`。
- **截断采样：** top-k/top-p 后 p、q 都需在同一截断集合重新归一化；将 p 的原始 softmax 与 q 的截断概率相除会改变 acceptance。
- **温度：** temperature 应在 logits 转概率之前一致应用。target 用 T=0.7、draft 用 T=1.0 再声称 exactness，仅在把 q 明确定义为 T=1.0 分布时成立；quality 变差但仍可 exact。
- **多 token batch：** target 返回 logits 的顺序、slot mapping 和 position ids 必须与 draft 候选一一对应。一个 offset 错位会在统计上看似“接受率下降”，实际是错误验证。
- **取消与断开：** 客户端在 verification 中途取消时，临时 KV 和随机数状态都应释放；不能把已产生但未发送的 bonus token 计入下一请求。
- **可复现：** 统计分布测试应固定 seed、并明确 CPU/GPU RNG。不要要求 speculative 与 baseline 每个 seed 得到逐 token 相同序列；正确要求是分布/频率一致。

### 29.3.4 约束下的 exactness

grammar 给出在状态 g 下的允许集合 (A(g))。正确流程是先构造

[
p^g(v)=\frac{p(v)1[v\in A(g)]}{\sum_{u\in A(g)}p(u)},\quad q^g(v)=\frac{q(v)1[v\in A(g)]}{\sum_{u\in A(g)}q(u)},
]

再用 (p^g,q^g) 做 accept-reject，并用提交 token 更新 g。若 (A(g)) 为空，应进入明确错误或 schema 回退策略，而不是把 EOS 当作万能合法 token。若 grammar backend 只给 target mask 而 draft 没有同一 mask，q 在非法 token 上的质量会流失，虽然可以通过 residual 修正保留 p^g，但接受率和实现复杂度会改变；工程上通常给 draft 同样 mask，减少无效候选。

## 29.4 四代草稿机制：Leviathan、Medusa、EAGLE、树与 Lookahead

### 29.4.1 独立小模型：简单但有部署成本

最直接的方法是为 7B/70B target 配一个小 draft model。draft 以低成本连续生成 k token，target 批量验证。优点是接口清晰、可独立量化、可在另一张 GPU/CPU 上运行；缺点是显存、权重加载和 tokenizer/采样逻辑重复。draft 与 target 的词表、chat template 和停止条件必须对齐，否则 acceptance 下降或无法 exact。

serving 中 draft 与 target 是两个资源池：draft 可能在同 GPU 上与 target 争夺显存，也可能通过 pipeline/远端 worker 引入 RPC 延迟。调度器要把 draft 计算纳入 token budget；“target forward 次数下降”不等于 E2E 变快，若 draft 传输和 kernel launch 成本超过节省的 target 工作，收益为负。

### 29.4.2 Medusa：同一 backbone 的多个预测头

Medusa 在 target 的隐藏状态之上增加多个 head，每个 head 预测不同未来位置，并通过树结构组合候选（事实：论文与官方实现提出多头 speculative decoding）。它避免独立 draft 模型的完整权重，但训练/微调和 head 之间的校准是新问题。第 j 个 head 的输入特征通常来自当前 hidden state 或前一预测，候选并非严格独立；实现必须保存每个树节点的 token、position 和父索引。

验证路径可把树展平为候选序列并使用 tree attention mask，一次 target forward 得到所有节点 logits；也可按层/深度分批。树宽过大时，candidate logits、attention mask、临时 KV 和 gather/scatter 可能超过节省的 target 计算。有效分支数不是配置的 `num_heads * topk`，而是经过去重、grammar mask、stop 条件和 page capacity 后真正送入 target 的节点数。

### 29.4.3 EAGLE：特征级草稿与训练边界

EAGLE 系列方法让 draft 在 target 的 feature/hidden-state 空间预测后续 token，再用 target 验证（事实：论文将 feature-level autoregressive draft 作为关键设计）。相比独立小模型，它可复用 target 的 embedding/feature，通常减少草稿参数；但 target 的隐藏状态接口、层选择、量化格式和训练 checkpoint 成为耦合点。若 serving 运行的 target revision 与 EAGLE head 不匹配，可能出现 silent acceptance 下降或数值错误。

工程验收要记录：head 依赖的 target 层与 hidden size、是否启用 tensor parallel、feature cache 的 page layout、draft head kernel 的 dtype、回退到 target-only 的条件。EAGLE 不能把“更接近 target 的 hidden feature”当作 exactness 证明；仍要使用 p/q accept-reject 或明确的无偏验证规则。

### 29.4.4 SpecInfer：树状验证与共享前缀

SpecInfer 把 speculative tokens 组织成 token tree，用树结构共享祖先计算并由 target 并行验证（事实：论文提出树状 speculative inference）。树能够容纳多个候选、不同 draft 来源和 top-k 分支，比单链 k token 更容易获得至少一条长接受路径；代价是 tree attention、分支调度和 KV 物理布局。

树验证的关键不在“把所有 token 拼在一起”，而在 attention mask：节点只能看到已提交前缀和自己的祖先，不能看到兄弟节点。若 mask 错把兄弟暴露给 target，p 分布改变；若 position id 把树深度和线性索引混淆，RoPE 相位错误。测试应构造二层、重复 token、共享祖先和空分支，逐节点对比 target-only logits。

### 29.4.5 Lookahead：n-gram 与多步候选的条件

Lookahead 类方法可从已生成文本中寻找 n-gram、利用自回归 lookahead slots 或维护候选窗口，再由 target 验证。它不一定需要额外模型，部署成本低，适合重复代码、模板化 JSON 或检索片段；对创造性文本，n-gram 命中少，候选质量 q 差，acceptance 下降。重复片段也可能把 grammar 状态带到错误边界，必须重新运行 parser，而不是因为 token 相同就直接复用。

### 29.4.6 方法比较表

| 机制 | q 的来源 | target 工作 | 主要状态 | 常见失败 | 适合的验收 |
| --- | --- | --- | --- | --- | --- |
| 独立 draft model | 小模型 logits | 链式 k token 验证 | draft KV、词表、RPC | 权重/模板不一致，draft 争显存 | token 分布、acceptance、draft/target wall time |
| Medusa | 多个 head 的候选 | 树或展平验证 | head 输出、父索引、tree mask | 分支爆炸、head 未校准 | 节点数、有效分支、mask correctness |
| EAGLE | hidden feature head | 链或小树验证 | feature cache、层映射 | target revision/dtype 不兼容 | logits 对齐、回退率、量化误差 |
| SpecInfer | 多草稿来源的 token tree | tree attention | node table、父子边、page ref | sibling 泄漏、KV scatter | 每节点位置与可见集合 |
| Lookahead | n-gram/窗口/slot | 链或小树验证 | 候选窗口、命中索引 | 重复率低、边界错 | workload 分层 acceptance |

表中的 target 工作不是“模型调用次数”这么简单；要按 target FLOPs、权重读取、attention KV bytes、kernel launch 和 scheduler round 量化。

## 29.5 Grammar、JSON、regex 与 CFG：从字符到 token

### 29.5.1 Grammar 状态机

约束解码器在每个已提交 token 后维护状态 g。对有限状态 grammar，g 可以是 DFA 状态；对 JSON Schema，g 通常包括栈、对象键集合、数组索引、字符串转义和数字状态；对 CFG，g 可能是解析栈和递归深度。解码器根据 tokenizer 的 token 字符串计算 `allowed(g)`，再把不允许的 token logits 设为 (-\infty)。token 可能跨越多个字符、包含引号和反斜杠，mask 必须模拟完整 token 消费并检查中间状态。

“字符 regex 看起来合法”不足以保证 token-level 合法。例如 regex `"[a-z]+"`，tokenizer 可能有 token `"ab`、`c"`、`"`；前两个 token 的中间状态不同。高效实现会预计算 token→字符序列的 transition，按 grammar state 缓存允许 token bitmap，并在 batch 内按 state 分组，减少每步逐 token Python 解释。

### 29.5.2 JSON Schema 的语义陷阱

JSON Schema 约束语法和部分结构语义，但不自动保证业务正确性。`type: integer` 不表示值在数据库允许范围；`required` 不表示字段之间业务一致；`additionalProperties: false` 可能拒绝未来兼容字段。structured decoding 的验收应分层：

1. tokenizer-level mask 让输出可解析；
2. parser 完成后做 schema validation；
3. 业务层检查范围、权限、引用和安全策略；
4. 工具调用前再确认 side effect 参数，并将原始 JSON 和校验错误保留在 trace。

schema 变更会使 grammar cache 失效。缓存键至少包括 canonical schema hash、tokenizer/vocab revision、grammar backend version、case/whitespace policy 和用户租户命名空间。把 Python 对象 id 当 cache key 会在多进程或重启后误命中；把未经 canonicalization 的 JSON 文本当 key 会让等价 schema 产生重复编译。

### 29.5.3 编译、缓存和回退

grammar 编译可能比短请求的整个 decode 更慢。scheduler 应把编译放在 admission 前或异步编译池，并设置超时、最大状态数、最大递归深度和缓存容量。编译未完成时有三种策略：排队等待、先用 unconstrained draft 再由 target mask（需保证 exactness），或拒绝请求。不能悄悄回退到 unconstrained 输出后仍标记为“JSON guaranteed”。

缓存命中不能跨模型/租户无条件共享。若 grammar 状态包含用户提供的 regex，拒绝路径和错误消息也可能泄漏 schema；多租户部署需要限制 schema 大小、编译 CPU 和 cache eviction。指标应分开 `grammar_compile_ms`、`grammar_cache_hit_total`、`grammar_mask_ms`、`grammar_dead_end_total` 和 schema hash 的高基数标签（后者通常只记录采样哈希，避免原始敏感 schema 进入 metrics）。

### 29.5.4 Dead end、EOS 与修复

当 grammar 状态没有允许 token 时，常见原因是 tokenizer 没有可消费的字符、schema 与 stop 条件冲突、或模型输出已进入不可修复字符串。强行允许 EOS 会产生截断 JSON；强行允许任意 token 会破坏约束。可靠流程是：

- 保留最近一个可恢复 parser checkpoint；
- 若 grammar 支持，回退 tentative branch，减少 draft k 或关闭 speculative；
- 向 target 请求 residual/重新采样，而不是从草稿复制非法 token；
- 达到重试上限后返回结构化错误（含 request id、grammar state 摘要和已提交 token 数），不上传完整敏感 prompt。

## 29.6 Speculative 与 grammar 的联合协议

### 29.6.1 正确的顺序

一轮联合执行可抽象为：

1. 从 committed prefix 读取 grammar state g 和 KV references；
2. 用相同 g 约束 draft，生成 k 个 tentative token 与 q logits；
3. 对每个 draft token 模拟 grammar transition，遇到 dead end 就截断候选树；
4. target 对剩余候选做一次链或树验证，计算 p；
5. 在每个位置用同一 allowed set 做 p^g/q^g 的 acceptance；
6. 提交接受前缀与 correction，按提交 token 更新 g 和 KV；
7. rollback 未提交 branch，释放 page/slot，记录 rejection 原因；
8. 运行 stop/schema validator，再向客户端发送 chunk。

如果第 3 步只是把非法 token 标成 reject，却仍用未 mask 的 q 概率计算 acceptance，输出分布不是约束后的 p^g。若第 6 步先发送 draft token、等 target 后再撤回，客户端看到的流式协议无法撤销，属于不可接受的状态泄漏。

### 29.6.2 Acceptance 与 grammar 的相互作用

grammar 越严格，p/q 的允许集合越小，可能提高 q 与 p 的相似度，也可能让 draft 大量质量落在被 mask 的 token 上而降低有效 acceptance。不能从 unconstrained acceptance 预测 JSON acceptance。应分层记录：候选数、grammar-valid 候选数、target accepted、grammar pre-reject、target residual reject、bonus token 和最终提交 token。

一个实用的自适应策略是根据滑动窗口的有效 acceptance 和 grammar dead-end 率调整 k：高 acceptance 且无 dead-end 时扩大 k；低 acceptance、grammar 编译慢或 p99 ITL 超过阈值时缩小 k；连续窗口失败则回退 target-only。回退状态要绑定 request，不应让一个复杂 schema 的失败改变整个 server 的全局 speculative 参数。

### 29.6.3 Rollback 的 KV 和 page 细节

对 paged KV，tentative branch 可以分配独占 page，接受后把 page ref 变成 committed；若 page 内混有 accepted 与 rejected token，则更新有效长度并清零/覆盖 rejected slots，避免下次 attention 读到旧值。对 radix cache，未提交 suffix 不应插入可共享节点，除非节点有 tentative lease；否则另一个请求可能命中未验证 token。树验证要为每个节点保存 parent page ref 和 grammar state snapshot，回滚只释放未被接受的子树。

copy-on-write 与引用计数的 race 是常见事故：请求 A 和 B 共享前缀，A speculative 分支触发 page split 时若忘记增加 refcount，B 的 decode 可能读到已被覆盖的 page。测试需要并发取消、同前缀多租户、分支在 page boundary 前后、以及 grammar reject 后立即插入新请求。

## 29.7 Serving scheduler：把 draft/verify 当作资源

### 29.7.1 调度步的扩展不等式

第 28 章的 token budget 需要扩展为候选与验证预算。对 batch B，请求 i 的 draft 长度为 (k_i)，最终提交 (a_i+1)（bonus/correction）个 token。一次迭代约束为

[
\sum_i (a_i+1) \le C_{commit},\quad
\sum_i k_i \le C_{draft},\quad
\sum_i V(k_i, b_i) \le C_{verify},
]

其中 (b_i) 是树分支节点数，(V) 近似 target attention、logits 和 mask 成本。显存约束还包括 committed KV、tentative KV、grammar state、tree metadata、draft activations 和 logits buffer：

[
M_{resident}+M_{tentative}(\sum k_i,\sum b_i)+M_{grammar}+M_{staging}\le M_{GPU}.
]

如果 scheduler 只按最终提交 token 计费，candidate tree 可以在验证时 OOM；只按 draft token 限制，则低 acceptance 时浪费 target budget。实现应在 admission 时估算上限，并在运行时根据实际 accepted/rollback 回收。

### 29.7.2 continuous batching 的两个安全点

在 iteration-level batching 中，draft 可以在 decode step 前或与 target worker 协同执行。安全点通常是：

- **draft admission point：** request 已拥有 committed KV、grammar state 和随机数 lease；分配 tentative slots 后才生成候选。
- **commit point：** target verification、residual sampling、grammar transition、KV event 全部完成；将 accepted token 原子地加入输出队列，随后才释放 rejected slots。

新请求不能在 tentative page 尚未清零时复用物理 page；客户端也不能在 commit event 前收到 token。异步 CUDA stream 下需要 event dependency 或 stream wait，不能把 Python list 更新当作 GPU memory fence。

### 29.7.3 优先级、公平和背压

speculative group 可能一次推进多个 token，因此“每请求一轮”会让高 acceptance 请求占用更多 GPU；反过来按 committed token 计费会惩罚低 acceptance、复杂 grammar 的请求。可采用 weighted deficit round robin：每请求有 deficit，draft candidate 消耗较低权重，target verify 和 grammar mask 按真实 cost 消耗较高权重；达到 TTFT deadline 的新请求保留 prefill 配额。

流式客户端慢时，发送队列背压不能让 engine 继续无限提交 speculative token。`max_unflushed_tokens` 应计入 request budget；超过上限，暂停 draft，保留 committed KV，必要时取消 tentative branch。否则内存和输出顺序会失控，且客户端断开后的 draft 计算全是浪费。

### 29.7.4 分布式 TP/DP 与 PD disaggregation

在 tensor parallel 中，draft 和 target 的 logits/all-reduce 必须与 TP rank 同步；Medusa/EAGLE head 若只在 rank 0 运行，需广播 hidden 或候选，通信可能抵消收益。data parallel/DP attention 下，同一 grammar schema 的 cache 可本地编译，但跨 replica 共享需要版本和内存一致性，不应假设命中。

Prefill/decode disaggregation（PD）把长 prompt prefill 与 decode 放在不同 worker。speculative draft 通常属于 decode 侧；若 target verify 需要远端 KV，候选树会扩大传输和缓存 lease。必须测量 NIC bytes、KV transfer、校验延迟和失败重试；不能把单机 acceptance 提升直接乘到 PD 集群吞吐。

## 29.8 性能模型：从 acceptance 到 goodput

### 29.8.1 单请求期望步数

令 draft 长度 k，逐 token acceptance 概率近似为 r，忽略位置相关性。接受前缀长度 A 的期望是

[
E[A]=\sum_{j=1}^{k}r^j=\frac{r(1-r^k)}{1-r},
]

本轮还提交一个 correction/bonus，因此期望进度 (E[S]=E[A]+1)。若每轮 draft cost 为 (C_D(k))，target verify cost 为 (C_T(k))，普通 target 单 token cost 为 (C_1)，粗略 speedup 是

[
Speedup\approx\frac{E[S]C_1}{C_D(k)+C_T(k)+C_{grammar}(k)}.
]

这不是性能定律：target batch 越大，C_T(k) 可能接近一次 forward 而非 k 倍；KV 长度、树分支、GPU occupancy、kernel fusion、网络和 scheduler 会改变常数。用它做决策时应把 measured `draft_ms`、`verify_ms`、`grammar_ms` 替换进去，并给出置信区间。

当 r 很低时，E[S] 接近 1，draft 只是额外开销；当 r 接近 1 时，k 太大又会增大临时 KV、grammar mask 和 rollback。最佳 k 通常在 acceptance 曲线、p99 ITL 和显存水位的交点，而不是固定 4/8/16 的神奇值。

### 29.8.2 端到端延迟分解

对 request i，建议记录

[
T_{e2e}=T_{queue}+T_{compile}+T_{prefill}+\sum_{round}(T_D+T_{verify}+T_{grammar}+T_{commit}+T_{network}).
]

`T_verify` 需要区分 target GPU compute、候选 tree mask、logits copy 与 residual sample；`T_grammar` 分为 cache lookup、状态 transition、mask kernel 和 schema validator。TTFT 发生在第一个 committed token，而不是第一个 draft token。ITL/TPOT 只对客户端收到的 committed token 计时；把 speculative candidate 也计入输出会虚高 tok/s。

### 29.8.3 成本、功耗和容量

draft 降低 target FLOPs 但增加 draft FLOPs、显存、功耗和运维复杂度。容量模型应同时估算：每秒 target requests、draft requests、GPU memory watermark、grammar compile CPU、网络 bytes、重试/回退率和错误预算。若 GPU 受 memory bandwidth 限制，减少 target kernel 次数可能显著；若受 scheduler/网络限制，draft 可能几乎无效。

报告中可以给出“每美元有效 token”或“每瓦有效 token”，但必须把 rejected candidate、grammar reject、重试和取消的 token 计入分子/分母定义。对高价值工具调用，错误率和 schema violation 的代价可能远高于 10% throughput；goodput 要写明质量和 SLO 门槛。

## 29.9 vLLM/SGLang 源码验收方法

### 29.9.1 vLLM 路径：scheduler、worker、guided decoding

vLLM 版本持续演进，建议从目标 tag 的 architecture overview、speculative decoding docs、structured outputs docs 和源码入口开始，固定 commit 后再下结论。审计 checklist：

1. 找到 request 从 API server 到 engine core 的字段：draft 配置、guided decoding backend、seed、stop、max tokens 是否完整传递。
2. 在 scheduler 中确认 waiting/running/swapped 的状态机、`num_scheduled_tokens`/token budget、prefill/decode 混合和 preemption 是否为 speculative 预留临时空间。
3. 在 model runner/worker 找到 candidate token、target logits、accepted count、bonus token、block table/slot mapping 和 CUDA event 的对应关系。
4. 检查 KV cache manager 对 accepted/rejected branch 的 allocation、copy-on-write、refcount、clear 和 free；不要只看 `allocated_blocks` 计数。
5. 检查 guided decoding/grammar backend 的 cache key、编译超时、mask 应用位置，以及 streaming output 是否只发送 committed token。
6. 对 metrics 记录 acceptance、draft/target latency、grammar compile、queue/prefill/decode、preemption 和 KV usage；确认 label cardinality 不会把 schema 原文写入 Prometheus。

官方 docs 的“支持 speculative/structured outputs”是能力声明；是否与特定 quantization、TP、beam、LoRA、tool call、chunked prefill 组合，必须运行目标版本的 tests/benchmarks。

### 29.9.2 SGLang 路径：scheduler、radix cache、grammar

SGLang 把 structured program、RadixAttention、grammar/regex、speculative backend 和 continuous batching 紧密结合。源码审计应从 request runtime、scheduler loop、radix cache、constraint/grammar state 和 benchmark scripts 追踪：

- radix cache 是否只插入 committed prefix，tentative speculative branch 是否有 lease；
- grammar state 是否作为请求状态随 batch 重排，是否在 radix 命中后恢复正确 parser checkpoint；
- tree/Medusa/EAGLE candidate 的 node id、parent、position 和 attention mask 是否在数据并行/张量并行下保持一致；
- structured output 的 HTTP stream 是否在 JSON object 完整或允许增量时发送，取消后 cache ref 是否释放；
- `bench_serving` 的 TTFT/ITL/TPOT/E2E 定义是否把 speculative accepted token、grammar compile 和 warmup 纳入。

Radix hit 只能说明共享前缀可复用，不能说明 grammar schema、采样参数或租户策略相同。把 schema 状态放在全局 radix key 中会减少命中但更安全；忽略 schema 则可能把一个请求的可见 token 误用于另一个约束。

### 29.9.3 最小源码证据包

每个版本提交以下证据：

- commit SHA、Python/CUDA/driver、模型与 tokenizer revision；
- 相关源文件路径和行号（scheduler、speculative worker、grammar backend、KV manager、metrics）；
- `--help` 或配置快照，明确 draft backend、k、grammar backend、cache、TP/DP；
- 单元测试命令与结果，特别是 acceptance/distribution、rollback、grammar dead-end、cancel、stream ordering；
- 受控 benchmark 的请求 JSONL、warmup、并发、输入输出长度、schema、seed、日志和 request-level metrics；
- 与 target-only baseline 的差异解释，以及哪些结论仍是推断。

不要把博客 headline 或默认配置截图当作源码证据。版本漂移时，manifest 中应标记“需复核”，而不是沿用旧行号。

## 29.10 CPU-only lab：精确性、grammar 和调度的可复现实验

实验脚本 `labs/ch29_speculative_lab.py` 只用 Python 标准库，实现三层 toy：

1. `exact_verify`：给定 p、q、draft token，按 min(1,p/q) 接受，拒绝时从 positive residual 采样 correction；
2. `constrained_speculative`：使用小型 JSON-like 状态机，在 draft 和 target 上应用同一 mask，统计 draft/accepted/reject/grammar；
3. `simulate`：用 baseline/speculative/lookahead 三种模式运行多个请求，按并发和 `max_batch_tokens` 估算逻辑时间。

运行示例：

```bash
python3 labs/ch29_speculative_lab.py \
  --seed 29 --requests 24 --concurrency 6 --output-len 32 \
  --draft-len 4 --accept-rate 0.72 --mode baseline \
  --output reports/ch29-target-only.json

python3 labs/ch29_speculative_lab.py \
  --seed 29 --requests 24 --concurrency 6 --output-len 32 \
  --draft-len 4 --accept-rate 0.72 --mode speculative --grammar \
  --grammar-reject-rate 0.10 \
  --output reports/ch29-speculative-json.json

python3 tests/test_ch29_speculative_lab.py
```

实验中的 `logical_time_ms` 是人为成本函数产生的离散时钟，不是 GPU wall time；`accept-rate` 是草稿质量参数，不是论文或生产测量。测试只证明协议方向：residual correction 在拒绝时推进一个 token；mask 后非法 token 概率为零；相同 seed 可复现；所有请求完成且最终输出 token 数符合合同。

### 29.10.1 如何读实验结果

先比较 target-only 与 speculative 的 `steps`、`target_tokens`、`draft_tokens`、`accepted_tokens`、`acceptance_rate` 和 TTFT；再比较 grammar 开关下的 `grammar_rejects`。若 speculative steps 下降但 logical time 上升，说明 draft/verify 常数抵消了候选收益；这正是生产中应关注的信号。若 grammar rejects 上升而 output count 不变，不能推出 JSON 语义正确，只能说明 toy 的 mask/retry bookkeeping 工作。

做受控扫描时一次只变一个参数：k=1/2/4/8；accept rate=0.2/0.5/0.8/1.0；grammar reject=0/0.1/0.5；并发=1/6/24；token budget=8/32/128。保存 stdout JSON 和命令行，避免只复制摘要表。对每个点至少重复多个 seed，报告均值、p95 和随机波动；toy 的 Bernoulli 不是硬件噪声模型。

## 29.10.2 理解检查：提交边界与可观测性

在进入真实 GPU benchmark 前，逐项检查：第一，客户端是否只收到 committed token，draft/rollback 是否永远不可见；第二，target logits、grammar state、KV slot 和 request generation id 是否能在同一个 trace 中对齐；第三，acceptance、grammar reject、target correction、compile time、TTFT/ITL/TPOT 是否分开统计；第四，取消、超时、schema dead end 和 page OOM 是否有可重现的最小样例。若任何一项无法回答，应把结果标成“协议未验收”，不要仅凭 tok/s 开启默认 speculative。

## 29.11 故障模式与排查 runbook

### 29.11.1 输出分布偏移

**症状：** speculative 与 target-only 的 token 频率、logprob 或安全拒绝率显著不同。**优先检查：** p/q 是否在同一截断和 temperature；acceptance 是否误用 batch 平均概率；拒绝时是否从 (p-q) 的正部采样；grammar mask 是否在两者归一化前后顺序一致；bonus token 是否漏掉。**隔离：** 关闭 grammar、k=1、使用人工小词表分布逐 token 穷举；再逐步开启 top-p、tree 和量化。

### 29.11.2 Grammar 合法率下降或死循环

**症状：** JSON parse error、重复重试、`grammar_dead_end` 激增。**检查：** tokenizer transition 是否消费完整 token；状态是否在 batch reorder/beam/tree 后正确迁移；EOS/stop 与 schema 是否冲突；schema canonicalization/cache key 是否命中旧版本；拒绝后 parser checkpoint 是否回到 committed state。**止损：** 设置最大 grammar retries/compile timeout，单请求回退 target-only constrained，超过阈值返回明确错误。

### 29.11.3 接受率突然变低

**症状：** target latency 不变但 accepted token 下降，或某些租户/长度分布特别差。**检查：** draft/target model revision、tokenizer、chat template、RoPE scaling、quantization、grammar schema、temperature、LoRA adapter 和上下文长度；统计按位置、请求、schema、语言和前缀命中切片。**动作：** 自适应减小 k；连续窗口低于阈值回退；保留 target-only baseline 以便容量预测。

### 29.11.4 OOM、page 泄漏和尾延迟

**症状：** `allocated` 低但 reserved 高、偶发 OOM、取消后水位不降、p99 ITL 尖峰。**检查：** tentative KV、tree metadata、draft activations、grammar cache、CUDA graph pool、page refcount、swap/recompute 和慢客户端队列；压测 page boundary 与取消。**动作：** 限制 candidate nodes 和 in-flight tentative bytes；在 commit/rollback event 后断言 page/ref 数守恒；把 draft 与 target 放到不同 GPU 或关闭 speculative 做对照。

### 29.11.5 流式顺序和取消竞态

**症状：** 客户端收到后来被 reject 的 token、重复 token、取消后仍有输出。**检查：** stream writer 是否只消费 committed queue；request generation id 是否随异步任务检查；取消是否撤销 draft、verify、detokenize、network flush；重试是否复用旧随机数。**测试：** 在 draft 完成、target logits 返回、grammar compile 完成、commit 前四个点注入 cancel/disconnect，并检查无 token 泄漏与 page 回收。

## 29.12 评测设计：质量、性能和正确性三张表

### 29.12.1 分布/质量表

固定 tokenizer、prompt 数据、seed 集合和 sampling config，比较 target-only 与 speculative+grammar：token-level KL/TV、nucleus inclusion、结构化 parse/schema pass、拒答/安全分类、工具参数校验和业务 end-to-end 结果。对随机采样不要用 exact string match 作为唯一正确性；对 deterministic temperature=0，仍需检查 logits、stop、grammar 和 rollback 是否一致。

### 29.12.2 性能/资源表

每个 request JSONL 至少包含 request id、arrival、input/output token、schema hash、cache hit、draft k、candidate nodes、draft/target/grammar ms、accepted/rejected、target forward、temporary/committed KV bytes、preempt/swap、TTFT、ITL、TPOT、E2E、cancel/error。报告 p50/p95/p99 和分桶（短/长 prompt、共享前缀、schema 复杂度、并发、acceptance）。GPU 端记录 kernel time、memory bandwidth、SM occupancy、PCIe/NIC bytes 和 power；CPU 端记录 grammar compile、parser CPU 与 queue。

### 29.12.3 正确性/安全表

覆盖人工小分布穷举、property-based 随机分布、batch reorder、tree sibling mask、page boundary、quantized logits、top-p、temperature、LoRA、grammar dead-end、schema recursion、cancel/retry、multi-tenant cache isolation。每次失败保存最小 repro：tokenizer revision、p/q logits（可脱敏）、grammar state、candidate tree、seed 和 commit。不要把生产 prompt 原文写入公共 artifact。

## 29.13 设计练习与参考答案要点

1. **练习：** p=[0.5,0.3,0.2]，q=[0.2,0.5,0.3]，草稿 token=1。求接受概率和 residual。**要点：** α=min(1,0.3/0.5)=0.6；Z=0.2；residual ∝ [0.3,0,0]，拒绝时必采 token 0；两条路径 token 1 的总概率为 0.5，证明一步分布恢复 p。
2. **练习：** grammar 允许集合 {0,2}，p mass [0.4,0.4,0.2]，q mass [0.1,0.8,0.1]。先 mask+normalize，再做 acceptance；说明若直接用原始 q 会发生什么。**要点：** p^g=[2/3,0,1/3]，q^g=[1/2,0,1/2]；原始 q 在非法 token 1 上浪费质量，直接 ratio 不代表约束 q，需明确是否接受无偏但低效的 residual。
3. **练习：** k=4，r=0.25/0.75，draft+verify 固定成本分别为 0.3/1.0，target 单 token 成本 1.0，估算哪种 r 值值得 speculative。**要点：** 用 (E[S]=1+r+r^2+r^3+r^4) 与成本比；再声明这只是独立位置、固定成本的推断，真实 kernel/grammar/queue 需测量。
4. **练习：** 两请求共享 radix 前缀，A 的 speculative branch 尚未 commit，B 命中该前缀。指出安全设计。**要点：** tentative node 不能作为可共享 committed node；使用 lease/独占 page 或只把 committed ancestor 插入 radix；取消 A 时释放 branch 并验证 B 的 refcount。
5. **练习：** JSON schema 编译耗时 200 ms，短输出 target-only 只需 40 ms。设计 admission。**要点：** 异步 compile pool、schema canonical hash cache、超时与最大状态数；编译未完成不得宣称 constrained guarantee，可排队或明确拒绝/回退，并把 compile time 纳入 TTFT。

## 29.14 章节交付和边界声明

本章交付：

- `docs/chapters/29-speculative-and-structured-decoding.md`：算法、证明、grammar、scheduler、性能模型、源码验收和 runbook；
- `labs/ch29_speculative_lab.py`：CPU-only exact accept-reject、有限状态 grammar、baseline/speculative/lookahead scheduler toy；
- `tests/test_ch29_speculative_lab.py`：拒绝校正、mask、grammar、完成性、可复现和 CLI JSON 合同；
- `reports/ch29-speculative-report.md`：命令、结果、观察、边界与下一步硬件实验；
- `evidence/ch29-speculative-manifest.json`：Leviathan、Medusa、EAGLE、SpecInfer、Lookahead、vLLM/SGLang 官方文档/源码和本地测量的证据索引；
- `website/sidebars.ts`：加入第 29 章。

**边界：** toy 没有真实 logits、tokenizer、CUDA kernel、GPU memory visibility、vLLM/SGLang API、JSON Schema 全语义、分布式通信、HTTP/TLS 或生产安全策略。论文 headline、accepted length、grammar parse pass 和 toy logical tok/s 都不是质量、SLO、容量或成本承诺。真实部署必须 pin 版本，运行 target-only 对照、分布/grammar contract、request-level trace、rollback/cancel/多租户测试，再决定是否为某个 workload 开启 speculative 或结构化约束。

## 29.15 参考资料（主来源）

- Leviathan, Kalman, Matias，《Fast Inference from Transformers via Speculative Decoding》，[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)。
- Chen 等，《Accelerating Large Language Model Decoding with Speculative Sampling》，[arXiv:2302.01318](https://arxiv.org/abs/2302.01318)。
- Cai 等，《Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads》，[arXiv:2401.10774](https://arxiv.org/abs/2401.10774)。
- Li 等，《EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty》，[arXiv:2401.15077](https://arxiv.org/abs/2401.15077)。
- Sun 等，《SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification》，[arXiv:2305.09781](https://arxiv.org/abs/2305.09781)。
- Fu 等，《Break the Sequential Dependency of LLM Inference Using Lookahead Decoding》，[arXiv:2402.02057](https://arxiv.org/abs/2402.02057)。
- vLLM 官方 speculative decoding 文档：[docs.vllm.ai](https://docs.vllm.ai/en/stable/features/speculative_decoding.html)；structured outputs 文档：[Structured Outputs](https://docs.vllm.ai/en/stable/features/structured_outputs.html)。
- vLLM 架构与源码：[architecture overview](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md)、[repository](https://github.com/vllm-project/vllm)。
- SGLang 结构化程序论文与文档：[SGLang paper](https://arxiv.org/abs/2312.07104)、[speculative decoding docs](https://docs.sglang.ai/advanced_features/speculative_decoding.html)、[repository](https://github.com/sgl-project/sglang)。
