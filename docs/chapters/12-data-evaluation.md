---
id: ch12-evaluation
title: 数据预处理与评估：tokenization、packing、采样、污染与统计有效性
slug: /chapters/12-data-evaluation
description: 把数据管道、基准评估、统计推断与人工质量门连成一条可审计链路，避免 token、样本和指标定义上的隐性偏差
sidebar_position: 12
level: systems
prerequisites:
  - ch01-python
  - ch04-performance-math
  - ch06-pytorch-execution
  - ch10-training-ops
learning_objectives:
  - 能解释 tokenizer 的词表、规范化、特殊 token 和版本漂移如何改变训练成本与模型行为
  - 能设计无泄漏的 sequence packing、边界掩码和 loss 归一化策略
  - 能计算混合数据的采样分布、温度采样、课程学习和有效样本量
  - 能用精确匹配与近似指纹检查训练集和 benchmark 的污染，并报告误报与漏报
  - 能用重复运行、置信区间、bootstrap、效应量和功效分析判断分数差异是否可信
  - 能区分测量不确定性、模型不确定性、数据分布漂移和评审者不一致
  - 能建立包含代码、数据、tokenizer、随机状态和硬件的可复现实验清单
  - 能设计人工评估的抽样、盲法、标注规范、仲裁和质量门
  - 能在 CPU-only 环境运行最小评估实验并解释失败模式
  - 能明确评估结论的适用边界、隐私边界和安全升级路径
estimated_hours: 20
hardware: CPU-only baseline; GPU optional
risk_level: L2
last_verified: 2026-10-05
---

# 第12章　数据预处理与评估：tokenization、packing、采样、污染与统计有效性

> 评估不是训练结束后才做的一张表，而是从原始字节到最终结论的一条测量链。一个看似更高的分数，可能来自 tokenizer 把同样的文本切成更少的 token，可能来自 packing 时把两个样本的答案互相看见，也可能来自 benchmark 已经混进训练集。反过来，一个真正有用的改进也可能被低功效的抽样、评审者噪声或不恰当的分母掩盖。本章的主线是建立“可解释、可复现、可审计”的评估契约：先明确测量单位和数据边界，再实现确定性的预处理，最后用统计与人工质量门决定是否发布。

## 12.1 先写评估契约，再写脚本

### 12.1.1 从问题到可观测量

评估契约至少回答六个问题：

1. 要回答什么问题？是下一个 token 预测、指令遵循、事实准确性、代码编译，还是安全拒答？
2. 观察单位是什么？字符、token、样本、对话、用户任务、文档，还是时间窗口？
3. 分母是什么？每个 token 等权、每个样本等权、每个任务等权，还是按业务暴露量加权？
4. 哪些数据允许进入训练、调参和最终测试？测试集何时冻结、由谁保管？
5. 需要多大差异才算有意义？给出最小实际效应（minimum practical effect, MPE），不要只追求统计显著。
6. 谁可以复核？保存哪些代码、清单、随机种子、日志和人工判定证据？

把契约写成一页表格或 YAML 比在 notebook 里口头约定更可靠。例如：

```yaml
metric: exact_match
unit: task
aggregation: macro_mean_by_task
primary_question: "新 tokenizer 是否让多语种任务在固定 FLOPs 下保持准确率？"
primary_split: test_v3_frozen
mpe: 0.5_percentage_point
seeds: [17, 29, 41, 53, 71]
ci: bootstrap_95_percentile
contamination_policy: exact_or_13gram_jaccard>=0.8 -> exclude_and_report
stop_rule: "五个种子全部完成后再看主要结论"
```

契约中的 `unit` 和 `aggregation` 尤其重要。若英语任务占 90%，按所有样本微平均会把少数语言的变化淹没；按任务宏平均则每个任务同权，方差通常更大。两者都可以报告，但必须明确哪一个是主要结论，不能在看到结果后挑分母。

### 12.1.2 数据流水线的证据链

把流水线画成可追踪的有向图：

```text
raw bytes
  -> decode/normalize (normalizer_version)
  -> tokenize (tokenizer_hash, vocab_size)
  -> filter/dedup (ruleset_hash)
  -> shard/sample (sampler_seed, weights)
  -> pack (max_len, boundary_mask)
  -> train/eval (code_commit, env_lock)
  -> aggregate (metric_version)
  -> human gate (rubric_version, annotator_ids)
```

每个节点输出一个 manifest，至少包含输入文件哈希、记录数、token 数、丢弃原因计数、随机种子和代码版本。不要只保存最终 parquet；如果无法回答“某个样本为何被丢弃”，就不能审计采样偏差。manifest 可用 JSON Lines，便于增量追加和流式验证。

### 12.1.3 三个常见但危险的捷径

- 用训练 loss 代表真实能力：loss 是分布加权的平均量，不能替代任务级指标。
- 在同一份测试集上反复调参：即使没有显式查看答案，选择行为也会让测试集成为验证集。
- 把 token 数当成样本数：不同语言、代码和数学文本的 token/字符比差异很大，成本和方差都会被误判。

## 12.2 Tokenization：词表是测量仪器的一部分

### 12.2.1 规范化、切分和特殊 token

典型 tokenizer 包含三层：

1. **规范化（normalization）**：Unicode NFC/NFKC、大小写、空白、全角半角、控制字符处理。
2. **预分词（pre-tokenization）**：按空白、标点、数字或语言规则拆出候选片段。
3. **子词模型**：BPE、Unigram、WordPiece 或字节级方案，把候选片段映射为词表 id。

每层都改变序列长度和可逆性。NFKC 会把某些兼容字符折叠，可能损失法律文本中的原貌；去除零宽字符有助于防止注入，但也可能破坏阿拉伯语或南亚文字。必须记录 normalizer 版本和可逆性测试结果：`decode(encode(x)) == x` 不成立时，要说明允许的规范化差异。

特殊 token（BOS、EOS、PAD、UNK、工具调用标记）应有唯一 id、明确插入规则和损失掩码。把 PAD 误当成 EOS 会让模型学会提前停止；把系统提示和用户内容使用同一损失权重，会掩盖指令跟随能力。对聊天模板，建议保存模板源文件和渲染后的 golden examples，各版本做 diff。

### 12.2.2 BPE 合并与序列长度

BPE 从字符或字节开始，反复选择频率最高的相邻对合并。若词表大小为 (V)，合并规则为 (M)，序列长度 (L) 决定自回归训练的注意力成本 (O(L^2))。在固定字符预算下，token 平均长度提高 10% 减少序列维度上的计算，还可能将注意力矩阵成本降低约 19%（((1/1.1)^2)），但这不是免费收益：罕见词拆得更碎会增加长尾序列，跨语言公平性也可能变差。

比较 tokenizer 时不能只看平均 tokens/字符。至少按语言、脚本、文档类型、数字/代码、URL 和表情符号分层报告 p50/p95。p95 反映最坏序列长度及截断风险，均值会掩盖少数语言的灾难性膨胀。

### 12.2.3 Tokenizer 漂移与可比性

训练中途升级 tokenizer 等于更换坐标系：词表 id、embedding 行、特殊 token 插入位置、数据游标和 checkpoint 都可能不兼容。安全做法是：

- 将 `tokenizer.json`、词表、normalizer 和模板一起内容寻址，记录 SHA-256
- 在 checkpoint 中保存 tokenizer hash，加载时强校验，禁止静默 fallback
- 对新旧 tokenizer 在固定语料上输出长度、OOV/UNK、脚本分层和边界样本 diff
- 若必须迁移，单独训练或映射 embedding，并把迁移前后结果视为不同实验系列

### 12.2.4 CPU 实验一：四种切分的长度与可逆性

以下脚本仅用标准库，比较字符、空白词、字节和简化 BPE。它不是生产 tokenizer，而是让读者看到“同一文本，成本单位已经不同”。

```python
from collections import Counter
import hashlib, math, unicodedata

TEXTS = [
    "中文和English混合，含有URL https://example.org/a?x=1。",
    "def f(x): return x ** 2  # code",
    "नमस्ते दुनिया — multilingual",
    "emoji 😀😀 and zero\u200bwidth",
]

def normalize(s):
    # 生产系统应固定规范化版本并保留原文哈希
    return unicodedata.normalize("NFC", s)

def char_tok(s):
    return list(s)

def word_tok(s):
    # 仅为教学：把空白切分，中文整句会成为一个片段
    return s.split()

def byte_tok(s):
    return list(normalize(s).encode("utf-8"))

def train_bpe(texts, merges=30):
    seqs = [list(normalize(t)) for t in texts]
    rules = []
    for _ in range(merges):
        pairs = Counter()
        for seq in seqs:
            pairs.update(zip(seq, seq[1:]))
        if not pairs:
            break
        pair, _ = pairs.most_common(1)[0]
        merged = "".join(pair)
        rules.append(pair)
        for i, seq in enumerate(seqs):
            out, j = [], 0
            while j < len(seq):
                if j + 1 < len(seq) and (seq[j], seq[j + 1]) == pair:
                    out.append(merged); j += 2
                else:
                    out.append(seq[j]); j += 1
            seqs[i] = out
    return rules

def apply_bpe(s, rules):
    seq = list(normalize(s))
    for pair in rules:
        out, j = [], 0
        while j < len(seq):
            if j + 1 < len(seq) and (seq[j], seq[j+1]) == pair:
                out.append("".join(pair)); j += 2
            else:
                out.append(seq[j]); j += 1
        seq = out
    return seq

rules = train_bpe(TEXTS)
for name, fn in [("char", char_tok), ("word", word_tok), ("byte", byte_tok),
                 ("bpe", lambda s: apply_bpe(s, rules))]:
    lengths = [len(fn(t)) for t in TEXTS]
    print(name, lengths, "mean=%.2f" % (sum(lengths)/len(lengths)))
print("bpe_rules_sha256", hashlib.sha256(repr(rules).encode()).hexdigest())
```

自审问题：如果把 `normalize` 改成 NFKC，哪些样本会改变？如果把 `word_tok` 用于中文，为什么长度看似很短却不代表更好的建模？实验输出中的规则哈希用于演示，生产环境应对完整 tokenizer 文件哈希。

## 12.3 Packing：提高利用率，但边界必须可证明

### 12.3.1 Padding 与 packing 的差别

固定长度 batch 的 padding 简单可靠，却浪费短样本的计算。packing 把多个样本拼到一个长度为 (L_{max}) 的序列中，减少 PAD。若样本长度为 (l_i)，无 packing 的利用率是 (sum_i l_i/(B L_{max}))；最佳装箱后的利用率接近 (sum_i l_i/(K L_{max}))，其中 (K) 是所需包数。离线或有限缓冲窗口内的 first-fit-decreasing（FFD）是常见近似：先按长度降序，再放入第一个可容纳的包。真正无缓冲的在线流不能预先全局排序；小规模实例可用整数规划或枚举核对装箱质量。

### 12.3.2 三种边界语义

1. **独立注意力（block-diagonal mask）**：包内每个样本只能看自己及其左侧 token。最安全，内核实现复杂。
2. **文档边界标记但全注意力**：通过 EOS 告诉模型边界，却仍允许跨样本看见内容。训练效率高但存在信息串扰，不能用于需要严格独立的评估。
3. **跨样本因果注意力**：后一个样本可以看前一个样本，适合把连续文档视作一个流，不适合把样本当独立观测。

若任务是 next-token 语言建模，跨文档上下文有时是预期的；若任务含答案标签或隐私隔离，必须用 block-diagonal 或在样本边界重置状态。配置里要显式写 `attention_boundary=independent|streaming`，不能仅靠注释。

### 12.3.3 Loss 归一化

pack 后每个样本长度不同，直接对所有 token 求平均等价于长文档权重大。可选策略：

- **token mean**：(mathcal{L}=sum_i ell_i/sum_i n_i)，适合关注每个 token 的预测。
- **sample mean**：(mathcal{L}=rac{1}{N}sum_i rac{1}{n_i}sum_jell_{ij})，每个样本等权。
- **task-balanced mean**：先按任务或语言求均值，再对组求宏平均。

报告中应同时保存分子、分母和掩码计数，便于复核。一个常见 bug 是把 PAD 的 label 设为 0 而不是 ignore_index，导致模型学会高频 PAD。

### 12.3.4 CPU 实验二：FFD packing 与边界检查

```python
from dataclasses import dataclass

@dataclass
class Item:
    sid: str
    n: int

def pack_ffd(items, capacity):
    if capacity <= 0 or any(x.n <= 0 or x.n > capacity for x in items):
        raise ValueError("先按契约处理空样本与超长样本")
    bins = []  # {"used": int, "items": [Item]}
    for item in sorted(items, key=lambda x: x.n, reverse=True):
        placed = False
        for b in bins:
            if b["used"] + item.n <= capacity:
                b["items"].append(item); b["used"] += item.n
                placed = True; break
        if not placed:
            bins.append({"used": item.n, "items": [item]})
    return bins

items = [Item(f"s{i}", n) for i, n in enumerate([7, 6, 5, 5, 4, 3, 2, 2, 1])]
bins = pack_ffd(items, capacity=8)
for i, b in enumerate(bins):
    print(i, b["used"], [x.sid for x in b["items"]])
    assert b["used"] <= 8
    # block-diagonal mask 的简化验证：不同样本之间不应存在可见边
    offsets, p = {}, 0
    for x in b["items"]:
        offsets[x.sid] = (p, p+x.n); p += x.n
    for a, (a0, a1) in offsets.items():
        for c, (c0, c1) in offsets.items():
            if a != c:
                assert not (a0 < c1 and c0 < a1), "区间重叠"
print("bins", len(bins), "utilization", sum(x.n for x in items)/(len(bins)*8))
```

代码只检查装箱区间不重叠；生产实现还需要验证 attention mask、position id、loss mask 和跨设备切分。建议构造两个极端样本：样本 A 的答案是“猫”，样本 B 的提示里出现“猫”。在独立 mask 下，B 的 loss 不应因 A 改变；把 A 的 token 替换为随机符号再比较，这是一个易于自动化的边界测试。

### 12.3.5 Packing 的诊断与在线监控

离线利用率高不代表线上吞吐稳定。需要记录每个包的有效 token 数、PAD 数、样本数、最大/最小长度、重排等待时间和跨 shard 边界。若长度分布出现双峰，FFD 可能把许多长样本单独放置，GPU 利用率下降；此时可以按长度分桶，但要避免分桶本身改变语言或任务分布。在线监控应按数据源和语言分层，否则一个高占比的英文源会掩盖少数语言的严重 padding。

评估时尤其要检查 position id 是否在边界重置。某些模型使用绝对位置 embedding，跨文档继续递增会让后半段样本处在训练中很少见的位置；另一些模型使用相对位置，重置与否影响较小。把 `position_policy` 写进 manifest，并在短序列单元测试中比较重置和不重置的 logits 差异。若使用 flash-attention 或 paged attention，需验证内核实际应用了边界元数据，而不是仅在 Python 层创建了一个未传入内核的 mask。

## 12.4 采样与课程：分布决定你测量的对象

### 12.4.1 混合数据的三层权重

设数据源 (k=1,dots,K)，原始大小为 (N_k)，目标采样概率为 (p_k)。一次 epoch 的期望样本数是 (p_k N)，并不等于使用全部源数据。常见三层权重：

- **源权重** (p_k)：新闻、代码、书籍、对话各自占比
- **样本权重** (w(x))：质量分、去重分、语言平衡、难度
- **时间权重**：最近数据或课程阶段的指数衰减

温度采样常写作 (p_kpropto q_k^alpha)，其中 (q_k) 是原始比例；(alpha<1) 抬高小源，(alpha>1) 强化大源。温度并不能自动消除重复：一个小而高度重复的源可能被过采样，造成记忆化。

### 12.4.2 课程学习与可比性

课程学习把难度或数据域随 step 改变。必须保存每个 step 的采样分布快照，否则恢复训练时可能使用不同分布。为了比较两个模型，至少固定总 token 预算、分层采样计划和评估触发点。若 A 在前半程看了更多简单样本，B 在相同 step 上分数更低并不说明 B 算法更差。

课程学习应有退出条件，例如当验证集上的困难子集达到阈值后再提高难度；不要让“难度”由测试集分数定义，否则把测试集变成控制回路的一部分。对在线数据，记录每个窗口的样本 id 范围和采样概率，以便发现时间漂移。

### 12.4.3 有效样本量与相关性

重复采样和近重复样本降低有效样本量。若样本相关系数近似为 (
ho)，每簇大小为 (m)，设计效应 (D=1+(m-1)
ho)，有效样本量 (n_{	ext{eff}}=n/D)。把一百万条高度相似网页当作一百万个独立证据，会严重低估置信区间。评估报告至少按来源、作者、时间或文档簇聚类 bootstrap。

### 12.4.4 CPU 实验三：温度采样与确定性

```python
import random, math

def temperature_probs(raw, alpha):
    z = sum(v ** alpha for v in raw.values())
    return {k: (v ** alpha) / z for k, v in raw.items()}

def draw(probs, n, seed):
    r = random.Random(seed)
    keys, cdf, s = [], [], 0.0
    for k, p in probs.items():
        keys.append(k); s += p; cdf.append(s)
    out = {k: 0 for k in keys}
    for _ in range(n):
        u = r.random()
        for k, t in zip(keys, cdf):
            if u <= t:
                out[k] += 1; break
    return out

raw = {"web": 0.80, "code": 0.15, "dialog": 0.05}
for a in [1.0, 0.7, 0.3]:
    p = temperature_probs(raw, a)
    print("alpha", a, "probs", {k: round(v, 3) for k, v in p.items()})
    print("draw", draw(p, 10000, seed=17))
assert draw(temperature_probs(raw, .7), 100, 17) == draw(temperature_probs(raw, .7), 100, 17)
```

将 `raw` 替换为真实源比例后，检查理论概率与实测计数的误差；误差过大通常意味着整数配额、分片边界或随机种子被错误处理。随机种子只能保证同一实现的确定性，不能让跨库、跨硬件的浮点归约天然一致。

## 12.5 Benchmark contamination：训练看过了，分数就不再是泛化证据

### 12.5.1 污染的层次

- **精确污染**：测试样本或其答案原文出现在训练数据。
- **近重复污染**：改写、HTML 清洗、大小写或标点变化后仍高度相似。
- **模板污染**：题干不同，但生成模板、选项模式或答案键被训练。
- **知识污染**：训练数据包含测试集发布后的解答、讨论或排行榜。
- **评估回路污染**：团队反复查看测试结果并据此调提示词、过滤规则或解码参数。

污染不是二元标签。建议报告“样本级命中率、token 覆盖率、命中来源和相似度分布”，并区分训练前已存在的公共资料与训练团队主动抓取的资料。

### 12.5.2 精确和近似检测

精确检测：对规范化后的文本计算 SHA-256，在流式哈希集合中查找。近似检测可用字符/词 n-gram 的 MinHash、SimHash 或 Bloom filter。阈值必须在独立的正负对上校准，不能凭经验设一个“0.8 就是污染”。长文档相似度高不等于包含答案；可把 benchmark 题干、选项、解析分别检测，避免只看整文档平均值。

一个可审计的规则示例：

```text
normalize_v3 后，13-gram MinHash Jaccard 估计 >= 0.8，且至少覆盖题干 60% -> review
题干与答案组合的精确 hash 命中 -> exclude_and_report
仅公共百科背景重叠，未命中题干/选项 -> retain_with_note
```

所有“排除”都要保留被排除样本 id 和原因。没有命中的样本也不能声称“没有污染”，只能说“在当前索引、规范化和阈值下未发现”。

### 12.5.3 时间切分与隐藏测试

对会随时间变化的知识任务，使用发布日期之前的训练数据和之后的测试数据，并冻结抓取时间。隐藏测试集应由独立团队生成或加密保管，访问日志可审计。若无法保证隔离，改用“污染敏感”的报告：同时给出公开测试、时间外推测试和新编人测集的结果。

### 12.5.4 CPU 实验四：n-gram Jaccard 污染筛查

```python
import re

def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()

def shingles(s, n=5):
    s = norm(s)
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}

def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))

train = [
    "巴黎是法国的首都，位于塞纳河畔。",
    "Python 的列表可以通过 append 添加元素。",
    "太阳系有八颗行星。",
]
test = [
    "巴黎是法国首都，位于塞纳河畔。",  # 改写，可能低于阈值
    "Rust 的所有权系统用于内存安全。",
]
for i, q in enumerate(test):
    scores = [jaccard(shingles(q), shingles(t)) for t in train]
    print("test", i, "max_jaccard", round(max(scores), 3))
```

教学脚本使用字符 5-gram，生产检测应按语言选择 token/字符 n-gram，并对长短文本做长度校正。把阈值设得过低会带来大量人工复核，过高则漏报。阈值的选择应在带标签的污染/非污染对上预注册。

### 12.5.5 污染命中后的分级处置

检测到相似样本后，不要直接把整个数据集判为“污染”。可按证据分级：L0 是仅主题相同；L1 是公共事实或通用代码片段重叠；L2 是题干或选项高度相似但没有答案；L3 是答案、解析或测试用例精确重叠。L0/L1 通常保留并在报告中说明，L2 进入人工复核或敏感性分析，L3 从主结果排除并给出排除前后分数。对于一组高度相关的 L3 样本，应按簇而不是逐行删除，避免残留改写版本继续影响结果。

处置记录需要包括：命中的索引版本、规范化版本、相似度算法和阈值、人工判定理由、样本来源、排除后的新分母，以及是否触发重跑。若 benchmark 由外部方提供，应把命中证据以最小片段和哈希形式回传，避免泄露整份测试集。所有阈值变化都应产生新版本，不要覆盖旧索引。

## 12.6 统计有效性：差异、区间与功效

### 12.6.1 把一次分数拆成变异来源

观测分数可分解为：

[
Y_{s,d,r}=\mu + \tau_s + \gamma_d + (\tau\gamma)_{s,d} + \epsilon_{s,d,r},
]

其中 (s) 是系统/模型，(d) 是数据集或任务，(r) 是随机种子或重复运行。\(	au_s\) 是系统主效应，\(gamma_d\) 是任务难度，交互项说明某系统只在特定域有效，\(epsilon\) 是抽样和测量噪声。至少保留任务级分数，避免只保留一个宏平均后无法估计异质性。

### 12.6.2 置信区间与 bootstrap

单次准确率 (\hat p=k/n) 可用 Wilson 区间；当 (n) 小或 (p) 接近 0/1 时，不要用对称的 (\hat p\pm1.96\sqrt{p(1-p)/n}\) 造成越界。对复杂指标（F1、BLEU、人工胜率）可按任务或文档簇重采样 bootstrap。配对比较时，优先对同一批样本计算差值 (d_i=Y_i^A-Y_i^B)，再 bootstrap (d_i)，方差通常更小。

95% 区间表达“在重复抽样程序下的覆盖行为”，不是“参数有 95% 概率在区间里”。不要把 CI 跨过 0 当作唯一决策规则；结合 MPE、成本和风险。

### 12.6.3 效应量、多重比较和停止规则

若同时比较 20 个模型、10 个任务和多种提示词，偶然显著几乎必然出现。预注册主要比较，或控制 FDR（如 Benjamini–Hochberg），并报告全部探索性结果。提前写停止规则：例如每个主要结论完成 5 个独立种子，或 CI 半宽小于 0.5 个百分点。看到“好看”的中间结果就停止，会夸大效应。

效应量应配合实际意义：准确率提升 0.2 个百分点若延迟增加 3 倍，可能不值得；安全拒答提升 2 个百分点但误拒率上升 8 个百分点，也不能只报前者。

### 12.6.4 功效分析与最小样本量

两比例检验的粗略样本量可由基线比例 (p_0)、目标差异 (\delta)、显著性水平 (\alpha) 和功效 (1-\beta) 估算；但任务级 benchmark 往往是配对、分层且相关，公式只用于规划。更稳妥的方法是用历史任务级分数模拟：按照估计协方差生成多次 A/B 数据，统计在预设检验下检测到 MPE 的比例。若功效不足，优先增加独立任务或减少比较次数，而不是重复同一题。

### 12.6.5 CPU 实验五：配对 bootstrap 与功效模拟

```python
import random, statistics

def paired_bootstrap(a, b, rounds=5000, seed=17):
    assert len(a) == len(b)
    diffs = [x-y for x, y in zip(a, b)]
    r = random.Random(seed); n = len(diffs); means = []
    for _ in range(rounds):
        sample = [diffs[r.randrange(n)] for _ in range(n)]
        means.append(sum(sample)/n)
    means.sort()
    lo = means[int(.025*rounds)]
    hi = means[int(.975*rounds)]
    return sum(diffs)/n, (lo, hi)

# 同一批任务上的分数，范围 0~1
A = [.80,.60,.90,.70,.75,.55,.88,.66,.72,.81]
B = [.78,.59,.88,.72,.74,.54,.86,.65,.70,.80]
print("mean_diff, ci95", paired_bootstrap(A, B))

def power_sim(n_tasks=20, true_delta=.02, sigma=.06, reps=3000, seed=1):
    # 明确假设：独立任务差值 ~ N(delta, sigma^2)，sigma 已知
    # 单侧 H0: delta <= 0，alpha=0.05；检验阈值固定而不随观察结果选取
    r = random.Random(seed); hits = 0
    threshold = 1.6448536269514722 * sigma / (n_tasks ** .5)
    for _ in range(reps):
        diffs = [r.gauss(true_delta, sigma) for _ in range(n_tasks)]
        if statistics.mean(diffs) > threshold:
            hits += 1
    power = hits/reps
    mc_se = (power*(1-power)/reps) ** .5
    return round(power, 3), round(mc_se, 4)
for n in [10, 20, 50, 100]:
    print("n_tasks", n, "power, monte_carlo_se", power_sim(n_tasks=n))
print("null_rejection", power_sim(n_tasks=50, true_delta=0.0))
```

`power_sim` 在明确的已知方差、独立正态任务差值假设下估计单侧检验功效；`null_rejection` 应接近 0.05，可检查第一类错误率，Monte Carlo 标准误表达模拟本身的精度。真实任务的方差未知、分布可能有界且有相关性，因此不能直接把该数值当作真实 benchmark 的功效。增加独立任务与增加种子的相对收益，要根据任务间和训练运行间的方差分量决定；本实验只改变独立任务数，不能证明所有场景都该少跑 seed。

### 12.6.6 分层与层级 bootstrap

当任务来自多个语言、客户或领域时，直接把所有样本混在一起 bootstrap 会把大层的方差主导整个区间。可采用两阶段重采样：先按层等概率抽取层，再在每层内有放回抽取任务或样本，最后按评估契约的权重聚合。这样得到的区间反映“层分布也可能变化”的不确定性。若上线流量的层比例固定，则应采用固定层权重，只在层内重采样。两种区间都可以提供，但要标注它们回答的是不同问题。

层级数据还需要防止同一作者、同一网页或同一会话同时出现在训练和测试。随机按行切分会把作者风格泄漏到两边，使区间异常窄。更稳妥的做法是先按 group id 切分，再在 group 内做 bootstrap；如果 group id 缺失，至少用 URL、会话、时间窗和近重复簇生成代理分组，并把代理规则的不确定性记录下来。

### 12.6.7 置换检验与非参数报告

当指标分布严重偏斜或样本量很小，正态近似可能误导。配对置换检验把每个样本的 A/B 标签以 0.5 概率交换，得到“无差异”下的差异分布。它不要求分数正态，但仍假设配对样本可交换。报告置换 p 值时，同时给出原始差异、置信区间和 MPE；不要用 p 值替代效应量。

对生成质量，均值可能被少数极端失败拉低。可以报告中位数、四分位距、失败率和最差分位数，随后说明这些指标是否是预注册的主要指标。若只在看到结果后增加“更好看”的分位数，就属于指标选择偏差。

### 12.6.8 宏平均、微平均与成本加权

宏平均把每个任务同权，适合回答“典型任务是否改善”；微平均把每个样本同权，适合回答“总体样本错误率”；成本加权则把延迟、算力或人工复核成本纳入业务决策。三者没有天然的优先级，关键是和问题一致。例如高风险任务数量少但代价大时，可把风险成本作为独立门，而不是把少数样本强行混入总体准确率。报告主指标后，应附上其他聚合方式的敏感性分析：如果结论只在一种聚合下成立，就说明存在结构性异质性。

## 12.7 不确定性与校准

### 12.7.1 不是所有“不确定”都一样

- **测量不确定性**：抽样、标注误差和随机种子造成的区间。
- **参数不确定性**：有限数据导致模型参数的后验或估计不稳定。
- **分布不确定性**：线上输入与评估集不同，出现 OOD 或概念漂移。
- **规范不确定性**：多个合理答案、价值冲突或任务定义含糊。

报告时标注不确定性的来源。模型给出 0.9 置信度，可能只是校准差；人工评审一致率低，可能是规范不明确而不是模型随机。

### 12.7.2 校准、选择性预测和风险覆盖

分类任务可用可靠性图、ECE（expected calibration error）和 Brier 分数。生成任务可改用“答案可验证率”“引用支持率”或选择性预测：只在置信度高于阈值时回答，其余转人工。报告覆盖率-风险曲线，而不是只给一个阈值点。阈值必须在验证集上选择，测试集只用于最终一次测量。

对安全任务，校准不能替代硬规则。高置信度的危险输出仍应由策略层拦截；不确定性分数本身不应暴露给攻击者以绕过限制。

### 12.7.3 分布漂移监控

线上应按时间窗、语言、设备、地区和任务类型分层监控。可用 PSI、KS 距离、embedding 邻域覆盖等指标触发复评。漂移阈值只是告警，不是自动发布或封禁依据；先抽样检查数据质量和隐私风险。

## 12.8 可复现实验：把“同样代码”变成可验证对象

### 12.8.1 最小实验包

一次可复现实验至少包含：

- 代码提交哈希和依赖锁文件（含编译器、CUDA/NCCL 版本）
- 数据 manifest、文件哈希、过滤/去重规则和 tokenizer hash
- 完整配置、命令行、环境变量中影响数值的部分
- 所有随机种子、分布式 rank、采样器状态和数据游标
- 硬件拓扑、精度模式、并行配置、确定性开关
- 原始逐样本输出、聚合脚本、指标版本和报告模板
- 失败运行及其原因，不只保存成功结果

### 12.8.2 确定性边界

CPU 上固定 Python、NumPy（若使用）和框架种子仍可能因线程调度、BLAS 或浮点归约顺序产生差异。GPU 上原子操作、混合精度和通信归约更明显。不要承诺“bitwise identical”除非实际验证；更实用的是定义容差，例如 loss 差异 <1e-6、主指标差异 <0.1 个百分点，并记录超出容差的原因。

### 12.8.3 检查点与评估续跑

评估作业也需要 checkpoint：保存已处理 shard、随机状态、聚合器的计数和逐样本结果哈希。恢复时先验证 manifest，再从下一个样本继续，避免重复计数。若结果文件采用追加写，使用临时文件 + fsync + 原子 rename，防止中断留下半行 JSON。

## 12.9 人工评估与质量门

### 12.9.1 先定义 rubric，再抽样

Rubric 应把抽象目标拆成可观察行为。以回答质量为例：

- 正确性：关键事实是否有证据支持，错误是否影响结论
- 完整性：是否覆盖问题要求，遗漏是否关键
- 相关性：是否直接回答，是否无关冗长
- 可执行性：步骤能否按描述完成，前置条件是否清楚
- 安全性：是否泄露敏感信息、提供危险步骤或绕过策略

每项用 0/1/2/3 四级并给正反例，避免评审者自行发明尺度。先冻结 rubric 和抽样框，后看模型结果；否则评审标准会随输出漂移。

### 12.9.2 抽样、盲法和配对比较

按任务类型、语言、风险等级和长度分层抽样。A/B 输出随机化顺序，隐藏模型名和版本。配对比较适合感知微小差异，绝对打分便于质量门。两者可并用：先问“哪个更好”，再分别给安全/正确性标签。对高风险样本全量人工复核，对低风险样本抽样估计。

### 12.9.3 评审者一致性与仲裁

至少 10% 样本由两名评审者独立标注，报告 Cohen κ、Krippendorff α 或简单一致率，并按标签分层。低一致率时先修订 rubric，再扩大样本；用多数票掩盖规范不清会制造虚假精确。仲裁者应看到原始证据和两份理由，不应只看分数。

评审者疲劳会导致后半段打分漂移。限制单批长度、插入已知金标准、记录完成时间和跳题比例。金标准不能直接暴露“正确答案”，否则评审者会机械点击。

### 12.9.4 质量门（quality gates）

一个可操作的发布门示例：

1. **数据门**：主测试集没有已确认的精确题目泄漏；近重复命中率低于预注册阈值，所有例外有审计记录。
2. **统计门**：主要指标 CI 上界/下界和 MPE 一致；随机训练比较按预注册计划完成独立种子（示例为 5 个，实际数量由功效决定）；没有未披露的测试集调参。
3. **安全门**：本次高风险集合观测到零关键违规，并报告该样本量下违规率的上置信界；拒答误伤率不超过基线 +2 个百分点。
4. **人工门**：关键领域双评审一致率 ≥0.8；高风险样本全部仲裁。
5. **复现门**：干净环境重跑在容差内；manifest、代码、日志和逐样本输出可下载。

任何一门失败都应阻止自动发布或降级到人工审批。门的阈值是风险策略，不是统计定理，需由产品、安全和领域负责人共同签字。


### 12.9.5 人工评估的运营闭环

人工评估不是一次性标注，而是一个持续校准的运营系统。启动阶段先用少量代表性样本做共同标注，汇总争议点并更新 rubric；正式阶段采用固定批次、随机顺序和隐形金标准；收尾阶段做漂移分析，比较同一评审者在不同日期对相似样本的判断。若漂移超过阈值，冻结该批结果并重新培训，而不是用后续分数“修正”前面的标签。

标注平台应保存题目版本、呈现顺序、评审者版本、开始/结束时间、跳过原因和修改历史。评审者可以提出“规范不足”标签，该标签不应被强行转化为错误或正确。对安全和隐私样本，采用最小可见字段和专门权限，必要时把原文脱敏后再分发。评审者的个人信息与样本内容分开存储，报告只使用随机 id。

对于成本高的领域，可使用两阶段策略：第一阶段用便宜的筛查模型或初级评审过滤明显通过/明显失败；第二阶段由领域专家复核边界和高风险样本。筛查模型不能直接决定安全通过，也不能把筛查分数当作专家标签。阶段间的抽样概率和误筛率必须进入最终估计，否则会产生验证偏差。

### 12.8.4 实验注册表与结果索引

把每次运行登记为一条不可变记录，而不是只在聊天里贴一个命令。注册表字段可以包括 experiment_id、parent_experiment_id、问题假设、主要指标、数据快照、tokenizer hash、配置 hash、代码提交、环境锁、随机种子、开始和结束时间、操作者、输出目录和状态。实验重跑时生成新的 experiment_id，并通过 parent 字段连接到原始运行；禁止在原记录上覆盖分数或删除失败原因。

结果索引要区分“观察结果”和“解释”。观察结果包括逐样本预测、原始计数、聚合前的分层分数和置信区间；解释包括为什么某层改善、为什么排除污染样本、为什么选择某个阈值。解释可以更新，观察结果不应被后处理脚本悄悄重写。报告生成器读取注册表和固定版本的聚合脚本，避免人工复制数字造成错位。

当多个实验共享同一缓存时，缓存键必须包含数据、tokenizer、模板、过滤规则和评估脚本的哈希。只用“模型名称 + split 名称”做键会把旧的分词结果复用到新 tokenizer。缓存命中和失效都写入日志，并在发布前做一次冷缓存重跑，确认结果不是缓存污染。

### 12.8.5 复现实验的差异预算

复现前先定义可接受差异预算。例如：逐 token loss 绝对差不超过 1e-6，任务宏平均差不超过 0.2 个百分点，人工胜率差不超过 1 个百分点。预算必须和指标量纲一致，不能把相对误差、百分点和标准差混用。若超出预算，按层定位：先检查数据和分词哈希，再检查采样器游标和随机状态，最后才检查硬件和浮点归约。

差异诊断应保留一小批固定探针样本。探针包含空文本、超长文本、Unicode 组合字符、表情符号、代码缩进、工具调用和边界 token。每次升级 tokenizer、框架或驱动都先跑探针，输出 token ids、attention mask、position ids、logits 摘要和解码文本。探针失败应阻止大规模评估，因为长时间运行后再追查会丢失故障发生的上下文。

## 12.9.6 质量门的例外、回滚与复审

质量门需要定义例外路径，否则团队会在临近发布时临时放宽阈值。每个例外包含触发门、风险说明、补偿控制、审批人、有效期限和回滚条件。例如数据门因版权审查尚未完成而暂停，不能用“先发布、稍后清理”代替；若确有内部灰度需求，应限制受众和数据范围，并在系统层阻止外部导出。

回滚不仅是恢复旧模型，还要恢复旧的 tokenizer、提示模板、采样配置和质量门版本。若新模型已经产生用户可见结果，保留受影响请求的时间窗和版本标签，便于定向重算或通知。不要把回滚后的分数与新版本分数混入同一条趋势线；在监控中标记版本边界和数据分布变化。

复审应问三个问题：失败是否由测量错误、模型变化还是数据变化引起？门的阈值是否仍符合当前风险和成本？是否需要增加新任务或隐藏测试来防止再次污染？复审结论写入实验注册表和质量报告，下一次发布可追踪地引用，而不是依赖个人记忆。

## 12.10 CPU-only 端到端评估实验

下面把 tokenization、packing、采样、污染和 bootstrap 串成一个可运行的最小流水线。它故意使用短字符串和简化指标，目标是验证账本与边界，而不是替代生产框架。

```python
from collections import Counter
import hashlib, random, re, statistics

RAW = [
    {"id":"a", "src":"docs", "text":"巴黎是法国的首都。", "score":.9},
    {"id":"b", "src":"code", "text":"def add(a,b): return a+b", "score":.8},
    {"id":"c", "src":"dialog", "text":"请解释什么是向量。", "score":.7},
    {"id":"d", "src":"docs", "text":"巴黎是法国首都。", "score":.85},
]
TEST = ["巴黎是法国的首都。", "解释向量。"]

def normalize(x):
    return re.sub(r"\s+", " ", x.lower()).strip()

def tok(x):
    return list(normalize(x))

def sha(x):
    return hashlib.sha256(normalize(x).encode()).hexdigest()

def sample(rows, seed=17):
    # 质量分只是示例：生产需要预注册的源/样本权重
    r = random.Random(seed)
    weights = [max(.01, row["score"]) for row in rows]
    out = []
    for _ in range(len(rows)):
        u = r.random() * sum(weights); acc = 0
        for row, w in zip(rows, weights):
            acc += w
            if u <= acc:
                out.append(row); break
    return out

def pack(rows, cap=24):
    bins, cur, used = [], [], 0
    for row in sorted(rows, key=lambda z: len(tok(z["text"])), reverse=True):
        n = len(tok(row["text"]))
        if n == 0 or n > cap:
            raise ValueError("空样本或超长样本需要显式策略")
        if used+n > cap and cur:
            bins.append((cur, used)); cur, used = [], 0
        cur.append(row); used += n
    if cur: bins.append((cur, used))
    return bins

def contamination(rows, test):
    hashes = {sha(r["text"]) for r in rows}
    return [q for q in test if sha(q) in hashes]

def paired_bootstrap(a, b, rounds=2000, seed=3):
    d = [x-y for x,y in zip(a,b)]; r = random.Random(seed)
    means = [statistics.mean(r.choices(d, k=len(d))) for _ in range(rounds)]
    means.sort()
    return statistics.mean(d), (means[int(.025*rounds)], means[int(.975*rounds)])

sampled = sample(RAW)
print("sample_ids", [r["id"] for r in sampled])
print("token_counts", {r["id"]: len(tok(r["text"])) for r in RAW})
print("packs", [[r["id"] for r in b[0]] for b in pack(sampled)])
print("exact_contamination", contamination(RAW, TEST))
print("sha_manifest", {r["id"]: sha(r["text"])[:12] for r in RAW})
A = [.8,.6,.9,.7,.75]; B = [.78,.59,.88,.72,.74]
print("paired_bootstrap", paired_bootstrap(A,B))
```

运行时应看到：同一 seed 的 `sample_ids`、`packs` 和哈希完全一致；测试集第一个问题被精确命中；bootstrap 的均值差约为 0.008，95% 百分位区间约为 [-0.006, 0.018]；区间跨过零，因此这五个玩具任务不能支持稳定提升的结论。将 `seed` 改为 18，采样顺序应变化但 manifest 规则不变。再添加测试题“巴黎是法国首都。”，检查精确命中来源，并用第一个实验中的改写对研究近重复阈值；短到不足 n 个字符的题目需走单独规则，不能因为 shingle 集为空就宣称无污染。

建议把这段实验拆成四个单元测试：

- tokenizer：规范化后 round-trip 和特殊 token id 不冲突
- packer：包内长度不超过容量，独立 mask 没有跨样本可见边
- sampler：固定 seed 可复现，经验比例在容差内接近理论概率
- evaluator：删除一个样本会更新分母和 CI，不能只更新分子

## 12.11 失败案例：分数变高但结论变差

以下案例为教学用的合成情景，百分比用于解释因果链，不代表特定公司的事故或实测结果。

### 案例一：Tokenizer 升级导致“免费”提升

团队把词表从 32k 换成 128k，验证 loss 下降，且每个 batch 的 token 数更少。后来发现新 tokenizer 对英文 URL 和代码切得更好，但中文长尾字符切分更碎（若无字节回退，还可能出现 UNK）；评估集恰好以英文为主。修复：按语言和文档类型分层报告，固定原始文本与算力预算分别比较，以任务质量和每字节负对数似然为补充，而非直接比较不同 tokenizer 的 token loss；在 checkpoint 中强校验 tokenizer hash。教训是 loss 的改善同时混合了表示变化和数据成本变化。

### 案例二：Packing 跨样本泄漏

为了提高吞吐，工程师只插入 EOS，没有 block-diagonal mask。问答 benchmark 的后一个问题可以看到前一个问题的答案，准确率提升 4 个百分点。修复：独立 mask + 边界置乱测试；对 streaming LM 单独命名和报告。教训是吞吐优化改变了任务定义。

### 案例三：公开 benchmark 被抓取

训练语料抓取了一个教程站点，里面包含公开测试题和解析。精确 hash 未命中，因为 HTML 清洗改变了空格；13-gram 检测发现题干覆盖率 85%。修复：按发布日期切分、保留原始 URL、将命中样本排除并在报告中量化影响。教训是“没有精确重复”不等于没有污染。

### 案例四：挑 seed 和 p 值

十个随机种子中，只有两个 seed 的提升显著，报告只展示这两个。复现时平均效果接近 0。修复：预注册所有 seed，报告均值、任务级分布和全量探索结果，使用多重比较校正。教训是停止规则和完整日志比漂亮表格更重要。

### 案例五：人工评审的顺序效应

评审者先看了基线的 100 条，再看新模型的 100 条，逐渐放宽“完整性”标准；新模型分数虚高。修复：A/B 随机化、盲法、插入金标准、限制批次长度，并对评审者做顺序效应分析。

### 案例六：高置信度的危险答案

模型对少见药物剂量给出 0.95 置信度，校准图显示平均置信度很高，但高风险子集没有证据。修复：将“有无可验证证据”作为独立质量门，危险域采用检索/人工复核和拒答策略；不把模型置信度当作安全批准。

## 12.12 理解检查（含答案）

### 检查 1：为什么 token 数不能直接当作样本数？

**答案：**不同语言、代码和格式的 token/字符比不同；按 token 加权会让长样本和高碎片化语言占更大权重。应同时报告字符、token、样本和任务级分母，并预先指定主要聚合方式。

### 检查 2：何时必须使用 block-diagonal mask？

**答案：**当 pack 中的样本应当相互独立，尤其包含答案标签、个人数据或需要严格 per-example loss 时。若任务明确把连续文档当作流，可使用 streaming attention，但必须在契约里声明并用不同基准比较。

### 检查 3：温度采样 α<1 会自动解决长尾问题吗？

**答案：**不会。它只改变源级概率，小源可能因此被过采样；源内部重复、质量差、时间漂移和语言覆盖仍需独立处理。还要计算有效样本量和实际 token 预算。

### 检查 4：CI 跨过 0 是否意味着模型没有价值？

**答案：**不一定。它表示当前抽样程序下无法排除 0 差异，可能是功效不足或任务异质性大。应结合 MPE、效应方向、任务级分布、成本和安全风险；增加独立任务或改善测量，比盲目重复同一题更有信息。

### 检查 5：为什么“未检测到污染”不能写成“没有污染”？

**答案：**检测依赖索引范围、规范化、相似度算法和阈值，都会产生漏报。正确表述是“在当前规则和可见语料下未发现”，并报告抽样审计、阈值校准和未覆盖来源。

### 检查 6：人工评审一致率低时，扩大样本是否总能解决？

**答案：**不能。若 rubric 含糊或任务有多种合理答案，扩大样本只会更精确地测量争议。应先用反例修订 rubric、训练评审者和设置仲裁，再决定是否增加样本。

## 12.13 练习

1. 为一个中英代码混合语料设计 tokenizer 报告：至少包含四个分层、两个边界样本和 tokenizer 漂移的回滚条件。
2. 给定长度 `[3,3,3,7,7,10]` 和容量 10，手算 FFD 包数与利用率，再设计一个能更好利用率的装箱顺序。
3. 编写一个 block-diagonal causal mask 的纯 Python 版本，并用随机替换前一文档 token 的方法验证没有跨文档影响。
4. 设计三源温度采样的预注册方案：给出原始比例、α、每个 epoch 的 token 预算、随机种子和偏差监控。
5. 为一个公开数学 benchmark 写污染审计计划：精确 hash、13-gram、时间切分、人工复核和例外处理都要有阈值。
6. 用历史任务级分数模拟 80% 功效检测 1 个百分点提升；比较增加 seed 与增加独立任务的成本。
7. 设计人工评审 rubric，包含正确性、安全性和可执行性；写出两个容易混淆的反例并说明仲裁规则。
8. 对本章端到端脚本添加“删除样本后自动重算 denominator”的测试，并提交一次可复现实验 manifest。

## 12.14 来源地图与延伸阅读

以下链接用于核对规范与经典方法，阅读时记录访问日期和版本，不把链接本身当作实验结果：

- Unicode 标准化与规范化：<https://unicode.org/reports/tr15/>。用于定义 NFC/NFKC 的边界和兼容字符风险。
- Hugging Face Tokenizers 文档：<https://huggingface.co/docs/tokenizers/>。用于工程化 tokenizer 训练、规范化、预分词和后处理。
- SentencePiece 论文与实现：<https://github.com/google/sentencepiece>。说明无需预分词的子词模型与可逆处理。
- 将 BPE 用于神经机器翻译的工作（Sennrich 等）：<https://aclanthology.org/P16-1162/>。解释神经机器翻译中的子词分割。
- MinHash/LSH 综述：<https://www.cs.princeton.edu/courses/archive/spring13/cos598C/lloyd-minhash.pdf>。用于近似集合相似度和重复检测。
- NIST/SEMATECH 统计手册：<https://www.itl.nist.gov/div898/handbook/>。涵盖置信区间、实验设计和功效基础。
- Efron 与 Tibshirani 的 bootstrap 介绍：<https://statweb.stanford.edu/~tibs/stat315a/lectures/boot.pdf>。用于理解重采样区间的前提。
- Benjamini–Hochberg FDR：<https://www.jstor.org/stable/2346101>。用于多重比较控制。
- Krippendorff alpha：<https://repository.upenn.edu/asc_papers/43/>。用于多评审者一致性分析。
- HELM 评估框架：<https://crfm.stanford.edu/helm/latest/>。展示多场景、多指标和透明报告的组织方式。
- BIG-bench：<https://github.com/google/BIG-bench>。提供任务定义、元数据和评估脚本的参考。
- ML Test Score：<https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/>。把数据、模型、监控和测试纳入上线质量门。
- Datasheets for Datasets：<https://dl.acm.org/doi/10.1145/3458723>。用于记录数据来源、用途、限制和风险。

来源地图的边界：这些资料描述方法或框架，不能证明某个模型、数据集或本章代码的具体数值。具体实验仍需保存自己的原始输出、版本和审计记录。

## 12.15 安全边界与升级路径

1. **隐私边界**：污染检测、逐样本日志和人工评审可能包含个人信息。只保存完成审计所需的最小字段，哈希不能替代访问控制；原文和 URL 按数据治理政策加密、限权和设定保留期。
2. **危险内容**：评估集可能含恶意代码、药物剂量、暴力或自伤内容。CPU 实验只使用无害短句；生产评估需隔离执行、内容分级、红队审批和紧急停机联系人。
3. **外部数据传输**：不要把受限测试集上传到第三方 tokenizer 或在线去重服务。需要外部服务时，先确认数据处理协议、区域和删除保证。
4. **自动发布边界**：质量门失败、污染不明、统计功效不足或安全高风险样本异常时，自动化只能暂停并创建审查任务，不得自行放宽阈值或删除失败记录。
5. **模型置信度**：置信度、校准和不确定性指标不构成医疗、法律、金融或人身安全建议。高风险决策必须由合格人员和领域流程复核。
6. **基准与奖励**：不要针对公开 benchmark 反复优化到过拟合；保留独立、隐藏或时间外推测试，并把发布门与业务风险而非单一榜单绑定。

当检测到疑似泄漏、数据处理越权、危险输出或无法解释的分数跃升时，升级路径是：冻结相关 checkpoint 和报告 → 保留原始证据与访问日志 → 通知数据治理/安全/领域负责人 → 复现与影响评估 → 决定排除、重跑或撤回结论。不要在证据尚未固定前覆盖 manifest、删掉失败运行或对外宣称“已修复”。

## 12.16 小结：把评估当作一条可回放的测量链

可靠评估的关键不是更复杂的单个指标，而是每一层都能回答“测了什么、如何加权、边界在哪里、误差多大、谁可以复核”。Tokenizer 决定成本和表示；packing 决定样本是否互相泄漏；采样决定训练分布；污染审计决定 benchmark 是否仍代表泛化；统计设计决定差异是否可信；人工质量门决定模型是否适合真实风险。把这些信息写进契约、manifest、测试和审计日志，才能在模型、数据或硬件变化后重放实验并解释结论。最终交付的不只是一个分数，而是一份带不确定性、限制条件和安全责任的证据包。
