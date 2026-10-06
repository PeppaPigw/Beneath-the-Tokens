---
id: phase2-23-decoding-and-sampling
title: 随机解码与采样：改变分布，还是少走几步
description: 从两个请求和一组可手算的 logits 出发，理解温度、截断、约束与投机解码的分界、证据和失败边界
slug: /phase2/chapters/23-decoding-and-sampling
sidebar_position: 123
phase: 2
chapter_number: 23
level: advanced
prerequisites:
  - ch01-ai-infrastructure
  - ch19-inference-optimization-accelerator-stack
learning_objectives:
  - 能把 logits、概率、熵和 surprisal 在一个小词表上逐步算出来
  - 能实现并测试 temperature、top-k、top-p、typical、min-p 与 Mirostat 的最小版本
  - 能解释 tokenizer、语法自动机和约束分布之间的边界
  - 能证明 speculative decoding 的残差校正为什么保持目标分布，并指出何时不再无损
  - 能运行 CPU 实验，报告支持集、KL、distinct-n、接受率和重复性限制
paper_count: 13
source_commits:
  - btt-phase2@5aaeeb530385b740107f0b88f113227709ca1443
  - transformers@v4.44.2
lab_paths:
  - labs/phase2/decoding_sampler.py
last_verified: 2026-10-06
source_commit: btt-phase2@5aaeeb530385b740107f0b88f113227709ca1443
lab_path: labs/phase2/decoding_sampler.py
estimated_hours: 10
---

# 随机解码与采样：改变分布，还是少走几步

> **本章地图**：先跟着两个请求看见“同一个模型为什么说出不同的话”，再用一个五 token 的分布手算温度和截断。随后我们把采样器分成两类：一类有意改变模型分布，另一类用草稿和校正减少目标模型的串行步数、但在条件满足时保持分布。最后把公式落到一个纯 Python CPU 实验。这里暂时不讨论训练目标、RLHF，也不把某个采样参数当成通用质量保证。本章的 `paper_count=13` 只统计正文中的 13 张主要教学卡（Holtzman、Typical、Mirostat、Contrastive、GCD、PICARD、Speculative、Medusa、EAGLE、SpecInfer、Draft & Verify、Lookahead、Min-p），不是把每个实现仓库或背景引用都算成论文。

## 问题边界：两个请求为什么走出不同的路径

服务刚热起来时，两个请求几乎同时到达。

- 请求 A：“请用一句话解释缓存命中。”产品希望回答自然一点，可以有少量变化。
- 请求 B：“返回一个包含 `answer` 和 `confidence` 的 JSON。”调用方会严格解析，少一个引号都算失败。

同一模型、同一前缀、同一个 GPU，A 的下一 token 可能从“命中”变成“缓存”，B 却必须只能从自动机允许的 token 中选择。A 用温度和 top-p 后，输出空间被重新加权；B 还多了一张“现在允许哪些 token”的表。若用一个便宜的小模型先写四个 token，再让大模型一次检查，这又是另一种改变：目标概率可以不动，只是把四次串行前向合成一轮草稿加一轮验证。

先记下三个容易混淆的问法。

1. **我想要更有变化的文本**：这是目标分布的选择问题，温度、top-k、top-p、typical、min-p 和 Mirostat 都会删掉或重排质量。
2. **我想要合法的 JSON/SQL**：这是把支持集与状态机相交的问题，通常也会改变分布；“合法”不等于“仍按未约束模型分布”。
3. **我想减少每个 token 都跑一次大模型的等待**：这是计算路径问题。speculative decoding、树式验证、自推测和 lookahead 只有在声明的 target 分布 p、有效的接受/拒绝与残差校正、以及正确的条件停止/回填都满足时，才可以称为无损加速；两边不需要共享同一随机数流。

本章研究的是单步和短序列的因果链：`logits → 变换后的分布 → 采样/校正 → 下一个上下文`。长文本的事实性、模型偏见和安全策略仍然是上层评测问题；一个更低的困惑度不会自动证明回答更有帮助。

## 直觉模型：先把“分布”看成一袋带重量的签条

想象模型把下一 token 写在五张签条上，签条有重量。重量总和是 1，抽签前你可以做三种动作：

- **加热或降温**：把所有重量的差距拉近或拉大。低温更像拿走最重的一张，高温让尾部签条也有机会。
- **截断**：只留下前 k 张、累计质量达到 p 的一小叠，或留下信息量接近平均值的签条；留下后要重新称重。
- **相交**：语法状态说“此处只能是数字或右括号”，于是把其他签条拿开，再重新称重。

这三种动作都可能改变“模型原来想抽到什么”的分布。它们解决的是质量、多样性或结构合法性，不是加速。

另一张图解释加速：小模型先按自己的袋子快速抽四张签，目标模型同时看四个位置。每个位置都掷一次接受硬币，接受概率是 `min(1, p/q)`。拒绝时不能简单选目标模型最重的签，否则随机目标会被偷偷改成贪心；要从 `[p-q]_+` 的残差袋子里抽。接受-拒绝两部分恰好拼回 p，这才是无损的核心。

```
请求前缀 h
       │ 目标模型给 logits z_t
       ▼
  softmax / temperature ──> p_t       (原分布可能已被改写)
       │
  top-k / top-p / typical / min-p / grammar mask
       ▼
       p'_t ──采样──> token x_t ──回填 h──> 下一步

草稿路径：h ──q 小模型串行提出 x1..xK──┐
                                      ├─ p 一次验证 + 接受/残差校正
                                      └─ 输出 1..K+1 个 token（目标分布可保持）
```

这里的“袋子”只是心智模型。实现中必须处理浮点下溢、tie-break、tokenizer 边界、空支持集和随机种子；这些小细节会把理论上的 p 变成另一个 p。

## 最小例子：先算一次，再决定要不要截断

我们暂时只有五个 token，编号 0–4，模型输出 logits

```
z = [2.0, 1.0, 0.2, -0.8, -1.5]
```

logits 不是概率，能整体加一个常数而不改变 softmax。先减去最大值 2，得到 `[0, -1, -1.8, -2.8, -3.5]`。取指数并归一化（这里四舍五入到 4 位）：

```
exp(z - 2) ≈ [1.0000, 0.3679, 0.1653, 0.0608, 0.0302]
总和 ≈ 1.6242
p     ≈ [0.6157, 0.2265, 0.1018, 0.0374, 0.0186]
```

如果 `T=2`，logits 先除以 2，概率会变平；如果 `T=0.5`，差距被放大，第一张签条更重。温度是对 logits 的变换，不是“在已经抽过签以后把随机数调大”。顺序也有影响：先温度再 top-p，和先 top-p 再温度，留下的支持集可能不同。实验必须把顺序写在命令里。

现在取 top-k，`k=2`。留下 token 0 和 1，质量为 `0.6157+0.2265=0.8422`，重新归一化：

```
p_top2 ≈ [0.7310586, 0.2689414, 0, 0, 0]
```

这已经不是原来的 p。可以用 KL 散度看方向：`KL(p_top2 || p)` 大于 0，因为支持集删掉了尾部。它不是“近似实现误差”，而是设计选择。

取 top-p，`p=0.8` 时按概率从大到小累加：token 0 的质量 0.6157，还不够；加入 token 1 后是 0.8422，于是支持集也恰好是 `{0,1}`。若把 `p` 提到 0.95，第三张签条加入后总和 0.9440，仍不足，再加入 token 3 才超过 0.95。支持集的大小随上下文改变，这正是 nucleus 方法相对固定 k 的直觉。

再算熵。用自然对数，

```
H(p) = -Σ p_i log p_i ≈ 1.06 nats = 1.54 bits
```

若一个 token 的 surprisal 是 `s_i=-log p_i`，token 0 的 surprisal 约 0.485 nats，token 4 的 surprisal 约 3.985 nats。熵是按 p 加权后的平均 surprisal，不是最高概率 token 的 surprisal。typical sampling 会看 `|s_i-H|`，所以“第二名”有时比“第一名”更接近典型信息量。

本节的可复算结论只有两个：任何截断都要重归一化；任何支持集变化都要在评测中报告。读者可以用 `python labs/phase2/decoding_sampler.py` 检查同一组数字。

## 最小例子续算：五张签条如何经过四种筛选器

前面只算了 top-2 和 top-p 的一个点。现在把同一组五项分布完整走一遍。这样做有一个好处：参数不再是抽象旋钮，读者能看到每个操作删掉了谁、留下了谁，以及为什么两个名字相近的筛选器会做出不同决定。

**温度：先改分数，后改变每一张签的相对重量**

原始 logits 是 `[2.0,1.0,0.2,-0.8,-1.5]`。当 `T=0.5` 时，分数除以 0.5，差距变为 `[4.0,2.0,0.4,-1.6,-3.0]`。为了计算稳定，减去最大值 4，再取指数并归一化，得到约

```
p_0.5 = [0.856701, 0.115942, 0.023408, 0.003168, 0.000781]
H_2(p_0.5) ≈ 0.7127 bits
```

第一项已经占 85.7%。这不是“随机数变小”这么简单：原来排名第三的 token 质量从 10.2% 降到 2.3%，而且这个变化会写回上下文。`T=2` 则把分数压成 `[1,0.5,0.1,-0.4,-0.75]`，归一化得到

```
p_2 = [0.410936, 0.249245, 0.167074, 0.101335, 0.071410]
H_2(p_2) ≈ 2.0647 bits
```

熵从 1.5359 bits 上升到 2.0647 bits，但“熵变高”只表示不确定性变高。它没有告诉我们第五个 token 是否有用，也没有告诉我们事实性是否变好。低温容易产生模式坍缩，高温容易把低概率尾部放大；最终参数必须由目标任务的独立评测决定。

**Top-k：固定袋子大小的硬选择**

`k=1` 会留下 token 0，等价于 greedy；`k=2` 留下 token 0、1，质量和为 0.84219395。重新归一化时不能拿 0.6157 和 0.2265 原样当概率，而要除以这个和：

```
p_top2(0) = 0.6156931154 / 0.8421939546 = 0.7310585786
p_top2(1) = 0.2265008392 / 0.8421939546 = 0.2689414214
```

这两个数看起来像 sigmoid，是因为前两项 logits 相差 1；它们之和是 1。`k=3` 时，前三项原始质量为 0.94396734，归一化后约 `[0.6523,0.2400,0.1077,0,0]`。注意固定 k 不看分布是否平坦：在另一个前缀中，排名第 2 和第 3 可能几乎一样，也可能相差巨大，top-k 都会留下相同数量的 token。

**Top-p：固定袋子质量的自适应选择**

对原始 p 从大到小累加：

```
加入 0：0.615693
加入 1：0.842194
加入 2：0.943967
加入 3：0.981408
加入 4：1.000000
```

因此 `p=0.8` 与 `p=0.9` 都留下 `{0,1}`，但 `p=0.95` 要加入 token 3，支持集变成 `{0,1,2,3}`。这是“自适应”的具体含义：不是每次保留 90% 的 token 数，而是保留累计质量首次达到 0.9 的最小前缀。若实现先排序、再在边界处把整个并列组都加入，支持集可能与本章 tie-break 不同；评测中应固定并记录 tie-break。

**Typical：平均信息量附近不等于概率最高**

原始分布的熵是 `H=1.0646 nats`。五个 token 的 surprisal 和到熵的距离约为：

```
token 0: s=0.4850，|s-H|=0.5796
token 1: s=1.4850，|s-H|=0.4204
token 2: s=2.2850，|s-H|=1.2204
token 3: s=3.2850，|s-H|=2.2204
token 4: s=3.9850，|s-H|=2.9204
```

所以典型排序是 `1,0,2,3,4`，不是常见的概率排序 `0,1,2,3,4`。`typical_p=0.9` 时按典型排序累加：先拿 token 1，质量 0.2265；再拿 token 0，质量 0.8422；再拿 token 2，质量 0.9440，达到阈值，于是支持集是 `{0,1,2}`。这个例子很适合提醒读者：典型采样可能保留第二名而暂时跳过第一名，但重归一化后的样本仍有 0.7311 的机会落在第一名。

典型采样的直觉来自“传递接近平均信息量的词”，不是一条关于人类语言的硬定律。低熵上下文中，熵很小，典型区间可能只剩极少 token；高熵上下文中，许多 token 的 surprisal 都接近熵。实现必须明确是先温度还是先典型筛选，因为温度会同时改变 p 和 H。

**Min-p：相对峰值的门槛**

设 `p_max=0.615693`。当 `min_p=0.1` 时阈值是 0.061569，保留 token 0、1、2；当 `min_p=0.5` 时阈值是 0.307846，只保留 token 0；`min_p=0` 则不删任何 token。它与 top-p 的不同之处是门槛随峰值缩放：分布很尖时，尾部迅速被清掉；分布平时，相对门槛允许更多候选。

多峰分布是 min-p 的反例。设分布为 `[0.45,0.12,0.12,0.11,0.10,0.10]`，第二个语义峰由两个各占 0.12 的 token 组成；`min_p=0.3` 的门槛是 `0.135`，因此只留下 0.45 的最高 token，整个第二个峰被删掉。top-p 也可能有类似问题，但控制量不同。使用 min-p 时应同时报告峰值、阈值、支持集和 KL，而不是只报告参数名。

**一个小的 KL 账本**

对 top-2，精确的新分布是 `[0.7310585786,0.2689414214,0,0,0]`。逐项代入

```
0.7310586 × log(0.7310586 / 0.6156931) ≈ 0.1256
0.2689414 × log(0.2689414 / 0.2265008) ≈ 0.0461
KL(p_top2 || p) ≈ 0.171745 nats
```

第二项不是把“剩余质量 0.1578”直接当 KL；KL 只比较新分布支持集中的项。反向的 `KL(p || p_top2)` 因为 p 在 token 2、3、4 上仍有正质量而 p_top2 为 0，按数学定义为无穷。工程代码若要报告这个方向，应先选择平滑方案并明确那已经是另一个指标。这个手算还告诉我们：一个看似小的支持集选择，不能写成“只带来 0.04 nats 的近似误差”。

## 正式定义前的熵、交叉熵与 KL 单位检查

解码讨论很容易把三个量混称为“困惑度”。把它们分开，读者才能知道实验到底在测什么。熵 `H(p)` 是分布自身的不确定性；交叉熵 `H(p,q)=-Σ p_i log q_i` 需要一个参考分布 p 和被评估分布 q；KL `D_KL(p||q)=H(p,q)-H(p)` 是非负的方向量。它们都可以用 nats（自然对数）或 bits（以 2 为底），但一张表不能混用单位。

对原始五项 p，`H(p)=1.0646 nats=1.5359 bits`。对低温 `p_0.5`，熵约 0.494 nats；对高温 `p_2`，熵约 1.431 nats。这里的“熵更低/更高”只是在同一词表、同一前缀下比较不确定性。若 top-k 删除了三个 token，支持集变小，熵下降并不意味着模型质量提高；它可能只是因为规则不允许模型选择那三个 token。

交叉熵要把真实 token 或参考分布写出来。例如若数据中的下一 token 是 token 2，原模型的 NLL 是 `-log(0.101773)=2.285 nats`；top-2 的 q(token 2)=0，因此 NLL 是无穷，说明该截断分布不可能解释这条真实观察。这个例子很直观地证明：采样文本看起来更干净，不能推出对原始数据的 NLL 更低。

KL 方向也影响解释。`D_KL(p_top2||p)=0.171745` 只在新支持集上求和；它回答“按新策略抽样时，相对原模型要付出多少信息代价”。反向 `D_KL(p||p_top2)` 因为原 p 在删掉的 token 上有正质量而 q 为零，数学上发散；它回答“原模型产生的所有可能性有多少无法被新策略解释”。工程报告应写清方向、平滑和零概率处理，不能只写“KL=0.17”。

一个简单的单位检查是把公式中的每项标出来：p、q 无量纲，log 概率是 nats，熵和 KL 是 nats/token；若乘以 token 数 T 才得到序列 nats。把熵直接和毫秒、吞吐或接受率相加是量纲错误。serving 报告可并列熵、KL、ITL，但不应把它们合成一个没有定义的“综合质量分”。

## 最小例子：约束规则怎样遇到 BPE

假设我们要生成一个极小 JSON，只允许 `{"a":1}` 或 `{"b":2}`。为了突出边界，使用一个人为的 token 表：

| token id | 解码文本 |
| ---: | --- |
| 0 | `{` |
| 1 | `"a"` |
| 2 | `"b"` |
| 3 | `:` |
| 4 | `1` |
| 5 | `2` |
| 6 | `}` |
| 7 | ` `（前导空格） |
| 8 | `"a":`（一个多字符 token） |

状态 `q0` 的合法前缀只有 `{`，所以允许 token 0；状态 `q1` 看到 `{` 后允许 token 1、2，也可能允许 token 7（如果 grammar 允许空格）；状态 `q2` 看到完整键后允许 token 3。若 tokenizer 提供 token 8，状态 q1 也可以直接走到“等待值”的状态 q3，因为它一次展开了 `"a":`。把 token 8 当作单个字符会错误拒绝合法路径；反过来，如果只检查 token 的第一个字符，某个包含非法后缀的 token 可能被错误放行。

一个 tokenizer-aware trie 的最小检查过程是：

1. 从 grammar 状态 q 和字符前缀出发，取一个候选 token 的完整解码文本。
2. 逐 code point 推进临时 grammar 状态；中途遇到非法字符就拒绝该 token。
3. token 全部展开且到达合法状态，才把 token id 放入 `allowed(q)`；若 token 结束在一个仍可继续的字符串内部，也保留下一状态。
4. 将不在 allowed 集的 logits 设为 `-∞`，然后重新 softmax；下一步使用新状态。

这个过程解释了为什么“字符白名单”不等于“token 白名单”。中文标点、Unicode NFC/NFD、JSON escape、token 的前导空格都会改变展开文本。空允许集应记录 q、前缀、tokenizer 版本和 schema hash 后失败；静默把 logits 恢复成原分布会产生不可解析输出。

约束还会影响统计评测。如果 grammar 条件分布只剩一个 token，熵自然降到 0；这不是模型突然变得更确定，而是规则删掉了其它可能。比较结构任务时要同时报告合法率、重试率、支持集大小和约束开销，不能只看生成质量。

## 最小例子：Speculative 的数值例与反例

取目标分布 `p=[0.6,0.3,0.1]`，草稿分布 `q=[0.55,0.35,0.1]`。草稿抽到 token 0 时，接受率 `min(1,0.6/0.55)=1`；抽到 token 1 时，接受率 `0.3/0.35≈0.8571`；抽到 token 2 时接受率 1。残差的未归一化质量是 `[0.05,0,0]`，所以拒绝后必然用 token 0 替代。这个 q 很接近 p，期望接受率会高。

换成 `q=[0.2,0.5,0.3]`：token 0 的接受率仍为 1，token 1 为 0.6，token 2 为 0.3333；残差是 `[0.4,0,0]`，仍集中在 token 0。这里草稿越偏向尾部，拒绝越多；但只要拒绝时按残差抽样，最终 token 的边缘仍是 `[0.6,0.3,0.1]`。效率变差不等于正确性变差。

一个常见错误是“draft token 与 target argmax 相等才接受”。对上面的 p、q，草稿 token 1 即使有 0.8571 的校正接受概率，也会被错误规则全部拒绝；错误规则产生的序列会向 token 0 偏置。另一个错误是拒绝后直接从 p 取 argmax，同样将残差的随机质量压成一个点。单步频率实验可以发现这两个错误：正确算法的经验分布随着样本数增加接近 p；错误算法会出现稳定的 mode 偏移。

接受—拒绝定理的精确边界值得单独写出来。它只要求 p 是你声明要采样的归一化 target，q 是任意归一化 proposal，且你能从 q 生成草稿、按每个 token 的 p/q 计算接受率、在拒绝时从 `[p-q]_+` 归一化残差采样。q 可以与 p 使用不同温度或不同截断，正确性仍成立；不同变换会改变 q、接受率和效率，也意味着用户配置不再相同。工程上推荐两边使用同一变换，是为了让“目标模型要实现的分布”明确且提高接受率，不是因为证明在不同变换下失效。

若 q(token)=0 而 p(token)>0，草稿永远不会提出该 token，所以它不进入“对已提出 x 计算 alpha”的分支；它的 p 质量会由残差 `[p-q]_+` 承担。不能把 `p/q` 的除零当成 NaN，也不能把这个未被提出的 token 当作一次 alpha=0 的候选。lab 的 `acceptance_probability` 为了让标量边界测试返回一个有限值，对 q=0 做了教学便利处理；生产多位置实现应先判断候选是否由 q 提出，再分别执行接受或残差抽样，不能只依赖这个 helper。

## 正式定义与推导：从 logits 到联合序列分布

**1. 自回归分解**

给定前缀 `h_t=x_<t` 和词表 `V`，模型给出 `z_t(i)`。定义

\[
p_\theta(i\mid h_t)=\frac{\exp z_t(i)}{\sum_{j\in V}\exp z_t(j)}.
\]

整段序列的概率是

\[
P_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
\]

这是定义，不是说各步独立；下一步的条件包含前一步采样结果。数值实现要用减去最大 logits 的 log-sum-exp，避免 `exp(1000)` 溢出。`float32` 足够做教学实验，但生产内核还需关注 GPU reduction 的舍入顺序。

**2. 温度、截断与熵**

温度 `T>0` 的分布为

\[
p_T(i\mid h)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}.
\]

`T→0` 时，若最高 logit 唯一，它的概率趋向 1；若有 m 个并列最高值，极限是在这 m 个 token 上均匀分布。实现里的确定性 tie-break 是另一个 argmax 行为，不能冒充 softmax 极限。`T=1` 保持 p；`T>1` 通常增大熵。这个“通常”依赖有限 logits；`T` 不能设为 0，在代码中应报错而不是产生 NaN。

top-k 的支持集是按 `p_i` 排序的前 k 项 `S_k`，

\[
p^{(k)}(i)=\frac{p_i\mathbf 1[i\in S_k]}{\sum_{j\in S_k}p_j}.
\]

top-p 的 `S_p` 是概率降序的最小前缀，满足 `Σ_{i∈S_p}p_i≥p`。典型采样先算

\[
H(p)=-\sum_i p_i\log p_i,\qquad s_i=-\log p_i,
\]

按 `|s_i-H|` 排序，累加到阈值 `τ`。这不是“选接近均值的 token 后不归一化”；筛选完成仍是一个新分布。

min-p 定义相对门槛 `θ=m·max_i p_i`，保留 `p_i≥θ`。它在峰值很高时自动收紧，在分布平坦时相对放宽。论文提出的质量—多样性观察不能直接迁移成“m=0.1 永远最好”：温度、模型、数据和截断顺序都会改变结果。

**3. 反馈控制与 Mirostat**

若希望每一步的平均 surprisal 接近目标困惑度 `PPL*`，可以在 log 坐标写 `μ*=log(PPL*)`。观察到 token 的 `ŝ_t=-log p(x_t)` 后，最小的反馈更新是

\[
\mu_{t+1}=\mu_t-\eta(\hat{s}_t-\mu^*).
\]

例如目标 `PPL*=4`，`μ*=log 4≈1.386`。若本次 token 的 surprisal 是 2.386，`η=0.2`，则新 `μ=μ_old-0.2`；下一步门槛应收紧。若 surprisal 是 0.386，则新 `μ=μ_old+0.2`，允许更大的候选集。这里的 `μ` 是控制状态，Mirostat-1 还要用 Zipf 尾部斜率把它映射到 k；Mirostat-2 使用更简化的动态阈值。本 lab 只实现可单测的标量更新，不假装复现完整 tail estimator。

反馈控制有两个边界。`η` 太大时会在“太惊讶/太不惊讶”之间振荡；模型分布不符合 Zipf 假设时，估计偏差不会因为多跑几步消失。目标 PPL 是可观测控制量，不是事实性或帮助性的代理指标。

**4. 约束是支持集与状态的相交**

设 grammar/FSM 当前状态为 q，能合法接出的 token 集为 `A(q)`。最简单的 logits mask 是

\[
\tilde z_i=\begin{cases}z_i&i\in A(q)\\-\infty&i\notin A(q).\end{cases}
\]

然后对 `\tilde z` softmax。若把“模型在 grammar 语言上的条件分布”定义为目标，这个重归一化是合理的；若目标仍是无约束 p，它显然变了分布。空集不是“全禁用后返回 eos”的小问题，而是 grammar、tokenizer 或 schema 的 bug，应带状态和前缀报错。

字符级 grammar 不能直接把字符白名单当 token 白名单。BPE token 可能一次包含多个字符，也可能是带前导空格的片段；必须构造“从当前字符前缀可接受的 token trie/FSA”。Unicode 归一化、大小写和 escape 规则再错一位，合法输出也会被拒绝。这个边界是约束系统最常见的生产故障之一。

**5. Speculative 的接受—拒绝证明**

设目标分布为 `p`，草稿分布为 `q`，草稿抽到了 token x。接受概率定义为

\[
\alpha(x)=\min(1,p(x)/q(x)).
\]

若接受，贡献概率 `q(x)α(x)=min(q(x),p(x))`。若拒绝，从残差

\[
r(x)=\frac{[p(x)-q(x)]_+}{\sum_y[p(y)-q(y)]_+}
\]

抽一个替代 token。拒绝发生的总质量是 `1-Σ_x min(p(x),q(x))`，乘上 `r(x)` 后正好得到 `[p(x)-q(x)]_+`。两部分相加：

\[
\min(p(x),q(x))+[p(x)-q(x)]_+=p(x).
\]

这是一行很短但很重要的证明：**不能用“草稿和目标 token 相等才接受”代替接受率，也不能在拒绝后直接从 p 贪心取最大值**。前者会偏向 q 与 mode，后者丢掉随机目标。

多 token 草稿在第 i 个位置用条件分布 `p_i、q_i`，一旦拒绝就丢弃后缀，因为后缀的条件上下文已经不成立；若全部 K 个接受，再从目标分布取额外 token。加速不等于每轮永远输出 K 个 token：接受率、验证批量效率、草稿开销和 KV 读写决定端到端收益。

## 机制与源码入口：一次请求经过哪些状态

读者任务：先不要看论文名，沿着 `labs/phase2/decoding_sampler.py` 找出三个状态变化：

1. `logits_to_probabilities` 在哪里减去最大值，为什么这一步不改分布？
2. `grammar_filter` 在哪里把禁用 token 设为 `-inf`，什么时候会抛出空支持集错误？
3. `speculative_acceptance_simulation` 在哪里停止草稿后缀，残差分布从哪一行来的？

本章的调用链是：

```
输入 logits
  └─ apply_temperature → logits_to_probabilities
       └─ 选择一个 support filter（top-k/top-p/typical/min-p）
            └─ grammar_filter（可选，重新 softmax）
                 └─ sample_categorical（固定 RNG）

目标 p + 草稿 q + 草稿 token
  └─ acceptance_probability
       ├─ 接受：追加 token，继续检查下一个位置
       └─ 拒绝：residual_distribution → 采样替代，丢弃后缀
```

实验代码故意不依赖 Transformers。这样读者可以把“策略”与框架的 batch、KV cache、CUDA kernel 分开。生产源码的调用链会更复杂：例如 Hugging Face `GenerationMixin` 在不同版本把 logits processors、warpers 和 stopping criteria 组合成不同顺序；vLLM 或 TensorRT-LLM 还会把采样放在批量 CUDA kernel。若要比较框架，必须固定版本和采样器顺序，不能只复制一个参数字典。

本 lab 中的随机性边界也很具体：`sample_categorical` 接收一个 `random.Random`，不使用全局随机状态；同一 seed 和同一浮点输入应得到同一输出。GPU 上的并行归约、不同 dtype 或不同 kernel 可能仍有最后一位差异，所以“固定 seed”不是跨硬件位级复现的保证。

### 固定版本的源码阅读任务

本仓库提交 `btt-phase2@5aaeeb530385b740107f0b88f113227709ca1443` 的 [softmax 片段](https://github.com/PeppaPigw/Beneath-the-Tokens/blob/5aaeeb530385b740107f0b88f113227709ca1443/labs/phase2/decoding_sampler.py#L50-L58) 只有 9 行：先调用温度缩放，排除 `-∞` mask，减去有限 logits 的最大值，再把指数和归一化。正常路径是返回和为 1 的列表；全是 `-∞`、NaN 或正无穷时，异常路径必须抛 `ValueError`，避免把空支持集伪装成概率。

[grammar mask 与 speculative 片段](https://github.com/PeppaPigw/Beneath-the-Tokens/blob/5aaeeb530385b740107f0b88f113227709ca1443/labs/phase2/decoding_sampler.py#L227-L268) 展示了另一个状态机：输入是同词表的 p、q 和 draft token，逐 token 计算 alpha；接受就追加并继续，拒绝就计算残差、采样替代并立即丢弃后缀。异常路径包括词表长度不一致、draft token 越界和空/非法概率。读者可以在 40 行以内逐行标出输入、状态和输出，再对照 `test_zero_draw_does_not_accept_zero_probability_token` 的 forced-RNG 回归测试。

## 把伪代码映射到一次请求

读者可以用下面的最小伪代码检查一轮 decode 的输入和输出，而不用先理解某个框架的 scheduler：

```text
z = model(prefix)
p = softmax(z / T)
p = support_filter(p, k, top_p, typical_p, min_p)
if grammar_state:
    z = mask(z, allowed(grammar_state))
    p = softmax(z)
x = sample(p, rng)
append(prefix, x)
advance(grammar_state, x)
```

第一行的 `z` 是模型输出，最后一行的状态才会影响下一轮。`support_filter` 若在概率上操作，要重新归一化；若在 logits 上操作，必须确认 `-∞` sentinel 被 softmax 正确识别。grammar mask 放在温度前后会产生不同 p，因此配置文件要保存 processor 顺序。异常路径包括模型 logits 全为 NaN、grammar 允许集为空、停止 token 已出现但循环仍继续、以及 RNG 被另一个请求共享导致 seed 复现失败。

speculative 一轮的伪代码稍有不同：

```text
for i in range(K):
    x_i = sample(q_i, rng)
    alpha_i = min(1, p_i[x_i] / q_i[x_i])
    if rng.random() < alpha_i:
        emit(x_i)
    else:
        emit(sample(residual(p_i, q_i), rng))
        discard(draft_suffix[i+1:])
        break
if all_accepted:
    emit(sample(p_{K+1}, rng))
```

`p_i、q_i` 是条件于当前前缀的分布，不是把一个向量复制 K 次就能代表真实模型。教学 lab 复用向量是为了让读者能打印整张表；迁移到模型后要把每个位置的 target logits、draft logits、accepted flag 和 residual checksum 写入 trace。若 `q_i[x_i]=0`，该 x_i 不可能由 q_i 提出，不能进入 alpha 分支；若实现捕获到这种输入，应该记录 proposal bug，而不是把除零吞掉。

```
正常：draft -> alpha<1 仍可能接受 -> 继续 verify -> emit
拒绝：draft -> alpha=0 -> residual -> emit替代 -> 丢弃后缀
错误：draft-match -> 直接贪心 -> 目标随机性被改写
```

这段状态图也是源码 review 的最小单元：审阅者可以逐项问“哪个函数拥有 RNG？哪个函数更新 grammar？拒绝后谁负责清理 KV？”如果答案散落在不同线程而没有 trace，性能和分布问题就很难区分。

## 两条路线的分界：改模型目标，还是减少目标前向

把方法分成下面两组，先问“最终 token 的边缘分布还是 p 吗？”

| 路线 | 代表方法 | 做了什么 | 目标分布 | 主要代价 |
| --- | --- | --- | --- | --- |
| 分布选择 | 温度、top-k、top-p、typical、min-p | 改 logits 或支持集后重归一化 | 变成 p' | 质量、熵、重复和校准改变 |
| 反馈/重排 | Mirostat、contrastive search | 在线控制 surprisal 或用隐藏状态重排 | 变成目标函数定义的 p' 或确定性序列 | 控制振荡、隐藏状态成本、长度偏置 |
| 结构约束 | FSM/CFG/JSON/SQL mask | 支持集与合法前缀相交 | grammar 条件分布（通常不同于无约束 p） | tokenizer 边界、死状态、状态爆炸 |
| 无损加速 | speculative、树验证、自推测 | 草稿 + target verify + 残差校正 | 在假设满足时仍为 p | draft 成本、低接受率、KV/显存 |
| 近似加速 | 只接受 draft-match、贪心验证、未经校正的 Medusa 变体 | 少做验证或省掉残差 | 一般是 p''，需测偏差 | 质量和可比性不可直接继承 |

“无损”只描述概率目标，不承诺延迟一定下降。草稿模型若与目标差异很大，接受率低，额外的草稿和验证可能比串行目标更慢。相反，top-p 即使让文本更好看，也不能标为无损，因为它明确删除了尾部。

## 接受—拒绝推导的逐行检查

把证明写成概率账本，可以发现实现中哪些数字应该被记录。仍用三个 token 的 `p=[0.6,0.3,0.1]` 与 `q=[0.2,0.5,0.3]`。先看草稿提出每个 token 的联合质量 `q(x)`，再看硬币接受后的质量：

| token | q(x) | p(x) | alpha | 接受质量 q·alpha | 残差未归一化 `[p-q]+` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.2 | 0.6 | 1 | 0.2 | 0.4 |
| 1 | 0.5 | 0.3 | 0.6 | 0.3 | 0 |
| 2 | 0.3 | 0.1 | 0.3333 | 0.1 | 0 |
| 合计 | 1 | 1 | — | 0.6 | 0.4 |

接受质量加总是 0.6，拒绝概率是 0.4。残差归一化后是 `[1,0,0]`，因此拒绝分支把剩下的 0.4 全部交给 token 0。最终 token 0 的总质量是接受分支 0.2 加残差分支 0.4，等于 0.6；token 1 是 0.3，token 2 是 0.1。证明不是凭“看起来校正了”成立，而是每一列相加刚好回到 p。

多位置时，把这一张表换成每个前缀的 `p_i、q_i`。如果第一个 draft token 被拒绝，第二个 token 的前缀已经不再是草稿所假设的前缀，所以只能丢弃后缀并从当前残差/target 继续。若所有 K 个 token 都接受，目标模型还要从最后一个已验证前缀抽取额外 token；省略这个额外 token 会让输出长度统计和目标序列分布都错一位。

### 三个容易混淆的“无损”说法

**分布无损**：对声明的 target p，接受—拒绝输出的边缘与 p 相同。它允许 q 很差，只是效率可能很差。**能力无损**：模型权重或训练后的任务能力没有下降，这是 Medusa-2 等联合训练需要独立评测的命题。**延迟无损**：几乎没有这个保证；draft、verify、通信和调度开销会随 batch 改变。论文把其中一个命题证明了，不能把三个都读成“无损”。

同样地，“输出和 target greedy 一致”只能说明一个确定性路径相同，不能说明随机采样的分布无偏。若目标是 greedy，接受条件可以简化为 token 相等；若目标是温度/top-p 后的随机 p'，必须使用 p' 和 q 的接受/残差。把采样目标从 p 改成 p' 是合理的产品选择，但要在配置和证据中明确。

### 经验 KL 的置信区间

假设运行 N 次单步输出，计数为 `c_i`，经验频率 `\hat p_i=(c_i+α)/(N+α|V|)`，其中 α 是为避免零项的平滑常数。`KL(\hat p||p)` 会同时受到随机抽样、平滑和 N 的影响；α=0 时，只要 `c_i=0` 而 p_i>0，经验 KL 的某个方向项可能仍然是 0，但反向方向会发散。报告时写清方向和 α，不要把一条未平滑的零计数当作“模型从来不生成”。

N=10000 在五 token toy 上能给一个有用的教学误差范围，但不能代表词表五万项的真实尾部。真实模型应按 token 频率分桶或使用校准曲线，同时检查停止 token、grammar mask 和截断导致的结构性零概率。若比较 two implementations，先确认它们声明的是同一个 p；否则 KL 差异可能只是目标不同，而不是实现错误。

## 采样器教学卡：每张卡都写清成功与失败

### Holtzman 等：nucleus/top-p 为什么出现

当时的问题是：最大似然模型的困惑度很低，但逐 token 取最大值会生成重复、无聊的文本；固定 beam 或 top-k 对不同上下文的尾部形状不够灵活。Holtzman 等观察到人类文本的概率质量与模型分布的可靠区域并不一致，提出用累计质量阈值选择 nucleus。实验采用 GPT-2 在 WebText 上的 open-ended continuation：5000 个 passages、每个最多 200 个生成 token，比较机器文本与人类文本的分布以及多样性、流畅性和连贯性；这不是“p=0.9 在所有聊天模型上最优”的证明。今天仍保留自适应支持集的思想，但 tokenizer、对齐训练和解码后处理都变了。证据账本：工作负载和 5000/200 的样本边界来自论文实验设置，硬件与重复口径应查正文而不能补猜。现在的实现通常把 nucleus 作为 logits warper，与温度、重复惩罚和停止器按显式顺序组合。

### Locally Typical Sampling：典型信息量而非最高概率

Meister 等把每个 token 的 surprisal 与局部熵比较，保留接近熵的 token。直觉是人类交流既不总选最可预测词，也不总追逐稀有词。论文在摘要与故事任务中报告相对 top-k/top-p 的竞争性质量并减少退化重复；任务规模、模型和阈值 `τ` 限定了这个结论。实现顺序尤其重要：典型筛选接在温度后，得到的熵和支持集与先筛选再温度不同。证据账本：工作负载是 abstractive summarization 与 story generation，指标包含自动/人工质量和退化重复，摘要页未报告硬件；现代实现把 `typical_p` 暴露为可组合的 logits warper，仍需锁定阈值和顺序。

### Mirostat：把困惑度当作反馈目标

Mirostat 的瓶颈是手调 top-k/p 很难跨上下文保持一致。它在线观察生成 token 的 surprisal，估计尾部斜率并调节候选集，目标是控制期望 PPL。论文的控制器与 Zipf 假设是关键，不是一个神奇的“质量旋钮”。当 `η`、目标 PPL 或分布形状不合适时，可能振荡或输出过于保守；论文的硬件和数据实验不能替代用户领域评测。证据账本：论文报告不同长度的生成、困惑度/交叉熵、重复与人工流畅性/连贯性，摘要没有给出硬件型号；今天的 Mirostat 2 实现常以动态阈值接在温度/截断之后，本章只保留可单测的控制状态。

### Contrastive search：重排高概率但相似的候选

对比搜索先取 top-k 候选，再用 `α log p(v|h) - (1-α) max_j cos(h_v,h_j)` 之类的目标惩罚与历史隐藏状态过于相似的候选。它常被误写成“更好的随机采样”，但核心是确定性重排序，目标已经变了。长上下文要保存或近似历史表示，成本和长度偏差都需测；论文在特定开放式生成基准的结果不能推出事实性提升。证据账本：论文覆盖 16 种语言的四类生成任务并用人工评价，摘要未给硬件与完整吞吐数字；今天的实现仍需隐藏状态缓存和 top-k 重排，训练表示与解码器是两个不同接口。

这里也要核对名称：`A Contrastive Framework for Neural Text Generation` 是训练/表示框架（arXiv:2202.06417），而本卡的解码论文是 `Contrastive Search Is What You Need for Neural Text Generation`（arXiv:2210.14140）。两个标题相近，引用时不能混用。

### Grammar-constrained decoding：合法结构不是免费午餐

PICARD 和后续 grammar-constrained 工作把解析器/自动机放到生成循环中，拒绝会导致语法错误的前缀。Geng 等论文展示了不微调也能把结构约束用于信息抽取、实体消歧和句法分析；它证明的是结构合法性与任务表现的组合，不是所有 schema 都能零成本运行。工程上需要缓存状态到 token 的可接受前缀，处理 tokenizer 的多字符 token，并对空集提供清晰错误。约束掩码改变支持集，除非你把 grammar 条件分布定义成新目标。证据账本：Geng 的工作负载是信息抽取、实体消歧和成分句法，PICARD 是 Spider/CoSQL text-to-SQL；指标是结构/任务得分，论文摘要没有硬件型号。今天的实现把解析器状态缓存为 tokenizer-aware 可接受前缀，仍必须处理死状态。

### Speculative decoding：接受—拒绝把串行依赖折叠

Leviathan、Kalman、Matias 的关键瓶颈是大型模型每个 token 都串行运行；小模型提出 K 个候选，目标模型一次验证。论文证明接受—拒绝校正后输出分布与目标一致，并在特定 T5 模型、硬件和任务上报告加速。实验中的速度取决于 draft/target 比例、batch、序列长度和实现，不能把论文倍数写成服务 SLO。证据账本：论文以 T5 草稿/目标和生成任务测加速及分布一致性，硬件与完整重复统计应以正文表格为准；今天的 runtime 通常把接受器、残差采样和 KV 管理拆成独立组件。

### Medusa：多头 proposal 的边界

Medusa 在同一 backbone 隐状态上增加多个预测头，构造候选树再由主模型验证。Medusa-1 冻结 backbone，并配合 rejection sampling/严格验证时可以保持原目标分布，因此论文把这一路线称为 lossless；Medusa-2 与 backbone 联合微调，目标是更好的预测与速度，模型本身已经变了，不能把联合训练后的质量称为对旧模型 target 的 exact preservation。Medusa 的 typical acceptance 方案主要是提高候选接受率的质量/效率折中，也不自动等于严格的原 target 分布校正。若实现只把多头 token 与主头做相等匹配，或省略残差，就应称近似加速并测 KL/任务质量，而不是继承“无损”标签。额外头、树宽和训练数据会改变收益。证据账本：Medusa 论文比较多种模型大小与训练流程，指标是 speedup 和生成质量，摘要未列硬件型号；今天的实现保留多头 proposal/树验证，但头的训练检查点、backbone 能力和严格接受规则必须一起锁定。

### EAGLE：预测特征而不是只预测 token

EAGLE 重新思考 draft 的不确定性，用目标模型的特征/轻量层产生候选，再用原 LM head 并行采样和验证。arXiv 论文报告了其模型家族上的接受率与速度，但这些数字依赖训练好的 draft head、硬件和实现版本。教学上保留两点：自推测仍需要明确的校正规则；“同一个 backbone”不等于“没有额外训练成本”。证据账本：EAGLE 评估 Vicuna、Llama2-Chat、Mixtral 以及对话、代码、数学、指令任务，报告接受/延迟/吞吐与分布保持，硬件细节需查正文；今天的实现把 feature draft head、目标 LM head 和校正器作为版本化 artifact。

### Lookahead decoding：没有外部 draft 也要付出结构复杂度

Lookahead 用 Jacobi/lookahead 序列并行提出和验证 n-gram，论文把算法描述为 exact，并报告在若干模型上的加速。它减少外部草稿依赖，却引入 n-gram 表、迭代收敛和缓存管理；在短序列、低并发或目标模型本身很快时，额外工作可能抵消收益。所谓 exact 仍需要实现遵守停止、随机性和数值假设，不能仅凭“输出相同几次”证明。证据账本：论文摘要报告 MT-Bench 最高约 1.8×、代码补全多 GPU 强扩展最高约 4×，硬件和重复次数在正文；今天的实现需额外维护 n-gram/迭代状态，短序列或低并发不应照搬倍数。

### Min-p：相对峰值的动态截断卡

**它要解决什么问题**：高温下 top-p 的累计质量阈值有时会把一整片低概率尾部放进来，产生不连贯或重复；固定 top-k 又不随模型信心变化。**核心想法**是用最高概率 `p_max` 做尺度，设 `θ=m·p_max`，保留 `p_i≥θ` 的 token 后重新归一化。它改变的是目标分布，和 speculative 的“保持 p”不是一回事。

**论文证据**：Nguyen 等在 GPQA、GSM8K、AlpacaEval Creative Writing 上比较了 Mistral/Llama 3 的 1B–123B 模型，报告高温条件下质量、多样性与人评偏好；摘要没有给出完整 GPU 型号、重复数和端到端延迟，因此只能迁移“相对峰值门槛”的机制，不能把论文的偏好写成线上保证。**今天的实现**已经在 Transformers、vLLM 等框架暴露 `min_p` 参数，但默认值、温度顺序和浮点 tie-break 仍由具体版本决定。

**限制与反例**：多峰分布可能只留下最高峰；`m→1` 近似 greedy，`m=0` 退化为原分布。评测至少保存 `m`、`p_max`、支持集大小、熵、`KL(p'||p)`、重复 n-gram 和任务成功率。若只报一条高温样本“更有创意”，无法区分阈值选择与 prompt 偶然性。

## 从单步分布到短序列：为什么局部选择会累积成全局差异

单步概率看起来只差一点，连续生成几步后会差很多。假设某个前缀下两个策略只在两个 token 上不同：原分布 `p=[0.7,0.3]`，截断后 `p'=[0.9,0.1]`。一位 token 的交叉熵差异是有限的；若四步都遇到相同形状的条件分布，某条序列的概率比会乘四次：

```
P_p(AAAA)=0.7^4=0.2401
P_p'(AAAA)=0.9^4=0.6561
比值约 2.73
```

真实模型每一步的前缀不同，不能把这个比例外推到整段文本，但它提醒我们：解码器不是最后一步的后处理器，而是在每一步把自己的选择写回上下文。一个被 top-k 删除的 token 可能本来会引导模型进入完全不同的主题；删除的影响会通过新的 hidden state 继续传播。

交叉熵和困惑度也要分开。若评测数据中的真实 token 是 y，单步 NLL 是 `-log p(y)`；PPL 是这组 NLL 的指数平均。采样器把 p 改成 p' 后，评测可以出现两种相反情况：样本看起来更自然，但对原始真实 token 的 NLL 变差；或者 PPL 轻微改善，但多样性和任务成功率下降。不能只看一个指标。

我们可以手算一个 KL。原 p 为 `[0.6157,0.2265,0.1018,0.0374,0.0186]`，top-2 后 p' 为 `[0.7310586,0.2689414,0,0,0]`。`KL(p'||p)=Σ_{i:p'_i>0}p'_i log(p'_i/p_i)`，只需计算前两项，约为 `0.171745 nats`（四舍五入 `0.1717`）。这个值明确说明 p' 与 p 不同；如果一段文字有 100 个条件步，局部偏差可能累积，必须在真实 prompt 上测经验 KL 或 token-level NLL，而不是凭单步直觉下结论。

还有一个方向问题：`KL(p'||p)` 衡量“从新策略看，原模型质量损失多少”；`KL(p||p')` 在 p' 删除了正概率 token 时直接变成无穷。后者正好表达 top-k 的硬删除风险，但在代码里要先处理零概率，不能把 `log(p/0)` 的浮点异常当作结果。报告 KL 时写清方向、零项处理和参考分布。

### 采样顺序是一条隐含的 API

把 logits processor 想成一串函数：`z → temperature → top-p → grammar → softmax`。函数一般不交换。例如上面的分布先 top-p=0.9 留下前三个 token，再 T=2，只会在三项中变平；先 T=2，尾部相对变重，top-p 可能留下四项。若 target 和 draft 在不同顺序上处理，得到的 p 与 q 会改变，接受率和效率也会改变；这会让两边不再对应同一套用户配置，但不是接受—拒绝正确性的定理要求。对任意已归一化的 proposal q（包括 q 某些位置为零），只要用声明的 target p 计算 `min(1,p/q)` 并用 `[p-q]_+` 残差，边缘分布仍是 p；因此“同一变换/顺序”是配置一致性和比较效率的建议，不是无损性证明的前提。

建议把顺序写进配置结构，而不是散落在服务启动参数中。一个可审计的配置至少包含：模型与 tokenizer 版本、温度、每种截断的阈值、grammar 状态、停止 token、随机数种子、是否在 draft 和 target 两边相同地执行。生产问题常常不是某个函数错，而是两个组件各自“默认正确”却组合成了不同的 p。

### 长度、停止和 EOS 的小陷阱

停止 token 也属于分布和测量边界。若 top-p 在 EOS 前执行，EOS 可能被删掉，服务只能到达 max_new_tokens 才停；若 grammar 不允许 EOS，合法结构完成后可能继续生成垃圾。统计 token-level distinct 时是否包含 EOS，统计 ITL 时是否把最后一次空输出算入分母，都要在实验记录中写出来。

同一 prompt 的长短输出不能直接比较重复率：短输出的 n-gram 样本少，长输出更容易碰到重复。按长度分层报告 distinct-n、重复 n-gram 和约束失败率，或至少给每个指标的样本数。这里不需要复杂统计模型，但需要承认分母改变了。

## 论文卡的证据账本：把“有效”拆成可核对的句子

到这里已经知道方法做什么，还需要问它在哪里被测过。下面的账本把“工作负载、主要指标、硬件可见性、今天的实现变化”放在同一张表里。表中写“摘要未报告”不是遗漏；如果一篇论文的摘要没有给出 GPU 型号，就不替作者补一个。

| 方法卡 | 论文实际比较的工作负载 | 论文报告的观察/指标 | 硬件与统计口径 | 今天的实现保留与改变 |
| --- | --- | --- | --- | --- |
| nucleus / Holtzman | GPT-2 WebText open-ended continuation：5000 passages、最多 200 tokens | 人类/机器分布、diversity、fluency、coherence 与退化 | 具体硬件、重复统计需查正文；不能把曲线当服务 SLO | 保留累计质量支持集；与重复惩罚、停止器和 batch sampler 组合 |
| locally typical | abstractive summarization、story generation | 自动和人工质量、退化重复 | 摘要未报告型号；阈值与模型需固定 | 保留 surprisal-熵排序；框架中成为可组合 warper |
| Mirostat | 不同长度的语言生成 | PPL/交叉熵随长度、重复、人工流畅性与连贯性 | 摘要未报告型号；控制器依赖 Zipf 假设 | Mirostat-1/2 使用反馈状态；服务需记录 eta、目标 PPL 与实际 surprisal |
| contrastive search | 四类文本生成、16 种语言 | 质量、人评与退化表达 | 摘要未给吞吐/硬件细节 | 保留 hidden-state 相似度重排；训练表示和推理重排分开 |
| grammar/GCD | 信息抽取、实体消歧、成分句法 | 结构合法率与下游任务分数 | 摘要未报告型号；输入相关 grammar 影响可比性 | 保留 grammar 状态；实际系统增加 tokenizer trie、缓存和死状态错误 |
| PICARD | Spider、CoSQL text-to-SQL | 解析/任务表现与状态可接受性 | 论文需按表格核对硬件和运行时；摘要只称 state-of-art | 从增量解析器演化为 schema/SQL/JSON 等多种约束接口 |
| speculative decoding | T5 草稿/目标和生成任务 | 端到端加速、接受率/输出分布一致性 | 正文才有硬件与重复，摘要不足以迁移倍数 | 接受器、残差采样、KV 管理拆分；多位置用条件 p_i/q_i |
| Medusa | 多种模型大小与训练流程 | Medusa-1/2 speedup 与生成质量 | 摘要未列型号；训练方案影响结论 | 采用预测头/候选树；严格校正才对原目标无损 |
| EAGLE | Vicuna、Llama2-Chat、Mixtral；对话、代码、数学、指令 | 接受/延迟、吞吐和分布保持 | 摘要给 Llama2-Chat 70B 的倍数，硬件细节在正文 | 训练 feature draft head；与目标 LM head、校正器一并版本化 |
| SpecInfer | 树式 speculative serving | 树验证延迟、计算和质量保持 | 需要按系统实验的模型/硬件表复核 | 共享前缀树和并行 verifier；树宽增加显存/调度压力 |
| Draft & Verify | Llama-2 及变体，自推测 | 早退 draft 的速度与原模型输出一致 | 摘要给最高约 1.99×，其余统计看正文 | 同一模型跳层作 proposal；仍要原模型完整 verify |
| Lookahead | MT-Bench、代码补全，单/多 GPU | 摘要报告最高约 1.8×、强扩展约 4× | 硬件与重复在正文；不能照搬到短 CPU 请求 | 无外部 draft 的 n-gram/Jacobi 迭代；新增状态与收敛开销 |
| min-p | GPQA、GSM8K、AlpacaEval Creative Writing，Mistral/Llama 3（1B–123B） | 高温下质量、多样性、人评偏好 | 论文版本含 ICLR 2025；具体 GPU 与重复按正文 | 保留相对峰值门槛；框架默认值和论文阈值不能混用 |

读表时有两个检查动作。第一，`paper_result` 只表示论文在自己的条件下测到某个数；“机制可迁移”是另一个 `inference`，需要自己的 workload。第二，硬件未报告并不代表论文没有硬件，只表示摘要不足；真正复现前应打开 PDF 的实验设置、表格和代码 commit，把版本写入 manifest。这样做比在来源地图中堆十几个标题更能防止错误迁移。

## 反事实练习：同一 logits 的四种不同选择

为了让读者能预测而不是只看输出，固定原始 p 并做四个反事实。

**反事实一：把 `T=0.5` 当作 top-k。** 低温仍给所有五个 token 正概率，而 top-k=1 直接把后四项置零；两者都更偏向 token 0，却不是同一分布。若只比较 argmax 文本，可能误以为算法相同；看 entropy/support/KL 就能立即区分。

**反事实二：把 `top-p=0.9` 当作“取前三名”。** 本例确实需要前三名，因为前三项质量是 0.943967；换一个平坦分布，例如 `[0.22,0.21,0.20,0.19,0.18]`，top-p=0.9 会留下五项，而 top-k=3 仍只留下三项。参数不能脱离分布形状解释。

**反事实三：把 typical 的 `τ=0.9` 当作“概率大于 0.9”。** typical 的 τ 是累计质量，不是单个 token 概率阈值；本例最高 token 只有 0.6157，却被保留。若把 τ 写成 `min_p`，会把绝对/相对两个不同坐标系混在一起。

**反事实四：先 grammar 再温度和先温度再 grammar。** grammar mask 后只有合法 token，温度只会在合法集合内重加权；先温度再 mask 则先改变所有候选的相对比例，再删除非法项。若最终合法集合相同，条件分布仍可能不同；约束任务要把 processor 顺序作为输入的一部分。

这些反事实可以直接变成单元测试：给出 logits、参数和预期支持集，断言每个函数不修改输入、输出和为 1、被屏蔽项概率为 0，并在极端温度/极端阈值下报出可读错误。测试的目的不是让 API 看起来完整，而是把“参数含义”钉在数值不变量上。

### 实验的控制面：先写“不能证明什么”

本实验不是把五项概率当成语言模型，而是把解码器最小的控制面隔离出来。输入是固定 logits，输出是概率、支持集、熵、采样 token 和接受记录；没有模型权重、没有 tokenizer、没有 GPU kernel。因而实验可以证明“这个实现满足归一化和接受—拒绝的局部不变量”，不能证明“某个温度使中文更自然”或“speculative 在某张卡上加速多少”。把边界写在实验之前，能避免看到漂亮的 toy 数字后才补免责声明。

基线是未截断的 `T=1` 分布；参数扫描（温度、top-k、top-p、typical、min-p 和 grammar）是读者可按下列 protocol 扩展的对照设计。本次已执行的 `--aggregate` artifact 只固定原始五项 logits、`top_p=0.9` 的 64 条长度 8 toy 序列，以及一轮接近 p 的 draft q；它汇总 distinct-2、接受率和 accepted token 数。grammar、q2 和其余阈值的扫描尚未声称已执行，读者应按配置逐项加入并保存独立 artifact。

每一个配置都要保存完整配置，而不是只保存均值：模型/词表标识（本例写 `toy-vocab-5`）、logits、温度、截断参数、grammar 状态、draft/target 分布、随机种子、代码 commit、Python 版本和输出路径。这样读者可以区分“参数改变了输出”与“脚本版本改变了输出”。随机种子只控制伪随机数，不消除浮点、排序 tie-break 或不同硬件 reduction 的差异。

### 五个 seed 的聚合与分位数边界

lab 的 `--aggregate` 运行 seed 0–4，并把每个 seed 的原始记录与 summary 写入 `reports/decoding-toy-seeds-0-4.json`。summary 对 distinct-2、接受率和 accepted token 数计算 min、p50、p95、max。这里的 p95 是“5 次重复实验的第 95 百分位”，不是请求延迟 p95；五个点只能给出很粗的范围，不能当置信区间。

示例中每个 seed 都会显示相同的原始熵和 top-p 支持集，因为 logits 没变；distinct-2 和 speculative 输出会随 seed 改变。若把 q 替换成更偏离 p 的向量，接受率分布会下降，但 toy 输出没有真正的 target forward 时间，所以不能从 accepted token 数推算 tokens/s。真实服务实验还要加入 draft_ms、verify_ms、target_calls/token、TTFT、ITL/TPOT、p50/p95/p99、显存和 batch，并在至少 100 个 prompt、多个长度桶上重复。

为什么仍然报告 min/max？它们能帮助发现一个 seed 的异常路径，例如残差为空、空支持集或停止 token 提前出现；但它们不能代替误差条。若要把实验扩展到真实模型，应预注册 prompt 集和 seed，使用 bootstrap 或置信区间，按长度和任务分层，报告失败样本而不是只保留通过样本。

### 验收不变量与一个故意失败的配置

运行测试前，读者可以先写下四条预测：

1. 每个合法概率向量的和在 `1±1e-12` 内；温度和截断不会制造负概率。
2. top-k、top-p、typical、min-p 的零概率位置集合与支持规则一致；`p=1` 和 `min_p=0` 保留全部正质量。
3. grammar 空允许集、越界 token、NaN logits 应显式报错，不应返回全 `-∞` 或静默恢复原分布。
4. speculative 中 alpha=0 的候选永远不会被接受，即便随机数恰好是 0；拒绝后输出长度不超过草稿长度，后缀被丢弃。

故意失败的配置是 `temperature=0`、`top_p=0`、`min_p=1.1` 和 `allowed_tokens=[]`。正确行为是 ValueError，错误行为是 NaN、死循环或把所有 token 当作可选。第二个故意失败是把接受条件写成 `draw <= alpha`；在强制 RNG 返回 0、alpha=0 时，它会错误接受零概率 token。这个测试看似边缘，却能阻止一个难以复现的概率泄漏。

### 从 toy 到真实模型的迁移清单

迁移第一步是替换 logits 来源，而不是同时接入框架、量化和 GPU。使用一个固定小模型，在单 batch、短 prompt 上打印每步 logits 的 checksum；再逐项打开 temperature、truncation、grammar。迁移第二步是固定 tokenizer，把 token id、解码文本、前缀状态和停止原因写入 trace。迁移第三步才增加 continuous batching、KV cache 和 speculative draft；每次只改变一个变量。

真实模型中，typical 和 min-p 的排序可能出现相同分数，需固定稳定排序；grammar mask 可能在 CUDA kernel 中用 finite sentinel 而不是 IEEE `-∞`，需检查 softmax 实现是否把 sentinel 当零质量；speculative target/draft 可能使用不同 dtype，残差的负数截断和归一化应在高精度累加。每个迁移阶段都保留 toy 的单元测试，并增加一条端到端“同一 seed、同一配置、同一 prompt”的 trace 对照。

## 可运行实验：一台 CPU 先把因果链跑通

### 实验问题与控制变量

我们不在 CPU 上假装测 GPU token/s，而问三个可复现问题：

1. 同一 logits 下，T、top-p、typical、min-p 如何改变熵、支持集和 distinct-2？
2. 草稿 q 越接近目标 p，单轮接受概率和输出长度是否上升？
3. 接受—拒绝残差是否把样本边缘拉回 p？

已执行的 aggregate 控制变量是五 token 词表、固定 logits、Python 版本、seed 0–4 和每个 seed 64 条长度 8 的 toy 序列；报告每个 seed 以及 min/p50/p95/max，不把 320 条 toy 序列外推为自然语言质量。命令：

```bash
python -m unittest discover -s labs/phase2 -p 'test_*.py'
python labs/phase2/decoding_sampler.py --seed 7 > reports/decoding-toy-seed7.json
python labs/phase2/decoding_sampler.py --aggregate > reports/decoding-toy-seeds-0-4.json
```

机器信息可用 `python --version` 和 `uname -a` 记录；本实验不需要 GPU、网络或第三方包。原始输出可重定向到 `reports/decoding-toy-seed7.json`。我们使用 `time.perf_counter`（若扩展脚本计时），预热一次后再测；本 lab 的 CLI 只输出分布和接受记录，避免把启动时间当作 token 延迟。

### 示例输出与解释

在 Python 3.11、seed=7 的一次示例中，原始概率约为 `[0.6157, 0.2265, 0.1018, 0.0374, 0.0186]`，熵为 `1.5359 bits`，top-p=0.9 的支持集大小为 3。64 条 toy 序列的 distinct-2 约为 0.02；这个数字很低是因为词表只有五项且序列很短，不能称作“模型重复率”。草稿 round 的接受率在该 seed 恰为 1.0，只说明这一次抽到的 token 都满足硬币，不说明期望接受率。

要测分布偏差，可把 `speculative_acceptance_simulation` 放进 10000 次循环，统计输出 token 的频率，再算经验 KL；接受规则正确时，误差应随样本数下降，但随机置信区间、浮点和同一分布复用会影响结果。更严格的单步实验为每个 token 独立提供 `p_i、q_i`，而不是像教学函数一样复用一个 p/q；章节中的证明适用于条件分布，lab 的简化只用于接口和故障演示。

### 可比指标

- **支持集大小**：非零 token 数；说明截断强度，不说明语义质量。
- **熵与 KL**：熵用 nats 或 bits 写单位；`KL(p'||p)` 以原 p 为参照，不能对称交换。
- **distinct-n、重复 n-gram**：多样性代理；短序列会有强烈小样本偏差。
- **接受率与每轮输出长度**：speculative 的机制指标；还要报告 draft 次数和 target calls/token。
- **延迟分位数**：真实 serving 才测 TTFT、ITL/TPOT、p50/p95/p99；本 CPU toy 不含模型前向，不能证明硬件收益。

## 从单步函数到 serving 指标：不要把接受率当作速度

在真实 serving 中，一次请求不只经历 `sample()`。prefill 先处理整个输入，decode 阶段每轮读取 KV、运行目标或 draft、执行 logits processor、采样并流式返回。TTFT 通常包含排队、prefill 和第一 token；ITL/TPOT 关注后续 token 间隔。speculative 主要可能改变 decode 阶段的 ITL，grammar 可能增加每步 mask/解析开销，temperature 本身通常只增加很小的算术工作。把三者放在同一“加速”标签下会把测量对象混在一起。

一个可解释的 trace 至少有这些时间戳：请求进入队列、prefill 开始/结束、draft 开始/结束、target verify 开始/结束、grammar mask 开始/结束、token emit、请求结束。每个时间戳带 request id、step、draft length、accepted length、support size、拒绝原因和 KV 版本。这样当 p95 变坏时，可以区分“等待更久”“draft 更慢”“verify 批量变小”“grammar 状态爆炸”四种原因。

**设计判断**：对开放文本，先锁定 p/T/top-p 和停止规则，再比较 target-only 与 speculative；对 JSON/SQL，先验证 grammar 合法率和空状态，再尝试 draft。一个高接受率的 draft 如果导致 target verify 的 batch 变小，端到端仍可能变慢；一个支持集很小的 grammar 如果减少重试，平均 token 数可能下降但解析 CPU 上升。指标必须同时覆盖工作量、延迟和失败路径。

### 一张最小 serving 表

| 阶段 | 输入/输出 | 主要指标 | 典型反例 |
| --- | --- | --- | --- |
| prefill | prompt token、KV | TTFT、prefill tok/s、峰值显存 | 长 prompt 让 decode 的改进被淹没 |
| draft | q、K 个候选 | draft ms、草稿长度、q 的熵 | q 很快但几乎全拒绝 |
| verify | p、候选树/批量 logits | verify ms、accepted/token、target calls/token | 接受率高但 batch 变小 |
| processor | logits、grammar 状态 | mask/排序 CPU 或 kernel 时间、support | 空集重试、tokenizer 状态爆炸 |
| emit | token、stream | ITL/TPOT、p50/p95/p99、断流率 | 平均变好但 p99 变坏 |

表格中的“典型反例”是实验假设，不是自然规律。读者应为每行写一个最小复现实验：固定其它阶段，增加一个变量，保存原始 trace。比如验证 batch 影响时，让 q 和 p 不变，只把并发从 1、2、4、8 改变；验证 grammar 影响时，让模型和温度不变，只替换 schema 的状态数。

### 何时该回退

回退应由可观察门槛触发：若 speculative 的 target calls/token 没有下降、p95 ITL 连续三个窗口超过 target-only 20%，或 grammar 空状态超过 0.5%，先回到同一 p 的 target-only/无约束 baseline，同时保留失败 trace。不要用更高温度“补救”一个 draft 失配，也不要在 JSON 失败时静默放宽 grammar。回退配置和原配置都要写入 experiment manifest，确保之后能区分真实修复和流量变化。

对于多租户系统，还要记录每租户的 prompt 长度、schema hash、draft 版本和采样参数。全局平均接受率可能掩盖某个租户的低接受率或 grammar 拒绝；公平的比较按租户和长度分层，并给出最差分位数。任何“系统快了”的结论都应回答：哪个阶段快了、谁受益、谁的 p95 变坏、目标分布是否仍然是声明的 p。

## 失效边界与失败诊所

### 诊所一：合法 JSON 偶尔被拒绝

**观察**：schema 测试的通过率从 99.8% 降到 94%，错误集中在字符串含有中文标点时。平均生成时间没变。

**看起来合理的假设**：模型概率变了；或者温度太高。

**最小排查**：记录拒绝前的字符前缀、token id、tokenizer 解码文本和 grammar 状态。用同一个前缀离线枚举“能使字符 DFA 继续可接受”的 token，而不是只检查 token 的第一个字符。

**证据**：若某 token 一次展开为两个 Unicode code point，而自动机只按字节推进，允许集会错误为空；若替换 `T=1` 仍失败且日志显示空集，原因是 tokenizer/FSA 边界，不是温度。

**修复与回滚**：使用 tokenizer-aware trie；统一 NFC 规范化和 JSON escape；空集包含 grammar 状态、前缀和 tokenizer 版本并立即失败。不要悄悄放行所有 token 作为“恢复”，那会把结构约束变成装饰。回滚时保留原始拒绝样本。

### 诊所二：speculative 比纯目标还慢

**观察**：target-only 每 token 20 ms，draft+verify 变成 24 ms；接受率只有 0.31。

**假设**：验证 kernel 没有批量化；draft 与 target 分布差得太远；或者 K 取太大。

**最小排查**：固定输入分别扫 `K∈{2,4,8,16}`，记录每轮接受长度、draft ms、verify ms、target calls/token 和 p95，而不只看平均 tok/s。先将 q 替换成接近 p 的模拟分布，区分“算法低接受”与“实现开销”。

**证据**：若 q 接近 p 时接受率升高且 verify 时间不变，草稿质量/窗口是瓶颈；若接受率高仍变慢，可能是内存搬运或批处理退化。接受率按 token 计，不能用“整轮全接受率”替代。

**修复与回滚**：减小 K、换更合适的 draft、把验证 logits 批量化；在低并发或短输出回退 target-only。回滚条件应写成 p95 或 tokens/target-call 的阈值，而不是“看起来不快”。

### 诊所三：温度越高，答案越“有创意”也越不稳

**观察**：T 从 0.8 调到 2.0，distinct-2 上升，但 JSON 失败和重复的长尾都上升。

**排查**：画每步熵、支持集大小和约束拒绝数；分别比较“先温度后 top-p”和“先 top-p 后温度”。检查 target 和 draft 各自声明的 p、q 是否已归一化、接受率是否按这两个分布计算。让两边使用同一温度/截断顺序是保持用户配置和提高接受率的建议；若两边使用不同变换，p/q 变了但只要残差校正仍基于声明的 p、q，接受—拒绝仍然精确。

**修复**：为结构任务使用 grammar 条件分布和较低熵；为开放式文本按验证集选择 T/p，而不是把创意代理当质量。记录 T、截断参数、随机种子和停止条件。不要用一条高温样本推断模型变聪明。

### 诊所四：离线 benchmark 看起来异常地好

**观察**：一个 top-p 配置在本地 prompt 集上重复率和任务得分同时提升，换成新模板后退化。

**可能原因**：prompt 泄漏、答案存在于上下文、调参集与测试集重叠、停止 token 不一致，或只挑选了成功样本。

**最小排查**：冻结 prompt 清单和版本 hash；给评测者隐藏解码参数；增加未见模板、负例和长度分层；报告所有失败和拒绝，而不是只写完成样本。用 token-level NLL 与任务分数分开保存。

**结论**：benchmark 是测量工具，不是授权信。若随机种子、prompt 或 tokenizer 未记录，结果不可复现；若只测试短句，不能推断长上下文的 p99 或 KV 影响。

## 失败注入矩阵：先制造症状，再排除解释

一个健康的基线很重要，但只有成功样本，读者学不到排障。下面的注入都在五 token lab 或一个最小服务 wrapper 中完成，目的是让“观察—假设—证据”形成闭环。

**注入 A：极低温度。** 把 T 设为 `1e-9`，观察第一项接近 1、其它项仍可能因浮点下溢变成 0。假设是“softmax 错了”；排查应先打印减最大值后的 logits 和有限支持，再把 T 改为 0.1、0.01 观察趋势。修复不是把所有低温强行改成 1，而是规定最小合法 T、记录 tie 行为，并在任务层决定是否接受近似 greedy。

**注入 B：极端阈值。** 把 top-p 设为 `1e-12`、min-p 设为 1。最小实现仍必须保留至少一个正质量 token并归一化；如果返回空列表，错误发生在参数校验而不是模型。对 min-p=1，若最大值唯一，输出应退化为该 token；若最大值并列，支持集应包含所有并列最大项。这个边界也能检验“稳定 tie-break”是否被错误地写进数学定义。

**注入 C：草稿完全失配。** 令 q 把全部质量放在目标低概率 token 上，记录 alpha、拒绝位置、残差和每轮输出长度。预期是接受率低、残差承担主要质量，输出仍按 p；若输出频率偏向 q 或 target argmax，说明实现省略了校正。此实验只测分布正确性，不测速度，因为 toy 没有前向时间。

**注入 D：grammar 空集。** 构造一个 schema 状态没有任何可接受 token，或者用错误的 tokenizer normalization 让所有候选都被拒绝。预期是带状态和前缀的明确错误；若服务继续生成，可能形成无限重试或把非法文本交给下游。修复后加入一个合法 escape token和一个真正非法 token的双向回归。

**注入 E：benchmark 泄漏。** 把用于选择 T/top-p 的 prompt 同时用于报告结果，再换一组未见模板。若得分和重复率突然恶化，不能只说“模型随机”；检查参数选择、停止规则和 prompt hash 是否泄漏。实验 artifact 应保存调参集与评估集的 hash，禁止从失败样本中静默删除难例。

矩阵的价值在于每一项只改一个因素。若同时改 T、grammar、draft 和 batch，观察到的改善无法归因；若只记录最终文本，无法判断是支持集、随机数还是后处理造成。即使最终选择回滚，失败 trace 也应进入 manifest，作为下一次改动的对照。

## 方案边界比较：同一工作负载下的选择

以下比较使用同一个五 token toy 分布和长度 8 的输出。它不是性能排行榜，而是一张设计决策草图。

| 选择 | 目标 | 需要保持的假设 | 可能收益 | 最坏情况/回滚 |
| --- | --- | --- | --- | --- |
| greedy/temperature | 可控的熵与确定性 | logits 数值稳定、停止条件固定 | 实现简单、p50 低 | 模式坍缩或高温垃圾；回到 T=1/验证集 |
| top-p vs top-k | 删除尾部噪声 | 排序和重归一化一致 | top-p 对分布形状自适应 | p 太小删掉必要 token；保留全量 p 做对照 |
| typical vs min-p | 以信息量或峰值相对阈值筛选 | 熵、峰值和参数记录 | 可能减少重复并保留多样性 | 多峰分布时误删整峰；报告 support/KL |
| Mirostat vs 固定 p | 在线控制 surprisal | 目标 PPL 与 eta 稳定 | 跨上下文少调参 | 振荡、目标与语义脱钩；降低 eta 或回退固定 p |
| grammar mask vs 后处理重试 | 生成中保证结构 | tokenizer-aware 状态自动机 | 解析失败率下降 | 空集和状态爆炸；记录拒绝并 fail fast |
| target-only vs speculative | 减少串行 target calls | 同一 p/q 变换和残差校正 | 高接受率时降低 ITL | draft 不匹配时更慢；按 p95 回退 |
| strict Medusa/EAGLE vs draft-match | 多头/特征草稿 | 严格验证才能无损 | 更少外部模型 | 省校正会变分布；测 KL 后再宣称近似 |

这里有一个重要的负结论：不能把“质量提升”和“无损”放在同一列。前者是选择一个你愿意评测的新目标，后者是证明输出仍按旧目标。生产评审应该分别签字。

## 源码观察与论文证据地图

本章 lab 的固定入口是本仓库 `btt-phase2@5aaeeb530385b740107f0b88f113227709ca1443`（`labs/phase2/decoding_sampler.py`）；它展示稳定的数值和状态转移，不代表任何 GPU serving 框架的 kernel 性能。若读者阅读 Transformers，可从 `transformers@v4.44.2` 的 generation 目录开始，先找 logits processor/warper 的调用顺序，再对照自己的版本；不要把 `main` 的路径写成历史证据。

论文来源按机制排列：Holtzman 的 nucleus 解释神经文本退化与自适应支持集；Meister 的 locally typical 给出 surprisal-熵排序；Basu 的 Mirostat 给出在线 PPL 控制；Su 的 contrastive search 给出隐藏状态重排；Geng、PICARD 相关工作说明 grammar/解析约束；Leviathan 给出接受—拒绝校正；Medusa、EAGLE、SpecInfer、Draft & Verify 和 Lookahead 分别探索多头、特征、自推测、树和无外部 draft 的验证路径。

读者应把每个“论文结果”拆成三问：原文在什么硬件和数据上测到？实验是否报告重复和分位数？今天的代码保留了哪个机制、改了哪些接口？manifest 中每条 claim 只写一个可核查陈述，限制字段会说明不能从它推断什么。

## 研究级边界：哪些结论不能从一张曲线推出

把论文和 toy 实验放到一起时，至少有五种“看起来像证据”的东西需要拆开。第一，**单步概率**只说明一个前缀下的选择，不说明长序列的语义；必须记录每步上下文或按长度分层。第二，**经验接受率**只说明草稿 token 被接受的比例，不说明验证 kernel 是否有足够并行度；要同时测 draft/verify 时间和 target calls/token。第三，**结构通过率**只说明输出符合 grammar，不说明字段值正确或业务约束满足；需要 schema 级别的任务检查。第四，**distinct-n 上升**可能来自输出变长或支持集变宽，不等同于人类偏好；应报告长度和重复 n-gram。第五，**论文 speedup**可能是在强扩展、多 GPU 或特定 batch 上得到，不能移植为单请求 CPU 或另一张卡的 p95。

这五种边界对应五个实验动作：固定上下文、记录每轮状态、保存失败样本、按长度/租户分层、复核硬件和版本。若无法执行动作，就把结论标成 `unverified_hypothesis`，而不是把“未来可以测”写成“已经成立”。manifest 的 limitation 字段应具体到“未报告硬件型号”“只复用单步 q”“不含 tokenizer”，让后来读者知道下一次该补哪个证据。

还要区分**实现缺陷**和**目标选择**。top-p 删除尾部是目标选择，不是 bug；grammar 空集是约束输入或 tokenizer 的错误；接受率低是 q 与 p 的关系，不一定是实现错；p95 变坏可能是 batch/内存，而不是接受—拒绝公式错。每类症状都有不同回滚：目标选择回到原 p 做对照，输入错误修复 schema/tokenizer，效率问题调 K 或 draft，公式错误立即停止并做分布校验。

如果读者把本章迁移到一个新 runtime，建议先写一页“声明目标”：原始 p、温度后的 p_T、截断后的 p'、grammar 条件分布，或它们的组合。然后为每个目标指定 sampler、校正器和证据。没有这张声明，团队很容易把“质量更好”“结构合法”“分布无损”“服务更快”四个不同结论混在一个 benchmark 表格里。

## 研究问题与理解检查

1. **为什么 top-p=0.9 不是“保留 90% 的 token”？** 先按概率排序，再累加到质量 0.9；支持集大小取决于分布形状。用上一节的 p，第三个 token 后质量约 0.944，因而需要第三个但不需要第四个。
2. **grammar mask 后是否还能说“无损”？** 若目标被重新定义为 grammar 条件分布，可以说对新目标做精确采样；相对原始无约束 p，它是分布改变。只有 speculative 的残差校正证明了对原 p 的边缘保持。
3. **接受率 0.8 是否保证 1.8× 加速？** 不保证。要扣掉 draft 时间、verify 时间、批处理和 KV 读写，并报告 target calls/token 与 p95。接受率只是必要线索，不是端到端因果结论。
4. **把拒绝 token 直接替换成 target 最大 token 会发生什么？** 残差 `[p-q]_+` 被丢弃，输出向 mode 偏移；用单步频率和经验 KL 可以把偏差测出来。
5. **Mirostat 的 PPL 目标达到后，事实性是否也达到？** 没有这样的推导。PPL 控制 surprisal 的统计量，事实性需独立数据、检索和人工/任务评测。

## 练习：从复算到设计

### 练习 A：复算支持集和熵

给 `p=[0.5,0.25,0.15,0.1]`，手算 top-k(k=2)、top-p(p=0.7) 的归一化概率，计算熵的表达式（保留自然对数）。验收标准：说明为什么两者支持集相同但参数含义不同，并给出至少 4 位小数。

### 练习 B：定位约束拒绝

构造一个 token 表：`0="{\""`、`1="answer"`、`2="："`、`3="}"`、`4="<前导空格>"`。设计一个只允许 JSON 键的 DFA，记录每个状态的 allowed token。故意把中文冒号按 ASCII 冒号处理，提交一条最短失败前缀和修复后的规范化规则。验收标准：失败日志含状态、前缀、token id，而非只写“解析失败”。

### 练习 C：接受率实验

令 `p=[0.6,0.3,0.1]`，分别取 `q1=[0.55,0.35,0.1]`、`q2=[0.2,0.5,0.3]`，K=4，各位置可先复用同一分布。运行 10000 个 seed，报告每个 q 的接受率、输出长度和经验 KL；再解释复用同一 q 与真实条件 draft 的差异。验收标准：固定算法后只改变 q，写出不能从 toy 结果推断的硬件结论。

### 练习 D：设计回退策略

为一个 p95<300 ms 且 JSON 通过率>99.5% 的服务选择参数。给出最小监控面板（TTFT、ITL、p95、拒绝率、support size、draft acceptance）和两个回滚阈值。验收标准：把“分布质量”与“加速正确性”分成两条告警，不以平均值掩盖尾延迟。

## 练习参考与延伸：把答案写成证据

练习不应只收一个数字。下面给出参考路径，读者可以先遮住答案，再检查自己的中间步骤是否与它们一致。

### 练习 A 的参考步骤

`p=[0.5,0.25,0.15,0.1]` 时，top-k(k=2) 的支持集是 token 0、1，质量和为 0.75，所以新概率是 `[0.6667,0.3333,0,0]`。top-p(p=0.7) 先取 token 0 得 0.5，再取 token 1 得 0.75，支持集碰巧相同，归一化结果也相同。它们的参数含义仍不同：top-k 先规定数量，top-p 先规定累计质量；把分布换成 `[0.4,0.2,0.15,0.15,0.1]` 后，top-k=2 仍取两项，top-p=0.7 需要前三项，因为前两项只有 0.6。

熵可写成 `-[0.5 log 0.5+0.25 log 0.25+0.15 log 0.15+0.1 log 0.1]`，约为 1.208 nats。截断后的熵要对重新归一化的分布重新计算，不能把原熵减去被删除质量；被删除 token 的位置和质量共同决定差异。验收时除了数值，还要检查读者是否写出单位以及 KL 的方向。

### 练习 B 的参考排查

若状态 q1 期望 ASCII `:`，token 表只有中文全角 `：`，最短失败前缀可以是 `{"answer"` 后的下一个字符。字符级 DFA 会把全角字符判为非法；tokenizer-aware trie 应在规范化步骤统一 NFC 与 schema 允许的标点，或者明确把全角标点加入 grammar。日志至少包括 `q1`、前缀的 Unicode code point、token id 2、token 解码文本和 tokenizer hash。若日志只有“解析失败”，无法判断是 schema、tokenizer 还是模型概率导致。

一个好的修复要有回归样本：同一前缀在修复前 allowed 集为空，修复后包含 token 2；同时保留一个真正非法的半角/全角混合字符串，确保“放宽规范化”没有把任意字符都放行。通过率上升不是唯一指标，还要确认非法样本仍被拒绝，状态缓存键没有把不同 schema 混在一起。

### 练习 C 的参考实验

`q1` 的逐 token 接受率为 `[1,0.8571,1]`（token 0、1、2），`q2` 为 `[1,0.6,0.3333]`。这只是条件于各 token 被提出时的 alpha，不是整轮接受率；整轮还由 q 抽到什么 token 和拒绝后何时停止决定。运行 10000 个独立 seed 时，应保存每次输出长度、拒绝位置、替代 token 和 RNG seed，然后按 token 计接受率。

正确残差在这两个例子都集中于 token 0，但 q2 的拒绝更频繁。若把拒绝替换为 target argmax，经验频率会进一步偏向 token 0；若把草稿后缀保留，后续条件分布不成立。报告应同时给原 p、经验输出频率、`KL(empirical||p)` 和置信区间，说明五 token、复用单步 q 与真实 draft 的差异。不能把“接受率高”直接换算成 GPU speedup。

### 练习 D 的参考决策

p95<300 ms 与 JSON 通过率>99.5% 是两个不同的门槛。最小面板可分为质量列（grammar 失败率、重试率、支持集、输出长度）和加速列（draft ms、verify ms、accepted/token、target calls/token、TTFT/ITL）。例如连续 5 分钟 p95 超过 300 ms，或者 grammar 失败率超过 0.5%，触发 target-only 与低温配置回退；若接受率低于 0.45 但 p95 尚未坏，先缩短 K、保留日志并观察，而不是立即把 target 分布改掉。

回滚阈值要注明窗口和分母：一次异常样本不能触发全局回滚，但连续三个窗口都越界应冻结新配置。回滚后仍保留失败样本，用于判断是 draft 失配、schema 漂移、GPU 争用还是 benchmark 输入变化。把质量和加速告警拆开，才能避免通过降低温度“修复”一个本来属于验证 kernel 的问题。

## 研究问题：从教学模型走向可证伪的系统问题

1. **变换顺序是否存在可预测的最优策略？** 在同一模型和 prompt 集上固定 `T∈{0.7,1,1.3}`、top-p、typical 和 grammar 顺序，测 NLL、合法率、distinct-2、KL 与 ITL。假设应写成“某类熵/峰值分布下顺序 A 的支持集更稳定”，而不是“顺序 A 总是最好”。
2. **接受率和端到端延迟的拐点在哪里？** 固定 target、draft 与 K，逐渐增加并发和上下文长度，记录 draft/verify/KV 时间的 p50/p95/p99。把接受率高但 draft 成本大、接受率低但 verify 可并行的两种反例分开；只看平均 accepted/token 会漏掉尾延迟。
3. **grammar 状态缓存如何影响多租户隔离？** 用相同前缀、不同 schema、不同 tokenizer 版本构造缓存键碰撞实验，检查合法率、缓存命中和错误泄漏。研究问题的验收不是“缓存更快”，而是不同租户不会读取彼此的 allowed 集。
4. **min-p 的相对门槛如何与模型校准关联？** 对峰值概率分桶，比较阈值、支持集、人工偏好和任务成功率。若高峰值桶的最佳 m 与低峰值桶不同，固定 m 的设计判断应被替换成分桶策略，并报告过拟合风险。
5. **无损的“目标分布”到底是哪一个？** 温度、top-p、grammar 和 repetition penalty 已经定义了一个 p'；speculative 应该对这个 p' 做校正，而不是对原始 softmax p 做校正。实验可故意让 target 使用 p'、draft 使用 q，验证残差仍恢复 p'；再记录错误地恢复 p 时的 KL。

每个研究问题都需要一个停止条件：样本数、prompt 版本、硬件和允许误差预先写好；若结果不能区分两个假设，记录为未决而不是挑一条曲线。研究级教材的价值不在于为每个方法给出冠军参数，而在于让下一次实验知道该改变什么、保留什么、怎样解释失败。

## 配置与隔离：采样参数也是请求状态

采样参数通常被放进请求 JSON，但在多租户 serving 中它们也属于状态和资源预算。租户 A 的 grammar、tokenizer 或 repetition penalty 不能沿用到租户 B；共享 grammar cache 时，键至少包含 schema hash、tokenizer 版本、Unicode 规范化规则和 allowed-set 版本。共享 draft 模型时，target/draft 的模型版本、温度、截断和停止 token 也必须进入接受 trace，避免把一个租户的 q 用在另一个租户的 p 上。

参数验证应在进入 scheduler 前完成：温度必须为正有限数，top-p 在 (0,1]，k 在词表范围，min-p 在 [0,1]，grammar 状态必须能枚举至少一个 token。失败请求返回结构化错误并释放已分配 KV；不能让一个空 grammar 状态占着 batch 等待重试。日志脱敏时仍保留参数 hash、状态编号和错误类别，以便复现而不泄露 prompt。

安全评测还要覆盖拒绝路径。攻击者可以构造极端高温、超大 K、巨大 schema 或不断触发空集的输入，消耗排序、解析和 KV。服务应设置每请求 token/状态/重试上限，监控 grammar CPU 时间和 speculative draft 开销，并在预算耗尽时回退到明确的错误或安全的 target-only 配置。回退不会让输出自动正确，但能避免用无限重试掩盖资源攻击。

### 可观测性字段的最小集合

一次解码 trace 可以用一行 JSON 表示：`request_id`、`step`、`prefix_hash`、`model_version`、`tokenizer_version`、`temperature`、截断参数、`grammar_state`、`support_size`、`entropy`、`draft_length`、`accepted_length`、`target_calls`、`stop_reason` 和 `error_class`。这些字段不要求保存完整 prompt，却足以回答“目标分布是什么、支持集如何变、为何在此处拒绝、回退是否生效”。

字段要有版本和单位：entropy 标注 nats 或 bits，latency 标注毫秒，长度标注 token；`null` 与 0 的含义不能混淆。没有 grammar 时写 `grammar_state=null`，不是 0；没有 draft 时 `accepted_length=0` 但 `draft_length=null`，不是一次零接受。数据管线应保留 schema 版本，让旧 trace 在新 parser 中回放时能明确报告“不兼容”，而不是静默套用默认值。

当 p95 回归时，先按 `error_class` 和 `support_size` 分组，再按模型/租户/长度切片；如果只有平均值，空 grammar 或长 schema 的少数请求会被淹没。复盘记录应把假设、命令、原始输出、修复和回滚条件绑定到同一个 experiment_id，和 manifest 的 claim 一一对应。

### 读完本章后的复盘

如果只能带走三句话，第一句是“先声明你要采样的分布”：温度、截断和 grammar 都可能把 p 换成 p'。第二句是“无损加速依赖校正”：draft 可以很差，但接受率和残差必须按声明的 target/proposal 计算，拒绝后要丢弃不再成立的后缀。第三句是“测量要对齐问题”：熵、KL、合法率、接受率、TTFT、ITL 和 p95 各自回答不同问题，不能用一个漂亮数字代替整条证据链。

下一章如果继续学习 grammar，应把本章的字符/BPE 例子扩展为可缓存的 DFA、CFG 与 schema；如果继续学习 speculative，应把单步 p/q 扩展为带 KV 的树验证，并在目标硬件上记录每次 verify 的 batch 和内存。无论走哪条路，先保留这个五 token baseline，任何优化都必须能解释它改变了哪个支持集、哪个状态或哪一段时间线。

### 教师评审提示

审阅这章时可让读者现场完成三个动作：不用框架手算 top-p 和 KL；把一个多字符 token 放进 grammar trie 并解释状态变化；用 forced RNG 证明 alpha=0 不会接受。若读者只能复述“top-k、top-p、speculative”而不能指出支持集、残差和异常路径，说明抽象还没有落到机制。若能画出 p/q 表、运行 aggregate 并指出 toy 不能证明 GPU speedup，才算完成从公式到证据的迁移。

### 最后一次自检

在提交实验前重新打开原始 JSON，确认概率和为一、支持集与参数相符、所有 seed 都有记录、失败样本没有被过滤。再打开 manifest，逐条检查来源版本、实验编号和限制，确保一条论文结果没有被误写成生产保证。这样一轮短小的自检，往往比再增加一个采样器名称更能提高章节的可信度。

### 证据优先的阅读节奏

先读问题边界和五 token 例子，再运行 seed=7；然后合上代码，自己写出残差表，最后回到 paper card 对照硬件、任务和指标。读者若跳过中间的数字推导，后面的“无损”很容易变成口号；读者若跳过限制字段，论文结果又会被误搬到自己的服务。每一步都应留下一个可检查的产物：手算表、trace、测试输出或回滚记录。

### 交付前的一句话

本章没有选择一个“最好参数”，而是把参数、约束、校正、测量和回滚放在同一条可复核路径上；这是解码工程最值得迁移的能力。

### 可复查的停止点

当实验达到预先设定的停止条件，就保存结果而不是继续挑选有利样本；若证据不足，明确写出未验证假设并等待下一次测量。

### 让失败留下痕迹

失败命令、异常状态和回滚原因都应保留在 artifact 中；没有失败样本，就无法证明修复确实缩小了问题边界。

### 复现优先于漂亮数字

先固定定义、版本和统计口径，再讨论质量与速度；否则数字无法比较。

### 记录不确定性

无法测量的量不应被默认为零，而应标记为未知并说明下一步。

### 证据可追溯

每个结论都应能回到代码、论文或输出。

## 术语表

- **logits**：模型最后一层给每个 token 的未归一化分数；可整体平移而不改变 softmax。
- **softmax**：把分数变成和为 1 的非负概率映射。
- **temperature / 温度**：除以 T 的分布变换；T 小变尖，T 大变平。
- **surprisal**：`-log p(token)`，单位通常是 nats/token；越小表示越可预测。
- **entropy / 熵**：按概率加权的平均 surprisal；衡量不确定性，不等于质量。
- **support / 支持集**：概率大于零的 token 集；截断后必须重新归一化。
- **nucleus/top-p**：最小累计质量前缀截断。
- **typical sampling**：按 surprisal 与局部熵的距离选择 token。
- **min-p**：按最大 token 概率的相对比例筛选。
- **Mirostat**：以目标 PPL 为反馈信号的在线采样控制器。
- **grammar/FSM**：描述合法前缀的有限状态机；tokenizer-aware 版本按 token 展开推进。
- **speculative decoding**：草稿分布 q 提议、目标分布 p 验证并以接受—拒绝校正的加速方法。
- **acceptance rate**：被接受的草稿 token 数除以草稿 token 数；不是端到端 speedup。
- **residual distribution**：`[p-q]_+` 归一化后的拒绝校正分布。
- **self-speculative**：同一大模型的早退层/轻量头作草稿，再由完整模型验证。
- **lossless**：相对明确定义的目标分布保持边缘分布；不表示数值、延迟或任务质量绝对不变。

## 来源地图与复现清单

### 稳定来源

- Holtzman 等，*The Curious Case of Neural Text Degeneration*，arXiv:1904.09751（预印本版本）。
- Meister 等，*Locally Typical Sampling*，TACL 2023，arXiv:2202.00666。
- Basu 等，*Mirostat*，ICLR 2021，arXiv:2007.14966。
- Su 等，*Contrastive Search Is What You Need for Neural Text Generation*，NeurIPS 2022，arXiv:2210.14140；训练框架另见 arXiv:2202.06417。
- Geng 等，*Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*，EMNLP 2023，arXiv:2305.13971；PICARD，EMNLP 2021。
- Leviathan 等，*Fast Inference from Transformers via Speculative Decoding*，ICML 2023，PMLR 202。
- Cai 等，*Medusa*，arXiv:2401.10774；Li 等，*EAGLE*，arXiv:2401.15077。
- Miao 等，*SpecInfer*，arXiv:2305.09781；Zhang 等，*Draft & Verify*，arXiv:2309.08168；Fu 等，*Lookahead Decoding*，arXiv:2402.02057。
- Nguyen 等，*Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*，arXiv:2407.01082。

### 从干净 checkout 重跑

1. 使用 Python 3.11+，进入仓库根目录；不需要 GPU 或联网。
2. 运行 `python -m py_compile labs/phase2/decoding_sampler.py labs/phase2/test_decoding_sampler.py`。
3. 运行 `python -m unittest discover -s labs/phase2 -p 'test_*.py' -v`，应得到 13 个测试通过。
4. 运行 `python labs/phase2/decoding_sampler.py --seed 7`，保存 JSON；比较概率和熵，允许最后 1e-12 的浮点误差。
5. aggregate 的已执行复现是 `python labs/phase2/decoding_sampler.py --aggregate > reports/decoding-toy-seeds-0-4.json`；按练习 C 扩展到 `{0,...,9999}` 时，另写新 experiment_id，报告接受率、范围和经验 KL。若改了 RNG 或排序 tie-break，请在 artifact 中写版本。
6. 该实验测的是标准库 CPU 的逻辑正确性和 toy 分布，不证明 GPU throughput、模型事实性、长上下文 p99 或任何论文 speedup。要做真实服务评测，另行固定模型权重、tokenizer、硬件、batch、温度/截断顺序和停止规则。

复现后清理 `reports/decoding-toy-*.json` 等临时输出即可；源代码和测试文件保留。遇到 URL 网络失败时，保留论文版本号和访问日期，不把暂时不可达误报为论文不存在。
