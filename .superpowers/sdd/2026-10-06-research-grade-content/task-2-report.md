# Task 2 完成报告：随机解码与采样（修订版）

日期：2026-10-06

## 状态

完成研究级二期第 23 章、证据 manifest、标准库 CPU sampler、多 seed artifact、测试、phase2 audit 覆盖和 Docusaurus sidebar/build 接入。章节正文约 47,146 字符，其中中文字符 20,023，符合 20–40k 中文字符目标；内容补齐了开场双请求、五 token running example、熵/KL 数字核算、tokenizer-aware grammar、speculative 接受—拒绝证明、论文证据卡、失败注入、研究问题和复现清单。

## 交付文件

- `docs/phase2/chapters/23-decoding-and-sampling.md`
  - `paper_count: 13` 明确只统计正文主要教学卡：Holtzman、locally typical、Mirostat、contrastive search、GCD、PICARD、speculative、Medusa、EAGLE、SpecInfer、Draft & Verify、Lookahead、min-p
  - 温度、top-k/top-p、typical、min-p 都沿同一五 token 分布手算；top-2 精确值 `[0.7310586, 0.2689414]`，`KL(p_top2||p)≈0.171745 nats`
  - 明确区分分布改变、grammar 条件分布和接受—拒绝无损加速；补充“任意归一化 q 的残差校正正确性”和“同变换顺序是效率/配置一致性建议而非定理前提”
  - Medusa-1（冻结 backbone + rejection sampling 可无损）与 Medusa-2（联合训练、质量/能力命题、typical acceptance 非 exact target preservation）已拆开
  - 提供固定 commit 的 40 行以内源码阅读入口，解释 softmax、grammar、speculative 正常/异常路径
- `docs/phase2/evidence/23-decoding.json`
  - 36 条 provenance 条目，涵盖论文 workload/metric/限制、版本、源码观察、实验测量、设计判断和本地 artifact
  - 使用 `https://github.com/PeppaPigw/Beneath-the-Tokens/...` 实际仓库路径；论文版本核对为 Mirostat v2、Lookahead v1、GCD v6、Medusa v3、EAGLE v3、min-p v8 等
- `labs/phase2/decoding_sampler.py`
  - 稳定 softmax、temperature、top-k/top-p/typical/min-p、熵、Mirostat scalar update、grammar mask、acceptance/residual、speculative simulation、distinct-n
  - `--seed 7` 单次输出和 `--aggregate` seed 0–4 多 seed 汇总；新增 min/p50/p95/max（明确不是服务延迟分位数）
  - 保留 `*_filter`、`temperature_scale`、`mirostat_step`、`grammar_mask` 发现式别名
- `labs/phase2/test_decoding_sampler.py`
  - 13 个测试：数值稳定、熵方向、四种截断、Mirostat、grammar 空/越界、残差、确定性、多 seed summary、alias API、forced RNG alpha=0 严格拒绝
- `reports/decoding-toy-seed7.json`
- `reports/decoding-toy-seeds-0-4.json`
- `website/sidebars.ts` 加入 `phase2/chapters/phase2-23-decoding-and-sampling`
- `scripts/audit_phase2.py` / `scripts/audit_phase2_rules.json` 默认同时覆盖一期 `chapters/*.md` 与二期 `phase2/chapters/*.md`

## 论文/主张核对

已用一手页面核对 arXiv、ACL Anthology、PMLR、OpenReview 入口与版本。Manifest 对每张论文卡分别记录 workload、metric、硬件/统计是否在摘要中报告、今天实现保留/改变了什么及限制；未报告的硬件没有补猜。论文 speedup 都明确为 paper result，不当作服务 SLO。

## 验证命令与结果

```text
python - <<'PY'  # phase2_template chapter + evidence validators
... chapter PASS
... evidence PASS
PY

python -m unittest discover -s labs/phase2 -p 'test_*.py' -v
Ran 13 tests — OK

python -m unittest discover -s scripts -p 'test_*.py' -v
Ran 32 tests — OK

python -m py_compile labs/phase2/*.py scripts/*.py
PASS

python scripts/audit_phase2.py --docs docs --sidebar website/sidebars.ts --report /tmp/phase2-final-audit.json
phase2 audit: PASS (0 errors)

python scripts/validate_content.py
validated 27 chapter(s)

PNPM_HOME=/tmp/pnpm-home pnpm --config.store-dir=/tmp/pnpm-store ...
# dependency installation was blocked by ignored core-js build scripts in this managed environment
cd website && ./node_modules/.bin/docusaurus build
SUCCESS — Generated static files in "build" (21s)
```

`pnpm ... build` 的依赖安装命令因 managed 环境的 `ERR_PNPM_IGNORED_BUILDS` 被阻断；已直接使用安装完成的 Docusaurus binary 成功构建并验证 sidebar 文档 id。

## 已知限制

1. lab 是五 token 标准库教学模型，不证明真实 Transformer GPU throughput、TTFT/ITL、显存、长上下文 p99 或论文 speedup。
2. speculative simulation 为教学方便复用单步 p/q；真实实现需要每个位置的条件 p_i/q_i，并测更大样本的经验 KL/置信区间。
3. grammar_filter 是 logits 支持集 mask，不是完整 JSON/CFG parser；tokenizer-aware trie、Unicode 规范化和状态缓存需在约束专题继续实现。
4. Mirostat 只实现可单测的 scalar feedback update，没有复现完整 Zipf tail estimator。
5. 多 seed artifact 只执行固定 top-p toy round；温度/top-k/typical/min-p/grammar/q2 扫描是明确标注的 reader protocol，不冒充已测结果。
6. frontmatter 当前依赖一期稳定 id `ch01-ai-infrastructure` 和 `ch19-inference-optimization-accelerator-stack`，后续 phase2 基础章落地后可迁移依赖 id。

## 提交

主 artifact 提交为 `5aaeeb530385b740107f0b88f113227709ca1443`（包含章节、lab、reports、sidebar、audit scope 和本报告）；随后 `1b816c0b76610aa39be26134faeda2144c10799b` 将 frontmatter/manifest 的 source URL 与 version 固定到该 artifact。
