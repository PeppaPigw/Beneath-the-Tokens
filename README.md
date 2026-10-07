# Beneath the Tokens

Beneath the Tokens 是一套面向 AI Infra 的研究型中文教材，目标是把读者从 Linux、网络和 GPU 基础，带到能够设计、测量、调试和运营生产级 AI 系统。

## 当前进度

主课程已完成 40 章，并按章节逐次通过内容校验和 GitHub Pages 部署。第 27–40 章组成深入的第二阶段，覆盖：

- KV cache 传输、connector、PagedAttention 与分层压缩
- vLLM、SGLang、随机/结构化解码与 speculative decoding
- MoE、Expert Parallelism、GPU 集群拓扑与 all-to-all
- 对象存储、并行文件系统、NVMe、数据加载与 checkpoint
- 编译器、图捕获、Inductor、XLA/MLIR、Triton/CUDA 与 kernel autotuning
- 在线推理调度、continuous batching、SLO、流式背压与多租户
- OTel、Prometheus、DCGM、Nsight、PyTorch Profiler、eBPF 与联合诊断
- 安全供应链、SBOM、provenance、GPU 隔离、成本/容量/能源
- 评测基准、质量与系统联合门禁，以及端到端生产 Capstone

## 全书方法

每章都从一个具体的工程问题或失败模式开始，依次给出：

1. 问题边界与心智模型
2. 机制、公式和系统控制流
3. 论文、官方文档、源码与版本证据
4. 可运行的 CPU-only toy lab 或可复现实验协议
5. 故障诊所、回滚边界和生产注意事项
6. 理解检查、练习与进一步研究问题

实验报告、phase2 audit 和 evidence manifest 与章节一起提交。toy 实验只证明明确的逻辑和账本，不把 CPU 数字冒充 GPU、NCCL 或生产吞吐保证。

## 从哪里开始

建议先读 [课程地图](docs/curriculum.md)，再按 1–40 章顺序阅读：

- [章节模板](docs/chapter-template.md)
- [实验与验证标准](docs/lab-standards.md)
- [来源与证据政策](docs/source-policy.md)

## 本地验证

```bash
python scripts/validate_content.py
python scripts/audit_phase2.py --docs docs --sidebar website/sidebars.ts
python3 tests/test_ch40_capstone_lab.py
pnpm --dir website typecheck
pnpm --dir website build
```

审计会检查 frontmatter、章节结构、来源链接、侧边栏引用、证据清单和可复现元数据。

## 在线站点

- GitHub 仓库：[PeppaPigw/Beneath-the-Tokens](https://github.com/PeppaPigw/Beneath-the-Tokens)
- GitHub Pages：[在线阅读](https://peppapigw.github.io/Beneath-the-Tokens/)

