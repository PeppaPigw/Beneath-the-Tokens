# Beneath the Tokens

Beneath the Tokens is a research-grade, first-principles textbook for AI infrastructure. It is written for readers who want to move from zero background to designing, debugging, measuring, and operating production AI systems.

## What this book teaches

The book follows one continuous path:

- Linux processes, files, memory, observability, and failure diagnosis
- Networking, performance mathematics, GPU/CUDA, PyTorch execution, and numerical precision
- Collective communication, distributed training, scheduling, checkpointing, and experiment operations
- Data systems, evaluation, artifact management, retrieval/RAG, inference execution, and serving
- Kubernetes GPU orchestration, Ray runtimes, inference optimization, observability, reliability, security, cost, platform engineering, and AI-infrastructure frontiers
- An end-to-end capstone that connects the layers into one auditable system

The phase-two research track adds deeper chapters on decoding and sampling, KV cache/PagedAttention, and related inference mechanisms. These chapters include paper evidence, fixed-version source maps, mathematical derivations, runnable CPU experiments, reports, and explicit boundaries between toy evidence and production claims.

## How to read

Start with [the curriculum](docs/curriculum.md), then read the chapters in order. Each chapter is organized around a concrete failure or design problem and progresses through:

1. a mental model
2. mechanism and equations
3. source-code and paper evidence
4. a runnable experiment or reproducible protocol
5. failure diagnosis
6. exercises and comprehension checks

The chapter authoring contract, lab standards, and source policy are part of the book itself:

- [Chapter template](docs/chapter-template.md)
- [Lab and verification standards](docs/lab-standards.md)
- [Source and evidence policy](docs/source-policy.md)

## Verification

The repository includes content validation and phase-two evidence audits. Before publishing changes, run:

```bash
python scripts/validate_content.py
python scripts/audit_phase2.py --docs docs --sidebar website/sidebars.ts
```

The checks cover frontmatter, chapter structure, evidence manifests, sidebar references, source links, and reproducibility metadata. GPU and production-serving performance claims are never inferred from the CPU toy labs.

## Site

The book is deployed as a static GitHub Pages site from the `website` directory. The public repository is:

https://github.com/PeppaPigw/Beneath-the-Tokens

