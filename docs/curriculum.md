# Curriculum

The book is organized as a dependency graph rather than a list of fashionable tools. Chapters are long-form; each final chapter targets at least 10,000 Chinese characters plus a runnable lab.

## Part I — The substrate

### 1. What AI infrastructure is
Workload taxonomy, control plane/data plane, latency/throughput/cost/reliability, the boundary between model code and systems code. Lab: a measured local inference service and a bottleneck report.

### 2. Linux, processes, filesystems, and observability
Processes, threads, syscalls, page cache, cgroups, namespaces, containers, signals, file descriptors, perf/eBPF basics. Lab: diagnose a deliberately stalled worker.

### 3. Networking for AI systems
TCP/UDP, congestion, RDMA concepts, collectives, serialization, backpressure, tail latency, topology. Lab: compare serialized, pipelined, and batched transfers.

### 4. Performance mathematics
Little’s Law, queueing intuition, Amdahl and Gustafson, roofline model, bandwidth/compute bounds, tail distributions, cost per token. Lab: build a workload model and validate it against measurements.

## Part II — Compute and acceleration

### 5. GPU architecture and CUDA execution
SMs, warps, memory hierarchy, occupancy, kernels, streams, events, synchronization, CUDA graphs. Lab: profile and repair a bandwidth-bound kernel.

### 6. PyTorch execution internals
Autograd graphs, eager vs compiled execution, dispatch, memory allocators, streams, graph breaks, checkpointing. Lab: trace one model through Python, dispatcher, kernels, and allocator.

### 7. Numerical computation and precision
FP32/BF16/FP16/FP8/INT8/INT4, rounding, stability, loss scaling, quantization error, calibration. Lab: compare accuracy, memory, throughput, and failure modes across precisions.

### 8. Distributed communication
All-reduce, all-gather, reduce-scatter, point-to-point, NCCL-style collectives, topology-aware communication, overlap. Lab: implement and benchmark a ring collective.

### 9. Distributed training
Data, tensor, pipeline, sequence, and expert parallelism; ZeRO/FSDP; activation checkpointing; optimizer state; fault recovery. Lab: derive a memory budget and train a small model across processes.

### 10. Large-scale training operations
Scheduling, elastic jobs, checkpoints, preemption, data locality, experiment tracking, reproducibility, cluster utilization. Lab: recover a failed training run without duplicating or corrupting state.

## Part III — Data and model lifecycle

### 11. Data systems for AI
Object storage, filesystems, manifests, formats, sharding, caching, lineage, quality, privacy, deduplication. Lab: build a streaming dataset with deterministic shards.

### 12. Data preprocessing and evaluation
Tokenization, packing, sampling, curriculum, benchmark contamination, statistical power, reproducible evaluation, human evaluation. Lab: design an evaluation set and quantify uncertainty.

### 13. Experiment and artifact management
Configs, immutable artifacts, registries, metadata, provenance, reproducible builds, dataset/model versioning. Lab: reproduce a model result from a clean checkout.

### 14. Retrieval, vector systems, and knowledge services
Embedding pipelines, indexes, hybrid retrieval, filtering, freshness, recall/latency trade-offs, RAG failure modes. Lab: measure retrieval recall and end-to-end answer latency.

## Part IV — Inference and serving

### 15. Inference execution
Prefill/decode, KV cache, batching, continuous batching, speculative decoding, memory planning, admission control. Lab: build a measurable OpenAI-compatible endpoint.

### 16. Model serving systems
Worker pools, routing, autoscaling, streaming, multi-tenancy, model loading, warmup, graceful degradation. Lab: deploy a service with SLOs and load-test it.

### 17. Kubernetes and GPU orchestration
Pods, scheduling, device plugins, operators, node pools, gang scheduling, quotas, topology, upgrades. Lab: run a GPU workload with explicit resource and failure policies.

### 18. Ray and distributed application runtimes
Actors, tasks, placement groups, Ray Serve, KubeRay, resource accounting, state, autoscaling. Lab: serve a model through a distributed runtime and compare it with direct Kubernetes deployment.

### 19. Inference optimization and accelerator stacks
TensorRT, compilation, kernel fusion, FlashAttention-style techniques, paged KV memory, quantization serving, hardware-specific limits. Lab: profile a before/after optimization and explain every gain.

## Part V — Reliability, economics, and security

### 20. Observability and debugging
Metrics, logs, traces, profiles, exemplars, GPU telemetry, structured events, causal debugging, incident response. Lab: resolve a tail-latency incident from telemetry only.

### 21. Reliability engineering for AI
SLOs, error budgets, retries, idempotency, circuit breakers, checkpoint integrity, chaos experiments, disaster recovery. Lab: inject failures and demonstrate bounded recovery.

### 22. Security, privacy, and supply chain
Identity, isolation, secrets, data governance, model theft, prompt/data exfiltration, signed artifacts, SBOM, sandboxing, abuse controls. Lab: threat-model and harden a serving endpoint.

### 23. Cost and capacity engineering
GPU economics, utilization, reservations, queueing, energy, carbon, rightsizing, spot/preemption, cost per training token and served token. Lab: choose a capacity plan from workload traces.

## Part VI — Architecture and frontier systems

### 24. End-to-end AI platform architecture
Control-plane/data-plane design, platform boundaries, tenancy, developer experience, golden paths, migration, governance. Lab: design and review a complete platform.

### 25. Mixture-of-experts and sparse systems
Routing, load balance, expert parallelism, communication pressure, capacity factors, failure modes. Lab: simulate routing skew and repair it.

### 26. Multimodal and agent infrastructure
Image/audio/video pipelines, tool execution, state, sandboxing, long-running jobs, evaluation, provenance, human approval. Lab: build a recoverable multimodal workflow.

### 27. Training and serving at hyperscale
Supernodes, data-center fabrics, topology, collective scheduling, checkpoint systems, fleet operations, incident economics. Lab: read and critique a frontier-scale systems paper.

### 28. Research-to-production synthesis
How to extract mechanisms from papers and repositories, design experiments, reject cargo cults, write decision records, and lead an AI infrastructure program. Capstone: implement, benchmark, operate, and defend a complete system.

## Chapter sequencing

Chapters 1–4 establish the reasoning tools. Chapters 5–10 explain how compute becomes distributed training. Chapters 11–14 cover the lifecycle of data and knowledge. Chapters 15–19 turn models into services. Chapters 20–23 make systems dependable and economical. Chapters 24–28 integrate the entire field.

## Required chapter artifacts

Every chapter must ship:

- chapter Markdown
- source map with papers, official docs, repositories, and version/date
- at least one runnable lab
- expected outputs and interpretation guide
- failure clinic
- six comprehension checks with answer key
- glossary additions
- reproducibility notes
