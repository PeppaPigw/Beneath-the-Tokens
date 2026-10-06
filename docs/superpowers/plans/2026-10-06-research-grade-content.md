# Research-Grade Textbook Content Rewrite Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the一期 AIGC-like summaries with a beginner-first, paper-to-source-to-experiment AI Infrastructure textbook organized into 40 research-grade专题章.

**Architecture:** Keep一期 chapters deployable while adding a versioned `docs/phase2/chapters/` tree. Every new chapter uses a shared teaching template and an evidence manifest. Five sample chapters (decoding, KV memory, KV transfer/connectors, vLLM, SGLang) establish the quality bar before the remaining chapters are migrated in dependency order.

**Tech Stack:** Markdown/MDX-compatible prose, KaTeX, Python standard-library labs, GitHub Actions evidence reports, versioned paper/source links.

**Spec:** `docs/phase2-design.md`

## Global Constraints

- Teach in the order problem → intuition → minimal example → formal definition → mechanism → boundary → practice.
- A chapter targets 20,000–40,000 Chinese characters, but repetition does not count as quality.
- Every important claim is labelled as fact, definition, derivation, paper result, source observation, measurement, inference, design judgement, or unverified hypothesis.
- Every framework claim names a repository commit/tag and a reading path.
- Every performance number states hardware, software, workload, repetitions, percentiles, and uncertainty.
- Old chapters are not deleted until their replacement passes audit and build.

## Review Focus

- A beginner must be able to follow the first three sections without knowing framework names.
- A paper result must not be silently presented as a production guarantee.
- A source link pointing at `main` must not be used as evidence for an old version without a version note.
- A formula must define variables and include at least one numerical example.
- A lab that cannot run on the available machine must provide an honest CPU/simulation fallback and state what it cannot prove.

---

### Task 1: Freeze teaching template and evidence manifest

**Files:**
- Create: `docs/phase2/chapter-template.md`
- Create: `docs/phase2/evidence-manifest.schema.json`
- Create: `docs/phase2/style-guide.md`
- Test: `scripts/test_phase2_template.py`

**Interfaces:**
- Chapter frontmatter contains `phase`, `chapter_number`, `level`, `prerequisites`, `learning_objectives`, `paper_count`, `source_commits`, `lab_paths`, `last_verified`.
- Evidence manifest stores claim, type, source URL, version, experiment id, limitation, and review date.

- [ ] Write fixture chapters showing valid, beginner-first, and invalid template cases.
- [ ] Implement deterministic template/evidence validation.
- [ ] Run fixtures and record expected errors.
- [ ] Commit `docs: define phase two teaching and evidence contracts`.

### Task 2: Write the decoding sample chapter

**Files:**
- Create: `docs/phase2/chapters/23-decoding-and-sampling.md`
- Create: `docs/phase2/evidence/23-decoding.json`
- Create: `labs/phase2/decoding_sampler.py`
- Test: `labs/phase2/test_decoding_sampler.py`

**Interfaces:**
- Lab exposes deterministic functions for logits→probabilities, top-k/top-p/typical/min-p, temperature, Mirostat update, grammar filtering, and speculative acceptance simulation.
- Chapter explicitly separates distribution-changing methods from lossless acceleration.

- [ ] Write an opening two-request example and the probability/entropy notation.
- [ ] Add paper teaching cards for Holtzman, locally typical sampling, Mirostat, contrastive search, grammar-constrained decoding, speculative decoding, Medusa, EAGLE, and lookahead.
- [ ] Implement and test reproducible CPU experiments, including acceptance-rate and diversity/quality trade-offs.
- [ ] Add failure clinics for tokenizer constraints, low acceptance, temperature misuse, and benchmark leakage.
- [ ] Run lab and audit, then commit `docs: add research-grade decoding chapter`.

### Task 3: Write KV memory and connector sample chapters

**Files:**
- Create: `docs/phase2/chapters/26-kv-cache-paged-attention.md`
- Create: `docs/phase2/chapters/27-kv-compression-and-tiering.md`
- Create: `docs/phase2/chapters/28-kv-transfer-and-connectors.md`
- Create: `docs/phase2/evidence/26-kv.json`, `27-kv-compression.json`, `28-kv-transfer.json`
- Create: `labs/phase2/kv_block_pool.py`, `labs/phase2/kv_transfer_sim.py`
- Test: corresponding CPU simulation tests

**Interfaces:**
- Simulations model logical→physical block tables, refcounts, prefix hashes, COW, eviction, transfer manifests, leases, checksums, retry, and recomputation.
- Chapters distinguish local cache correctness, remote transfer consistency, and serving-level latency.

- [ ] Derive KV bytes/token and paged block fragmentation using a numerical example.
- [ ] Teach PagedAttention, prefix caching, KV quantization/compression, LMCache, Mooncake, DistServe, NIXL and connector semantics as separate historical mechanisms.
- [ ] Implement controlled simulations and report hit rate, memory, transfer bandwidth, recomputation, p95 latency, and failure recovery.
- [ ] Run chapter and lab audits, then commit `docs: add KV memory and transfer chapters`.

### Task 4: Write vLLM and SGLang sample chapters

**Files:**
- Create: `docs/phase2/chapters/29-vllm.md`, `docs/phase2/chapters/30-sglang.md`
- Create: evidence manifests and version matrixes for both
- Create: CPU scheduler/cache analog labs and tests

**Interfaces:**
- Each chapter records exact repository tags/commits, source paths, call chains, metrics names, version changes, and fair benchmark commands.
- Labs are explicit teaching models, not claims of GPU parity.

- [ ] Explain vLLM V0→V1→Model Runner V2 and SGLang RadixAttention in beginner-first order.
- [ ] Trace request lifecycle, scheduler, KV manager/cache, executor, sampler, streaming, metrics, and failure paths.
- [ ] Compare both frameworks with the same workload model and state what cannot be compared fairly.
- [ ] Run evidence/link/build audits and commit each framework independently.

### Task 5: Migrate remaining chapters by dependency order

**Files:**
- Create: `docs/phase2/chapters/01-25`, `31-40` as replacement chapters
- Modify: `website/sidebars.ts`, `docs/curriculum.md`
- Test: phase-two audit, all lab tests, Docusaurus build

**Interfaces:**
- Each replacement chapter links prerequisites by stable id and declares evidence/lab manifests.
- Sidebar can expose一期 and二期 during migration, then remove一期 entries only after replacement approval.

- [ ] Rewrite foundations before dependent chapters; do not merely append paragraphs to一期 text.
- [ ] Review each chapter for beginner clarity and AIGC-pattern violations before merge.
- [ ] Run full validation and deploy only after all replacement chapters pass.
- [ ] Commit each chapter or coherent pair separately with its audit report.
