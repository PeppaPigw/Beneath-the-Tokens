# Reading Layout and Content Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add accessible independent left-sidebar/right-TOC folding and a CI audit that protects the new textbook's structural and evidence standards.

**Architecture:** Keep Docusaurus static. Swizzle only the smallest theme layout boundary needed to render two client-side disclosure controls; CSS controls layout and localStorage stores preferences. A Python audit reads Markdown/frontmatter and emits human-readable failures plus a machine-readable evidence report; it runs before the existing Docusaurus build.

**Tech Stack:** Docusaurus 3, React/TypeScript, CSS, Python 3 standard library, pnpm GitHub Actions.

**Spec:** `docs/phase2-design.md`

## Global Constraints

- Markdown remains the source of truth; no backend or client database.
- Sidebar and TOC must be independently foldable, keyboard accessible, and usable on mobile.
- Existing一期 content must remain buildable during migration.
- Audit failures must distinguish missing/invalid URLs from network errors.
- No content-quality claim may be inferred from character count alone.

## Review Focus

- A page deep-linked with either panel already hidden must still show readable content and heading links.
- Refresh, back/forward, private browsing, and malformed localStorage must not break rendering.
- Mobile drawer Escape/focus behavior must not trap or lose focus.
- A missing prerequisite or duplicate frontmatter id must identify both files.
- A paper URL timeout must not be reported as a false 404.

---

### Task 1: Establish theme layout boundary

**Files:**
- Create: `website/src/theme/DocPage/Layout/index.tsx`
- Create: `website/src/theme/DocPage/Layout/styles.module.css`
- Test: `website/src/theme/DocPage/Layout/layout.test.tsx` (or a deterministic DOM smoke script if the repository has no test runner)

**Interfaces:**
- Consumes Docusaurus `DocPage/Layout` children and existing theme context.
- Produces `data-btt-sidebar-collapsed` and `data-btt-toc-collapsed` state attributes plus buttons with stable `aria-controls` values.

- [ ] Write a smoke test asserting both controls render, have accessible names, and toggling one does not change the other.
- [ ] Run the smoke test and record the initial failure.
- [ ] Implement the wrapper using React state initialized from `localStorage` only in an effect-safe path; preserve SSR output.
- [ ] Run the smoke test and a TypeScript check.
- [ ] Commit `feat: add independent reading panel controls`.

### Task 2: Implement responsive behavior and persistence

**Files:**
- Modify: `website/src/theme/DocPage/Layout/styles.module.css`
- Modify: `website/src/css/custom.css`
- Test: `website/src/theme/DocPage/Layout/layout.test.tsx`

**Interfaces:**
- Consumes Task 1 state attributes.
- Produces desktop width expansion, mobile drawer styles, Escape handling, and durable preferences under versioned localStorage keys.

- [ ] Add tests for independent persistence, malformed storage fallback, and Escape closing the mobile drawer.
- [ ] Implement desktop collapsed widths, mobile drawer transitions, focus-visible styles, reduced-motion media query, and no-JS-readable fallback.
- [ ] Run tests at desktop and mobile viewport breakpoints; verify keyboard focus and heading links.
- [ ] Commit `feat: make reading panels responsive and persistent`.

### Task 3: Add chapter metadata presentation

**Files:**
- Create: `website/src/theme/DocItem/Content/index.tsx`
- Modify: chapter frontmatter template and `docs/chapter-template.md`
- Test: metadata rendering smoke test

**Interfaces:**
- Consumes optional `level`, `estimated_hours`, `prerequisites`, `paper_count`, `source_commit`, `lab_path`, `last_verified` frontmatter.
- Produces an accessible metadata strip with omission-safe rendering.

- [ ] Test full, partial, and absent metadata cases.
- [ ] Implement metadata rendering without changing article Markdown semantics.
- [ ] Run build and inspect one old and one new-style document.
- [ ] Commit `feat: expose chapter learning metadata`.

### Task 4: Build the phase-two content audit

**Files:**
- Create: `scripts/audit_phase2.py`
- Create: `scripts/audit_phase2_rules.json`
- Create: `reports/.gitkeep`
- Test: `scripts/test_audit_phase2.py`

**Interfaces:**
- Command: `python scripts/audit_phase2.py --docs docs --report reports/phase2-content.json`.
- Exit 0 only when structural rules pass; report contains per-file checks, skipped network checks, and error classes.

- [ ] Write tests for duplicate ids, dangling prerequisites/sidebar ids, missing pedagogical sections, unpaired fences, placeholder words, script syntax, URL 404, and URL timeout.
- [ ] Implement frontmatter/heading/parser checks with only the standard library; never turn network timeout into 404.
- [ ] Implement optional URL checking with bounded concurrency only if explicitly enabled; default CI uses cached/declared evidence to avoid flaky builds.
- [ ] Run the audit against一期 and inspect the report.
- [ ] Commit `ci: add phase two content audit`.

### Task 5: Integrate CI and verify deployment

**Files:**
- Modify: `.github/workflows/content-ci.yml`
- Modify: `.github/workflows/deploy-pages.yml`
- Modify: `README.md`

**Interfaces:**
- Content CI runs validator, phase-two audit, script syntax checks, then Docusaurus build.
- Pages workflow consumes the same lockfile and generated `website/build` artifact.

- [ ] Add a CI step that uploads the JSON audit as an artifact even on content failure.
- [ ] Run local audit, `pnpm --filter beneath-the-tokens-site build`, and existing content validation.
- [ ] Push a small CI commit and verify both workflow conclusions and the live Pages HTTP response.
- [ ] Commit `ci: enforce phase two audit in content pipeline`.
