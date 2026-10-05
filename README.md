# Beneath the Tokens

**Beneath the Tokens** is a production-oriented, first-principles textbook for Artificial Intelligence Infrastructure.

The goal is not to memorize frameworks. It is to understand how data, models, kernels, accelerators, networks, schedulers, storage, serving systems, observability, and organizations compose into reliable AI systems.

## Learning contract

- Start from zero and build toward staff/principal-level AI infrastructure reasoning
- Every chapter combines concepts, mathematics, systems mechanisms, source-code reading, experiments, failure analysis, and exercises
- Each chapter targets at least 10,000 Chinese characters in the final edition
- Claims are linked to papers, official documentation, source code, and reproducible measurements
- The repository distinguishes explanatory text, runnable labs, reference implementations, and production caveats
- No chapter is considered complete until its comprehension checks, lab, and verification checklist are present

## Site

The book is published as a static GitHub Pages site. The current site scaffold uses Docusaurus 3; the content remains plain Markdown so it can be audited, versioned, searched, and rendered by other toolchains.

See:

- [Curriculum](docs/curriculum.md)
- [Chapter authoring contract](docs/chapter-template.md)
- [Lab and verification standards](docs/lab-standards.md)
- [Source and evidence policy](docs/source-policy.md)

## Status

All 26 long-form chapters are now published. Each chapter has frontmatter, source-oriented explanations, failure analysis, comprehension checks, exercises, and a lab or lab specification; chapters with runnable standard-library experiments include the corresponding script under labs/. The Docusaurus site scaffold, sidebar, and CI are present. A final remote build and content audit remains the release gate before treating the GitHub Pages deployment as verified.
