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

Current progress: ten long-form chapters are published, each with frontmatter, source-oriented explanations, labs or lab specifications, failure analysis, comprehension checks, and exercises. The site scaffold and CI are present; the first GitHub Pages build still needs to run in Actions before the published site is considered verified. Chapters will continue to be added as independently reviewed commits.

Content CI now runs the existing content validator, the deterministic phase-two audit, audit unit tests, Python syntax checks, website typechecking, the layout smoke test, and a production Docusaurus build. Runs that reach the audit step upload `reports/phase2-content.json` for review, including when validation or the audit finds content errors. The Pages workflow installs from the committed lockfile and publishes the generated `website/build` directory.
