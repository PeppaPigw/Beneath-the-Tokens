# Publishing architecture

The book uses Docusaurus 3 with React/TypeScript and MDX-capable Markdown rendering.

The book stays plain Markdown in docs/, independent of the site generator. The site provides navigation, table of contents, formula rendering, code highlighting, search integration points, and GitHub Pages deployment. Interactive labs remain separate from prose and are embedded through stable identifiers. A future milestone can freeze a curriculum version without rewriting current chapter URLs.

Planned layout:

~~~text
docs/                 authoritative chapters and editorial contracts
website/              Docusaurus shell
labs/                 independent browser or container experiments
packages/             reusable lab components and measurement helpers
scripts/              content, links, references, and safety validators
refs/                 machine-readable source maps
.github/workflows/    content CI and Pages deployment
~~~

No backend is assumed. Browser-safe labs use Web Workers/WASM/Pyodide where appropriate; cluster labs provide a bounded local fallback and an explicit resource warning.
