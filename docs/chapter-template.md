# Chapter authoring contract

Every chapter is a small, auditable course.

## Required structure

1. Why this problem exists
2. Prerequisites and learning outcomes
3. One-sentence mental model
4. First-principles derivation
5. System mechanism, from request to hardware
6. Minimal implementation
7. Production implementation reading
8. Measured laboratory
9. Failure clinic
10. Trade-offs and rejected alternatives
11. Paper and repository synthesis
12. Six comprehension checks
13. Exercises: recall, derivation, implementation, diagnosis, design
14. Summary and next dependency
15. Source map and reproducibility record

## Writing rules

- Define every symbol before using it
- Separate fact, measurement, inference, and design advice
- Never use a framework name as an explanation
- Show the control flow and data movement
- Include at least one counterexample and one failure that looks healthy
- Explain what a metric can and cannot prove
- Prefer diagrams that answer a question over decorative diagrams
- Keep code runnable on a documented environment
- Pin versions for labs while explaining what is version-sensitive
- Use Chinese as the main language, retaining precise English terms on first use

## Completion gate

A chapter is ready only when:

- the prose is at least 10,000 Chinese characters
- the dependency and prerequisite links are correct
- code and lab commands work from a clean checkout
- expected observations are recorded
- six comprehension checks have answer keys
- an independent review found and addressed factual, causal, and pedagogical defects
- all claims have a source or are labeled as design judgment
