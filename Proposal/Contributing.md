# Contributing - Proposal & Documentation

This file covers how we manage the proposal document and documentation updates in this repository. Read this before making any changes.

---

## LaTeX Structure

The proposal is written in LaTeX with a modular file structure to minimise merge conflicts. **Do not edit `main.tex` directly** unless you are adding a new section import.

```
docs/
└── proposal/
    ├── main.tex               # Cover page + imports only — do not edit content here
    ├── sections/
    │   ├── abstract.tex
    │   ├── introduction.tex
    │   ├── problem_statement.tex
    │   ├── literature.tex
    |   |── literature
    |   |   |── ann.tex
    |   |   |── auditing.tex
    |   |   |── cancer.tex
    |   |   |── critical_analysis.tex
    |   |   |── ea.tex
    |   |   |── ga.tex
    |   |   |── nsga.tex
    |   |   |── sam.tex
    |   |   |── sota.tex
    |   |   |── xai.tex
    │   ├── methodology.tex
    |   |   |── ann.tex
    |   |   |── evaluation.tex
    |   |   |── expected_results.tex
    |   |   |── experiments.tex
    |   |   |── ga.tex
    |   |   |── nsga.tex
    |   |   |── overview.tex
    |   |   |── sam.tex
    │   └── timeline.tex
    └── references.bib
```

Each section is a separate `.tex` file. **Work only in your assigned file.** This isolates changes and makes conflicts easier to resolve.

---

## Section Assignments

| Section | File | Assigned to |
|---------|------|-------------|
| Abstract | `abstract.tex` | TBD |
| Introduction + Objectives | `introduction.tex` | TBD |
| Problem Statement + Scope | `problem_statement.tex` | TBD |
| Literature Review (SOTA XAI + Critical Analysis) | `literature.tex` | TBD |
| Methodology | `methodology.tex` | TBD |
| Timeline / Gantt chart | `timeline.tex` | TBD |

Assignments will be updated here once confirmed. Check back before starting work.

---

## Workflow

1. Branch off `dev` — never off `main`
2. Your branch for documentation is `docs/proposal`
3. Make your changes in your assigned `.tex` file only
4. Commit with a clear message, e.g. `docs: expand SOTA XAI subsection in literature`
5. Open a pull request into `dev` when your section is ready for review
6. At least one other team member should review before merging

---

## Key Pending Changes (from supervisor feedback)

These are the gaps identified — tick them off as they are completed.

- [ ] Add SOTA XAI subsection (SHAP, LIME, LRP, Integrated Gradients, Anchors) with comparison table
- [ ] Strengthen Section 3.7 critical analysis to critique cited papers, not only identify gaps
- [ ] Add formal objectives list (to evaluate / to determine / to compare / to demonstrate)
- [ ] Add scope and limitations paragraph
- [ ] Name research paradigm explicitly as proof by demonstration
- [ ] Add CART decision tree as SOTA rule extraction baseline in methodology
- [ ] Add technical specifications (software stack, compute environment, runtimes)
- [ ] Cut abstract to 100–200 words
- [ ] Replace Gantt chart placeholder with completed chart
- [ ] Resolve all reference placeholders (priority: Biecek & Samek 2024)
- [ ] Convert first-person constructions to third person throughout
- [ ] Fill patience parameter and run count placeholders in Section 4

---

## Commit Message Convention

```
docs: <what you changed>
```

Examples:
```
docs: add SOTA XAI comparison table to literature section
docs: resolve Biecek & Samek reference placeholder
docs: cut abstract to 180 words
```