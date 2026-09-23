# Contributing — Proposal & Documentation

This file covers how we manage the proposal document and documentation updates in this repository. Read this before making any changes.

For the full breakdown of what needs to be written, why, and how — including research starting points for each SOTA method — see the **[Proposal Revision Guide](https://docs.google.com/document/d/1uksUNk0Xf9MOwRIdzfIkmajpppzs4Nk-w-diyGziJT4/edit?usp=sharing)**. That document is the single source of truth for all revision tasks. Claim your tasks in the tracker at the bottom of it.

---

## Git Commit Message Convention

```
docs: <what you changed>
```

Examples:
```
docs: add SOTA XAI comparison table to literature section
docs: activate scope and limitations paragraph
docs: cut abstract to 180 words
docs: paste existing Section 3.2 content into cancer.tex
```

## LaTeX Structure

The proposal is written in LaTeX with a modular file structure to minimise merge conflicts. **Do not edit `main.tex` directly** unless you are adding a new section import.

```
docs/
└── proposal/
    ├── main.tex                        # Cover page + imports only — do not edit content here
    ├── sections/
    │   ├── abstract.tex
    │   ├── introduction.tex
    │   ├── problem_statement.tex
    │   ├── literature.tex              # Importer only — edit files in literature/ subfolder
    │   ├── literature/
    │   │   ├── xai.tex                 # XAI overview + SOTA comparison (NEW — needs writing)
    │   │   ├── cancer.tex
    │   │   ├── sam.tex
    │   │   ├── ann.tex
    │   │   ├── ea.tex
    │   │   ├── ga.tex
    │   │   ├── nsga.tex
    │   │   ├── auditing.tex
    │   │   ├── sota.tex
    │   │   └── critical_analysis.tex   # Needs new critique paragraphs (see Revision Guide)
    │   ├── methodology.tex             # Importer only — edit files in methodology/ subfolder
    │   ├── methodology/
    │   │   ├── overview.tex            # Research paradigm + tech specs (NEW — needs writing)
    │   │   ├── ann.tex
    │   │   ├── ga.tex
    │   │   ├── nsga.tex
    │   │   ├── sam.tex
    │   │   ├── evaluation.tex          # Needs CART baseline addition (see Revision Guide)
    │   │   ├── experiments.tex         # Needs CCLE stretch objective (see Revision Guide)
    │   │   └── expected_results.tex
    │   └── timeline.tex
    └── references.bib
```

Each section is a separate `.tex` file. **Work only in your assigned file.** This isolates changes and makes conflicts easier to resolve.

New content that needs to be written is marked in **red** (`\textcolor{red}{...}`) inside the relevant files. Once you have written the content, remove the `\textcolor{red}` wrapper.

---

## Section Assignments

Assignments are managed in the **[Proposal Revision Guide](../PROPOSAL_REVISION_GUIDE.md)** — use the task tracker table at the bottom of that document to claim tasks.

---

## Workflow

1. Branch off `dev` — never off `main`
2. Your branch for documentation is `docs/proposal`
3. Make your changes in your assigned `.tex` file only
4. Commit with a clear message (see convention below)
5. Open a pull request into `dev` when your section is ready for review
6. At least one other team member should review before merging

---

## Pending Changes Checklist

- [ ] Fix 3 remaining reference placeholders in `references.bib`
- [ ] Replace Gantt chart placeholder in `timeline.tex`
- [ ] Cut abstract to 100–200 words in `abstract.tex`
- [ ] Write SOTA XAI paragraphs (SHAP, LIME, LRP, IG, Anchors) in `literature/xai.tex`
- [ ] Activate objectives list in `introduction.tex`
- [ ] Activate scope and limitations in `problem_statement.tex`
- [ ] Activate research paradigm statement in `methodology/overview.tex`
- [ ] Fill technical specifications in `methodology/overview.tex`
- [ ] Add paper critique paragraphs to `literature/critical_analysis.tex`
- [ ] Add BioXAI competition paragraph to `literature/critical_analysis.tex`
- [ ] Activate CART decision tree baseline in `methodology/evaluation.tex`
- [ ] Activate CCLE stretch objective in `methodology/experiments.tex`
- [ ] Paste existing content into all scaffolded literature and methodology files
- [ ] Fill patience parameter placeholder in `methodology/ann.tex`
- [ ] Fill run count placeholder in `methodology/ga.tex`
- [ ] First person → third person pass across all `.tex` files
- [ ] Remove flagged phrases (see Revision Guide)

---

