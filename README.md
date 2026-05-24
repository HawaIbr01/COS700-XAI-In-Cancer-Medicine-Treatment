# COS700 — Diagnosing and Repairing XAI Fragility
### SAM-Optimised Rules for Cancer Drug Response

Diagnosing XAI fragility in cancer drug response prediction: a RED XAI pipeline using SAM regularisation and evolutionary rule extraction on genomic data.

This project implements a controlled RED XAI diagnostic pipeline that uses the generalisation gap as an indicator of explanation fragility. DeepCCDS is trained on GDSC data under Adam (baseline) and SAM (intervention), symbolic if-then rules are extracted via GA and NSGA-II, and rule quality is evaluated across fidelity, comprehensibility, and biological validity using PROMETHEE II and GSEA pathway enrichment.

---
## Team

| Name & Surname | Student Number | LinkedIn | Personal Website |
|----------------|---------------|----------|-----------------|
| Angelique Breedt | u23542838 | [LinkedIn](#) | [Website](#) |
| Bahiya Hoosen | u22598546 | [LinkedIn](#) | [Website](#) |
| Hawa Ibrahim | u20418494 | [LinkedIn](#) | [Website](#) |
| Melisa Ledwaba | u19296950 | [LinkedIn](#) | [Website](#) |
| Sakhile Manzi | u26838002 | [LinkedIn](#) | [Website](#) |

---

## Supervisors
- Prof. Nelishia Pillay
- Dr. Nyathi Thambo

**University of Pretoria — COS700 Research Project 2026**

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Branching Convention](#branching-convention)
- [Team](#team)
- [Supervisors](#supervisors)

---

## Repository Structure

```
├── docs/               # Proposal, reports, and project documentation
├── src/
│   ├── data/           # Preprocessing and dataset utilities
│   ├── model/          # DeepCCDS baseline and SAM-regularised training
│   ├── extraction/     # GA and NSGA-II rule extraction (DEAP)
│   └── evaluation/     # PROMETHEE II ranking, Wilcoxon tests, GSEA
├── notebooks/          # Exploratory analysis and visualisations
└── tests/
```

---

## Branching Convention

| Prefix | Purpose | Example |
|--------|---------|---------|
| `main` | Stable, supervisor-presentable state only. Never commit directly. | — |
| `dev` | Integration branch. All branches merge here before `main`. | — |
| `docs/` | Documentation — proposal, reports, README updates | `docs/proposal` |
| `feat/` | New code functionality, one branch per component | `feat/baseline-model` |
| `chore/` | Setup, config, tooling | `chore/project-setup` |
| `fix/` | Bug fixes | `fix/gsea-pathway-query` |

**Rules:**
- Branch off `dev`, not `main`
- Merge back into `dev` via pull request
- `main` is updated only at stable project milestones
- One person, one branch — avoid working on the same branch simultaneously

### Current branches

```
main
dev
docs/proposal
feat/baseline-model          # Angelique
feat/sam-intervention        # Bahiya
feat/ga-rule-extraction      # Melisa
feat/nsga2-rule-extraction   # Sakhile
feat/evaluation-pipeline     # Hawa
chore/project-setup
```
