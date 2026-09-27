# Clever Hans Attribution Audit — Baseline vs. Pre- vs. Post-Investigation SAM

Qualitative sanity check only (§4.2.4, Pahde et al. framework; vanilla input
gradients over the 38 ssGSEA pathway-activity features, `n_samples_per_split
= 256`, same method/JSON schema across all runs — see each audit JSON's own
`interpretation_note`). **This is not a formal Clever Hans test or an
enrichment analysis** — read every claim below descriptively, not as proof.

Three conditions compared:

| Condition | Checkpoint | Gap / Val R² / Val RMSE (context only — see source scorecards, not duplicated here) |
|---|---|---|
| Baseline (Adam, WD=0) | `artifacts/checkpoints/baseline/` | 0.3369 / 0.8565 / 1.0299 — `artifacts/sam_investigation/clever_hans/baseline/baseline_gap_scorecard.json` |
| Pre-investigation SAM sweep, ρ=0.2 (the config actually deployed before this investigation; ρ=0.01/0.05/0.1 also audited, shown where relevant) | `artifacts/checkpoints/sam/` | 0.3122 / 0.8602 / 1.0165 — `artifacts/sam_investigation/clever_hans/pre_investigation_sam_sweep/sam_gap_scorecard_rho_0p2.json` |
| Post-investigation final SAM, ρ=0.5 + WD=1e-3 (Stage 4c confirmatory rerun — the recommended config) | `artifacts/sam_investigation/confirmatory/` | 0.0485 / 0.8444 / 1.0722 — `artifacts/sam_investigation/confirmatory/confirmatory_scorecard.json` |

---

## 1. Top-10 pathways and coefficient of variation across all three

**Top-10 by mean |gradient|, validation split:**

| Baseline | SAM ρ=0.2 (pre-inv.) | SAM ρ=0.5+WD (final) |
|---|---|---|
| Estrogen signaling | Phospholipase D signaling | Focal adhesion |
| Focal adhesion | Estrogen signaling | B cell receptor signaling |
| Natural killer cell mediated cytotoxicity | Apelin signaling | Estrogen signaling |
| Cholinergic synapse | Insulin secretion | Regulation of actin cytoskeleton |
| cAMP signaling | PI3K-Akt signaling | Cholinergic synapse |
| Apelin signaling | Cholinergic synapse | Th17 cell differentiation |
| Phospholipase D signaling | ErbB signaling | Phospholipase D signaling |
| Renin secretion | MAPK signaling | ErbB signaling |
| ErbB signaling | Thyroid hormone signaling | Natural killer cell mediated cytotoxicity |
| Wnt signaling | Calcium signaling | Chemokine signaling |

**Pathways shared across all three top-10s:** Focal adhesion, Estrogen
signaling, Cholinergic synapse, Phospholipase D signaling, ErbB signaling,
Natural killer cell mediated cytotoxicity (6/10) — the "core" attribution
signature is largely stable across the whole arc.

**Dropped out by the final config** (present in baseline's top-10, absent
from the final SAM top-10): Apelin signaling, cAMP signaling, Renin
secretion, Wnt signaling.

**Entered in the final config** (absent from baseline's top-10, present in
the final SAM top-10): B cell receptor signaling, Regulation of actin
cytoskeleton, Th17 cell differentiation, Chemokine signaling.

**`mean_coefficient_of_variation`** (train / val, all from the relevant audit
JSON):

| Condition | Train CV | Val CV |
|---|---|---|
| Baseline | 0.792 | 0.783 |
| SAM ρ=0.01 | 0.801 | 0.789 |
| SAM ρ=0.05 | 0.804 | 0.801 |
| SAM ρ=0.1 | 0.847 | 0.824 |
| SAM ρ=0.2 (pre-inv.) | 0.872 | 0.884 |
| **SAM ρ=0.5+WD (final)** | **0.840** | **0.850** |

CV rises monotonically through the pre-investigation sweep (0.79→0.88 val),
then comes back down slightly at the final ρ=0.5+WD config (0.85) — but it
does **not** return to the baseline's 0.78. By this specific measure, the
final config's attribution is still less internally-consistent
sample-to-sample than the unregularised baseline, just less erratic than the
ρ=0.2 sweep point immediately before it.

---

## 2. Does the final SAM config show more evenly-spread attribution than baseline? — **mixed, not clean**

Two additional scale-invariant concentration measures were computed from each
audit's `per_pathway_mean_abs_gradient` (share of total |gradient| mass held
by the top-3 / top-10 pathways out of all 38):

| Condition | Top-3 share (val) | Top-10 share (val) | Total |gradient| mass (val) |
|---|---|---|---|
| Baseline | 11.7% | 35.1% | 28.5 |
| SAM ρ=0.2 (pre-inv.) | 12.0% | 32.5% | 22.6 |
| SAM ρ=0.5+WD (final) | **19.0%** | **43.4%** | 7.2 |

**The evidence here is mixed, not a clean "SAM balances attribution" story:**

- The **total magnitude** of attribution shrinks sharply (~4× smaller by the
  final config) — expected, since WD=1e-3 shrinks weight/gradient scale
  generally, independent of any "balance" question.
- But looking at **shape**, not scale: the final config's top pathways
  actually hold a **larger** share of the total attribution mass than the
  baseline's do (top-10 share rises from 35% to 43%, top-3 from 12% to 19%),
  and its `mean_coefficient_of_variation` (0.84–0.85) is **higher** than the
  baseline's (0.78–0.79), not lower.
- The train/val gap in both CV (final: +0.010; baseline: −0.009; ρ=0.2:
  +0.012) and top-3 concentration (final: +0.005; baseline: −0.007; ρ=0.2:
  −0.002) is small and similar in magnitude across all three conditions —
  none of them shows a materially tighter train/val attribution match than
  the others by this measure.

**Plain read:** on the metrics this audit actually computes, the final
SAM ρ=0.5+WD config does **not** show more evenly-spread or more
train/val-consistent attribution than the Adam baseline — if anything,
attribution is somewhat *more* concentrated in fewer pathways and slightly
*more* sample-to-sample variable. The investigation's headline result (an
84.6% reduction in the numeric generalisation gap, `journal.md` Stage 4) is
real and separately verified, but this particular qualitative attribution
check does not independently corroborate a "SAM produces more balanced
representations" (Springer et al.) story — it neither confirms nor
contradicts it cleanly, and the concentration numbers point the other way.
This should not be rounded up into a cleaner narrative than the numbers
support.

---

## 3. Biological plausibility — plain-language read only

Across all three conditions, the top-attributed pathways remain a recognisable
mix of canonical cancer-relevant signaling: **ErbB signaling** (classic
oncogenic receptor tyrosine kinase pathway, therapeutic target in multiple
cancers) and **PI3K-Akt signaling** (canonical oncogenic survival pathway,
frequently dysregulated in cancer) appear prominently in the baseline and
pre-investigation sweep; **Focal adhesion** and **Regulation of actin
cytoskeleton** (cell migration/invasion, relevant to metastatic potential)
and **ErbB signaling** remain prominent in the final config. Estrogen
signaling (breast-cancer-relevant) and Phospholipase D signaling (implicated
in oncogenic transformation and drug resistance) appear consistently across
all three. The pathways that enter the final config's top-10 but weren't
there in the baseline — B cell receptor signaling, Th17 cell differentiation,
chemokine signaling, NK-cell-mediated cytotoxicity — lean more toward
immune/tumour-microenvironment biology than classic cell-intrinsic oncogenic
signaling; this is not implausible for a drug-response panel that includes
immunomodulatory compounds, but it is a qualitatively different flavour of
"plausible" than ErbB/PI3K-Akt, and is worth a one-line caveat in the write-up
rather than being folded into "same story as baseline" without comment.
Overall: the final config's top pathways remain **biologically plausible**
for cancer drug response, in the same loose sense the baseline's did — this
is a plain-language read, not a formal enrichment test (that is what the
later GSEA stage is for).

---

## Sources

- `artifacts/sam_investigation/clever_hans/baseline/baseline_clever_hans_audit.json`
- `artifacts/sam_investigation/clever_hans/pre_investigation_sam_sweep/sam_clever_hans_audit_rho_0p{01,05,1,2}.json`
- `artifacts/sam_investigation/clever_hans/post_investigation_sam_final/sam_clever_hans_audit_rho_0p5_wd.json`
- Gap/R²/RMSE context: `artifacts/sam_investigation/clever_hans/baseline/baseline_gap_scorecard.json`,
  `artifacts/sam_investigation/clever_hans/pre_investigation_sam_sweep/sam_gap_scorecard_rho_0p2.json`,
  `artifacts/sam_investigation/confirmatory/confirmatory_scorecard.json`
