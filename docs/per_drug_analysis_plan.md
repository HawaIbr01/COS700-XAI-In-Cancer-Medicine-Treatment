# Per-drug rule-extraction pilot (Stage 5b) — frozen analysis plan

Written before any extraction was run. Design decisions below were made by Hawa; items marked
**[OPEN]** are interpretations the implementer had to make and that still need Hawa's confirmation
(none of them can be settled by looking at rule quality). The experiment closes on the day it is run:
no new experiments are chosen after seeing results. Bug fixes and re-runs are allowed and are logged in
`artifacts/per_drug/RERUN_LOG.md`.

## 1. Design (frozen)

| Item | Value |
|---|---|
| Arms | (1) Adam baseline `artifacts/checkpoints/baseline`; (2) Adam + weight decay `artifacts/sam_investigation/adam_wd_control`; (3) SAM (+ weight decay, plateau stopping rule) `artifacts/sam_investigation/confirmatory`. Arms 1 and 3 use the existing class_pred files; arm 2's are `artifacts/per_drug/data/predictions/adam_wd_*.parquet` (same recipe; recipe re-verified to reproduce arms 1 and 3 exactly). |
| Extractors | GA (`src/extraction/ga.py`): pop 35–45 (per-seed draw), 50 generations, cx 0.8, mut 0.2, aging 8, fitness 0.45 fidelity + 0.35 accuracy + 0.20 comprehensibility. NSGA-II (`src/extraction/nsga.py`): pop 40, 50 generations, cx 0.9, mut 0.2, fidelity vs length, no accuracy term. Same sparse initialisation, operators and aggregation as the global runs. |
| Unit | One drug. All of the drug's rows (train + val + test of the networks, ≈ 817), one row per cell line. The network's original split is kept (`net_split`). |
| Cross-validation | 5 folds **by cell line** within the drug; fold seed 42 (balanced seeded permutation). For each fold: extract on the 4 training folds, score on the held-out fold. |
| Inner split | Inside the 4 training folds, 20 % of cell lines (seeded: 4242 + fold) form the validation part used **only** by the GA accuracy term; fidelity uses the rest. NSGA-II ignores it. |
| Coverage floor | Fixed count `min_cover_n = 25` cell lines, applied to the fidelity part (replaces the 2 % fraction in both extractors and in the GA pruning floor). Implemented as `25 / n_rows`, which is exact (tested). |
| Seeds | 30 per fold by default (run seed = 42 + run_id, as in the global runs). The final number is set after Stage 1 timing and then frozen. |
| Held-out scoring | For every rule in a pool: fidelity = macro-F1 of the rule's fixed majority class vs the network's class_pred (same arm) on the held-out rows the rule covers; coverage count; length. Also fidelity and counts separately for held-out rows that were in the network's TRAIN split vs its val/test splits. Rules covering < 10 held-out rows are flagged, not dropped. Rules covering 0 held-out rows have undefined fidelity (NaN) and are excluded from medians; their number is reported. |
| Pool | The pooled non-dominated rules over the seeds of one (drug, arm, extractor, fold), exactly as `aggregate_runs` builds it in the global runs. |

Every run writes a manifest line (git commit, design hash, arm, extractor, drug, fold, seed, seconds) to
`artifacts/per_drug/<stage>/manifest.jsonl`; each (drug, arm, extractor, fold) cell is its own checkpoint file.

## 2. Outcomes

**Primary** (per drug × arm × extractor): median held-out fidelity across the pool's rules, averaged over the 5 folds
(folds with an empty pool are left out of the average and counted).

**Secondary**
- PROMETHEE II net flow φ (mean over a pool's rules; per-arm mean over folds).
- Median rule length.
- Stability of the gene set. Fold gene set = all genes in any rule of that fold's pool. *Primary:* mean pairwise Jaccard
  over the 10 fold pairs. *Secondary:* Kuncheva index (only for pairs of equal-size sets; n = 572 features) and mean pairwise
  Jaccard across seeds (per-seed gene sets, within a fold, averaged over folds). Empty sets give NaN; pairs involving them are
  left out of the mean and their number is reported.
- Positive-control hit (see §5): descriptive only (4 drugs), no test.

## 3. Planned comparisons

1. **SAM effect** = SAM − Adam + weight decay.
2. **Weight-decay effect** = Adam + weight decay − Adam.

Each is computed per extractor and per metric on the per-drug values (paired by drug; n = number of pilot drugs).
- Paired Wilcoxon signed-rank test across drugs (two-sided, exact where possible, zero differences dropped by the default rule), one test per metric × extractor × comparison.
- Benjamini–Hochberg within each metric family (the family = all tests of that metric: 2 comparisons × 2 extractors).
- Bootstrap 95 % CI of the median paired difference: 10 000 resamples of drugs (percentile interval, fixed seed).
- TOST equivalence on held-out fidelity, margin ±0.05, **exploratory pending supervisor sign-off**. Implemented as two one-sided
  Wilcoxon signed-rank tests on the paired differences (H0: d ≤ −0.05, and H0: d ≥ +0.05); p_TOST = the larger of the two p-values.
  Confirmed by Hawa (2026-10-10): one-sided Wilcoxon tests, exploratory.

## 4. PROMETHEE II

Existing implementation (`notebooks/03_promethee_ranking.ipynb`: `pymcdm.PROMETHEE_II`, V-shape preference, p = observed range, q = 0)
and the existing criteria and weights (provisional, not signed off by supervisors):

| Criterion | Direction | Weight |
|---|---|---|
| held-out fidelity (`fidelity_test` in the notebook) | maximise | 0.5 / 0.7 = **0.714** |
| rule length | minimise | 0.2 / 0.7 = **0.286** |
| ~~stability (variance across cell lines of per-cell-line agreement, ≥ 10 covered rows per cell line)~~ | dropped | (was 0.3) |

**Decisions (Hawa, 2026-10-10)**
1. **Stability criterion dropped, other two weights renormalised to 0.714 / 0.286.** Reason: it is undefined for single-drug data (a drug has at most
   one row per cell line, so no cell line reaches ≥ 10 covered rows and every rule would get NaN).
2. **One PROMETHEE II run per fold, per drug and extractor**, over the rules of the three arms' pools for that fold, with fidelity measured on that
   fold's held-out rows (`fidelity_test` = held-out fidelity of the fold). φ is only comparable inside one run; each arm's φ is its mean over the
   fold's rules, then averaged over the 5 folds.
3. **TOST-style one-sided Wilcoxon tests**, exploratory (see §3).

Rules with undefined held-out fidelity cannot enter PROMETHEE and are excluded from it (and counted).

## 5. Positive controls

Erlotinib → EGFR_mut; PLX-4720 → BRAF_mut; Olaparib → BRCA1_mut or BRCA2_mut; Selumetinib → BRAF_mut or KRAS_mut.
Hit = the gene appears with condition **High** in any rule of the pool (per fold; summarised as folds hit out of 5). Also logged: whether it appears in the
top PROMETHEE-ranked rule. Caveat to carry into the write-up: the `_mut` flags are very common in this data set (BRAF_mut High in 38 % of cell lines,
EGFR_mut 44 %), so `High` almost certainly means "any recorded variant in the gene", not "activating mutation" (see `artifacts/per_drug/STAGE0_REPORT.md` §D).

## 6. Pilot drugs

Chosen by `scripts/select_pilot_drugs.py` before any extraction and stored in `artifacts/per_drug/pilot_drugs.json`:
best-predicted 5 (highest per-drug validation macro-F1 of class_pred vs the true class, mean over the 3 arms, ties alphabetical), the 4 positive controls,
and 5 drawn with seed 700 from the remaining candidates (379 candidates). A drug in two groups counts once. Note: the best-predicted group is selected on
network-val rows that are also part of the held-out folds; this selection effect is a known limitation.

## 7. Over-representation analysis (Stage 2 item 4)

Pooled KEGG over-representation (`gseapy.enrichr`, library `KEGG_2021_Human`, background = the 534 mutation-feature genes, BH-corrected) per arm,
using genes from `High` mutation conditions in all rules across drugs. Named "over-representation analysis", not GSEA. Needs network access to Enrichr; if it
is not reachable the gene lists are saved and the failure is reported, nothing is substituted.

## 9. Decisions added 2026-10-10 (Hawa)

**Positive-control interpretation.** For features flagged in more than 30 % of cell lines (BRAF_mut, EGFR_mut), a hit counts as evidence and a miss is
inconclusive, because the flag looks like "any recorded variant" rather than an activating mutation.

**Arm definitions and limitation.** The "SAM" arm is SAM + weight decay (rho 0.5, WD 1e-3). The two weight-decay arms (Adam + WD, SAM + WD) reconstruct
their inputs about 10x worse than the Adam baseline (validation reconstruction MSE: mutation 0.1445 / 0.1440 vs 0.0124; drug 0.1087 / 0.1096 vs 0.0012).
This is a limitation: the arms differ in the autoencoder branches as well as in the optimiser.

**Stage 1 timing and seed choice.** Erlotinib, 1 worker, 90 seed runs, 0 errors: GA 7.77 s/seed, NSGA-II 0.95 s/seed, per-cell overhead about 1.5 s / 0.35 s,
peak RSS 735 MB per process. Per drug about 130.8 x seeds + 28 s on one core. The pilot uses **10 seeds and 3 workers** (estimate 14 drugs x 10 seeds: 5.2 h single
core, about 1.7 h with 3 workers; scaling assumed linear). Details: `artifacts/per_drug/STAGE1_REPORT.md`.

## 10. POST-HOC DIAGNOSTICS (added 2026-10-11, after seeing RESULTS.md)

Everything in this section was decided **after** `artifacts/per_drug/RESULTS.md` had been produced. It is not part of the frozen design, does not
change any primary number (RESULTS.md and the primary CSVs are unchanged), re-runs no extraction, and is stored in `artifacts/per_drug/diagnostics/`
(report: `POST_HOC_DIAGNOSTICS.md`; code: `src/per_drug/diagnostics.py`; files prefixed `posthoc_`). Added, and why:

1. **Fidelity-metric behaviour** (share of evaluations with fidelity exactly 1.0, share with a single network class among covered rows, deciles, net_train vs net_other):
   because the macro-F1 call has no `labels` argument, so its value depends on whether the covered rows contain one class or two.
2. **Accuracy and coverage as a sensitivity analysis** (same paired comparisons as §3): a metric without the single-class artefact.
3. **Matched random-rule chance baseline** (200 random rules per real rule, same number of conditions, >= 25 covered extraction rows): to place the observed fidelity against chance.
4. **Stability definitions, set sizes and chance Jaccard** for those sizes out of 572 features.
5. **Rule-length units** (number of conditions) and a BH correction across all planned comparisons (Kuncheva rows with n < 5 excluded), next to the within-metric BH.
6. **Timing estimates for 379 drugs** (3 workers; 10, 5, 3 seeds; GA, NSGA-II and NSGA-II only) from the pilot manifest.
7. **Positive-control feasibility**: the single-condition rule `feature = High` on all rows of each control drug, against each network's class_pred and the true class, compared with random single-condition rules.
   Purpose: separate "features and network do not carry the signal" from "the extractor missed it".
8. **Original global rule pools (19 rules), POST-HOC** (added 2026-10-11, after the pilot results): per-rule agreement with the network's class_pred and coverage on
   train/val/test, per-pool share of test rows covered by >= 1 and by exactly 1 rule, and a matched random-rule baseline (200 draws per rule, same length, 2% train coverage floor).
   Why: to put the pilot's agreement and fidelity numbers next to the global pools under the same functions. Files: `posthoc_global_*.csv`, `POST_HOC_DIAGNOSTICS_global_pools.md`; code `src/per_drug/diagnostics_global.py`.

## 8. What will not be done

No new experiments, parameters or drugs chosen after seeing results; no parameter set from rule quality; Stage 1 (Erlotinib, 3 seeds) is used for timing and
correctness only.
