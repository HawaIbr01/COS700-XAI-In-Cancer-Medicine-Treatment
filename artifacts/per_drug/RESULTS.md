# Per-drug pilot: RESULTS (numbers only)

Drugs: 14; cells: 420 (expected 420); seeds per cell: [10].

## Drug-level means by arm and extractor (mean over drugs)

| extractor | arm | heldout_fidelity | phi | length | stability_jaccard_folds | stability_kuncheva_folds | stability_jaccard_seeds | heldout_fidelity_net_train | heldout_fidelity_net_other |
|---|---|---|---|---|---|---|---|---|---|
| ga | adam | 0.4647 | -0.0367 | 1.7714 | 0.1101 | 0.1818 | 0.0315 | 0.4917 | 0.7292 |
| ga | adam_wd | 0.5154 | 0.0113 | 1.8571 | 0.1016 | 0.1113 | 0.0331 | 0.5160 | 0.7446 |
| ga | sam | 0.5345 | 0.0446 | 1.5929 | 0.1125 | 0.1947 | 0.0346 | 0.5393 | 0.7757 |
| nsga | adam | 0.4381 | -0.0010 | 1.9286 | 0.3800 | 0.5743 | 0.0041 | 0.4570 | 0.6676 |
| nsga | adam_wd | 0.4510 | 0.0108 | 2.0286 | 0.3261 | 0.5152 | 0.0046 | 0.4823 | 0.6049 |
| nsga | sam | 0.4763 | 0.0338 | 1.8714 | 0.3585 | 0.5490 | 0.0053 | 0.4922 | 0.6714 |

## Counts

| extractor | arm | empty pools (folds) | rules covering <10 held-out rows | rules with NaN held-out fidelity | empty fold gene sets |
|---|---|---|---|---|---|
| ga | adam | 0 | 157 | 0 | 0 |
| ga | adam_wd | 0 | 185 | 0 | 0 |
| ga | sam | 0 | 138 | 0 | 0 |
| nsga | adam | 0 | 36 | 0 | 0 |
| nsga | adam_wd | 0 | 31 | 0 | 0 |
| nsga | sam | 0 | 29 | 0 | 0 |

## Planned comparisons (paired by drug)

| metric | extractor | comparison | n | median diff | 95% CI | Wilcoxon p | BH p | TOST p (±0.05) |
|---|---|---|---|---|---|---|---|---|
| heldout_fidelity | ga | SAM_minus_AdamWD | 14 | 0.0105 | [-0.0430, 0.0752] | 0.5830 | 0.7773 | 0.1338 |
| heldout_fidelity | ga | AdamWD_minus_Adam | 14 | 0.0060 | [-0.0178, 0.0605] | 0.2958 | 0.7773 | 0.2131 |
| heldout_fidelity | nsga | SAM_minus_AdamWD | 14 | 0.0038 | [-0.0549, 0.0604] | 0.9515 | 0.9515 | 0.0676 |
| heldout_fidelity | nsga | AdamWD_minus_Adam | 14 | -0.0043 | [-0.0366, 0.0740] | 0.4631 | 0.7773 | 0.0247 |
| phi | ga | SAM_minus_AdamWD | 14 | 0.0191 | [-0.0355, 0.1117] | 0.5016 | 1.0000 | NaN |
| phi | ga | AdamWD_minus_Adam | 14 | 0.0262 | [-0.0348, 0.1205] | 0.1937 | 0.7749 | NaN |
| phi | nsga | SAM_minus_AdamWD | 14 | -0.0101 | [-0.1553, 0.1851] | 1.0000 | 1.0000 | NaN |
| phi | nsga | AdamWD_minus_Adam | 14 | -0.0111 | [-0.1570, 0.2074] | 0.9515 | 1.0000 | NaN |
| length | ga | SAM_minus_AdamWD | 14 | -0.1500 | [-0.4000, -0.1000] | 0.0067 | 0.0268 | NaN |
| length | ga | AdamWD_minus_Adam | 14 | 0.0500 | [-0.0500, 0.2000] | 0.2116 | 0.2116 | NaN |
| length | nsga | SAM_minus_AdamWD | 14 | -0.2000 | [-0.3000, -0.1000] | 0.0168 | 0.0336 | NaN |
| length | nsga | AdamWD_minus_Adam | 14 | 0.1500 | [-0.1000, 0.3000] | 0.1613 | 0.2116 | NaN |
| stability_jaccard_folds | ga | SAM_minus_AdamWD | 14 | 0.0210 | [0.0044, 0.0374] | 0.0906 | 0.3623 | NaN |
| stability_jaccard_folds | ga | AdamWD_minus_Adam | 14 | 0.0014 | [-0.0387, 0.0197] | 0.8552 | 0.8552 | NaN |
| stability_jaccard_folds | nsga | SAM_minus_AdamWD | 14 | 0.0680 | [-0.0444, 0.1247] | 0.3910 | 0.5213 | NaN |
| stability_jaccard_folds | nsga | AdamWD_minus_Adam | 14 | -0.0602 | [-0.1964, 0.0763] | 0.3258 | 0.5213 | NaN |
| stability_kuncheva_folds | ga | SAM_minus_AdamWD | 1 | 0.1440 | [0.1440, 0.1440] | 1.0000 | 1.0000 | NaN |
| stability_kuncheva_folds | ga | AdamWD_minus_Adam | 2 | -0.1199 | [-0.1297, -0.1102] | 0.5000 | 1.0000 | NaN |
| stability_kuncheva_folds | nsga | SAM_minus_AdamWD | 14 | 0.0842 | [-0.1738, 0.2893] | 0.5525 | 1.0000 | NaN |
| stability_kuncheva_folds | nsga | AdamWD_minus_Adam | 13 | -0.0018 | [-0.2234, 0.1680] | 0.7869 | 1.0000 | NaN |
| stability_jaccard_seeds | ga | SAM_minus_AdamWD | 14 | -0.0009 | [-0.0057, 0.0052] | 1.0000 | 1.0000 | NaN |
| stability_jaccard_seeds | ga | AdamWD_minus_Adam | 14 | 0.0023 | [-0.0014, 0.0054] | 0.2412 | 0.7773 | NaN |
| stability_jaccard_seeds | nsga | SAM_minus_AdamWD | 14 | 0.0007 | [-0.0015, 0.0025] | 0.5830 | 0.7773 | NaN |
| stability_jaccard_seeds | nsga | AdamWD_minus_Adam | 14 | 0.0005 | [-0.0015, 0.0026] | 0.5830 | 0.7773 | NaN |

## Positive controls (folds with a hit, out of 5)

| drug | features | extractor | arm | folds with hit in pool | folds where top-phi rule (3-arm union) contains the feature (High) |
|---|---|---|---|---|---|
| Erlotinib | EGFR_mut | ga | adam | 0/5 | 0/5 |
| Erlotinib | EGFR_mut | ga | adam_wd | 0/5 | 0/5 |
| Erlotinib | EGFR_mut | ga | sam | 0/5 | 0/5 |
| Erlotinib | EGFR_mut | nsga | adam | 0/5 | 0/5 |
| Erlotinib | EGFR_mut | nsga | adam_wd | 0/5 | 0/5 |
| Erlotinib | EGFR_mut | nsga | sam | 0/5 | 0/5 |
| Olaparib | BRCA1_mut,BRCA2_mut | ga | adam | 0/5 | 0/5 |
| Olaparib | BRCA1_mut,BRCA2_mut | ga | adam_wd | 0/5 | 0/5 |
| Olaparib | BRCA1_mut,BRCA2_mut | ga | sam | 1/5 | 0/5 |
| Olaparib | BRCA1_mut,BRCA2_mut | nsga | adam | 0/5 | 0/5 |
| Olaparib | BRCA1_mut,BRCA2_mut | nsga | adam_wd | 0/5 | 0/5 |
| Olaparib | BRCA1_mut,BRCA2_mut | nsga | sam | 0/5 | 0/5 |
| PLX-4720 | BRAF_mut | ga | adam | 0/5 | 0/5 |
| PLX-4720 | BRAF_mut | ga | adam_wd | 0/5 | 0/5 |
| PLX-4720 | BRAF_mut | ga | sam | 0/5 | 0/5 |
| PLX-4720 | BRAF_mut | nsga | adam | 0/5 | 0/5 |
| PLX-4720 | BRAF_mut | nsga | adam_wd | 0/5 | 0/5 |
| PLX-4720 | BRAF_mut | nsga | sam | 0/5 | 0/5 |
| Selumetinib | BRAF_mut,KRAS_mut | ga | adam | 0/5 | 0/5 |
| Selumetinib | BRAF_mut,KRAS_mut | ga | adam_wd | 0/5 | 0/5 |
| Selumetinib | BRAF_mut,KRAS_mut | ga | sam | 0/5 | 0/5 |
| Selumetinib | BRAF_mut,KRAS_mut | nsga | adam | 0/5 | 0/5 |
| Selumetinib | BRAF_mut,KRAS_mut | nsga | adam_wd | 0/5 | 0/5 |
| Selumetinib | BRAF_mut,KRAS_mut | nsga | sam | 0/5 | 0/5 |

## Wall-clock (seconds, summed over cells)

| extractor | arm | total s | mean s per cell |
|---|---|---|---|
| ga | adam | 10450 | 149.3 |
| ga | adam_wd | 10281 | 146.9 |
| ga | sam | 10497 | 150.0 |
| nsga | adam | 1219 | 17.4 |
| nsga | adam_wd | 1230 | 17.6 |
| nsga | sam | 1233 | 17.6 |

## Pooled KEGG over-representation (High mutation genes per arm; background 534 genes; BH)

| arm | n genes |
|---|---|
| adam | 175 |
| adam_wd | 183 |
| sam | 160 |

Terms with adjusted p < 0.05 per arm:

- adam: 0 of 220 terms
- adam_wd: 0 of 219 terms
- sam: 0 of 205 terms

## Definitions and counts used (no interpretation)

- Source: `artifacts/per_drug/pilot/` (420 cells = 14 drugs x 3 arms x 2 extractors x 5 folds; 10 seeds, 3 workers, design hash 8036bd516e6d; manifest of 4200 seed runs in `pilot/manifest.jsonl`). No cell was re-run (see `RERUN_LOG.md`).
- Per-drug value = mean over the 5 folds of the fold's median held-out fidelity (all folds were defined; 0 empty pools).
- PROMETHEE II: one run per fold, drug and extractor over the three arms' rules; criteria held-out fidelity (max, weight 5/7) and length (min, weight 2/7); stability criterion dropped.
- Fold gene set = feature names of all rules in the pool (this includes pathway-activity feature names as well as mutation genes). Kuncheva is defined only for equal-size pairs, hence the small n in some rows.
- TOST p = max of two one-sided Wilcoxon p-values (margin 0.05), exploratory. BH is applied within each metric (4 tests); the phi, length and stability metrics are compared with Wilcoxon only.
- KEGG: `gseapy.enrichr`, KEGG_2021_Human, background 534 mutation genes, genes with High in any rule per arm over all drugs/extractors/folds; Enrichr was reachable.
- Per-fold and per-drug tables: `results_tidy.csv`, `results_drug_level.csv`; controls: `positive_controls.csv`; timings: `wallclock.csv`.
