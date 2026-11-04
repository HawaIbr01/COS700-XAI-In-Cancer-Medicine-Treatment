# Per-drug pilot: RESULTS (numbers only)

Drugs: 379; cells: 11370 (expected 11370); seeds per cell: [10].

## Drug-level means by arm and extractor (mean over drugs)

| extractor | arm | heldout_fidelity | phi | length | stability_jaccard_folds | heldout_fidelity_net_train | heldout_fidelity_net_other |
|---|---|---|---|---|---|---|---|
| ga | adam | 0.5116 | -0.0106 | 1.7280 | 0.0992 | 0.5324 | 0.7376 |
| ga | adam_wd | 0.5379 | 0.0161 | 1.7449 | 0.0985 | 0.5612 | 0.7584 |
| ga | sam | 0.5385 | 0.0177 | 1.7330 | 0.0990 | 0.5603 | 0.7580 |
| nsga | adam | 0.4447 | -0.0213 | 1.9467 | 0.3361 | 0.4568 | 0.6354 |
| nsga | adam_wd | 0.4560 | 0.0126 | 1.9530 | 0.3481 | 0.4686 | 0.6331 |
| nsga | sam | 0.4648 | 0.0375 | 2.0383 | 0.3707 | 0.4753 | 0.6569 |

## Counts

| extractor | arm | empty pools (folds) | rules covering <10 held-out rows | rules with NaN held-out fidelity | empty fold gene sets |
|---|---|---|---|---|---|
| ga | adam | 0 | 4168 | 0 | 0 |
| ga | adam_wd | 0 | 4229 | 5 | 0 |
| ga | sam | 0 | 4154 | 2 | 0 |
| nsga | adam | 0 | 968 | 0 | 0 |
| nsga | adam_wd | 0 | 918 | 0 | 0 |
| nsga | sam | 0 | 857 | 0 | 0 |

## Planned comparisons (paired by drug)

| metric | extractor | comparison | n | median diff | mean diff | zero diffs | Wilcoxon p | BH p |
|---|---|---|---|---|---|---|---|---|
| heldout_fidelity | ga | SAM_minus_AdamWD | 379 | -0.0008 | 0.0006 | 1 | 0.8902 | 0.8902 |
| heldout_fidelity | ga | AdamWD_minus_Adam | 379 | 0.0066 | 0.0263 | 1 | 0.0001 | 0.0002 |
| heldout_fidelity | ga | sam_vs_adam (proposal contrast, confounded) | 379 | 0.0101 | 0.0269 | 1 | 0.0000 | 0.0000 |
| heldout_fidelity | nsga | SAM_minus_AdamWD | 379 | 0.0057 | 0.0089 | 0 | 0.0986 | 0.1183 |
| heldout_fidelity | nsga | AdamWD_minus_Adam | 379 | 0.0076 | 0.0113 | 0 | 0.0086 | 0.0129 |
| heldout_fidelity | nsga | sam_vs_adam (proposal contrast, confounded) | 379 | 0.0141 | 0.0201 | 0 | 0.0000 | 0.0000 |
| phi | ga | SAM_minus_AdamWD | 379 | 0.0061 | 0.0017 | 0 | 0.5013 | 0.5013 |
| phi | ga | AdamWD_minus_Adam | 379 | 0.0144 | 0.0266 | 0 | 0.0001 | 0.0003 |
| phi | ga | sam_vs_adam (proposal contrast, confounded) | 379 | 0.0280 | 0.0283 | 0 | 0.0000 | 0.0000 |
| phi | nsga | SAM_minus_AdamWD | 379 | 0.0237 | 0.0249 | 0 | 0.0493 | 0.0591 |
| phi | nsga | AdamWD_minus_Adam | 379 | 0.0333 | 0.0339 | 0 | 0.0023 | 0.0034 |
| phi | nsga | sam_vs_adam (proposal contrast, confounded) | 379 | 0.0763 | 0.0588 | 0 | 0.0000 | 0.0000 |
| length | ga | SAM_minus_AdamWD | 379 | 0.0000 | -0.0119 | 61 | 0.6143 | 0.9214 |
| length | ga | AdamWD_minus_Adam | 379 | 0.0000 | 0.0169 | 47 | 0.2750 | 0.5500 |
| length | ga | sam_vs_adam (proposal contrast, confounded) | 379 | 0.0000 | 0.0050 | 52 | 0.7680 | 0.9216 |
| length | nsga | SAM_minus_AdamWD | 379 | 0.1000 | 0.0852 | 47 | 0.0000 | 0.0000 |
| length | nsga | AdamWD_minus_Adam | 379 | 0.0000 | 0.0063 | 50 | 0.9359 | 0.9359 |
| length | nsga | sam_vs_adam (proposal contrast, confounded) | 379 | 0.1000 | 0.0916 | 55 | 0.0000 | 0.0000 |
| stability_jaccard_folds | ga | SAM_minus_AdamWD | 379 | 0.0007 | 0.0005 | 0 | 0.8521 | 0.9250 |
| stability_jaccard_folds | ga | AdamWD_minus_Adam | 379 | -0.0006 | -0.0006 | 0 | 0.6274 | 0.9250 |
| stability_jaccard_folds | ga | sam_vs_adam (proposal contrast, confounded) | 379 | -0.0006 | -0.0001 | 0 | 0.9250 | 0.9250 |
| stability_jaccard_folds | nsga | SAM_minus_AdamWD | 379 | 0.0200 | 0.0227 | 0 | 0.0973 | 0.2919 |
| stability_jaccard_folds | nsga | AdamWD_minus_Adam | 379 | 0.0045 | 0.0120 | 1 | 0.4565 | 0.9129 |
| stability_jaccard_folds | nsga | sam_vs_adam (proposal contrast, confounded) | 379 | 0.0250 | 0.0346 | 3 | 0.0071 | 0.0427 |

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
| ga | adam | 67121 | 35.4 |
| ga | adam_wd | 66988 | 35.4 |
| ga | sam | 67074 | 35.4 |
| nsga | adam | 7472 | 3.9 |
| nsga | adam_wd | 7463 | 3.9 |
| nsga | sam | 7451 | 3.9 |
