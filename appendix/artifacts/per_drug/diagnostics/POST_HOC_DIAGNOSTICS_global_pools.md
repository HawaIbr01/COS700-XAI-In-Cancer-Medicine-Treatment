# POST-HOC DIAGNOSTICS: original global rule pools (19 rules)

Decided after seeing the pilot results; not part of the frozen design. Read-only on `artifacts/rules/{ga,nsga}_{baseline,sam}.json` and `data/processed/real_extraction/`. Code: `src/per_drug/diagnostics_global.py`. Numbers only.

Agreement = share of the rule's covered rows whose network class_pred (Adam baseline for the `*_baseline` pools, SAM for the `*_sam` pools, from `real_extraction/predictions/`) equals the rule class; coverage = covered rows / rows of the split (train 245,845; val 30,720; test 30,728). Stored train coverage and train fidelity of every rule were reproduced to within 6e-5 (assertion in the script).

## 1. Per-rule agreement and coverage

| id                 |   length | rule_class   |   coverage_train |   coverage_val |   coverage_test |   agreement_train |   agreement_val |   agreement_test |
|:-------------------|---------:|:-------------|-----------------:|---------------:|----------------:|------------------:|----------------:|-----------------:|
| ga_baseline_0001   |        3 | sensitive    |           0.0221 |         0.0208 |          0.0216 |            0.9233 |          0.9343 |           0.9155 |
| ga_baseline_0002   |        2 | sensitive    |           0.0245 |         0.023  |          0.0242 |            0.9206 |          0.9178 |           0.9141 |
| ga_baseline_0003   |        2 | sensitive    |           0.024  |         0.024  |          0.0248 |            0.9185 |          0.9322 |           0.9187 |
| ga_baseline_0004   |        4 | sensitive    |           0.029  |         0.0283 |          0.0299 |            0.9134 |          0.8998 |           0.9217 |
| ga_baseline_0005   |        3 | sensitive    |           0.0289 |         0.0271 |          0.03   |            0.9088 |          0.9147 |           0.9066 |
| ga_baseline_0006   |        1 | sensitive    |           0.1427 |         0.14   |          0.1455 |            0.8656 |          0.8603 |           0.8716 |
| ga_baseline_0007   |        2 | sensitive    |           0.0254 |         0.0248 |          0.0252 |            0.8535 |          0.8633 |           0.8516 |
| ga_sam_0001        |        4 | sensitive    |           0.0238 |         0.0245 |          0.0225 |            0.9661 |          0.9721 |           0.9754 |
| ga_sam_0002        |        3 | sensitive    |           0.0241 |         0.0229 |          0.0239 |            0.9659 |          0.9701 |           0.9673 |
| ga_sam_0003        |        7 | sensitive    |           0.0247 |         0.0246 |          0.024  |            0.946  |          0.955  |           0.954  |
| ga_sam_0004        |        2 | sensitive    |           0.0494 |         0.0495 |          0.0488 |            0.9431 |          0.9494 |           0.9467 |
| ga_sam_0005        |        2 | sensitive    |           0.0414 |         0.0386 |          0.0419 |            0.9196 |          0.9292 |           0.9207 |
| ga_sam_0006        |        1 | sensitive    |           0.1427 |         0.14   |          0.1455 |            0.8802 |          0.8812 |           0.8819 |
| nsga_baseline_0001 |        3 | sensitive    |           0.0212 |         0.0195 |          0.0205 |            0.8578 |          0.8512 |           0.8667 |
| nsga_baseline_0002 |        2 | sensitive    |           0.0553 |         0.0531 |          0.0532 |            0.8262 |          0.8204 |           0.8501 |
| nsga_baseline_0003 |        1 | sensitive    |           0.1807 |         0.1785 |          0.1823 |            0.8238 |          0.8243 |           0.8349 |
| nsga_sam_0001      |        4 | sensitive    |           0.0222 |         0.0223 |          0.0222 |            0.836  |          0.8732 |           0.8331 |
| nsga_sam_0002      |        2 | sensitive    |           0.0294 |         0.0295 |          0.0315 |            0.8026 |          0.8133 |           0.806  |
| nsga_sam_0003      |        1 | sensitive    |           0.1548 |         0.1534 |          0.1592 |            0.779  |          0.777  |           0.7858 |

All 19 rules predict 'sensitive'. Fidelity on test (macro-F1, same function as the pilot) is in `posthoc_global_rules_agreement_coverage.csv` (0.440-0.494).

## 2. Test-row coverage per pool

| pool          |   n_rules |   n_test_rows |   share_test_covered_by_ge1_rule |   share_test_covered_by_exactly_1_rule |   share_test_covered_by_ge2_rules |   max_rules_covering_a_test_row |
|:--------------|----------:|--------------:|---------------------------------:|---------------------------------------:|----------------------------------:|--------------------------------:|
| ga_baseline   |         7 |         30728 |                           0.1547 |                                 0.0672 |                            0.0875 |                               5 |
| ga_sam        |         6 |         30728 |                           0.1546 |                                 0.0643 |                            0.0903 |                               5 |
| nsga_baseline |         3 |         30728 |                           0.1823 |                                 0.1291 |                            0.0532 |                               3 |
| nsga_sam      |         3 |         30728 |                           0.1767 |                                 0.1434 |                            0.0333 |                               3 |

## 3. Chance baseline (200 random rules per real rule, same length, train coverage >= 2%)

Floor = 2% of train rows = at least 4,917 of 245,845 rows (the extraction's `MIN_COVERAGE = 0.02`). Random rule: distinct random features, level uniform over the feature's observed train levels, rejection-sampled (cap 10,000 attempts per real rule; 0 rules fell short of 200 draws; attempts per pool in the table), class = majority of the pool's network class_pred on the covered train rows (tie -> 'resistant'), scored on train/val/test. Seed 20261012. 'Above' = strictly above the 95th percentile of the rule's own 200 draws.

|                                | 0           | 1      | 2             | 3        | 4      |
|:-------------------------------|:------------|:-------|:--------------|:---------|:-------|
| pool                           | ga_baseline | ga_sam | nsga_baseline | nsga_sam | ALL_19 |
| n_rules                        | 7           | 6      | 3             | 3        | 19     |
| total_attempts                 | 1726        | 3960   | 673           | 837      | 7196   |
| mean_real_agreement_train      | 0.9005      | 0.9368 | 0.8359        | 0.8058   | 0.8868 |
| mean_random_agreement_train    | 0.5432      | 0.5504 | 0.5423        | 0.5435   | 0.5454 |
| mean_p95_agreement_train       | 0.6262      | 0.6333 | 0.6369        | 0.6104   | 0.6276 |
| share_real_above_own_p95_train | 1.0         | 1.0    | 1.0           | 1.0      | 1.0    |
| mean_real_agreement_val        | 0.9032      | 0.9428 | 0.832         | 0.8211   | 0.8915 |
| mean_random_agreement_val      | 0.5433      | 0.5499 | 0.5424        | 0.5424   | 0.5451 |
| mean_p95_agreement_val         | 0.6271      | 0.6326 | 0.6298        | 0.6172   | 0.6277 |
| share_real_above_own_p95_val   | 1.0         | 1.0    | 1.0           | 1.0      | 1.0    |
| mean_real_agreement_test       | 0.9         | 0.941  | 0.8505        | 0.8083   | 0.8907 |
| mean_random_agreement_test     | 0.5473      | 0.5505 | 0.5476        | 0.5433   | 0.5477 |
| mean_p95_agreement_test        | 0.6319      | 0.6369 | 0.6381        | 0.615    | 0.6318 |
| share_real_above_own_p95_test  | 1.0         | 1.0    | 1.0           | 1.0      | 1.0    |

Random-rule mean fidelity (macro-F1): 0.351-0.355 on every split; share of real rules above their own 95th percentile of fidelity: 1.0 in every pool and split (columns `mean_random_fidelity_*`, `share_real_fid_above_own_p95_*` in `posthoc_global_random_baseline_summary.csv`).

## Could not be verified

- The GA run logs do not print the coverage floor; the 0.02 value is taken from `run_logged.MIN_COVERAGE` and the NSGA-II logs, and is consistent with the lowest GA rule coverage (0.0221).
- The SAM pools were extracted on `real_extraction/predictions/sam_*.parquet`, which I earlier showed to equal the confirmatory checkpoint's predictions (0 mismatches); I did not re-check that here.
- The null is one particular choice (uniform over observed levels); real rules were selected on fidelity, so beating it is expected.
