# POST-HOC DIAGNOSTICS (decided after seeing RESULTS.md)

Not part of the frozen design. The primary `artifacts/per_drug/RESULTS.md` is unchanged. Code: `src/per_drug/diagnostics.py` (read-only on `artifacts/per_drug/pilot/`; no extraction re-run). All CSVs here are prefixed `posthoc_`. Numbers only.

## 1. Fidelity metric

Function used for every fidelity number (`src/common/metrics.py`):

```python
def fidelity_f1(rule_class_preds: Sequence[str], model_class_preds: Sequence[str]) -> float:
    """
    Fidelity of a rule (or rule set) to the black-box model, measured as
    macro F1-score between the rule's predicted classes and the model's
    predicted classes on the same rows.

    Both sequences must already be mapped to the discrete response
    classes defined in artifacts/data/response_classes.json
    (see docs/CONTRACTS.md §0.4 and methodology §4.2.2 / §4.6.1) —
    this function does not do that mapping itself. Use rule_class_preds()
    above to derive these sequences from a features dataframe.

    Args:
        rule_class_preds: predicted class per row, from the rule.
        model_class_preds: predicted class per row, from DeepCCDS
            (baseline or SAM — whichever condition is being evaluated).

    Returns:
        macro F1-score in [0, 1].
    """
    if len(rule_class_preds) != len(model_class_preds):
        raise ValueError(
            f"length mismatch: rule_class_preds has {len(rule_class_preds)} rows, "
            f"model_class_preds has {len(model_class_preds)}"
        )
    if len(rule_class_preds) == 0:
        raise ValueError("cannot compute fidelity on an empty rule coverage set")

    return float(f1_score(model_class_preds, rule_class_preds, average="macro", zero_division=0))
```

Called as `fidelity_f1(Series(rule_class), Series(network class_pred on covered rows))`: `f1_score(y_true=network, y_pred=rule, average='macro', zero_division=0)`, **no `labels=` argument**, so the label set is the union of the classes present in y_true and y_pred. A rule always predicts one fixed class c. Consequences (checked against sklearn on 600 random cases; closed form below reproduces the stored fidelity of all 2,153 held-out rule evaluations to 1e-12):

- Covered rows all of class c: labels = {c}, F1 = 1.0 (fidelity exactly 1.0).
- Covered rows all of the other class: labels = {c, d}, both F1 = 0 -> fidelity 0.0.
- Both classes covered, share of class c = p: F1(c) = 2p/(1+p), F1(d) = 0 (zero_division=0) -> fidelity = p/(1+p) < 0.5 for every p < 1.

So the metric takes the value 1.0 only when the covered rows contain a single network class (equal to the rule class), and it is capped below 0.5 whenever both classes are covered. In the data: max fidelity with two classes covered = 0.4865-0.4921 (per arm x extractor); fidelity==1.0 and not single-class: 0 cases.

| extractor   | arm     |   n_rule_evals |   n_nan_fidelity |   share_fidelity_exactly_1 |   share_fidelity_exactly_0 |   share_single_network_class |    p10 |    p20 |    p30 |    p40 |    p50 |    p60 |    p70 |    p80 |    p90 |
|:------------|:--------|---------------:|-----------------:|---------------------------:|---------------------------:|-----------------------------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| ga          | adam    |            512 |                0 |                     0.0918 |                     0.002  |                       0.0938 | 0.344  | 0.381  | 0.4084 | 0.4286 | 0.4375 | 0.451  | 0.4667 | 0.4762 | 0.4902 |
| ga          | adam_wd |            552 |                0 |                     0.1594 |                     0.0036 |                       0.163  | 0.3333 | 0.3786 | 0.4    | 0.4211 | 0.4375 | 0.4545 | 0.4702 | 0.4848 | 1      |
| ga          | sam     |            480 |                0 |                     0.1729 |                     0.0021 |                       0.175  | 0.3457 | 0.4    | 0.4167 | 0.4348 | 0.4444 | 0.4568 | 0.4706 | 0.4879 | 1      |
| nsga        | adam    |            203 |                0 |                     0.0345 |                     0.0049 |                       0.0394 | 0.365  | 0.4    | 0.4129 | 0.4286 | 0.44   | 0.4486 | 0.4586 | 0.4679 | 0.4762 |
| nsga        | adam_wd |            214 |                0 |                     0.0701 |                     0      |                       0.0701 | 0.3509 | 0.3889 | 0.4091 | 0.4236 | 0.4367 | 0.4508 | 0.4615 | 0.4688 | 0.4824 |
| nsga        | sam     |            192 |                0 |                     0.0781 |                     0      |                       0.0781 | 0.3718 | 0.4065 | 0.4167 | 0.4339 | 0.4421 | 0.45   | 0.4583 | 0.4706 | 0.4848 |

Share columns are fractions of held-out rule evaluations (rows = rules in a fold's pool scored on that fold's held-out rows; 0 with undefined fidelity). p10..p90 = fidelity deciles over those evaluations.

| extractor   | arm     |   fid==1 & single_class |   fid==1 & not single_class |   fid<=0.5 & single_class |   fid max when 2 classes covered |
|:------------|:--------|------------------------:|----------------------------:|--------------------------:|---------------------------------:|
| ga          | adam    |                      47 |                           0 |                         1 |                           0.4912 |
| ga          | adam_wd |                      88 |                           0 |                         2 |                           0.4912 |
| ga          | sam     |                      83 |                           0 |                         1 |                           0.4921 |
| nsga        | adam    |                       7 |                           0 |                         1 |                           0.4902 |
| nsga        | adam_wd |                      15 |                           0 |                         0 |                           0.4889 |
| nsga        | sam     |                      15 |                           0 |                         0 |                           0.4865 |

### net_other (rows the network did not train on) vs net_train

Held-out rows are split by the network's own split: net_train = the network's train rows; net_other = its val/test rows (20% of rows). Rule-level medians (not the drug-level means used in RESULTS.md):

| extractor   | arm     |   n_rules_total |   n_rules_no_net_other_cover |   mean_n_cov_net_train |   mean_n_cov_net_other |   share_single_class_net_train |   share_single_class_net_other |   share_fid1_net_train |   share_fid1_net_other |   median_fid_net_train |   median_fid_net_other |   median_fid_net_train_matched_size |
|:------------|:--------|----------------:|-----------------------------:|-----------------------:|-----------------------:|-------------------------------:|-------------------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|------------------------------------:|
| ga          | adam    |             512 |                           49 |                12.7852 |                 3.3218 |                         0.127  |                         0.6134 |                 0.123  |                 0.5529 |                 0.4375 |                 1      |                              0.65   |
| ga          | adam_wd |             552 |                           73 |                12.3442 |                 3.3674 |                         0.1902 |                         0.6117 |                 0.183  |                 0.5386 |                 0.4305 |                 1      |                              0.65   |
| ga          | sam     |             480 |                           51 |                13.1167 |                 3.6247 |                         0.2062 |                         0.634  |                 0.1979 |                 0.5734 |                 0.4444 |                 1      |                              0.65   |
| nsga        | adam    |             203 |                           15 |                14.7931 |                 3.6968 |                         0.0837 |                         0.5    |                 0.0739 |                 0.4574 |                 0.439  |                 0.4615 |                              0.595  |
| nsga        | adam_wd |             214 |                           17 |                14.229  |                 3.7411 |                         0.1121 |                         0.4264 |                 0.1121 |                 0.3756 |                 0.4361 |                 0.4286 |                              0.5833 |
| nsga        | sam     |             192 |                            9 |                15.125  |                 3.6011 |                         0.1198 |                         0.5137 |                 0.1198 |                 0.4536 |                 0.4415 |                 0.4444 |                              0.6438 |

`median_fid_net_train_matched_size` = net_train fidelity recomputed on random subsets of the covered net_train rows with the same size as the rule's net_other coverage (20 draws per rule, rules with net_train coverage >= net_other coverage > 0).

Class balance does not differ between the subsets (share of network class 'resistant', row-weighted over the 14 drugs: Adam 0.515 train / 0.516 other; Adam+WD 0.508 / 0.508; SAM 0.524 / 0.531; `posthoc_1_class_balance_by_net_split.csv`).

Why net_other fidelity is higher, from these numbers: net_other covers about 3.3-3.7 held-out rows per rule against 12.3-15.1 for net_train, so the covered rows are far more often a single class (0.43-0.63 vs 0.08-0.21), and a single-class covered set is the only way to reach fidelity above 0.5. Matching the sample size raises net_train GA median fidelity from 0.44 to 0.65, so the covered-row count explains part of the gap for GA; for NSGA-II the matched net_train median (0.58-0.64) is above the net_other median (0.43-0.46). The rest of the GA gap (0.65 vs 1.00) is **not explained and not verified** here. Rules with no net_other coverage (9-73 per arm x extractor) have undefined net_other fidelity and drop out of that median.

## 2. Sensitivity metric: plain agreement (accuracy) and coverage (SENSITIVITY ANALYSIS)

Accuracy = share of the rule's covered held-out rows whose network class equals the rule class; coverage = covered held-out rows / held-out rows. Per cell: median over the pool's rules; per drug: mean over 5 folds; comparisons as in the plan (paired Wilcoxon, BH within metric, bootstrap 95% CI of the median paired difference, 10,000 resamples, seed 20261010).

| extractor   | arm     |   accuracy |   coverage |   fidelity |
|:------------|:--------|-----------:|-----------:|-----------:|
| ga          | adam    |     0.8034 |     0.0822 |     0.4647 |
| ga          | adam_wd |     0.8075 |     0.0888 |     0.5154 |
| ga          | sam     |     0.8337 |     0.0927 |     0.5345 |
| nsga        | adam    |     0.7584 |     0.1056 |     0.4381 |
| nsga        | adam_wd |     0.7655 |     0.0984 |     0.451  |
| nsga        | sam     |     0.7757 |     0.1028 |     0.4763 |

| metric   | extractor   | comparison        |   n_drugs_paired |   median_diff |   ci95_lo_median_diff |   ci95_hi_median_diff |   wilcoxon_p |   wilcoxon_p_bh_within_metric |
|:---------|:------------|:------------------|-----------------:|--------------:|----------------------:|----------------------:|-------------:|------------------------------:|
| accuracy | ga          | SAM_minus_AdamWD  |               14 |        0.0212 |               -0.036  |                0.089  |       0.3258 |                        0.9515 |
| accuracy | ga          | AdamWD_minus_Adam |               14 |        0.0088 |               -0.0164 |                0.0441 |       0.7148 |                        0.9515 |
| accuracy | nsga        | SAM_minus_AdamWD  |               14 |        0.0011 |               -0.0702 |                0.0957 |       0.8077 |                        0.9515 |
| accuracy | nsga        | AdamWD_minus_Adam |               14 |        0.008  |               -0.0707 |                0.0486 |       0.9515 |                        0.9515 |
| coverage | ga          | SAM_minus_AdamWD  |               14 |       -0.0018 |               -0.0093 |                0.0075 |       0.8552 |                        0.8552 |
| coverage | ga          | AdamWD_minus_Adam |               14 |        0.0058 |               -0.0047 |                0.0139 |       0.2166 |                        0.4331 |
| coverage | nsga        | SAM_minus_AdamWD  |               14 |        0.0041 |               -0.0073 |                0.0098 |       0.5016 |                        0.6688 |
| coverage | nsga        | AdamWD_minus_Adam |               14 |       -0.0099 |               -0.0175 |                0.002  |       0.0785 |                        0.314  |

## 3. Chance baseline (matched random rules)

For every real rule in every fold's pool (2,153 rules), 200 random rules with the same number of conditions: distinct random features out of 572, level drawn uniformly from the levels observed for that feature in the fold's extraction rows (`fid` part, 550-ish rows), rejection-sampled until the rule covers >= 25 of those rows (cap 10,000 attempts per real rule). Class = majority of the arm's class_pred on the covered extraction rows (ties -> 'resistant', as pandas `.mode().iat[0]` in the real extractors). Scored on the same fold's held-out rows with the same fidelity function (and with accuracy). Seeds: `SeedSequence([20261011, drug, arm, extractor, fold])`. Draw failures: 0 rules with fewer than 200 draws, 0 cap hits. Random draws whose held-out coverage was empty (fidelity undefined): 0-3 per arm x extractor.

| extractor   | arm     |   n_rules |   mean_random_fidelity |   mean_of_per_rule_p95_fidelity |   mean_real_fidelity |   share_real_above_own_p95 |   share_real_at_or_above_own_p95 |   drug_level_mean_real_minus_random_fidelity |   mean_random_accuracy |   mean_of_per_rule_p95_accuracy |   mean_real_accuracy |   share_real_acc_above_own_p95 |   drug_level_mean_real_minus_random_accuracy |
|:------------|:--------|----------:|-----------------------:|--------------------------------:|---------------------:|---------------------------:|---------------------------------:|---------------------------------------------:|-----------------------:|--------------------------------:|---------------------:|-------------------------------:|---------------------------------------------:|
| ga          | adam    |       512 |                 0.3464 |                          0.4129 |               0.4702 |                     0.6543 |                           0.668  |                                       0.1383 |                 0.5349 |                          0.7053 |               0.7571 |                         0.6543 |                                       0.2399 |
| ga          | adam_wd |       552 |                 0.3542 |                          0.418  |               0.5038 |                     0.6286 |                           0.6377 |                                       0.1731 |                 0.5532 |                          0.7206 |               0.7587 |                         0.6286 |                                       0.2215 |
| ga          | sam     |       480 |                 0.354  |                          0.4172 |               0.5186 |                     0.6958 |                           0.7146 |                                       0.1878 |                 0.5534 |                          0.7151 |               0.7812 |                         0.6958 |                                       0.2448 |
| nsga        | adam    |       203 |                 0.3461 |                          0.415  |               0.4435 |                     0.7044 |                           0.7143 |                                       0.0976 |                 0.5344 |                          0.7114 |               0.7583 |                         0.7044 |                                       0.2228 |
| nsga        | adam_wd |       214 |                 0.3546 |                          0.4194 |               0.462  |                     0.6776 |                           0.6869 |                                       0.108  |                 0.5543 |                          0.7244 |               0.7589 |                         0.6776 |                                       0.2041 |
| nsga        | sam     |       192 |                 0.3531 |                          0.4232 |               0.471  |                     0.6719 |                           0.6823 |                                       0.1174 |                 0.5515 |                          0.7212 |               0.7729 |                         0.6719 |                                       0.2191 |

Definitions: 'mean random' = mean over all draws of all rules; 95th percentile = mean over rules of each rule's own 95th percentile; 'share above' = share of real rules whose fidelity is strictly above the 95th percentile of their own 200 draws (the 'at or above' column counts ties); drug-level difference = per drug mean over rules of (real - mean of own random draws), then mean over the 14 drugs. Number of real rules by length: `posthoc_3_rules_by_length.csv`. Attempts used: `posthoc_3_random_baseline_summary.csv` (total_attempts, total_draws).

Caveat: rules with real fidelity exactly 1.0 or tied with the percentile are only counted as above when strictly greater. Random rules are drawn from the extraction-row level distribution and are not conditioned on the real rules' features, so real rules (selected on fidelity) are expected to beat them; the comparison shows magnitude, not independent validation.

## 4. Stability: what each measure uses

- **Fold gene set (primary, Jaccard over 10 fold pairs; Kuncheva):** per drug x arm x extractor x fold, the union of `genes` over all rules in that fold's pooled output (`aggregate_runs` over the 10 seeds). The 'genes' are the feature_map symbols of the rule's conditions, so they include pathway-activity feature names (38 pathway features) and mutation-gene names (534); universe N = 572.
- **Seed gene set (secondary):** per drug x arm x extractor x fold x seed, the union of `genes` over the rules of that single seed's front (10 sets per fold); Jaccard over the 45 seed pairs within a fold, averaged over folds.
- Pairs involving an empty set are dropped (0 empty fold sets occurred).
- Chance Jaccard = exact E[|A∩B|/|A∪B|] for two independent uniformly random subsets of the observed sizes out of 572 (hypergeometric), averaged over the same pairs. Kuncheva's index has expected value 0 by construction.

| extractor   | arm     |   mean_fold_set_size |   mean_seed_set_size |   obs_jaccard_folds |   chance_jaccard_folds |   obs_jaccard_seeds |   chance_jaccard_seeds |
|:------------|:--------|---------------------:|---------------------:|--------------------:|-----------------------:|--------------------:|-----------------------:|
| ga          | adam    |              10.8429 |               9.0643 |              0.1101 |                 0.009  |              0.0315 |                 0.0076 |
| ga          | adam_wd |              12.0429 |               9.2057 |              0.1016 |                 0.0095 |              0.0331 |                 0.0078 |
| ga          | sam     |               9.6286 |               9.2157 |              0.1125 |                 0.0081 |              0.0346 |                 0.0078 |
| nsga        | adam    |               3.7    |               2.7143 |              0.38   |                 0.0035 |              0.0041 |                 0.0027 |
| nsga        | adam_wd |               3.9857 |               2.8429 |              0.3261 |                 0.0037 |              0.0046 |                 0.0028 |
| nsga        | sam     |               3.4714 |               2.7414 |              0.3585 |                 0.0033 |              0.0053 |                 0.0027 |

(Means over the 14 drugs; sizes are numbers of distinct features.)

## 5. Rule length and corrected p-values

Rule length = number of conditions (`len(rule['conditions'])`, `src.common.metrics.comprehensibility`). The length effects in RESULTS.md are in conditions: median paired difference SAM − Adam+WD = −0.15 conditions (GA, 95% CI [−0.40, −0.10]) and −0.20 (NSGA-II, [−0.30, −0.10]); those are medians of per-drug means of per-fold median lengths.

| extractor   | arm     |   n_rules |   mean_length_all_rules |   median_length_all_rules |   max_length |   drug_level_mean_of_fold_median_length |
|:------------|:--------|----------:|------------------------:|--------------------------:|-------------:|----------------------------------------:|
| ga          | adam    |       512 |                  2.0117 |                         2 |            6 |                                  1.7714 |
| ga          | adam_wd |       552 |                  2.1214 |                         2 |            7 |                                  1.8571 |
| ga          | sam     |       480 |                  1.8542 |                         2 |            6 |                                  1.5929 |
| nsga        | adam    |       203 |                  2.0837 |                         2 |            6 |                                  1.9286 |
| nsga        | adam_wd |       214 |                  2.1449 |                         2 |            5 |                                  2.0286 |
| nsga        | sam     |       192 |                  1.9896 |                         2 |            6 |                                  1.8714 |

`wilcoxon_p_bh_all_comparisons` = BH over all 24 planned comparisons (6 metrics x 2 extractors x 2 comparisons) excluding the two Kuncheva GA rows with n < 5 (n = 1 and 2), i.e. a family of 22; `wilcoxon_p_bh_within_metric` is the original within-metric BH. The primary `comparisons.csv` is unchanged.

| metric                   | extractor   | comparison        |   n_drugs_paired |   median_diff |   ci95_lo_median_diff |   ci95_hi_median_diff |   wilcoxon_p |   wilcoxon_p_bh_within_metric | in_bh_all_family   |   wilcoxon_p_bh_all_comparisons |
|:-------------------------|:------------|:------------------|-----------------:|--------------:|----------------------:|----------------------:|-------------:|------------------------------:|:-------------------|--------------------------------:|
| heldout_fidelity         | ga          | SAM_minus_AdamWD  |               14 |        0.0105 |               -0.043  |                0.0752 |       0.583  |                        0.7773 | True               |                          0.8016 |
| heldout_fidelity         | ga          | AdamWD_minus_Adam |               14 |        0.006  |               -0.0178 |                0.0605 |       0.2958 |                        0.7773 | True               |                          0.7964 |
| heldout_fidelity         | nsga        | SAM_minus_AdamWD  |               14 |        0.0038 |               -0.0549 |                0.0604 |       0.9515 |                        0.9515 | True               |                          1      |
| heldout_fidelity         | nsga        | AdamWD_minus_Adam |               14 |       -0.0043 |               -0.0366 |                0.074  |       0.4631 |                        0.7773 | True               |                          0.8016 |
| phi                      | ga          | SAM_minus_AdamWD  |               14 |        0.0191 |               -0.0355 |                0.1117 |       0.5016 |                        1      | True               |                          0.8016 |
| phi                      | ga          | AdamWD_minus_Adam |               14 |        0.0262 |               -0.0348 |                0.1205 |       0.1937 |                        0.7749 | True               |                          0.7581 |
| phi                      | nsga        | SAM_minus_AdamWD  |               14 |       -0.0101 |               -0.1553 |                0.1851 |       1      |                        1      | True               |                          1      |
| phi                      | nsga        | AdamWD_minus_Adam |               14 |       -0.0111 |               -0.157  |                0.2074 |       0.9515 |                        1      | True               |                          1      |
| length                   | ga          | SAM_minus_AdamWD  |               14 |       -0.15   |               -0.4    |               -0.1    |       0.0067 |                        0.0268 | True               |                          0.1473 |
| length                   | ga          | AdamWD_minus_Adam |               14 |        0.05   |               -0.05   |                0.2    |       0.2116 |                        0.2116 | True               |                          0.7581 |
| length                   | nsga        | SAM_minus_AdamWD  |               14 |       -0.2    |               -0.3    |               -0.1    |       0.0168 |                        0.0336 | True               |                          0.1846 |
| length                   | nsga        | AdamWD_minus_Adam |               14 |        0.15   |               -0.1    |                0.3    |       0.1613 |                        0.2116 | True               |                          0.7581 |
| stability_jaccard_folds  | ga          | SAM_minus_AdamWD  |               14 |        0.021  |                0.0044 |                0.0374 |       0.0906 |                        0.3623 | True               |                          0.6642 |
| stability_jaccard_folds  | ga          | AdamWD_minus_Adam |               14 |        0.0014 |               -0.0387 |                0.0197 |       0.8552 |                        0.8552 | True               |                          1      |
| stability_jaccard_folds  | nsga        | SAM_minus_AdamWD  |               14 |        0.068  |               -0.0444 |                0.1247 |       0.391  |                        0.5213 | True               |                          0.8016 |
| stability_jaccard_folds  | nsga        | AdamWD_minus_Adam |               14 |       -0.0602 |               -0.1964 |                0.0763 |       0.3258 |                        0.5213 | True               |                          0.7964 |
| stability_kuncheva_folds | ga          | SAM_minus_AdamWD  |                1 |        0.144  |                0.144  |                0.144  |       1      |                        1      | False              |                        nan      |
| stability_kuncheva_folds | ga          | AdamWD_minus_Adam |                2 |       -0.1199 |               -0.1297 |               -0.1102 |       0.5    |                        1      | False              |                        nan      |
| stability_kuncheva_folds | nsga        | SAM_minus_AdamWD  |               14 |        0.0842 |               -0.1738 |                0.2893 |       0.5525 |                        1      | True               |                          0.8016 |
| stability_kuncheva_folds | nsga        | AdamWD_minus_Adam |               13 |       -0.0018 |               -0.2234 |                0.168  |       0.7869 |                        1      | True               |                          1      |
| stability_jaccard_seeds  | ga          | SAM_minus_AdamWD  |               14 |       -0.0009 |               -0.0057 |                0.0052 |       1      |                        1      | True               |                          1      |
| stability_jaccard_seeds  | ga          | AdamWD_minus_Adam |               14 |        0.0023 |               -0.0014 |                0.0054 |       0.2412 |                        0.7773 | True               |                          0.7581 |
| stability_jaccard_seeds  | nsga        | SAM_minus_AdamWD  |               14 |        0.0007 |               -0.0015 |                0.0025 |       0.583  |                        0.7773 | True               |                          0.8016 |
| stability_jaccard_seeds  | nsga        | AdamWD_minus_Adam |               14 |        0.0005 |               -0.0015 |                0.0026 |       0.583  |                        0.7773 | True               |                          0.8016 |

## 6. Timing for 379 drugs on 3 workers

Inputs from `pilot/manifest.jsonl` (measured with 3 workers running concurrently): mean seconds per seed run GA 14.80, NSGA-II 1.69; mean per-cell overhead (aggregation + held-out scoring) GA 0.73 s, NSGA-II 0.64 s. Per drug: 15 cells (3 arms x 5 folds) per extractor. Hours = 379 x CPU-seconds per drug / 3 workers / 3600, assuming perfect packing of cells over workers and no per-drug data-loading cost (about 10 s per drug, not included). Check: the same model predicts 11637 s for the 14 pilot drugs at 10 seeds, observed 12,232 s (+5%).

| plan         |   seeds |   cpu_s_per_drug |   hours_379_drugs_3_workers |   hours_379_drugs_1_worker_equiv |
|:-------------|--------:|-----------------:|----------------------------:|---------------------------------:|
| GA+NSGA-II   |      10 |        2493.55   |                     87.505  |                         262.515  |
| GA+NSGA-II   |       5 |        1257.05   |                     44.1133 |                         132.34   |
| GA+NSGA-II   |       3 |         762.456  |                     26.7566 |                          80.2697 |
| GA only      |      10 |        2230.54   |                     78.2755 |                         234.827  |
| GA only      |       5 |        1120.78   |                     39.3311 |                         117.993  |
| GA only      |       3 |         676.878  |                     23.7534 |                          71.2602 |
| NSGA-II only |      10 |         263.004  |                      9.2295 |                          27.6885 |
| NSGA-II only |       5 |         136.271  |                      4.7821 |                          14.3463 |
| NSGA-II only |       3 |          85.5783 |                      3.0032 |                           9.0095 |

## 7. Positive-control feasibility (POST-HOC, read-only)

Single-condition rule `feature = High` on ALL rows of the control drug, scored against each of the three networks' class_pred and against the true class. 'Best direction' = the class (sensitive/resistant) with higher agreement; agreement on covered rows = precision of that class; recall = share of that class's rows covered. Reference: 2,000 random single-condition rules per drug and target (random feature, observed level, coverage >= 25 rows, class = majority of the same target on covered rows, scored in-sample on the same rows), giving the 95th percentile of agreement/fidelity. `share_random_acc_ge_best` = share of random draws with agreement at least that of the feature rule. Both directions are in `posthoc_positive_control_feasibility.csv`.

| drug        | feature   | target      | rule_class   |   n_rows |   n_feature_High |   share_feature_High |   agreement_covered |   precision |   recall |   base_rate_class |   rand_acc_p95 |   share_random_acc_ge_best |   fidelity_f1 |   rand_fid_p95 |
|:------------|:----------|:------------|:-------------|---------:|-----------------:|---------------------:|--------------------:|------------:|---------:|------------------:|---------------:|---------------------------:|--------------:|---------------:|
| Erlotinib   | EGFR_mut  | net_adam    | sensitive    |      860 |              376 |               0.4372 |              0.5612 |      0.5612 |   0.4763 |            0.5151 |         0.6231 |                     0.252  |        0.3595 |         0.3839 |
| Erlotinib   | EGFR_mut  | net_adam_wd | resistant    |      860 |              376 |               0.4372 |              0.508  |      0.508  |   0.4021 |            0.5523 |         0.597  |                     0.9235 |        0.3369 |         0.3738 |
| Erlotinib   | EGFR_mut  | net_sam     | sensitive    |      860 |              376 |               0.4372 |              0.5638 |      0.5638 |   0.4482 |            0.55   |         0.6327 |                     0.328  |        0.3605 |         0.3875 |
| Erlotinib   | EGFR_mut  | true_class  | sensitive    |      860 |              376 |               0.4372 |              0.5559 |      0.5559 |   0.4918 |            0.4942 |         0.5902 |                     0.1175 |        0.3573 |         0.3711 |
| Olaparib    | BRCA1_mut | net_adam    | resistant    |      864 |              288 |               0.3333 |              0.5278 |      0.5278 |   0.3207 |            0.5486 |         0.6042 |                     0.77   |        0.3455 |         0.3766 |
| Olaparib    | BRCA1_mut | net_adam_wd | sensitive    |      864 |              288 |               0.3333 |              0.5833 |      0.5833 |   0.3485 |            0.5579 |         0.6638 |                     0.3295 |        0.3684 |         0.399  |
| Olaparib    | BRCA1_mut | net_sam     | sensitive    |      864 |              288 |               0.3333 |              0.5278 |      0.5278 |   0.3439 |            0.5116 |         0.6121 |                     0.344  |        0.3455 |         0.3797 |
| Olaparib    | BRCA1_mut | true_class  | resistant    |      864 |              288 |               0.3333 |              0.5035 |      0.5035 |   0.3318 |            0.5058 |         0.5902 |                     0.871  |        0.3349 |         0.3711 |
| Olaparib    | BRCA2_mut | net_adam    | sensitive    |      864 |              312 |               0.3611 |              0.5032 |      0.5032 |   0.4026 |            0.4514 |         0.6042 |                     0.9705 |        0.3348 |         0.3766 |
| Olaparib    | BRCA2_mut | net_adam_wd | sensitive    |      864 |              312 |               0.3611 |              0.609  |      0.609  |   0.3942 |            0.5579 |         0.6648 |                     0.1985 |        0.3785 |         0.3993 |
| Olaparib    | BRCA2_mut | net_sam     | sensitive    |      864 |              312 |               0.3611 |              0.5353 |      0.5353 |   0.3778 |            0.5116 |         0.6066 |                     0.2815 |        0.3486 |         0.3776 |
| Olaparib    | BRCA2_mut | true_class  | sensitive    |      864 |              312 |               0.3611 |              0.5128 |      0.5128 |   0.3747 |            0.4942 |         0.5863 |                     0.494  |        0.339  |         0.3696 |
| PLX-4720    | BRAF_mut  | net_adam    | sensitive    |      864 |              333 |               0.3854 |              0.5195 |      0.5195 |   0.4303 |            0.4653 |         0.5986 |                     0.848  |        0.3419 |         0.3745 |
| PLX-4720    | BRAF_mut  | net_adam_wd | sensitive    |      864 |              333 |               0.3854 |              0.5255 |      0.5255 |   0.4353 |            0.4653 |         0.5862 |                     0.761  |        0.3445 |         0.3696 |
| PLX-4720    | BRAF_mut  | net_sam     | sensitive    |      864 |              333 |               0.3854 |              0.5886 |      0.5886 |   0.4109 |            0.5521 |         0.6285 |                     0.171  |        0.3705 |         0.3859 |
| PLX-4720    | BRAF_mut  | true_class  | sensitive    |      864 |              333 |               0.3854 |              0.5435 |      0.5435 |   0.4067 |            0.515  |         0.5769 |                     0.1385 |        0.3521 |         0.3659 |
| Selumetinib | BRAF_mut  | net_adam    | sensitive    |      864 |              333 |               0.3854 |              0.5135 |      0.5135 |   0.4286 |            0.4618 |         0.6    |                     0.9295 |        0.3393 |         0.375  |
| Selumetinib | BRAF_mut  | net_adam_wd | sensitive    |      864 |              333 |               0.3854 |              0.5075 |      0.5075 |   0.4401 |            0.4444 |         0.6091 |                     0.9725 |        0.3367 |         0.3785 |
| Selumetinib | BRAF_mut  | net_sam     | sensitive    |      864 |              333 |               0.3854 |              0.5345 |      0.5345 |   0.4331 |            0.4757 |         0.5987 |                     0.332  |        0.3483 |         0.3745 |
| Selumetinib | BRAF_mut  | true_class  | sensitive    |      864 |              333 |               0.3854 |              0.5075 |      0.5075 |   0.4183 |            0.4676 |         0.6067 |                     0.98   |        0.3367 |         0.3776 |
| Selumetinib | KRAS_mut  | net_adam    | sensitive    |      864 |              187 |               0.2164 |              0.6096 |      0.6096 |   0.2857 |            0.4618 |         0.6034 |                     0.042  |        0.3787 |         0.3763 |
| Selumetinib | KRAS_mut  | net_adam_wd | sensitive    |      864 |              187 |               0.2164 |              0.615  |      0.615  |   0.2995 |            0.4444 |         0.6111 |                     0.0445 |        0.3808 |         0.3793 |
| Selumetinib | KRAS_mut  | net_sam     | sensitive    |      864 |              187 |               0.2164 |              0.631  |      0.631  |   0.2871 |            0.4757 |         0.6045 |                     0.03   |        0.3869 |         0.3767 |
| Selumetinib | KRAS_mut  | true_class  | sensitive    |      864 |              187 |               0.2164 |              0.6043 |      0.6043 |   0.2797 |            0.4676 |         0.6021 |                     0.049  |        0.3767 |         0.3758 |

Agreement above the random 95th percentile: Selumetinib/KRAS_mut against all four targets (0.604-0.631 vs about 0.60-0.61 p95, share of random draws at least as good 0.03-0.05); no other control feature is above it (shares 0.12-0.98).

## Could not be verified / limits

- Why GA net_other fidelity reaches 1.0 more often than a size-matched net_train subset (see §1): not established.
- The meaning of `_mut` = High (any variant vs activating) is still unverified, so a low agreement of `feature = High` does not distinguish 'network does not carry the signal' from 'flag is too broad'; the table only shows that, in these data, the single-feature rules do not separate the classes beyond random single-condition rules (except KRAS_mut for Selumetinib).
- Random-rule levels are drawn from the observed levels of the fold's extraction rows; the extractors' own sampling distribution differs, so this baseline is one particular choice of null.
- The chance Jaccard assumes uniform random feature selection; real rules are not uniform over features.
- Timing assumes linear scaling to 3 workers and ignores per-drug loading.
