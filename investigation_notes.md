# Investigation notes (read-only)

Method: code reading plus read-only pandas queries on `data/processed/real_extraction/`. Nothing was modified. Line numbers refer to the working tree as of 2026-10-08. The tree has many uncommitted changes (`git status` shows `M` on most artifacts).

Note: `run_logged.py` is at `src/extraction/run_logged.py`, not at the repo root.

---

## 1. Drug filter

**Can the extractors run on one drug's rows? Yes, with no structural obstacle.** Nothing in the scorer depends on multiple drugs. `drug_name` is already a column in `gdsc_processed.parquet`.

Where rows are loaded:
- GA: `src/extraction/ga.py:100` has `df = pd.read_parquet(data_dir / "gdsc_processed.parquet")` inside `load_condition_data` (def at line 91). Train and val are both derived from this `df` (`_load_split`, lines 119-126), so one filter covers both.
- NSGA-II: `src/extraction/nsga.py:74` has the same read inside `load_condition_data` (def at line 72). It then filters by split at line 78 and merges predictions at lines 80-86.
- Callers: `src/extraction/run_logged.py:183` (GA) and `:260` (NSGA) call these loaders. Both use `DATA_DIR` (`run_logged.py:65`).
- Scorer: `src/extraction/common.py` holds only the encoding, operators and `predict_with_rule`. The actual scoring is `_score_genotype`/`make_scorer` in `ga.py:169-236` and `nsga.py`. Both work on whatever `df` they are handed. Coverage is `train_mask.mean()` (`ga.py:203`), compared with `MIN_COVERAGE = 0.02` (`run_logged.py:59`).

**Smallest change.** Add one line right after each read (`ga.py:100`, `nsga.py:74`):
```python
df = df[df["drug_name"] == drug_name].reset_index(drop=True)
```
`drug_name` could come from a new `--drug` CLI argument (`run_logged.py:356-365`) or from an environment variable, and be passed into `load_condition_data`. The later `df["row_id"].isin(...)` split filter and the `merge` with predictions then work unchanged.

Side effects to keep in mind (not changed):
- Without `--tag`, `run_logged.py` writes the normal pool name and runs `compare_pools` (lines 404-408). A single-drug run would need `--tag` and its own `--out-dir`, or it could overwrite the `{ga,nsga}_{baseline,sam}` outputs.
- `ensure_response_classes` (`run_logged.py:148-155`) reads the whole parquet. It is harmless but not drug-filtered.
- The 2% coverage floor would apply to about 650-865 rows per drug, so a rule must cover at least about 13-17 rows.
- The `response_class` and `class_pred` labels use a per-drug median threshold (`meta.json`: `per_drug_median_train`). Single-drug classes are therefore roughly balanced by construction.

---

## 2. Adam + weight-decay control model

A checkpoint exists. **No predictions exist for it.**

| File | Size (bytes) | Modified |
|---|---|---|
| `artifacts/sam_investigation/adam_wd_control/model.pt` | 4,414,229 | 2026-09-13 02:47 |
| `.../adam_wd_control/meta.json` | 1,667 | 2026-09-27 20:39 |
| `.../adam_wd_control_scorecard.json` | 1,519 | 2026-09-27 20:39 |
| `.../adam_wd_control_history.json` | 30,219 | 2026-09-27 20:39 |
| `.../adam_wd_control_history.jsonl` | 23,110 | 2026-09-27 20:39 |
| `.../adam_wd_control_progress.json` | 274 | 2026-09-27 20:39 |
| `.../adam_wd_control.log` | 31,153 | 2026-09-27 20:39 |
| `.../_partial_run_2026-09-10_backup/` (log, history.jsonl, progress.json, meta.json; **no model.pt**) | 21,710 / 14,725 / 271 / 1,407 | 2026-09-27 20:39 |
| `src/investigations/stage4b_adam_wd_control.py` (producer script) | not measured | not measured |

Modified dates are from the 2026-09-27 copy or checkout. Only `model.pt` keeps its original training-time stamp.

Findings:
- **The gap is not 0.2582 for the saved model.** The saved `meta.json`, scorecard and journal (`docs/investigations/sam-generalisation/journal.md:740-750`) give **gap = 0.2729**, with 187 epochs, a rerun started 2026-09-11, runtime 19,158 s, `produced_by: src.investigations.stage4b_adam_wd_control`, and git commit 6f13fb6.
- **0.2582 and epoch 157 come from the original 2026-09-05 run**, which saved metrics only. That run is described in `journal.md:395-398` and `SAM_Investigation_Full_Report.md:179, 184`. `meta.json` ("run_note") says the rerun was made to add a loadable `model.pt`. The 0.2582 run has no surviving weights.
- The 0.2582 run's own scorecard is no longer on disk. It was overwritten by the rerun, which now says 0.2729. The only record of 0.2582 is prose.
- **Predictions:** there are none for this model. Searching the repo for `adam_wd` and `wd_control` finds no parquet. `artifacts/predictions/` and `data/processed/real_extraction/predictions/` contain only `baseline_*` and `sam_*` (train/val/test). `data/processed/real_extraction_inputs_{baseline,sam_final}.parquet` are also baseline and SAM only.
- Predictions could be generated from `model.pt`, using the same path that made `sam_final` (`src/data/prepare_real_extraction_data.py`, `_model_from_checkpoint`). That is a code path, not a file.
- Other `.pt` files, for reference: `artifacts/checkpoints/{baseline,sam}/model.pt` (4.4 MB each, 2026-09-27) and `artifacts/sam_investigation/confirmatory/model.pt` (2026-09-13 02:44). The Stage 5 SAM checkpoint is `confirmatory` (gap 0.0485 per `real_extraction/meta.json`), not `artifacts/checkpoints/sam`.

---

## 3. Drug count (413 vs 379)

Drugs are dropped by a deliberate shortlist filter in the data-prep step. They are not lost in any join.
- Code: `src/data/prepare_real_extraction_data.py:82-89`, `build_shortlist()`. It keeps drugs whose tested-cell-line count is >= `max(min_samples_per_drug, low_sanity_floor)`. It is applied at lines 154-159: `df = full_df[full_df["drug_name"].isin(shortlist)]`.
- Config: `configs/data_prep.yaml`, `drug_shortlist.min_samples_per_drug: 500` and `low_sanity_floor: 50`.
- Recorded in `data/processed/real_extraction/meta.json`: `n_drugs_total: 413`, `n_drugs_shortlisted: 379`, `min_samples_per_drug: 500`, and `PROVISIONAL: true`.
- Rationale: `artifacts/data_prep/report.md:155-163` says 379/413 drugs have >=500 total samples. A threshold of 600 would keep 340/413. It recommends 500.
- **Caveat:** `configs/data_prep.yaml` and `meta.json` both flag the 500 threshold and the discretisation mode as PROVISIONAL, pending the project lead's confirmation.
- The 34 dropped drugs are not named in the files I read. I did not compute which they are, though the shortlist and full lists are both available. Drugs that remain have 645-865 rows each (mean ~811).
- The 413 figure persists in docs (`BASELINE_README.md:39`, journal line 197) because the model and baseline were trained on all 413 drugs (`drug_fingerprint_matrix` is 413x1024). Only the extraction dataset is filtered to 379.

---

## 4. Positive controls

`gdsc_processed.parquet` has 307,293 rows, 379 drugs and 866 cell lines.

**Encoding caveat:** mutation features are not stored as 0/1. They are categorical `{'Low','High'}` (`feature_map.json` gives `partition_edges: None` for them). I took `High` as "mutated" (value 1). The raw `GDSC_mutation_input.csv` is 0/1, and its mutant counts (BRAF 336, EGFR 382, KRAS 187, TP53 625, PIK3CA 253, BRCA1 291, BRCA2 317 across 878 cell lines) closely match the processed High counts across 866 cell lines. That supports the mapping, but I did not trace the discretisation code itself.

**Mutation features present** in `feature_map.json` (572 features): `BRAF_mut`, `EGFR_mut`, `KRAS_mut`, `TP53_mut`, `PIK3CA_mut`, `BRCA1_mut`, `BRCA2_mut`. All seven are present, with real gene symbols.

**Drugs present.** Absent: Vemurafenib. The full list of 379 drug names is in the appendix at the end of this file.

| Drug | Class | Cell lines | Matching mutation | Cells with matching mutation = High |
|---|---|---|---|---|
| PLX-4720 | BRAF | 864 | BRAF_mut | 333 |
| Dabrafenib | BRAF | 862 | BRAF_mut | 333 |
| Vemurafenib | BRAF | **absent** | | |
| Erlotinib | EGFR | 860 | EGFR_mut | 376 |
| Gefitinib | EGFR | 863 | EGFR_mut | 376 |
| Afatinib | EGFR | 864 | EGFR_mut | 377 |
| Lapatinib | EGFR | 861 | EGFR_mut | 376 |
| Olaparib | PARP | 864 | BRCA1_mut / BRCA2_mut | 288 / 312 |
| Talazoparib | PARP | 864 | BRCA1_mut / BRCA2_mut | 288 / 312 |
| Selumetinib | MEK | 864 | BRAF_mut (KRAS_mut = 187) | 333 |
| Trametinib | MEK | 864 | BRAF_mut (KRAS_mut = 187) | 333 |
| Refametinib | MEK | 864 | BRAF_mut (KRAS_mut = 187) | 333 |

Whole-dataset counts of High cell lines (out of 866): BRAF 333, EGFR 378, KRAS 187, TP53 618, PIK3CA 252, BRCA1 289, BRCA2 313.

**Concern:** these prevalences are very high. BRAF is mutated in 38% of lines and EGFR in 44%, whereas hotspot activating mutations are far rarer in GDSC. The `_mut` feature is probably "any mutation in the gene", not "activating hotspot". This matters for positive-control logic because `BRAF_mut = High` includes passenger mutations. This is an inference from the numbers, not something I verified in code.

---

## 5. Training time

The record is uneven: runtime is saved in metadata for the investigation runs, but I found no explicit runtime for the baseline.

| Model | Record found | Source |
|---|---|---|
| **Adam baseline** (`artifacts/checkpoints/baseline`, 90 epochs) | **No runtime stored.** `meta.json` has no `runtime_s`. `baseline_training_history.json` holds only per-epoch losses (no timestamps). The only figure is the vague "~5 hours" for a full SAM run in `CLAUDE.md:85`. The journal calls this the "~5 h documented baseline" (`journal.md:314`), which is not backed by a stored number. BASELINE_README says "~15 minutes to a few hours" (line 159). | `artifacts/checkpoints/baseline/meta.json`, `CLAUDE.md:85`, `journal.md:314`, `BASELINE_README.md:159` |
| **SAM, confirmatory (final)** | **Two runs.** Original (2026-09-05): 77,248 s (~21.5 h), 250-epoch cap, not converged, slowed by CPU contention. Rerun (2026-09-10, producing the current `model.pt`): **48,400 s (~13.4 h)**, `runtime_s` in the confirmatory `meta.json` and scorecard. | `confirmatory/meta.json:42`, `journal.md:314, 538` |
| **SAM, `checkpoints/sam`** (rho sweep series, 200-epoch runs) | No runtime in `meta.json` or the registry (`grep runtime_s` found none). | `artifacts/checkpoints/sam/meta.json`, `artifacts/reports/sam_runs_registry.json` |
| **Adam+WD control** | Rerun (the saved model): **19,157.6 s (~5.3 h)**, 187 epochs, `runtime_s` in `meta.json` and the scorecard. Original run (the 0.2582 one): 23,823 s (~6.6 h), described in the journal only, with no file. | `adam_wd_control/meta.json`, `journal.md:395-396, 746-747` |
| **Stage 3 sweep** (16 configs, 10% subsample) | ~2h20 in total. SAM epoch ~4.8 s on subsample, ~90 s at full scale. | `journal.md:107, 127` |
| **GA/NSGA-II extraction** | Per-run seconds: GA baseline 30 runs, 1,103 s total (mean 36.8 s); GA SAM 571 s (19.0 s); NSGA baseline 50 s (1.7 s); NSGA SAM 64 s (2.1 s). Logs also print a total wall time. | `outputs/logs/*_runs.csv` (`seconds` column), `*_rerun.log` |

I found no wandb or tensorboard directories and no shell history in the repo. Notebooks exist (`notebooks/01-06`), but I did not search them for timing.

**Fairness caveat:** all of these runtimes were affected by host CPU contention, and the journal says so itself, so they are not comparable across models. Epoch counts also differ (90 for the baseline, 187 for Adam+WD, 250 for SAM).

---

## 6. Stability metric

Source: `src/evaluation/stage5_utils.py`.

- `MIN_ROWS_PER_CELL = 10` (line 25).
- **`rule_stability` (lines 85-101):**
  - It applies the rule mask to the split.
  - It groups the covered rows by `cell_name` (default `group_col`).
  - It skips any cell line with `len(grp) < min_rows`, i.e. fewer than 10 covered rows.
  - If fewer than 2 groups remain, it returns `(nan, n_groups, [])`.
  - Otherwise it returns `metrics.stability()` (`src/common/metrics.py:113-148`). That is the variance of per-cell macro-F1 fidelity, and it raises `ValueError` for fewer than 2 subgroups.
- **`stability_agreement` (lines 104-127):** the same grouping and the same minimum of 10 rows per group. It returns `np.var` of the per-cell agreement rate, and returns `nan` if fewer than 2 groups.
- **Minimum required:** at least 2 cell lines that each have at least 10 covered rows in the evaluated split. This is applied in the test split, so each rule needs at least 20 covered test rows concentrated in at least 2 cell lines.
- **Current data:** the mean is ~355 drug rows per cell line over 866 cell lines (`gdsc_processed.parquet`), so full-data stability works in practice.
- **Single-drug data:** **No, it does not work.** Each cell line has at most one row per drug (the dataset has unique cell/drug pairs; Olaparib has 864 rows and 864 cell lines). The `>= 10` rows-per-cell condition can never be met, so every rule gets `nan` and `n_groups = 0`. The `nan` then flows into PROMETHEE and any stability-based criterion, which may break or drop that criterion. To use stability on single-drug data, `group_col` would have to change to something with many rows per group (for example tissue or cancer type). The parquet columns I listed are `row_id, cell_name, cell_idx, drug_name, drug_idx, ln_ic50, split` plus 572 features, and I did not see a tissue column. So there may be no suitable grouping variable in this dataset without external annotation.

---

## Summary table

| Question | Answer | Confidence | Evidence |
|---|---|---|---|
| 1. Single-drug filter | Yes. Add `df = df[df.drug_name == X]` after the parquet read in `ga.py:100` and `nsga.py:74` (plus a CLI arg). Use `--tag` and a separate `--out-dir` to avoid overwriting outputs. | High | `ga.py:91-126`, `nsga.py:72-95`, `run_logged.py:183, 260, 356-408` |
| 2. Adam+WD model | Checkpoint exists (`adam_wd_control/model.pt`), but it is the rerun with **gap 0.2729**. The 0.2582 run (epoch 157) has no surviving weights. **No predictions exist** for it. | High (checkpoint and no predictions); medium on whether 0.2582 is the version you want | `adam_wd_control/meta.json`, scorecard, `journal.md:395-398, 740-750`, Full Report `:179` |
| 3. 413 vs 379 | 34 drugs are dropped deliberately by a `min_samples_per_drug = 500` shortlist (provisional). Not a join loss. | High | `prepare_real_extraction_data.py:82-89, 154-159`, `configs/data_prep.yaml`, `real_extraction/meta.json`, `data_prep/report.md:155-163` |
| 4. Positive controls | All 7 mutation features exist, stored as Low/High (High = mutated). Vemurafenib is absent; all other requested drugs are present. Counts are in the table above (e.g. PLX-4720: 864 cells, 333 BRAF High). Mutation prevalences look implausibly high for activating hotspots (BRAF 38%, EGFR 44%). | High on counts; medium on "High = 1" and the prevalence interpretation | `feature_map.json`, parquet query, raw `GDSC_mutation_input.csv` |
| 5. Training time | Stored `runtime_s`: SAM confirmatory 48,400 s (rerun; original 77,248 s); Adam+WD 19,158 s (original 23,823 s). **No runtime for the Adam baseline.** All runs affected by CPU contention. | High for present values; medium that nothing is missing in unsearched notebooks | `confirmatory/meta.json`, `adam_wd_control/meta.json`, `journal.md:314, 395, 538, 746`, `outputs/logs/*_runs.csv` |
| 6. Stability | Variance of per-cell fidelity (or agreement rate). It needs at least 2 cell lines with at least 10 covered rows each. With one row per cell line (single-drug) it always returns NaN. | High | `stage5_utils.py:25, 85-127`, `metrics.py:113-148`, parquet group sizes |

---

## Appendix: drugs in `gdsc_processed.parquet` (379)

(5Z)-7-Oxozeaenol, 5-Fluorouracil, 5-azacytidine, A-366, A-83-01, ABT737, ACY-1215, AGI-5198, AGI-6780, AGK2, AICA Ribonucleotide, AKT inhibitor VIII, AMG-319, AR-42, ARRY-520, AS601245, AS605240, AST-1306, AT-7519, AT13148, AT7867, AZ20, AZ6102, AZ960, AZD1208, AZD1332, AZD1480, AZD2014, AZD3514, AZD3759, AZD4547, AZD4877, AZD5153, AZD5363, AZD5438, AZD5582, AZD5991, AZD6094, AZD6482, AZD6738, AZD7762, AZD8055, AZD8186, AZD8835, AZD8931, Acetalax, Afatinib, Afuresertib, Alectinib, Alisertib, Alpelisib, Amuvatinib, Apitolisib, Ara-G, Avagacestat, Axitinib, BAM7, BAY-61-3606, BEN, BI-2536, BIBF-1120, BIBR-1532, BIX02189, BMS-345541, BMS-536924, BMS-754807, BPTES, BX-912, BX795, Belinostat, Bexarotene, Bicalutamide, Bleomycin, Bortezomib, Bosutinib, Bromosporine, Bryostatin 1, Buparlisib, C-75, CAY10566, CAY10603, CCT-018159, CCT007093, CCT245232, CD532, CHIR-99021, CI-1033, CI-1040, CP466722, CP724714, CPI-613, CPI-637, CRT0105950, CUDC-101, CX-5461, CZC24832, Cabozantinib, Camptothecin, Capivasertib, Cediranib, Cisplatin, Crizotinib, Cyclophosphamide, Cytarabine, DMOG, Dabrafenib, Dacarbazine, Dacinostat, Dactinomycin, Dactolisib, Daporinad, Dasatinib, Dihydrorotenone, Dinaciclib, Docetaxel, Doramapimod, Doxorubicin, EHT-1864, EPZ004777, EPZ5676, ETP-45835, Elephantin, Elesclomol, Embelin, Entinostat, Entospletinib, Enzastaurin, Epirubicin, Epothilone B, Erlotinib, Etoposide, FH535, FMK, FR-180204, FTI-277, FTY-720, Fedratinib, Flavopiridol, Fludarabine, Foretinib, Fulvestrant, GDC0810, GNE-317, GSK-J4, GSK-LSD1, GSK1059615, GSK1070916, GSK1904529A, GSK2110183B, GSK2578215A, GSK2606414, GSK269962A, GSK2801, GSK2830371, GSK343, GSK429286A, GSK591, GSK650394, GSK690693, GW-2580, GW441756, Gallibiscoquinazole, Gefitinib, Gemcitabine, HG6-64-1, I-BET-151, I-BET-762, I-BRD9, I-CBP112, IC-87114, ICL-SIRT078, IMD-0354, IOX2, IPA-3, IWP-2, Ibrutinib, Idelalisib, Ipatasertib, Irinotecan, Ispinesib Mesylate, JNJ38877605, JNK Inhibitor VIII, JQ1, JQ12, JW-7-24-1, KIN001-244, KIN001-266, KIN001-270, KU-55933, KU-60019, Kobe2602, L-Oxonoreleagnine, LCL161, LDN-193189, LFM-A13, LGK974, LIMK1 inhibitor BMS4, LJI308, LMP744, LY2109761, Lapatinib, Leflunomide, Lenalidomide, Lestaurtinib, Linifanib, Linsitinib, Luminespib, MG-132, MIM1, MIRA-1, MK-1775, MK-2206, MK-8776, ML323, MN-64, Masitinib, Methotrexate, Midostaurin, Mitomycin-C, Mitoxantrone, Motesanib, Mycophenolic acid, N-acetyl cysteine, NG-25, NPK76-II-72-1, NSC-207895, NSC-87877, NSC319726, NU7441, NVP-ADW742, NVP-BHG712, Navitoclax, Nelarabine, Nilotinib, Niraparib, OF-1, OSI-027, OSI-930, OSU-03012, OTX015, Obatoclax Mesylate, Olaparib, Omipalisib, Osimertinib, Oxaliplatin, P22077, PAC-1, PCI-34051, PD0325901, PD173074, PF-00299804, PF-4708671, PF-562271, PFI-1, PFI-3, PFI3, PHA-793887, PI-103, PIK-93, PLX-4720, POMHEX, PRIMA-1MET, PRT062607, Paclitaxel, Palbociclib, Panobinostat, Pazopanib, Pelitinib, Pemetrexed, Pevonedistat, Phenformin, Pictilisib, Pilaralisib, Piperlongumine, Podophyllotoxin bromide, Ponatinib, Pyridostatin, QL-X-138, QL-XI-92, QL-XII-47, QS11, Quizartinib, RO-3306, RU-SKI 43, RVX-208, Rapamycin, Refametinib, Ribociclib, Romidepsin, Rucaparib, Ruxolitinib, SB216763, SB505124, SB52334, SB590885, SCH772984, SGC-CBP30, SGC0946, SL0101, SN-38, SNX-2112, STF-62247, SU11274, Sabutoclax, Sapitinib, Savolitinib, Schweinfurthin A, Selisistat, Selumetinib, Sepantronium bromide, Serdemetan, Shikonin, Sinularin, Sorafenib, Sphingosine Kinase 1 Inhibitor II, Staurosporine, T0901317, TAK-715, TPCA-1, TW 37, TWS119, Talazoparib, Tamoxifen, Tanespimycin, Taselisib, Telomerase Inhibitor IX, Temozolomide, Temsirolimus, Teniposide, Tenovin-6, Thapsigargin, Tipifarnib, Tivozanib, Topotecan, Torin 2, Trametinib, Tretinoin, Trichostatin A, Tubastatin A, UMI-77, UNC0379, UNC0638, UNC0642, UNC1215, Ulixertinib, Uprosertib, VE-822, VE821, VNLG/124, VX-11e, VX-702, Veliparib, Venetoclax, Vinblastine, Vinorelbine, Vismodegib, Vorinostat, Voxtalisib, WEHI-539, WHI-P97, WIKI4, WYE-125132, WZ3105, WZ4003, Wee1 Inhibitor, Wnt-C59, XAV939, XMD13-2, XMD14-99, Y-39983, YK-4-279, YM201636, ZM447439, ZSTK474, Zibotentan, Zoledronate, alpha-lipoic acid, glutathione, kb NB 142-70
