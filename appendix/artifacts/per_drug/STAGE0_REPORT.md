# Stage 0 report (per-drug pilot)

Branch `per-drug-pilot`, created from `afa1d6e62e575b7f1381f78a15876141a128f8c5` (HEAD of `feat/evaluation-pipeline`).
The working tree already had ~195 modified files before this work; they are line-ending (CRLF vs LF) changes only
(`git diff --ignore-space-at-eol` is empty). They were left alone and are not committed here. Only files listed in the commit were touched.

## B. Gap numbers (recomputed with one code path)

`scripts/recompute_gaps.py` → `artifacts/per_drug/data/gap_recompute.json`. Same code for all three: `src.model.baseline.run_epoch`
(eval mode, no grad), MSE loss, gap = val loss − train loss, on `artifacts/data` (all 413 drugs, the rows the networks were trained on).

| Checkpoint | Epochs | Stopping rule | gap recomputed (val−train) | gap in `meta.json` | train / val / test loss | gap on the 379-drug extraction subset |
|---|---|---|---|---|---|---|
| Adam baseline `artifacts/checkpoints/baseline` | 90 | reverse early stopping (`baseline.py`) | 0.33694 | 0.33694 | 0.7373 / 1.0743 / 1.0741 | 0.33313 |
| Adam + WD `adam_wd_control/model.pt` | 187 | val-loss plateau, patience 30 (`stage4b_adam_wd_control`, git 6f13fb6, rerun started 2026-09-11) | **0.27291** | 0.27291 | 1.0027 / 1.2756 / 1.2717 | 0.26995 |
| SAM ρ=0.5 + WD `confirmatory/model.pt` | 250 (hit the 250 cap) | val-loss plateau, patience 30, max 250 (`stage4_confirmatory`, git cf6c950; runtime 48,400 s) | **0.04850** | 0.04850 | 1.3548 / 1.4033 / 1.4038 | 0.04818 |

Where the other numbers come from (not picked between; the saved checkpoints reproduce 0.2729 and 0.0485):
- **0.2582** = the original 2026-09-05 Adam+WD run, plateau-stopped at epoch 157 (`SAM_Investigation_Full_Report.md:179, 184`; `journal.md` ~395–398). That run saved metrics only; its weights and scorecard no longer exist (the 2026-09-11 rerun overwrote the scorecard), so the 0.2582 is prose-only and cannot be recomputed.
- **0.0517** = the original Stage 4 SAM run (2026-09-04/05, 250-epoch cap, not converged, 77,248 s; metrics only), quoted in `adam_wd_control_scorecard.json` (`comparison.sam_rho05_wd1e3_stage4`), `journal.md:318`, `results_summary.md`. `writeup_audit.md:181` already lists 0.05173 vs 0.04850 (−6.25 %).
- Reading the table: the SAM network has a larger val loss (1.40) than the Adam networks (1.07, 1.28), and a much larger train loss (1.35 vs 0.74 / 1.00).

## C. Adam + weight-decay predictions

`scripts/make_adam_wd_predictions.py` → `artifacts/per_drug/data/predictions/adam_wd_{train,val,test}.parquet` (gitignored by `*.parquet`; regenerate with the script) and `adam_wd_verification.json`.
Same recipe as `prepare_real_extraction_data` (`_model_from_checkpoint` → `run_inference` → `discretise` with the per-drug train-median thresholds from `real_extraction/meta.json`).
- Recipe check: re-deriving baseline and SAM (confirmatory) predictions with it reproduces the shipped files exactly (0 class_pred mismatches, max |Δ ln_ic50_pred| = 0.0, all splits).
- Schema check vs `sam_*.parquet`: identical columns (`row_id`, `ln_ic50_pred` float32, `class_pred` str), dtypes, row_id sets; 245,845 / 30,720 / 30,728 rows.
- Gap reproduces: 0.27291 (table above), the checkpoint's own recorded value.

## D. Mutation flag (`*_mut`)

Trace: raw `data/deepccds_precomputed/GDSC_mutation_input.csv` (878 cell lines × 534 gene columns + `cell_idx`, `cell_line`; values 0/1; from the DeepCCDS repo,
https://github.com/hanjunwei-lab/DeepCCDS, per CLAUDE.md) → `src/data/precomputed_inputs.py:load_mutation_matrix` → `src/data/preprocess.py:174–183`
(`mutation_labels = np.where(mutation_matrix == 1, "High", "Low")`, columns named `<GENE>_mut`, categorical [Low, High]; commit f174991) → `gdsc_processed.parquet`.
So: **High = raw value 1, Low = raw value 0**, nothing else. The raw file is in the repo folder (gitignored) but has no codebook, and the DeepCCDS repository is not in this repo.
What I can and cannot say:
- The project's own write-up describes the matrix as "1 = non-synonymous mutation in a driver gene" (`SAM_Investigation_Full_Report.md:35`). I could not verify that against any source in the repo; no script or document says how DeepCCDS made the calls, or whether "likely-functional" filtering was applied.
- Observed frequencies (raw file, 878 lines): BRAF 38.3 % (336), EGFR 43.5 % (382), KRAS 21.3 % (187), TP53 71.2 % (625), BRCA1 33.1 % (291), BRCA2 36.1 % (317); mean 117 of 534 genes flagged per cell line (median 96). These are high for hotspot/activating calls, which suggests a broad definition, but that is an inference from the counts, not something established. **Whether `_mut` = "any mutation" or "only likely-functional" is therefore unresolved; I stopped on this item as instructed.** This matters for how the positive-control hits are interpreted.

## E. Pilot drugs

`scripts/select_pilot_drugs.py` → `artifacts/per_drug/pilot_drugs.json` (+ `pilot_drug_val_f1_all.csv`). 379 candidates. Per-drug validation macro-F1 = network val split rows (66–93 rows/drug for the top ones), sklearn macro-F1 of class_pred vs the true class, mean of 3 arms.
- Best-predicted 5: SCH772984 (0.909), PD0325901 (0.884), QL-X-138 (0.877), Irinotecan (0.873), Lestaurtinib (0.870).
- Positive controls: Erlotinib, PLX-4720, Olaparib, Selumetinib (all present, none overlaps the best-predicted five).
- Random 5 (seed 700, from the 370 remaining): CHIR-99021, Capivasertib, Nelarabine, XMD13-2, Daporinad.
- **Final count: 14 drugs, no drug in two groups.**

## G. Code and tests

- `ga.py`, `nsga.py`: `--drug` (filter right after the parquet read; unknown drug raises) and `--min-cover-n` (fixed row floor, overrides `--min-coverage`; default unchanged, so the global runs reproduce). New `src/extraction/coverage.py` (`coverage_floor`, exact; tested for every count). Files converted from CRLF to LF so the diff shows only the real change.
- `run_logged.py`: **`--tag` and `--out-dir` already existed**; added `--drug` and `--min-cover-n` (and a drug slug in output names; the shipped-pool comparison is skipped when `--drug` is used).
- New `src/per_drug/` (config, data/fold logic, cell runner, held-out scoring + stability, resumable driver). The old stability function was not touched.
- `tests/test_per_drug.py` (synthetic data only): (i) filter == manual filter, (ii) no cell line in both extraction and held-out part, every cell line held out exactly once, (iii) same seed → identical output (both extractors), (iv) 2-generation / 2-seed dry run for both extractors incl. manifest, resume and refusal to mix designs; plus coverage-floor exactness, held-out scoring vs manual, Jaccard/Kuncheva, control hits. 10 passed.
- A real-data dry run (Erlotinib, fold 0, 3 arms × 2 extractors, 2 generations × 2 seeds) completed; used for correctness only.
- Not mine, found on the way: `tests/test_ga.py` fails at import (`rule_mask` no longer exported by `src/extraction/common.py`); `tests/test_per_drug.py` does not depend on it.
- Environment: this machine reports 3 GB RAM total (CLAUDE.md says ~8 GB) and 8 logical CPUs.
