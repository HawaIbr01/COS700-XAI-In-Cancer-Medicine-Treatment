# SAM Generalisation-Gap Investigation — Consolidated Results Table

Built for the post-rerun write-up audit (2026-09-13). Every number below is
read directly from a committed JSON artifact — see the **Source** column for
exact provenance. No retraining was done to build this table.

All runs use the identical full-data DeepCCDS split (`config_hash=a2070aedcc19`,
`split_hash=dc2241cf73c6`; train 255,628 / val 31,963 / test 31,952), except
where noted.

| Condition | Gen. gap | Val R² | Val RMSE | Train loss | Val loss | Test loss | Epochs | Stopping rule | Checkpoint (loadable?) |
|---|---|---|---|---|---|---|---|---|---|
| Adam baseline (WD=0) | 0.3369 | 0.8565 | 1.0299 | 0.7373 | 1.0743 | 1.0741 | 90 | paired: val-plateau (patience=30, min_delta=1e-4) **and** train-loss-still-falling (min_consecutive=10) — the "reverse-early-stopping" rule built for a deliberately-overfitting baseline | `artifacts/checkpoints/baseline/model.pt` — **yes** (verified, loads cleanly) |
| SAM ρ=0.01, WD=0 | 0.3656 | 0.8577 | 1.0255 | 0.6977 | 1.0632 | 1.0661 | 122 | same paired rule as Adam baseline | none saved — **n/a** |
| SAM ρ=0.05, WD=0 | 0.3661 | 0.8593 | 1.0197 | 0.6840 | 1.0501 | 1.0585 | 162 | same paired rule | none saved — **n/a** |
| SAM ρ=0.1, WD=0 | 0.3600 | 0.8587 | 1.0218 | 0.6936 | 1.0536 | 1.0701 | 199 | same paired rule | none saved — **n/a** |
| SAM ρ=0.2, WD=0 | 0.3122 | 0.8602 | 1.0165 | 0.7297 | 1.0419 | 1.0394 | 215 | same paired rule | `artifacts/checkpoints/sam/model.pt` — **yes** (verified, loads cleanly) |
| Adam + WD=1e-3 control (rerun, Stage 4e) | 0.2729 | 0.8616 | 1.0111 | 1.0027 | 1.2756 | 1.2717 | 187 (converged) | val-plateau only (patience=30, min_delta=1e-4) — matched to the SAM confirmatory rule, deliberately **not** the paired rule above (see journal Stage 4b) | `artifacts/sam_investigation/adam_wd_control/model.pt` — **meta.json says it exists; file is absent from this working tree — see Item 9** |
| SAM ρ=0.5, WD=1e-3 (confirmatory rerun, Stage 4c — "this rerun") | 0.0485 | 0.8444 | 1.0722 | 1.3548 | 1.4033 | 1.4038 | 250 (hit max-epoch cap; `converged: false`, val loss flat/non-widening from ~epoch 150) | val-plateau only, MAX_EPOCHS=250 safety cap | `artifacts/sam_investigation/confirmatory/model.pt` — **meta.json says it exists; file is absent from this working tree — see Item 9** |

**Original (pre-rerun) SAM ρ=0.5+WD confirmatory numbers**, for reference (see
Item 6 for the reproducibility comparison): gap=0.0517, val R²=0.8450, val
RMSE=1.0703, epoch 250 (also non-converged). These were the numbers in
`artifacts/sam_investigation/confirmatory/confirmatory_scorecard.json` before
the Stage 4c rerun overwrote that file with the rerun's own numbers (git
commit `43707e6` → `61d85eb`); they only survive in git history and in the
Stage 4 journal narrative now, not as a live JSON file.

## Sources

- Adam baseline: `artifacts/reports/baseline_gap_scorecard.json`
- SAM ρ=0.01/0.05/0.1/0.2: `artifacts/reports/sam_gap_scorecard_rho_0p{01,05,1,2}.json` (`sam_gap_scorecard.json` is an identical copy of the ρ=0.2 file — checked numerically equal)
- Adam+WD control: `artifacts/sam_investigation/adam_wd_control/adam_wd_control_scorecard.json` + `meta.json`
- SAM ρ=0.5+WD confirmatory rerun: `artifacts/sam_investigation/confirmatory/confirmatory_scorecard.json` + `meta.json`
- Checkpoint existence verified by direct filesystem check (`ls`) and, for the two loadable ones, by actually loading them in `src/investigations/write_up_loss_breakdown.py` (item 3).

## Notes

- Rows for ρ=0.01/0.05/0.1 never had a `model.pt` saved — only the final ρ=0.2
  configuration was checkpointed for the original Adam-vs-SAM handoff
  (`docs/CONTRACTS.md` only specifies `checkpoints/{baseline,sam}`; the sweep
  points were diagnostic only). This is expected, not a gap.
- The Adam+WD and SAM ρ=0.5+WD checkpoints are a different situation: both were
  explicitly built (Stage 4c/4e) for handoff to the rule-extraction arms, are
  documented in the journal as saved and verified, and are gitignored by
  design — but the files themselves are not present on this machine right now.
  See `writeup_audit.md` Item 9.
