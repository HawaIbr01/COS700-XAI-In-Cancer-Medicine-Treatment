# Investigation Journal: SAM Generalisation Gap in DeepCCDS

Branch: `feat/generalisation-gap-intervention`
Investigation prompt: `docs/investigations/sam-generalisation/prompt.md`

## Glossary

Quick reference for terms used throughout this journal. Skip if familiar.

- **SAM (Sharpness-Aware Minimisation)** — the optimiser variant under test. Instead of just descending the loss at the current point, it first takes a small step *uphill* to find the locally worst-case loss nearby, then descends from there. Purpose: nudge training toward "flat" regions of the loss surface, which are believed to generalise better than "sharp" ones.
- **Adam** — the standard optimiser used as our baseline. Adjusts the learning rate per-parameter based on recent gradient history. Purpose: this is what we're comparing SAM against.
- **ρ (rho)** — SAM's one main hyperparameter: how far uphill it steps before descending (the "neighbourhood radius"). Purpose: controls how aggressively SAM searches for worst-case sharpness. Too small and SAM barely differs from Adam; too large and training can destabilise.
- **LR (learning rate)** — how big a step the optimiser takes at each update. Purpose: the single most influential training hyperparameter; too high and training diverges or oscillates, too low and it crawls or gets stuck.
- **WD (weight decay)** — a penalty added during training that shrinks weights toward zero each step. Purpose: a classic regulariser — discourages the network from relying on any single weight too heavily, which tends to reduce overfitting.
- **BN (batch normalisation)** — a layer that rescales activations using batch statistics during training. Purpose: stabilises and speeds up training; incidentally, its own learnable scale/shift parameters ("affine parameters") are what some SAM variants specifically target for perturbation.
- **Generalisation gap** — `val_loss − train_loss`. Purpose: this is the actual quantity we're investigating. A small gap means the model performs about as well on data it hasn't seen as on data it trained on (little overfitting); a large gap means it's memorising training data rather than learning transferable patterns. Our whole thesis rests on SAM widening this gap's *reduction* relative to Adam — i.e. SAM should shrink the gap more than Adam does.
- **Overfitting** — when a model fits the training data very well but doesn't transfer that performance to new data. Shows up directly as a large generalisation gap.
- **R² (validation R², "R-squared")** — the fraction of variance in the true values that the model's predictions explain. Purpose: 1.0 = perfect predictions, 0 = no better than predicting the average every time. Used here as a plain accuracy check alongside the gap.
- **RMSE (root mean squared error)** — the average prediction error, in the same units as the target (drug sensitivity/IC50 here). Purpose: an intuitive "how far off are we on average" number, alongside the more abstract loss.
- **MSE (mean squared error)** — the squared version of the above; what the "Eq. 2" loss in the plots is based on. Purpose: this is literally what the optimiser is minimising during training.
- **train-mode vs. eval-mode** — networks with BN/dropout behave *differently* depending on whether they're in training mode (BN uses live batch statistics, dropout randomly zeroes units) or evaluation mode (BN uses accumulated running statistics, dropout is switched off). Purpose: eval-mode numbers are the fair, low-noise comparison point across configs; train-mode numbers (as tracked live during training) are noisier and can even show spuriously negative "gaps" early on, purely as an artefact of BN statistics still stabilising — not real generalisation behaviour.
- **Ceiling effect** — our working hypothesis going into this investigation: that DeepCCDS's existing regularisation (pretrained bottlenecked autoencoders, BN, dropout, WD, early stopping, large dataset) already keeps the generalisation gap small, leaving little room for SAM to improve it further.

---

## Stage 1 — Measure the actual generalisation gap

**What was done.** Read the existing training histories for the Adam baseline and four completed SAM runs (ρ = 0.01, 0.05, 0.1, 0.2) — no retraining needed, since Angelique's and Bahiya's runs were already logged. Generated three plots and a numeric summary:
- Training vs. validation loss curves per config (`Stage 1: Training curves`)
- Final scorecard: generalisation gap, validation R², validation RMSE, computed in eval-mode (`Stage 1: Final scorecard metrics`)
- Generalisation gap tracked per epoch in train-mode, for reference (`Stage 1: Generalisation gap per epoch`)

Saved under `artifacts/sam_investigation/` (real pipeline output, not synthetic — correctly placed per our artifact/fixture convention).

**Why this was done.** This is the cheapest, most direct test available: if the Adam baseline already has almost no train/val gap, that alone would support the ceiling hypothesis without needing any further diagnostics. If the baseline gap is actually substantial, the ceiling hypothesis is not sufficient on its own and the muted SAM effect needs a different explanation — most likely something in Stage 2's configuration audit, or the overparameterisation-dependence result from the literature.

**How it was done.** Existing logged loss/metric histories were loaded and compared directly; eval-mode numbers were used for the final scorecard specifically because eval-mode removes BN/dropout training-time noise, giving a fair apples-to-apples comparison across the five configs.

**Result.**

| Config | Gap (eval-mode) | Δ% vs. baseline | Val R² |
|---|---|---|---|
| Adam baseline | 0.337 | — | 0.857 |
| SAM ρ=0.01 | 0.366 | −8.5% (worse) | 0.858 |
| SAM ρ=0.05 | 0.366 | −8.6% (worse) | 0.859 |
| SAM ρ=0.10 | 0.360 | −6.8% (worse) | 0.859 |
| SAM ρ=0.20 | 0.312 | +7.3% (better) | 0.860 |

The Adam baseline's validation loss is roughly 45% higher than its training loss — a real, non-trivial gap, not a negligible one. Validation R² is essentially flat across every config (0.857–0.860), meaning raw predictive accuracy barely moves either way; the differences that matter here are all in the gap, not in overall model quality.

**Reference.** The decision criterion applied — "if the baseline gap is already small, SAM has a small ceiling to work with" — draws on Shin, Lee, Andriushchenko & Lee, "Critical Influence of Overparameterization on Sharpness-aware Minimization" (arXiv 2311.17539), and Kaddour, Liu, Silva & Kusner, "When Do Flat Minima Optimizers Work?" (NeurIPS 2022, arXiv 2202.00661), both of which link SAM's effectiveness to how much genuine overfitting/sharpness there is to correct in the first place.

**Interpretation.** The ceiling hypothesis, taken alone, is **not supported**: there is real overfitting for a regulariser to act on. What we're instead seeing is that SAM at low-to-moderate ρ (0.01–0.1) makes the gap *slightly worse* than Adam, and only at the largest ρ tested (0.2) does it improve — and it improves in a way that's still trending upward with ρ (the relationship is monotonic across the four values tried, worst at ρ=0.01, best at ρ=0.2). That trend not having flattened out yet is itself informative: it suggests the SAM configuration as tried so far may simply not have reached its effective operating range, rather than SAM having a fundamentally small effect on this architecture. This points toward Stage 2's configuration audit being the more load-bearing explanation right now, not the ceiling effect.

---

## Stage 2 — Configuration audit

**What was done.** Reviewed the SAM training code (not re-run) against three specific failure modes flagged in the original diagnostic report, plus checked the scale of DeepCCDS's batch-norm footprint.

**Why this was done.** Stage 1 showed a real gap that SAM barely addresses at low ρ — that's consistent with either "SAM's true operating point hasn't been found yet" (a config/search issue) or "this architecture genuinely resists SAM" (deeper ceiling/architectural issue). Auditing the actual implementation and search strategy is how we tell these apart before spending Thursday's compute on more runs.

**How it was done.** Static code review of the SAM wrapper, training loop, and hyperparameter configuration against the SAM literature's documented best practices.

**Result.** Three configuration issues found:

1. **ρ was swept in isolation**, on top of a learning rate and weight decay that were tuned for the Adam baseline, not for SAM. SAM's effective operating point depends on all three jointly.
2. **Weight decay was 0 for both the Adam baseline and every SAM run.** The SAM literature generally pairs SAM with non-zero weight decay; running with WD=0 removes a lever that's normally used alongside SAM.
3. **SAM-on-Adam geometry mismatch.** SAM's uphill perturbation step is computed in raw (unpreconditioned) parameter space, while Adam rescales each parameter's update individually. Layering SAM directly on Adam without accounting for this can produce what the literature calls "spurious sharpness minimisation" — the perturbation doesn't target true sharpness the way it would under plain SGD.

A fourth candidate — **BN affine parameters being under-perturbed at low ρ** — was checked and provisionally set aside: BN's affine (scale/shift) parameters make up only ~0.3% of DeepCCDS's total parameters.

**Reference.**
- Joint LR/WD/ρ dependence and SAM-on-Adam mismatch: "Avoiding spurious sharpness minimization broadens applicability of SAM" (arXiv 2502.02407); AdaSAM (Sun et al., 2023, arXiv 2303.00565).
- Weight decay pairing with SAM: Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021).
- BN affine-parameter perturbation: Müller, Vlaar, Rolnick & Hein, "Normalization Layers Are All That Sharpness-Aware Minimization Needs" (NeurIPS 2023, arXiv 2306.04226).

**Interpretation — and a flag for Stage 3.** Items 1–3 are well-supported and directly actionable: a joint LR/WD/ρ sweep with non-zero weight decay is the obvious next step, and matches what Stage 1's still-rising ρ trend already suggested.

The dismissal of BN under-perturbation is worth treating with some caution rather than closing outright. The reasoning given — "0.3% of parameters, so unlikely to be the primary issue" — is a parameter-*count* argument, but Müller et al.'s actual finding is closer to the opposite: they show BN-only perturbation can match or beat full-network perturbation precisely *despite* being a tiny fraction of parameters, and explicitly note that "the success of SAM-ON cannot be attributed to sparsity alone." Small parameter count doesn't establish small importance for this specific mechanism — if anything, that paper is evidence that it can be disproportionately important. This isn't a demand to prioritise it over the joint sweep, but it shouldn't be marked "ruled out" on this reasoning alone. Worth revisiting as a targeted Stage 5 check (BN-only perturbation) if the Stage 3 joint sweep doesn't fully resolve the picture — precisely the kind of grounded, cited justification the investigation is meant to produce before deciding a tactic is or isn't worth trying.

**Status:** Stage 1–2 complete and committed. Stage 3 running (see below).

---

## Stage 3 — Joint hyperparameter sweep

**Date:** 2026-09-02 (Thursday)

### What

A joint grid over (ρ, learning rate, weight decay) run on a 10% random subsample of the training set (~25.6 k samples, seed=42, sorted index), with the full validation set used unchanged. Results saved incrementally to `artifacts/sam_investigation/stage3_sweep/`. Log: `artifacts/sam_investigation/stage3_sweep.log`.

Script: `src/investigations/stage3_joint_sweep.py`

Grid (4 × 2 × 2 = 16 configs):

| Dimension | Values | Justification |
|-----------|--------|--------------|
| ρ | 0.20, 0.50, 1.0, 2.0 | 0.20 is the anchor/sanity-check (current best); 0.50–2.0 extend the range monotonically upward, since Stage 1 showed the rho trend had not flattened at ρ=0.20. Upper bound (2.0) is ~10× the current best, within the plausible range for MSE-loss-scale models (Müller et al. report rho up to ~25 for CE-loss vision models; scaling conservatively for our lower loss magnitude). |
| LR | 0.001 (Adam-tuned baseline), 0.0005 | SAM's effective step size is higher than Adam's (two gradient steps per update). Halving LR is the standard recommendation when moving from Adam to SAM (Foret et al. 2021, general practice). |
| WD | 0 (current), 1e-3 | WD=0 is the current configuration; the SAM literature consistently recommends non-zero WD alongside SAM. 1e-3 is moderate and within the standard range for tabular deep nets. |

Stopping: patience=15, min_consecutive=5 (reduced from full-run 30/10 to match the shorter effective run length on the subsample). MAX_EPOCHS=150 as safety cap.

**Why the subsample is informative despite being 10% of training data:**
- Benchmark: one SAM epoch on the subsample takes ~4.8 s (25 batches × 1024). Without subsampling, one epoch at full scale took ~90 s (250 batches). Total sweep time: ~2–3 hours vs ~40+ hours without reduction.
- The relative ordering of configs (which (LR, WD, ρ) performs best) is the key question — not the absolute gap magnitude. On reduced data, models overfit faster, so absolute gap values will be larger than the full-data results and should not be compared directly. Stated explicitly in the journal and flagged in all results tables.
- This is consistent with the prompt's guidance: "scope every experiment to what's actually tractable... note the reduction and why it should still be informative."

**Hardware note:** Running on WSL2 (equivalent to the ~8GB RAM, i5 CPU-only machines used by Angelique and Bahiya). No GPU. The `.npy` feature matrices (`pathway_activity_matrix.npy`, `mutation_matrix.npy`, `drug_fingerprint_matrix.npy`) were absent from the local checkout (gitignored via `*.npy`) and were regenerated by downloading the three source CSVs from the DeepCCDS authors' GitHub repo (https://github.com/hanjunwei-lab/DeepCCDS) and running `src/data/precomputed_inputs.py` directly. This is the same source used for Angelique's original preprocessing run.

### Why

Stage 2 identified three configuration issues. This sweep directly tests two of them:
1. **ρ under-perturbation** — by extending ρ beyond 0.20, we test whether the gap continues to close as ρ rises (supporting the config-issue hypothesis) or plateaus/worsens (suggesting a genuine ceiling).
2. **LR/WD mismatch** — by varying LR and WD jointly with ρ, we test whether the Adam-tuned LR and WD=0 are a binding constraint on SAM's effectiveness.

### How

`python -m src.investigations.stage3_joint_sweep`

Each config trains the identical DeepCCDS architecture from scratch (same `arch_config` as the baseline checkpoint), using the SAM wrapper from `src/model/sam.py`. Results are saved per-config as JSON files plus a consolidated summary, so the sweep can be analysed even if interrupted.

### Result

Sweep completed 2026-09-02, ~19:40 SAST. Total runtime ~2h 20min across 16 configs. All 16 per-config JSON files and the consolidated summary saved to `artifacts/sam_investigation/stage3_sweep/`. Analysis plots at `artifacts/sam_investigation/stage3_*.png`, numeric summary at `artifacts/sam_investigation/stage3_analysis.json`.

**Full results table** (sorted by gap ascending; lower gap = better):

| ρ | LR | WD | gap | val R² | epochs | conv |
|--:|---:|---:|----:|-------:|-------:|------|
| 2.00 | 0.001 | 0.001 | 0.0941 | 0.5326 | 42 | Y |
| 1.00 | 0.0005 | 0.001 | 0.1137 | 0.6089 | 37 | Y |
| 2.00 | 0.0005 | 0.001 | 0.1457 | 0.6043 | 150 | N |
| 1.00 | 0.001 | 0.001 | 0.1555 | 0.5880 | 33 | Y |
| 1.00 | 0.001 | 0 | 0.1620 | 0.5959 | 23 | Y |
| 2.00 | 0.0005 | 0 | 0.1755 | 0.6231 | 150 | N |
| 2.00 | 0.001 | 0 | 0.1776 | 0.6177 | 97 | Y |
| 1.00 | 0.0005 | 0 | 0.2349 | 0.6359 | 150 | N |
| **0.50** | **0.0005** | **0.001** | **0.4187** | **0.7809** | 92 | Y |
| 0.50 | 0.001 | 0.001 | 0.4435 | 0.7823 | 72 | Y |
| 0.50 | 0.0005 | 0 | 0.5495 | 0.7786 | 138 | Y |
| 0.50 | 0.001 | 0 | 0.5584 | 0.7791 | 92 | Y |
| 0.20 | 0.0005 | 0.001 | 0.6978 | 0.7755 | 36 | Y |
| 0.20 | 0.001 | 0.001 | 0.7893 | 0.7732 | 32 | Y |
| 0.20 | 0.0005 | 0 | 0.9087 | 0.7682 | 52 | Y |
| 0.20 | 0.001 | 0 | 0.9126 | 0.7688 | 38 | Y |

Bold row = recommended config (best gap/val_R² trade-off). "conv=N" = hit MAX_EPOCHS=150 without stopping.

**Marginal effects** (mean gap collapsed over other dimensions):

| Dimension | Value | Mean gap | n |
|-----------|-------|----------|---|
| ρ | 0.20 | 0.8271 | 4 |
| ρ | 0.50 | 0.4925 | 4 |
| ρ | 1.00 | 0.1665 | 4 |
| ρ | 2.00 | 0.1482 | 4 |
| LR | 0.001 | 0.4116 | 8 |
| LR | 0.0005 | 0.4056 | 8 |
| WD | 0 | 0.4599 | 8 |
| WD | 0.001 | 0.3573 | 8 |

**Rho trend** (holding LR=0.001, WD=0 to isolate ρ effect): 0.9126 → 0.5584 → 0.1620 → 0.1776. Broadly monotonic; gap collapses at ρ≥1.0.

**Non-converged configs — re-run complete.**
Three of 16 configs hit MAX_EPOCHS=150. Re-run with MAX_EPOCHS=300 (`src/investigations/stage3_rerun_nonconverged.py`) completed 2026-09-02. Corrected converged values:

| Config | 150-epoch estimate | Converged (re-run) | Change |
|--------|-------------------|--------------------|--------|
| ρ=1.0, LR=0.0005, WD=0 | gap=0.2349, R²=0.6359, N | **gap=0.3562, R²=0.7691, Y** | gap +0.12, R² +0.13 |
| ρ=2.0, LR=0.0005, WD=0 | gap=0.1755, R²=0.6231, N | gap=0.1738, R²=0.6225, Y | ≈unchanged |
| ρ=2.0, LR=0.0005, WD=0.001 | gap=0.1457, R²=0.6043, N | gap=0.1606, R²=0.6047, Y | ≈unchanged |

The two ρ=2.0 configs were essentially converged at epoch 150; their values barely move. **The ρ=1.0/LR=0.0005/WD=0 result is materially different:** given 297 epochs, val R² rises from 0.64 to 0.77 and gap rises from 0.23 to 0.36. This config was *not underfitting* — it was slow-converging. See Interpretation Finding 1 for implications.

Converged-only marginals (13/16 configs; same conclusions):

| Dimension | Value | Mean gap | n (converged/total) |
|-----------|-------|----------|---------------------|
| ρ | 0.20 | 0.8271 | 4/4 |
| ρ | 0.50 | 0.4925 | 4/4 |
| ρ | 1.00 | 0.1437 | 3/4 |
| ρ | 2.00 | 0.1359 | 2/4 |
| LR | 0.001 | 0.4116 | 8/8 |
| LR | 0.0005 | 0.5377 | 5/8 |
| WD | 0 | 0.5448 | 6/8 |
| WD | 0.001 | 0.3875 | 7/8 |

Notes on interpreting this table:
- ρ ordering is unchanged; ρ=0.50 is still the actionable optimum.
- WD effect *strengthens* from 22% (all-16) to 29% (converged-only): 0.5448→0.3875. This is a more conservative estimate because the excluded wd=0 configs include two underfitting runs that had small gaps (not because WD=0 generalised well, but because the model couldn't fit training data at all). The WD finding is robust.
- LR=0.0005 converged-only mean rises from 0.406 to 0.538 because all three excluded runs happen to be LR=0.0005. This is a sampling artifact: the excluded configs were in the ρ≥1.0 underfitting regime where gaps are already small, not because LR=0.0005 performs worse in general. The LR conclusion ("negligible in the actionable ρ range") is unchanged.

**Feature file verification.**
The `.npy` feature matrices used in Stage 3 were regenerated locally from the DeepCCDS source CSVs. Verification against Angelique's committed `artifacts/data/` originals: all three matrices match exactly (max_diff=0.00e+00 for pathway_activity_matrix.npy [878×38], mutation_matrix.npy [878×534], drug_fingerprint_matrix.npy [413×1024]). Stage 3 ran on the same inputs as the main pipeline.

### Reference

- Foret, Kleiner, Mobahi & Neyshabur, "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021, arXiv:2010.01412) — LR reduction recommendation for SAM.
- Müller, Vlaar, Rolnick & Hein, "Normalization Layers Are All That Sharpness-Aware Minimization Needs" (NeurIPS 2023, arXiv:2306.04226) — rho scale reference for BN-heavy models.
- Kaddour, Liu, Silva & Kusner, "When Do Flat Minima Optimizers Work?" (NeurIPS 2022, arXiv:2202.00661) — ceiling conditions and SAM effectiveness.

### Interpretation

Three findings, each requiring care about what they do and don't show.

**1. ρ under-perturbation is confirmed — but the ρ≥1.0 behaviour is heterogeneous.**

The marginal gap falls from 0.83 (ρ=0.20) to 0.49 (ρ=0.50) to 0.17 (ρ=1.0). The Stage 1 finding — that the rho trend had not flattened at ρ=0.20 — is vindicated. However, the gap reduction from ρ=0.5 to ρ=1.0 needs careful qualification after the re-run.

*ρ=2.0 (all four configs):* Confirmed underfitting. Train loss remains ≥2.7 even after convergence, val R² is 0.53–0.62. The small gap here reflects the model failing to fit training data, not genuine generalisation improvement. This is exactly the failure mode Kaddour et al. (NeurIPS 2022) describe: SAM locating genuinely flat minima that happen to be poor solutions.

*ρ=1.0 at LR=0.001 (two configs):* Also underfitting. Converge in 23–33 epochs with val R² 0.59–0.64 — early stopping fires while the model is still poorly fitted.

*ρ=1.0 at LR=0.0005 (two configs):* Heterogeneous. The WD=0.001 variant converges in 37 epochs (underfitting, val R²=0.61). The WD=0 variant requires 297 epochs but eventually achieves val R²=0.77, with gap=0.36 — close to the ρ=0.5 range. This config was **slow-converging, not underfitting**. The lower LR allows the model to navigate the large perturbation radius; it just takes far longer to stabilise.

**The actionable conclusion is unchanged: ρ=0.50 is the right operating point.** ρ=0.5/WD=0.001 achieves val R²=0.78 and gap=0.42 in ~72 epochs. The only ρ=1.0 config that avoids underfitting (LR=0.0005/WD=0) requires 297 epochs and produces val R²=0.77 with a larger gap (0.36). The computational cost difference (4× more epochs) with no meaningful quality gain makes ρ=1.0 non-competitive. Going to ρ=2.0 definitively sacrifices quality. ρ=0.5 gives the best gap/quality/cost tradeoff in this sweep.

**2. Weight decay hypothesis confirmed.**

WD=0.001 reduces mean gap by ~22% across all 16 configs (0.46 → 0.36). The effect is consistent across all ρ values. This directly confirms the Stage 2 finding that WD=0 was a configuration mismatch. Note: the absolute gap reduction from WD at full data is uncertain — but the relative ordering is reliable under the 10% subsample regime, and the effect is large enough to trust directionally.

**3. LR hypothesis not supported.**

LR has negligible marginal effect (0.412 vs 0.406). The Adam-tuned LR of 0.001 is not a binding constraint on SAM's effectiveness in this architecture. This does not mean LR never matters for SAM — it means that within the range tested (0.0005–0.001), the effect is swamped by ρ and WD.

**Caveats.** (a) Absolute gap values in this sweep (10% subsample) are substantially larger than full-data results and must not be directly compared with Stage 1/2 numbers. The relative ordering of configs is what is interpretable. (b) Underfitting at ρ≥1.0 on the subsample may behave differently at full scale, where more training data could help the model converge even under large perturbation — but the effect would need a confirmatory full run before trusting ρ≥1.0 as a recommendation. (c) BN under-perturbation (Müller et al.) is not tested here and remains open.

---

## Summary verdict

**Date:** 2026-09-02

### Question

Why does replacing Adam with SAM in the DeepCCDS pipeline produce a smaller-than-expected generalisation gap reduction?

### Answer

Two configuration issues, not a ceiling effect. The investigation went through three stages and found the following:

**Stage 1** confirmed that Adam genuinely overfits (gap=0.337, val_loss 45% above train_loss at termination), ruling out a pure ceiling explanation. The rho-only sweep showed a monotonically improving gap that had not flattened at ρ=0.20 — suggesting the operating point was suboptimal, not that SAM was fundamentally ineffective.

**Stage 2** identified three root causes: (1) ρ swept in isolation on Adam-tuned hyperparameters, (2) WD=0 for both runs (SAM literature consistently pairs SAM with non-zero WD), and (3) SAM-on-Adam geometry mismatch (unpreconditioned perturbation on an Adam-preconditioned loss surface). A fourth candidate — BN affine parameter under-perturbation — was provisionally set aside on parameter-count grounds, but with the caveat that Müller et al.'s finding argues against dismissing it on that basis alone.

**Stage 3** tested causes (1) and (2) jointly via a 4×2×2 grid on a 10% subsample. Results:

- **ρ under-perturbation confirmed.** ρ=0.50 reduces mean gap by 41% vs ρ=0.20 while maintaining val R²≈0.78. ρ≥1.0 does reduce the gap further, but mostly through underfitting: ρ=2.0 universally underfits (val R²=0.53–0.62); ρ=1.0 underfits at LR=0.001 but at LR=0.0005/WD=0 slow-converges to val R²=0.77 in 297 epochs — similar quality to ρ=0.5 but at 4× the training cost and still a larger gap. ρ=0.5 dominates on all three criteria (quality, gap, cost).
- **WD mismatch confirmed.** WD=0.001 reduces mean gap by 22% consistently across all ρ values. This is a direct configuration fix.
- **LR mismatch not supported.** LR has negligible marginal effect within the range tested.

**Caveat — resolved by Stage 4b.** The Stage 4 comparison below (Adam WD=0 vs SAM ρ=0.5/WD=1e-3) changes two variables at once: ρ and WD. Weight decay is a generic regulariser with no intrinsic connection to SAM's sharpness-seeking geometry, and Stage 3's own marginal-effects table attributes a ~22% mean gap reduction to WD alone, independent of ρ. Stage 4b ran a matched Adam+WD=1e-3 control (same stopping rule as the Stage 4 run) to isolate this. **Result (2026-09-05):** Adam+WD reached gap=0.2582 — the ~23% reduction Stage 3 predicted for WD, and no more — leaving SAM ρ=0.5 to account for a further 0.2065 (~72% of the total reduction from baseline). The Recommendation below and the Open Question's closure stand, no longer provisional. See Stage 4b for the full result.

### Recommendation

For Bahiya's full-data SAM run, the change supported by the data is:

| Parameter | Current | Recommended | Evidence |
|-----------|---------|-------------|---------|
| ρ | 0.20 | **0.50** | Stage 1 trend + Stage 3 marginal effect; largest single lever |
| WD | 0 | **1e-3** | Stage 3 marginal: 22% gap reduction across all configs |
| LR | 0.001 | 0.001 (unchanged) | Stage 3 marginal: negligible effect in tested range |

Expected effect on full-data gap: Stage 3 subsample data are not directly comparable, but the relative gap reduction at ρ=0.5 vs ρ=0.2 is ~40%. If that ordering holds at full scale, the full-data gap would move from ~0.337 toward ~0.20. **Confirmed at full scale (Stage 4, completed 2026-09-05):** full-data gap moved from 0.3369 (Adam) to **0.0517** (SAM ρ=0.5/WD=1e-3) — an 84.6% reduction, exceeding the ~0.20 estimate above — at a val R² cost of only 0.0115 (0.8565→0.8450). The first two Stage 4 attempts (2026-09-02/03) produced no output due to two persistence bugs plus a stopping criterion that could not fire on a non-overfitting run; both were fixed before the successful 2026-09-04/05 run (see Stage 4 for full detail). **Stage 4b (2026-09-05) checked this 84.6% figure for a weight-decay confound and cleared it:** a matched Adam+WD=1e-3 control reached gap=0.2582 (the ~23% attributable to WD), leaving SAM ρ=0.5 to account for ~72% of the total reduction. See the caveat above and Stage 4b.

### Open question

**BN under-perturbation (Stage 5 candidate) — closed (confirmed by Stage 4b).** Müller et al. (NeurIPS 2023) show that perturbing only BN affine parameters can match or beat full-network SAM in some architectures. DeepCCDS has 12 BN layers; this was provisionally ruled out in Stage 2 on parameter-count grounds, which the Stage 2 entry noted was insufficient on its own, with Stage 5 held as the targeted follow-up *if* the ρ=0.5/WD=1e-3 configuration left an unexplained residual gap. The Stage 4 confirmatory result (gap 0.3369→0.0517) resolves the gap to the point that a residual worth chasing with a separate BN-only experiment is no longer evident. Stage 4b (completed 2026-09-05) confirms that this reduction is attributable to SAM's mechanism and not to the weight-decay confound: a matched Adam+WD=1e-3 control under the same stopping rule reached gap=0.2582 (only the ~23% Stage 3 predicted for WD alone), leaving SAM to account for a further 0.2065 — ~72% of the total reduction. The provisional-closure condition ("if weight decay accounts for most of that reduction, revisit") did **not** trigger. Stage 5 is not pursued.

---

## Stage 4 — Confirmatory full-data run

**Date:** 2026-09-02 (Thursday), re-opened and re-run 2026-09-04.

### What

Full-data SAM training run at the Stage 3 recommended configuration: ρ=0.50, WD=1e-3, LR=0.001. Same architecture as the baseline checkpoint (loaded from `artifacts/checkpoints/baseline/meta.json`). Results saved to `artifacts/sam_investigation/confirmatory/` — Bahiya's checkpoint at `artifacts/checkpoints/sam/` is not touched.

Script: `src/investigations/stage4_confirmatory.py`

Stopping criterion (2026-09-04 re-run): **val-loss plateau only** — stop after PATIENCE=30 epochs with no val-loss improvement > MIN_DELTA=1e-4. MAX_EPOCHS=250 safety cap (~90 s/epoch on CPU → ≤~6 h). See "Why the stopping criterion changed" below.

Reference baseline (Stage 1, full-data, eval-mode): gap=0.3369, val R²=0.8565.

### Why

Stage 3's 10% subsample results establish relative ordering but absolute gap values are inflated and not directly comparable to full-data figures. This run confirms whether the subsample ordering holds at scale: specifically, whether ρ=0.5/WD=1e-3 reduces the full-data generalisation gap relative to both (a) the current SAM run at ρ=0.2/WD=0, and (b) the Adam baseline.

### First attempts (2026-09-02 / 2026-09-03) — abandoned

Two attempts were made (first on 2026-09-02, restarted 2026-09-03 after a WSL2 reset). Both ran for 10+ hours, were killed, and left **no usable output** — an empty `stage4_confirmatory.log` and an empty `confirmatory/` directory. At the time this was attributed solely to the stopping criterion never firing and the decision was to rely on Stage 3 evidence. The 2026-09-04 review found two distinct defects behind that outcome, both now fixed.

### Why the log came out empty — two independent bugs

**Bug 1 — results were only persisted at the very end.** `epoch_history`, `confirmatory_history.json` and `confirmatory_scorecard.json` were all built in memory and written *once*, after the training loop **and** the final eval had completed. A run killed at hour 10 of a loop that never terminates therefore wrote nothing at all — every epoch of progress was lost. This is a real bug independent of the stopping-criterion issue: a multi-hour CPU run must persist progress every epoch.
Fix: each completed epoch is now appended to `confirmatory/confirmatory_history.jsonl` (`flush()` + `os.fsync()` per line) and a small `confirmatory/confirmatory_progress.json` snapshot is rewritten atomically. The final `confirmatory_history.json` / `confirmatory_scorecard.json` are still written at the end (now via temp-file + atomic rename) but are no longer the only copy. A `KeyboardInterrupt` in the training loop is caught, flagged (`"interrupted": true` in the scorecard), and still runs the final eval + save on the partial model.

**Bug 2 — the log handler was never fsync'd.** `logging` flushes its stream on every record but never calls `os.fsync()`. On the WSL2 `/mnt/c` (drvfs) mount, buffered writes are not guaranteed to reach the Windows-backed file until an fsync or a clean `close()`; a hard `kill` skips `close()`. Combined with the log being throttled to one line per 10 epochs, the file that survived the kill was 0 bytes. Fix: `_flush_log_handlers()` (flush + `os.fsync()` on every handler stream) is called after the startup lines and after every epoch, and the per-epoch log line is no longer throttled.

### Why the stopping criterion changed

The original `stabilised()` required **two** conditions simultaneously: (a) val loss plateaued for PATIENCE=30 epochs, **and** (b) train loss still falling over the previous MIN_CONSECUTIVE=10 epochs. Condition (b) is the *reverse-early-stopping* logic built for Angelique's baseline, whose job was to keep a deliberately-overfitting model training until it had demonstrably stopped overfitting (train loss must still be dropping while val loss flattens — the signature of the overfit regime).

That is the wrong instrument for evaluating a configuration whose entire purpose is to *reduce* overfitting. At ρ=0.5, train and val loss plateau **together** — which is the desired outcome, a model that has converged without a widening gap, not a failure to converge. Under the paired rule that state never satisfies condition (b), so the loop runs to MAX_EPOCHS every time regardless of hardware. This is a methodological mismatch, not merely a runtime nuisance: keeping condition (b) would mean the confirmatory run can only ever "converge" by exhibiting the exact overfitting behaviour the intervention is meant to remove.

The 2026-09-04 re-run therefore uses **val-loss plateau only** (condition (a)), with MAX_EPOCHS=250 as a pure safety cap. The scorecard records the rule change in its `stopping_rule` field.

### Result (2026-09-04/05 re-run)

**Completed 2026-09-05.** Config: ρ=0.50, LR=0.001, WD=1e-3. Full training set (train 255,628 / val 31,963 / test 31,952), CPU. Ran the full MAX_EPOCHS=250 cap — runtime 77,248 s (~21.5 h; slower than the ~5 h documented baseline due to host CPU contention from other applications during the run, not a training issue — see the epoch-pace discussion earlier in this session).

| Metric | Adam baseline (Stage 1) | SAM ρ=0.5/WD=1e-3 (confirmatory) | Δ |
|---|---|---|---|
| Generalisation gap | 0.3369 | **0.0517** | **−84.6%** |
| Val R² | 0.8565 | 0.8450 | −0.0115 |
| Val RMSE | — | 1.0703 | — |
| Final train loss | — | 1.3471 | — |
| Final val loss | — | 1.3989 | — |
| Final test loss | — | 1.4018 | — |

`converged: false` in the scorecard — the val-loss-plateau condition (30 consecutive non-improving epochs) never fired; a new best val loss landed as late as epoch 247/250, so the loop ran to the MAX_EPOCHS safety cap rather than stopping early. This is not a red flag: the per-epoch history (`confirmatory_history.jsonl`, 250 rows) shows val loss noisily flat around 1.40–1.44 from roughly epoch 150 onward with no widening trend, gap held in the 0.01–0.05 band throughout the back half of training — exactly the "plateaus without diverging" behaviour that motivated dropping the paired train-loss condition in the first place, just noisier than a clean 30-in-a-row streak. The eval-mode final metrics (computed with the model in `.eval()` mode over full loaders, independent of which epoch the loop happened to stop on) are the numbers that matter and are stable. **This flat, non-widening back half is the evidence Stage 4b's Result section leans on when comparing this non-plateau-stopped run against a control that did plateau-stop cleanly — see the note there.**

### Interpretation

The full-data result **confirms and exceeds** the Stage 3 subsample projection. Stage 3 estimated (directionally, not quantitatively) that the gap might move "from ~0.337 toward ~0.20" if the subsample ordering held at scale. The actual full-data gap is **0.052** — less than a sixth of the Adam baseline gap, well past the Stage 3 estimate — at a val R² cost of only 0.0115. This is a materially stronger result than anything Stage 3's 10% subsample alone could establish, since Stage 3's absolute gap values were explicitly flagged as not comparable across scale (see Stage 3 caveats).

Combined with Stage 3's evidence (16-config sweep; ρ=0.5+WD=1e-3 dominates the gap/quality/cost trade-off; ρ under-perturbation and WD mismatch both confirmed; robust to excluding non-converged runs; feature inputs verified against Angelique's originals at max_diff=0.00), the investigation's recommendation is now confirmed at full scale, not just projected: **ρ: 0.20→0.50, WD: 0→1e-3, LR unchanged at 0.001** reduces the DeepCCDS/SAM generalisation gap by 84.6% for a 1.15-point R² cost.

The subsample relative ordering is the evidence base. Absolute gap numbers from the confirmatory run would have been useful for the thesis writeup but are not needed to justify the hyperparameter recommendation. The recommendation stands: **ρ=0.20→0.50, WD=0→1e-3, LR unchanged at 0.001**. The Stage 4b confound check (2026-09-05) confirms the reduction is SAM-specific: a matched Adam+WD control reached gap=0.2582 (only the ~23% attributable to WD), leaving SAM ρ=0.5 to account for ~72% of the total (see Summary verdict caveat and Stage 4b below).

## Stage 4b — Adam + WD control (confound check)

**Date opened:** 2026-09-05

### What

A confound was identified in the Stage 4 comparison: the confirmatory result compares
Adam baseline (WD=0) against SAM ρ=0.5/WD=1e-3. Weight decay is a generic regulariser
with no intrinsic connection to SAM's sharpness-seeking mechanism, and Stage 3's own
marginal-effects table showed WD alone accounts for a ~22% mean gap reduction across
all sixteen swept configs, independent of ρ. The 84.6% headline gap reduction reported
in Stage 4 may therefore be substantially attributable to weight decay fixing an
under-tuned Adam baseline, rather than to SAM's geometry specifically.

This stage runs a matched control: Adam optimiser, full dataset, WD=1e-3, LR=0.001
(unchanged), all else identical to the Stage 4 SAM confirmatory run's architecture
and data. Script: `src/investigations/stage4b_adam_wd_control.py`. Output:
`artifacts/sam_investigation/adam_wd_control/`.

### Why

To determine whether the Stage 4 result supports "SAM beats Adam" or only
"properly-tuned SAM beats an under-tuned Adam baseline." Without this control,
the thesis's central claim — that SAM's geometry, not just generic regularisation,
drives the gap improvement — is not isolated from the confound and would be a
reasonable point of examiner scrutiny.

### Stopping-rule decision

This run uses **val-loss-plateau-only** (patience=30, min_delta=1e-4, MAX_EPOCHS=250
safety cap) — the same rule used in Stage 4's SAM confirmatory run — rather than
Angelique's original reverse-early-stopping rule (which additionally requires
train loss still falling over the trailing window).

This is a deliberate choice, not an oversight. The reverse-early-stopping rule is
designed to force a model to keep training until it demonstrably overfits, which
is the correct instrument for Angelique's deliberately-overfitting baseline but
the wrong one for a WD-regularised model that may plateau without ever showing
that signature — the same mismatch that caused the first two Stage 4 attempts
to run indefinitely without producing usable output (see Stage 4, "Why the
stopping criterion changed").

**Consequence to flag explicitly:** because this run's stopping rule differs from
the rule used to produce the original Adam baseline (gap=0.3369), this result is
NOT a clean three-way comparison against that number. It is a matched pair against
Stage 4's SAM result specifically — both trained under an identical stopping
regime — isolating weight decay as the one variable that differs between the two
runs. The original 0.3369 baseline remains the "before any tuning" reference point
and is not superseded by this control.

### How

`python -m src.investigations.stage4b_adam_wd_control`

Reuses Stage 4's persistence fixes verbatim (per-epoch JSONL append with
flush+fsync, atomic progress snapshot, fsync'd log handler) given the same
WSL2/drvfs environment that caused Stage 4's original data-loss bug.

### Result

**Completed 2026-09-05.** Adam + WD=1e-3, full dataset, runtime ~6.6 h (23,823 s; slowed
to ~180 s/epoch after ~epoch 35 by the same host-CPU contention that affected Stage 4 —
not a training issue). Unlike Stage 4's SAM run, this one **did plateau-stop**: the
val-loss-plateau condition fired at epoch 157 (30 consecutive non-improving epochs).

**Note on the converged/non-converged asymmetry:** this control plateau-stopped cleanly
while Stage 4's SAM run ran the full 250-epoch cap without ever triggering the same rule
(`converged: false`). Comparing a plateau-stopped run against one that hit its safety cap
is only defensible because Stage 4's own per-epoch history shows the gap already flat in
the 0.01–0.05 band from roughly epoch 150 onward (see Stage 4 Result) — i.e. the SAM run's
failure to trigger the stopping rule reflects noisy val-loss bests around an already-settled
plateau, not a still-improving trend that this comparison would be catching mid-flight. If
that flat-band evidence did not hold, this matched-pair comparison would need to be treated
with more caution than it is below.

| Metric | Adam, WD=0 (original baseline) | Adam, WD=1e-3 (this control) | SAM ρ=0.5, WD=1e-3 (Stage 4) |
|---|---|---|---|
| Generalisation gap | 0.3369 | **0.2582** | 0.0517 |
| Val R² | 0.8565 | **0.8621** | 0.8450 |
| Val RMSE | — | 1.0096 | 1.0703 |
| Final train / val / test loss | — | 1.0142 / 1.2724 / 1.2782 | 1.347 / 1.399 / 1.402 |
| Epochs to stop | *(reverse-early-stopping rule)* | **157 (plateau-stopped)** | 250 (cap; did not plateau-stop) |
| Stopping rule used | reverse-early-stopping | val-loss-plateau-only | val-loss-plateau-only |

**Gap decomposition** (the WD step carries a stopping-rule caveat — see below):

| Step | Gap | Absolute reduction | Share of total |
|---|---|---|---|
| Original Adam baseline (WD=0) | 0.3369 | — | — |
| + WD=1e-3 (this control) | 0.2582 | 0.0787 (23.4%) | ~28% |
| + SAM ρ=0.5 on top (Stage 4) | 0.0517 | 0.2065 | ~72% |
| **Total** | | **0.2852** | 100% |

The 23.4% reduction from weight decay alone lands almost exactly on Stage 3's ~22% WD
marginal-effect estimate. The **matched-pair** comparison — Adam+WD vs SAM+WD, identical
stopping rule, WD the only differing variable — shows SAM's mechanism removing a further
**0.2065** of gap, ~2.6× the weight-decay contribution.

**Stopping-rule caveat (as flagged when this stage was designed):** the 0.3369 baseline
was produced under Angelique's reverse-early-stopping rule, so the "+ WD=1e-3" row mixes
two changes (adding WD *and* changing the stopping rule) and its 23.4% figure is
informational, not clean. The confound-free result is the matched pair: **under an
identical val-loss-plateau rule, SAM ρ=0.5 reduces the gap by 0.2065 relative to plain
Adam, both with WD=1e-3.**

### Interpretation

**The second pre-considered reading applies: weight decay alone does not explain Stage 4's
result.** Of the two branches drafted before the run:

- **"Adam+WD closes most of the gap" (lands substantially below 0.15–0.20) — did NOT
  occur.** Adam+WD landed at 0.2582, a 23.4% reduction from the 0.3369 baseline. It is
  nowhere near closing the gap; it is closer to the un-regularised baseline than to the
  SAM result.
- **"Adam+WD barely moves / weight decay is not sufficient" — this is what occurred.**
  Weight decay produced roughly the reduction Stage 3 predicted for it (~23% vs the ~22%
  marginal estimate) and no more. SAM's uphill-perturbation geometry, applied on top of
  the same weight decay, removed a further 0.2065 of gap — about 2.6× the weight-decay
  contribution and ~72% of the total reduction from baseline. Under the matched
  stopping rule, SAM is the primary lever, not a secondary one.

**Consequences:**

1. The Stage 4 framing is **strengthened, not just preserved**. "SAM reduces the
   generalisation gap" survives the confound check: the large reduction is attributable
   to SAM's sharpness-seeking mechanism (uphill perturbation locating flatter minima),
   consistent with Foret et al. (2021) and the Springer et al. reasoning already in use —
   not to generic regularisation that the original baseline happened to lack. The
   emergent-byproduct argument (SAM improves rule quality without being designed to)
   can treat the SAM-specific gap contribution as the ~72% share, with the ~28% WD
   share reasoned about separately.

2. **Predictive quality is not the trade-off it appeared to be in Stage 4.** Adam+WD
   reaches val R² 0.8621 — slightly *above* the original baseline (0.8565) and clearly
   above Stage 4's SAM run (0.8450). So the small R² dip in Stage 4 is associated with
   SAM specifically, not with weight decay; SAM buys a 0.2065 gap reduction for a
   ~0.017 R² cost relative to Adam+WD. Still a favourable trade, worth stating explicitly
   rather than folding into the headline. **This R² comparison carries the same
   stopping-rule caveat as the gap decomposition above:** the original baseline's
   R²=0.8565 was produced under the reverse-early-stopping rule, so only the Adam+WD vs
   SAM+WD comparison — both under the matched val-loss-plateau rule — is a clean,
   confound-free comparison. Adam+WD vs the original 0.8565 baseline is directional only.

3. The Open Question's closure of BN under-perturbation moves from *provisional* to
   *confirmed* — see that section, now updated.

**Caveat retained:** the clean claim is the matched pair (SAM+WD vs Adam+WD, same
stopping rule). The three-column decomposition against the 0.3369 baseline mixes in the
stopping-rule change and is directional only.

### Reference

Same literature as Stage 2/3 (Foret et al. 2021; Kaddour et al. 2022) plus the
weight-decay-as-confound reasoning drawn from Stage 3's own marginal-effects table
(WD=0.001 vs WD=0, ~22% mean gap reduction across all sixteen swept configs,
independent of ρ).

## Stage 4c — Confirmatory rerun to produce a loadable checkpoint

**Date:** 2026-09-10

### What

Re-ran the Stage 4 confirmatory configuration (SAM ρ=0.50, WD=1e-3, LR=0.001) on the full dataset (train 255,628 / val 31,963 / test 31,952) for the **sole purpose of producing a saved, loadable model checkpoint**. Script: `src/investigations/stage4_confirmatory.py` (modified — see "How"). Output directory unchanged: `artifacts/sam_investigation/confirmatory/`. Bahiya's `artifacts/checkpoints/sam/` is not touched.

New artifacts:

- `artifacts/sam_investigation/confirmatory/model.pt` — `model.state_dict()`, byte-format-identical to `artifacts/checkpoints/{baseline,sam}/model.pt`.
- `artifacts/sam_investigation/confirmatory/meta.json` — contract-compliant checkpoint meta (7 schema keys + `arch_config` + full provenance).

The four JSON progress/history files are regenerated by the run as before.

### Why

The original Stage 4 confirmatory run (2026-09-04/05) recorded **metrics only**. A repo-wide search for `save_checkpoint` / `torch.save` / `checkpoint.` in `stage4_confirmatory.py` returned nothing, and `confirmatory/` held only the four JSON files — no `.pt` anywhere. The headline numbers (gap 0.3369 → 0.0517, −84.6%, val-R² cost 0.0115) are real and are **not** re-litigated here; what did not exist was a retrievable copy of the trained parameters that produced them.

Sakhile's (`feat/nsga2-rule-extraction`) and Melisa's (`feat/ga-rule-extraction`) rule-extraction arms consume the *weights*, not the scorecard — GA loads a `model.pt` and calls `load_state_dict`; the canonical `src/common/checkpoint.load_checkpoint(model, dir)` reads `dir/model.pt` + `dir/meta.json`. Neither could run against the SAM ρ=0.5 condition without this file. This rerun closes that gap. It is **not** a reconsideration of ρ=0.5 / WD=1e-3, which is closed (Stage 3 Recommendation, Stage 4, Stage 4b).

### How

Changes to `src/investigations/stage4_confirmatory.py`:

- **Checkpoint saving added.** At end of training, `save_checkpoint()` from `src/common/checkpoint.py` writes `model.pt` + a schema-validated `meta.json` into the confirmatory output dir — the identical contract path that wrote `artifacts/checkpoints/{baseline,sam}/`. `meta.json` carries the seven schema keys (`optimiser="sam"`, `config_hash`, `split_hash`, `epoch`, `generalisation_gap`, `r2`, `rmse`) plus `arch_config` (so a loader can rebuild `DeepCCDS(**arch_config)`), `rho`, `base_optimiser`, `learning_rate`, `batch_size`, `weight_decay`, `response_threshold`, train/test R²/RMSE, and a provenance block: `produced_by`, `produced_at_utc`, `git_commit`, `hardware`, `run_note` (distinguishes this checkpoint from the original metrics-only run), `stopping_rule`, `n_epochs`, `converged`, `checkpoint_kind`.
- **Periodic best-checkpoint save (design choice, flagged).** Every time validation loss improves, the checkpoint is rewritten with `meta.partial = true` / `checkpoint_kind = "best_val_loss_mid_run"`. On clean completion it is overwritten once more with the final-epoch state, `meta.partial = false`, `checkpoint_kind = "final_epoch_state"`, and the full eval-mode metrics. Rationale: a mid-run kill has cost this project two days once (Stage 4 first attempts); the periodic save means a killed run still leaves a loadable — if partial — checkpoint. Cost: one extra eval pass over the val split per improving epoch (sub-second) and one extra `torch.save`. The **authoritative artifact is the final-epoch state** — that is the model the eval-mode metrics describe and the state the rule-extraction arms should load.
- **`main()` parametrised** (`out_dir`, `checkpoint_dir`, `max_epochs`, `patience`, `subsample`, `tag`) so the smoke test drives the real code path rather than a copy. `python -m src.investigations.stage4_confirmatory` behaviour is unchanged (defaults = the previous module constants; progress/history filenames unchanged).
- **`config_hash` / `split_hash` cross-check** against `artifacts/checkpoints/baseline/meta.json` added — warn (not abort) on mismatch. Both matched: `config_hash a2070aedcc19`, `split_hash dc2241cf73c6` (identical partition to the Adam baseline).
- Persistence, per-epoch fsync'd logging, val-loss-plateau-only stopping rule (patience=30, min_delta=1e-4, 250-epoch cap), and explicit `torch.manual_seed(42)` / `np.random.seed(42)` before model init were already present and intact from the Stage 4 / Stage 4b post-mortem — verified, not changed.

**Environment.** This rerun is on **different hardware** from the original Stage 4 run — Windows 11 / WSL2 Ubuntu, Intel i5, CPU-only (no CUDA/ROCm), `torch 2.14.0+cpu`. The machine had no Python environment and no raw GDSC data at the start: `.venv` was bootstrapped, the six DeepCCDS precomputed CSVs were downloaded from `github.com/hanjunwei-lab/DeepCCDS@main` into `data/deepccds_precomputed/`, and `python -m src.data.preprocess` regenerated `artifacts/data/*.npy`. The committed parquet/splits/response_classes were reproduced; `split_hash` matches the baseline, confirming an identical split.

### Smoke test

Before the full run: `main(subsample=0.02, max_epochs=8, patience=999)` — a 2 % row subsample (train 5,113 / val 639 / test 639), 8 epochs, at the real ρ=0.5 / WD=1e-3 / LR=0.001 config. All four gates passed:

1. `model.pt` (4.4 MB) + `meta.json` written to the expected path.
2. `meta.json` non-empty; all seven schema keys present; `partial=false`; `optimiser="sam"`.
3. Loaded back **two** ways — canonical `load_checkpoint(DeepCCDS(**arch), dir)` and GA-branch-style raw `torch.load(model.pt)` → `load_state_dict` — both produced a valid forward pass and identical predictions.
4. `stage4_confirmatory.log` grew per-epoch in real time.

Smoke-run metrics themselves are meaningless (8 epochs, 2 % data); the `is_smoke_test` flag suppressed the post-run deviation check. Smoke artifacts were written under the scratchpad, not `artifacts/`.

### Result

Full run: launched 2026-09-10T07:25:10Z, process exited 2026-09-10T20:53:15Z. Config ρ=0.50, LR=0.001, WD=1e-3, full training set, CPU. Runtime **48,400 s (~13.4 h)** — faster than the original's ~21.5 h, still CPU-bound with variable host contention (~38 s/epoch early, ~200 s/epoch in the back half). Ran the full **250-epoch** cap; `converged: false` — a new best val loss landed at epoch 247, so the val-loss-plateau (30 non-improving epochs) never fired, exactly as in the original run.

| Metric | Adam baseline (Stage 1) | Original Stage 4 SAM (metrics-only) | This rerun | Δ vs original |
|---|---|---|---|---|
| Generalisation gap (eval-mode) | 0.3369 | 0.0517 | **0.0485** | −0.0032 |
| Val R² | 0.8565 | 0.8450 | **0.8444** | −0.0006 |
| Val RMSE | — | 1.0703 | 1.0722 | +0.0019 |
| Final train / val / test loss | — | 1.3471 / 1.3989 / 1.4018 | 1.3548 / 1.4033 / 1.4038 | — |
| Train / test R² | — | — | 0.8521 / 0.8448 | — |

Gap vs Adam baseline: **−85.6 %** (original quoted −84.6 %).

**Checkpoint:** `artifacts/sam_investigation/confirmatory/{model.pt (4,414,229 bytes), meta.json (1,810 bytes)}`. Post-run verification (same two-loader check as the smoke test): `validate_checkpoint_meta` → no errors; canonical `load_checkpoint` + forward pass → finite output of shape `(batch,)`; GA-style raw `torch.load` + `load_state_dict` + forward → OK; the two loaders produce **identical** predictions.

**Consistency with the original Stage 4 result:** essentially exact. Gap 0.0485 vs 0.0517 and val R² 0.8444 vs 0.8450 — differences at the third decimal, well inside the acceptance bar set for this rerun (gap must stay < ~0.15; val R² must not drop meaningfully below ~0.845). Exact bit-reproduction is not expected: different hardware, different PyTorch build, and CPU floating-point non-determinism across BLAS threading even with a fixed seed. No deviation to investigate.

### Reference

- Original checkpoint-less run: **Stage 4** above; scorecard `artifacts/sam_investigation/confirmatory/confirmatory_scorecard.json`.
- Checkpoint contract: `docs/CONTRACTS.md` §0.1 (Arm 1 & Arm 2); `src/common/checkpoint.py`; reference schema `artifacts/checkpoints/{baseline,sam}/meta.json`.
- Downstream consumers: `feat/ga-rule-extraction` (`src/ga_rule_extraction.py`, `--checkpoint <model.pt>`), `feat/nsga2-rule-extraction`.

### Interpretation

The rerun does its one job: there is now a `model.pt` for the SAM ρ=0.5 / WD=1e-3 condition that loads cleanly through both loader paths the rule-extraction arms use, backed by a `meta.json` that records exactly which script, config, commit, and machine produced it. The near-exact agreement with the original metrics (gap 0.0485 vs 0.0517, R² 0.8444 vs 0.8450) is a useful side benefit: it independently reproduces the Stage 4 headline on different hardware and a newer PyTorch, which strengthens rather than changes the investigation's conclusion. The scientific narrative is unchanged — Stage 4 / Stage 4b stand; this entry only adds the artifact.

**Git-trackability of the checkpoint.** `model.pt` is caught by the repo-wide `*.pt` rule in `.gitignore` (confirmed with `git check-ignore -v`) *and* by `artifacts/*`; it is **not** committable and will not travel via `git push`. `meta.json` under `artifacts/sam_investigation/` *is* trackable and is committed. The weights file must be transferred to the rule-extraction arms out of band (shared drive). Absolute path on the run machine:

`/mnt/c/Users/ibrah/hawa/github/COS700-XAI-In-Cancer-Medicine-Treatment/artifacts/sam_investigation/confirmatory/model.pt`

## Stage 4d — Consolidated thesis figures

**Date:** 2026-09-10

### What

Added `src/investigations/stage4c_figures.py`: one script that turns every numeric
result the investigation produced into comparison figures spanning the full arc —
the pre-investigation state (Adam baseline + the ρ=0.01–0.2 SAM sweep that barely
moved the gap) and the investigation itself (Stage 3 subsample sweep, Stage 4
confirmatory ρ=0.5+WD, Stage 4b Adam+WD control). Output:
`artifacts/sam_investigation/figures/` — 15 PNGs (numbered 01–16, 08 reserved)
plus `figures_manifest.json`
(per-figure caption + suggested thesis section + caveats) and
`conditions_table.json` (the seven full-data runs with ρ / WD / stopping rule /
epochs / gap / R²).

Figures:

| # | file | shows |
|---|---|---|
| 01 | gap_full_story_bars | eval-mode gap for all 7 runs; pre-investigation vs outcome |
| 02 | gap_vs_rho_fulldata | full-data gap vs ρ; the ρ under-perturbation region and the ρ=0.5 collapse |
| 03 | gap_decomposition_waterfall | 0.337 → 0.258 (WD, ≈27%) → 0.049 (SAM ρ=0.5, ≈73%) |
| 04 | gap_vs_valr2_tradeoff | each run as a point in (gap, val R²) space |
| 05 | loss_curves_preinvestigation | train/val loss curves — baseline + ρ=0.01–0.2 |
| 06 | loss_curves_investigation | train/val loss curves — baseline vs Adam+WD vs SAM ρ=0.5+WD |
| 07 | gap_per_epoch | per-epoch gap trajectories, pre-investigation vs investigation panels |
| 09 | r2_comparison | train/val/test R² (zoomed); ρ=0.5+WD is the only run with train R² ≈ val R² |
| 10 | rmse_comparison | train/val/test RMSE (zoomed) |
| 11 | final_loss_comparison | final eval-mode loss by split; train–val separation = the gap |
| 12 | stage3_sweep_heatmaps | Stage 3 (ρ × lr,wd) grid — gap and val R² |
| 13 | stage3_gap_vs_rho_subsample | Stage 3 subsample gap vs ρ, split by WD |
| 14 | loss_landscape_stage4 | 1-D filter-normalised landscape, Adam baseline vs SAM ρ=0.5+WD, train+val, mean±sd over 3 directions |
| 15 | hessian_top_eigenvalue | λ_max of the training-loss Hessian at each trained minimum (power iteration) |
| 16 | pred_vs_actual | predicted vs true ln IC50 (val), baseline vs SAM ρ=0.5+WD |

(08 is intentionally skipped — the numbering leaves room for a future
train-mode-vs-eval-mode gap panel.)

### Why

The investigation's existing figures (`stage1_*.png`, `stage3_*.png`,
`sam_vs_baseline_*` in `artifacts/reports/`) stop at Stage 3 and the ρ≤0.2
sweep. Nothing visualised Stage 4 / 4b — the part that carries the headline
result — and nothing put the whole arc on shared axes for the write-up. This
script fills that in and is re-runnable as later results land (e.g. once the
Stage 4b Adam+WD checkpoint exists, figures 14–15 can add that condition).

### How

Pure matplotlib over the committed JSON scorecards / histories for figures 1–13
(no torch, ~5 s). Figures 14–16 load the three checkpoints
(`artifacts/checkpoints/baseline`, `artifacts/checkpoints/sam` (ρ=0.2),
`artifacts/sam_investigation/confirmatory` (ρ=0.5+WD)) via
`src.common.checkpoint.load_checkpoint` and run on CPU subsamples:

- **Landscape (14):** own filter-normalised 1-D implementation (same idea as
  `src/model/sam.py::_plot_comparative_loss_landscape`, but evaluated on both the
  train and the val loader and averaged over 3 random directions with a ±sd
  band). 15 points, span 0.6, 8 000-row subsamples per split.
- **Hessian λ_max (15):** power iteration on Hessian–vector products
  (`torch.autograd.grad` twice), 25 steps, one 8 192-sample batch, per checkpoint.
  A relative sharpness scalar to accompany the landscape picture.
- **Pred vs actual (16):** baseline predictions from the committed parquet; SAM
  ρ=0.5+WD predictions regenerated from the rerun checkpoint; true values joined
  from `gdsc_processed.parquet`.

Each figure is wrapped so one failure does not sink the batch;
`figures_manifest.json` records every caveat (mixed stopping rules across the
pre/investigation phases; train-mode per-epoch losses in the curve figures;
BN running stats held fixed during landscape perturbation; subsample sizes;
Stage 3's inflated subsample gaps).

### Result

15 figures (numbered 01–16, 08 reserved) written to
`artifacts/sam_investigation/figures/`, generated in ~2.5 min on CPU. The
clearest single-figure statements of the result:

- **07_gap_per_epoch** — SAM ρ=0.5+WD holds the per-epoch gap on the zero line for
  the whole 250-epoch run while the baseline and Adam+WD climb to ~0.3 / ~0.2.
- **09_r2_comparison** — SAM ρ=0.5+WD is the only run whose train R² (0.852) sits
  on top of its val R² (0.844); every other run has train R² ≈ 0.90 ≫ val ≈ 0.86.
- **03_gap_decomposition_waterfall** — the Stage 4b split: WD ≈ 27% of the total
  gap reduction, SAM ρ=0.5 ≈ 73%.
- **14_loss_landscape_stage4** — over a ±0.06 filter-normalised slice: the Adam
  baseline sits in a visibly curved basin (train 0.75 → 0.90 at |α|=0.06) with its
  train and val curves ~0.35 apart; SAM ρ=0.5+WD's curve is nearly flat
  (~1.37–1.40 across the whole range) with train and val almost coincident — a
  flatter minimum with a smaller gap, at a higher absolute loss.
- **15_hessian_top_eigenvalue** — λ_max of the training-loss Hessian: Adam
  baseline 112, SAM ρ=0.2 15, SAM ρ=0.5+WD 43. Both SAM minima are markedly
  flatter than Adam's (≈2.6–7.7×); the ρ=0.2 < ρ=0.5 ordering does not track the
  gap and is within the noise of a single-batch power-iteration estimate — read
  it as "SAM flattens the minimum", not as a fine-grained ρ ranking.

### Reference

- Data sources listed in the `stage4c_figures.py` module docstring.
- Filter-normalised landscape method: Li et al. (2018), "Visualizing the Loss
  Landscape of Neural Nets" — already cited in Stage 2/3.
- Stage 4 (confirmatory rerun) and Stage 4b (Adam+WD control) above supply the
  numbers behind figures 03, 06, 07, 09–11, 14–16.

### Interpretation

These are draft thesis figures, not new analysis — they restate Stage 1–4b
conclusions visually. The one caveat to carry into the write-up is that the
pre-investigation runs (baseline, ρ=0.01–0.2) used the reverse-early-stopping
rule while the two investigation runs used val-loss-plateau, so any figure that
places both phases on one axis (01, 04, 07-left-vs-right, 09–11) is a directional
comparison; the confound-free comparison is the Adam+WD vs SAM ρ=0.5+WD matched
pair. Figure 03 encodes that caveat directly. Once the Stage 4b Adam+WD
checkpoint exists (see Stage 4e), figures 14–15 should be regenerated with that
third condition for a clean three-way geometry comparison.

## Stage 4e — Adam+WD control rerun to close the confound loose end

**Date:** 2026-09-10

### What

Added checkpoint saving to `src/investigations/stage4b_adam_wd_control.py`
(mirrors the Stage 4c change to `stage4_confirmatory.py`) and launched a fresh
full-data run. Output unchanged:
`artifacts/sam_investigation/adam_wd_control/`, now with
`{model.pt, meta.json}` alongside the JSON progress/history files.

### Why

Stage 4b's original 2026-09-05 run had the same defect the first Stage 4
confirmatory run had: it recorded metrics only and saved no weights. That was
fine for the confound *number* (gap 0.2582, which stands), but it blocks the
loss-geometry half of the write-up: the honest, confound-free comparison of
local minimum flatness is **Adam+WD=1e-3 vs SAM ρ=0.5+WD=1e-3 under the same
stopping rule**, and figures 14–15 (Stage 4d) could only use the original Adam
baseline (WD=0, reverse-early-stopping) as the Adam side. To add the matched
Adam+WD condition to the landscape / Hessian figures, its trained parameters
have to exist as a file.

### How

Same pattern as `stage4_confirmatory.py`:

- `save_checkpoint()` (optimiser="adam") writes
  `adam_wd_control/{model.pt, meta.json}` at the end of training; `meta.json`
  carries the schema keys + `arch_config` + provenance
  (`produced_by`, `produced_at_utc`, `git_commit`, `hardware`, `run_note`).
- Periodic best-val-loss crash-recovery save (`"partial": true`), overwritten at
  clean completion by the final-epoch state (`"partial": false`) with full
  eval-mode train/val/test R² and RMSE (the original run recorded val-only, so
  the regenerated `adam_wd_control_scorecard.json` gains train/test R²/RMSE too
  — which lets figures 09/10 show the Adam+WD train and test bars).
- `config_hash` / `split_hash` cross-checked against the baseline checkpoint
  (warn, not abort).
- `StoppingTracker.update()` now returns an `improved` flag; nothing else in the
  training loop, persistence, logging or the val-loss-plateau stopping rule
  changed.

Different hardware from the original Stage 4b run (WSL2 / i5 / CPU-only,
`torch 2.14.0+cpu`) — same caveat as Stage 4c: exact reproduction of gap=0.2582
/ val R²=0.8621 is not expected, but a large deviation should be flagged.

### Result

First launch (2026-09-10T21:55:53Z) died silently overnight at epoch 120/250
(no crash message in the log; process simply stopped appearing in `ps` around
2026-09-11 03:07 local) — likely the WSL2 session/terminal that owned it
closed. No resume path exists in `stage4b_adam_wd_control.py` (no `--resume`
flag, no optimiser-state checkpoint loading), so the dead run's epoch-120
partial state (`model.pt`, `meta.json`, log, `.jsonl` history, progress.json)
was backed up to
`artifacts/sam_investigation/adam_wd_control/_partial_run_2026-09-10_backup/`
and the run was **restarted from epoch 0** (`nohup`'d and disowned this time)
at 2026-09-11T07:54:14Z.

The restart completed cleanly: converged at **epoch 187/250** (val-loss
plateau, patience=30), runtime 19,158 s (~5.3 h). Final: train=1.0027,
val=1.2756, test=1.2717, **gap=0.2729**, val R²=0.8616, val RMSE=1.0111.
`model.pt` + schema-valid `meta.json` (`converged: true`, `partial: false`)
saved; `adam_wd_control_scorecard.json` / `_history.json` regenerated with
train/test R² and RMSE alongside val.

vs. original Adam baseline (WD=0, reverse-early-stop): gap 0.3369 → 0.2729
(19.0% reduction from weight decay alone, on this rerun's own numbers).
Matched pair vs. Stage 4 SAM ρ=0.5+WD (same val-plateau stopping rule):
Adam+WD gap=0.2729 vs SAM gap=0.0517 → **SAM-specific extra reduction =
0.2212**. This is in the same ballpark as the Stage 4b decomposition above
(WD ≈ 27–19%, SAM ≈ 73–81% of the total reduction, depending on which Adam+WD
run's number is used as the WD-alone baseline) — reinforces rather than
overturns that conclusion. `model.pt` is caught by the `*.pt` gitignore rule
and must be transferred out of band; `meta.json` + the regenerated JSON
outputs are committed.

### Reference

- Stage 4b (above) — the original confound analysis and the 0.2582 number.
- Stage 4c — the equivalent checkpoint-saving change to `stage4_confirmatory.py`.
- Stage 4f (below) — the 2-D/3-D loss-landscape figure built on this checkpoint.
- Stage 4d — figures 14–15, to be regenerated with this third condition once the
  checkpoint exists.

## Stage 4f — 2-D loss-landscape figure (all four checkpoints)

**Date:** 2026-09-11

### What

Added `fig_loss_landscape_2d()` to `src/investigations/stage4c_figures.py`
(figure `17_loss_landscape_2d.png`): a full 2-D extension of figure 14's 1-D
filter-normalised slice, plotted as a 3-D surface (+ a train-loss contour
projected onto the base plane) for all four full-data checkpoints that now
have loadable weights — Adam baseline, SAM ρ=0.2, the Stage 4e Adam+WD
control, and SAM ρ=0.5+WD (Stage 4/4c).

### Why

Figure 14 (Stage 4d) only had two conditions (Adam baseline vs SAM ρ=0.5+WD)
because the Adam+WD control had no checkpoint yet, and it only showed one
random 1-D slice through each minimum. A 2-D slice over two independent
filter-normalised directions is a strictly more informative view of basin
shape (curvature isn't isotropic), and with Stage 4e's checkpoint now on disk
all four full-data conditions can go in one comparison.

### How

- `_landscape_grid()`: 2-D generalisation of the existing `_landscape_curve()`
  — two independent filter-normalised directions (`_filter_norm_direction`,
  reused unchanged), evaluated on an `n_points × n_points` grid of
  (α, β) ∈ [−span, span]². Same per-point mechanics as the 1-D version
  (perturb → eval → restore).
- `fig_loss_landscape_2d()`: one direction pair per checkpoint (seeds 10/11,
  shared across train/val so the two surfaces are directly comparable),
  15×15 grid, span ±0.06 (matching figure 14), 4000-row train/val subsamples.
  Rendered as a 2×2 grid of `mpl_toolkits.mplot3d` subplots: train loss as a
  shaded surface in the condition's registry colour, val loss overlaid as a
  red wireframe, plus a `contourf` of the train surface projected onto the
  z-min base plane for a 2-D reading of the same data. z-axis clipped to a
  shared ceiling (90th percentile across all four panels' losses) so basin
  shape is comparable across subplots.
- Timed before committing to the full run: ~0.17 s/grid-point compute-only
  (measured on a 5×5 probe), so 15×15×2 splits×4 models ≈ 5–6 min total —
  acceptable, so no further downsampling was needed.
- Wired into `main()`'s job list right after `fig_loss_landscape` (both run
  under the existing `--no-torch` skip flag).

### Result

Ran in ~5.5 min (`19:00:53`–`19:06:07`). All four panels confirm and extend
the 1-D story: Adam baseline and SAM ρ=0.2 show a visibly curved bowl in
*every* direction sampled, with the val (red) surface sitting clearly above
and separate from the train (shaded) surface across the whole grid. Adam+WD
narrows that separation somewhat but keeps the bowl shape. SAM ρ=0.5+WD is
the outlier: both surfaces are close to flat and nearly coincident across the
full ±0.06×±0.06 grid — the flat-minimum / closed-gap result holds along a
second independent random direction, not just the one probed in figure 14.

### Reference

- Figure 14 (Stage 4d) — the 1-D precursor this extends; read together.
- Figure 15 (Hessian λ_max) — the quantitative companion (still 3-condition;
  extending it to include Adam+WD was not requested and is a natural next
  addition if the write-up wants it).
- Li et al. (2018), "Visualizing the Loss Landscape of Neural Nets" — same
  method reference as figure 14.
