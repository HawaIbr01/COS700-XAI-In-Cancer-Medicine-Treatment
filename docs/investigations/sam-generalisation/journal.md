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

### Reference

- Foret, Kleiner, Mobahi & Neyshabur, "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021, arXiv:2010.01412) — LR reduction recommendation for SAM.
- Müller, Vlaar, Rolnick & Hein, "Normalization Layers Are All That Sharpness-Aware Minimization Needs" (NeurIPS 2023, arXiv:2306.04226) — rho scale reference for BN-heavy models.
- Kaddour, Liu, Silva & Kusner, "When Do Flat Minima Optimizers Work?" (NeurIPS 2022, arXiv:2202.00661) — ceiling conditions and SAM effectiveness.

### Interpretation

Three findings, each requiring care about what they do and don't show.

**1. ρ under-perturbation is confirmed — but there is a regime boundary at ρ≈0.5.**

The marginal gap falls from 0.83 (ρ=0.20) to 0.49 (ρ=0.50) to 0.17 (ρ=1.0). The Stage 1 finding — that the rho trend had not flattened at ρ=0.20 — is vindicated. However, the gap reduction from ρ=0.5 to ρ=1.0 is not the same kind of improvement. At ρ=1.0 the training-set losses are ~3.0 (vs ~1.3–1.5 at ρ=0.5) and val R² drops from ~0.78 to ~0.59–0.64. This is not better generalisation: the model is failing to fit the training data at all. The gap shrinks because train loss rises, not because val loss falls. ρ≥1.0 is finding flat regions that are poor solutions — the exact failure mode Kaddour et al. (NeurIPS 2022) describe: SAM can locate genuinely flat minima that happen to be bad ones, not just well-generalising ones.

**The actionable conclusion: ρ=0.50 is the right operating point.** It reduces the gap to ~0.42–0.56 (vs ~0.70–0.91 at ρ=0.20) while keeping val R² at ~0.78, essentially matching the ρ=0.20 quality. Going higher sacrifices prediction quality for a gap reduction that reflects underfitting, not generalisation.

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

- **ρ under-perturbation confirmed.** ρ=0.50 reduces mean gap by 41% vs ρ=0.20 while maintaining val R²≈0.78. ρ≥1.0 does reduce the gap further, but by causing underfitting (val R² collapses to 0.53–0.64), not by improving generalisation.
- **WD mismatch confirmed.** WD=0.001 reduces mean gap by 22% consistently across all ρ values. This is a direct configuration fix.
- **LR mismatch not supported.** LR has negligible marginal effect within the range tested.

### Recommendation

For Bahiya's full-data SAM run, the change supported by the data is:

| Parameter | Current | Recommended | Evidence |
|-----------|---------|-------------|---------|
| ρ | 0.20 | **0.50** | Stage 1 trend + Stage 3 marginal effect; largest single lever |
| WD | 0 | **1e-3** | Stage 3 marginal: 22% gap reduction across all configs |
| LR | 0.001 | 0.001 (unchanged) | Stage 3 marginal: negligible effect in tested range |

Expected effect on full-data gap: Stage 3 subsample data are not directly comparable, but the relative gap reduction at ρ=0.5 vs ρ=0.2 is ~40%. If that ordering holds at full scale, the full-data gap would move from ~0.337 toward ~0.20. This is an estimate, not a guarantee — a confirmatory full run at ρ=0.5, WD=1e-3 is the right next step before treating this as closed.

### Open question

**BN under-perturbation (Stage 5 candidate).** Müller et al. (NeurIPS 2023) show that perturbing only BN affine parameters can match or beat full-network SAM in some architectures. DeepCCDS has 12 BN layers; this was provisionally ruled out in Stage 2 on parameter-count grounds, which the Stage 2 entry noted is insufficient on its own. If the recommended ρ=0.5/WD=1e-3 run does not fully resolve the gap, BN-only perturbation is the targeted Stage 5 experiment. It is not recommended speculatively — only if the above configuration change leaves an unexplained residual gap.