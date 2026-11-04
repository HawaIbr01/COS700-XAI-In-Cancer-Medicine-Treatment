# SAM Generalisation-Gap Investigation Journal

**Branch:** `feat/generalisation-gap-intervention`  
**Started:** 2026-09-02  
**Deadline:** Friday 2026-09-05 morning  
**Author:** Hawa (u20418494), Evaluation arm  
**Context:** COS700 Honours thesis — investigating why the Adam-vs-SAM generalisation gap is smaller than expected.

This journal is written sequentially as work is done, not reconstructed at the end. Each entry follows the schema: **What / Why / How / Result / Reference / Interpretation.**

---

## Orientation (pre-Stage-1)

Before running any experiments, I read `docs/CONTRACTS.md`, `src/model/baseline.py`, `src/model/sam.py`, `src/model/deepccds.py`, and all existing artifacts under `artifacts/reports/`.

Key facts established:

| Item | Finding |
|------|---------|
| Baseline optimiser | Adam, `lr=0.001`, `weight_decay=0` (plain Adam, no WD) |
| SAM base optimiser | Adam, same `lr=0.001`, `weight_decay=0` |
| SAM rho values tried | 0.01, 0.05, 0.10, 0.20 |
| LR/WD joint sweep? | **No** — rho was swept in isolation on top of Adam-tuned LR/WD |
| Autoencoders frozen during SAM? | **No** — `model.parameters()` passes all params |
| Batch size | 1024 (both baseline and SAM) |
| Architecture | DeepCCDS: mutation AE (534→300→100→30) + drug AE (1024→300→100→30) + ssGSEA head (98→98→98→1) |
| BN layers | 12 total: 5 per AE (3 encoder + 2 decoder), 2 in head |
| Total parameters | ~1,093,233 (linear + BN affine) |
| BN affine params | ~3,712 = **0.3% of total** |
| Stopping rule | patience=30, min_consecutive=10, min_delta=1e-4 (identical for baseline and SAM) |
| Dropout | 0.0 (disabled, unregularised Stage 1 baseline) |

The stopping rule deliberately confirms stabilisation rather than stopping at best-val-loss: it waits until val_loss hasn't improved for 30 epochs *and* training loss is still falling. The saved checkpoint is the "deliberately-overfit endpoint" (per `baseline.py` docstring §4.2.3). This means the scorecard gap is measured at the confirmed-overfitting point, not the minimum-val-loss point.

**Gap metric note:** Per-epoch gaps in training histories = `val_loss(eval_mode) − train_loss(train_mode)`. Scorecard gaps = both in eval mode. Scorecard gaps are systematically larger because eval-mode disables BN mini-batch normalisation (uses running stats instead), which lowers train_loss in eval mode.

---

## Stage 1 — Measure the Actual Gap

**Date:** 2026-09-02

### What

Extracted and compared train/validation loss curves and convergence-point metrics for:
- Adam baseline (1 run)
- SAM with ρ ∈ {0.01, 0.05, 0.10, 0.20} (4 runs, from Bahiya's rho sweep)

All training histories already existed in `artifacts/reports/` — no re-runs were needed.

Ran: `python -m src.investigations.stage1_gap_analysis`

Plots saved to `artifacts/sam_investigation/`:
- `stage1_training_curves.png` — train and val loss curves for all 5 configs
- `stage1_gap_per_epoch.png` — per-epoch generalisation gap for all 5 configs
- `stage1_scorecard_comparison.png` — bar charts of final eval-mode gap, val_R², val_RMSE

Numeric summary: `artifacts/sam_investigation/stage1_gap_summary.json`

### Why

This is the cheapest and most informative check: if the Adam baseline's gap is already small and stable, that is direct evidence for the ceiling hypothesis (DeepCCDS is already well-regularised and SAM has little slack to work with). The size and shape of the gap also tells us whether overfitting is real or artefactual.

### How

Reading existing `artifacts/reports/baseline_training_history.json` and `sam_training_history_rho_*.json`. Scorecard data from `baseline_gap_scorecard.json` and `sam_gap_scorecard_rho_*.json`. Script: `src/investigations/stage1_gap_analysis.py`.

### Result

**Table 1 — Final eval-mode scorecard metrics at convergence**

| Config | Gap (eval) | Δ% vs baseline | val R² | val RMSE | epochs |
|--------|-----------|----------------|--------|----------|--------|
| Adam baseline | 0.3369 | — | 0.8565 | 1.0299 | 90 |
| SAM ρ=0.01 | 0.3656 | **−8.5%** ↑worse | 0.8577 | 1.0255 | 122 |
| SAM ρ=0.05 | 0.3661 | **−8.6%** ↑worse | 0.8593 | 1.0197 | 162 |
| SAM ρ=0.10 | 0.3600 | **−6.8%** ↑worse | 0.8587 | 1.0218 | 199 |
| SAM ρ=0.20 | 0.3122 | **+7.3%** ↓better | 0.8602 | 1.0165 | 215 |

(Positive Δ% = gap reduced = better; negative = gap widened = worse.)

**Key observations from per-epoch curves:**
- The Adam baseline gap became positive at epoch 6 and widened steadily to ~0.27 (train-mode metric) by epoch 80, confirming real and progressive overfitting.
- At the scorecard's eval-mode measurement, the baseline gap is 0.337 — val_loss is ~45% above train_loss.
- SAM ρ=0.01–0.10 all produced *wider* gaps than the baseline at convergence, despite training for far more epochs (122–199 vs 90). The gap curves show that small rho values slow training without meaningfully flattening the loss landscape.
- Only ρ=0.20 outperformed the baseline, reducing the gap from 0.337 to 0.312 (7.3%). Its per-epoch curve takes longer to stabilise (215 epochs) but achieves a slightly lower val_loss.
- All SAM variants achieve marginally better val_R² and val_RMSE than the baseline — the improvement is consistent but tiny (val_R² ranges 0.8577–0.8602 vs baseline 0.8565). The differences are ~0.001–0.004 in R² units, which could be within run-to-run noise (no multiple seeds were used).
- Test_R² closely tracks val_R² for all configs (within 0.001), confirming the splits are internally consistent.

### Reference

- Foret, Kleiner, Mobahi & Neyshabur, "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021, arXiv:2010.01412) — SAM's baseline claim is a measurable generalisation gap reduction. We observe this only weakly (7.3% at best).
- Kaddour, Liu, Silva & Kusner, "When Do Flat Minima Optimizers Work?" (NeurIPS 2022, arXiv:2202.00661) — analyses conditions under which SAM generalisation benefits materialise.

### Interpretation

**The Adam baseline IS overfitting** — the gap is not small. val_loss is 45% above train_loss at convergence, and the gap widened monotonically from epoch 6 onward. This does **not** support a pure "ceiling effect" interpretation; there is genuine slack for a regulariser to work with.

**However, SAM barely captures that slack.** Only ρ=0.20 improved the gap, and only by 7.3%. The three smaller rho values all *worsened* the gap, suggesting they are under-perturbing the weights relative to the loss landscape geometry of this model. The monotonic pattern (rho=0.01 worst → rho=0.20 best) is consistent with the rho being systematically too small across the entire sweep, and the "best" rho value (0.20) being the largest tried — i.e., the optimal rho for this model may lie above 0.20.

**Decision point per prompt §Stage 1:** The gap is not already small (it is ~0.337), so we do not have direct evidence for the ceiling hypothesis from this metric alone. The real question becomes: *why* is SAM failing to close this gap more substantially? This points us directly to Stage 2 (configuration audit).

**Preliminary ceiling evidence (partial):** The R² gap between train and val is only 0.046 (0.903 vs 0.857). In R² terms the model does not look dramatically overfit. The MSE loss gap looks larger because of the MSE scale — but the *relative* predictive performance gap is modest. This is consistent with the "heavy prior regularisation" argument (BN throughout, large dataset ~320k samples, ssGSEA prior knowledge branch), but is not conclusive without sharpness diagnostics.

---

## Stage 2 — Audit the SAM Configuration

**Date:** 2026-09-02

### What

Systematic audit of the SAM implementation against each configuration question raised in the investigation prompt. All code references are to `src/model/sam.py` and `src/model/baseline.py` on branch `feat/generalisation-gap-intervention`.

### Why

To determine whether the muted SAM effect is due to a genuine ceiling (the model is already well-regularised) or a configuration artefact (SAM is misconfigured for this architecture). A clean config audit tells us which Stage 3/4 experiments are meaningful.

### How

Code inspection. Reference implementation used for comparison: davda54/sam on GitHub (the de facto standard PyTorch SAM implementation, cited in multiple SAM papers).

### Result — Item-by-Item Audit

---

#### 2.1 SAM wrapping scope: all params or partial?

**Finding: SAM wraps ALL trainable parameters.**

`sam.py:1082–1087`:
```python
optimizer = SAM(
    model.parameters(),          # <- all parameters
    base_optimizer_cls=torch.optim.Adam,
    rho=rho,
    lr=LEARNING_RATE,
)
```

`model.parameters()` includes both autoencoders (mutation AE + drug AE) and the head. The autoencoders are **not frozen** during SAM training — they are fine-tuned jointly. This is consistent with `deepccds.py`'s comment that end-to-end training with Eq. 2 (joint MSE + reconstruction losses) is the design intent.

**Assessment: config looks correct.** SAM is applied to the full parameter space as intended.

**One concern flagged:** The global gradient norm used for perturbation scaling (`sam.py:198–280`, `_grad_norm()`) is computed over ALL parameters in a single L2 norm. This means the perturbation magnitude per parameter is:

    ε_i = (∂L/∂θ_i) × rho / ||∇L||₂

For a model with very differently-sized branches (mutation AE has 387k params, drug AE has 682k params, head has 20k params), the effective perturbation is dominated by the large AE gradients. This is the "fixed-radius ρ sensitivity to weight re-scaling across branches" concern from ASAM (Kwon et al., 2021).

---

#### 2.2 Rho values tried and sweep design

**Finding: rho was swept in isolation; LR and weight decay were NOT jointly re-tuned.**

The SAM runs used:
- rho ∈ {0.01, 0.05, 0.10, 0.20}
- LR = 0.001 (inherited directly from `baseline.py`)
- weight_decay = 0 (inherited — see 2.3 below)

The sweep was sequential (each rho value run independently) rather than a joint grid over (LR, WD, rho).

**Assessment: identified issue.** The SAM literature consistently notes that SAM's optimal LR and WD differ from Adam's. SAM's perturbation adds an implicit regularisation effect that changes the effective learning rate, and the base Adam step needs re-tuning. Running rho sweeps at the Adam-tuned LR without WD is a plausible source of under-performance.

**Rho range assessment:** Values tried (0.01–0.20) are within the typical SGD/vision-task range reported in the original SAM paper. However, Müller et al. (2023) report that BN-heavy networks may need rho up to ~25 (absolute scale). Even accounting for the difference in loss scale (our MSE loss vs cross-entropy), the upper end of the sweep (0.20) shows the best result and is clearly not converging toward a maximum — this is consistent with the optimal rho lying above 0.20.

---

#### 2.3 Weight decay: same for Adam and SAM?

**Finding: weight decay = 0 for BOTH baseline and SAM.**

`baseline.py:197`:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# plain Adam, no weight decay
```

`sam.py:1082–1087`:
```python
optimizer = SAM(
    model.parameters(),
    base_optimizer_cls=torch.optim.Adam,
    rho=rho,
    lr=LEARNING_RATE,
    # no weight_decay kwarg — defaults to 0
)
```

**Assessment: identified issue.** The SAM literature (Foret et al. 2021, Zhuang et al. 2022 "Surrogate Gap Minimization") typically recommends non-zero weight decay alongside SAM, as WD provides additional regularisation that interacts with SAM's sharpness penalty. With WD=0, SAM is operating without complementary L2 regularisation. This is a direct consequence of the LR/WD sweep not being done.

---

#### 2.4 SAM perturbation-normalisation: is the implementation correct?

**Finding: the implementation is correct for standard SAM but uses global (not per-parameter) gradient norm.**

The perturbation step (`sam.py:197–225`) computes:
```
grad_norm = ||∇L||₂  (global, over all parameters)
scale     = rho / (grad_norm + 1e-12)
ε_i       = grad_i × scale   (for each parameter i)
```

This matches the reference implementation (davda54/sam `first_step`). The reference SAM paper (Foret et al., eq. 2) defines the perturbation as:

    ε̂(w) = rho × ∇wL(w) / ||∇wL(w)||₂

This is correctly implemented. The SAM implementation also correctly:
- Uses `_disabled_running_stats(model)` for the second forward pass to prevent BN running statistics from being corrupted by perturbed weights (`sam.py:312–352`). This matches the SAM-BN best practice.
- Restores `e_w` via `pop` in `second_step` (`sam.py:234–265`) to avoid accumulation.

**Assessment: config looks correct** for standard unpreconditioned SAM. No implementation bugs found.

**One nuance:** The implementation does not use *m-sharpness* (Kwon et al. 2022), which sub-samples the gradient norm over mini-batches to get a sharper, more localised perturbation estimate. Standard SAM at batch size 1024 estimates sharpness over the full batch, which Kaur et al. (2023) and Keskar et al. (2017) note may decouple from generalisation at large batch sizes.

---

#### 2.5 Batch size and m-sharpness

**Finding: batch_size = 1024 (large batch); no m-sharpness.**

`baseline.py:57`:
```python
BATCH_SIZE = 1024
```

`sam.py:68–74` (imports from baseline):
```python
from src.model.baseline import BATCH_SIZE  # 1024
```

At batch size 1024, sharpness estimates are averaged over 1024 samples per step. Keskar et al. (2017) and Kaur et al. (2023) both show that large-batch training tends to converge to sharper minima AND that the correlation between sharpness (as measured by λ_max or SAM's ε-ball loss) and generalisation weakens at large batch sizes.

For m-sharpness, the batch would need to be sub-sampled further (e.g. batch size 32–256) for each perturbation gradient estimate. This is not currently implemented.

**Assessment: identified issue.** Large-batch SAM is a known weaker regime. The dataset contains ~320k samples; with batch_size=1024, each epoch covers ~312 gradient steps. The sharpness estimate per step is computed over 1024 samples (the full batch), not a sub-batch.

---

#### 2.6 SAM-on-Adam geometry mismatch

**Finding: standard unpreconditioned SAM is applied to Adam-preconditioned geometry.**

Adam maintains per-parameter adaptive learning rates (using first and second moment estimates). SAM's perturbation is an *unpreconditioned* ℓ₂ step — it treats all parameter coordinates equally regardless of Adam's per-coordinate scaling. This creates a geometry mismatch: Adam navigates a preconditioned space, but SAM's ε-ball is defined in the original (non-preconditioned) ℓ₂ space.

The consequence is that parameters with small Adam step sizes (e.g. rarely-updated feature directions) receive proportionally larger perturbations relative to their typical update magnitudes, while high-curvature directions (where Adam is more aggressive) may be under-perturbed.

**Assessment: identified issue.** This is the "spurious sharpness minimization" concern described in "Avoiding spurious sharpness minimization broadens applicability of SAM" (arXiv:2502.02407) and AdaSAM (arXiv:2303.00565). Since Adam is the base optimiser here, the geometry mismatch is directly applicable.

---

#### 2.7 BN affine parameter representation

**Finding: BN affine parameters are 0.3% of total parameters (~3,712 / ~1,093,233).**

BN layers: 12 total (5 per AE encoder/decoder, 2 in head).  
BN affine params (γ, β per channel): 3,712.  
Total params: ~1,093,233.

The Müller et al. (2023) "Normalization Layers Are All That SAM Needs" result applies specifically when BN parameters *dominate* the sharpness signal. In DeepCCDS, BN affine parameters are 0.3% of total parameters. The large autoencoder linear layers (mutation AE: 387k params, drug AE: 682k params) dominate the parameter space.

**Assessment: BN under-perturbation is unlikely to be the primary issue here.** The Müller et al. finding is less directly applicable to this architecture than to, say, a ResNet where BN layers are a larger fraction of parameters relative to convolution layers.

---

#### 2.8 Multi-branch joint loss and "fake flat minima"

**Finding: SAM optimises the joint loss (MSE + mutation AE reconstruction + drug AE reconstruction) globally.**

`deepccds.py` (compute_loss, Eq. 2): `loss = prediction_MSE + mutation_recon_MSE + drug_recon_MSE`.

SAM perturbs weights in the direction of steepest ascent on this joint loss. A minimum that is flat in the joint-loss surface may still be sharp per-branch — the AE reconstruction loss terms can mask sharpness in the prediction head.

**Assessment: plausible but not confirmed.** The "global sharpness over multi-branch joint loss" concern (arXiv:2503.23430) applies here in principle. However, verifying this requires branch-level loss decomposition, which is not yet done.

---

### Stage 2 Summary

| Item | Assessment | Issue? |
|------|-----------|--------|
| SAM wrapping scope | All parameters (correct) | No |
| Autoencoders frozen? | No — jointly trained (correct) | No |
| rho sweep design | Isolated (not joint LR/WD sweep) | **Yes** |
| Weight decay | 0 for both (not recommended for SAM) | **Yes** |
| SAM perturbation normalisation | Correct (matches davda54 reference) | No |
| Batch size / m-sharpness | 1024 / no m-sharpness | **Yes (soft)** |
| SAM-on-Adam geometry | Unpreconditioned ℓ₂ on Adam-preconditioned space | **Yes** |
| BN under-perturbation | BN = 0.3% of params; unlikely dominant issue | No |
| Multi-branch joint loss | Plausible concern; unverified | Flagged |

**Overall Stage 2 verdict:** Three clear configuration issues found (no joint LR/WD/rho sweep, WD=0, SAM-on-Adam geometry mismatch) plus one soft issue (large batch size). The BN under-perturbation hypothesis is unlikely to be the primary driver given the architecture's parameter distribution. The configuration issues are sufficient motivation to proceed to Stage 3 (joint hyperparameter sweep).

### Reference

- Foret et al., "SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021, arXiv:2010.01412) — perturbation formula, reference implementation.
- Müller, Vlaar, Rolnick & Hein, "Normalization Layers Are All That Sharpness-Aware Minimization Needs" (NeurIPS 2023, arXiv:2306.04226) — BN-layer focus finding; less applicable here given BN param fraction.
- Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks" (ICML 2021, arXiv:2102.11600) — scale-invariant perturbation; addresses the multi-branch weight-scale mismatch.
- "Avoiding spurious sharpness minimization broadens applicability of SAM" (arXiv:2502.02407) — Adam geometry mismatch.
- Kaur, Cohen & Lipton, "On the Maximum Hessian Eigenvalue and Generalization" (2023) — large-batch sharpness–generalisation decoupling.
- Keskar, Mudigere, Nocedal, Smelyanskiy & Tang, "On Large-Batch Training for Deep Learning" (ICLR 2017, arXiv:1609.04836) — large-batch convergence to sharp minima.

### Interpretation

Three identified configuration issues could each contribute to the muted SAM effect:

1. **rho under-perturbation + no joint LR/WD sweep** — most likely primary cause. The rho values tried (0.01–0.20) show a monotonic trend (higher rho → better), and rho=0.20 is the best but clearly not converged. The Adam-tuned LR (0.001) and WD=0 may not be appropriate for SAM.

2. **SAM-on-Adam geometry mismatch** — plausible secondary cause. The unpreconditioned ℓ₂ perturbation step in Adam's non-Euclidean geometry creates "spurious sharpness minimisation" — SAM may be flattening directions that Adam had already handled efficiently, while missing directions that matter for generalisation.

3. **Large batch size** — a soft contributing factor. At batch 1024, sharpness estimates are noisier and the sharpness–generalisation link is weaker. But this is not easily changed without touching the baseline contract.

The ceiling hypothesis is **not ruled out** — the architecture's heavy regularisation (BN throughout, 320k samples, prior-knowledge ssGSEA branch) genuinely limits how much additional regularisation SAM can provide. But the configuration issues are sufficient alternative explanations that must be tested first (Stage 3).

---

## Wednesday Check-in (2026-09-02)

**What's been confirmed:**
- Stage 1 complete: all training artifacts read, plots generated, numeric summary saved to `artifacts/sam_investigation/`. The baseline shows real overfitting (gap=0.337) but SAM closes it only marginally (best: 7.3% at ρ=0.20). The rho values smaller than 0.20 all worsened the gap.
- Stage 2 complete (audit): three configuration issues identified — (a) rho swept in isolation without joint LR/WD retuning; (b) weight_decay=0 for both baseline and SAM, not recommended for SAM; (c) SAM-on-Adam geometry mismatch. BN under-perturbation hypothesis is unlikely to be primary given BN params = 0.3% of total.

**What's still open:**
- Stage 3 (joint LR/WD/rho sweep on reduced data) — not yet designed or run.
- Stage 4 (sharpness diagnostics — optional) — deferred until Stage 3 is done.
- Stage 5 (ASAM/BN-only variants — optional) — not planned unless Stage 2 had implicated BN under-perturbation more strongly; BN-only perturbation is unlikely to be the fix here.

**Friday deadline assessment:** Still realistic. Stage 3 is the critical remaining piece. I need to design a tractable joint grid (small enough to run on 8GB CPU within ~1 day) and run it Thursday. If the grid results are clear, Stage 4 can be skipped. Stage 5 can be dropped (ASAM is not the strongest hypothesis for this architecture).

**Scope flag:** If the Stage 3 sweep suggests the optimal rho is very large (>5, as per Müller et al.'s BN-heavy regime) and the results are still ambiguous, I will flag this explicitly rather than expanding scope — the investigation mandate is to determine "why is the gap small," not to find a configuration that maximally widens the gap.

---

## Stage 3 — Joint Hyperparameter Sweep

*[To be filled in during Thursday 2026-09-03]*

---

## Summary Verdict

*[To be filled in after Stage 3]*

---

## Appendix: Artifact Paths

| Artifact | Path |
|----------|------|
| Stage 1 training curves | `artifacts/sam_investigation/stage1_training_curves.png` |
| Stage 1 per-epoch gap | `artifacts/sam_investigation/stage1_gap_per_epoch.png` |
| Stage 1 scorecard bar chart | `artifacts/sam_investigation/stage1_scorecard_comparison.png` |
| Stage 1 numeric summary | `artifacts/sam_investigation/stage1_gap_summary.json` |
| Stage 1 analysis script | `src/investigations/stage1_gap_analysis.py` |
| Existing baseline training history | `artifacts/reports/baseline_training_history.json` |
| Existing SAM training histories | `artifacts/reports/sam_training_history_rho_*.json` |
| Existing scorecards | `artifacts/reports/{baseline,sam}_gap_scorecard*.json` |
