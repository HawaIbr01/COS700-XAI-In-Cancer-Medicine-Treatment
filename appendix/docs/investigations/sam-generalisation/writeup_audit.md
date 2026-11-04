# Post-Rerun Write-Up Verification Audit

Date: 2026-09-13. Branch: `feat/sam-intervention`. Verification-only pass —
no training was run. Every finding below is backed by a file/line citation or
a command whose output is quoted.

---

## 1. Loss definition — **ready for write-up**

Canonical implementation: **`src/model/deepccds.py:240-251`**, function
`compute_loss(model_out, y)`:

```python
loss = nn.functional.mse_loss(model_out["prediction"], y)
if "mutation_recon" in model_out:
    loss = loss + nn.functional.mse_loss(model_out["mutation_recon"], model_out["mutation_input"])
if "drug_recon" in model_out:
    loss = loss + nn.functional.mse_loss(model_out["drug_recon"], model_out["drug_input"])
```

This is an **unweighted sum** of the three MSE terms — prediction MSE +
mutation-AE reconstruction MSE + drug-AE reconstruction MSE — exactly matching
"Eq. 2" as described in `COS700_Project_Orientation_and_Decisions.md`. No
scaling/weighting coefficients are applied to any term.

Every training/eval script that computes this loss imports this single
function — confirmed by import grep across the codebase:
`src/model/baseline.py:45`, `src/model/sam.py:58`,
`src/investigations/stage4_confirmatory.py:65`,
`src/investigations/stage4b_adam_wd_control.py:79`,
`src/investigations/stage4c_figures.py:531,702`. There is no second,
diverging implementation in any production path.

Minor note (not a problem): `src/model/train_baseline.py:107-115` contains a
byte-for-byte identical duplicate `compute_loss`, but it is only imported by
`tests/test_baseline_model.py:10` — never by a training or investigation
script. No inconsistency risk for the write-up.

---

## 2. Weight-decay scope — **ready for write-up**

- `src/investigations/stage4b_adam_wd_control.py:323`:
  `optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)`
- `src/investigations/stage4_confirmatory.py:346-349`:
  ```python
  optimizer = SAM(
      model.parameters(),
      base_optimizer_cls=torch.optim.Adam,
      rho=RHO, lr=LR, weight_decay=WD,
  )
  ```

Both are a **single optimiser call** over `model.parameters()`, where `model =
DeepCCDS(**arch_config)` (line 322 / 345 respectively in each file).
`DeepCCDS.__init__` (`src/model/deepccds.py:175-176`) registers
`self.mutation_ae` and `self.drug_ae` as submodules of the same `nn.Module`,
so `model.parameters()` — and therefore `weight_decay=1e-3` — covers the
prediction head **and** both autoencoders. There is no separate optimiser or
parameter group that scopes WD to only part of the network.

---

## 3. Prediction-loss vs. reconstruction-loss breakdown — **needs attention (partially blocked)**

None of the three scorecards (`confirmatory_scorecard.json`,
`adam_wd_control_scorecard.json`, `baseline_gap_scorecard.json`) log
prediction-MSE and reconstruction-MSE separately — each only logs the combined
`final_{train,val,test}_loss`. Confirmed by inspecting all three files' key
sets directly.

Standalone script written: **`src/investigations/write_up_loss_breakdown.py`**
(no training; loads a checkpoint, runs one eval-mode forward pass over the val
split, computes prediction MSE and both reconstruction MSEs separately).
Output: **`artifacts/sam_investigation/write_up_loss_breakdown.json`**.

**Only 2 of the 4 requested conditions could be run directly**, because their
`model.pt` files do not exist on this machine (see Item 9 — this is a real
gap, not a retraining question, so it is flagged rather than worked around by
training):

| Condition | Prediction MSE | Mutation recon MSE | Drug recon MSE | Combined (= reported loss) | Recon as % of loss |
|---|---|---|---|---|---|
| Adam baseline | 1.06066 | 0.012385 | 0.001221 | 1.07427 | 1.27% |
| SAM ρ=0.2 | 1.03326 | 0.007830 | 0.000797 | 1.04188 | 0.83% |

Cross-check: `combined_eq2_loss` reproduces the scorecards'
`final_val_loss` to 6+ significant figures for both (1.074268 vs recorded
1.074268; 1.041883 vs recorded 1.041883), confirming the script's method is
correct.

**For the two WD=1e-3 conditions, the checkpoint is missing, but the same
quantity can still be derived exactly** (not approximated) from existing
JSON, because `rmse()` (`src/investigations/stage4_confirmatory.py:229-243`)
computes prediction-only RMSE on the same eval-mode pass as `eval_split()`'s
combined loss (`src/investigations/stage4_confirmatory.py:195-206`) — so
`final_val_loss − val_rmse² ≡ mutation_recon_mse + drug_recon_mse` exactly.
This identity was verified against the two directly-measured conditions above
(exact match to 6 decimal places) before being applied to the two
checkpoint-less conditions:

| Condition | final_val_loss | val_rmse² (=prediction MSE) | Combined recon MSE (derived) | Recon as % of loss |
|---|---|---|---|---|
| Adam + WD=1e-3 | 1.275574 | 1.022386 | **0.253189** | **19.85%** |
| SAM ρ=0.5+WD=1e-3 | 1.403261 | 1.149718 | **0.253543** | **18.07%** |

**This directly answers the question the task posed.** Reconstruction loss is
~1% of total loss with WD=0, but jumps to ~18-20% with WD=1e-3 — and the two
WD=1e-3 conditions land within 0.3% of each other in absolute recon-MSE terms
regardless of optimiser (Adam vs SAM). Weight decay is measurably degrading
reconstruction quality, roughly equally for both optimisers; this is **not**
a SAM-specific side effect. The one thing not available without the missing
checkpoints is the **mutation-vs-drug split** of that combined recon number
for the two WD conditions — the derivation above gives the sum, not the two
individual terms.

**Verdict: ready for write-up on the headline finding (WD inflates
reconstruction loss ~18-20x, symmetrically across optimisers); needs
attention if the report wants the WD-condition mutation/drug split, which
requires recovering the missing checkpoints (see Item 9) — not retraining
this audit's job to do, so flagged rather than attempted.**

---

## 4. Loss-landscape figure loss definition — **ready for write-up**

Both `fig_loss_landscape()` (`src/investigations/stage4c_figures.py:571-604`)
and `fig_loss_landscape_2d()` (`stage4c_figures.py:629-696`) call the shared
helper `_eval_loss()` (`stage4c_figures.py:529-538`), which does:

```python
from src.model.deepccds import compute_loss
...
tot += compute_loss(out, y).item() * p.size(0)
```

**This is the full combined Eq. 2 loss** (prediction MSE + both
reconstruction MSEs) at every perturbed grid point — **not** a prediction-only
MSE computed separately for the figure. The y-axis label
(`stage4c_figures.py:590`) is explicit about this: `"loss (Eq. 2, MSE-based)"`.
`fig_hessian()` (`stage4c_figures.py:699-714`) uses the same `compute_loss`
for its gradient/Hessian computation.

**Wording implication for the report/captions:** the "flatter minimum" finding
in figures 14/15/17 describes flatness of the *combined* training objective's
surface, not of the prediction-error surface alone. Given Item 3's finding
that reconstruction loss is a much larger fraction of total loss for the two
WD=1e-3 conditions (~18-20%) than for the two WD=0 conditions (~1%), some of
the landscape flattening those figures show could in principle be influenced
by the reconstruction term's own geometry, not purely by prediction-error
geometry. The figures and existing journal text do not disentangle this —
worth a caveat sentence in the caption/discussion if the report wants to be
precise about what "flat" refers to.

---

## 5. Consolidated results table — **ready for write-up**

Written to **`docs/investigations/sam-generalisation/results_summary.md`**.
Covers all 7 full-data conditions (Adam baseline; SAM ρ=0.01/0.05/0.1/0.2;
Adam+WD control; SAM ρ=0.5+WD confirmatory rerun) with gap, val R², val RMSE,
train/val/test loss, epochs, stopping rule, and checkpoint loadability +
exact path for each. Every number in it is sourced from a specific committed
JSON file (sources listed at the bottom of that file).

Flags surfaced while building it (both carried into Item 9): the Adam+WD and
SAM ρ=0.5+WD checkpoint paths are recorded in `meta.json` but the `model.pt`
files themselves are not present on this machine.

---

## 6. Reproducibility check on the new checkpoint — **ready for write-up (small, explained delta)**

Original Stage 4 confirmatory run (git commit `43707e6`, before being
overwritten by the rerun) vs. the Stage 4c rerun (current
`confirmatory_scorecard.json` / `meta.json`, git commit `61d85eb`):

| Metric | Original | Rerun | Δ (rerun − original) | Relative |
|---|---|---|---|---|
| Generalisation gap | 0.05173 | 0.04850 | −0.00323 | −6.25% |
| Val R² | 0.84497 | 0.84441 | −0.00057 | −0.067% |
| Val RMSE | 1.07030 | 1.07225 | +0.00195 | +0.18% |

These are small deltas. Both runs are `converged: false` (hit the 250-epoch
safety cap rather than a clean plateau — `confirmatory_scorecard.json`,
`meta.json`), which means both trajectories were still drifting slightly at
the cutoff epoch, so some run-to-run variation is expected even under a fixed
seed. The `meta.json` provenance block records different hardware/software
between the runs (`torch 2.14.0+cpu`, differing host) than the original
2026-09-04/05 run. The direction and size of the delta (a *smaller* gap and
RMSE change well under 1%) is consistent with ordinary floating-point/
hardware nondeterminism, not with a methodology or code regression, and the
journal (`docs/investigations/sam-generalisation/journal.md:562`) draws the
same conclusion. **Not flagged as needing investigation.**

One documentation note: `confirmatory_scorecard.json` currently displays the
**rerun's** numbers, not the original run's — the original numbers were
overwritten in place (see `results_summary.md` for the git-commit citation)
and now only exist in git history and in the journal's prose. Anyone opening
that JSON file today without reading the journal would not see the original
numbers used in this comparison.

---

## 7. Outstanding open item — BN-affine under-perturbation — **visible, not decided (per instructions)**

Journal Stage 2 (`journal.md:71,78-80`) provisionally set aside "BN affine
parameters under-perturbed at low ρ" as a cause, on the grounds that BN
affine params are ~0.3% of DeepCCDS's parameters. The Stage 2 interpretation
itself flags this as methodologically weak: Müller, Vlaar, Rolnick & Hein
(NeurIPS 2023, arXiv 2306.04226) show BN-only perturbation can match
full-network SAM *despite* being a tiny parameter fraction — the opposite of
what a parameter-count argument would predict — and the journal explicitly
says the item "shouldn't be marked 'ruled out' on this reasoning alone,"
proposing a dedicated Stage 5 BN-only-perturbation experiment as the correct
follow-up if Stage 3 didn't fully resolve the picture.

**Current state, as of the journal's own later entry** (`journal.md:271`,
under Stage 3's "Open question"): the item was revisited, not dropped. It
is marked **"closed (confirmed by Stage 4b)"** — but closed by an argument,
not by running the originally-proposed Stage 5 BN-only experiment. The
reasoning given: the provisional-closure condition set in Stage 2/3 ("revisit
if weight decay accounts for most of the reduction") did not trigger — Stage
4b's confound check attributed ~72% of the gap reduction to SAM specifically
(not to weight decay), so the residual gap left over is judged too small to
be worth a dedicated BN-only follow-up. `journal.md:271` states this
explicitly: **"Stage 5 is not pursued."**

So, plainly: the specific methodological weakness Stage 2 flagged (a
parameter-count argument that Müller et al.'s own paper argues against) was
never actually re-tested — no BN-only perturbation experiment exists in this
codebase. What happened instead is that the team judged the *conclusion* it
would have tested (whether an unexplained residual gap remains) to already be
resolved by Stage 4b's numbers, making the targeted experiment moot in their
judgement. Whether "the residual gap is small enough that the original
methodological objection no longer matters" is a satisfying closure for the
thesis write-up, or whether reviewers would want to see the citation-level
concern (parameter-count reasoning contradicted by the cited paper) addressed
more directly, is **Hawa's and the team's call** — not something this audit
is deciding. Flagging only that it is visible and documented, not accidentally
dropped.

---

## 8. Journal completeness — **needs attention (two stages incomplete)**

Checked every `## Stage` header in
`docs/investigations/sam-generalisation/journal.md` for a complete
What/Why/How/Result/Reference/Interpretation set (as `###` headers or
equivalent clearly-labelled bold lead-ins) and for date ordering.

| Stage | What | Why | How | Result | Reference | Interpretation | Date order |
|---|---|---|---|---|---|---|---|
| 1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ok |
| 2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ok |
| 3 | ✓ (L90) | ✓ (L113) | ✓ (L119) | ✓ (L125) | ✓ (L199) | ✓ (L205) | ok (2026-09-02) |
| 4 | ✓ (L279) | ✓ (L289) | **missing** — content exists as unlabelled subsections ("First attempts", "Why the log came out empty", "Why the stopping criterion changed", L293-310) rather than a `### How` header | ✓ (L312) | **missing** — no `### Reference` section at all | ✓ (L327) | ok (2026-09-02/04/05) |
| 4b | ✓ (L339) | ✓ (L354) | ✓ (L385) | ✓ (L393) | ✓ (L485) | ✓ (L440) — note: appears *before* Reference, reverse of the usual order, but present | ok (2026-09-05) |
| 4c | ✓ (L496) | ✓ (L507) | ✓ (L513) | ✓ (L536) | ✓ (L554) | ✓ (L560) | ok (2026-09-10) |
| 4d | ✓ (L572) | ✓ (L608) | ✓ (L617) | ✓ (L642) | ✓ (L665) | ✓ (L673) | ok (2026-09-10) |
| 4e | ✓ (L689) | ✓ (L697) | ✓ (L709) | ✓ (L732) | ✓ (L763) | **missing** — interpretive content ("reinforces rather than overturns that conclusion") is folded into the Result section (L752-761) instead of its own header | ok (2026-09-10) |
| 4f | ✓ (L775) | ✓ (L784) | ✓ (L793) | ✓ (L815) | ✓ (L826) | **missing** — interpretive content folded into Result (L817-824) | ok (2026-09-11) |

The journal has **no entry at all for this rerun-verification audit** —
expected, since the deliverable for this pass is this file
(`writeup_audit.md`), not a journal entry; add one to `journal.md` only if
the team wants the audit itself logged as a stage.

**Verdict: mostly solid.** Stage 4's missing `How`/`Reference` headers and
Stage 4e/4f's missing `Interpretation` headers are formatting/completeness
gaps, not missing content — the substance is present as prose elsewhere in
each stage, just not under the expected header. Worth a quick pass to add the
headers before submission if the marking rubric checks for the
What/Why/How/Result/Reference/Interpretation structure explicitly per stage.

---

## 9. Checkpoint/handoff status — **needs attention (real gap, flagged rather than fixed)**

**Contractually required checkpoints (per `docs/CONTRACTS.md`, Arm 1 & Arm 2)
are fine:**
- `artifacts/checkpoints/baseline/model.pt` — present, loads cleanly (used
  directly in Item 3's script).
- `artifacts/checkpoints/sam/model.pt` (SAM ρ=0.2) — present, loads cleanly.

These are the only two checkpoints `docs/CONTRACTS.md` §0.1 actually commits
Sakhile's/Melisa's extraction arms to consuming, and both are unblocked.

**The investigation's recommended-configuration checkpoints are missing from
this machine:**
- `artifacts/sam_investigation/confirmatory/model.pt` (SAM ρ=0.5+WD=1e-3) —
  `meta.json` is present and describes a checkpoint that was built (Stage 4c,
  `journal.md:492-566`) and post-run verified two ways (canonical
  `load_checkpoint` + a GA-branch-style raw `torch.load`/`load_state_dict`,
  both producing identical predictions — `journal.md:529-550`). **The file
  itself does not exist anywhere findable on this machine** — confirmed with
  `find` across the whole repo and a broader search of `/mnt/c/Users` and the
  WSL home directory. It is correctly gitignored (`*.pt` in `.gitignore`),
  so its absence from git is expected — but the journal's recorded absolute
  path for it, `/mnt/c/Users/ibrah/hawa/github/COS700-XAI-In-Cancer-Medicine-Treatment/artifacts/sam_investigation/confirmatory/model.pt`
  (`journal.md:566`), is on a **different user directory that does not exist
  on this machine at all** (`/mnt/c/Users/ibrah/` is not present). The
  project directory has evidently moved since Stage 4c was run
  (current path is under `/mnt/c/Users/hawai/2026/UP/HONOURS 2026/...`), and
  the gitignored weights file was not carried over.
- `artifacts/sam_investigation/adam_wd_control/model.pt` (Adam+WD=1e-3
  control) — same situation: `meta.json` describes a verified, saved
  checkpoint (Stage 4e, `journal.md:685-761`), but no `model.pt` exists in
  this working tree, and no absolute path is recorded for it in the journal
  to search for.
- Figures 14/15/17 (`artifacts/sam_investigation/figures/`) that were built
  from these checkpoints are already generated, committed PNGs and are
  **not** affected — they were produced while the checkpoints still existed
  on the original run machine.

**Practical consequence:** if Sakhile's/Melisa's extraction branches are
meant to also run against the SAM ρ=0.5+WD (recommended) or Adam+WD
conditions — as opposed to just the two contractually-committed checkpoints
— **that handoff is not currently possible from this repo state.** The
weights either need to be located (most likely only ever existed on the
original run machine's local disk under the old path, which is now gone) or
regenerated by re-running `stage4_confirmatory.py` / `stage4b_adam_wd_control.py`
with checkpoint saving (as Stage 4c/4e already did once). **Per this audit's
ground rules, no retraining was attempted to fix this — it is flagged for
Hawa/the team to decide how to proceed**, not treated as resolved.

**Verdict: contractual handoff (baseline, SAM ρ=0.2) — ready. Investigation
handoff (Adam+WD, SAM ρ=0.5+WD) — needs attention, currently blocked, and the
team should decide whether the write-up needs those two checkpoints for
anything beyond what the already-committed figures and JSON scorecards
provide.**

---

## Summary

| # | Item | Verdict |
|---|---|---|
| 1 | Loss definition | ready for write-up |
| 2 | Weight-decay scope | ready for write-up |
| 3 | Prediction vs. reconstruction breakdown | needs attention — 2/4 conditions directly measured, 2/4 derived exactly from existing JSON (mutation/drug split unavailable for those 2 without the missing checkpoints) |
| 4 | Loss-landscape figure loss definition | ready for write-up — full Eq. 2 loss confirmed, with a wording caveat suggested for captions |
| 5 | Consolidated results table | ready for write-up — `results_summary.md` |
| 6 | Reproducibility check | ready for write-up — small, explained delta |
| 7 | BN-affine open item | visible and documented — closure is argument-based, not experiment-based; decision to accept that is the team's |
| 8 | Journal completeness | needs attention — Stage 4 missing How/Reference headers; Stage 4e/4f missing Interpretation headers (content present as prose, just unlabelled) |
| 9 | Checkpoint/handoff status | needs attention — contractual checkpoints fine; investigation checkpoints (Adam+WD, SAM ρ=0.5+WD) missing from disk, path in journal points to a directory that no longer exists |

## Artifacts produced by this audit

- `src/investigations/write_up_loss_breakdown.py` (new script)
- `artifacts/sam_investigation/write_up_loss_breakdown.json` (new output)
- `docs/investigations/sam-generalisation/results_summary.md` (new)
- `docs/investigations/sam-generalisation/writeup_audit.md` (this file)

No existing checkpoint, report, or journal entry was modified.
