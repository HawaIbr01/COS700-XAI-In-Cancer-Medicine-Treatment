# Prompt for Claude Code: Investigate the Muted SAM Generalisation-Gap Effect

Paste everything below into Claude Code in the repo root, on the `feat/generalisation-gap-intervention` branch (create/switch to it first if not already there).

**Timeline: this needs to be closed off by Friday morning/afternoon.** Today is Wednesday. Prioritise ruthlessly — Stage 1–2 are non-negotiable, Stage 3 should be scoped down if needed to fit, Stage 4–5 are genuinely optional and should be dropped rather than rushed if time runs short. Check in with me at the end of each day (Wed/Thu) with a short status note so we can redirect early if something isn't panning out, rather than discovering on Friday morning that a stage didn't work.

**Scope discipline:** this is a diagnostic exercise to explain an already-observed result, not a new research thread. If an experiment or literature trail starts pointing toward something that would require a new model, a new dataset, or materially expands beyond "why is the Adam-vs-SAM gap small, and is it real," stop, flag it, and ask before continuing. We are not trying to open a new study — we're trying to close this one out with a defensible answer by Friday.

**Hardware:** running locally on an 8GB RAM, i5 CPU, 512GB SSD MacBook (no dedicated GPU). Angelique and Bahiya likely used similar-spec machines for their runs, so this should be broadly consistent with how the rest of the pipeline has been run so far — but scope every experiment (dataset size, batch size, sweep grid size, Hessian estimation method) to what's actually feasible on this hardware. If a planned step (e.g. full-network λmax, a large joint grid) isn't feasible at full scale locally, say so explicitly in the journal, use a reduced/subsampled version instead, and note the reduction and why it should still be informative — don't silently skip it or silently scale it down without flagging it.

---

## Context

I'm on a COS700 Honours research team. Our thesis tests whether replacing Adam with SAM (Sharpness-Aware Minimisation) in the DeepCCDS drug-response framework widens the train/validation generalisation gap enough to produce better symbolically-extracted rules downstream. Bahiya has run SAM with several rho values and the generalisation-gap difference vs. the Adam baseline is smaller than expected.

I've had a research report compiled (attached below / see `docs/sam_investigation_report.md` if I've saved it into the repo) that proposes two competing explanations:

1. **Genuine ceiling effect** — DeepCCDS is already heavily regularised (pretrained bottlenecked autoencoders, batch-norm throughout, dropout, weight decay, early stopping, ~320k samples, a fixed ssGSEA prior-knowledge pathway branch), so there's little sharp-minima "slack" left for SAM to remove. Backed by: Shin, Lee, Andriushchenko & Lee, "Critical Influence of Overparameterization on Sharpness-aware Minimization" (arXiv 2311.17539); Kaddour, Liu, Silva & Kusner, "When Do Flat Minima Optimizers Work?" (NeurIPS 2022, arXiv 2202.00661).

2. **Configuration artefacts masquerading as a ceiling**, specifically:
   - **ρ under-perturbing batch-norm affine parameters.** DeepCCDS uses BN in every layer; a ρ tuned in typical SGD/vision ranges may barely touch BN params, which carry most of SAM's benefit. Source: Müller, Vlaar, Rolnick & Hein, "Normalization Layers Are All That Sharpness-Aware Minimization Needs" (NeurIPS 2023, arXiv 2306.04226).
   - **SAM-on-Adam geometry mismatch.** SAM's perturbation is an unpreconditioned ℓ2 step; Adam's per-coordinate scaling can create "spurious sharpness minimization." Source: "Avoiding spurious sharpness minimization broadens applicability of SAM" (arXiv 2502.02407); AdaSAM (arXiv 2303.00565).
   - **Fixed-radius ρ sensitivity to weight re-scaling** across DeepCCDS's differently-scaled branches (two pretrained autoencoders + fusion head). Source: Kwon et al., ASAM (ICML 2021, arXiv 2102.11600).
   - **Large batch size (1024)** sitting in a regime where sharpness and generalisation are known to decouple. Source: Kaur, Cohen & Lipton, "On the Maximum Hessian Eigenvalue and Generalization" (2023); Keskar et al., "On Large-Batch Training for Deep Learning" (ICLR 2017, arXiv 1609.04836).
   - **Global sharpness over a multi-branch joint loss** potentially finding "fake flat minima" that look flat overall but stay sharp per-branch. Source: "Global Sharpness-Aware Minimization Is Suboptimal in Domain Generalization" (arXiv 2503.23430).

I need you to **run actual experiments to determine which explanation holds**, not just review code. Follow the staged plan below. This is time-boxed to a weekend, so prioritise Stage 1–2 first and treat Stage 3–4 as time permits.

## Ground rules

- **Grounded reasoning over favourable outcomes.** The point of this investigation is to find out what's actually true, not to find a configuration that produces a bigger generalisation gap so the thesis hypothesis "wins." If a fix is applied (larger rho, ASAM, BN-only perturbation, joint LR/WD/rho retuning, etc.), it must be justified by a specific, cited piece of literature that applies to *our* architecture/context — not selected because it happened to move the number in the direction we want. If a tactic is tried and it doesn't have a clear justification for why it should apply here (e.g. a fix designed for CNNs with no stated relevance to a tabular multi-branch autoencoder model), say so and treat a resulting change in the numbers with scepticism rather than as a resolution. Every decision in the journal should be defensible on its own terms if someone at the viva asks "why did you try that, and why do you believe the result?"
- **Orient yourself first.** Read `CONTRACTS.md`, the orientation doc (likely `COS700_Project_Orientation_and_Decisions.md`), and the current state of `feat/baseline-model` and `feat/sam-intervention` before changing anything. Confirm you understand where the Adam baseline training run and the SAM training run each live, what config format they use, and what checkpoints/logs already exist before re-running anything expensive.
- **Don't silently touch `feat/ga-rule-extraction` or `feat/nsga2-rule-extraction`.** This is an evaluation/diagnostic exercise on the model layer only.
- **Never mix synthetic and real data.** Follow the existing `tests/fixtures/synthetic/` vs `artifacts/` separation.
- **Any change to shared contracts (`schema.py`, `metrics.py`, config dataclasses) needs a flagged PR description per `CONTRACTS.md §6`** — don't make silent breaking changes.
- **Ask before anything expensive or destructive** (e.g. a full retrain sweep on real data, deleting checkpoints, HPC job submission) — propose the plan and cost estimate first, get a go-ahead, then run it.
- If something in this brief turns out to be wrong once you look at the actual code/data (e.g. batch size isn't actually 1024, SAM wrapping already looks correct, BN isn't actually used everywhere) — **stop and flag the discrepancy rather than forcing the plan to fit.**

## Staged investigation

### Stage 1 — Measure the actual gap (do this first)
- Locate the Adam baseline's training logs / re-run if needed on a small held-out slice to get train vs. validation loss (and PCC/RMSE) curves across epochs.
- Locate or reproduce the same for the SAM run(s) already completed (whatever rho values Bahiya tried).
- Plot both. Compute the numeric train–val gap at convergence for each.
- **Decision point:** if the Adam baseline's gap is already small and stable (no meaningful overfitting), that's direct evidence for the ceiling hypothesis. Note this explicitly either way — it doesn't resolve everything alone, but it's the cheapest and most informative check.

### Stage 2 — Audit the SAM configuration
Check and report on, with code references (file/line):
- Is SAM wrapping **all** trainable parameters in the joint stage, or only part of the network? Are the pretrained autoencoders frozen or fine-tuned during the SAM stage?
- What rho values were actually tried, and were they swept **jointly** with learning rate and weight decay, or swept alone on top of Adam-tuned LR/WD?
- Is weight decay the same value for the Adam and SAM runs, or was it increased for SAM (as the SAM literature typically recommends)?
- Confirm the SAM implementation's perturbation-normalisation step is correct (compare against a reference implementation, e.g. https://github.com/davda54/sam, cited in the report).
- What batch size is actually used, and does it match m-sharpness expectations (per-step batch vs. accumulated)?

Report back clearly: "config looks correct" or "found issue X at file:line" for each item — this determines whether Stage 3/4 experiments are even meaningful yet.

### Stage 3 — Joint hyperparameter sweep
- Design and run a **joint grid over (learning rate, weight decay, rho)** — not rho in isolation — on a reduced/fast-iterating slice of data, sized to what's actually tractable on an 8GB CPU-only machine within roughly a day. Keep the grid small and deliberate (e.g. a handful of points per dimension) rather than exhaustive — this is a diagnostic sweep to test the "under-perturbation" and "LR/WD mismatch" hypotheses, not a hyperparameter optimisation exercise. Reasonable rho range: extend beyond whatever was tried before, including at least one value an order of magnitude larger (batch-norm-heavy nets reportedly need rho up to ~25 per Müller et al., vs. typical SGD/vision values of 0.05–0.5) — check what's already been tried before deciding new points, and justify each grid point chosen.
- Record train/val gap for each config in a results table.

### Stage 4 — Sharpness diagnostics (optional — only if Stage 1–3 leave real ambiguity and time remains)
- Compute a sharpness proxy for the best Adam solution and the best SAM solution:
  - **Filter-normalised 1D loss slices** around each minimum, per Li et al., "Visualizing the Loss Landscape of Neural Nets" (arXiv 1712.09913) — use their filter-normalisation method so ReLU/BN scale-invariance doesn't distort the plot.
  - **Largest Hessian eigenvalue (λmax)** via power iteration (e.g. PyHessian or a lightweight custom implementation), using a subsampled batch/dataset if full-dataset estimation isn't tractable on this hardware — note the subsampling explicitly.
- Compare: is the Adam solution actually sharp? Did SAM measurably flatten it? Do these track the train/val gap or decouple (per Kaur et al.)?
- Only worth doing if Stage 1–3 haven't already given a clear, defensible answer — don't run this just because it's in the plan if the ceiling-vs-config question is already settled by cheaper evidence.

### Stage 5 — Targeted SAM variants (optional, drop first if time is short)
- Try **ASAM** (scale-invariant rho) and/or **BN-only perturbation** (SAM-ON, per Müller et al.) only if Stage 2 identified a specific, plausible mechanism (BN under-perturbation, weight-scale mismatch across branches) that these directly address. Don't try them speculatively just to see if the number moves — see the grounded-reasoning rule above.

## Deliverable: investigation journal

Write directly to `docs/investigations/sam-generalisation/journal.md` — this file already exists and has the Stage 1–2 entries written up in the required format; treat it as the source of truth for what's been done so far and **append** to it, don't overwrite or restart it. Update it **as you go**, not reconstructed at the end — commit journal updates alongside each stage's work so there's a real trail, not a single end-of-week write-up.

If a term or acronym comes up that isn't already covered in the journal's Glossary section, add a short entry for it there on first use (same style as the existing entries: what it stands for, and its purpose in plain terms) — don't assume the reader already knows SAM/optimisation jargon. Don't re-explain terms that are already in the glossary.

For every stage/step include:

- **What** was done (exact command/script/config, or code change with diff/file reference)
- **Why** it was done (which hypothesis or report finding it tests)
- **How** it was done (method, tools, any settings/hyperparameters used)
- **Result** (numbers, plots saved to disk with paths, or "no change observed")
- **Reference** — cite the specific paper(s) from the list above (or others you find necessary) that motivated this step, in the same style as our report (author, venue/year, arXiv id if applicable)
- **Interpretation** — what this result does or doesn't tell us, and whether it supports "ceiling" or "configuration issue"

End the journal with a **summary verdict section**: which hypothesis the evidence supports, what remains uncertain, and what (if anything) the team should change in the SAM training pipeline before continuing on to Melisa/Sakhile's rule extraction. Be explicit about which conclusions are well-supported by evidence gathered this week versus which are still plausible-but-untested — don't overstate certainty to produce a tidier verdict. This journal is going directly into our project report as justification for our methodology decisions, so precision on what was tried and why matters more than narrative polish.

## Note on where things stand (Stage 1–2 already done)

Stage 1 (measuring the actual gap) and Stage 2 (configuration audit) are complete — see `docs/investigations/sam-generalisation/journal.md` for full detail. Headline: the ceiling hypothesis alone is **not supported** (the Adam baseline shows a real ~0.34 generalisation gap, not a negligible one); the muted SAM effect looks more likely to be a search/configuration issue — ρ was swept in isolation rather than jointly with LR/WD, weight decay was 0 throughout, and there's a known SAM-on-Adam geometry mismatch in the literature. The ρ-vs-gap relationship across the four SAM runs tried (0.01, 0.05, 0.1, 0.2) is monotonically improving and hadn't flattened out by ρ=0.2, suggesting SAM's effective operating point may be further out than what's been tried.

One item from Stage 2 needs care rather than being treated as closed: BN under-perturbation was provisionally set aside on a "BN affine params are only 0.3% of the network" argument, but the cited paper (Müller et al., NeurIPS 2023) actually shows that BN-only perturbation can match/beat full-network perturbation *despite* being a tiny parameter fraction — so small parameter count doesn't establish small importance here. Don't treat this as ruled out; if Stage 3's joint sweep doesn't resolve things, this is a reasonable candidate for a targeted Stage 5 check, with the actual justification (not just parameter count) stated in the journal.

Proceed to Stage 3 (joint LR/WD/ρ sweep, with weight decay now included as a swept dimension rather than fixed at 0).

## Daily check-in

At the end of Wednesday and Thursday, stop and give me a short status summary: what's been confirmed, what's still open, and whether the Friday deadline still looks realistic given what's been found so far — so we can cut scope or redirect early rather than late.

Save all generated plots/tables under a clearly labelled `artifacts/sam_investigation/` (or `tests/fixtures/synthetic/sam_investigation/` if run on synthetic/reduced data — keep this distinction explicit per our existing convention) and reference their paths from the journal.