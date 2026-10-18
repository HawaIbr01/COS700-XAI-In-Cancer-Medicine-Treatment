# Team discussion brief: Stage 4b — Adam+WD confound check

**Prepared by:** Hawa
**For:** team discussion once Stage 4b results are in
**Related:** `docs/investigations/sam-generalisation/journal.md`, Stage 4 / Stage 4b

## The one-paragraph version

Stage 4's headline result — SAM (ρ=0.5, WD=1e-3) reduces the generalisation gap by
84.6% versus the Adam baseline (WD=0) — changes two things at once: SAM's ρ and
weight decay. Weight decay is a generic regulariser unrelated to SAM's mechanism,
and our own Stage 3 sweep showed it accounts for a ~22% mean gap reduction on its
own. Stage 4b runs a matched Adam+WD=1e-3 control to find out how much of the 84.6%
is actually SAM, and how much is just "someone finally turned WD on."

## Why this matters for the thesis, not just the numbers

Our core thesis claim is specifically that **SAM's flatness-seeking geometry**
drives rule-quality improvement — not "any regularisation helps." That's the basis
for our positioning against the explanation-guided-training literature (R2ET, REGEX,
etc.): SAM is explanation-agnostic, so any rule-quality gain is an emergent
byproduct of geometric regularisation, not a designed objective. If most of Stage
4's gap reduction turns out to be attributable to weight decay rather than SAM's
perturbation step, that specific novelty argument weakens — even though the
practical finding ("properly-tuned SAM beats an under-tuned baseline") would still
be true.

## Two outcomes, two different conversations

**A. Adam+WD closes most of the gap** (lands well below the ~0.15–0.20 range)
→ Weight decay is the primary driver. SAM's ρ tuning is a secondary contributor on
top of a properly regularised baseline, not the main story. Implications to discuss:
- Does the thesis narrative need to reframe around "SAM adds a smaller marginal
  improvement over a correctly regularised baseline" rather than "SAM substantially
  outperforms Adam"?
- Does this change what checkpoint should feed rule extraction — SAM+WD, or is
  Adam+WD alone a legitimate comparison point too?
- Does the SAM-novelty-vs-explanation-guided-training argument need additional
  support to survive this (e.g. isolating the geometric contribution some other way)?

**B. Adam+WD barely moves** (stays close to the original 0.337)
→ Weight decay alone doesn't explain it; SAM's geometry is doing real work. This
strengthens the existing Stage 4 framing — the control becomes supporting evidence
rather than a defensive footnote, and the current thesis narrative holds without
change.

## The concrete decision to make as a team

**Should Bahiya's official `feat/sam-intervention` checkpoint be retrained at
ρ=0.5/WD=1e-3 (Stage 4's config) and used for downstream rule extraction (Melisa's
GA, Sakhile's NSGA-II), replacing the original ρ=0.2/WD=0 checkpoint?**

This hasn't been done yet — everything in Stages 3/4/4b has been diagnostic work
run separately, without touching Bahiya's branch or checkpoint. Nothing has been
overwritten. This is the point where that changes, if the team agrees it should.

Relevant to factor in regardless of outcome A or B above: the recommended config
(ρ=0.5, WD=1e-3) reduces the gap substantially either way — the Stage 4b result
mainly changes *why*, and how much of the credit belongs to SAM specifically versus
generic regularisation. Whether that distinction changes the retrain decision, or
just changes how we write about it, is worth deciding explicitly.

## Status at time of this brief

- Stage 4 (SAM confirmatory, full data): complete. Original run, weights not saved: gap=0.0517, val R²=0.8450.
  Saved checkpoint (rerun): gap=0.0485, val R²=0.8444.
- Stage 4b (Adam+WD control): *[in progress / complete — update before the meeting]*
- No changes made to Angelique's or Bahiya's branches or checkpoints.
