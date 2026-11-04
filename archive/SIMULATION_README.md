# COS700 Part 4 — NSGA-II rule extraction, simulation harness

Self-contained standalone environment for building and debugging
`src/extraction/nsga.py` before Arm 1 (baseline-model) and Arm 2
(sam-intervention) deliver real artifacts. Everything here conforms to
Branch Contracts §0.1–§0.4 exactly, so swapping in real data later requires
no code changes — only replacing files under `artifacts/`.

## Layout

```
configs/pipeline.yaml          single source of truth (§0.2)
src/common/                    config.py, metrics.py, schema.py, checkpoint.py (§0.3)
src/extraction/
  common.py                    shared chromosome encoding + operators (prototype)
  nsga.py                      the actual NSGA-II implementation (§4.4)
scripts/simulate_data.py       generates everything Arm 1/2 would normally produce
artifacts/                     generated — data/, checkpoints/, predictions/, rules/
```

## Running it

```bash
# 1. Generate synthetic data/checkpoints/predictions (Arm 1 + Arm 2 stand-ins)
python scripts/simulate_data.py

# 2. Smoke test — tiny pop/gens, fast iteration while debugging
python -m src.extraction.nsga --condition baseline --n-runs 3 --pop 12 --n-gen 10
python -m src.extraction.nsga --condition sam       --n-runs 3 --pop 12 --n-gen 10

# 3. Full run — pop=40, gens=50, runs=30 from pipeline.yaml (~1 min per condition)
python -m src.extraction.nsga --condition baseline
python -m src.extraction.nsga --condition sam
```

Output: `artifacts/rules/nsga_baseline.json`, `artifacts/rules/nsga_sam.json`.
Both validate against `src/common/schema.py`'s `validate_rule_pool` on write —
if either fails, the script raises before touching disk.

## Synthetic data design

`scripts/simulate_data.py` plants two "true" decision rules so the search has
something real to recover, not pure noise:

- `EGFR_expr High & TP53_expr Low → resistant`
- `MYC_expr High → resistant`

Label noise differs by condition — **baseline 25%, sam 10%** — modelling
the fact that SAM regularisation (§4.5) is supposed to reduce the
generalisation gap. This gives you a checkable property once Evaluation
compares the two: sam's rule pool should show higher average fidelity for
comparable rule lengths.

## Known finding: degenerate rules at the default settings

Running the full pipeline surfaced something worth flagging rather than
quietly fixing. The two objectives specified in §4.4 are **fidelity** and
**length only** — coverage isn't an objective. Consequence, confirmed by an
actual run: **96 of 105 baseline rules (91%) cover under 1% of rows and hit
fidelity 1.0 trivially** — predicting the majority class among 1–3 rows is
guaranteed to match, so these dominate on the fidelity axis for free. This
isn't a bug in the implementation; it's what the spec as written produces.

The branch contract already flags this as an open question ("whether
accuracy is folded in as a third objective or left to Evaluation" — §4.4
"Likely to change"). This run is concrete evidence it needs resolving
*before* Evaluation gets the pools, not after — a PROMETHEE ranking over a
pool that's 91% statistical noise isn't meaningful.

A `--min-coverage` flag is provided as an opt-in fix (default `0.0`, so the
spec-exact behaviour ships unchanged unless you ask for it):

```bash
python -m src.extraction.nsga --condition baseline --min-coverage 0.02
```

This zeroes a genotype's fidelity below the coverage floor, so it can never
survive `selNSGA2`'s dominance check against a genotype with real coverage.
Tested result: 105 degenerate rules → 6 genuinely informative ones spanning
length 0–7 and coverage 2%–100%, including a rule that recovers the planted
`MYC High → resistant` structure almost exactly (fidelity 0.446, coverage
16.6%). Recommend raising this with Melisa/Hawa before the real run —
whatever's decided should apply to GA too, or the GA-vs-NSGA comparison in
§4.6 is comparing a filtered pool against an unfiltered one.

## Swapping in real data

Once Arm 1 and Arm 2 deliver:

1. Replace `artifacts/data/*`, `artifacts/checkpoints/*`, `artifacts/predictions/*`
   with their real outputs (same filenames, same schema).
2. Confirm `config_hash` matches across everything — if it doesn't, you're
   comparing incompatible runs (Branch Contracts §6).
3. Re-run `python -m src.extraction.nsga --condition {baseline,sam}` — no
   code changes needed.

## Integration points still open

- `src/extraction/common.py` is a solo prototype. Diff against Melisa's GA
  arm once it lands and reconcile — the two extractors must use identical
  operators for the comparison in §4.6 to mean anything.
- `metrics.stability()` is an intentional `NotImplementedError` stub —
  it's Hawa's arm to define (§4.6), not needed for extraction.
- Which split fidelity is fit against (`train`, here) isn't pinned down in
  the branch contract — confirm this matches Melisa's GA arm.
