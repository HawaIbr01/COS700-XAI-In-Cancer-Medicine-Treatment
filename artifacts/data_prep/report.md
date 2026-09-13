# Real-Data Preparation for Sakhile's/Melisa's Rule Extraction

**PROVISIONAL — Two decisions here are recommendations, not settled: Hawa needs to
confirm or override both (`DISCRETISATION_MODE`, `MIN_SAMPLES_PER_DRUG`,
`configs/data_prep.yaml`) before this data is treated as final for the write-up.**

Sakhile and Melisa can start real runs against the output today (Part 4/5 below prove
it), but nothing here should be cited in the report until that sign-off happens.

---

## 0. A correction to the task's premise (read this first)

Two claims in the original task brief turned out not to match the repo's actual state.
Flagging both loudly, per the ground rules, rather than quietly working around them.

**0.1 — NSGA-II is not "synthetic-fixtures-only."** `src/extraction/nsga.py` is a real,
already-working implementation (its own header says so, lines 20–35) and its
`load_condition_data()` (nsga.py:72–95) already reads real data:
`artifacts/data/gdsc_processed.parquet` + `splits.json` + `feature_map.json`
(all real, produced by `src/data/preprocess.py`) merged with real
`artifacts/predictions/{condition}_{split}.parquet`. `artifacts/rules/nsga_baseline.json`
and `nsga_sam.json` (committed, dated 2026-09-11) are genuine output of a real 30-run
NSGA-II search over the full, real, **all-413-drugs-pooled** dataset — not synthetic. Only
GA (`src/extraction/ga.py:82`, `raise NotImplementedError`) is actually blocked on
missing real data; NSGA-II already ran against it, just without a drug shortlist or a
reconsidered threshold.

**0.2 — The synthetic notebooks are not the extractors' schema source.** The task asked
to reuse `notebooks/01_synthetic_data_foundation.ipynb`'s per-drug CSV schema
(`cell_line_id`/`true_class`/`baseline_pred_class`/… ) because it supposedly
"established the schema `src/extraction/common.py` and Sakhile's/Melisa's extractors
expect." Both notebooks' own headers say otherwise: `01_synthetic_data_foundation.ipynb`
(cell `cded3627`) and `02_synthetic_rule_pools.ipynb` (cell `377fa4a9`) both open with
**"Section: Evaluation layer"** — they are Hawa's own fixtures for testing PROMETHEE
II/CART/GSEA, explicitly built because "real DeepCCDS predictions and real GA/NSGA-II
rule sets aren't ready yet." `common.py` never defines a CSV schema at all — it's
chromosome encode/decode only (`src/extraction/common.py:44–76`). The schema GA/NSGA-II
actually consume is `docs/CONTRACTS.md` §0.1 (lines 17–33) / §0.4 (lines 62–85),
implemented in `nsga.py`'s `load_condition_data`.

**What this changes:** Part 4 below outputs data in the *real*, already-working
contract shape (filtered + re-discretised `gdsc_processed.parquet` + `predictions/*`),
not the synthetic notebook's per-drug CSV shape — because that's what actually makes
"Sakhile and Melisa can start real runs today" true. The Part 5 instruction to verify
against "whatever loader `common.py`/the extractors actually use" independently points
the same way. A flattened, human-readable file is also produced under the exact
filenames the task requested, for manual inspection — see Part 4.3.

**0.3 — Mutation status isn't in the rule-extraction feature set yet, contrary to an
assumption in the task brief.** `src/data/preprocess.py`'s module docstring (lines
22–56) already flags this: `feature_map.json` / `gdsc_processed.parquet`'s discretised
columns are built **only** from the 38 pathway-activity scores; the 534-gene mutation
matrix exists as a raw cached `.npy` (model input only) but was never merged into the
discretised feature set GA/NSGA-II condition rules on. Per the ground rules ("reuse
locked components... don't reinvent them... if anything seems to require changing one
of these, stop and flag it"), this script does **not** invent a mutation discretisation
scheme — it attaches pathway-activity buckets exactly as already computed, and mutation
status is simply not available as a rule antecedent until that gap (already documented,
not introduced here) is closed.

---

## Part 1 — Per-drug sample distribution

Computed across all 413 drugs in `artifacts/data/gdsc_processed.parquet` (866 unique
cell lines, 319,543 total cell-line×drug rows). Script:
`src/data/analyze_drug_distribution.py`. Output:
`artifacts/data_prep/per_drug_sample_distribution.json` +
`artifacts/data_prep/per_drug_sample_distribution.png`.

| stat | value |
|---|---|
| n drugs | 413 |
| min | 112 |
| 5th pct | 355 |
| 25th pct | 782 |
| median | 815 |
| 75th pct | 857 |
| max | 865 |
| mean / std | 773.7 / 137.1 |

**Correction to the task brief's framing:** the per-drug count is *not* "highly
variable" in the way implied — it's tightly clustered (25th–75th pct = 782–857) with a
short tail. Under the task's suggested low-sanity floor (**<50 tested cell lines**),
**zero drugs** are flagged. The one genuine outlier is **`torin2`, at 112** — the next
smallest is 339 (`Remodelin`), a clean break in the distribution. `torin2` is worth
excluding regardless of the final `MIN_SAMPLES_PER_DRUG` (Part 3 already does, at any
reasonable setting).

---

## Part 2 — Discretisation threshold investigation

**Step 1 — literature search.** No full-text PDF of a GDSC resource paper (Yang et al.
2013, or a Wang et al. GDSC paper) exists anywhere in this repo — only bibliography
entries (`docs/Proposal/references.bib:479-481`, citing Yang2013). The only compiled PDF
in the repo is this project's own report (`docs/Report/main.pdf`). Reporting plainly, per
the task's own instruction not to strain: **no published convention was found in what's
actually cited in this project**, because the source material to search isn't here. The
project's own methodology text already anticipated this: `docs/Proposal/sections/methodology/ann.tex:51-52`
states the response is "discretised... by thresholding LN IC50, **with the exact
threshold fixed at the implementation stage**" — i.e. this was always deferred to
exactly this decision point, not accidentally skipped.

**Current baked-in default (for context):** `configs/pipeline.yaml:8` sets
`threshold: median # provisional`; `src/data/preprocess.py:172-176` resolves this as
**one global value — the median of `ln_ic50` over the TRAIN split, pooled across all 413
drugs** — currently `2.901433` (recorded identically in all three checkpoints' `meta.json`
`response_threshold` fields, e.g. `artifacts/checkpoints/baseline/meta.json`).

**Step 2/3 — three options, and the concrete evidence behind the recommendation.**

| | (a) single fixed global cutoff (current default) | (b) per-drug median | (c) published convention |
|---|---|---|---|
| **What it does** | One LN_IC50 value for all 413 drugs | Each drug's own train-split median | N/A — not found (Step 1) |
| **Class balance per drug** | Measured directly (below) — badly skewed for many drugs | ~50/50 by construction for every drug | — |
| **Macro-F1 fidelity fairness** | Degenerate for the worst-affected drugs | Fair across all drugs by construction | — |
| **Downside** | — | Cross-drug rule comparison less direct (a "resistant" rule for drug X and drug Y aren't pinned to the same LN_IC50 value); needs enough TRAIN rows per drug to compute a stable median (motivates Part 3) | — |

Quantified evidence for the skew claim (computed directly from
`artifacts/data/gdsc_processed.parquet`'s existing `response_class` column, i.e. option
(a) as it stands today): across the 413 drugs, the **minority-class fraction** per drug
has median 0.165 and mean 0.193 — i.e. a typical drug is already skewed roughly 84/16,
not 50/50. **20 drugs are perfectly single-class** (100% sensitive or 100% resistant —
e.g. `torin2`, `Bortezomib`, `Romidepsin`, `Tivozanib`) under the current global
threshold, and **112/413 (27%) have a minority class under 5%**. For those rules, a
trivial "always predict the majority class" model already gets a perfect or
near-perfect macro-F1 fidelity score — the rule search has nothing real to distinguish,
exactly the fairness problem the task asked to consider.

**Recommendation: (b) per-drug median**, computed from that drug's own TRAIN-split
`ln_ic50` only (mirroring the existing global convention's train-only rule, just at
per-drug granularity — see `src/data/discretisation.py` docstring). This directly fixes
the measured skew (Part 5 confirms class balance recovers to ~50/50 per drug after this
change) without touching the locked feature-side z-score discretisation, the 2%
coverage floor, or any extractor code.

**Config override:** `configs/data_prep.yaml` → `discretisation.mode`
(`DISCRETISATION_MODE`), currently set to `per_drug_median_train`. Switching to the old
global behaviour, or to a literal fixed cutoff, is a one-line edit — no rebuild
(`src/data/discretisation.py:compute_thresholds`/`discretise`).

---

## Part 3 — Drug shortlist & minimum-sample threshold

**Reasoning, tied to the locked 2% coverage floor** (`nsga.py`'s `--min-coverage`,
already flagged in that file's own comments, lines 137-147, as a fix for "a rule
matching 1-3 rows trivially hits fidelity 1.0"): a 2% coverage floor is only a
meaningful filter if 2% of a drug's usable (train-split) rows is itself a real sample
size. At `MIN_SAMPLES_PER_DRUG = 500` (≈ 500×0.8 ≈ 400 train rows per drug in this
dataset), 2% coverage ≈ **8–10 covered rows** — enough that a rule's fidelity score
isn't dominated by a handful of trivially-matched rows. Checked directly against the
real train-split counts per drug (not an 80% approximation): **379/413 drugs** have
≥500 total (≥ ~500 train) samples; the next threshold tested (600) drops to 340/413, a
much bigger cut for only 2 more guaranteed covered rows at the floor.

**Recommendation: `MIN_SAMPLES_PER_DRUG = 500`** → shortlist of **379/413 drugs**
(91.8% of the drug panel retained; only the tail — 34 drugs, `torin2` through the low-600s
range — excluded). This is a materially larger shortlist than "highly variable
counts" might suggest (see Part 1's correction); the panel is mostly uniform, with a
short excluded tail.

**Config override:** `configs/data_prep.yaml` → `drug_shortlist.min_samples_per_drug`
(`MIN_SAMPLES_PER_DRUG`), currently `500`; `drug_shortlist.low_sanity_floor` (currently
`50`, per Part 1) is always applied as an additional floor regardless.

---

## Part 4 — Real-data build

Script: `src/data/prepare_real_extraction_data.py`
(`python -m src.data.prepare_real_extraction_data`, optional `--skip-sam`).
**No retraining** — inference only, reusing `src.model.baseline.run_epoch` (the same
eval-mode code path `baseline.py`/`sam.py` already use) rather than a new forward-pass
implementation.

**Gate check (ground rules):** `artifacts/sam_investigation/confirmatory/model.pt` **is
present** on this machine (confirmed before running). Both baseline and SAM-side
predictions were generated.

**Checkpoints used:**
- `artifacts/checkpoints/baseline/model.pt` — Adam baseline.
- `artifacts/sam_investigation/confirmatory/model.pt` — SAM ρ=0.5, weight_decay=1e-3
  (**not** the older `artifacts/checkpoints/sam/model.pt`, ρ=0.2/no WD — the
  confirmatory checkpoint's own `meta.json` `run_note` states it exists specifically
  "to produce a loadable model.pt for the GA / NSGA-II rule-extraction arms").

### 4.1 — Real extractor contract (drop-in runnable today)

```
data/processed/real_extraction/
├── data/
│   ├── gdsc_processed.parquet   # filtered to 379 shortlisted drugs, re-discretised, 307,293 rows
│   ├── splits.json              # same row_ids as above, split into train/val/test
│   └── feature_map.json         # unchanged copy — features not touched
├── predictions/
│   ├── baseline_{train,val,test}.parquet   # row_id, ln_ic50_pred, class_pred
│   └── sam_{train,val,test}.parquet        # from the CONFIRMATORY checkpoint
└── meta.json                    # provenance, thresholds, shortlist, PROVISIONAL flag
```

Same column names/types as `artifacts/data/*` and `artifacts/predictions/*` — only the
row set and the `response_class`/`class_pred` values differ. Verified runnable
unmodified (Part 5):

```bash
python -m src.extraction.nsga --condition baseline --data-dir data/processed/real_extraction/data
python -m src.extraction.nsga --condition sam      --data-dir data/processed/real_extraction/data
```

(GA, once `src/extraction/ga.py`'s `NotImplementedError` is resolved, should take the
same `--data-dir` shape.)

### 4.2 — Antecedent features

Pathway-activity Low/Intermediate/High buckets carried through **exactly as already
computed** by `src/data/preprocess.py` (locked z-score k=3 partitioning, untouched) —
this script only filters rows and overwrites `response_class`. Mutation status is not
attached, per the Part 0.3 finding above (not currently part of the rule-extraction
feature set anywhere in the repo, not something this task should silently add).

### 4.3 — Human-readable combined files (task's requested filenames)

```
data/processed/real_extraction_inputs_baseline.parquet    # 307,293 rows
data/processed/real_extraction_inputs_sam_final.parquet   # 307,293 rows
```

One row per (cell line, drug) pair: `row_id, cell_name, drug_name, split`, the 38
pathway-activity feature columns, `true_ln_ic50, true_class, ln_ic50_pred, class_pred`
— both true and predicted discretised labels, as required for the accuracy term
Melisa's (not-yet-built) GA fitness function will need in addition to fidelity.

---

## Part 5 — Sanity checks

Script: `src/data/verify_real_extraction_data.py` →
`artifacts/data_prep/part5_sanity_checks.json` (`all_ok: true`).

1. **Loads through the real consuming code, unmodified** — `src.extraction.nsga.load_condition_data`
   (not a bespoke loader) successfully reads both conditions × all 3 splits; also ran
   the actual `python -m src.extraction.nsga` CLI end-to-end against this data
   (`--out-dir` pointed at a scratch dir, not `artifacts/rules/`, to avoid touching the
   real committed rule pools) — completed without error.
2. **Row count ballpark** — 307,293 rows in the output exactly equals
   `full_data['drug_name'].isin(shortlist).sum()` on the original 319,543-row dataset —
   no rows gained or dropped beyond the intended drug filter.
3. **Per-drug class balance, before/after** — spot-checked 6 drugs across the shortlist's
   count range (3 smallest, 3 largest): all land at 48–52% minority class under the new
   per-drug threshold (e.g. `alpha-lipoic acid`, one of the 20 drugs that was **100%
   single-class** under the old global threshold, is now 50.4/49.6). `class_pred`
   balance is a bit noisier (model error), but never worse than ~62/38 in the sample
   checked — nowhere near the 99/1 degenerate case the task asked to guard against.

---

## Summary of file locations

| what | path |
|---|---|
| Config (both provisional knobs) | `configs/data_prep.yaml` |
| Part 1 output | `artifacts/data_prep/per_drug_sample_distribution.{json,png}` |
| Part 2/4 discretisation logic | `src/data/discretisation.py` |
| Part 3/4 build script | `src/data/prepare_real_extraction_data.py` |
| Part 5 verification | `src/data/verify_real_extraction_data.py` → `artifacts/data_prep/part5_sanity_checks.json` |
| Real extractor-ready data | `data/processed/real_extraction/{data,predictions}/`, `meta.json` |
| Human-readable combined files | `data/processed/real_extraction_inputs_{baseline,sam_final}.parquet` |

---

**Two decisions here are provisional recommendations, not settled — Hawa needs to
confirm or override both (`DISCRETISATION_MODE`, `MIN_SAMPLES_PER_DRUG`) before this
data is treated as final for the write-up.**
