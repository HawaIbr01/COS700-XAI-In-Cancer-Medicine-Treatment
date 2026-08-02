# feat/baseline-model — Stage 1 Baseline (Angelique Breedt)

**Branch:** `feat/baseline-model`
**Covers:** Proposal Section 4.2 (ANN: Baseline Training and Overfitting Diagnosis)
**Last updated:** 2 August 2026

---

## 1. Summary for my supervisor (short version)

I built and verified the Stage 1 baseline pipeline for our Red XAI diagnostic
framework: the DeepCCDS architecture, the data pipeline feeding it, and the
deliberately-unregularised training run that establishes our "Fragility Baseline"
(the overfit checkpoint that Stage 2's SAM regularisation will later be compared
against).

**What's built and verified:**
- DeepCCDS architecture (pathway input + mutation autoencoder + drug autoencoder +
  feedforward head) implemented to match the published paper's exact specifications
  (Wu et al., 2025) — I cross-checked every layer size against their Experimental
  Section.
- A full data pipeline wired to real GDSC data: I use the DeepCCDS authors' own
  precomputed pathway-activity, mutation, and drug-fingerprint features (published
  directly on their GitHub), verified against the real files (shapes matched their
  paper exactly: 878 cell lines × 38 pathways, 878 × 534 driver genes, 413 drugs ×
  1024-bit fingerprints, 319,543 total cell-drug pairs matching their stated dataset
  size exactly).
- 17 automated unit tests, all passing, covering the architecture, the loss
  function, the data loaders, and the overfitting-detection logic.
- I ran a 5-epoch correctness check on two independent machines (my own machine and
  a cloud sandbox) and got matching results to 3-4 decimal places — train R²≈0.84,
  val R²≈0.83 — confirming the pipeline is genuinely correct, not a fluke.
- A full training run (using the real "stop once the generalisation gap stabilises"
  rule from our proposal, not a fixed epoch count) is running now.

**What's still open, honestly:**
- The full training run needs to finish and be reported (in progress as of tonight).
- I used the DeepCCDS authors' precomputed pathway/mutation/drug features rather
  than independently re-deriving them from raw GDSC files (which would require
  building a GSEA/Random-Walk-with-Restart pipeline from scratch — a substantial
  independent undertaking). **I'd like your input on whether this is an acceptable
  scoping decision for the proposal**, or whether the team should independently
  derive these features ourselves for full methodological independence.
- The Clever Hans qualitative sanity check (Section 4.2.4) still needs a small
  update to point at the real trained model instead of an earlier placeholder.
- The discretised Low/Intermediate/High gene-expression bins that Melisa and
  Sakhile need for rule extraction come from a separate raw-GDSC-file pipeline
  that still needs the RNA-seq expression file to complete.

---

## 2. What this branch actually does

This implements Proposal Section 4.2: it trains DeepCCDS with a standard Adam
optimiser, **no regularisation**, deliberately pushed past the point of good
generalisation, using a "reverse early stopping" rule — training doesn't stop when
validation loss is best, it stops once validation loss has clearly plateaued or
worsened for 10 consecutive epochs, confirming the model has settled into an
overfit state. That overfit checkpoint is our **Fragility Baseline**: the reference
point Stage 2 (SAM regularisation, Bahiya's branch) will be compared against to
show whether flattening the loss landscape actually improves explanation quality.

---

## 3. Full directory structure

```
feat-baseline-model/
├── BASELINE_README.md              <- this file
├── requirements.txt                <- Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── precomputed_inputs.py   <- loads the DeepCCDS authors' real pathway/
│   │   │                              mutation/drug feature files + train/val/test splits
│   │   ├── dataset.py              <- PyTorch Dataset classes (DeepCCDSDataset = real
│   │   │                              3-branch data; GDSCDataset = older raw-GDSC path)
│   │   └── preprocessing.py        <- raw-GDSC-file pipeline (GDSC1+GDSC2 merge,
│   │                                   cell line ID mapping, Z-scoring, expression
│   │                                   binning) — separate, still-incomplete path
│   │
│   └── model/
│       ├── __init__.py
│       ├── deepccds.py             <- the DeepCCDS architecture itself
│       ├── train_baseline.py       <- Stage 1 training loop (real + placeholder paths)
│       └── clever_hans_audit.py    <- overfitting sanity check (Section 4.2.4)
│
├── tests/
│   ├── __init__.py
│   ├── test_baseline_model.py      <- 10 tests: architecture, loss, stopping logic
│   └── test_precomputed_data.py    <- 7 tests: data loaders, real dataset class
│
├── reference_run/                  <- my verified 5-epoch correctness-check output
│   ├── baseline_adam_real_summary.json
│   └── baseline_adam_real_history.json
│
├── data/                           <- NOT COMMITTED TO GIT (see Section 5)
│   ├── deepccds_precomputed/       <- the 6 CSVs from the DeepCCDS GitHub repo
│   └── raw/                        <- (only needed for the raw-GDSC path, optional for now)
│
└── checkpoints/                    <- NOT COMMITTED TO GIT — created when training runs
    ├── baseline_adam_real.pt
    ├── baseline_adam_real_summary.json
    └── baseline_adam_real_history.json
```

---

## 4. File-by-file breakdown

### `src/data/precomputed_inputs.py`
Loads four things straight from the DeepCCDS authors' own published data (see
Section 6 for why I used these instead of building them from scratch):
- `load_pathway_activity()` — 878 cell lines × 38 KEGG pathway activity scores
  (their pre-computed ssGSEA output)
- `load_mutation_matrix()` — 878 cell lines × 534 binary driver-gene mutation calls
  (COSMIC Cancer Gene Census genes)
- `load_drug_fingerprints()` — 413 drugs × 1024-bit Morgan molecular fingerprints
- `load_split()` — the train/val/test row lists (cell index, drug index, LN_IC50
  target) for a given split

I verified every one of these shapes against the numbers stated in the actual
published paper before trusting them.

### `src/data/dataset.py`
Two PyTorch `Dataset` classes:
- **`DeepCCDSDataset`** — the one actually in use. Wraps the precomputed files
  above into `(pathway_activity, mutation, drug_fingerprint, LN_IC50)` samples.
  Built deliberately as index lookups into three small matrices rather than one
  huge pre-expanded tensor, to keep memory usage flat regardless of dataset size
  (this matters — an earlier attempt at expanding everything into one big merged
  table caused out-of-memory crashes).
- **`GDSCDataset`** — the older class for the raw-GDSC-file path (still used by
  `preprocessing.py`). Kept for the expression-binning work Melisa/Sakhile need.

### `src/data/preprocessing.py`
A separate, still-incomplete pipeline that works directly from the raw GDSC1/GDSC2
Excel files, `model_list.csv`, and `driver_genes.csv` (all real files I've
downloaded and tested this against). Handles:
- Merging GDSC1 + GDSC2 (I verified this reproduces the exact de-duplication rule
  the DeepCCDS paper describes — keeping the GDSC2 value when a cell-drug pair
  appears in both)
- Cell line ID mapping (COSMIC_ID ↔ SANGER_MODEL_ID)
- Z-score normalisation and Low/Intermediate/High expression binning

This path still needs the RNA-seq gene expression file to fully complete — it's
not used for the ANN training right now (that uses the precomputed pathway
activity instead), but it's the source of the discretised expression bins that
Melisa and Sakhile's rule-extraction algorithms need.

### `src/model/deepccds.py`
The DeepCCDS architecture itself, matching the published paper's Experimental
Section exactly:
- Pathway branch: takes the 38-dim pathway activity directly (ssGSEA is a
  preprocessing step, not a learned layer — I corrected an earlier version of
  this that had it wrong)
- Mutation & drug autoencoders: encoder hidden dims (300, 100), 30-dim bottleneck,
  BatchNorm + ReLU, Sigmoid output (matching the paper's grid-search result exactly)
- Feedforward head: two hidden layers sized to match the 98-dim concatenated input,
  BatchNorm + ReLU, unactivated single-neuron output for LN_IC50 regression
- Combined training loss (their Eq. 2): prediction MSE + both autoencoders'
  reconstruction MSE, jointly minimised

### `src/model/train_baseline.py`
The Stage 1 training loop. Two paths:
- **`train_baseline_real()`** — the one I actually use. Trains the full
  architecture on real data, plain Adam (lr=1e-3, batch=1024, matching the paper's
  own grid search), no regularisation, "reverse early stopping" (stops once
  validation loss has plateaued/worsened for 10 straight epochs).
- **`train_baseline_proxy()`** — an earlier fallback path using a random
  projection standing in for pathway data, kept only as a plumbing test. Not
  meant to produce real results — clearly flagged with a `WARNING` field in its
  output.

### `src/model/clever_hans_audit.py`
Gradient-based attribution check (Section 4.2.4) — looks at which genes the
overfit model relies on most, flags biologically implausible ones. **Still points
at the older placeholder path** — needs a quick update to audit the real trained
model instead. On my list to finish.

### `tests/test_baseline_model.py` + `tests/test_precomputed_data.py`
17 automated tests total, all passing, covering: architecture output shapes, the
combined loss calculation, the generalisation-gap stopping logic, and the real
data loaders (using small synthetic files so tests run fast without needing the
full dataset).

### `reference_run/`
The exact output from my verified 5-epoch correctness check, kept as a reference
point. I ran this on two separate machines and got matching results (train R²
0.8398 vs 0.8400, val R² 0.8289 vs 0.8299) — strong evidence the pipeline is
genuinely correct and reproducible, not a fluke.

---

## 5. Data files: what to get, where from, and what NOT to commit

### Never commit these to git

Add this to `.gitignore` if it's not already there:
```
data/
checkpoints/
__pycache__/
*.pyc
.pytest_cache/
```

Raw and processed data files are large, aren't code, and GDSC's own licence terms
mean we shouldn't be redistributing their data via our own GitHub repo anyway.
Every team member downloads their own local copy.

### What each teammate needs, and from where

**For training the ANN (what I've been running) — `data/deepccds_precomputed/`:**

Download the `data/` folder from https://github.com/hanjunwei-lab/DeepCCDS
(small CSVs, a few MB total — click "Code" → "Download ZIP", then pull these six
files out):
```
GDSC_ssgsea_input.csv
GDSC_mutation_input.csv
GDSC_SMILE_input.csv
GDSC_train_IC50_by_borh_cv00.csv
GDSC_valid_IC50_by_borh_cv00.csv
GDSC_test_IC50_by_borh_cv00.csv
```
Place them in `data/deepccds_precomputed/` at the repo root (a sibling of `src`,
**not** inside `src/data` — that's the Python package folder, keep code and data
separate).

**For the raw-GDSC path (expression binning for rule extraction) —
`data/raw/`:**
```
GDSC1_fitted_dose_response_27Oct23.xlsx   (GDSC bulk download / Cell Model Passports)
GDSC2_fitted_dose_response_27Oct23.xlsx   (same)
model_list_20260724.csv                   (Cell Model Passports "Model List")
driver_genes_20241212.csv                 (Cell Model Passports "Cancer Driver Genes")
```
Still needed: the RNA-seq expression file ("RNA-Seq Summary" on Cell Model
Passports) — this is the one piece blocking this path from being complete.

Note: GDSC's old download site (cancerrxgene.org) has migrated — everything now
lives at https://cellmodelpassports.sanger.ac.uk/downloads.

---

## 6. Why I used the DeepCCDS authors' precomputed features (please flag to supervisor)

Building the real pathway-activity signal from scratch is a substantial
bioinformatics pipeline in its own right: a curated protein-protein interaction
network, Random-Walk-with-Restart diffusion seeded from driver genes, then GSEA
against 323 KEGG pathways to select the 38 that matter. Similarly, the mutation
matrix and drug fingerprints each need their own separate raw-data processing.

Rather than re-deriving all of this independently under a tight timeline, I found
that the DeepCCDS authors published their own exact, paper-verified versions of
all three directly in their GitHub repository. I checked every file's shape and
row count against the numbers explicitly stated in their published paper, and
they matched exactly (878 cell lines, 38 pathways, 534 driver genes, 1024-bit
fingerprints, 319,543 total training pairs).

**This is a genuine scoping decision that affects what our project can claim
credit for**, and I think it's worth explicitly raising with our supervisor: are
we comfortable building our Red XAI diagnostic pipeline on top of the authors'
precomputed features, or does the project need us to independently derive these
ourselves for full methodological independence? Either way is defensible, but I'd
rather have that confirmed than assumed.

---

## 7. How to run this

```powershell
py -m pip install -r requirements.txt

# Place the 6 precomputed CSVs in data/deepccds_precomputed/ (see Section 5)

py -m pytest tests/ -v              # confirm 17/17 pass first

py -m src.model.train_baseline      # runs the real Stage 1 training loop
```

Training doesn't stop at a fixed epoch count — it runs until the generalisation
gap has clearly plateaued (validation loss non-improving for 10 consecutive
epochs), per our proposal's Section 4.2.3 stopping rule. This could take anywhere
from 15 minutes to a few hours depending on your machine; it prints progress every
epoch and stops itself once done.

Once it finishes, you'll have:
```
checkpoints/baseline_adam_real.pt              <- the trained model weights
checkpoints/baseline_adam_real_summary.json    <- final metrics
checkpoints/baseline_adam_real_history.json    <- per-epoch training curve
```

---

## 8. Handoff notes for teammates

**Melisa / Sakhile (rule extraction):** `DeepCCDSDataset` gives real pathway,
mutation, and drug tensors plus real LN_IC50 targets right now — you can start
building your GA/NSGA-II extraction code against these interfaces immediately,
ahead of the final trained checkpoint. For the discretised Low/Intermediate/High
expression bins specifically, those still come from the separate raw-GDSC path
(`preprocessing.py`), which is blocked on the RNA-seq file — I'll update you once
that's resolved.

**Bahiya (SAM branch):** `train_baseline_real()` in `train_baseline.py` is your
direct template — identical architecture, identical data, identical loss. The
only thing that changes for your branch is swapping the plain Adam optimiser step
for SAM's sharpness-aware perturbation step.

**Hawa (project coordination):** current status is: architecture and data
pipeline built and verified (17/17 tests passing, cross-machine reproducibility
confirmed), full training run in progress, Clever Hans audit needs a small update,
and the precomputed-vs-derived-features scoping decision needs supervisor input.
