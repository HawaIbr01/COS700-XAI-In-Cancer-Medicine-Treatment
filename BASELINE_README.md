# feat/baseline-model-2 — Stage 1 Baseline (Angelique Breedt, u23542838)

**Replaces:** `feat/baseline-model` (kept around read-only; this branch is a
clean rebase off `dev`, built to match the contracts in `docs/CONTRACTS.md`
exactly, so the rest of the team can pull it without surprises).

**Covers:** proposal §4.1–§4.2 (research approach, dataset & preprocessing
interface, Stage 1 baseline, Clever Hans confirmation) and `docs/CONTRACTS.md`
§1 (Arm 1 — this arm also owns the shared data contract everyone else depends on).

**Target:** fully working and trained by 15 Aug (hard deadline in the 5 Aug
brief: 22 Aug, alongside Bahiya's SAM branch).

---

## 1. What changed vs the old `feat/baseline-model` branch

The old branch was built before `dev` had the shared-contract skeleton
(`src/common/`, `configs/pipeline.yaml`, the `artifacts/` layout) that
`chore/project-setup` and the 5 Aug merge landed. Rather than retrofitting
contracts onto code written against a different shape, this branch starts
clean from `dev` and fills in every `TODO` in the stub files it already
contains (`src/data/preprocess.py`, `src/model/deepccds.py`,
`src/model/baseline.py`).

The actual modelling work (architecture, data loaders, training loop, Clever
Hans audit) carries over from the old branch's design almost unchanged —
that part was already solid and tested — it's now just wired into the
contract's artifact/schema system so Bahiya, Melisa, Sakhile, and Hawa can
consume it exactly the way `docs/CONTRACTS.md` §1 describes.

---

## 2. What's implemented

- **`src/data/precomputed_inputs.py`** — loaders for the DeepCCDS authors'
  own precomputed pathway-activity / mutation / drug-fingerprint / split
  files (unchanged from the old branch, still real, still paper-verified:
  878 cell lines × 38 pathways, 878 × 534 driver genes, 413 drugs × 1024-bit
  fingerprints, 319,543 total pairs).
- **`src/data/preprocess.py`** — **new.** Builds the four contract artifacts
  (`docs/CONTRACTS.md` §0.1): `gdsc_processed.parquet`, `splits.json`,
  `feature_map.json`, `response_classes.json`, plus three cached `.npy`
  matrices baseline.py/sam.py load for training. Z-scores the 38 pathway
  activities across the full cell-line population and discretises them into
  High/Low/Intermediate per `pipeline.yaml`'s thresholds, for rule extraction.
  Bins `ln_ic50` into sensitive/resistant using the train-split median.
  **See the big warning in this file's docstring and §3 below before this
  feeds GSEA.**
- **`src/data/dataset.py`** — memory-efficient `DeepCCDSDataset`, indexing
  into three small cached matrices rather than one huge expanded tensor
  (this genuinely matters — an early attempt at a fully expanded table OOM'd).
- **`src/model/deepccds.py`** — the real architecture, cross-checked layer by
  layer against the paper's Experimental Section: pathway activity fed
  directly (ssGSEA is preprocessing, not a learned layer), mutation/drug
  autoencoders (300→100→30, BatchNorm+ReLU, Sigmoid decode), feedforward head
  (two hidden layers sized to the 98-dim concatenated input), combined Eq. 2
  loss (prediction MSE + both reconstruction MSEs).
- **`src/model/baseline.py`** — Stage 1 training: Adam (lr=1e-3, batch=1024,
  no regularisation), the §4.2.3 "early stopping in reverse" rule (patience
  30, confirmed against a 10-epoch train-still-falling check — see §4 below,
  this fixes a real bug from the old branch), generalisation-gap/R²/RMSE
  scorecard, `save_checkpoint(..., optimiser="adam")`, predictions parquet
  for all three splits, loss-curve figure.
- **`src/model/clever_hans_audit.py`** — input-gradient attribution over the
  38 pathway features (§4.2.4), comparing train vs val attribution
  consistency; writes a JSON summary + bar chart. Now points at the real
  trained model (the old branch's version was still on a placeholder).
- **Tests** (`tests/test_deepccds_model.py`, `tests/test_preprocess.py`) —
  architecture shapes, the combined loss, the (now fixed) stopping-tracker
  logic, and a full `preprocess()` run against tiny synthetic CSVs shaped
  like the real files. All pass locally; run `pytest tests/ -v` to confirm
  on your machine before training.

---

## 3. Known limitation — please read before anyone runs GSEA on this data

This is the same scoping issue the old branch's README flagged:

**The model's inputs are real** (38 real ssGSEA pathway-activity scores, a
real 534-gene mutation matrix, real 1024-bit drug fingerprints — all from the
DeepCCDS authors' own published, paper-verified files).

**The rule-extraction feature set is not gene-level.** `feature_map.json`'s
38 entries are each a *KEGG pathway* (Z-scored, discretised
High/Low/Intermediate), not an individual gene — because we don't have a raw
RNA-seq expression file wired up yet (that path needs Cell Model Passports'
"RNA-Seq Summary" download; see §5). Each entry's `gene_symbol` field
currently holds the **pathway name as a placeholder**, with a `note` field
saying so explicitly. GA/NSGA-II can build and evaluate rules over these
pathway-level features fine (fidelity/comprehensibility/stability all still
work), but **Hawa's GSEA step (§4.6.3) needs real gene symbols** to run a
meaningful KEGG/Enrichr over-representation test — passing pathway names in
as if they were genes will not produce valid enrichment results.

Two ways to close this, neither of which I've done because doing it without
the real source data would mean fabricating gene-pathway associations:
1. Pull the DeepCCDS paper's Supplementary Table S1 (leading-edge gene set
   per pathway) and use that as the real feature → genes mapping.
2. Finish the raw-GDSC RNA-seq path (§5) for genuine per-gene features.

**Team decision needed:** is (1) an acceptable stopgap for the current
milestone, or does this block on (2)? Flag it in the next stand-up — this
affects Hawa's arm directly and Melisa/Sakhile's `genes` field indirectly.

---

## 4. A bug I found and fixed vs the old branch

The old branch's full 500-epoch training run (`checkpoints/baseline_adam_real_summary.json`
in the previous branch) hit the `MAX_EPOCHS` cap without its stopping rule
ever firing — `has_stabilised()` required *strictly* non-improving validation
loss across a fixed window, which noisy real validation loss essentially
never satisfies. That run still produced a usable overfit checkpoint (train
R² ≈0.92 vs val R² ≈0.85), but it wasn't a genuinely *confirmed* stabilisation
per §4.2.3, just a cap being hit.

`StoppingTracker` in this branch's `baseline.py` replaces that with a
standard best-value + patience counter (patience=30, matching §4.2.3's stated
number), with a secondary check that training loss is still meaningfully
lower than 10 epochs ago (the "train loss keeps falling" half of the §4.2.3
description). This is a **breaking change to the stopping logic** — flagging
it here rather than silently changing behaviour. Covered by
`test_stopping_tracker_triggers_on_plateau` / `..._does_not_trigger_while_still_improving`.

---

## 5. How to run this

```bash
pip install -r requirements.txt
```

**Get the data** (same six files as before — download the `data/` folder
from https://github.com/hanjunwei-lab/DeepCCDS, "Code" → "Download ZIP"):

```
GDSC_ssgsea_input.csv
GDSC_mutation_input.csv
GDSC_SMILE_input.csv
GDSC_train_IC50_by_borh_cv00.csv
GDSC_valid_IC50_by_borh_cv00.csv
GDSC_test_IC50_by_borh_cv00.csv
```

Place them in `data/deepccds_precomputed/` at the repo root (sibling of
`src`, not inside it — `data/` is git-ignored, never commit these).

```bash
pytest tests/ -v                     # confirm the suite passes first

python -m src.data.preprocess        # writes artifacts/data/* (a few seconds)
python -m src.model.baseline         # Stage 1 training — see below for timing
```

Training does not stop at a fixed epoch count — it runs until the
generalisation gap has genuinely stabilised (§4.2.3 / §4 above). This could
take anywhere from ~15 minutes to a few hours depending on your machine; it
logs every epoch and stops itself.

When it finishes you'll have, all under `artifacts/`:
```
checkpoints/baseline/{model.pt, meta.json}
predictions/baseline_{train,val,test}.parquet
reports/baseline_gap_scorecard.json
reports/baseline_training_history.json
reports/baseline_loss_curve.png
reports/baseline_clever_hans_audit.json
reports/baseline_clever_hans_attribution.png
```

None of `artifacts/`, `data/`, or `*.npy`/`*.pt`/`*.parquet` get committed
(see `.gitignore`) — every teammate reproduces these locally from the same
code + the same six CSVs, which is exactly what `config_hash`/`split_hash`
in `meta.json` are there to let people confirm.

---

## 6. Handoff notes for the team

**Bahiya (`feat/sam-intervention`):** `src/model/deepccds.py`'s `DeepCCDS`
class is unchanged in interface from what `docs/CONTRACTS.md` describes —
`from src.model.deepccds import DeepCCDS, compute_loss` gives you the exact
architecture and the exact Eq. 2 loss the baseline trained with.
`src/model/baseline.py`'s `train_baseline()` is your structural template:
same data (`DeepCCDSDataset`), same architecture, same loss — the only thing
that should change in `sam.py` is wrapping `optimizer` with SAM's
neighbourhood-perturbation step. Reuse `artifacts/data/*` as-is; don't
re-run `preprocess.py` from your branch.

**Melisa / Sakhile (rule extraction):** `artifacts/data/gdsc_processed.parquet`
has real discretised High/Low/Intermediate columns you can build rules over
right now, and `artifacts/predictions/baseline_*.parquet` gives you the
model's real `class_pred` to use as the fidelity target — you don't need to
run the model yourselves. **Please read §3 above before wiring `genes` into
anything GSEA-adjacent** — the feature set is pathway-level, not gene-level,
for now.

**Hawa (evaluation / coordination):** current status —
architecture + full data pipeline done and tested against real DeepCCDS
data, training runnable end-to-end (`pytest tests/` green,
9+8 unit/integration tests plus the carried-over model tests all pass), Clever
Hans audit wired to the real model. The one open item that needs a team
decision is §3 (pathway-vs-gene feature identity for GSEA) — please put it
on the agenda.

---

## 7. Directory structure (this branch's additions/changes only)

```
src/
├── common/                    <- unchanged, shared contract code (config/schema/metrics/checkpoint)
├── data/
│   ├── precomputed_inputs.py  <- loaders for the DeepCCDS authors' real CSVs
│   ├── preprocess.py          <- NEW: builds artifacts/data/* per docs/CONTRACTS.md §0.1
│   └── dataset.py             <- NEW: memory-efficient DeepCCDSDataset over the artifacts
└── model/
    ├── deepccds.py            <- real architecture (was a placeholder stub on dev)
    ├── baseline.py            <- real Stage 1 training loop (was a TODO stub on dev)
    └── clever_hans_audit.py   <- NEW: §4.2.4 sanity check, points at the real model

tests/
├── test_deepccds_model.py     <- NEW: architecture + stopping-logic unit tests
└── test_preprocess.py         <- NEW: preprocess() integration test (synthetic data)
```
