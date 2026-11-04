# Branch Contracts — COS700 RED XAI Pipeline

**Status: starting point, not set in stone.** These contracts exist so that when something changes (and it will), we can see *exactly* what changed and who it affects. If you need to break a contract, say so in the team channel and update this file in the same PR.

The guiding principle: **an arm is defined by the artifacts it produces and their format, not by its internal code.** As long as the artifacts below match the agreed schema, each person can implement their arm however they like.

---

## 0. Shared foundations (owned by `chore/project-setup`)

Set up *before* anyone starts their arm, so all five branches agree on the same contracts. Everyone branches off `dev` after this lands.

### 0.1 Artifact layout (the "drop zones")

Arms communicate through files on disk under `artifacts/`, not by calling each other's code.

```
artifacts/
├── data/
│   ├── gdsc_processed.parquet     # rows = cell line–drug pairs, cols = features + ln_ic50
│   ├── splits.json                # {"train": [row_ids], "val": [...], "test": [...]}
│   ├── feature_map.json           # feature_col -> {gene_symbol, partition_edges}
│   └── response_classes.json      # {threshold, per_row_class}
├── checkpoints/
│   ├── baseline/{model.pt, meta.json}
│   └── sam/{model.pt, meta.json}
├── predictions/
│   └── {baseline|sam}_{train|val|test}.parquet   # row_id, ln_ic50_pred, class_pred
├── rules/
│   └── {ga|nsga}_{baseline|sam}.json             # rule-pool schema (0.4)
└── reports/
    └── ...                                        # rankings, stats, enrichment, plots
```

### 0.2 Single source of truth: `configs/pipeline.yaml`

Every arm reads shared parameters from here. **No arm hardcodes a split ratio, threshold, or seed.**

```yaml
seed: 42
split: {train: 0.8, val: 0.1, test: 0.1}      # fixed, hashed, never re-drawn
discretisation:                                # feature discretisation (Z-score)
  high_z: 1.0                                  # Z > 1  -> High
  low_z: -1.0                                  # Z < -1 -> Low  (else Intermediate)
response_binning:                              # target -> rule classes
  target: ln_ic50
  threshold: median                            # provisional; fixed during pilot
  classes: [sensitive, resistant]
ga:   {generations: 50, pop_min: 35, pop_max: 45, runs: 30}
nsga: {generations: 50, pop: 40, runs: 30}
```

### 0.3 Shared code: `src/common/`

To stop three people implementing fidelity three different ways, these live in one place and everyone imports them:

- `config.py` — loads `pipeline.yaml`, exposes `config_hash()`.
- `schema.py` — read/write + validation for every artifact below.
- `metrics.py` — `fidelity_f1(rule_class_preds, model_class_preds)`, `comprehensibility(rule)`, `stability(rule, subgroups)`. **These three functions are the contract for rule quality — GA, NSGA-II, and Evaluation all call the same ones.**
- `checkpoint.py` — `save_checkpoint`, `load_checkpoint`, `meta.json` I/O.

### 0.4 Rule-pool schema (`artifacts/rules/*.json`)

The single most important interface in the project. Both extractors emit exactly this; Evaluation consumes exactly this.

```json
{
  "meta": {"extractor": "ga", "condition": "baseline",
           "runs": 30, "config_hash": "…"},
  "rules": [
    {
      "id": "ga_baseline_0001",
      "conditions": [{"feature": "EGFR_expr", "partition": "High"},
                     {"feature": "TP53_expr", "partition": "Low"}],
      "genes": ["EGFR", "TP53"],
      "predicted_class": "resistant",
      "length": 2,
      "fidelity_f1": 0.82,
      "coverage": 0.15,
      "run_ids": [1, 7, 22],
      "non_dominated": true
    }
  ]
}
```

`genes` is mandatory and must be resolvable via `feature_map.json` — Evaluation's GSEA step depends on it.

### 0.5 Dependency graph

```
chore/project-setup
      │
      ▼
feat/baseline-model (data + Adam baseline)  ──► data/, checkpoints/baseline/, predictions/baseline/
      │
      ▼
feat/sam-intervention (reuses arch+data)    ──► checkpoints/sam/, predictions/sam/
      │
      ├──────────────► feat/ga-rule-extraction    ──► rules/ga_{baseline,sam}.json
      ├──────────────► feat/nsga2-rule-extraction ──► rules/nsga_{baseline,sam}.json
      │                        │
      ▼                        ▼
                 feat/evaluation-pipeline ──► reports/
```

Each extractor runs **twice** (once per checkpoint). Evaluation consumes all four rule pools.

---

## 1. `feat/baseline-model` — Angelique (u23542838)

**Methodology:** §4.1–§4.2 (Research approach, Dataset & Preprocessing Interface, Stage 1 baseline, Clever Hans confirmation). This arm also owns the **shared data contract** everyone else depends on.

**Receives:** raw GDSC sub-datasets; the `chore/project-setup` skeleton and `pipeline.yaml`.

**Implements (`src/data/`, `src/model/baseline.py`, `src/model/deepccds.py`):**
- Preprocessing per §4.2.2: merge the three GDSC sub-datasets, dedup, handle missing values, one-hot categoricals, Z-score normalise expression, discretise features into High/Low/Intermediate.
- The fixed 80/10/10 split (seeded, hashed) and the response-class binning of `ln_ic50` into sensitive/resistant.
- The **DeepCCDS architecture module** (`deepccds.py`) — defined once here, imported unchanged by SAM.
- Train Stage 1 baseline: Adam, lr 0.001, no regularisation, early-stopping-in-reverse (patience 30; gap "stable" when val loss plateaus/rises ≥10 epochs).
- Generalisation-gap scorecard (R², RMSE on `ln_ic50`), loss-landscape baseline plot, and the Clever Hans attribution sanity check (lightweight LRP / input gradients).

**Produces (the handoff):**
- `artifacts/data/*` — processed dataset, `splits.json`, `feature_map.json` (incl. **feature→gene mapping**), `response_classes.json`.
- `artifacts/checkpoints/baseline/{model.pt, meta.json}` — `meta.json` carries optimiser, arch config, gap metrics, epoch, `config_hash`, `split_hash`.
- `artifacts/predictions/baseline_{train,val,test}.parquet` — `ln_ic50_pred` and `class_pred` per row.
- Fragility Baseline scorecard + loss-landscape figure in `reports/`.

**Hand-off notes for the next people:**
- **SAM (Bahiya):** import `deepccds.py` and the data artifacts as-is; change *only* the optimiser/training loop. Do not touch preprocessing or the split.
- **GA/NSGA (Melisa/Sakhile):** treat the split, `feature_map`, and `response_classes` as read-only. Use `predictions/baseline_*` as the fidelity target; you do **not** need to run the model.
- **Evaluation (Hawa):** the class definitions and `feature_map` here are what GSEA and stability slicing rely on.

**Acceptance:** artifacts validate against `schema.py`; baseline shows high train performance but degraded test vs the published DeepCCDS reference (overfitting confirmed); checkpoint reloads and reproduces the recorded predictions.

**Likely to change:** exact response threshold (pilot may move it off median); patience values; whether preprocessing is cached as parquet vs npz.

---

## 2. `feat/sam-intervention` — Bahiya (u22598546)

**Methodology:** §4.5 (SAM regularisation, Stage 2).

**Receives:** `deepccds.py`, all `artifacts/data/*`, and the baseline training harness from Arm 1.

**Implements (`src/model/sam.py`):**
- Wrap the identical DeepCCDS architecture with the SAM optimiser (neighbourhood ε-ball perturbation step); everything else — architecture, data, splits, normalisation — held constant.
- Retrain to Stage 2; record the reduced generalisation gap and a loss-landscape plot directly comparable to the baseline's.

**Produces (the handoff):**
- `artifacts/checkpoints/sam/{model.pt, meta.json}` (same schema as baseline; `optimiser: sam`, plus `rho`/ε).
- `artifacts/predictions/sam_{train,val,test}.parquet`.
- Stage 2 gap scorecard + comparative loss-landscape figure in `reports/`.

**Hand-off notes:** GA/NSGA and Evaluation consume `checkpoints/sam` and `predictions/sam` through the **identical interface** as baseline — the only thing that differs downstream is the `condition` tag (`baseline` vs `sam`). Nothing else in their code should need to branch.

**Acceptance:** SAM checkpoint validates; generalisation gap is measurably smaller than baseline under the same metrics; loss landscape is visibly flatter; predictions reload reproducibly.

**Likely to change:** SAM `rho`, learning rate, epoch budget (SAM ~2× slower per epoch — accounted for in the timeline).

---

## 3. `feat/ga-rule-extraction` — Melisa (u19296950)

**Methodology:** §4.3 (GA via DEAP).

**Receives:** `artifacts/data/*`, both checkpoints/predictions (baseline + SAM), and `src/common/metrics.py`.

**Implements (`src/extraction/ga.py`, shared encoding in `src/extraction/common.py`):**
- DEAP GA over discretised features. Chromosome = vector of feature-partition indices with `-1` = "feature excluded".
- Single-objective (scalarised) fitness combining fidelity (`fidelity_f1` vs the model's `class_pred`), accuracy on held-out data, and comprehensibility (rule length) — using the shared metric functions.
- Tournament selection, two-point crossover, mutation, rule pruning, aging-evolution diversity; 50 generations, pop 35–45; **30 independent seeded runs**; aggregate non-dominated rules into one pool per condition.

**Produces (the handoff):** `artifacts/rules/ga_baseline.json` and `artifacts/rules/ga_sam.json`, conforming to the 0.4 schema (every rule carries `conditions`, `genes`, `predicted_class`, `length`, `fidelity_f1`, `coverage`).

**Hand-off notes:** Evaluation reads these two files and nothing else from this arm. Do not pre-rank or pre-filter beyond non-domination — ranking is Evaluation's job (a single unified PROMETHEE run needs the raw pool). Coordinate with Sakhile on `common.py` so GA and NSGA share one encoding/rule-decode.

**Acceptance:** both pools validate; `genes` resolve via `feature_map`; rules reproduce their reported `fidelity_f1` when recomputed by `metrics.py`; runs are seed-reproducible.

**Likely to change:** fitness weighting, pruning thresholds, pop size, run count (pending pilot + alignment with Yedjour et al.).

---

## 4. `feat/nsga2-rule-extraction` — Sakhile (u26838002)

**Methodology:** §4.4 (NSGA-II via DEAP).

**Receives:** identical inputs to Arm 3, plus the shared `common.py` encoding.

**Implements (`src/extraction/nsga.py`):**
- DEAP NSGA-II on the same discretised data and checkpoints. **True multi-objective:** maximise fidelity (macro-F1 vs model `class_pred`), minimise rule length.
- Fast non-dominated sorting + crowding distance; binary tournament (crowded-comparison); two-point crossover + mutation matched to GA for comparability; 50 generations, same pop; **30 runs**; output the aggregated Pareto front per condition.

**Produces (the handoff):** `artifacts/rules/nsga_baseline.json`, `artifacts/rules/nsga_sam.json` — **same schema as GA** (this is what makes GA-vs-NSGA a fair comparison downstream).

**Hand-off notes:** because the schema is shared with GA, Evaluation treats all four pools uniformly. The only distinguishing fields are `meta.extractor` and `meta.condition`. Keep crossover/mutation operators identical to GA except where the multi-objective design requires otherwise, and note any deviation in `meta`.

**Acceptance:** as Arm 3, plus the output is a genuine non-dominated front (no rule dominated on both objectives).

**Likely to change:** same as GA; plus whether accuracy is folded in as a third objective or left to Evaluation.

---

## 5. `feat/evaluation-pipeline` — Hawa (u20418494)

**Methodology:** §4.6 (PROMETHEE II, statistical testing, GSEA, CART baseline).

**Receives:** all four `artifacts/rules/*.json`, `predictions/*`, `feature_map.json`, and `src/common/metrics.py`.

**Implements (`src/evaluation/`):**
- **PROMETHEE II** (`pymcdm`) over criteria fidelity / comprehensibility / stability. Any comparison (baseline vs SAM, or evolutionary vs CART) goes through a **single unified run** so net flows (φ) are commensurable; take top-10 per condition.
- **Statistical test:** Mann–Whitney U (`scipy`) on the φ distributions of the two conditions (two independent samples), α = 0.05.
- **GSEA** (`GSEApy`) over-representation of each rule pool's `genes` against KEGG/Enrichr; report enrichment p-values.
- **CART baseline** (`scikit-learn`, depth-bounded) fitted to each model's predictions, rules extracted and pushed through the *same* PROMETHEE run for a SOTA comparison.
- Compute `stability` per rule (variance across GDSC cancer subtypes) using the shared metric.

**Produces (the handoff):** `artifacts/reports/` — ranked rule tables, φ values, Mann–Whitney results, pathway-enrichment tables, and the comparison figures that feed the final report. This arm is the terminal consumer; its outputs go to the write-up.

**Hand-off notes:** you own the definition of `stability` and the PROMETHEE weight vector — publish them back into `pipeline.yaml`/`metrics.py` so extractors can optionally report the same numbers. Flag early if any rule pool lacks resolvable `genes`.

**Acceptance:** a single command reproduces all four scorecards, the Mann–Whitney verdict, and the enrichment tables from the artifacts on disk, with no manual steps.

**Likely to change:** PROMETHEE preference functions/weights; stability definition; whether CCLE (stretch) reuses this pipeline unchanged (it should).

---

## 6. Working agreement

- Branch off `dev`; PR back into `dev`; `main` only at milestones.
- If you change any artifact schema, `pipeline.yaml`, or a `metrics.py` signature, it's a **breaking change**: announce it and update this file + `schema.py` in the same PR.
- Prefer consuming **artifacts** (files) over importing another arm's code. The only shared *code* is `src/common/`.
- Every artifact carries a `config_hash`; if hashes don't match across arms, you're comparing incompatible runs — stop and re-sync.