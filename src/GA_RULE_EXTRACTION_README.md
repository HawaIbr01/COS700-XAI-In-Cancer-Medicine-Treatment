# GA Rule Extraction — `ga_rule_extraction.py`

**Covers:** Proposal Section 4.3 (Genetic Algorithms for Rule Extraction)
**Wired against:** `feat/baseline-model` branch (`BASELINE_README.md`, 2 Aug 2026)
**Status:** GA logic complete and unit-tested. Blocked on real gene-expression
bins (RNA-seq file) — ships with a `--synthetic` mode so development and
testing can continue in the meantime.

---

## 1. What this script does

Given a trained DeepCCDS checkpoint (Stage 1 Adam baseline, or Stage 2 SAM),
this script extracts symbolic **if-then rules** that approximate the model's
own predictions, using a single-objective Genetic Algorithm built on DEAP —
exactly as specified in Proposal Section 4.3:

1. **Encodes** each candidate rule as a chromosome: one gene per discretised
   gene-expression feature, valued `-1` (don't care / excluded), `0` (Low),
   `1` (Intermediate), or `2` (High).
2. **Scores** each rule on three objectives, combined into a weighted-sum
   fitness (Section 4.3.3):
   - **Fidelity** — macro-F1 of the rule's predictions vs. the black-box
     model's own predictions (not ground truth).
   - **Accuracy** — macro-F1 vs. the real discretised LN IC50 outcome.
   - **Comprehensibility** — penalises rules with many active conditions
     (Rule Explosion, Section 3.6.1).
3. **Evolves** the population for 50 generations (tournament selection,
   two-point crossover, structural mutation) with population size 35–45,
   matching the Yedjour et al. baseline configuration cited in the proposal.
4. **Prunes** low-support rule conditions after each generation
   (Section 4.3.4).
5. **Repeats 30 independent runs** with different seeds and aggregates the
   non-dominated ("Pareto front") rule sets from each run into one pool
   (Section 4.3.5), deduplicated by rule signature.
6. **Exports** the pooled rules to JSON, ready for the PROMETHEE II ranking
   stage (Section 4.6.1) and, downstream, GSEA pathway enrichment
   (Section 4.6.3).

Run it once per model checkpoint — once against Stage 1 (Adam) and once
against Stage 2 (SAM) — to produce the two rule pools the proposal's
statistical comparison (Mann–Whitney U, Section 4.6.2) needs.

---

## 2. Why this isn't a plug-and-play generic script

Two things in `BASELINE_README.md` mean this script has to bridge **two
different feature spaces**, not just load "the dataset":

| | Model's own inputs | Rule antecedents |
|---|---|---|
| **What** | 38-dim pathway activity, 534-dim mutation calls, 1024-bit drug fingerprint | Discretised (Low/Int/High) **raw gene expression** |
| **Indexed by** | cell line-drug **pair** (319,543 rows) | **cell line** (878 rows) |
| **Source** | `src/data/precomputed_inputs.py` (ready now) | `src/data/preprocessing.py` (blocked — needs RNA-seq file) |
| **Why different** | DeepCCDS replaces raw expression with pathway-level summaries by design (Section 3.4.1) | Rules need gene-level antecedents so they can be cross-referenced against KEGG/Enrichr later (Section 4.6.3) |

So the model predicts on one feature space, but rules must be built and
graded for fidelity on a *different* feature space, aligned back to the
model's predictions row-for-row. `align_rule_frame_to_predictions()`
handles this: each cell line's expression bins are broadcast across every
drug pair that cell line appears in, which is exactly what Section 4.2.2
describes ("each row represents a unique cell line–drug combination").

---

## 3. Current blocker: gene-expression bins

Per `BASELINE_README.md` Sections 4 and 8, the Low/Intermediate/High
discretised expression bins come from `src/data/preprocessing.py` (the
raw-GDSC path), which is itself blocked on the RNA-seq expression file from
Cell Model Passports. That file isn't downloaded/available yet.

**Until it lands**, run this script with `--synthetic`: it generates
placeholder gene-expression bins (and, if needed, placeholder model
predictions) of the correct shape (878 cell lines) so the full GA pipeline
— encoding, fitness, evolution, pruning, multi-run aggregation, export —
can be developed, tested, and tuned right now. This matches Angelique's
handoff note that GA/NSGA-II development can proceed "ahead of the final
trained checkpoint."

**Once real bins exist**, replace the body of `load_gene_expression_bins()`
with a call into `preprocessing.py`'s real output — the function signature
and return contract are already documented in its docstring so this is a
small, contained change.

Two smaller TODOs are marked inline for whoever has `src/model/deepccds.py`
open, since the README describes but doesn't show the exact code:

- `DeepCCDS`'s constructor signature (does it take explicit dimension args,
  or are 38/534/1024 hardcoded?)
- `DeepCCDS.forward()`'s return signature (the combined loss in Eq. 2
  implies it returns the prediction alongside both autoencoders'
  reconstructions — the unpacking in `_unpack_prediction()` assumes the
  LN IC50 prediction is either the sole return value or the first element
  of a tuple; adjust if that's wrong).

---

## 4. How to run it today (synthetic mode)

```powershell
py -m pip install deap scikit-learn numpy torch

py ga_rule_extraction.py --condition baseline_adam --synthetic --generations 50 --runs 30
```

This produces `rules_baseline_adam.json` built entirely from placeholder
data — useful for validating the pipeline, tuning GA hyperparameters, and
building the downstream PROMETHEE II/GSEA scripts against a realistic
output shape, but **not real results**. Every synthetic-mode run prints a
`SYNTHETIC MODE` warning so it's never mistaken for the real thing.

---

## 5. How to run it for real (once bins are unblocked)

```powershell
py ga_rule_extraction.py ^
    --condition baseline_adam ^
    --checkpoint checkpoints/baseline_adam_real.pt ^
    --data-root data/deepccds_precomputed ^
    --split test ^
    --generations 50 ^
    --runs 30 ^
    --output rules_baseline_adam.json
```

Then, once Bahiya's SAM checkpoint is ready:

```powershell
py ga_rule_extraction.py ^
    --condition sam_regularised ^
    --checkpoint checkpoints/sam_regularised_real.pt ^
    --data-root data/deepccds_precomputed ^
    --split test ^
    --generations 50 ^
    --runs 30 ^
    --output rules_sam_regularised.json
```

Both JSON outputs share the same schema (rule vector, readable rule string,
fidelity, accuracy, comprehensibility, support, consequent class, model
condition), so they can be loaded directly into a single PROMETHEE II
matrix as required by Section 4.6.1.

**Note on the LN IC50 threshold:** Section 4.2.2 states this is "fixed at
the implementation stage." This script defaults to a median split if
`--ln-ic50-threshold` isn't passed, and prints a warning when it does so —
confirm the real clinically-motivated cutoff with the team before treating
any output as final.

---

## 6. Where to put this file in the repo

Suggested location, alongside the existing `src/` layout:

```
feat-baseline-model/
├── src/
│   ├── data/
│   ├── model/
│   └── xai/
│       └── ga_rule_extraction.py   <- this file
```

placing it under `src/xai/` lets it import `from src.model.deepccds import
DeepCCDS` and `from src.data.precomputed_inputs import ...` the same way
`train_baseline.py` does, without any path hacking.

---

## 7. Function reference

| Function | Purpose |
|---|---|
| `load_black_box_model(checkpoint_path)` | Loads a DeepCCDS checkpoint (Stage 1 or 2). |
| `load_precomputed_split_arrays(data_root, split)` | Loads pathway/mutation/drug-fingerprint arrays + split rows, gathered per cell-drug pair. |
| `load_gene_expression_bins(...)` | Loads (or synthesises) Low/Int/High expression bins, per cell line. **Currently blocked in non-synthetic mode.** |
| `align_rule_frame_to_predictions(bins, cell_idx)` | Broadcasts per-cell-line bins onto per-pair rows. |
| `get_black_box_predictions(model, ...)` | Forward pass + LN IC50 threshold → fidelity target. |
| `run_pipeline(...)` | Orchestrates the above (real or synthetic) and runs the GA. |
| `extract_rules(...)` | The 30-run GA orchestration + rule pool aggregation (Section 4.3.5). |
| `run_single_ga(...)` | One GA run: init → evolve → prune → Pareto-filter. |
| `RuleFitnessEvaluator` | Computes fidelity/accuracy/comprehensibility and the weighted-sum fitness. |
| `export_rules(rules, path)` | Writes the rule pool to JSON for PROMETHEE II / GSEA. |

---

## 8. Testing status

- ✅ Core rule logic (`rule_matches`, `rule_consequent_class`,
  `readable_rule`, `rule_length`, `rule_support`) — unit-tested against
  synthetic data, behaves correctly.
- ✅ New alignment logic (`load_gene_expression_bins` synthetic mode,
  `align_rule_frame_to_predictions` cell-line→pair broadcast, the blocked
  path's error message) — unit-tested, behaves correctly.
- ⚠️ Full DEAP evolutionary loop (`run_single_ga`, `extract_rules`,
  `run_pipeline`) — implemented per Section 4.3.3–4.3.5 and structurally
  sound, but **not yet run end-to-end** in this environment because
  `deap` couldn't be installed here (no network access in the sandbox
  this was written in). Run `pip install deap scikit-learn` and the
  `--synthetic` smoke test above before relying on it for real results.
- ⏳ Real (non-synthetic) path — untestable until the RNA-seq-derived
  gene-expression bins and a real checkpoint exist.
