# Stage 5 readiness audit — rule extraction → evaluation

Branch audited: `feat/evaluation-pipeline` (checked out, up to date with `origin/feat/evaluation-pipeline`). Read-only: no files were modified. `jupyter`/`nbconvert`/`ipykernel` were installed into the local `.venv` to satisfy question 6 (not added to `requirements.txt` — see §6).

**Note on governing docs:** `COS700_Project_Orientation_and_Decisions.md` does not exist anywhere in this repository's git history, on any branch (`git rev-list --all --objects | grep -i orientation` returns nothing). `docs/CONTRACTS.md` and `docs/sam_investigation_journal.md` / `docs/investigations/sam-generalisation/journal.md` do exist and were used instead, but nothing below should be read as confirming or denying a decision that only the (missing) orientation doc would have recorded — those points are marked NEEDS DECISION with "no record found," not "confirmed open."

---

## 1. Rule pool availability — **BLOCKED** (GA ready, NSGA-II not)

**GA (Melisa) — real.** `artifacts/rules/ga_baseline.json` / `ga_sam.json` exist, have a single clean commit `19699a3` ("rules from the baseline and sam", 2026-09-23), and their `conditions` reference real feature names (`Axon_guidance_activity`, `C-type_lectin_receptor_signaling_pathway_activity`, …) matching `artifacts/data/feature_map.json`'s 38 real ssGSEA pathway columns exactly. This is genuine GDSC output, not synthetic.

**NSGA-II (Sakhile) — not real.** `artifacts/rules/nsga_baseline.json` / `nsga_sam.json` trace back to commit `828503a`, titled **"feat: NSGA-II extraction with full synthetic simulation harness."** Their rule `conditions` reference feature names like `TP53_expr`, `NRAS_expr`, `EGFR_expr`, `MYC_expr`, `KRAS_expr`, `BRAF_expr`, `STK11_expr` — none of which exist in the real `feature_map.json` (confirmed: `'TP53_expr' in feature_map → False`, checked against all 38 real keys). These match the illustrative example names used in `docs/CONTRACTS.md`'s §0.4 schema docstring, not any real data source. A second variant, `artifacts/rules_v2/nsga_{baseline,sam}.json` (5–6 rules each, evidently a rerun with a coverage floor applied — see §4), has the identical problem. **Neither location Sakhile's arm writes to currently contains real-data output.**

**Blocked on the final SAM/baseline checkpoint?** No — this specific blocker is resolved. `data/processed/real_extraction/meta.json`'s `conditions_run.sam` field records a completed run against `artifacts/sam_investigation/confirmatory/model.pt` (SAM ρ=0.5, WD=1e-3 — the intended final checkpoint), producing 307,293 real rows across 379/413 shortlisted drugs, at `git_commit: "c7f316c"`. The real-extraction data-prep step itself is done and usable; the checkpoint is not what's holding NSGA-II back.

---

## 2. Schema compatibility — **READY (GA) / BLOCKED (NSGA-II)**

`src/common/schema.py:43-49`: `VALID_EXTRACTORS = {"ga", "nsga", "cart"}`, with a comment confirming `"cart"` was added 2026-08-08 by Hawa for notebook 06 and announced as a breaking change per CONTRACTS.md §6. The CART-rejection fix is in place.

Ran `validate_rule_pool()` live (with the real `feature_map.json`) against every currently-committed real-location rule pool:

| File | Result |
|---|---|
| `artifacts/rules/ga_baseline.json` | **VALID** |
| `artifacts/rules/ga_sam.json` | **VALID** |
| `artifacts/rules/nsga_baseline.json` | **415 errors** — unresolvable features/genes on ~103 of 105 rules, plus 1 rule with empty `conditions`/`genes` |
| `artifacts/rules/nsga_sam.json` | **19 errors** — same two problem classes |
| `artifacts/rules_v2/nsga_baseline.json` | **22 errors** |
| `artifacts/rules_v2/nsga_sam.json` | **17 errors** |

The empty-`conditions`/`genes` rule in each NSGA-II pool is not new: commit `95249c9` (2026-09-11, "fix pre-handoff errors found while auditing rule-extraction readiness for Melisa/Sakhile") already found and explicitly left this one ("Not fixed, flagged for Sakhile … Whether to drop it or special-case it in the schema is a call for her, not made here"). It is still there, unresolved, three weeks later. The feature-name mismatch is a separate, previously-unflagged problem (that audit only fixed a stale `config_hash`, not feature resolvability).

Since `save_rule_pool()` validates before writing, these files necessarily passed validation *at the time they were written* — against whatever `feature_map.json` the synthetic simulation harness was using then, not the current real one. They have not been regenerated since.

---

## 3. `predict_with_rule()` and `metrics.stability()` — **READY**

- `predict_with_rule()` is defined exactly once, in `src/extraction/common.py:68-89`. Its docstring: *"RESOLVED (was KNOWN GAP): predict_with_rule() below is that extraction. nsga.py's `_score_genotype()` now calls it instead of computing the majority vote inline."*
- `nsga.py` imports it at line 59 and calls it at line 134 (`_score_genotype`).
- `ga.py` imports it at line 60 and calls it at line 208 (`_score_genotype`). No duplicate/local reimplementation exists in either file — confirmed by reading both scoring functions in full.
- `src/common/metrics.py:113-148` implements `stability()` as variance of per-subgroup `fidelity_f1`, with the module docstring stating: *"stability() … is owned conceptually by the Evaluation arm (Hawa) — the implementation below is the confirmed, non-stub version."* This matches the described canonical status.

**Caveat carried to §6/§7:** `stability()` itself is fine, but it requires a subgroup label (e.g. `cancer_type`) to group rows by. The real `artifacts/data/gdsc_processed.parquet` has no such column (checked directly — see §7 item 6), so nothing currently supplies its required input for real data.

---

## 4. Coverage threshold (C6) and sparse initialisation (C10) — **NEEDS DECISION / not applied identically**

**Coverage threshold.** Neither extractor defaults to 2%:
- `nsga.py --min-coverage` default = **0.0** (off). Comment at line 141: *"Off by default (0.0) to match the branch contract exactly; opt in with --min-coverage."*
- `ga.py --min-coverage` default = **0.01** (1%).

Empirical confirmation via the actual coverage values in the committed pools:
- `artifacts/rules/nsga_baseline.json`: 103/105 rules have coverage below 2% (min 0.13%) — consistent with the 0.0 default having been used.
- `artifacts/rules_v2/nsga_baseline.json`: all 5 rules have coverage ≥ 2.13% — consistent with a separate, non-default rerun using `--min-coverage 0.02`, but this rerun was never promoted to the canonical `artifacts/rules/` location, and GA was never given an equivalent 2%-floor rerun.
- `artifacts/rules/ga_baseline.json`: all 12 rules have coverage ≥ 1.00% exactly at the floor in several cases — consistent with GA's own 1% default, not 2%.

No committed pool from either extractor was produced with a matching, explicit 2% floor applied to both.

**Sparse initialisation.** `random_gene()` in `src/extraction/common.py:50-51` is `random.randint(-1, N_PARTITIONS - 1)` — a uniform draw over 4 outcomes per gene (dense: with ~38-98 features, the expected individual has the large majority of genes active, not 1-5). Both `ga.py` and `nsga.py` import this same function (no divergence between the two — they're equally *not* sparse). A correct sparse sampler already exists, but only in `notebooks/02_synthetic_rule_pools.ipynb` cell 11 — `sample_sparse_chromosome(n_features, rng, max_active=5)`, whose own docstring explains *"why uniform random_gene() per position doesn't work at this feature count"* — and is used solely to fabricate the synthetic test fixtures. It was never ported into `src/extraction/common.py` for the real extractors.

---

## 5. Open decisions still blocking — all four remain **NEEDS DECISION**

| Decision | Status found | Evidence |
|---|---|---|
| PROMETHEE weights (fidelity 0.5 / stability 0.3 / comprehensibility 0.2), `vshape` | **Still provisional** | `notebooks/03_promethee_ranking.ipynb` cell 6 (markdown): *"Weights — also provisional … hasn't been discussed with the team yet."* Cell 13 writes `promethee_config.json` with `"status": "provisional -- not yet confirmed with team, see §3 markdown"` verbatim. |
| CART scope ("time-permitting"?) | **No record found** | No occurrence of "time permit"/"stretch goal" tied to CART anywhere in `docs/` or code. `06_cart_baseline.ipynb` exists and (against synthetic fixtures) produces `cart_baseline.json`/`cart_sam.json`, but no real `artifacts/rules/cart_*.json` exists yet. Cannot confirm or deny the decision from repo evidence alone. |
| Per-drug sample threshold / drug shortlist | **Explicitly still provisional** | `configs/data_prep.yaml` header: *"PROVISIONAL -- NOT YET CONFIRMED BY HAWA … do not treat these two values … as final."* `data/processed/real_extraction/meta.json["PROVISIONAL"]` is `true` with the identical note, even though the data was already generated and is in use. |
| Target discretisation threshold (median split) | **Textually still provisional, functionally already in use** | `configs/pipeline.yaml`: `threshold: median  # provisional; fixed during pilot` — the comment reads as stale (says "provisional" but also "fixed"), and a concrete resolved value (`2.901433`) is already baked into both shipped checkpoints' `meta.json`. The *separate* `configs/data_prep.yaml` discretisation mode (`per_drug_median_train`, for the real-extraction handoff only) is under the same explicit PROVISIONAL header as the drug shortlist above. |

---

## 6. Notebook execution status — **BLOCKED**, and not only for real-data reasons

Environment note: `jupyter`, `nbconvert`, `ipykernel` were **not installed** in `.venv` and are **not listed** in `requirements.txt` — installed for this audit only (`pip install nbconvert ipykernel jupyter_client`), not persisted anywhere in the repo.

Ran `jupyter nbconvert --to notebook --execute` on each, output redirected to a scratch directory (tracked `.ipynb` files untouched):

| Notebook | Result | Detail |
|---|---|---|
| `03_promethee_ranking.ipynb` | **FAIL** | `FileNotFoundError: [Errno 2] No such file or directory: '../tests/fixtures/synthetic/feature_map.json'` — fails on the first data-loading cell, before any real-vs-synthetic question is even reached. |
| `04_statistical_testing.ipynb` | **PASS** | Only because it reads a pre-existing, already-committed `tests/fixtures/synthetic/reports/promethee_ranked_rules.csv` rather than depending on 03's (currently broken) output at run time. |
| `05_gsea_enrichment.ipynb` | **FAIL** | Identical `FileNotFoundError` on the same missing file. |
| `06_cart_baseline.ipynb` | **FAIL** | Identical `FileNotFoundError` on the same missing file. |

**Root cause, traced:** `tests/fixtures/synthetic/feature_map.json` does not exist anywhere in git history (checked with `git log --all -- tests/fixtures/synthetic/feature_map.json`). `notebooks/02_synthetic_rule_pools.ipynb` cell 7 *builds* an equivalent `feature_map` dict in memory (and uses it in cell 13 to validate the fabricated pools before writing them), but never writes it to `tests/fixtures/synthetic/feature_map.json` on disk. Confirmed by listing the actual directory contents — no such file, at any path, in that fixture tree.

**Real-data schema mismatch, confirmed by direct code inspection (not just the failures above):** none of notebooks 03, 04, 05, or 06 reference `artifacts/rules/`, `artifacts/data/`, or `data/processed/real_extraction/` anywhere. All four hardcode `FIXTURE_DIR = Path("../tests/fixtures/synthetic")` (03 cell 1, 04 cell 1, 05 cell 1, 06 cell 1) and a fixed `POOL_NAMES = ["ga_baseline", "ga_sam", "nsga_baseline", "nsga_sam"]`. Even once the missing `feature_map.json` is fixed, these notebooks will keep exercising only the synthetic fixture path — pointing them at real data is unstarted work, not a currently-blocked feature.

Notebook 03's own markdown (cell 14, "Known gaps, to address when real data arrives") already anticipates this: *"Once Melisa's real GA branch lands, ga_baseline/ga_sam here (currently fabricated stand-ins) get replaced with real output — no notebook changes needed, since this reads through the same `schema.load_rule_pool()` interface either way."* That's true for the *interface* (both real and synthetic pools go through `load_rule_pool()`), but false for the *path* — the `FIXTURE_DIR` constant itself still needs to change to point at `artifacts/rules/` (or `data/processed/real_extraction/`) and `artifacts/data/feature_map.json`; nothing does that automatically.

---

## 7. Gaps for stage 5 — concrete punch list

1. **`src/evaluation/pipeline.py` is an unimplemented stub.** `run_evaluation()` is a TODO list ending in `raise NotImplementedError`. CONTRACTS.md §5's acceptance criterion ("a single command reproduces all four scorecards … with no manual steps") has no code path outside the notebooks at all.
2. **Notebooks 03/05/06 are currently broken even against synthetic fixtures** — missing `tests/fixtures/synthetic/feature_map.json`. This is the first fix needed, independent of real-data readiness.
3. **None of notebooks 03-06 read real paths.** All four hardcode `tests/fixtures/synthetic/`. Wiring in `artifacts/rules/*.json`, `artifacts/data/feature_map.json`, and real predictions is unstarted.
4. **NSGA-II has no real, schema-valid rule pool in the location Evaluation reads from.** `artifacts/rules/nsga_{baseline,sam}.json` and `rules_v2/nsga_{baseline,sam}.json` all fail `validate_rule_pool()` against the real feature map (placeholder gene-symbol features from an old "synthetic simulation harness" commit). Sakhile's branch needs an actual run against real data (`artifacts/data/` or `data/processed/real_extraction/data/`) before Evaluation can consume it.
5. **Both NSGA-II pools also carry one never-fixed trivial rule** (empty `conditions`/`genes`) flagged since 2026-09-11 and still unresolved.
6. **`stability()` has no real-data input to work with.** It needs a subgroup label (e.g. `cancer_type`); `artifacts/data/gdsc_processed.parquet`'s actual columns (checked directly) contain no `cancer_type` or `tissue` field of any kind. This needs a data-pipeline addition (a GDSC metadata join), not just a notebook change, before real stability numbers are possible at all.
7. **Rule pools carry no provenance of which `--data-dir` or which SAM predictions file produced them**, so this can't be verified after the fact from the artifact alone. Circumstantial evidence is concerning: `artifacts/predictions/sam_{train,val,test}.parquet` (the default location both extractors' CLIs read from) was last regenerated in commit `b158c51` ("run 3: different rho", 2026-08-27) — the *same* commit that last touched the now-superseded ρ=0.2 `artifacts/checkpoints/sam/model.pt` — and has never been regenerated from the confirmatory ρ=0.5+WD checkpoint. GA's rule pool was produced 2026-09-23, well after the confirmatory checkpoint existed, but since GA's CLI defaults to reading `artifacts/data`/`artifacts/predictions` (not `data/processed/real_extraction/`), its `sam`-condition rules may well be built on the stale ρ=0.2 checkpoint's predictions rather than the intended final SAM result, unless it was explicitly invoked with `--data-dir data/processed/real_extraction/data`. This can't be confirmed or ruled out from the committed artifact.
8. **Four open team decisions remain textually unconfirmed** (PROMETHEE weights/preference function, CART scope, drug-shortlist/discretisation-mode, and the main response threshold's stale-looking "provisional" comment) — see §5. The referenced orientation-and-decisions document doesn't exist in the repo at all, so there is no single authoritative place recording which of these are actually locked.
9. **Coverage threshold and sparse initialisation aren't implemented as specified in either real extractor** (§4) — a correct sparse-init implementation exists only in the synthetic-fixture notebook and was never ported to `src/extraction/common.py`.
10. **`requirements.txt` doesn't list `jupyter`/`nbconvert`** — the documented setup (`pip install -r requirements.txt`) doesn't give a teammate the ability to execute these notebooks at all without an undocumented extra install.
