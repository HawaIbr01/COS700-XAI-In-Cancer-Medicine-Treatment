import nbformat as nbf
from pathlib import Path
NB = Path(__file__).parent

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return n
md = lambda s: ("md", s.strip("\n"))
code = lambda s: ("code", s.strip("\n"))

SETUP = '''
import os, sys, json
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir("..")            # run from the repo root so relative paths match the pipeline
sys.path.insert(0, ".")
import numpy as np, pandas as pd
'''

# ===================================================================== 02b
n02b = nb([
md('''
# 02b — Extraction evolution logs (GA and NSGA-II)

**Section:** between rule extraction and the evaluation layer
**Inputs:** `artifacts/rules/logs/*_generations.csv`, `*_runs.csv` and `artifacts/rules/runs/*_run_fronts.json`, written by `src/extraction/run_logged.py`
**Why this exists:** the original `ga.py` / `nsga.py` runs saved only the pooled rules, so there was no record of how the searches evolved. `run_logged.py` repeats the **same** extraction (same seeds, operators and parameters, calling `run_ga` / `run_nsga2` unchanged) and records every generation of every run. The pooled rules it produced were compared with the shipped pools (`*_repro_check.txt`) and are identical, so nothing downstream had to be redone.

**How to read the numbers.** 30 independent runs per (extractor, condition), 50 generations each. Curves show the median across runs with the inter-quartile band. A rule always predicts one class, so macro-F1 fidelity is `p/(1+p)` (p = share of covered rows in the rule's class): it can only exceed 0.5 if the covered region is perfectly pure, and 0.46 / 0.49 correspond to about 85% / 96% agreement with the model.
'''),
code(SETUP + '''
import matplotlib.pyplot as plt
LOG = Path("artifacts/rules/logs"); RUNS = Path("artifacts/rules/runs"); FIG = Path("artifacts/reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
C = {"baseline": "#2a78d6", "sam": "#eb6834"}          # validated categorical slots 1-2
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "font.size": 9})
g = {(e, c): pd.read_csv(LOG / f"{e}_{c}_generations.csv") for e in ("ga", "nsga") for c in ("baseline", "sam")}
r = {(e, c): pd.read_csv(LOG / f"{e}_{c}_runs.csv") for e in ("ga", "nsga") for c in ("baseline", "sam")}
for k, v in g.items(): print(k, v.shape, "runs:", v.run_id.nunique())

def band(ax, df, col, cond, label=None):
    s = df.groupby("gen")[col].quantile([.25, .5, .75]).unstack()
    ax.plot(s.index, s[.5], color=C[cond], lw=2, label=label or cond)
    ax.fill_between(s.index, s[.25], s[.75], color=C[cond], alpha=.18, lw=0)
'''),
md('''
## 1. GA: fitness over generations and the aging effect

`ga.py` replaces individuals older than `max_age = 8` generations by random immigrants. Children are clones that inherit their parent's `age`, so in a population that starts all at age 0 every individual ages in lockstep and the **whole population** is replaced at generation 9, 18, 27, 36 and 45. The curves below test that directly.
'''),
code('''
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for c in ("baseline", "sam"):
    band(ax[0], g[("ga", c)], "fit_max", c)
    band(ax[1], g[("ga", c)], "fit_avg", c)
    band(ax[2], g[("ga", c)], "n_age0", c)
ax[0].set_title("Best weighted fitness in population"); ax[1].set_title("Mean weighted fitness (-1 = below 2% floor)")
ax[2].set_title("Individuals of age 0 (new immigrants)")
for a in ax:
    a.set_xlabel("generation")
    for x in (9, 18, 27, 36, 45): a.axvline(x, color="grey", lw=.6, ls=":")
ax[0].legend(frameon=False, title="model")
plt.tight_layout(); plt.savefig(FIG / "ga_evolution.png"); plt.show()

# quantitative check of the lockstep-replacement hypothesis
d = pd.concat([g[("ga", c)].assign(condition=c) for c in ("baseline", "sam")])
d["all_new"] = d.n_age0 == d.popsize
wipe_gens = sorted(d.loc[d.all_new & (d.gen > 0), "gen"].unique())
share = d.loc[d.gen.isin(wipe_gens)].groupby("gen").all_new.mean()
print("generations at which the entire population was replaced in some run:", wipe_gens)
print("share of the 60 GA runs with a full replacement at those generations:\\n", share.round(2).to_string())
print("\\nfull replacements at any OTHER generation:", int(d.loc[d.all_new & (d.gen > 0) & ~d.gen.isin(wipe_gens)].shape[0]))
'''),
md('''
**Reading.** If the population is entirely age 0 at generations 9, 18, 27, 36 and 45 in (nearly) every run, then each GA run is effectively **five or six short searches of about nine generations**, with the 10-slot HallOfFame carrying the best rules across the restarts. The rules that reach the pool are therefore the best of several short restarts, not the end of one 50-generation search. This is a property of the implemented algorithm, not of the data; it is reported as a limitation and a future-work item (stagger or reset ages for offspring).
'''),
md('''
## 2. GA: what the search finds, and how diverse the population is
'''),
code('''
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for c in ("baseline", "sam"):
    band(ax[0], g[("ga", c)], "best_fidelity_valid", c)
    band(ax[1], g[("ga", c)], "n_unique_genotypes", c)
    band(ax[2], g[("ga", c)], "mean_length", c)
ax[0].set_title("Best fidelity among rules above the 2% floor"); ax[1].set_title("Distinct genotypes in population")
ax[2].set_title("Mean rule length (active conditions)")
for a in ax: a.set_xlabel("generation")
ax[0].legend(frameon=False, title="model")
plt.tight_layout(); plt.savefig(FIG / "ga_fidelity_diversity.png"); plt.show()
'''),
md('''
## 3. NSGA-II: convergence and diversity

NSGA-II logs every run with `front_size = 40 = pop`, i.e. the whole population is in front 0. Identical individuals do not dominate each other, so a population of copies of a handful of rules still reports a full front. `front0_unique` counts the distinct rules in that front.
'''),
code('''
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for c in ("baseline", "sam"):
    band(ax[0], g[("nsga", c)], "best_above_fidelity", c)
    band(ax[1], g[("nsga", c)], "n_unique_genotypes", c)
    band(ax[2], g[("nsga", c)], "front0_unique", c)
ax[0].set_title("Best fidelity among rules above the 2% floor"); ax[1].set_title("Distinct genotypes in population (of 40)")
ax[2].set_title("Distinct rules in front 0 (reported front size = 40)")
for a in ax: a.set_xlabel("generation")
ax[0].legend(frameon=False, title="model")
plt.tight_layout(); plt.savefig(FIG / "nsga_evolution.png"); plt.show()

rows = []
for c in ("baseline", "sam"):
    d = g[("nsga", c)]
    last = d[d.gen == d.gen.max()]
    # generation after which the best fidelity never improves again, per run
    conv = d.groupby("run_id").apply(lambda x: int(x.loc[x.best_above_fidelity.fillna(-1) >= x.best_above_fidelity.max() - 1e-12, "gen"].min()))
    rows.append({"condition": c, "median generation best fidelity is reached": conv.median(),
                 "runs reaching it by gen 15": int((conv <= 15).sum()),
                 "median distinct genotypes at gen 50 (of 40)": last.n_unique_genotypes.median(),
                 "median distinct front-0 rules at gen 50": last.front0_unique.median(),
                 "runs with <= 5 distinct front-0 rules": int((last.front0_unique <= 5).sum())})
pd.DataFrame(rows).set_index("condition").T
'''),
md('''
## 4. Final comparison across the 30 independent runs

Each run is an independent replicate (different seed), so per-run results are the properly independent sample. For every run: the best fidelity (train) among its final front.
'''),
code('''
best = []
for e in ("ga", "nsga"):
    for c in ("baseline", "sam"):
        fr = json.load(open(RUNS / f"{e}_{c}_run_fronts.json"))
        for run, rules in fr.items():
            best.append({"extractor": e, "condition": c, "run_id": int(run),
                         "best_fidelity_train": max(x["fidelity_f1"] for x in rules), "front_rules": len(rules)})
best = pd.DataFrame(best)
best.to_csv(Path("artifacts/reports") / "per_run_best_fidelity.csv", index=False)
print(best.groupby(["extractor", "condition"]).best_fidelity_train.describe().round(4).to_string())
fig, ax = plt.subplots(figsize=(6.5, 3.6))
pos = {("ga","baseline"):0, ("ga","sam"):1, ("nsga","baseline"):2.6, ("nsga","sam"):3.6}
for k, x in pos.items():
    v = best[(best.extractor == k[0]) & (best.condition == k[1])].best_fidelity_train
    ax.boxplot(v, positions=[x], widths=.6, showfliers=False, medianprops=dict(color="black"),
               boxprops=dict(color=C[k[1]]), whiskerprops=dict(color=C[k[1]]), capprops=dict(color=C[k[1]]))
    ax.scatter(np.full(len(v), x) + np.random.default_rng(0).uniform(-.18, .18, len(v)), v, s=14, color=C[k[1]], alpha=.7, zorder=3)
ax.set_xticks([.5, 3.1]); ax.set_xticklabels(["GA", "NSGA-II"]); ax.set_ylabel("best fidelity in the run (train)")
ax.set_title("Per-run best fidelity: baseline (blue) vs SAM (orange)")
plt.tight_layout(); plt.savefig(FIG / "per_run_best_fidelity.png"); plt.show()
'''),
])

# ===================================================================== 03
n03 = nb([
md('''
# 03 — PROMETHEE II ranking (real rule pools)

**Section:** Evaluation layer
**Depends on:** `artifacts/rules/{ga,nsga}_{baseline,sam}.json` (real extraction, 572 features, per-drug labels), `data/processed/real_extraction/`
**Feeds into:** `04_statistical_testing.ipynb`, `05_gsea_enrichment.ipynb`

This replaces the synthetic-fixture version. Changes from it, each deliberate:

| Item | Synthetic version | This version | Why |
|---|---|---|---|
| Data | `tests/fixtures/synthetic` | real pools + corrected extraction dataset | the real rules exist now |
| Fidelity criterion | value stored in the pool (train) | **re-scored on the held-out test split** with the rule's class fixed from train | train-for-search, test-for-reporting |
| Stability subgroup | `cancer_type` (6 subtypes) | **cell line** | no cancer-type / tissue annotation exists in the DeepCCDS precomputed files used by this project (it ships in Cell Model Passports, which was not downloaded) |
| Stability measure | variance of per-subgroup macro-F1 (`metrics.stability`) | variance of per-subgroup **agreement rate**; the contract version is reported alongside | see §3: macro-F1 of a one-class rule jumps from ~0.49 to 1.0 at perfect purity, which dominates the variance |
| Run | one PROMETHEE II over all rules | unchanged: **one** unified run over all 19 rules | φ values from separate runs are not comparable |

Criteria and weights follow methodology §4.6.1: fidelity 0.5 (maximise), stability 0.3 (minimise variance), comprehensibility 0.2 (minimise length). V-shape preference with `p` = observed range, `q = 0`, as in the synthetic notebook (still provisional).
'''),
code(SETUP + '''
from pymcdm.methods import PROMETHEE_II
from scipy.stats import spearmanr
from src.common import metrics
from src.evaluation import stage5_utils as U

pools, rule_lookup, feature_map = U.load_pools()
for n, p in pools.items():
    print(f"{n}: {len(p['rules'])} rules, runs={p['meta']['runs']}, config_hash={p['meta']['config_hash']}")
table = U.rule_table(pools)
print("\\nall rules predict:", table.predicted_class.unique().tolist())
test = U.load_split("test"); train = U.load_split("train")
print("test rows:", len(test), "| cell lines:", test.cell_name.nunique(), "| drugs:", test.drug_name.nunique())
'''),
md('''
## 1. Split structure (matters for how to read "held-out")

The corrected dataset splits **rows** (cell line × drug pairs) at random. Check which cell lines and drugs the splits share.
'''),
code('''
cells = {s: set(df.cell_name) for s, df in (("train", train), ("test", test))}
drugs = {s: set(df.drug_name) for s, df in (("train", train), ("test", test))}
print("cell lines in test that also appear in train:", len(cells["test"] & cells["train"]), "of", len(cells["test"]))
print("drugs in test that also appear in train:     ", len(drugs["test"] & drugs["train"]), "of", len(drugs["test"]))
'''),
md('''
Every test cell line and drug also appears in train. Rules read cell-line features only, so test fidelity measures agreement with the model on **unseen (cell line, drug) pairs for cell lines the search has already seen**. It is a check against overfitting to specific rows, not a test on new cell lines. This goes in the limitations.
'''),
md('''
## 2. Re-score every rule on the test split

The rule's class is fixed (chosen by majority vote on train). `fidelity` compares it with the model's `class_pred`; `accuracy` compares it with the true per-drug class.
'''),
code('''
rows = []
for _, t in table.iterrows():
    rule = rule_lookup[t["id"]]
    s, _ = U.score_on_split(rule, test, t["condition"])
    rows.append({"id": t["id"], "fidelity_test": s["fidelity"], "accuracy_test": s["accuracy"], "coverage_test": s["coverage"],
                 "n_test_rows": s["n_rows"], "n_test_cells": s["n_cells"],
                 "model_agreement_test": s["share_pred_class_model"], "truth_agreement_test": s["share_pred_class_true"]})
table = table.merge(pd.DataFrame(rows), on="id")
table[["id", "length", "fidelity_train", "fidelity_test", "coverage_train", "coverage_test", "n_test_cells",
       "model_agreement_test", "truth_agreement_test", "accuracy_test"]].round(4)
'''),
md('''
**Reading.**
- Train and test fidelity agree to about 0.005, as expected when test shares cell lines with train.
- `model_agreement_test` is the share of covered test rows where the model says the rule's class, and is what fidelity encodes (`fidelity = p/(1+p)`). Rules agree with the model on roughly 79–98% of what they cover.
- `truth_agreement_test` is agreement with the true per-drug label: lower (about 73–88%), because the model is not perfect; the rule explains the model, not the data.
- Rules sit at coverage of about 2–3% (or 14–18% for one-condition rules). The 2% floor is **binding**: the search pushes coverage down to the floor to buy purity. Fidelity is therefore conditional on the floor value (future work: sweep it).
- At the floor a rule covers only about 18–27 distinct cell lines (`n_test_cells`) out of 866, so "2% of rows" is closer to "20 cell lines, each with many drugs". Rows within a cell line are not independent.
'''),
md('''
## 3. Stability

Subgroup = cell line (≥ 10 covered test rows). For each subgroup, the **agreement rate** is the share of its covered rows on which the model's class equals the rule's class; stability is the variance of that across subgroups (lower = more stable). The contract definition (`metrics.stability`, variance of per-subgroup macro-F1) is computed too, to show why it is not used for ranking.
'''),
code('''
rows = []
for _, t in table.iterrows():
    rule = rule_lookup[t["id"]]
    va, ng, per = U.stability_agreement(rule, test, t["condition"])
    vf, _, perf = U.rule_stability(rule, test, t["condition"])
    rows.append({"id": t["id"], "stability": va, "stability_contract_f1": vf, "n_stability_groups": ng,
                 "min_group_agreement": min(per), "share_groups_perfectly_pure": float(np.mean(np.array(per) == 1.0))})
table = table.merge(pd.DataFrame(rows), on="id")
table[["id", "condition", "model_agreement_test", "stability", "stability_contract_f1", "n_stability_groups",
       "share_groups_perfectly_pure"]].round(4)
'''),
code('''
# Does the contract version track model agreement, or "how many subgroups happen to be perfectly pure"?
print("Spearman(contract stability, share of perfectly pure subgroups):",
      round(spearmanr(table.stability_contract_f1, table.share_groups_perfectly_pure)[0], 2))
print("Spearman(smooth stability,   share of perfectly pure subgroups):",
      round(spearmanr(table.stability, table.share_groups_perfectly_pure)[0], 2))
print("\\nContract stability, mean by model condition (GA rules only):")
print(table[table.extractor == "ga"].groupby("condition")[["model_agreement_test", "stability", "stability_contract_f1"]].mean().round(4))
'''),
md('''
In the contract version the SAM GA rules, which agree most with the model, receive the **worst** stability, because their subgroups are more often perfectly pure and a pure subgroup scores 1.0 against about 0.49 for a 97%-pure one. That is an artefact of macro-F1 on one-class rules, not instability. The smooth version is used for the ranking; the contract version is carried through the sensitivity analysis in §6.
'''),
md('''
## 4. Are the conditions informative? (drop-one check, train rows)

`retention = coverage(rule) / coverage(rule without that condition)`. A retention near 1.0 would mean the condition removes nothing.
'''),
code('''
vac = []
for rid in table.id:
    for v in U.condition_vacuity(rule_lookup[rid], train):
        vac.append({"id": rid, **v})
vac = pd.DataFrame(vac)
vac["is_mutation"] = vac.feature.str.endswith("_mut")
print("conditions in total:", len(vac), "| mutation-side:", int(vac.is_mutation.sum()), "| near-vacuous (retention >= 0.95):", int(vac.vacuous.sum()))
print(vac[vac.vacuous][["id", "feature", "partition", "retention"]].round(3).to_string(index=False))
print("\\nmutation 'Low' (= not mutated) conditions:", int(((vac.partition == "Low") & vac.is_mutation).sum()),
      "| median retention:", round(vac[(vac.partition == "Low") & vac.is_mutation].retention.median(), 3))
vac.to_csv(U.REPORT_DIR / "condition_drop_one_retention.csv", index=False)
'''),
md('''
Nearly all conditions do remove rows: even "not mutated" conditions typically cut 5–50% of the region, i.e. they refine which cell lines are covered. Only a couple are near-vacuous, so `length` is a fair comprehensibility proxy here.
'''),
md('''
## 5. PROMETHEE II, one unified run over all 19 rules
'''),
code('''
CRITERIA = ["fidelity_test", "length", "stability"]
WEIGHTS = np.array([0.5, 0.2, 0.3])          # fidelity, comprehensibility, stability
TYPES = np.array([1, -1, -1])                # 1 = maximise, -1 = minimise

def run_promethee(df, criteria, weights):
    m = df[criteria].to_numpy(float)
    assert not np.isnan(m).any()
    p = m.max(axis=0) - m.min(axis=0); q = np.zeros(len(criteria))
    return PROMETHEE_II("vshape", p=p, q=q)(m, weights, TYPES), p, q

phi, p, q = run_promethee(table, CRITERIA, WEIGHTS)

def phi_manual(m, w, types, p):
    n = len(m); pi = np.zeros((n, n))
    for j in range(m.shape[1]):
        d = (m[:, None, j] - m[None, :, j]) * types[j]
        pi += w[j] * np.clip(d / p[j], 0, 1)
    return (pi.sum(1) - pi.sum(0)) / (n - 1)

chk = phi_manual(table[CRITERIA].to_numpy(float), WEIGHTS, TYPES, p)
print("pymcdm vs independent numpy implementation, max |difference|:", float(np.abs(phi - chk).max()))
assert np.allclose(phi, chk)
table["phi"] = phi
ranked = table.sort_values("phi", ascending=False).reset_index(drop=True)
ranked.index += 1
print("sum of phi (must be ~0):", round(float(phi.sum()), 12))
ranked[["id", "extractor", "condition", "fidelity_test", "length", "stability", "coverage_test", "phi"]].round(4)
'''),
code('''
# top-10 per (extractor, condition) pool = every rule (pools have 3-7 rules)
top = ranked.groupby(["extractor", "condition"]).head(10)
print(top.groupby(["extractor", "condition"]).size())
print("\\nBest rule per pool:")
print(ranked.groupby(["extractor", "condition"]).head(1)[["id", "fidelity_test", "length", "stability", "phi"]].round(4).to_string())
print("\\nMean phi by pool:")
print(ranked.groupby(["extractor", "condition"]).phi.agg(["mean", "median", "count"]).round(3))
'''),
md('''
## 6. Sensitivity (supplementary): does the ranking depend on the provisional weights, or on the stability definition?
'''),
code('''
variants = {
    "base (0.5/0.2/0.3)": (CRITERIA, WEIGHTS),
    "equal weights":        (CRITERIA, np.array([1/3, 1/3, 1/3])),
    "fidelity-heavy":       (CRITERIA, np.array([0.7, 0.1, 0.2])),
    "stability-heavy":      (CRITERIA, np.array([0.3, 0.2, 0.5])),
    "length-heavy":         (CRITERIA, np.array([0.3, 0.4, 0.3])),
    "contract stability":   (["fidelity_test", "length", "stability_contract_f1"], WEIGHTS),
    "train fidelity":       (["fidelity_train", "length", "stability"], WEIGHTS),
}
sens = pd.DataFrame({"id": table.id})
for name, (crit, w) in variants.items():
    sens[name] = run_promethee(table, crit, w)[0]
base_rank = sens["base (0.5/0.2/0.3)"].rank(ascending=False)
out = []
for name in variants:
    ph = sens[name]
    top3 = list(sens.loc[ph.sort_values(ascending=False).index[:3], "id"])
    out.append({"variant": name, "Spearman vs base": round(spearmanr(base_rank, ph.rank(ascending=False))[0], 3), "top 3": top3})
pd.DataFrame(out)
'''),
code('''
# mean phi by condition under each variant (does the SAM-vs-baseline picture change?)
tmp = sens.merge(table[["id", "extractor", "condition"]], on="id")
tmp.groupby(["extractor", "condition"])[list(variants)].mean().round(3)
'''),
md('''
## 7. One matched comparison: the same rule under both models

A rule with identical conditions found under both baseline and SAM is the cleanest like-for-like comparison available (same region of cell lines, different model).
'''),
code('''
sig = {rid: tuple(sorted((c["feature"], c["partition"]) for c in rule_lookup[rid]["conditions"])) for rid in table.id}
table["_sig"] = table.id.map(sig)
dup = table[table.duplicated("_sig", keep=False)].sort_values("_sig")
dup[["id", "condition", "fidelity_test", "model_agreement_test", "stability", "coverage_test"]].round(4) if len(dup) else print("no identical rules across conditions")
'''),
md('''
## 8. Save
'''),
code('''
U.REPORT_DIR.mkdir(parents=True, exist_ok=True)
save = ranked.drop(columns=["_sig"], errors="ignore")
save.to_csv(U.REPORT_DIR / "promethee_ranked_rules.csv", index=False)
top.drop(columns=["_sig"], errors="ignore").to_csv(U.REPORT_DIR / "promethee_top10_per_condition.csv", index=False)
sens.to_csv(U.REPORT_DIR / "promethee_sensitivity.csv", index=False)
json.dump({"criteria": CRITERIA, "weights": WEIGHTS.tolist(), "types": TYPES.tolist(), "preference_function": "vshape",
           "p": p.tolist(), "q": q.tolist(), "config_hash": pools["ga_baseline"]["meta"]["config_hash"],
           "stability": "variance across cell lines (>=10 covered test rows) of per-cell-line agreement rate",
           "fidelity": "macro-F1 vs model class_pred on the TEST split, rule class fixed from train",
           "status": "weights and preference function provisional (not signed off by supervisors)"},
          open(U.REPORT_DIR / "promethee_config.json", "w"), indent=2)
print("saved to", U.REPORT_DIR)
'''),
md('''
## Limitations specific to this notebook

- All 19 rules predict "sensitive"; no "resistant" rule exists, so the pools describe only the sensitive side of the models.
- Stability uses cell lines because tissue / cancer type is unavailable; it is a proxy for the §4.6.1 subtype criterion, not the same quantity.
- Weights and the V-shape preference function are provisional; §6 shows how much they matter.
- Test shares cell lines and drugs with train (§1), so test fidelity does not measure generalisation to new cell lines.
- Rules are global and drug-blind (they read cell-line features only), which caps how pure a covered region can be.
'''),
])

# ===================================================================== 04
n04 = nb([
md('''
# 04 — Statistical testing (Mann–Whitney U)

**Section:** Evaluation layer
**Depends on:** `03_promethee_ranking.ipynb` (`artifacts/reports/promethee_ranked_rules.csv`), `02b_extraction_evolution.ipynb` (`per_run_best_fidelity.csv`)
**Feeds into:** the results write-up

## Design

**Primary test (methodology §4.6.2, locked):** two-sided Mann–Whitney U on the unified φ values, α = 0.05, for
1. baseline vs SAM (pools GA and NSGA-II within each condition), and
2. GA vs NSGA-II (pools baseline and SAM within each extractor).

**Why it needs a companion analysis.** The pooled rule sets are small (10 vs 9 and 13 vs 6 rules). Rules within a pool are not independent draws: they are the survivors of a Pareto filter, many come from the same run, and φ values come from one pairwise-comparison run, so they are mutually dependent (they sum to zero). The test is run as specified, but its power is very low, and it is read together with:
- the same test on each raw criterion (to see what drives any difference),
- an exact permutation test (to confirm the p-values without the normal approximation), and
- a **supplementary per-run analysis**: the 30 extraction runs per (extractor, condition) are genuine independent replicates, so comparing the best fidelity each run achieved gives a far better-powered test of the same question.
'''),
code(SETUP + '''
from scipy.stats import mannwhitneyu, wilcoxon, permutation_test
from src.evaluation import stage5_utils as U
REPORT = U.REPORT_DIR
ranked = pd.read_csv(REPORT / "promethee_ranked_rules.csv")
print(f"loaded {len(ranked)} ranked rules"); print(ranked.groupby(["extractor", "condition"]).size())

def mwu(a, b, la, lb, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - 2 * u / (len(a) * len(b))        # rank-biserial; sign follows the original notebook (negative: A tends to be higher)
    return {"comparison": f"{la} vs {lb}", "n_a": len(a), "n_b": len(b), "median_a": float(np.median(a)),
            "median_b": float(np.median(b)), "U": float(u), "p_value": float(p), "significant_at_0.05": bool(p < alpha),
            "rank_biserial": float(r)}
'''),
md('''
## 1. Primary test on φ
'''),
code('''
groups = {
    "baseline vs sam": (ranked[ranked.condition == "baseline"], ranked[ranked.condition == "sam"]),
    "ga vs nsga":      (ranked[ranked.extractor == "ga"],       ranked[ranked.extractor == "nsga"]),
}
primary = pd.DataFrame([mwu(a.phi, b.phi, *k.split(" vs ")) for k, (a, b) in groups.items()])
primary
'''),
md('''
## 2. What drives it: the same test on each criterion, plus an exact permutation check
`rank_biserial` is negative when group A tends to have the **higher** value of that column.
'''),
code('''
rows = []
for k, (a, b) in groups.items():
    for col in ["phi", "fidelity_test", "stability", "length", "coverage_test"]:
        r = mwu(a[col], b[col], *k.split(" vs "))
        perm = permutation_test((a[col].to_numpy(float), b[col].to_numpy(float)),
                                lambda x, y: np.median(x) - np.median(y), n_resamples=20000, random_state=0,
                                alternative="two-sided").pvalue
        rows.append({"groups": k, "variable": col, "median_a": r["median_a"], "median_b": r["median_b"],
                     "U": r["U"], "p_mwu": r["p_value"], "p_permutation_(diff of medians)": float(perm), "rank_biserial": r["rank_biserial"]})
per_criterion = pd.DataFrame(rows)
per_criterion.round(4)
'''),
md('''
## 3. How much can this sample detect?

Simulated power of the two-sided Mann–Whitney test at α = 0.05 for a shift of d standard deviations between two normal groups of the observed sizes.
'''),
code('''
rng = np.random.default_rng(1)
def power(n1, n2, d, sims=3000):
    hit = 0
    for _ in range(sims):
        hit += mannwhitneyu(rng.normal(0, 1, n1), rng.normal(d, 1, n2), alternative="two-sided").pvalue < .05
    return hit / sims
pw = pd.DataFrame({"shift d (SD)": [0.5, 1.0, 1.5, 2.0],
                   "power, 10 vs 9 (baseline vs SAM)": [power(10, 9, d) for d in (0.5, 1, 1.5, 2)],
                   "power, 13 vs 6 (GA vs NSGA-II)":   [power(13, 6, d) for d in (0.5, 1, 1.5, 2)],
                   "power, 30 vs 30 (per-run, §4)":     [power(30, 30, d) for d in (0.5, 1, 1.5, 2)]})
pw.round(2)
'''),
md('''
## 4. Supplementary: per-run comparison (30 independent runs per group)

Statistic per run: the best fidelity (train) in the run's final front. Runs are independent replicates. For baseline vs SAM the same seeds were used in both conditions, so the 30 runs are also **paired by seed** (identical starting populations); both the independent test and the paired Wilcoxon signed-rank test are shown. GA vs NSGA-II uses the independent test only.
'''),
code('''
pr = pd.read_csv(REPORT / "per_run_best_fidelity.csv")
rows = []
for e in ("ga", "nsga"):
    a = pr[(pr.extractor == e) & (pr.condition == "baseline")].sort_values("run_id")
    b = pr[(pr.extractor == e) & (pr.condition == "sam")].sort_values("run_id")
    r = mwu(a.best_fidelity_train, b.best_fidelity_train, f"{e}: baseline", f"{e}: sam")
    w = wilcoxon(a.best_fidelity_train.to_numpy(), b.best_fidelity_train.to_numpy(), zero_method="wilcox")
    r.update({"paired_wilcoxon_p": float(w.pvalue), "runs_where_sam_higher": int((b.best_fidelity_train.to_numpy() > a.best_fidelity_train.to_numpy()).sum()),
              "mean_diff_sam_minus_baseline": float((b.best_fidelity_train.to_numpy() - a.best_fidelity_train.to_numpy()).mean())})
    rows.append(r)
for c in ("baseline", "sam"):
    a = pr[(pr.extractor == "ga") & (pr.condition == c)]; b = pr[(pr.extractor == "nsga") & (pr.condition == c)]
    rows.append(mwu(a.best_fidelity_train, b.best_fidelity_train, f"{c}: ga", f"{c}: nsga"))
per_run = pd.DataFrame(rows)
per_run.round(5)
'''),
md('''
## 5. Save
'''),
code('''
primary.to_csv(REPORT / "mannwhitney_results.csv", index=False)
per_criterion.to_csv(REPORT / "mannwhitney_per_criterion.csv", index=False)
per_run.to_csv(REPORT / "mannwhitney_per_run_supplementary.csv", index=False)
pw.to_csv(REPORT / "mannwhitney_power.csv", index=False)
print("saved to", REPORT)
'''),
md('''
## Interpretation notes (fill from the printed results)

- State the primary result exactly as computed, with the effect size and the sample sizes, and say that the test has little power at these n (§3).
- Use §2 to say which criterion carries any difference, and §4 to say whether a larger independent sample (runs) supports or contradicts the primary result.
- Do not describe a non-significant result as evidence of "no effect"; it is absence of evidence at this sample size.
'''),
])

# ===================================================================== 05
n05 = nb([
md('''
# 05 — GSEA / pathway over-representation (real rules)

**Section:** Evaluation layer
**Depends on:** the four rule pools, `03_promethee_ranking.ipynb` (ranked table), `feature_map.json`
**Methodology:** §4.6.3, GSEApy over-representation of the genes in each pool's rules against KEGG

## What is tested, and what is not

- **Gene-level test (mutation-side conditions only).** Genes come from conditions on `<GENE>_mut` features (real HGNC symbols). The **background is the 534 COSMIC driver genes** the mutation features were drawn from (not the whole genome), because rules can only ever select from those 534.
- **A "Low" mutation condition means "not mutated".** Such genes are not evidence that the gene's biology matters, so the **primary query list uses only genes required to be mutated (`High`)**. A list with all mutation genes is a secondary analysis.
- **Pathway-side conditions are not gene lists.** The 38 pathway-activity features are KEGG pathways already (ssGSEA scores), so they are reported as direct pathway usage in §3, with no enrichment test.

## Network note
The Enrichr / KEGG service is **not reachable from the sandbox where the other notebooks were run**. Cell 4 queries Enrichr and, if it cannot connect, prints a message and still saves the query lists so the identical analysis can be run where there is network access (or by pasting the lists into the Enrichr web page). No enrichment result is fabricated or substituted.
'''),
code(SETUP + '''
import gseapy as gp
from src.evaluation import stage5_utils as U
pools, rule_lookup, feature_map = U.load_pools()
ranked = pd.read_csv(U.REPORT_DIR / "promethee_ranked_rules.csv")
mut_genes_all = sorted({v["gene_symbol"] for k, v in feature_map.items() if k.endswith("_mut")})
print("background (driver genes available to rules):", len(mut_genes_all))
'''),
md('''
## 1. Genes appearing in each pool's rules
'''),
code('''
rows = []
for name, pool in pools.items():
    for r in pool["rules"]:
        for c in r["conditions"]:
            if c["feature"].endswith("_mut"):
                rows.append({"pool": name, "rule": r["id"], "gene": feature_map[c["feature"]]["gene_symbol"], "partition": c["partition"]})
genes_long = pd.DataFrame(rows)
genes_long["condition"] = genes_long.pool.str.split("_").str[1]
genes_long["extractor"] = genes_long.pool.str.split("_").str[0]
print("mutation-side conditions:", len(genes_long), "| of which 'High' (mutated):", int((genes_long.partition == "High").sum()))
summary = genes_long.groupby("pool").agg(conditions=("gene", "size"), genes=("gene", "nunique"),
            mutated_High=("partition", lambda s: int((s == "High").sum())))
summary
'''),
code('''
def lists(df):
    return {
        "mutated (High) genes": sorted(df.loc[df.partition == "High", "gene"].unique()),
        "all mutation genes":   sorted(df["gene"].unique()),
    }
query = {}
for name in pools: query[name] = lists(genes_long[genes_long.pool == name])
for cond in ("baseline", "sam"): query[f"{cond} (GA+NSGA-II)"] = lists(genes_long[genes_long.condition == cond])
ql = pd.DataFrame([{"query": k, "list_type": t, "n_genes": len(g), "genes": ", ".join(g)} for k, d in query.items() for t, g in d.items()])
ql.to_csv(U.REPORT_DIR / "gsea_query_gene_lists.csv", index=False)
ql
'''),
md('''
With 0–7 genes per list, any over-representation test has very little power: with a 534-gene background, a gene set must contain nearly all of a short list to reach significance. Results should be read as exploratory.
'''),
md('''
## 2. Over-representation against KEGG (needs network access to Enrichr)
'''),
code('''
LIBRARY = "KEGG_2021_Human"
results, failed = [], None
for qname, d in query.items():
    for ltype, glist in d.items():
        if len(glist) < 2:
            continue
        try:
            res = gp.enrichr(gene_list=glist, gene_sets=LIBRARY, background=mut_genes_all, outdir=None).results
        except Exception as e:                         # no network / service unavailable
            failed = f"{type(e).__name__}: {str(e)[:200]}"
            break
        res.insert(0, "list_type", ltype); res.insert(0, "query", qname)
        results.append(res)
    if failed: break
if failed:
    print("Enrichr not reachable from this environment ->", failed)
    print("The query lists are saved in", U.REPORT_DIR / "gsea_query_gene_lists.csv", "- re-run this notebook where Enrichr is reachable.")
    enr = pd.DataFrame()
else:
    enr = pd.concat(results, ignore_index=True)
    enr.to_csv(U.REPORT_DIR / "gsea_enrichment_results.csv", index=False)
    print(enr.sort_values("Adjusted P-value").head(15)[["query", "list_type", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]].to_string(index=False))
    print("\\nterms with adjusted p < 0.05:", int((enr["Adjusted P-value"] < 0.05).sum()), "of", len(enr))
'''),
md('''
## 3. Pathway-feature usage (no external database needed)

Which of the 38 KEGG pathway-activity features the rules condition on, and in which direction.
'''),
code('''
rows = []
for name, pool in pools.items():
    for r in pool["rules"]:
        for c in r["conditions"]:
            if c["feature"].endswith("_activity"):
                rows.append({"pool": name, "rule": r["id"], "pathway": feature_map[c["feature"]]["gene_symbol"], "partition": c["partition"]})
pw = pd.DataFrame(rows)
usage = (pw.groupby(["pathway", "partition"]).agg(rules=("rule", "nunique"), pools=("pool", lambda s: ", ".join(sorted(set(s)))))
           .reset_index().sort_values("rules", ascending=False))
usage.to_csv(U.REPORT_DIR / "pathway_feature_usage.csv", index=False)
usage
'''),
code('''
# how much of the pool depends on each pathway, and is the same pathway used under both models?
share = pw.groupby(["pool"]).apply(lambda d: d.groupby("pathway").rule.nunique() / len(pools[d.name]["rules"])).unstack(0).fillna(0).round(2)
share.loc[share.max(axis=1) >= 0.3]
'''),
md('''
## Interpretation notes

- Report the pathway usage (§3) as the main biologically readable output: which KEGG pathways the model's sensitive-region rules rely on, and whether SAM and baseline rely on the same ones.
- Report the gene-level enrichment (§2) with its power limitation. A null result there is expected at these list sizes and is not evidence against biological relevance.
- Rules are drug-blind and cover about 20 cell lines at the floor, so a pathway such as B-cell receptor signalling may be marking a lineage (haematological cell lines) rather than a drug-specific mechanism. That is a hypothesis for future work (needs tissue annotation), not a finding.
'''),
])

for name, n in {"02b_extraction_evolution.ipynb": n02b, "03_promethee_ranking.ipynb": n03,
                "04_statistical_testing.ipynb": n04, "05_gsea_enrichment.ipynb": n05}.items():
    nbf.write(n, NB / name)
print("built")
