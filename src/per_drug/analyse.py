"""Stage 2 analysis of the per-drug pilot (docs/per_drug_analysis_plan.md).

Reads <root>/cells/<drug>/<arm>__<ext>__fold<k>.json and writes, next to <root>:
  results_tidy.csv           one row per drug x arm x extractor x fold
  results_drug_level.csv     one row per drug x arm x extractor (fold-averaged, stability)
  comparisons.csv            planned comparisons (paired Wilcoxon, BH within metric, median/mean difference)
  comparisons.md             markdown view of comparisons.csv
  positive_controls.csv      hit / top-rule hit per control drug x arm x extractor x fold
  wallclock.csv              wall-clock seconds per cell and per arm x extractor
  kegg_gene_lists.csv, kegg_ora_results.csv   pooled KEGG over-representation per arm (only with --kegg)
  RESULTS.md                 numbers only

PROMETHEE II: once per fold, per drug and extractor, over the union of the three arms' pools for that
fold, criteria held-out fidelity (max, 5/7) and length (min, 2/7); V-shape, p = observed range, q = 0.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.per_drug.config import DESIGN
from src.per_drug.scoring import POSITIVE_CONTROLS, jaccard, pairwise_stats

ARMS = ["adam", "adam_wd", "sam"]
EXTS = ["ga", "nsga"]
W = np.array([0.5, 0.2]) / 0.7            # fidelity, length (stability dropped, renormalised)
TYPES = np.array([1, -1])
FEATURE_MAP = Path("data/processed/real_extraction/data/feature_map.json")


def promethee(fid, length):
    """phi per rule (pymcdm PROMETHEE_II, V-shape, p = range, q = 0). NaN array if < 2 rules."""
    from pymcdm.methods import PROMETHEE_II
    n = len(fid)
    if n < 2:
        return np.full(n, np.nan)
    m = np.column_stack([fid, length]).astype(float)
    p = m.max(axis=0) - m.min(axis=0)
    p[p == 0] = 1.0                      # all-equal criterion contributes 0 either way
    return np.asarray(PROMETHEE_II("vshape", p=p, q=np.zeros(2))(m, W, TYPES), float)


def load_cells(root: Path) -> dict:
    cells = {}
    for f in sorted((root / "cells").glob("*/*.json")):
        c = json.loads(f.read_text())
        c.pop("traces", None)          # monitoring only; never used here, so not kept in memory
        k = c["key"]
        cells[(k["drug"], k["arm"], k["extractor"], k["fold"])] = c
    return cells


def _nanmedian(x):
    x = [v for v in x if v is not None and not math.isnan(v)]
    return float(np.median(x)) if x else float("nan")


def _nanmean(x):
    x = [v for v in x if v is not None and not math.isnan(v)]
    return float(np.mean(x)) if x else float("nan")


def build_tidy(cells: dict):
    rows, ctrl_rows = [], []
    drugs = sorted({k[0] for k in cells})
    for drug in drugs:
        for ext in EXTS:
            for fold in range(DESIGN["n_folds"]):
                # PROMETHEE over the three arms' scored rules for this fold
                recs = []
                for arm in ARMS:
                    c = cells.get((drug, arm, ext, fold))
                    if c is None:
                        continue
                    pool = {r["id"]: r for r in c["pool"]}
                    for h in c["heldout"]:
                        recs.append((arm, h, pool[h["id"]]))
                ok = [(a, h, r) for a, h, r in recs if not math.isnan(h["fidelity"])]
                phi = promethee([h["fidelity"] for _, h, _ in ok], [h["length"] for _, h, _ in ok]) if ok else np.array([])
                phi_of = {h["id"]: v for (_, h, _), v in zip(ok, phi)}
                top_id = ok[int(np.nanargmax(phi))][1]["id"] if len(ok) > 1 and not np.isnan(phi).all() else None
                rule_of = {r["id"]: r for _, _, r in recs}
                for arm in ARMS:
                    c = cells.get((drug, arm, ext, fold))
                    if c is None:
                        continue
                    h = c["heldout"]
                    fid = [x["fidelity"] for x in h]
                    pool_genes = {g for r in c["pool"] for g in r["genes"]}
                    row = {
                        "drug": drug, "arm": arm, "extractor": ext, "fold": fold,
                        "n_pool_rules": len(c["pool"]), "n_scored_rules": int(sum(not math.isnan(v) for v in fid)),
                        "n_rules_nan_heldout_fidelity": int(sum(math.isnan(v) for v in fid)),
                        "n_rules_low_heldout_coverage_flag": int(sum(x["low_coverage_flag"] for x in h)),
                        "median_heldout_fidelity": _nanmedian(fid),
                        "median_heldout_fidelity_net_train": _nanmedian([x["fidelity_net_train"] for x in h]),
                        "median_heldout_fidelity_net_other": _nanmedian([x["fidelity_net_other"] for x in h]),
                        "median_length": _nanmedian([x["length"] for x in h]),
                        "mean_phi": _nanmean([phi_of.get(x["id"], float("nan")) for x in h]),
                        "n_rules_in_promethee": int(sum(1 for x in h if x["id"] in phi_of)),
                        "n_pool_genes": len(pool_genes),
                        "pool_genes": ";".join(sorted(pool_genes)),
                        "train_rows_fid": c["sizes"]["fid"], "heldout_rows": c["sizes"]["heldout"],
                        "wall_clock_s": c["wall_clock_s"],
                    }
                    if drug in POSITIVE_CONTROLS:
                        feats = POSITIVE_CONTROLS[drug]
                        hit_ids = [r["id"] for r in c["pool"]
                                   if any(cd["feature"] in feats and cd["partition"] == "High" for cd in r["conditions"])]
                        top_rule = rule_of.get(top_id) if top_id else None
                        top_hit = bool(top_rule is not None and top_rule["id"].split("|")[1] == arm and any(
                            cd["feature"] in feats and cd["partition"] == "High" for cd in top_rule["conditions"]))
                        # top-ranked rule over the three-arm union; "in top rule" is about that single rule
                        top_feats_hit = bool(top_rule is not None and any(
                            cd["feature"] in feats and cd["partition"] == "High" for cd in top_rule["conditions"]))
                        top_arm = top_rule["id"].split("|")[1] if top_rule else None
                        row.update({"control_hit_in_pool": bool(hit_ids), "control_n_rules_with_hit": len(hit_ids)})
                        ctrl_rows.append({
                            "drug": drug, "control_features": ",".join(feats), "arm": arm, "extractor": ext, "fold": fold,
                            "hit_in_pool": bool(hit_ids), "n_rules_with_hit": len(hit_ids), "n_pool_rules": len(c["pool"]),
                            "top_phi_rule_arm": top_arm, "top_phi_rule_contains_control_High": top_feats_hit,
                            "top_phi_rule_is_this_arm_and_contains_control_High": top_hit,
                            "flag_prevalence_gt_30pct": any(f in ("BRAF_mut", "EGFR_mut") for f in feats),
                        })
                    rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(ctrl_rows)


def drug_level(tidy: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (drug, arm, ext), g in tidy.groupby(["drug", "arm", "extractor"]):
        g = g.sort_values("fold")
        sets = [set(s.split(";")) if s else set() for s in g["pool_genes"]]
        ps = pairwise_stats(sets)
        out.append({
            "drug": drug, "arm": arm, "extractor": ext, "n_folds": len(g),
            "heldout_fidelity": _nanmean(g["median_heldout_fidelity"]),
            "n_folds_fidelity_defined": int(g["median_heldout_fidelity"].notna().sum()),
            "heldout_fidelity_net_train": _nanmean(g["median_heldout_fidelity_net_train"]),
            "heldout_fidelity_net_other": _nanmean(g["median_heldout_fidelity_net_other"]),
            "phi": _nanmean(g["mean_phi"]), "n_folds_phi_defined": int(g["mean_phi"].notna().sum()),
            "length": _nanmean(g["median_length"]),
            "n_rules_low_coverage_flag": int(g["n_rules_low_heldout_coverage_flag"].sum()),
            "n_rules_nan_heldout_fidelity": int(g["n_rules_nan_heldout_fidelity"].sum()),
            "n_empty_pools": int((g["n_pool_rules"] == 0).sum()),
            "stability_jaccard_folds": ps["jaccard"], "n_fold_sets_empty": ps["n_empty_sets"],
            "n_fold_pairs_jaccard_defined": ps["n_pairs_jaccard_defined"],
            "wall_clock_s": float(g["wall_clock_s"].sum()),
        })
    return pd.DataFrame(out)


def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    idx = np.where(~np.isnan(p))[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out


def _wilcoxon(d, **kw):
    d = np.asarray(d, float)
    if len(d) == 0 or np.all(d == 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(d, **kw).pvalue)
    except ValueError:
        return float("nan")


def comparisons(dl: pd.DataFrame) -> pd.DataFrame:
    metrics = {"heldout_fidelity": "fidelity", "phi": "phi", "length": "length",
               "stability_jaccard_folds": "stability_jaccard_folds"}
    comps = {"SAM_minus_AdamWD": ("sam", "adam_wd"), "AdamWD_minus_Adam": ("adam_wd", "adam"),
             "sam_vs_adam (proposal contrast, confounded)": ("sam", "adam")}
    rows = []
    for metric in metrics:
        for ext in EXTS:
            for cname, (a, b) in comps.items():
                wide = dl[dl.extractor == ext].pivot(index="drug", columns="arm", values=metric)
                d = (wide[a] - wide[b]).dropna()
                n = len(d)
                r = {"metric": metric, "extractor": ext, "comparison": cname, "n_drugs_paired": n,
                     "n_drugs_total": len(wide), "median_diff": float(d.median()) if n else float("nan"),
                     "mean_diff": float(d.mean()) if n else float("nan"),
                     "n_zero_diffs": int((d == 0).sum()), "wilcoxon_p": _wilcoxon(d)}
                rows.append(r)
    df = pd.DataFrame(rows)
    df["wilcoxon_p_bh"] = np.nan
    for metric, idx in df.groupby("metric").groups.items():
        df.loc[idx, "wilcoxon_p_bh"] = bh(df.loc[idx, "wilcoxon_p"].to_numpy())
    return df


def kegg(tidy_cells: dict, outdir: Path):
    """Genes in High mutation conditions of any rule, pooled per arm over drugs/extractors/folds."""
    fm = json.loads(FEATURE_MAP.read_text())
    background = sorted({v["gene_symbol"] for k, v in fm.items() if k.endswith("_mut")})
    lists = {a: set() for a in ARMS}
    for (drug, arm, ext, fold), c in tidy_cells.items():
        for r in c["pool"]:
            for cd in r["conditions"]:
                if cd["feature"].endswith("_mut") and cd["partition"] == "High":
                    lists[arm].add(cd["feature"][:-4])
    gl = pd.DataFrame([{"arm": a, "n_genes": len(g), "genes": ";".join(sorted(g)), "n_background": len(background)}
                       for a, g in lists.items()])
    gl.to_csv(outdir / "kegg_gene_lists.csv", index=False)
    res, err = [], None
    try:
        import gseapy as gp
        for a, g in lists.items():
            if len(g) < 2:
                continue
            e = gp.enrichr(gene_list=sorted(g), gene_sets="KEGG_2021_Human", background=background, outdir=None).results
            e.insert(0, "arm", a)
            res.append(e)
    except Exception as ex:
        err = f"{type(ex).__name__}: {str(ex)[:300]}"
    if res:
        pd.concat(res, ignore_index=True).to_csv(outdir / "kegg_ora_results.csv", index=False)
    return gl, (pd.concat(res, ignore_index=True) if res else None), err


def fmt(x, nd=4):
    return "NaN" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_md(outdir, tidy, dl, comp, ctrl, wall, gl, enr, err, n_seeds_note):
    L = ["# Per-drug pilot: RESULTS (numbers only)", "",
         f"Drugs: {tidy.drug.nunique()}; cells: {len(tidy)} (expected {tidy.drug.nunique() * 30}); {n_seeds_note}.", ""]
    L += ["## Drug-level means by arm and extractor (mean over drugs)", ""]
    cols = ["heldout_fidelity", "phi", "length", "stability_jaccard_folds",
            "heldout_fidelity_net_train", "heldout_fidelity_net_other"]
    t = dl.groupby(["extractor", "arm"])[cols].mean().reset_index()
    L += ["| extractor | arm | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for _, r in t.iterrows():
        L.append(f"| {r.extractor} | {r.arm} | " + " | ".join(fmt(r[c]) for c in cols) + " |")
    L += ["", "## Counts", ""]
    cnt = dl.groupby(["extractor", "arm"]).agg(empty_pools=("n_empty_pools", "sum"), low_cov_rules=("n_rules_low_coverage_flag", "sum"),
                                              nan_fid_rules=("n_rules_nan_heldout_fidelity", "sum"),
                                              empty_fold_sets=("n_fold_sets_empty", "sum")).reset_index()
    L += ["| extractor | arm | empty pools (folds) | rules covering <10 held-out rows | rules with NaN held-out fidelity | empty fold gene sets |", "|---|---|---|---|---|---|"]
    for _, r in cnt.iterrows():
        L.append(f"| {r.extractor} | {r.arm} | {r.empty_pools} | {r.low_cov_rules} | {r.nan_fid_rules} | {r.empty_fold_sets} |")
    L += ["", "## Planned comparisons (paired by drug)", "",
          "| metric | extractor | comparison | n | median diff | mean diff | zero diffs | Wilcoxon p | BH p |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        L.append(f"| {r.metric} | {r.extractor} | {r.comparison} | {r.n_drugs_paired} | {fmt(r.median_diff)} | "
                 f"{fmt(r.mean_diff)} | {r.n_zero_diffs} | {fmt(r.wilcoxon_p)} | {fmt(r.wilcoxon_p_bh)} |")
    L += ["", "## Positive controls (folds with a hit, out of 5)", "",
          "| drug | features | extractor | arm | folds with hit in pool | folds where top-phi rule (3-arm union) contains the feature (High) |", "|---|---|---|---|---|---|"]
    if len(ctrl):
        g = ctrl.groupby(["drug", "control_features", "extractor", "arm"]).agg(
            hit=("hit_in_pool", "sum"), top=("top_phi_rule_contains_control_High", "sum")).reset_index()
        for _, r in g.iterrows():
            L.append(f"| {r.drug} | {r.control_features} | {r.extractor} | {r.arm} | {r.hit}/5 | {r.top}/5 |")
    L += ["", "## Wall-clock (seconds, summed over cells)", "", "| extractor | arm | total s | mean s per cell |", "|---|---|---|---|"]
    for _, r in wall.iterrows():
        L.append(f"| {r.extractor} | {r.arm} | {r.total_s:.0f} | {r.mean_cell_s:.1f} |")
    if gl is not None:                   # KEGG section only when the KEGG step was run (--kegg)
        L += ["", "## Pooled KEGG over-representation (High mutation genes per arm; background 534 genes; BH)", ""]
        L += ["| arm | n genes |", "|---|---|"] + [f"| {r.arm} | {r.n_genes} |" for _, r in gl.iterrows()]
        if err:
            L += ["", f"Enrichr not run: {err}. Gene lists saved in kegg_gene_lists.csv."]
        elif enr is not None:
            L += ["", "Terms with adjusted p < 0.05 per arm:", ""]
            for a, g in enr.groupby("arm"):
                s = g[g["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")
                L.append(f"- {a}: {len(s)} of {len(g)} terms" + ("".join(f"; {r.Term} (adj p {r['Adjusted P-value']:.3g}, overlap {r.Overlap})" for _, r in s.head(10).iterrows())))
    (outdir / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--kegg", action="store_true", help="run the pooled KEGG enrichment step (needs network); off by default")
    ap.add_argument("--no-kegg", action="store_true", help="force the KEGG step off (the default)")
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root.parent
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(root)
    tidy, ctrl = build_tidy(cells)
    tidy.drop(columns=["pool_genes"]).to_csv(out / "results_tidy.csv", index=False)
    tidy[["drug", "arm", "extractor", "fold", "pool_genes"]].to_csv(out / "fold_gene_sets.csv", index=False)
    dl = drug_level(tidy)
    dl.to_csv(out / "results_drug_level.csv", index=False)
    comp = comparisons(dl)
    comp.to_csv(out / "comparisons.csv", index=False)
    (out / "comparisons.md").write_text(comp.round(4).to_markdown(index=False) + "\n", encoding="utf-8")
    ctrl.to_csv(out / "positive_controls.csv", index=False)
    wall = tidy.groupby(["extractor", "arm"]).wall_clock_s.agg(total_s="sum", mean_cell_s="mean").reset_index()
    tidy[["drug", "arm", "extractor", "fold", "wall_clock_s"]].to_csv(out / "wallclock.csv", index=False)
    if a.kegg and not a.no_kegg:
        gl, enr, err = kegg(cells, out)
    else:
        gl, enr, err = None, None, None
    n_seeds = sorted({c["key"]["n_seeds"] for c in cells.values()})
    write_md(out, tidy, dl, comp, ctrl, wall, gl, enr, err, f"seeds per cell: {n_seeds}")
    print("wrote outputs to", out)


if __name__ == "__main__":
    main()
