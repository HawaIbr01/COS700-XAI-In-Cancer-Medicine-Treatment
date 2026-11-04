"""
Helpers shared by the Stage 5 evaluation notebooks (03 PROMETHEE, 04 Mann-Whitney,
05 GSEA) so each notebook stays short and the logic lives in one place.

Everything here reads the corrected extraction dataset
(data/processed/real_extraction/) and the four rule pools
(artifacts/rules/{ga,nsga}_{baseline,sam}.json). It never touches the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import metrics
from src.common.schema import load_rule_pool

POOL_NAMES = ["ga_baseline", "ga_sam", "nsga_baseline", "nsga_sam"]
DATA_DIR = Path("data/processed/real_extraction/data")
PRED_DIR = Path("data/processed/real_extraction/predictions")
RULES_DIR = Path("artifacts/rules")
REPORT_DIR = Path("artifacts/reports")
MIN_ROWS_PER_CELL = 10   # a cell line needs >= this many covered test rows to count as a stability subgroup


def load_pools(rules_dir: Path = RULES_DIR, data_dir: Path = DATA_DIR):
    """Load + validate the four pools; returns (pools, rule_lookup, feature_map)."""
    feature_map = json.load(open(data_dir / "feature_map.json"))
    pools = {n: load_rule_pool(rules_dir / f"{n}.json", feature_map=feature_map) for n in POOL_NAMES}
    hashes = {n: p["meta"]["config_hash"] for n, p in pools.items()}
    assert len(set(hashes.values())) == 1, f"config_hash differs between pools: {hashes}"
    lookup = {r["id"]: r for p in pools.values() for r in p["rules"]}
    return pools, lookup, feature_map


def load_split(split: str, data_dir: Path = DATA_DIR, pred_dir: Path = PRED_DIR) -> pd.DataFrame:
    """One split of gdsc_processed + both models' class_pred
    (columns class_pred_baseline, class_pred_sam)."""
    df = pd.read_parquet(data_dir / "gdsc_processed.parquet")
    df = df[df["split"] == split].reset_index(drop=True)
    for cond in ("baseline", "sam"):
        p = pd.read_parquet(pred_dir / f"{cond}_{split}.parquet")[["row_id", "class_pred"]]
        p = p.rename(columns={"class_pred": f"class_pred_{cond}"})
        df = df.merge(p, on="row_id", how="inner", validate="one_to_one")
    return df


def rule_table(pools: dict) -> pd.DataFrame:
    rows = []
    for name, pool in pools.items():
        for r in pool["rules"]:
            rows.append({
                "id": r["id"], "pool": name, "extractor": pool["meta"]["extractor"],
                "condition": pool["meta"]["condition"], "predicted_class": r["predicted_class"],
                "length": r["length"], "fidelity_train": r["fidelity_f1"], "coverage_train": r["coverage"],
                "n_runs_found": len(r["run_ids"]),
            })
    return pd.DataFrame(rows)


def score_on_split(rule: dict, df: pd.DataFrame, condition: str):
    """Fidelity (vs the model's class_pred) and accuracy (vs the true per-drug class) of one
    rule on one split. The rule's class is FIXED (it was chosen on train)."""
    col = f"class_pred_{condition}"
    mask = metrics.evaluate_rule_mask(df, rule["conditions"])
    n = int(mask.sum())
    out = {"coverage": float(mask.mean()), "n_rows": n,
           "n_cells": int(df.loc[mask, "cell_name"].nunique()) if n else 0,
           "n_drugs": int(df.loc[mask, "drug_name"].nunique()) if n else 0}
    if n == 0:
        out.update(fidelity=np.nan, accuracy=np.nan, share_pred_class_model=np.nan, share_pred_class_true=np.nan)
        return out, mask
    rule_pred = pd.Series(rule["predicted_class"], index=df.index[mask])
    model_pred = df.loc[mask, col]
    truth = df.loc[mask, "response_class"].astype(str)
    out["fidelity"] = metrics.fidelity_f1(rule_pred, model_pred)
    out["accuracy"] = metrics.fidelity_f1(rule_pred, truth)
    out["share_pred_class_model"] = float((model_pred == rule["predicted_class"]).mean())
    out["share_pred_class_true"] = float((truth == rule["predicted_class"]).mean())
    return out, mask


def rule_stability(rule: dict, df: pd.DataFrame, condition: str, group_col: str = "cell_name",
                   min_rows: int = MIN_ROWS_PER_CELL):
    """metrics.stability() with subgroup = cell line (see notebook 03 for why).
    Returns (variance, n_groups, per_group_fidelity list)."""
    col = f"class_pred_{condition}"
    mask = metrics.evaluate_rule_mask(df, rule["conditions"])
    sub = df.loc[mask]
    rp, mp = {}, {}
    for g, grp in sub.groupby(group_col, observed=True):
        if len(grp) < min_rows:
            continue
        mp[g] = grp[col]
        rp[g] = pd.Series(rule["predicted_class"], index=grp.index)
    if len(rp) < 2:
        return np.nan, len(rp), []
    per = [metrics.fidelity_f1(rp[g], mp[g]) for g in rp]
    return metrics.stability(rp, mp), len(rp), per


def stability_agreement(rule: dict, df: pd.DataFrame, condition: str, group_col: str = "cell_name",
                        min_rows: int = MIN_ROWS_PER_CELL):
    """Smooth stability: variance, across subgroups, of the per-subgroup AGREEMENT RATE
    (share of the subgroup's covered rows on which the model's class equals the rule's class).

    Why not metrics.stability() as the ranking criterion: that function takes the variance of
    per-subgroup macro-F1, and for a rule that always predicts ONE class, macro-F1 is
    discontinuous in purity: p/(1+p) < 0.5 for any mixed subgroup, but exactly 1.0 when every
    covered row in the subgroup agrees. A 97%-pure subgroup scores ~0.49 and a 100%-pure one
    1.0, so the variance is dominated by whether some subgroup happens to be perfectly pure
    rather than by how consistently the rule tracks the model. The agreement rate is the
    continuous quantity underlying it. Both versions are reported in notebook 03.
    Returns (variance, n_groups, per_group_agreement list)."""
    col = f"class_pred_{condition}"
    mask = metrics.evaluate_rule_mask(df, rule["conditions"])
    sub = df.loc[mask]
    per = []
    for g, grp in sub.groupby(group_col, observed=True):
        if len(grp) < min_rows:
            continue
        per.append(float((grp[col] == rule["predicted_class"]).mean()))
    if len(per) < 2:
        return np.nan, len(per), per
    return float(np.var(per)), len(per), per


def condition_vacuity(rule: dict, df_train: pd.DataFrame, retain_threshold: float = 0.95):
    """Drop-one analysis on TRAIN rows. For each condition,
        retention = coverage(rule) / coverage(rule with that one condition removed).
    A retention near 1.0 means the condition removes almost nothing the other conditions had
    not already removed, i.e. it is near-vacuous (typically '<gene>_mut Low' = 'not mutated',
    which almost every cell line satisfies). For a one-condition rule the retention is the
    rule's own coverage. Returns a list of dicts (one per condition)."""
    conds = rule["conditions"]
    full = metrics.evaluate_rule_mask(df_train, conds)
    out = []
    for i, c in enumerate(conds):
        rest = conds[:i] + conds[i + 1:]
        if rest:
            m_rest = metrics.evaluate_rule_mask(df_train, rest)
            ret = float(full.sum() / m_rest.sum()) if m_rest.sum() else np.nan
        else:
            ret = float(full.mean())
        out.append({"feature": c["feature"], "partition": c["partition"], "retention": ret,
                    "vacuous": bool(ret >= retain_threshold)})
    return out
