"""Held-out scoring of a rule pool, stability across folds/seeds, positive-control hits.

Fidelity of a rule on a row set = macro-F1 between the rule's FIXED class (majority
class on the extraction rows) and the network's class_pred on the rows the rule covers
(src.common.metrics.fidelity_f1, the project's definition).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from src.common import metrics
from src.per_drug.config import DESIGN

POSITIVE_CONTROLS = {
    "Erlotinib": ["EGFR_mut"],
    "PLX-4720": ["BRAF_mut"],
    "Olaparib": ["BRCA1_mut", "BRCA2_mut"],
    "Selumetinib": ["BRAF_mut", "KRAS_mut"],
}


def _fid(pred_class: str, model_preds: pd.Series) -> float:
    if len(model_preds) == 0:
        return float("nan")
    return metrics.fidelity_f1(pd.Series(pred_class, index=model_preds.index), model_preds)


def score_rule(rule: dict, held: pd.DataFrame, arm: str) -> dict:
    """Held-out fidelity, coverage, length; fidelity split by the network's own split."""
    col = f"class_pred_{arm}"
    mask = metrics.evaluate_rule_mask(held, rule["conditions"])
    cov = held.loc[mask]
    on_train = cov["net_split"] == "train"
    n = len(cov)
    return {
        "id": rule["id"], "length": int(rule["length"]), "predicted_class": rule["predicted_class"],
        "n_covered": int(n), "fidelity": _fid(rule["predicted_class"], cov[col]),
        "n_covered_net_train": int(on_train.sum()), "fidelity_net_train": _fid(rule["predicted_class"], cov.loc[on_train, col]),
        "n_covered_net_other": int((~on_train).sum()), "fidelity_net_other": _fid(rule["predicted_class"], cov.loc[~on_train, col]),
        "low_coverage_flag": bool(n < DESIGN["min_heldout_rows_flag"]),
        "train_fidelity": float(rule["fidelity_f1"]), "train_coverage_frac": float(rule["coverage"]),
    }


def score_pool(pool: list, held: pd.DataFrame, arm: str) -> list:
    return [score_rule(r, held, arm) for r in pool]


# ---------------------------------------------------------------- stability

def jaccard(a: set, b: set) -> float:
    """NaN if either set is empty (undefined for the purposes of this study)."""
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)


def kuncheva(a: set, b: set, n: int = DESIGN["kuncheva_n_features"]) -> float:
    """Kuncheva consistency index for two subsets of equal size k out of n features:
    (r*n - k^2) / (k*(n - k)), r = |A ∩ B|. NaN unless |A| == |B| = k with 0 < k < n."""
    if len(a) != len(b) or not a or len(a) >= n:
        return float("nan")
    k, r = len(a), len(a & b)
    return (r * n - k * k) / (k * (n - k))


def _nanmean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def pairwise_stats(sets: list) -> dict:
    """Mean pairwise Jaccard / Kuncheva over all unordered pairs, ignoring undefined pairs."""
    jac = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    kun = [kuncheva(a, b) for a, b in itertools.combinations(sets, 2)]
    return {"n_sets": len(sets), "n_empty_sets": sum(1 for s in sets if not s),
            "n_pairs": len(jac), "n_pairs_jaccard_defined": sum(1 for x in jac if not math.isnan(x)),
            "jaccard": _nanmean(jac),
            "n_pairs_kuncheva_defined": sum(1 for x in kun if not math.isnan(x)),
            "kuncheva": _nanmean(kun)}


def pool_gene_set(pool: list) -> set:
    return {g for r in pool for g in r["genes"]}


# ------------------------------------------------------- positive controls

def control_hit(drug: str, rules: list, feature_map: dict | None = None) -> dict:
    """Hit = a control feature appears with partition High in some rule of `rules`."""
    feats = POSITIVE_CONTROLS.get(drug)
    if feats is None:
        return {"is_control": False}
    hit = [r["id"] for r in rules
           if any(c["feature"] in feats and c["partition"] == "High" for c in r["conditions"])]
    return {"is_control": True, "features": feats, "hit": bool(hit), "hit_rule_ids": hit}
