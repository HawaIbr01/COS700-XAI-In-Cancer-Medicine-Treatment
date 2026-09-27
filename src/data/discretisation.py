"""
Part 2 of the real-data preparation task: LN_IC50 -> sensitive/resistant
discretisation, gated behind DISCRETISATION_MODE (configs/data_prep.yaml).

This does NOT touch or reinterpret the locked z-score k=3 FEATURE
discretisation (High/Low/Intermediate pathway buckets, cfg.discretisation
in configs/pipeline.yaml, implemented in src/data/preprocess.py) -- that is
a different axis (antecedent features) from this module's concern (the
RESPONSE/target variable that rules are scored against). Nothing here
changes feature_map.json or the partition codes GA/NSGA-II already share.

Mirrors the existing convention in src/model/baseline.py's
_class_from_threshold / train_baseline(): a threshold is derived from
TRAIN-split true LN_IC50 only (never val/test, to avoid leakage), then
applied uniformly to true labels and model predictions across all splits.
The only thing this module changes is the GRANULARITY of that threshold
(one global value vs. one value per drug vs. a fixed literal), which is
exactly what DISCRETISATION_MODE controls.

See artifacts/data_prep/report.md Part 2 for the investigation and
recommendation behind the default ("per_drug_median_train").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VALID_MODES = {"global_median_train", "per_drug_median_train", "fixed_cutoff"}


def compute_thresholds(
    df: pd.DataFrame,
    mode: str,
    fixed_cutoff_ln_ic50: float | None = None,
    drug_col: str = "drug_name",
    split_col: str = "split",
    target_col: str = "ln_ic50",
) -> dict[str, float] | float:
    """Compute the discretisation threshold(s), from TRAIN-split rows only.

    Returns either a single float (global_median_train / fixed_cutoff) or a
    {drug_name: threshold} dict (per_drug_median_train).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"DISCRETISATION_MODE must be one of {VALID_MODES}, got {mode!r}")

    train = df.loc[df[split_col] == "train"]

    if mode == "fixed_cutoff":
        if fixed_cutoff_ln_ic50 is None:
            raise ValueError("mode='fixed_cutoff' requires fixed_cutoff_ln_ic50 to be set")
        return float(fixed_cutoff_ln_ic50)

    if mode == "global_median_train":
        return float(train[target_col].median())

    # per_drug_median_train
    per_drug = train.groupby(drug_col)[target_col].median()
    missing = set(df[drug_col].unique()) - set(per_drug.index)
    if missing:
        raise ValueError(
            f"{len(missing)} drug(s) have no TRAIN rows to compute a per-drug threshold from "
            f"(e.g. {sorted(missing)[:5]}) -- these must be excluded from the shortlist before "
            f"calling this function, since they cannot get a per-drug threshold."
        )
    return per_drug.to_dict()


def discretise(
    ln_ic50: np.ndarray | pd.Series,
    threshold: dict[str, float] | float,
    drug_name: np.ndarray | pd.Series | None,
    classes: list[str],
) -> np.ndarray:
    """Apply threshold(s) to a vector of LN_IC50 values (true or predicted).

    GDSC convention (also used by src/model/baseline.py's
    _class_from_threshold): lower LN_IC50 = more drug-sensitive, so
    ln_ic50 < threshold -> classes[0] ("sensitive"), else classes[1]
    ("resistant"). Ties (== threshold) go to "resistant", matching the
    existing `<` comparison in _class_from_threshold exactly.
    """
    sensitive, resistant = classes
    ln_ic50 = np.asarray(ln_ic50)

    if isinstance(threshold, dict):
        if drug_name is None:
            raise ValueError("a per-drug threshold dict requires drug_name to align rows to drugs")
        drug_name = np.asarray(drug_name)
        thresh_vec = np.array([threshold[d] for d in drug_name], dtype=float)
    else:
        thresh_vec = float(threshold)

    return np.where(ln_ic50 < thresh_vec, sensitive, resistant)
