"""Per-drug data: one drug's rows, the three arms' class_pred, cell-line CV folds.

Unit = one drug (<= 1 row per cell line). The drug's rows from ALL splits
(train/val/test of the networks) are kept; the original split is retained in the
`split` column (aliased `net_split`) so held-out scoring can report agreement on
rows the network trained on vs rows it did not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.per_drug.config import DESIGN, ROOT

REAL = ROOT / "data" / "processed" / "real_extraction"
NEW_PRED = ROOT / "artifacts" / "per_drug" / "data" / "predictions"
# arm -> (directory, file prefix) of {prefix}_{split}.parquet with columns row_id, class_pred
ARM_PRED = {
    "adam": (REAL / "predictions", "baseline"),
    "adam_wd": (NEW_PRED, "adam_wd"),
    "sam": (REAL / "predictions", "sam"),
}
SPLITS = ("train", "val", "test")


@dataclass
class DrugData:
    drug: str
    df: pd.DataFrame            # one row per cell line; cols incl. class_pred_<arm>, true_class, net_split
    feature_cols: list
    feature_map: dict
    fold_of: dict               # cell_name -> fold index


@dataclass
class FoldParts:
    fid: pd.DataFrame           # extraction rows used for fidelity (training folds minus inner val)
    inner_val: pd.DataFrame     # GA accuracy term only
    heldout: pd.DataFrame       # held-out fold, scoring only


def load_drug_frame(drug: str, data_dir: Path = REAL / "data", arm_pred: dict = ARM_PRED,
                    arms=None) -> DrugData:
    arms = list(arms or arm_pred)
    # row filter pushed into the parquet read: only this drug's rows are materialised (frame verified
    # identical to read-all-then-filter for 5 drugs; ~120 MB lower transient peak, 2-5x faster)
    df = pd.read_parquet(data_dir / "gdsc_processed.parquet", filters=[("drug_name", "==", drug)])
    df = df[df["drug_name"] == drug].reset_index(drop=True)          # right after the read
    if df.empty:
        raise ValueError(f"no rows for drug {drug!r}")
    if df["cell_name"].duplicated().any():
        raise ValueError(f"{drug}: more than one row per cell line")
    feature_map = json.loads((data_dir / "feature_map.json").read_text())
    feature_cols = list(feature_map)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns missing from parquet: {missing[:5]}")

    for arm in arms:
        d, prefix = arm_pred[arm]
        preds = pd.concat([pd.read_parquet(d / f"{prefix}_{s}.parquet")[["row_id", "class_pred"]] for s in SPLITS])
        preds = preds.rename(columns={"class_pred": f"class_pred_{arm}"})
        n0 = len(df)
        df = df.merge(preds, on="row_id", how="inner", validate="one_to_one")
        if len(df) != n0:
            raise ValueError(f"{drug}/{arm}: predictions missing for {n0 - len(df)} rows")
    df = df.assign(true_class=df["response_class"].astype(str), net_split=df["split"].astype(str))
    df = df.sort_values("cell_name").reset_index(drop=True)
    # feature columns -> plain object arrays: identical comparisons, much faster masks at ~800 rows
    feats = pd.DataFrame({c: df[c].astype(str).astype(object) for c in feature_cols}, index=df.index)
    df = pd.concat([df.drop(columns=feature_cols), feats], axis=1)[list(df.columns)]
    return DrugData(drug, df, feature_cols, feature_map,
                    assign_folds(df["cell_name"].tolist(), DESIGN["n_folds"], DESIGN["fold_seed"]))


def assign_folds(cell_names, n_folds: int, seed: int) -> dict:
    """Seeded, balanced assignment of cell lines to folds (sizes differ by <= 1)."""
    cells = sorted(set(cell_names))
    if len(cells) != len(cell_names):
        raise ValueError("duplicate cell lines")
    perm = np.random.RandomState(seed).permutation(len(cells))
    return {cells[i]: j % n_folds for j, i in enumerate(perm)}


def fold_parts(dd: DrugData, fold: int, inner_frac: float = DESIGN["inner_val_frac"],
               inner_seed: int = DESIGN["inner_seed"]) -> FoldParts:
    df = dd.df
    is_held = df["cell_name"].map(dd.fold_of).to_numpy() == fold
    held = df[is_held]
    train = df[~is_held]
    cells = sorted(train["cell_name"])
    perm = np.random.RandomState(inner_seed + fold).permutation(len(cells))
    n_val = int(round(inner_frac * len(cells)))
    val_cells = {cells[i] for i in perm[:n_val]}
    in_val = train["cell_name"].isin(val_cells).to_numpy()
    return FoldParts(fid=train[~in_val].reset_index(drop=True),
                     inner_val=train[in_val].reset_index(drop=True),
                     heldout=held.reset_index(drop=True))
