"""
PyTorch Dataset for the processed GDSC data, loading the locked split produced by
preprocessing.py so every branch trains/evaluates on identical data (Section 4.2.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROCESSED_DIR = Path("data/processed")


class GDSCDataset(Dataset):
    """Wraps the processed GDSC parquet file for a given split ('train', 'val', 'test').

    Returns (gene_expression_tensor, ln_ic50_target) pairs. Gene expression is the
    continuous Z-scored representation -- the ANN (DeepCCDS baseline / SAM model)
    trains on this; the discretised `_bin` columns are only consumed by the GA /
    NSGA-II rule-extraction branches, not here.
    """

    def __init__(self, split: str, processed_dir: Path = PROCESSED_DIR):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split!r}")

        df = pd.read_parquet(processed_dir / "gdsc_processed.parquet")
        split_idx = np.load(processed_dir / "split_indices.npz")[f"{split}_idx"]
        with open(processed_dir / "feature_columns.json") as f:
            manifest = json.load(f)

        self.feature_cols = manifest["feature_cols"]
        self.df = df.iloc[split_idx].reset_index(drop=True)
        self.split = split

        self.X = torch.tensor(self.df[self.feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(self.df["LN_IC50"].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)
