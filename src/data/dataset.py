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


class DeepCCDSDataset(Dataset):
    """Real three-branch DeepCCDS dataset, built from the authors' precomputed
    input files (see src/data/precomputed_inputs.py). Returns
    (pathway_activity, mutation, drug_fingerprint, ln_ic50) per sample.

    MEMORY NOTE: this is deliberately built as index lookups into three SMALL base
    matrices (878x38 pathway, 878x534 mutation, 413x1024 drug -- a few MB total),
    not as one big pre-expanded (319543, 1596) tensor. Expanding to a full merged
    tensor is what caused OOM crashes in earlier attempts (each of the 319,543 rows
    would duplicate whichever cell line / drug's rows it reuses many times over).
    Keeping the three small matrices in memory and indexing into them per __getitem__
    (or per-batch, via the default_collate stacking) keeps memory flat regardless of
    dataset size. Mutation/drug data is kept as int8 (binary values) and only cast to
    float32 per-batch in the training loop, not for the whole dataset up front.
    """

    def __init__(self, split: str, precomputed_dir: Path = Path("data/deepccds_precomputed")):
        from src.data.precomputed_inputs import (
            load_drug_fingerprints, load_mutation_matrix, load_pathway_activity, load_split,
        )

        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split!r}")

        self.pathway_activity = load_pathway_activity(precomputed_dir)   # (878, 38) float32
        self.mutation = load_mutation_matrix(precomputed_dir)            # (878, 534) int8
        self.drug_fingerprint = load_drug_fingerprints(precomputed_dir)  # (413, 1024) int8

        split_df = load_split(split, precomputed_dir)
        self.cell_idx = split_df["cell_idx"].to_numpy(dtype=np.int64)
        self.drug_idx = split_df["drug_idx"].to_numpy(dtype=np.int64)
        self.y = split_df["IC50"].to_numpy(dtype=np.float32)  # already LN_IC50, verified against real file
        self.split = split

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        c, d = self.cell_idx[idx], self.drug_idx[idx]
        pathway = torch.from_numpy(self.pathway_activity[c])                  # (38,) float32
        mutation = torch.from_numpy(self.mutation[c].astype(np.float32))      # (534,) float32, cast per-sample not for the whole dataset
        drug = torch.from_numpy(self.drug_fingerprint[d].astype(np.float32))  # (1024,) float32
        y = torch.tensor([self.y[idx]], dtype=torch.float32)
        return pathway, mutation, drug, y
