"""
PyTorch Dataset wrapping the contract artifacts produced by preprocess.py
(docs/CONTRACTS.md §0.1), for use by both baseline.py (Adam) and sam.py (SAM)
so both train on the byte-identical data (§4.5.1).

Returns (pathway_activity, mutation, drug, ln_ic50) per sample — the RAW,
non-Z-scored ssGSEA pathway activity (the paper feeds ssGSEA output directly
into the network; the Z-scored/discretised columns in gdsc_processed.parquet
are for rule extraction only, not model input).

MEMORY NOTE: built as index lookups into three small cached matrices
(pathway_activity_matrix.npy, mutation_matrix.npy, drug_fingerprint_matrix.npy
— a few MB total) rather than expanding everything into one big
(319543, ~1600) tensor, which caused OOM crashes in earlier testing. See
preprocess.py's module docstring for the full explanation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ARTIFACTS_DATA_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "data"


class DeepCCDSDataset(Dataset):
    """One split ('train' | 'val' | 'test') of the DeepCCDS 3-branch dataset,
    built from artifacts/data/* (must run `python -m src.data.preprocess` first).
    """

    def __init__(self, split: str, artifacts_data_dir: Path = ARTIFACTS_DATA_DIR):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split!r}")

        parquet_path = artifacts_data_dir / "gdsc_processed.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Missing {parquet_path}. Run `python -m src.data.preprocess` first "
                f"(see README.md)."
            )

        df = pd.read_parquet(parquet_path, columns=["row_id", "split", "cell_idx", "drug_idx", "ln_ic50"])
        df = df.loc[df["split"] == split].reset_index(drop=True)

        self.row_ids = df["row_id"].to_numpy(dtype=np.int64)
        self.cell_idx = df["cell_idx"].to_numpy(dtype=np.int64)
        self.drug_idx = df["drug_idx"].to_numpy(dtype=np.int64)
        self.y = df["ln_ic50"].to_numpy(dtype=np.float32)
        self.split = split

        self.pathway_activity = np.load(artifacts_data_dir / "pathway_activity_matrix.npy")  # (878, 38) float32
        self.mutation = np.load(artifacts_data_dir / "mutation_matrix.npy")                  # (878, 534) int8
        self.drug_fingerprint = np.load(artifacts_data_dir / "drug_fingerprint_matrix.npy")  # (413, 1024) int8

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        c, d = self.cell_idx[idx], self.drug_idx[idx]
        pathway = torch.from_numpy(self.pathway_activity[c].copy())                   # (38,) float32
        mutation = torch.from_numpy(self.mutation[c].astype(np.float32))              # (534,) float32
        drug = torch.from_numpy(self.drug_fingerprint[d].astype(np.float32))          # (1024,) float32
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        row_id = int(self.row_ids[idx])
        return row_id, pathway, mutation, drug, y
