"""
Integration test for src/data/preprocess.py, using tiny synthetic stand-ins
for the DeepCCDS precomputed CSVs (same column counts as the real files —
38 pathways / 534 genes / 1024 fingerprint bits — but only a handful of rows)
so this runs fast without needing the real ~12MB download.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data import preprocess as preprocess_module
from src.data.precomputed_inputs import N_PATHWAYS, N_MUTATION_GENES, N_DRUG_FINGERPRINT_BITS
from src.common.schema import validate_splits, validate_feature_map, validate_response_classes


@pytest.fixture()
def synthetic_precomputed_dir(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_cells, n_drugs = 6, 4

    pathway_cols = [f"pathway_{i}" for i in range(N_PATHWAYS)]
    ssgsea = pd.DataFrame(rng.normal(size=(n_cells, N_PATHWAYS)), columns=pathway_cols)
    ssgsea.insert(0, "cell_name", [f"CELL{i}" for i in range(n_cells)])
    ssgsea.to_csv(tmp_path / "GDSC_ssgsea_input.csv")

    gene_cols = [f"gene_{i}" for i in range(N_MUTATION_GENES)]
    mutation = pd.DataFrame(rng.integers(0, 2, size=(n_cells, N_MUTATION_GENES)), columns=gene_cols)
    mutation.insert(0, "cell_line", [f"CELL{i}" for i in range(n_cells)])
    mutation.to_csv(tmp_path / "GDSC_mutation_input.csv")

    bit_cols = [f"bit_{i}" for i in range(N_DRUG_FINGERPRINT_BITS)]
    drugs = pd.DataFrame(rng.integers(0, 2, size=(n_drugs, N_DRUG_FINGERPRINT_BITS)), columns=bit_cols)
    drugs.insert(0, "drug_name", [f"DRUG{i}" for i in range(n_drugs)])
    drugs.to_csv(tmp_path / "GDSC_SMILE_input.csv")

    # every cell-drug pair, split roughly 3 train / 1 val / rest test, disjoint pairs
    rows = [(c, d) for c in range(n_cells) for d in range(n_drugs)]
    rng.shuffle(rows)
    n = len(rows)
    split_points = {"train": rows[: n // 2], "val": rows[n // 2: n // 2 + n // 4], "test": rows[n // 2 + n // 4:]}
    for split, pairs in split_points.items():
        df = pd.DataFrame({
            "cell_name": [f"CELL{c}" for c, d in pairs],
            "cell_idx": [c for c, d in pairs],
            "drug_name": [f"DRUG{d}" for c, d in pairs],
            "drug_idx": [d for c, d in pairs],
            "IC50": rng.normal(size=len(pairs)),
        })
        filename = {"train": "GDSC_train_IC50_by_borh_cv00.csv",
                    "val": "GDSC_valid_IC50_by_borh_cv00.csv",
                    "test": "GDSC_test_IC50_by_borh_cv00.csv"}[split]
        df.to_csv(tmp_path / filename, index=False)

    return tmp_path


def test_preprocess_writes_valid_contract_artifacts(synthetic_precomputed_dir, tmp_path, monkeypatch):
    out_dir = tmp_path / "artifacts_out" / "data"
    monkeypatch.setattr(preprocess_module, "ARTIFACTS_DATA_DIR", out_dir)

    preprocess_module.preprocess(precomputed_dir=synthetic_precomputed_dir)

    assert (out_dir / "gdsc_processed.parquet").exists()
    assert (out_dir / "splits.json").exists()
    assert (out_dir / "feature_map.json").exists()
    assert (out_dir / "response_classes.json").exists()
    assert (out_dir / "pathway_activity_matrix.npy").exists()
    assert (out_dir / "mutation_matrix.npy").exists()
    assert (out_dir / "drug_fingerprint_matrix.npy").exists()

    df = pd.read_parquet(out_dir / "gdsc_processed.parquet")
    with open(out_dir / "splits.json") as f:
        splits = json.load(f)
    assert not validate_splits(splits, all_row_ids=set(df["row_id"].tolist()))

    with open(out_dir / "feature_map.json") as f:
        feature_map = json.load(f)
    assert not validate_feature_map(feature_map)
    assert len(feature_map) == N_PATHWAYS + N_MUTATION_GENES

    with open(out_dir / "response_classes.json") as f:
        response_classes = json.load(f)
    assert not validate_response_classes(response_classes)
    assert set(response_classes["per_row_class"].values()) <= {"sensitive", "resistant"}

    # discretised columns only ever take the three valid partition labels
    discretised_cols = [c for c in df.columns if c.endswith("_activity")]
    assert len(discretised_cols) == N_PATHWAYS
    for col in discretised_cols:
        assert set(df[col].unique()) <= {"High", "Low", "Intermediate"}

    # mutation columns: binary, so only High/Low -- never Intermediate
    mutation_cols = [c for c in df.columns if c.endswith("_mut")]
    assert len(mutation_cols) == N_MUTATION_GENES
    for col in mutation_cols:
        assert set(df[col].unique()) <= {"High", "Low"}
