"""
Unit tests for the real (non-placeholder) data path: precomputed_inputs.py loaders
and DeepCCDSDataset. Uses small synthetic CSVs matching the DeepCCDS authors' real
file structure (verified column counts/names against the actual files), not the
full real data, so these run fast and deterministically in CI.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.precomputed_inputs import (
    N_DRUG_FINGERPRINT_BITS, N_MUTATION_GENES, N_PATHWAYS,
    load_drug_fingerprints, load_mutation_matrix, load_pathway_activity, load_split,
)


@pytest.fixture
def synthetic_precomputed_dir(tmp_path: Path) -> Path:
    n_cells, n_drugs = 5, 4
    rng = np.random.default_rng(0)

    ssgsea = pd.DataFrame(
        rng.normal(size=(n_cells, N_PATHWAYS)),
        columns=[f"pathway_{i}" for i in range(N_PATHWAYS)],
    )
    ssgsea.insert(0, "cell_name", [f"CELL{i}" for i in range(n_cells)])
    ssgsea.to_csv(tmp_path / "GDSC_ssgsea_input.csv")

    mutation = pd.DataFrame(
        rng.integers(0, 2, size=(n_cells, N_MUTATION_GENES)),
        columns=[f"GENE{i}" for i in range(N_MUTATION_GENES)],
    )
    mutation.insert(0, "cell_line", [f"CELL{i}" for i in range(n_cells)])
    mutation.to_csv(tmp_path / "GDSC_mutation_input.csv")

    smile = pd.DataFrame(
        rng.integers(0, 2, size=(n_drugs, N_DRUG_FINGERPRINT_BITS)),
        columns=[str(i) for i in range(N_DRUG_FINGERPRINT_BITS)],
    )
    smile.insert(0, "drug_name", [f"DRUG{i}" for i in range(n_drugs)])
    smile.to_csv(tmp_path / "GDSC_SMILE_input.csv")

    for split, fname in [
        ("train", "GDSC_train_IC50_by_borh_cv00.csv"),
        ("val", "GDSC_valid_IC50_by_borh_cv00.csv"),
        ("test", "GDSC_test_IC50_by_borh_cv00.csv"),
    ]:
        split_df = pd.DataFrame({
            "cell_name": ["CELL0", "CELL1"],
            "cell_idx": [0, 1],
            "drug_name": ["DRUG0", "DRUG1"],
            "drug_idx": [0, 1],
            "IC50": [1.5, -2.3],
        })
        split_df.to_csv(tmp_path / fname, index=False)

    return tmp_path


@pytest.fixture
def synthetic_artifacts_data_dir(tmp_path: Path) -> Path:
    """Minimal artifacts/data/-shaped directory (post-preprocess.py layout),
    matching what DeepCCDSDataset actually reads: gdsc_processed.parquet +
    the three cached .npy matrices — not the raw precomputed-input CSVs
    (those are covered by the synthetic_precomputed_dir fixture above and
    src/data/preprocess.py's own tests)."""
    rng = np.random.default_rng(0)
    n_cells, n_drugs = 2, 2

    pd.DataFrame({
        "row_id": [0, 1],
        "split": ["train", "train"],
        "cell_idx": [0, 1],
        "drug_idx": [0, 1],
        "ln_ic50": [1.5, -2.3],
    }).to_parquet(tmp_path / "gdsc_processed.parquet")

    np.save(tmp_path / "pathway_activity_matrix.npy",
            rng.normal(size=(n_cells, N_PATHWAYS)).astype(np.float32))
    np.save(tmp_path / "mutation_matrix.npy",
            rng.integers(0, 2, size=(n_cells, N_MUTATION_GENES)).astype(np.int8))
    np.save(tmp_path / "drug_fingerprint_matrix.npy",
            rng.integers(0, 2, size=(n_drugs, N_DRUG_FINGERPRINT_BITS)).astype(np.int8))

    return tmp_path


def test_load_pathway_activity_shape(synthetic_precomputed_dir):
    arr, cell_names, pathway_cols = load_pathway_activity(synthetic_precomputed_dir)
    assert arr.shape == (5, N_PATHWAYS)
    assert arr.dtype == np.float32
    assert len(cell_names) == 5
    assert len(pathway_cols) == N_PATHWAYS


def test_load_mutation_matrix_shape_and_dtype(synthetic_precomputed_dir):
    arr, cell_names, gene_names = load_mutation_matrix(synthetic_precomputed_dir)
    assert arr.shape == (5, N_MUTATION_GENES)
    assert arr.dtype == np.int8
    assert len(cell_names) == 5
    assert len(gene_names) == N_MUTATION_GENES


def test_load_drug_fingerprints_shape(synthetic_precomputed_dir):
    arr = load_drug_fingerprints(synthetic_precomputed_dir)
    assert arr.shape == (4, N_DRUG_FINGERPRINT_BITS)
    assert arr.dtype == np.int8


def test_load_split_returns_expected_columns(synthetic_precomputed_dir):
    df = load_split("train", synthetic_precomputed_dir)
    assert list(df.columns) == ["cell_name", "cell_idx", "drug_name", "drug_idx", "IC50"]
    assert len(df) == 2


def test_load_split_rejects_bad_split_name(synthetic_precomputed_dir):
    with pytest.raises(ValueError):
        load_split("bogus", synthetic_precomputed_dir)


def test_deepccds_dataset_end_to_end(synthetic_artifacts_data_dir):
    from src.data.dataset import DeepCCDSDataset

    ds = DeepCCDSDataset("train", artifacts_data_dir=synthetic_artifacts_data_dir)
    assert len(ds) == 2

    row_id, pathway, mutation, drug, y = ds[0]
    assert pathway.shape == (N_PATHWAYS,)
    assert mutation.shape == (N_MUTATION_GENES,)
    assert drug.shape == (N_DRUG_FINGERPRINT_BITS,)
    assert y.shape == ()
    assert abs(y.item() - 1.5) < 1e-5


def test_deepccds_dataset_rejects_bad_split(synthetic_artifacts_data_dir):
    from src.data.dataset import DeepCCDSDataset

    with pytest.raises(ValueError):
        DeepCCDSDataset("bogus", artifacts_data_dir=synthetic_artifacts_data_dir)
