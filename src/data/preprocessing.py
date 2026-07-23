"""
GDSC preprocessing pipeline.

Implements the preprocessing interface described in Section 4.2.2 of the proposal:
  1. Merge the three GDSC sub-datasets (cell line genomic features, drug chemical
     descriptors, drug response screening measurements).
  2. Remove duplicate records and handle missing feature entries.
  3. One-hot encode categorical features.
  4. Z-score normalise continuous gene expression variables.
  5. Discretise Z-scores into {Low, Intermediate, High} expression bins for the
     downstream symbolic rule extraction pipeline (GA / NSGA-II branches).
  6. Produce the LN(IC50) regression target.
  7. Emit a locked 80/10/10 train/val/test split, saved with a fixed seed so every
     branch (baseline, SAM, GA, NSGA-II, evaluation) trains/evaluates on identical data.

This follows the open-source GDSC preprocessing approach referenced in the proposal
(Alipour, 2024) as a starting point, adapted to our column layout.

NOTE FOR THE TEAM: this assumes the three raw GDSC files sit in `data/raw/`:
    - data/raw/gdsc_expression.csv      (cell line x gene expression matrix)
    - data/raw/gdsc_drug_response.csv   (cell line, drug, IC50)
    - data/raw/gdsc_drug_descriptors.csv (drug chemical descriptors)
Adjust RAW_* paths below once we confirm the actual GDSC export filenames/columns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42  # locked across the whole pipeline -- do not change without team agreement

RAW_EXPRESSION_PATH = Path("data/raw/gdsc_expression.csv")
RAW_RESPONSE_PATH = Path("data/raw/gdsc_drug_response.csv")
RAW_DRUG_DESC_PATH = Path("data/raw/gdsc_drug_descriptors.csv")

PROCESSED_DIR = Path("data/processed")

# Discretisation thresholds from Section 4.2.2 (biologically informed bins on Z-scores)
HIGH_EXPR_THRESHOLD = 1.0
LOW_EXPR_THRESHOLD = -1.0

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def load_raw_tables(
    expression_path: Path = RAW_EXPRESSION_PATH,
    response_path: Path = RAW_RESPONSE_PATH,
    drug_desc_path: Path = RAW_DRUG_DESC_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw GDSC sub-datasets. Raises a clear error if files are missing
    rather than failing on a cryptic pandas error, since this is usually the first
    thing that breaks when someone clones the repo without the data."""
    missing = [p for p in (expression_path, response_path, drug_desc_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw GDSC file(s): {missing}. Place the GDSC exports under "
            f"data/raw/ (see module docstring) before running preprocessing."
        )
    expression = pd.read_csv(expression_path)
    response = pd.read_csv(response_path)
    drug_desc = pd.read_csv(drug_desc_path)
    logger.info(
        "Loaded raw tables: expression=%s response=%s drug_desc=%s",
        expression.shape, response.shape, drug_desc.shape,
    )
    return expression, response, drug_desc


def merge_sub_datasets(
    expression: pd.DataFrame,
    response: pd.DataFrame,
    drug_desc: pd.DataFrame,
    cell_line_key: str = "COSMIC_ID",
    drug_key: str = "DRUG_ID",
) -> pd.DataFrame:
    """Merge expression, response and drug-descriptor tables into a single
    cell-line x drug tabular dataset (one row per cell line-drug combination)."""
    merged = response.merge(expression, on=cell_line_key, how="inner")
    merged = merged.merge(drug_desc, on=drug_key, how="inner")
    logger.info("Merged dataset shape before cleaning: %s", merged.shape)
    return merged


def remove_duplicates_and_handle_missing(
    df: pd.DataFrame,
    missing_col_thresh: float = 0.3,
) -> pd.DataFrame:
    """Drop duplicate cell-line/drug records, drop columns with excessive missingness,
    then impute remaining numeric gaps with the column median (robust to outliers,
    which are common in IC50 screening data)."""
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", before - len(df))

    frac_missing = df.isna().mean()
    drop_cols = frac_missing[frac_missing > missing_col_thresh].index.tolist()
    if drop_cols:
        logger.info("Dropping %d columns with >%.0f%% missing values", len(drop_cols), missing_col_thresh * 100)
        df = df.drop(columns=drop_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def one_hot_encode_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical metadata columns (e.g. tissue type, cancer subtype)."""
    present = [c for c in categorical_cols if c in df.columns]
    if not present:
        logger.warning("None of the requested categorical columns %s were found", categorical_cols)
        return df
    return pd.get_dummies(df, columns=present, dtype=float)


def compute_ln_ic50(df: pd.DataFrame, ic50_col: str = "IC50") -> pd.DataFrame:
    """Add the LN(IC50) regression target used throughout the study (Section 3.2.1)."""
    if (df[ic50_col] <= 0).any():
        n_bad = (df[ic50_col] <= 0).sum()
        logger.warning("%d rows have non-positive IC50; dropping before log transform", n_bad)
        df = df[df[ic50_col] > 0].copy()
    df["LN_IC50"] = np.log(df[ic50_col])
    return df


def zscore_normalise(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score normalise continuous gene expression features across all samples.
    Returns the transformed frame plus the per-column mean/std so the same
    transform can be re-applied to held-out data (e.g. CCLE, Section 4.7.1)."""
    means = df[feature_cols].mean()
    stds = df[feature_cols].std().replace(0, 1.0)  # guard against constant columns
    df_z = df.copy()
    df_z[feature_cols] = (df[feature_cols] - means) / stds
    return df_z, means, stds


def discretise_expression(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Bin Z-scored expression into {Low, Intermediate, High} categories for the
    downstream GA / NSGA-II rule-extraction pipelines (Section 4.2.2, Section 4.3.2).
    Adds new `<gene>_bin` columns rather than overwriting the continuous features,
    since the ANN trains on the continuous values and only rule extraction needs bins.
    """
    df_binned = df.copy()
    for col in feature_cols:
        bins = pd.cut(
            df[col],
            bins=[-np.inf, LOW_EXPR_THRESHOLD, HIGH_EXPR_THRESHOLD, np.inf],
            labels=["Low", "Intermediate", "High"],
        )
        df_binned[f"{col}_bin"] = bins
    return df_binned


def make_fixed_split(n_rows: int, seed: int = RANDOM_SEED) -> SplitIndices:
    """Locked 80/10/10 train/val/test split (Section 4.2.2). This is the single
    source of truth for the split -- every branch must call this function (or load
    the saved index files it produces) rather than re-splitting independently,
    otherwise rule-quality metrics across branches are not comparable."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_rows)
    n_train = int(n_rows * TRAIN_FRAC)
    n_val = int(n_rows * VAL_FRAC)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return SplitIndices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def save_processed(df: pd.DataFrame, splits: SplitIndices, feature_cols: list[str], out_dir: Path = PROCESSED_DIR) -> None:
    """Persist the processed dataframe, the split indices, and the feature-column
    manifest so downstream branches (SAM, GA, NSGA-II, evaluation) can load an
    identical, reproducible dataset without re-running preprocessing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "gdsc_processed.parquet")
    np.savez(
        out_dir / "split_indices.npz",
        train_idx=splits.train_idx, val_idx=splits.val_idx, test_idx=splits.test_idx,
    )
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "seed": RANDOM_SEED}, f, indent=2)
    logger.info("Saved processed dataset + split indices to %s", out_dir)


def run_pipeline(
    categorical_cols: list[str] | None = None,
    ic50_col: str = "IC50",
) -> pd.DataFrame:
    """End-to-end preprocessing entry point. Run this once; every branch consumes
    the artefacts it writes to data/processed/."""
    categorical_cols = categorical_cols or ["TISSUE_TYPE", "CANCER_SUBTYPE"]

    expression, response, drug_desc = load_raw_tables()
    merged = merge_sub_datasets(expression, response, drug_desc)
    cleaned = remove_duplicates_and_handle_missing(merged)
    encoded = one_hot_encode_categoricals(cleaned, categorical_cols)
    with_target = compute_ln_ic50(encoded, ic50_col=ic50_col)

    gene_cols = [c for c in with_target.columns if c.startswith("GENE_")]
    if not gene_cols:
        raise ValueError(
            "No gene expression columns found (expected columns prefixed 'GENE_'). "
            "Update the prefix to match the actual GDSC export column naming."
        )

    zscored, means, stds = zscore_normalise(with_target, gene_cols)
    binned = discretise_expression(zscored, gene_cols)

    splits = make_fixed_split(len(binned))
    save_processed(binned, splits, gene_cols)

    means.to_csv(PROCESSED_DIR / "gene_means.csv")
    stds.to_csv(PROCESSED_DIR / "gene_stds.csv")

    logger.info("Preprocessing complete: %d rows, %d gene features, target=LN_IC50", len(binned), len(gene_cols))
    return binned


if __name__ == "__main__":
    run_pipeline()
