"""
GDSC preprocessing pipeline.

UPDATED against the real GDSC release 8.5 (27 Oct 2023) files -- confirmed the current
release as of this writing. Column names below were checked directly against
`GDSC2_fitted_dose_response_27Oct23.xlsx` (969 cell lines, 295 drugs, no missing/duplicate
keys, LN_IC50 already computed for you -- no log-transform step needed).

Three raw inputs, two of which are still needed on top of the response file:

  1. Drug response  (HAVE THIS):
     GDSC2_fitted_dose_response_27Oct23.xlsx
     Columns actually present: DATASET, NLME_RESULT_ID, NLME_CURVE_ID, CELL_LINE_NAME,
     SANGER_MODEL_ID, CANCER_TYPE, DRUG_ID, DRUG_NAME, PUTATIVE_TARGET, PATHWAY_NAME,
     MIN_CONC, MAX_CONC, LN_IC50, AUC, RMSE, Z_SCORE.
     Note LN_IC50 is already provided -- we do NOT recompute it from a raw IC50 column.

  2. Gene expression (STILL NEEDED):
     Cell_line_RMA_proc_basalExp.txt(.zip) from
     https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html
     This file is genes-in-rows, cell-lines-in-columns (the opposite orientation of
     what we want) with a `GENE_SYMBOLS` id column and `GENE_title` description column,
     followed by one column per cell line named `DATA.<COSMIC_ID>`. `load_expression()`
     below transposes this into the (n_cell_lines, n_genes) layout we need.

  3. Cell line ID mapping (STILL NEEDED):
     Cell_Lines_Details.xlsx from the bulk_download page.
     Needed because the expression file is keyed by COSMIC_ID, but the response file
     is keyed by SANGER_MODEL_ID / CELL_LINE_NAME -- these tables don't share a key
     directly. `load_cell_line_mapping()` builds the COSMIC_ID -> SANGER_MODEL_ID map.

  4. Drug chemical descriptors (per the DeepCCDS paper's own Experimental Section --
     confirmed against the paper, not a placeholder anymore): DeepCCDS does NOT use
     anything GDSC publishes natively for drug features. It queries PubChemPy for a
     SMILES string per drug, then converts SMILES to a 1024-bit Morgan fingerprint
     via the R package `rcdk`. GDSC's own PUTATIVE_TARGET / PATHWAY_NAME columns are
     NOT what the paper feeds to the drug autoencoder -- don't substitute them in.
     This is still a separate, not-yet-implemented step -- flagging as a real gap.

  5. Gene mutation data (per the paper, also NOT what's in the GDSC bulk-download
     files above): DeepCCDS uses a BINARY mutation matrix restricted to the 534
     COSMIC Cancer Gene Census (CGC) driver genes, built from GDSC's separate
     mutation annotation files (not the expression or response files). Also not yet
     wired up.

  6. Pathway mask / the "38 pathways" (per the paper's Experimental Section, this is
     a derived, non-trivial bioinformatics asset): built by (a) taking a curated
     human PPI network, (b) seeding Random-Walk-with-Restart from the 534 CGC driver
     genes to get a diffused gene ranking, (c) running GSEA of that ranking against
     323 KEGG pathways, (d) keeping the 38 pathways with ES > 0 and FDR < 0.2. This
     is a substantial pipeline in its own right. RECOMMENDATION: check the DeepCCDS
     GitHub (https://github.com/hanjunwei-lab/DeepCCDS) and the paper's Table S1
     (Supporting Information) for a precomputed gene->pathway mapping before
     re-deriving this from scratch -- it's very likely already published there.

Preprocessing steps implemented in THIS file (Section 4.2.2, plus the paper's own
drug-response pairing rule described in their Experimental Section):
  1. Merge GDSC1 + GDSC2 drug response, keeping the GDSC2 LN_IC50 value for any
     (cell line, drug) pair present in both -- this is the exact rule the DeepCCDS
     paper states it used, and how they got from raw pairs (517k+ across both
     releases) toward their final 319,543 (the rest of the reduction comes from
     requiring expression+mutation availability and cancer types with >=10 cell
     lines, applied later in the pipeline).
  2. Merge expression (via COSMIC_ID -> SANGER_MODEL_ID mapping) with drug response.
  3. Remove duplicate records and handle missing feature entries.
  4. One-hot encode categorical features (CANCER_TYPE).
  5. Z-score normalise continuous gene expression variables.
  6. Discretise Z-scores into {Low, Intermediate, High} expression bins for the
     downstream symbolic rule extraction pipeline (GA / NSGA-II branches).
  7. Emit a locked 80/10/10 train/val/test split, saved with a fixed seed so every
     branch (baseline, SAM, GA, NSGA-II, evaluation) trains/evaluates on identical data.
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

RAW_RESPONSE_GDSC1_PATH = Path("data/raw/GDSC1_fitted_dose_response_27Oct23.xlsx")
RAW_RESPONSE_GDSC2_PATH = Path("data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx")
RAW_EXPRESSION_PATH = Path("data/raw/Cell_line_RMA_proc_basalExp.txt")
RAW_CELL_LINE_DETAILS_PATH = Path("data/raw/Cell_Lines_Details.xlsx")

PROCESSED_DIR = Path("data/processed")

# Discretisation thresholds from Section 4.2.2 (biologically informed bins on Z-scores)
HIGH_EXPR_THRESHOLD = 1.0
LOW_EXPR_THRESHOLD = -1.0

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

GENE_COL_PREFIX = "GENE_"  # applied when we rename expression columns, e.g. "TP53" -> "GENE_TP53"


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _load_single_response_file(response_path: Path, dataset_label: str) -> pd.DataFrame:
    if not response_path.exists():
        raise FileNotFoundError(
            f"Missing drug response file at {response_path}. Download it from "
            f"https://www.cancerrxgene.org/downloads/bulk_download and place it there."
        )
    df = pd.read_excel(response_path)
    required = {"SANGER_MODEL_ID", "DRUG_ID", "CANCER_TYPE", "LN_IC50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_label} file is missing expected columns: {missing}")
    logger.info("Loaded %s: %s rows, %d unique cell lines, %d unique drugs",
                dataset_label, len(df), df["SANGER_MODEL_ID"].nunique(), df["DRUG_ID"].nunique())
    return df


def load_drug_response(
    gdsc1_path: Path = RAW_RESPONSE_GDSC1_PATH,
    gdsc2_path: Path = RAW_RESPONSE_GDSC2_PATH,
) -> pd.DataFrame:
    """Load and merge GDSC1 + GDSC2 fitted dose-response files, following the exact
    rule the DeepCCDS paper states it used: 'For cell-drug pairs duplicated in GDSC1
    and GDSC2, the LN IC50 value from the newer version (GDSC2) was retained.'
    (Wu et al. 2025, Experimental Section -- 'Preparing Data from GDSC').

    LN_IC50 is already computed by GDSC's curve-fitting pipeline (gdscIC50/nlme
    model) in both files -- we use it as-is rather than recomputing from a raw
    IC50 column, which neither file even contains.
    """
    gdsc1 = _load_single_response_file(gdsc1_path, "GDSC1")
    gdsc2 = _load_single_response_file(gdsc2_path, "GDSC2")

    key_cols = ["SANGER_MODEL_ID", "DRUG_ID"]
    gdsc1_only = gdsc1.merge(
        gdsc2[key_cols].drop_duplicates(), on=key_cols, how="left", indicator=True
    )
    gdsc1_only = gdsc1_only[gdsc1_only["_merge"] == "left_only"].drop(columns=["_merge"])

    combined = pd.concat([gdsc1_only, gdsc2], ignore_index=True)
    n_overlap = len(gdsc1) - len(gdsc1_only)
    logger.info(
        "Combined GDSC1+GDSC2: %d rows (%d overlapping pairs resolved in favour "
        "of GDSC2, per the DeepCCDS dedup rule)", len(combined), n_overlap,
    )
    return combined


def load_expression(expression_path: Path = RAW_EXPRESSION_PATH) -> pd.DataFrame:
    """Load and transpose Cell_line_RMA_proc_basalExp.txt.

    Raw layout: rows = genes, columns = [GENE_SYMBOLS, GENE_title, DATA.<COSMIC_ID>, ...].
    We transpose to rows = cell lines (indexed by COSMIC_ID), columns = genes
    (renamed to `GENE_<symbol>` so downstream code can select feature columns by prefix).
    """
    if not expression_path.exists():
        raise FileNotFoundError(
            f"Missing gene expression file at {expression_path}. Download "
            f"Cell_line_RMA_proc_basalExp.txt.zip from "
            f"https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html, "
            f"unzip it, and place the .txt file there."
        )
    raw = pd.read_csv(expression_path, sep="\t")
    if "GENE_SYMBOLS" not in raw.columns:
        raise ValueError(
            "Expected a 'GENE_SYMBOLS' column in the expression file -- got "
            f"{list(raw.columns)[:5]}... Check the file wasn't corrupted during download/unzip."
        )

    data_cols = [c for c in raw.columns if c.startswith("DATA.")]
    if not data_cols:
        raise ValueError("No 'DATA.<COSMIC_ID>' columns found in the expression file.")

    gene_symbols = raw["GENE_SYMBOLS"]
    expr = raw[data_cols].T  # transpose: now rows = cell lines, cols = genes
    expr.columns = [f"{GENE_COL_PREFIX}{g}" for g in gene_symbols]
    expr.index = [c.replace("DATA.", "") for c in data_cols]
    expr.index.name = "COSMIC_ID"
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.reset_index()
    expr["COSMIC_ID"] = expr["COSMIC_ID"].astype(str)

    logger.info("Loaded expression: %d cell lines, %d genes", expr.shape[0], expr.shape[1] - 1)
    return expr


def load_cell_line_mapping(details_path: Path = RAW_CELL_LINE_DETAILS_PATH) -> pd.DataFrame:
    """Load Cell_Lines_Details.xlsx and return a (COSMIC_ID, SANGER_MODEL_ID) map.

    Needed because the expression file is keyed by COSMIC_ID while the response
    file is keyed by SANGER_MODEL_ID -- there's no direct shared key otherwise.
    NOTE: exact column names in Cell_Lines_Details.xlsx have shifted across GDSC
    releases (e.g. "COSMIC identifier" vs "COSMIC_ID", "Sample Name" vs
    "SANGER_MODEL_ID"). If this raises a KeyError, open the file and tell me the
    actual header row/column names and I'll fix the mapping.
    """
    if not details_path.exists():
        raise FileNotFoundError(
            f"Missing cell line details file at {details_path}. Download "
            f"Cell_Lines_Details.xlsx from https://www.cancerrxgene.org/downloads/bulk_download."
        )
    df = pd.read_excel(details_path)
    cosmic_col = next((c for c in df.columns if "cosmic" in c.lower()), None)
    model_col = next((c for c in df.columns if "sanger" in c.lower() and "model" in c.lower()), None)
    if cosmic_col is None or model_col is None:
        raise ValueError(
            f"Could not auto-detect COSMIC/SANGER_MODEL_ID columns in {details_path}. "
            f"Available columns: {list(df.columns)}. Update load_cell_line_mapping() "
            f"with the exact column names."
        )
    mapping = df[[cosmic_col, model_col]].dropna().drop_duplicates()
    mapping.columns = ["COSMIC_ID", "SANGER_MODEL_ID"]
    mapping["COSMIC_ID"] = mapping["COSMIC_ID"].astype(int).astype(str)
    logger.info("Loaded cell line ID mapping: %d entries", len(mapping))
    return mapping


def merge_sub_datasets(
    expression: pd.DataFrame,
    response: pd.DataFrame,
    cell_line_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Merge expression (via COSMIC_ID -> SANGER_MODEL_ID) with drug response into
    a single cell-line x drug tabular dataset (one row per cell line-drug combination)."""
    expr_with_model_id = expression.merge(cell_line_mapping, on="COSMIC_ID", how="inner")
    merged = response.merge(expr_with_model_id, on="SANGER_MODEL_ID", how="inner")
    logger.info("Merged dataset shape before cleaning: %s", merged.shape)
    if len(merged) == 0:
        raise ValueError(
            "Merge produced 0 rows -- SANGER_MODEL_ID values likely don't overlap between "
            "the response file and the cell-line mapping. Double check both files are from "
            "the same GDSC release."
        )
    return merged


def remove_duplicates_and_handle_missing(
    df: pd.DataFrame,
    missing_col_thresh: float = 0.3,
) -> pd.DataFrame:
    """Drop duplicate cell-line/drug records, drop columns with excessive missingness,
    then impute remaining numeric gaps with the column median (robust to outliers,
    which are common in IC50 screening data)."""
    before = len(df)
    df = df.drop_duplicates(subset=["SANGER_MODEL_ID", "DRUG_ID"])
    logger.info("Dropped %d duplicate cell-line/drug rows", before - len(df))

    frac_missing = df.isna().mean()
    drop_cols = frac_missing[frac_missing > missing_col_thresh].index.tolist()
    if drop_cols:
        logger.info("Dropping %d columns with >%.0f%% missing values", len(drop_cols), missing_col_thresh * 100)
        df = df.drop(columns=drop_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def one_hot_encode_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical metadata columns (e.g. CANCER_TYPE)."""
    present = [c for c in categorical_cols if c in df.columns]
    if not present:
        logger.warning("None of the requested categorical columns %s were found", categorical_cols)
        return df
    return pd.get_dummies(df, columns=present, dtype=float)


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


def run_pipeline(categorical_cols: list[str] | None = None) -> pd.DataFrame:
    """End-to-end preprocessing entry point. Run this once; every branch consumes
    the artefacts it writes to data/processed/."""
    categorical_cols = categorical_cols or ["CANCER_TYPE"]

    response = load_drug_response()  # now merges GDSC1 + GDSC2, per the DeepCCDS paper's rule
    expression = load_expression()
    cell_line_mapping = load_cell_line_mapping()

    merged = merge_sub_datasets(expression, response, cell_line_mapping)
    cleaned = remove_duplicates_and_handle_missing(merged)
    encoded = one_hot_encode_categoricals(cleaned, categorical_cols)

    gene_cols = [c for c in encoded.columns if c.startswith(GENE_COL_PREFIX)]
    if not gene_cols:
        raise ValueError(
            f"No gene expression columns found (expected columns prefixed '{GENE_COL_PREFIX}'). "
            "Check load_expression() ran correctly."
        )

    zscored, means, stds = zscore_normalise(encoded, gene_cols)
    binned = discretise_expression(zscored, gene_cols)

    splits = make_fixed_split(len(binned))
    save_processed(binned, splits, gene_cols)

    means.to_csv(PROCESSED_DIR / "gene_means.csv")
    stds.to_csv(PROCESSED_DIR / "gene_stds.csv")

    logger.info("Preprocessing complete: %d rows, %d gene features, target=LN_IC50", len(binned), len(gene_cols))
    return binned


if __name__ == "__main__":
    run_pipeline()
