"""
Loaders for the DeepCCDS authors' own precomputed input files
(https://github.com/hanjunwei-lab/DeepCCDS, `data/` folder — verified against
the real files):

    GDSC_ssgsea_input.csv     878 cell lines x 38 pathways        (+ cell_name col)
    GDSC_mutation_input.csv   878 cell lines x 534 genes          (+ cell_line col)
    GDSC_SMILE_input.csv      413 drugs x 1024-bit fingerprint    (+ drug_name col)
    GDSC_{train,valid,test}_IC50_by_borh_cv00.csv
                              (cell_name, cell_idx, drug_name, drug_idx, IC50) rows,
                              319,543 total (255,628 / 31,963 / 31,952) — matches the
                              paper's stated dataset size and "both split" strategy
                              exactly (cell lines AND drugs are non-overlapping across
                              train/val/test — the most stringent of the paper's four
                              splitting strategies).

    IC50 in these files is already LN_IC50 (verified range ~-9.8 to 12.7, matching
    the paper's Figure 3A).

WHY THESE INSTEAD OF DERIVING OUR OWN: building the real pathway-activity signal
from scratch means a curated PPI network, Random-Walk-with-Restart diffusion
seeded from 534 COSMIC CGC driver genes, then GSEA against 323 KEGG pathways to
select the 38 that matter (paper's "Characterizing Cancer Driver Signals through
a Prior Knowledge Network"). That is a substantial independent bioinformatics
pipeline. The DeepCCDS authors published their own exact, paper-verified versions
of pathway activity, the mutation matrix, and the drug fingerprints directly in
their repo. Every shape below has been checked against the numbers stated in
their Advanced Science paper and matches exactly.

**Flag to supervisor (kept from earlier scoping discussion):** using the
authors' precomputed features rather than independently re-deriving them from
raw GDSC files is a legitimate, but real, scoping decision — worth confirming
explicitly is acceptable for the proposal before results from this path are
reported as final. See README.md §5 ("Known limitations") for the full context.

cell_idx / drug_idx in the split files are POSITIONAL row indices into
GDSC_ssgsea_input.csv / GDSC_mutation_input.csv (same cell ordering) and
GDSC_SMILE_input.csv respectively — NOT SANGER_MODEL_ID or GDSC's own DRUG_ID.
Do not join these against raw GDSC1/GDSC2 files by ID; they're keyed by the
plain cell_name / drug_name strings used in the original DeepCCDS training run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PRECOMPUTED_DIR = Path(__file__).resolve().parents[2] / "data" / "deepccds_precomputed"

N_PATHWAYS = 38
N_MUTATION_GENES = 534
N_DRUG_FINGERPRINT_BITS = 1024


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the `data/` folder from "
            f"https://github.com/hanjunwei-lab/DeepCCDS and copy the six CSVs into "
            f"{path.parent} — see README.md §4 for the exact file list."
        )


def load_pathway_activity(precomputed_dir: Path = DEFAULT_PRECOMPUTED_DIR) -> tuple[np.ndarray, list[str], list[str]]:
    """Returns (activity, cell_names, pathway_names).
    activity: (878, 38) float32 array, row-aligned to cell_idx in the split files.
    """
    path = precomputed_dir / "GDSC_ssgsea_input.csv"
    _require(path)
    df = pd.read_csv(path, index_col=0)
    cell_col = "cell_name" if "cell_name" in df.columns else df.columns[0]
    pathway_cols = [c for c in df.columns if c != cell_col]
    if len(pathway_cols) != N_PATHWAYS:
        raise ValueError(f"Expected {N_PATHWAYS} pathway columns, got {len(pathway_cols)}")
    arr = df[pathway_cols].to_numpy(dtype=np.float32)
    cell_names = df[cell_col].astype(str).tolist() if cell_col in df.columns else df.index.astype(str).tolist()
    logger.info("Loaded pathway activity: %s", arr.shape)
    return arr, cell_names, pathway_cols


def load_mutation_matrix(precomputed_dir: Path = DEFAULT_PRECOMPUTED_DIR) -> tuple[np.ndarray, list[str], list[str]]:
    """Returns (mutation, cell_names, gene_names).
    mutation: (878, 534) int8 array (binary driver-gene mutation calls), row-aligned to cell_idx.
    """
    path = precomputed_dir / "GDSC_mutation_input.csv"
    _require(path)
    df = pd.read_csv(path, index_col=0)
    cell_col = "cell_line" if "cell_line" in df.columns else df.columns[0]
    gene_cols = [c for c in df.columns if c != cell_col]
    if len(gene_cols) != N_MUTATION_GENES:
        raise ValueError(f"Expected {N_MUTATION_GENES} gene columns, got {len(gene_cols)}")
    arr = df[gene_cols].to_numpy(dtype=np.int8)
    cell_names = df[cell_col].astype(str).tolist() if cell_col in df.columns else df.index.astype(str).tolist()
    logger.info("Loaded mutation matrix: %s", arr.shape)
    return arr, cell_names, gene_cols


def load_drug_fingerprints(precomputed_dir: Path = DEFAULT_PRECOMPUTED_DIR) -> np.ndarray:
    """Returns (413, 1024) int8 array (binary Morgan fingerprint bits), row-aligned to drug_idx."""
    path = precomputed_dir / "GDSC_SMILE_input.csv"
    _require(path)
    df = pd.read_csv(path, index_col=0)
    drug_col = "drug_name" if "drug_name" in df.columns else df.columns[0]
    bit_cols = [c for c in df.columns if c != drug_col]
    if len(bit_cols) != N_DRUG_FINGERPRINT_BITS:
        raise ValueError(f"Expected {N_DRUG_FINGERPRINT_BITS} fingerprint columns, got {len(bit_cols)}")
    arr = df[bit_cols].to_numpy(dtype=np.int8)
    logger.info("Loaded drug fingerprints: %s", arr.shape)
    return arr


def load_split(split: str, precomputed_dir: Path = DEFAULT_PRECOMPUTED_DIR) -> pd.DataFrame:
    """Returns the (cell_name, cell_idx, drug_name, drug_idx, IC50) rows for one split."""
    filename = {
        "train": "GDSC_train_IC50_by_borh_cv00.csv",
        "val": "GDSC_valid_IC50_by_borh_cv00.csv",
        "test": "GDSC_test_IC50_by_borh_cv00.csv",
    }.get(split)
    if filename is None:
        raise ValueError(f"split must be one of train/val/test, got {split!r}")
    path = precomputed_dir / filename
    _require(path)
    df = pd.read_csv(path)
    logger.info("Loaded %s split: %d rows", split, len(df))
    return df
