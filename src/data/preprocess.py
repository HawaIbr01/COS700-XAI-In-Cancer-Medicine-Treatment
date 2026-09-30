"""
GDSC preprocessing: build the four shared data artifacts every other arm
depends on (docs/CONTRACTS.md §0.1, Arm 1).

OWNER: Angelique (feat/baseline-model)
Methodology: proposal §4.2.2.

Produces (read-only for every other arm from this point on):
    artifacts/data/gdsc_processed.parquet
    artifacts/data/splits.json
    artifacts/data/feature_map.json
    artifacts/data/response_classes.json

...plus three small cached matrices (NOT part of the official §0.1 schema, but
harmless additions used by baseline.py / sam.py to avoid re-reading the raw
CSVs every run and to keep memory flat — see the "why not one big merged
table" note below):
    artifacts/data/pathway_activity_matrix.npy   (878, 38)  float32
    artifacts/data/mutation_matrix.npy           (878, 534) int8
    artifacts/data/drug_fingerprint_matrix.npy   (413, 1024) int8

===============================================================================
IMPORTANT — READ BEFORE USING THIS FOR GSEA (flag for Hawa's evaluation arm):
===============================================================================
This pipeline uses the DeepCCDS authors' precomputed data (see
src/data/precomputed_inputs.py for the full justification). That means the
*model's* input features are real: 38 ssGSEA pathway-activity scores, a real
534-driver-gene mutation matrix, and real 1024-bit drug fingerprints — all
paper-verified.

BUT the discretised feature set this file builds for rule extraction
(§0.4 rule-pool schema's `conditions` / `feature_map.json`'s `gene_symbol`) is
built from those same 38 *pathway* activity scores, Z-scored and binned into
High/Low/Intermediate — NOT from individual per-gene expression. There is no
raw RNA-seq expression file wired up yet (that path needs the "RNA-Seq Summary"
download from Cell Model Passports — see README.md §5). So:

  - Each entry in feature_map.json is keyed by a KEGG PATHWAY name (e.g.
    "MAPK_signaling_pathway_activity"), not a single gene.
  - `gene_symbol` for each entry is currently set to the pathway name itself
    as a placeholder, NOT a real HGNC gene symbol. GSEA over-representation
    (§4.6.3) against KEGG/Enrichr needs real gene symbols to be meaningful —
    passing pathway names into that step will not produce a valid enrichment
    test. Do not treat rules['genes'] from this data as biologically real
    until this is resolved.
  - Two ways to close this gap, neither of which I've done here because doing
    it without the real source data would mean inventing gene-pathway
    associations: (1) pull the DeepCCDS paper's Supplementary Table S1, which
    lists the actual leading-edge gene set per pathway, and use that as the
    feature -> genes mapping; or (2) finish the raw-GDSC RNA-seq path in this
    same module once that file is available, giving genuine per-gene features.

This is the same limitation flagged in earlier team discussion — repeating it
here loudly because it is easy to silently lose when code moves between
branches.
===============================================================================

Design note — why the discretised features live in the row-level parquet
rather than a smaller per-cell-line table: the rule-pool schema evaluates a
rule's conditions with a boolean mask over `gdsc_processed.parquet` directly
(src/common/metrics.py: evaluate_rule_mask), so every cell-drug row needs its
own copy of that cell line's discretised pathway columns. This duplicates the
38 category values across every drug the cell line appears against, which is
intentional and matches how rule extraction is expected to consume the data.

Design note — why the three cached matrices exist separately from the
parquet: the *continuous* pathway/mutation/drug tensors DeepCCDS actually
trains on are small in their un-expanded form (878x38, 878x534, 413x1024 —
a few MB total). Expanding them into one big (319543, ~1600) merged tensor
is what caused an out-of-memory crash in earlier local testing (every one of
the 319,543 rows would duplicate whichever cell line's / drug's features it
reuses many times over). Keeping the three small matrices as .npy caches and
indexing into them per-row (via cell_idx/drug_idx, both preserved in the
parquet) keeps memory flat regardless of dataset size. See src/data/dataset.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.schema import (
    validate_splits,
    validate_feature_map,
    validate_response_classes,
)
from src.data.precomputed_inputs import (
    DEFAULT_PRECOMPUTED_DIR,
    load_drug_fingerprints,
    load_mutation_matrix,
    load_pathway_activity,
    load_split,
)

ARTIFACTS_DATA_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "data"


def _sanitise_column(name: str) -> str:
    """KEGG pathway names contain spaces/slashes/parens — make them safe
    parquet/JSON column names while staying human-readable."""
    return (
        name.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        + "_activity"
    )


def _discretise(z: np.ndarray, high_z: float, low_z: float) -> np.ndarray:
    out = np.full(z.shape, "Intermediate", dtype=object)
    out[z > high_z] = "High"
    out[z < low_z] = "Low"
    return out


def preprocess(precomputed_dir: Path = DEFAULT_PRECOMPUTED_DIR) -> None:
    cfg = load_config()

    # ---- 1. load the DeepCCDS authors' real precomputed inputs -----------
    pathway_activity, cell_names, pathway_names = load_pathway_activity(precomputed_dir)
    mutation_matrix, mutation_cell_names, mutation_gene_names = load_mutation_matrix(precomputed_dir)
    drug_fingerprints = load_drug_fingerprints(precomputed_dir)

    splits_raw = {s: load_split(s, precomputed_dir) for s in ("train", "val", "test")}

    # ---- 2. build the row-level long table (one row per cell-drug pair) --
    frames = []
    for split_name, df in splits_raw.items():
        d = df.copy()
        d["split"] = split_name
        frames.append(d)
    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.rename(columns={"IC50": "ln_ic50"})
    long_df.insert(0, "row_id", np.arange(len(long_df), dtype=np.int64))

    # ---- 3. Z-score pathway activity across the full cell-line population
    #         (§4.2.2: "Z-score normalised across all samples", done prior to
    #         the split, mirroring the paper's own preprocessing order) and
    #         discretise for rule extraction. Model input keeps the RAW
    #         (non-Z-scored) ssGSEA activity — see dataset.py — since the
    #         paper feeds ssGSEA output directly into the network.
    high_z = cfg.discretisation.high_z
    low_z = cfg.discretisation.low_z
    pathway_mean = pathway_activity.mean(axis=0)
    pathway_std = pathway_activity.std(axis=0)
    pathway_std[pathway_std == 0] = 1.0  # guard against a degenerate constant pathway
    pathway_z = (pathway_activity - pathway_mean) / pathway_std  # (878, 38)

    feature_cols = [_sanitise_column(p) for p in pathway_names]
    discretised = {
        col: _discretise(pathway_z[:, j], high_z, low_z) for j, col in enumerate(feature_cols)
    }
    # discretised[col] is (878,) aligned to cell_idx -> broadcast onto every
    # row that references that cell_idx. Stored as `category` dtype, not
    # object/str: at 572 discretised columns x 319,543 rows, object dtype
    # (a full Python string per cell) pushed gdsc_processed.parquet's
    # in-memory footprint past this project's 3.7GB WSL2 ceiling and got the
    # real-extraction prep step OOM-killed (each column only ever holds 2-3
    # distinct values, so this is a pure dtype fix -- rule_mask's `==`
    # comparisons work identically against a categorical column).
    cell_idx_per_row = long_df["cell_idx"].to_numpy()
    for col in feature_cols:
        long_df[col] = pd.Categorical(
            discretised[col][cell_idx_per_row], categories=["Low", "Intermediate", "High"]
        )

    # ---- 3b. mutation status: already binary (0/1), no Z-score/threshold
    #          needed -- straight High/Low on the raw calls. Real HGNC gene
    #          symbols (mutation_gene_names), suffixed _mut to avoid
    #          colliding with the pathway _activity columns above. Same
    #          cell_idx_per_row broadcast as the pathway block, same
    #          category-dtype reasoning as above.
    mutation_feature_cols = [f"{g.strip()}_mut" for g in mutation_gene_names]
    mutation_labels = np.where(mutation_matrix == 1, "High", "Low")  # (878, 534)
    for j, col in enumerate(mutation_feature_cols):
        long_df[col] = pd.Categorical(mutation_labels[cell_idx_per_row, j], categories=["Low", "High"])

    # ---- 4. response-class binning (§4.2.2 / §4.6.1: continuous ln_ic50 ->
    #         sensitive/resistant for rule-fidelity classification). Threshold
    #         fixed from the TRAIN split only (cfg.response_binning.threshold,
    #         currently "median" — provisional per pipeline.yaml, see
    #         docs/CONTRACTS.md §0.2). Lower ln_ic50 = more drug-sensitive
    #         (paper: "a lower IC50 indicates a more sensitive cell line").
    train_ln_ic50 = long_df.loc[long_df["split"] == "train", "ln_ic50"]
    if cfg.response_binning.threshold == "median":
        threshold = float(train_ln_ic50.median())
    else:
        threshold = float(cfg.response_binning.threshold)

    sensitive_label, resistant_label = cfg.response_binning.classes
    long_df["response_class"] = np.where(
        long_df["ln_ic50"] < threshold, sensitive_label, resistant_label
    )

    # ---- 5. write artifacts, validating every one before writing ---------
    ARTIFACTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    long_df.to_parquet(ARTIFACTS_DATA_DIR / "gdsc_processed.parquet")

    splits = {
        "train": long_df.loc[long_df["split"] == "train", "row_id"].tolist(),
        "val": long_df.loc[long_df["split"] == "val", "row_id"].tolist(),
        "test": long_df.loc[long_df["split"] == "test", "row_id"].tolist(),
    }
    split_errors = validate_splits(splits, all_row_ids=set(long_df["row_id"].tolist()))
    if split_errors:
        raise ValueError(f"splits.json failed validation: {split_errors}")
    with open(ARTIFACTS_DATA_DIR / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f)

    feature_map = {
        col: {
            "gene_symbol": pathway_names[j],
            "note": (
                "PLACEHOLDER: this is a KEGG pathway name, not a single HGNC gene "
                "symbol -- no per-gene expression is wired up yet. See the module "
                "docstring in this file and README.md Sec 5 before using this "
                "for GSEA (Sec 4.6.3)."
            ),
            "partition_edges": [low_z, high_z],
        }
        for j, col in enumerate(feature_cols)
    }
    feature_map.update({
        col: {
            "gene_symbol": gene,
            "note": (
                "Real HGNC gene symbol (binary driver-gene mutation status from "
                "GDSC_mutation_input.csv) -- unlike the pathway-activity entries "
                "above, gene_symbol here is not a placeholder."
            ),
            "partition_edges": None,
        }
        for gene, col in zip(mutation_gene_names, mutation_feature_cols)
    })
    fm_errors = validate_feature_map(feature_map)
    if fm_errors:
        raise ValueError(f"feature_map.json failed validation: {fm_errors}")
    with open(ARTIFACTS_DATA_DIR / "feature_map.json", "w", encoding="utf-8") as f:
        json.dump(feature_map, f, indent=2)

    response_classes = {
        "threshold": threshold,
        "target": cfg.response_binning.target,
        "classes": cfg.response_binning.classes,
        "threshold_source": "train_split_median",
        "per_row_class": {
            str(rid): cls
            for rid, cls in zip(long_df["row_id"], long_df["response_class"])
        },
    }
    rc_errors = validate_response_classes(response_classes)
    if rc_errors:
        raise ValueError(f"response_classes.json failed validation: {rc_errors}")
    with open(ARTIFACTS_DATA_DIR / "response_classes.json", "w", encoding="utf-8") as f:
        json.dump(response_classes, f)

    # ---- 6. cached small matrices for baseline.py / sam.py's DataLoader --
    np.save(ARTIFACTS_DATA_DIR / "pathway_activity_matrix.npy", pathway_activity)
    np.save(ARTIFACTS_DATA_DIR / "mutation_matrix.npy", mutation_matrix)
    np.save(ARTIFACTS_DATA_DIR / "drug_fingerprint_matrix.npy", drug_fingerprints)

    print(f"Wrote {len(long_df):,} rows -> {ARTIFACTS_DATA_DIR}")
    print(f"  splits: train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,}")
    print(f"  response threshold ({cfg.response_binning.threshold}, from train): {threshold:.4f}")
    print(f"  feature_map: {len(feature_map)} features "
          f"({len(feature_cols)} pathway-activity, see limitation note above; "
          f"{len(mutation_feature_cols)} mutation)")


if __name__ == "__main__":
    preprocess()
