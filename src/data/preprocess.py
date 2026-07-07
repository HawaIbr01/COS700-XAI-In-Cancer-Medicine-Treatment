"""
GDSC preprocessing: merge, clean, normalise, discretise, split.

OWNER: Angelique (feat/baseline-model)
Methodology: §4.2.2

CONTRACT (docs/CONTRACTS.md Arm 1):
    Produces (read-only for every other arm from this point on):
        artifacts/data/gdsc_processed.parquet
        artifacts/data/splits.json
        artifacts/data/feature_map.json
        artifacts/data/response_classes.json

TODO (Angelique):
    - Merge the three GDSC sub-datasets (cell line features, drug
      info, drug response measurements).
    - Deduplicate, handle missing values, one-hot encode categoricals.
    - Z-score normalise gene expression features.
    - Build feature_map.json: {feature_col: {"gene_symbol": ...,
      "partition_edges": [cfg.discretisation.low_z, cfg.discretisation.high_z]}}
      — every extraction/evaluation arm resolves genes through this.
    - Discretise expression into High/Low/Intermediate using
      cfg.discretisation (Z > high_z -> High, Z < low_z -> Low, else
      Intermediate).
    - Bin the continuous ln_ic50 target into response classes using
      cfg.response_binning (threshold starts at "median" — pilot may
      fix this to a specific value; once fixed, update
      configs/pipeline.yaml, not a hardcoded value here).
    - Produce the fixed 80/10/10 split (seeded via cfg.seed), save as
      splits.json (row ids per split — validated by
      src.common.schema.validate_splits, which checks for zero
      overlap).
    - Write response_classes.json: {"threshold": ..., "per_row_class":
      {row_id: "sensitive"|"resistant"}}.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.common.config import load_config
from src.common.schema import (
    validate_splits,
    validate_feature_map,
    validate_response_classes,
)

ARTIFACTS_DATA_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "data"


def preprocess() -> None:
    cfg = load_config()

    # TODO: load raw GDSC sub-datasets (paths TBD once data is sourced)
    # TODO: merge, dedup, handle missing values, one-hot encode
    # TODO: Z-score normalise expression features
    # TODO: build feature_map (feature -> gene_symbol + partition_edges)
    # TODO: discretise features per cfg.discretisation
    # TODO: bin ln_ic50 into response classes per cfg.response_binning
    # TODO: seeded 80/10/10 split per cfg.split, using cfg.seed
    # TODO: write all four artifacts, validating each before writing:
    #   ARTIFACTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    #   (df).to_parquet(ARTIFACTS_DATA_DIR / "gdsc_processed.parquet")
    #   assert not validate_splits(splits); json.dump(splits, ...)
    #   assert not validate_feature_map(feature_map); json.dump(...)
    #   assert not validate_response_classes(response_classes); json.dump(...)
    raise NotImplementedError("preprocess: see TODOs above")


if __name__ == "__main__":
    preprocess()
