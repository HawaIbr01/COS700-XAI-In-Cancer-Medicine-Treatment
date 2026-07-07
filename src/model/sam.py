"""
Stage 2: retrain DeepCCDS with Sharpness-Aware Minimisation.

OWNER: Bahiya (feat/sam-intervention)
Methodology: §4.5

CONTRACT (docs/CONTRACTS.md Arm 2):
    Receives: src.model.deepccds.DeepCCDS (UNCHANGED), artifacts/data/*
    Produces:
        artifacts/checkpoints/sam/{model.pt, meta.json}
        artifacts/predictions/sam_{train,val,test}.parquet
        artifacts/reports/sam_gap_scorecard.json (+ loss-landscape figure,
            directly comparable to baseline's)

    The ONLY thing that should differ from src/model/baseline.py is
    the optimiser. Architecture, data, splits, normalisation, and the
    training/eval loop shape must all stay identical — that's what
    makes the baseline-vs-SAM comparison valid (§4.5.1).

TODO (Bahiya):
    - Implement / import a SAM optimiser wrapper (neighbourhood
      epsilon-ball perturbation step, §4.5.2):
          1. compute gradient at theta
          2. ascent step to theta + epsilon (worst-case perturbation)
          3. compute gradient at theta + epsilon
          4. descent step using that gradient, applied at original theta
    - Reuse DeepCCDS(**same arch_config as baseline) — load
      arch_config from the baseline checkpoint meta if easiest, or
      from configs/pipeline.yaml if it's added there.
    - Reuse the identical data loading / split / preprocessing from
      Angelique's arm (import, don't reimplement).
    - Train, compute generalisation_gap/r2/rmse the same way as
      baseline, save via save_checkpoint(..., optimiser="sam",
      extra_meta={"rho": <SAM neighbourhood size>}).
    - Produce the comparative loss-landscape figure (same axes/scale
      as baseline's, so they can be placed side by side).
"""

from __future__ import annotations

from pathlib import Path

from src.common.config import load_config
from src.common.checkpoint import save_checkpoint, hash_split, peek_meta
from src.model.deepccds import DeepCCDS

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def train_sam() -> None:
    cfg = load_config()

    # TODO: confirm architecture matches baseline exactly, e.g.:
    #   baseline_meta = peek_meta(ARTIFACTS_DIR / "checkpoints" / "baseline")
    #   assert baseline_meta["config_hash"] == cfg.config_hash
    # TODO: load identical data/splits used by baseline
    # TODO: instantiate DeepCCDS(**same arch_config)
    # TODO: wrap optimiser with SAM (see docstring above)
    # TODO: training loop (same shape as baseline.py, different optimiser)
    # TODO: compute generalisation_gap, r2, rmse
    # TODO: save_checkpoint(model, ARTIFACTS_DIR / "checkpoints" / "sam",
    #                        optimiser="sam", config_hash=cfg.config_hash,
    #                        split_hash=hash_split(splits), epoch=...,
    #                        extra_meta={"rho": ...})
    # TODO: write predictions parquet
    # TODO: comparative loss-landscape figure vs baseline
    raise NotImplementedError("train_sam: see TODOs above")


if __name__ == "__main__":
    train_sam()
