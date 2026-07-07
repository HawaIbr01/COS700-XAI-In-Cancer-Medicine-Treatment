"""
Stage 1: train the unregularised overfitting baseline.

OWNER: Angelique (feat/baseline-model)
Methodology: §4.2.3 (Stage 1), §4.2.4 (Clever Hans sanity check)

CONTRACT (docs/CONTRACTS.md Arm 1):
    Produces:
        artifacts/checkpoints/baseline/{model.pt, meta.json}
        artifacts/predictions/baseline_{train,val,test}.parquet
        artifacts/reports/baseline_gap_scorecard.json (+ loss-landscape figure)

TODO (Angelique):
    - Load artifacts/data/* (gdsc_processed.parquet, splits.json).
    - Train DeepCCDS with Adam (lr=0.001), no regularisation.
    - Early-stopping-in-reverse: patience=30 epochs, stop when train
      loss keeps falling while val loss has plateaued/risen for >=10
      consecutive epochs (§4.2.3).
    - Compute generalisation_gap = val_loss - train_loss, R^2, RMSE
      on LN_IC50.
    - Save via src.common.checkpoint.save_checkpoint(..., optimiser="adam").
    - Run the Clever Hans attribution sanity check (§4.2.4): LRP /
      input-gradients on a sample of train+val predictions, check for
      diffuse / biologically incoherent relevance patterns.
    - Write predictions parquet with columns: row_id, ln_ic50_pred,
      class_pred (class_pred uses the response-class threshold from
      artifacts/data/response_classes.json).
"""

from __future__ import annotations

from pathlib import Path

from src.common.config import load_config
from src.common.checkpoint import save_checkpoint, hash_split
from src.model.deepccds import DeepCCDS

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def train_baseline() -> None:
    cfg = load_config()

    # TODO: load artifacts/data/gdsc_processed.parquet + splits.json
    # TODO: build DataLoaders from the fixed train/val/test split
    # TODO: instantiate DeepCCDS(**arch_config)
    # TODO: Adam optimiser, lr=0.001, MSE loss on ln_ic50
    # TODO: training loop with early-stopping-in-reverse (patience=30,
    #       stop condition: val plateau/rise for >=10 consecutive epochs)
    # TODO: compute generalisation_gap, r2, rmse on held-out data
    # TODO: save_checkpoint(model, ARTIFACTS_DIR / "checkpoints" / "baseline",
    #                        optimiser="adam", config_hash=cfg.config_hash,
    #                        split_hash=hash_split(splits), epoch=..., ...)
    # TODO: write predictions parquet
    # TODO: run Clever Hans attribution sanity check, save figures/summary
    #       to artifacts/reports/
    raise NotImplementedError("train_baseline: see TODOs above")


if __name__ == "__main__":
    train_baseline()
