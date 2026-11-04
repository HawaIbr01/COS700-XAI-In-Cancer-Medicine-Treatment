"""
Stage 1: train the unregularised overfitting baseline ("Fragility Baseline").

OWNER: Angelique (feat/baseline-model)
Methodology: §4.2.3 (Stage 1 training), §4.2.4 (Clever Hans sanity check).

CONTRACT (docs/CONTRACTS.md Arm 1):
    Produces:
        artifacts/checkpoints/baseline/{model.pt, meta.json}
        artifacts/predictions/baseline_{train,val,test}.parquet
        artifacts/reports/baseline_gap_scorecard.json (+ loss-curve figure,
            + Clever Hans attribution summary)

Requires artifacts/data/* to already exist — run `python -m src.data.preprocess`
first (see README.md).

Stopping rule ("early stopping in reverse", §4.2.3): we deliberately do NOT
stop at the epoch of best validation loss. Training keeps going with
patience=30 epochs to *confirm* the generalisation gap has genuinely
stabilised (train loss still falling while val loss has stopped improving),
rather than stopping the moment val loss ticks up once. The checkpoint we
save is the model's state at the point stabilisation is confirmed — this
deliberately-overfit endpoint is the Fragility Baseline that Stage 2 (SAM,
Bahiya's branch) is compared against.

Usage:
    python -m src.model.baseline
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.common.checkpoint import save_checkpoint, hash_split
from src.data.dataset import DeepCCDSDataset
from src.model.deepccds import DeepCCDS, compute_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "baseline"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

LEARNING_RATE = 1e-3     # paper's grid search ("Parameter Setting of Feedforward Neural Network")
BATCH_SIZE = 1024        # paper's grid search
MAX_EPOCHS = 500         # hard safety cap — see has_stabilised() below
PATIENCE = 30            # §4.2.3: confirmation patience
MIN_CONSECUTIVE = 10     # §4.2.3: minimum consecutive non-improving epochs reported alongside PATIENCE
MIN_DELTA = 1e-4         # minimum change in val loss to count as "improvement"
SEED = 42


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    generalisation_gap: float


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def generalisation_gap(val_loss: float, train_loss: float) -> float:
    """Generalisation Gap = Validation Loss - Training Loss (Eq. 1, §4.2.3)."""
    return val_loss - train_loss


def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


class StoppingTracker:
    """Implements the §4.2.3 "early stopping in reverse" rule.

    We track the best (lowest) validation loss seen so far and count
    consecutive epochs since it last improved by more than MIN_DELTA.
    Stabilisation is confirmed once that streak reaches PATIENCE epochs *and*
    training loss is still meaningfully lower than it was MIN_CONSECUTIVE
    epochs ago (i.e. the model is still fitting the training set while
    validation has genuinely plateaued/worsened — not just noisy).
    """

    def __init__(self, patience: int = PATIENCE, min_consecutive: int = MIN_CONSECUTIVE,
                 min_delta: float = MIN_DELTA):
        self.patience = patience
        self.min_consecutive = min_consecutive
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.no_improve_streak = 0
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    def update(self, train_loss: float, val_loss: float) -> None:
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.no_improve_streak = 0
        else:
            self.no_improve_streak += 1

    def stabilised(self) -> bool:
        if self.no_improve_streak < self.patience:
            return False
        if len(self.train_loss_history) < self.min_consecutive + 1:
            return False
        train_still_falling = (
            self.train_loss_history[-1] < self.train_loss_history[-(self.min_consecutive + 1)]
        )
        return train_still_falling


def run_epoch(model: DeepCCDS, loader: DataLoader, optimizer=None, device: str = "cpu"
              ) -> tuple[float, torch.Tensor, torch.Tensor, np.ndarray]:
    """Runs one epoch. If `optimizer` is None, runs in eval mode with no grad.
    Returns (avg_loss, y_true, y_pred, row_ids)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_samples = 0.0, 0
    all_true, all_pred, all_row_ids = [], [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for row_id, pathway, mutation, drug, y in loader:
            pathway, mutation, drug, y = pathway.to(device), mutation.to(device), drug.to(device), y.to(device)
            out = model(pathway, mutation=mutation, drug=drug)
            loss = compute_loss(out, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * pathway.size(0)
            n_samples += pathway.size(0)
            all_true.append(y.detach())
            all_pred.append(out["prediction"].detach())
            all_row_ids.append(row_id.numpy() if torch.is_tensor(row_id) else np.asarray(row_id))

    avg_loss = total_loss / n_samples
    return avg_loss, torch.cat(all_true), torch.cat(all_pred), np.concatenate(all_row_ids)


def _class_from_threshold(ln_ic50: np.ndarray, threshold: float, sensitive: str, resistant: str) -> np.ndarray:
    return np.where(ln_ic50 < threshold, sensitive, resistant)


def train_baseline(device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    set_seed(cfg.seed)

    if not (DATA_DIR / "gdsc_processed.parquet").exists():
        raise FileNotFoundError(
            f"{DATA_DIR / 'gdsc_processed.parquet'} not found. "
            f"Run `python -m src.data.preprocess` first."
        )

    with open(DATA_DIR / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(DATA_DIR / "response_classes.json", "r", encoding="utf-8") as f:
        response_classes = json.load(f)
    threshold = response_classes["threshold"]
    sensitive_label, resistant_label = response_classes["classes"]

    train_ds = DeepCCDSDataset("train")
    val_ds = DeepCCDSDataset("val")
    test_ds = DeepCCDSDataset("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepCCDS(dropout=0.0).to(device)  # no regularisation — Stage 1 (§4.2.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # plain Adam, no weight decay

    tracker = StoppingTracker()
    history: list[EpochMetrics] = []
    final_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _, _ = run_epoch(model, train_loader, optimizer, device)
        val_loss, _, _, _ = run_epoch(model, val_loader, None, device)
        gap = generalisation_gap(val_loss, train_loss)

        tracker.update(train_loss, val_loss)
        history.append(EpochMetrics(epoch, train_loss, val_loss, gap))
        final_epoch = epoch

        logger.info(
            "epoch %3d | train_loss=%.4f val_loss=%.4f gen_gap=%.4f no_improve_streak=%d",
            epoch, train_loss, val_loss, gap, tracker.no_improve_streak,
        )

        if tracker.stabilised():
            logger.info(
                "Generalisation gap stabilised after %d epochs "
                "(val loss non-improving for %d epochs, patience=%d). Stopping.",
                epoch, tracker.no_improve_streak, PATIENCE,
            )
            break
    else:
        logger.warning(
            "Reached MAX_EPOCHS=%d without the patience-based stabilisation check firing. "
            "Proceeding with the current (final-epoch) weights as the Fragility Baseline "
            "regardless — the gap is still reported below so this is visible, not hidden.",
            MAX_EPOCHS,
        )

    # ---- final scorecard: R^2 / RMSE on all three splits ------------------
    train_loss, y_train_true, y_train_pred, train_row_ids = run_epoch(model, train_loader, None, device)
    val_loss, y_val_true, y_val_pred, val_row_ids = run_epoch(model, val_loader, None, device)
    test_loss, y_test_true, y_test_pred, test_row_ids = run_epoch(model, test_loader, None, device)

    gap = generalisation_gap(val_loss, train_loss)
    scorecard = {
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "final_test_loss": test_loss,
        "generalisation_gap": gap,
        "train_r2": r_squared(y_train_true, y_train_pred),
        "val_r2": r_squared(y_val_true, y_val_pred),
        "test_r2": r_squared(y_test_true, y_test_pred),
        "train_rmse": rmse(y_train_true, y_train_pred),
        "val_rmse": rmse(y_val_true, y_val_pred),
        "test_rmse": rmse(y_test_true, y_test_pred),
        "n_epochs": final_epoch,
        "optimiser": "adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "regularisation": "none",
        "stopping_rule": f"patience={PATIENCE}, min_consecutive={MIN_CONSECUTIVE}, min_delta={MIN_DELTA}",
        "data_source": "DeepCCDS authors' precomputed features (real, paper-verified — see precomputed_inputs.py)",
    }

    # ---- checkpoint (docs/CONTRACTS.md Arm 1) ------------------------------
    save_checkpoint(
        model,
        CHECKPOINT_DIR,
        optimiser="adam",
        config_hash=cfg.config_hash,
        split_hash=hash_split(splits),
        epoch=final_epoch,
        generalisation_gap=gap,
        r2=scorecard["val_r2"],
        rmse=scorecard["val_rmse"],
        extra_meta={
            "arch_config": model.arch_config(),
            "train_r2": scorecard["train_r2"],
            "test_r2": scorecard["test_r2"],
            "train_rmse": scorecard["train_rmse"],
            "test_rmse": scorecard["test_rmse"],
            "response_threshold": threshold,
        },
    )

    # ---- predictions parquet (row_id, ln_ic50_pred, class_pred) ----------
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, row_ids, y_pred in (
        ("train", train_row_ids, y_train_pred),
        ("val", val_row_ids, y_val_pred),
        ("test", test_row_ids, y_test_pred),
    ):
        y_pred_np = y_pred.cpu().numpy()
        preds_df = pd.DataFrame({
            "row_id": row_ids,
            "ln_ic50_pred": y_pred_np,
            "class_pred": _class_from_threshold(y_pred_np, threshold, sensitive_label, resistant_label),
        })
        preds_df.to_parquet(PREDICTIONS_DIR / f"baseline_{split_name}.parquet")

    # ---- reports: scorecard json + loss-curve figure -----------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "baseline_gap_scorecard.json", "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    with open(REPORTS_DIR / "baseline_training_history.json", "w", encoding="utf-8") as f:
        json.dump([asdict(h) for h in history], f, indent=2)

    try:
        _plot_loss_curve(history, REPORTS_DIR / "baseline_loss_curve.png")
    except Exception as exc:  # pragma: no cover — plotting must never break training
        logger.warning("Could not save loss-curve figure: %s", exc)

    logger.info("Stage 1 baseline complete. Scorecard: %s", json.dumps(scorecard, indent=2))

    # ---- Clever Hans sanity check (§4.2.4) --------------------------------
    from src.model.clever_hans_audit import run_clever_hans_audit
    run_clever_hans_audit(model, device=device)

    return scorecard


def _plot_loss_curve(history: list[EpochMetrics], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h.epoch for h in history]
    train_loss = [h.train_loss for h in history]
    val_loss = [h.val_loss for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, label="train loss", color="#1f77b4")
    ax.plot(epochs, val_loss, label="val loss", color="#d62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (Eq. 2, MSE-based)")
    ax.set_title("Stage 1 baseline (Adam, no regularisation) — training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    train_baseline()
