"""
Stage 1: train the unregularised overfitting baseline (Section 4.2.3).

DeepCCDS is trained with plain Adam (lr=1e-3), no weight decay / dropout / other
regularisation. We deliberately do NOT stop at the point of best validation loss --
instead we use "early stopping in reverse": training keeps going with patience=30
purely to *confirm* the generalisation gap has stabilised (i.e. training loss keeps
falling while validation loss has plateaued or risen for >=10 consecutive epochs).
This intentionally-overfit checkpoint is the Fragility Baseline that Stage 2 (SAM,
feat/sam-intervention branch) is compared against.

Usage:
    python -m src.model.train_baseline
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import GDSCDataset
from src.model.deepccds import DeepCCDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 500
PLATEAU_PATIENCE = 30           # confirmation patience (Section 4.2.3)
PLATEAU_MIN_CONSECUTIVE = 10    # min consecutive non-improving epochs to call it stabilised
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
    """Generalisation Gap = Validation Loss - Training Loss (Eq. 1, Section 4.2.3)."""
    return val_loss - train_loss


def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None, device="cpu") -> tuple[float, torch.Tensor, torch.Tensor]:
    """Runs one epoch. If `optimizer` is None, runs in eval mode with no grad."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_samples = 0.0, 0
    all_true, all_pred = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            n_samples += X.size(0)
            all_true.append(y.detach())
            all_pred.append(preds.detach())

    avg_loss = total_loss / n_samples
    return avg_loss, torch.cat(all_true), torch.cat(all_pred)


def has_stabilised(val_history: list[float], min_consecutive: int = PLATEAU_MIN_CONSECUTIVE) -> bool:
    """True once validation loss has plateaued or increased for the last
    `min_consecutive` epochs (Section 4.2.3 stopping criterion)."""
    if len(val_history) < min_consecutive + 1:
        return False
    recent = val_history[-(min_consecutive + 1):]
    non_improving = all(recent[i + 1] >= recent[i] - 1e-6 for i in range(len(recent) - 1))
    return non_improving


def train_baseline(
    n_genes: int | None = None,
    device: str = "cpu",
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict:
    set_seed()

    train_ds = GDSCDataset("train")
    val_ds = GDSCDataset("val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_genes = n_genes or train_ds.n_features
    model = DeepCCDS(n_genes=n_genes, dropout=0.0).to(device)  # no regularisation for Stage 1

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # plain Adam, no weight decay

    history: list[EpochMetrics] = []
    val_loss_history: list[float] = []
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_val_true, y_val_pred = run_epoch(model, val_loader, criterion, None, device)

        gap = generalisation_gap(val_loss, train_loss)
        history.append(EpochMetrics(epoch, train_loss, val_loss, gap))
        val_loss_history.append(val_loss)

        logger.info(
            "epoch %3d | train_loss=%.4f val_loss=%.4f gen_gap=%.4f",
            epoch, train_loss, val_loss, gap,
        )

        best_state = model.state_dict()  # Stage 1 deliberately keeps the LATEST state,
        # not the best-val checkpoint -- we want the overfit endpoint, not early-stopped weights.

        if has_stabilised(val_loss_history):
            logger.info(
                "Generalisation gap stabilised after %d epochs (val loss non-improving "
                "for %d consecutive epochs). Stopping.", epoch, PLATEAU_MIN_CONSECUTIVE,
            )
            break
    else:
        logger.warning("Reached MAX_EPOCHS=%d without a confirmed stabilised gap.", MAX_EPOCHS)

    final_val_loss, y_val_true, y_val_pred = run_epoch(model, val_loader, criterion, None, device)
    final_train_loss, y_train_true, y_train_pred = run_epoch(model, train_loader, criterion, None, device)

    summary = {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_generalisation_gap": generalisation_gap(final_val_loss, final_train_loss),
        "train_r2": r_squared(y_train_true, y_train_pred),
        "val_r2": r_squared(y_val_true, y_val_pred),
        "train_rmse": rmse(y_train_true, y_train_pred),
        "val_rmse": rmse(y_val_true, y_val_pred),
        "n_epochs": len(history),
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "regularisation": "none",
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_dir / "baseline_adam.pt")
    with open(checkpoint_dir / "baseline_adam_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(checkpoint_dir / "baseline_adam_history.json", "w") as f:
        json.dump([asdict(h) for h in history], f, indent=2)

    logger.info("Stage 1 baseline complete. Summary: %s", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train_baseline()
