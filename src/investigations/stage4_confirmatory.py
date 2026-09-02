"""
Stage 4 — confirmatory full-data run: SAM at ρ=0.5, WD=1e-3, LR=0.001.

Validates the Stage 3 subsample finding on the full training set, using the same
stopping criterion as the original pipeline (PATIENCE=30, MIN_CONSECUTIVE=10).
Results are saved to artifacts/sam_investigation/confirmatory/ (NOT Bahiya's
artifacts/checkpoints/sam/ — that checkpoint must not be overwritten).

Run from repo root:
    python -m src.investigations.stage4_confirmatory
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DeepCCDSDataset
from src.model.deepccds import DeepCCDS, compute_loss
from src.model.sam import SAM, _disabled_running_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path(__file__).resolve().parents[2]
            / "artifacts" / "sam_investigation" / "stage4_confirmatory.log",
            mode="w",
        ),
    ],
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "artifacts" / "sam_investigation" / "confirmatory"
BASELINE_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "baseline"

# Recommended config from Stage 3
RHO = 0.50
LR  = 0.001
WD  = 1e-3

# Stopping criterion — identical to baseline.py / sam.py production runs
BATCH_SIZE      = 1024
MIN_DELTA       = 1e-4
PATIENCE        = 30
MIN_CONSECUTIVE = 10
MAX_EPOCHS      = 500

# Stage 1 baseline reference (full-data, eval-mode)
BASELINE_GAP = 0.3369
BASELINE_R2  = 0.8565


class StoppingTracker:
    def __init__(self):
        self.best_val_loss    = float("inf")
        self.no_improve_streak = 0
        self.train_history: list[float] = []
        self.val_history:   list[float] = []

    def update(self, train_loss: float, val_loss: float) -> None:
        self.train_history.append(train_loss)
        self.val_history.append(val_loss)
        if val_loss < self.best_val_loss - MIN_DELTA:
            self.best_val_loss = val_loss
            self.no_improve_streak = 0
        else:
            self.no_improve_streak += 1

    def stabilised(self) -> bool:
        if self.no_improve_streak < PATIENCE:
            return False
        if len(self.train_history) < MIN_CONSECUTIVE + 1:
            return False
        return (
            self.train_history[-1]
            < self.train_history[-(MIN_CONSECUTIVE + 1)]
        )


def eval_split(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for _, pathway, mutation, drug, y in loader:
            pathway, mutation, drug, y = (
                pathway.to(device), mutation.to(device),
                drug.to(device), y.to(device),
            )
            out = model(pathway, mutation=mutation, drug=drug)
            total_loss += compute_loss(out, y).item() * pathway.size(0)
            n += pathway.size(0)
    return total_loss / n


def r_squared(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for _, pathway, mutation, drug, y in loader:
            pathway, mutation, drug, y = (
                pathway.to(device), mutation.to(device),
                drug.to(device), y.to(device),
            )
            out = model(pathway, mutation=mutation, drug=drug)
            all_true.append(y)
            all_pred.append(out["prediction"])
    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")


def rmse(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for _, pathway, mutation, drug, y in loader:
            pathway, mutation, drug, y = (
                pathway.to(device), mutation.to(device),
                drug.to(device), y.to(device),
            )
            out = model(pathway, mutation=mutation, drug=drug)
            all_true.append(y)
            all_pred.append(out["prediction"])
    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def train_epoch_sam(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for _, pathway, mutation, drug, y in loader:
        pathway, mutation, drug, y = (
            pathway.to(device), mutation.to(device),
            drug.to(device), y.to(device),
        )
        out  = model(pathway, mutation=mutation, drug=drug)
        loss = compute_loss(out, y)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        with _disabled_running_stats(model):
            out2  = model(pathway, mutation=mutation, drug=drug)
            loss2 = compute_loss(out2, y)
        loss2.backward()
        optimizer.second_step(zero_grad=True)
        total_loss += loss.item() * pathway.size(0)
        n += pathway.size(0)
    return total_loss / n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("Config: rho=%.2f  LR=%s  WD=%s  MAX_EPOCHS=%d", RHO, LR, WD, MAX_EPOCHS)

    with open(BASELINE_CHECKPOINT_DIR / "meta.json", encoding="utf-8") as f:
        arch_config = json.load(f)["arch_config"]

    train_ds = DeepCCDSDataset("train")
    val_ds   = DeepCCDSDataset("val")
    test_ds  = DeepCCDSDataset("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    logger.info(
        "Dataset sizes — train: %d  val: %d  test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    torch.manual_seed(42)
    np.random.seed(42)
    model     = DeepCCDS(**arch_config).to(device)
    optimizer = SAM(
        model.parameters(),
        base_optimizer_cls=torch.optim.Adam,
        rho=RHO, lr=LR, weight_decay=WD,
    )
    tracker   = StoppingTracker()
    t0        = time.time()
    final_epoch = 0
    converged   = False

    epoch_history = []  # list of {epoch, train_loss, val_loss, gap}

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch_sam(model, train_loader, optimizer, device)
        val_loss   = eval_split(model, val_loader, device)
        tracker.update(train_loss, val_loss)
        final_epoch = epoch

        gap = val_loss - train_loss
        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "gap":        round(gap,         6),
        })

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            logger.info(
                "epoch %3d | train=%.4f val=%.4f gap=%.4f | no_imp=%d | %.0fs elapsed",
                epoch, train_loss, val_loss, gap,
                tracker.no_improve_streak, elapsed,
            )

        if tracker.stabilised():
            logger.info("Stopping criterion met at epoch %d.", epoch)
            converged = True
            break
    else:
        logger.warning("Reached MAX_EPOCHS=%d without stopping criterion firing.", MAX_EPOCHS)

    # Final eval-mode metrics
    final_train = eval_split(model, train_loader, device)
    final_val   = eval_split(model, val_loader,   device)
    final_test  = eval_split(model, test_loader,  device)
    val_r2      = r_squared(model, val_loader, device)
    val_rmse    = rmse(model, val_loader, device)
    runtime     = time.time() - t0

    gen_gap = final_val - final_train

    logger.info(
        "FINAL | train=%.4f val=%.4f test=%.4f gap=%.4f val_r2=%.4f val_rmse=%.4f | "
        "epochs=%d conv=%s runtime=%.0fs",
        final_train, final_val, final_test, gen_gap, val_r2, val_rmse,
        final_epoch, converged, runtime,
    )

    # Compare with baseline
    gap_reduction = (BASELINE_GAP - gen_gap) / BASELINE_GAP * 100
    r2_delta      = val_r2 - BASELINE_R2
    logger.info(
        "vs. Adam baseline | gap: %.4f → %.4f (%.1f%%) | val_r2: %.4f → %.4f (Δ=%.4f)",
        BASELINE_GAP, gen_gap, gap_reduction, BASELINE_R2, val_r2, r2_delta,
    )

    scorecard = {
        "config": {"rho": RHO, "lr": LR, "wd": WD},
        "n_epochs": final_epoch,
        "converged": converged,
        "stopping_rule": f"patience={PATIENCE}, min_consecutive={MIN_CONSECUTIVE}, min_delta={MIN_DELTA}",
        "final_train_loss": final_train,
        "final_val_loss":   final_val,
        "final_test_loss":  final_test,
        "generalisation_gap": gen_gap,
        "val_r2":   val_r2,
        "val_rmse": val_rmse,
        "gap_vs_baseline": {
            "baseline_gap": BASELINE_GAP,
            "baseline_r2":  BASELINE_R2,
            "gap_reduction_pct": round(gap_reduction, 2),
            "r2_delta": round(r2_delta, 4),
        },
        "runtime_s": round(runtime, 1),
        "dataset_sizes": {
            "train": len(train_ds),
            "val":   len(val_ds),
            "test":  len(test_ds),
        },
        "max_epochs_cap": MAX_EPOCHS,
    }

    out_scorecard = OUT_DIR / "confirmatory_scorecard.json"
    out_history   = OUT_DIR / "confirmatory_history.json"

    with open(out_scorecard, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    logger.info("Saved: %s", out_scorecard)

    with open(out_history, "w", encoding="utf-8") as f:
        json.dump(epoch_history, f, indent=2)
    logger.info("Saved: %s", out_history)


if __name__ == "__main__":
    main()
