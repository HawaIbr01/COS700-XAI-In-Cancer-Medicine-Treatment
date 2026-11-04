"""
Stage 3 re-run — the three configs that hit MAX_EPOCHS=150 in the original sweep.

Re-runs with MAX_EPOCHS=300 so they can converge. Overwrites the original
per-config JSON files in artifacts/sam_investigation/stage3_sweep/.

Non-converged configs (all ρ≥1.0, lr=0.0005):
  rho=1.0, lr=0.0005, wd=0
  rho=2.0, lr=0.0005, wd=0
  rho=2.0, lr=0.0005, wd=0.001

Run from repo root:
  python -m src.investigations.stage3_rerun_nonconverged
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.common.config import load_config
from src.data.dataset import DeepCCDSDataset
from src.model.deepccds import DeepCCDS, compute_loss
from src.model.sam import SAM, _disabled_running_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "artifacts" / "sam_investigation" / "stage3_sweep"
BASELINE_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "baseline"

# Same subsample as original sweep
SUBSAMPLE_FRAC = 0.10
SUBSAMPLE_SEED = 42
BATCH_SIZE     = 1024
MIN_DELTA      = 1e-4
PATIENCE       = 15
MIN_CONSECUTIVE = 5
MAX_EPOCHS     = 300   # raised from 150

NON_CONVERGED = [
    (1.0, 0.0005, 0.0),
    (2.0, 0.0005, 0.0),
    (2.0, 0.0005, 1e-3),
]


class StoppingTracker:
    def __init__(self):
        self.best_val_loss = float("inf")
        self.no_improve_streak = 0
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    def update(self, train_loss: float, val_loss: float) -> None:
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        if val_loss < self.best_val_loss - MIN_DELTA:
            self.best_val_loss = val_loss
            self.no_improve_streak = 0
        else:
            self.no_improve_streak += 1

    def stabilised(self) -> bool:
        if self.no_improve_streak < PATIENCE:
            return False
        if len(self.train_loss_history) < MIN_CONSECUTIVE + 1:
            return False
        return (
            self.train_loss_history[-1]
            < self.train_loss_history[-(MIN_CONSECUTIVE + 1)]
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


def train_epoch_sam(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for _, pathway, mutation, drug, y in loader:
        pathway, mutation, drug, y = (
            pathway.to(device), mutation.to(device),
            drug.to(device), y.to(device),
        )
        out = model(pathway, mutation=mutation, drug=drug)
        loss = compute_loss(out, y)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        with _disabled_running_stats(model):
            out2 = model(pathway, mutation=mutation, drug=drug)
            loss2 = compute_loss(out2, y)
        loss2.backward()
        optimizer.second_step(zero_grad=True)
        total_loss += loss.item() * pathway.size(0)
        n += pathway.size(0)
    return total_loss / n


def run_config(rho, lr, wd, arch_config, sub_train_loader, val_loader, device):
    label = f"rho={rho} lr={lr} wd={wd}"
    logger.info("--- Starting: %s (MAX_EPOCHS=%d)", label, MAX_EPOCHS)
    torch.manual_seed(SUBSAMPLE_SEED)
    np.random.seed(SUBSAMPLE_SEED)

    model = DeepCCDS(**arch_config).to(device)
    optimizer = SAM(
        model.parameters(),
        base_optimizer_cls=torch.optim.Adam,
        rho=rho, lr=lr, weight_decay=wd,
    )
    tracker = StoppingTracker()
    t0 = time.time()
    final_epoch = 0
    converged = False

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch_sam(model, sub_train_loader, optimizer, device)
        val_loss   = eval_split(model, val_loader, device)
        tracker.update(train_loss, val_loss)
        final_epoch = epoch
        if epoch % 10 == 0 or epoch == 1:
            logger.info("  %s | epoch %3d train=%.4f val=%.4f gap=%.4f no_imp=%d",
                        label, epoch, train_loss, val_loss,
                        val_loss - train_loss, tracker.no_improve_streak)
        if tracker.stabilised():
            logger.info("  %s | stopped at epoch %d", label, epoch)
            converged = True
            break
    else:
        logger.warning("  %s | reached MAX_EPOCHS=%d without stopping", label, MAX_EPOCHS)

    final_train = eval_split(model, sub_train_loader, device)
    final_val   = eval_split(model, val_loader, device)
    val_r2      = r_squared(model, val_loader, device)
    runtime     = time.time() - t0

    result = {
        "rho": rho, "lr": lr, "wd": wd,
        "n_epochs": final_epoch,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "generalisation_gap": final_val - final_train,
        "val_r2": val_r2,
        "runtime_s": round(runtime, 1),
        "converged": converged,
        "max_epochs_used": MAX_EPOCHS,
    }
    logger.info("  %s | DONE: gap=%.4f val_r2=%.4f epochs=%d conv=%s time=%.0fs",
                label, result["generalisation_gap"], val_r2, final_epoch, converged, runtime)
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    with open(BASELINE_CHECKPOINT_DIR / "meta.json", encoding="utf-8") as f:
        arch_config = json.load(f)["arch_config"]

    full_train_ds = DeepCCDSDataset("train")
    val_ds        = DeepCCDSDataset("val")
    n_sub = int(len(full_train_ds) * SUBSAMPLE_FRAC)
    rng   = np.random.default_rng(SUBSAMPLE_SEED)
    sub_idx = np.sort(rng.choice(len(full_train_ds), size=n_sub, replace=False))
    sub_train_loader = DataLoader(Subset(full_train_ds, sub_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader       = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    logger.info("Subsample: %d / %d  |  val: %d", n_sub, len(full_train_ds), len(val_ds))

    for rho, lr, wd in NON_CONVERGED:
        result = run_config(rho, lr, wd, arch_config, sub_train_loader, val_loader, device)
        slug = f"rho{rho}_lr{lr}_wd{wd}".replace(".", "p")
        out_path = OUT_DIR / f"config_{slug}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved: %s", out_path)

    logger.info("Re-run complete.")


if __name__ == "__main__":
    main()
