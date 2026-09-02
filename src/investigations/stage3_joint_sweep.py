"""
Stage 3 — Joint (learning_rate, weight_decay, rho) hyperparameter sweep.

Tests two hypotheses from Stage 2:
  (a) rho was too small — the optimal operating point lies above ρ=0.20.
  (b) LR/WD were not jointly retuned — Adam-tuned LR=0.001, WD=0 may be
      suboptimal for SAM.

Grid (4 × 2 × 2 = 16 configs):
  rho:  0.20 (anchor/sanity-check), 0.50, 1.0, 2.0
  lr:   0.001 (Adam-tuned baseline), 0.0005 (half-LR, standard SAM recommendation)
  wd:   0 (current), 1e-3 (moderate, recommended for SAM)

Data: 10% random subsample of training set (~25.6k samples), seed=42.
The full validation and test sets are used unchanged.
Subsample rationale:
  - Reduces per-config runtime from ~5h to ~10-15 min (10x speedup).
  - Relative ordering of configs expected to be informative for the
    full-scale regime; absolute gap values will differ from full-data runs.
  - This is flagged explicitly: do not compare absolute gap numbers
    from this sweep with Bahiya's full-data results.

Stopping: patience=15, min_consecutive=5 (reduced from full-run 30/10).
MAX_EPOCHS = 150 (safety cap).

Saves results per-config and a sweep summary to:
  artifacts/sam_investigation/stage3_sweep/

Run from repo root:
  python -m src.investigations.stage3_joint_sweep [--dry-run]

Use --dry-run to print the grid without running anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.common.config import load_config
from src.data.dataset import DeepCCDSDataset
from src.model.deepccds import DeepCCDS, compute_loss
from src.model.sam import SAM, _disabled_running_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "artifacts" / "sam_investigation" / "stage3_sweep"
BASELINE_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "baseline"

# ------------------------------------------------------------------ #
# Sweep parameters (see module docstring for justification)
# ------------------------------------------------------------------ #

RHO_VALUES   = [0.20, 0.50, 1.0, 2.0]
LR_VALUES    = [0.001, 0.0005]
WD_VALUES    = [0, 1e-3]

SUBSAMPLE_FRAC = 0.10   # fraction of training set to use
SUBSAMPLE_SEED = 42

PATIENCE        = 15
MIN_CONSECUTIVE = 5
MIN_DELTA       = 1e-4
MAX_EPOCHS      = 150
BATCH_SIZE      = 1024


# ========================================================================== #
# Stopping tracker (same logic as baseline/sam.py, shorter patience)
# ========================================================================== #

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


# ========================================================================== #
# Evaluation
# ========================================================================== #

def eval_split(model: DeepCCDS, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for _, pathway, mutation, drug, y in loader:
            pathway, mutation, drug, y = (
                pathway.to(device), mutation.to(device),
                drug.to(device), y.to(device),
            )
            out = model(pathway, mutation=mutation, drug=drug)
            loss = compute_loss(out, y)
            total_loss += loss.item() * pathway.size(0)
            n += pathway.size(0)
    return total_loss / n


def r_squared(model: DeepCCDS, loader: DataLoader, device: str) -> float:
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


# ========================================================================== #
# SAM training epoch
# ========================================================================== #

def train_epoch_sam(
    model: DeepCCDS,
    loader: DataLoader,
    optimizer: SAM,
    device: str,
) -> float:
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


# ========================================================================== #
# Single config training run
# ========================================================================== #

@dataclass
class ConfigResult:
    rho: float
    lr: float
    wd: float
    n_epochs: int
    final_train_loss: float
    final_val_loss: float
    generalisation_gap: float
    val_r2: float
    runtime_s: float
    converged: bool


def run_config(
    rho: float,
    lr: float,
    wd: float,
    arch_config: dict,
    sub_train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
) -> ConfigResult:
    label = f"rho={rho} lr={lr} wd={wd}"
    logger.info("--- Starting: %s", label)

    torch.manual_seed(SUBSAMPLE_SEED)
    np.random.seed(SUBSAMPLE_SEED)

    model = DeepCCDS(**arch_config).to(device)
    optimizer = SAM(
        model.parameters(),
        base_optimizer_cls=torch.optim.Adam,
        rho=rho,
        lr=lr,
        weight_decay=wd,
    )

    tracker = StoppingTracker()
    t0 = time.time()
    final_epoch = 0
    converged = False

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch_sam(model, sub_train_loader, optimizer, device)
        val_loss = eval_split(model, val_loader, device)
        gap = val_loss - train_loss
        tracker.update(train_loss, val_loss)
        final_epoch = epoch

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  %s | epoch %3d train=%.4f val=%.4f gap=%.4f no_imp=%d",
                label, epoch, train_loss, val_loss, gap, tracker.no_improve_streak,
            )

        if tracker.stabilised():
            logger.info("  %s | stopped at epoch %d", label, epoch)
            converged = True
            break
    else:
        logger.warning("  %s | reached MAX_EPOCHS=%d without stopping", label, MAX_EPOCHS)

    # Final eval-mode scorecard (both train and val in eval mode for fair comparison)
    final_train = eval_split(model, sub_train_loader, device)
    final_val = eval_split(model, val_loader, device)
    val_r2 = r_squared(model, val_loader, device)
    runtime = time.time() - t0

    result = ConfigResult(
        rho=rho, lr=lr, wd=wd,
        n_epochs=final_epoch,
        final_train_loss=final_train,
        final_val_loss=final_val,
        generalisation_gap=final_val - final_train,
        val_r2=val_r2,
        runtime_s=round(runtime, 1),
        converged=converged,
    )
    logger.info(
        "  %s | DONE: gap=%.4f val_r2=%.4f epochs=%d time=%.0fs",
        label, result.generalisation_gap, result.val_r2, final_epoch, runtime,
    )
    return result


# ========================================================================== #
# Main
# ========================================================================== #

def main(dry_run: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    cfg = load_config()

    # ------------------------------------------------------------------ #
    # Load architecture config from baseline checkpoint
    # ------------------------------------------------------------------ #
    meta_path = BASELINE_CHECKPOINT_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No baseline meta.json at {meta_path}. "
            "Baseline checkpoint is required."
        )
    with open(meta_path, encoding="utf-8") as f:
        baseline_meta = json.load(f)
    arch_config = baseline_meta["arch_config"]
    logger.info("Loaded baseline arch_config: %s", arch_config)

    # ------------------------------------------------------------------ #
    # Build grid
    # ------------------------------------------------------------------ #
    grid = list(product(RHO_VALUES, LR_VALUES, WD_VALUES))
    logger.info(
        "Stage 3 sweep: %d configs (rho=%s, lr=%s, wd=%s)",
        len(grid), RHO_VALUES, LR_VALUES, WD_VALUES,
    )
    logger.info("Subsample: %.0f%% of training set, seed=%d", SUBSAMPLE_FRAC * 100, SUBSAMPLE_SEED)
    logger.info("Stopping: patience=%d, min_consecutive=%d, max_epochs=%d", PATIENCE, MIN_CONSECUTIVE, MAX_EPOCHS)

    if dry_run:
        print("\nDRY RUN — grid (rho, lr, wd):")
        for rho, lr, wd in grid:
            print(f"  rho={rho}, lr={lr}, wd={wd}")
        return

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    full_train_ds = DeepCCDSDataset("train")
    val_ds = DeepCCDSDataset("val")

    # 10% subsample of training set, random with fixed seed
    n_sub = int(len(full_train_ds) * SUBSAMPLE_FRAC)
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    sub_idx = np.sort(rng.choice(len(full_train_ds), size=n_sub, replace=False))
    sub_train_ds = Subset(full_train_ds, sub_idx)

    sub_train_loader = DataLoader(sub_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(
        "Subsample train size: %d / %d (%.1f%%)",
        n_sub, len(full_train_ds), 100 * n_sub / len(full_train_ds),
    )
    logger.info("Val size: %d", len(val_ds))

    # ------------------------------------------------------------------ #
    # Run grid
    # ------------------------------------------------------------------ #
    results: list[ConfigResult] = []
    sweep_start = time.time()

    for i, (rho, lr, wd) in enumerate(grid, start=1):
        logger.info("=== Config %d / %d: rho=%.2f lr=%g wd=%g ===", i, len(grid), rho, lr, wd)

        result = run_config(rho, lr, wd, arch_config, sub_train_loader, val_loader, device)
        results.append(result)

        # Save incremental result
        per_config_path = OUT_DIR / f"config_rho{rho}_lr{lr}_wd{wd}.json".replace(".", "p")
        with open(per_config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

    total_time = time.time() - sweep_start

    # ------------------------------------------------------------------ #
    # Save sweep summary
    # ------------------------------------------------------------------ #
    summary = {
        "stage": 3,
        "description": (
            "Joint (rho, lr, wd) sweep on 10% training subsample. "
            "Tests rho under-perturbation and LR/WD mismatch hypotheses."
        ),
        "subsample_frac": SUBSAMPLE_FRAC,
        "subsample_n_train": n_sub,
        "subsample_seed": SUBSAMPLE_SEED,
        "full_train_n": len(full_train_ds),
        "val_n": len(val_ds),
        "patience": PATIENCE,
        "min_consecutive": MIN_CONSECUTIVE,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "total_runtime_s": round(total_time, 1),
        "grid": {"rho": RHO_VALUES, "lr": LR_VALUES, "wd": WD_VALUES},
        "results": [asdict(r) for r in results],
    }
    summary_path = OUT_DIR / "stage3_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved sweep summary: %s", summary_path)

    # ------------------------------------------------------------------ #
    # Print table
    # ------------------------------------------------------------------ #
    print()
    print("=" * 80)
    print("STAGE 3 — Joint Sweep Results (10% subsample, eval-mode gaps)")
    print("=" * 80)
    print(f"{'rho':>6} {'lr':>8} {'wd':>8} {'gap':>8} {'val_R²':>8} {'epochs':>7} {'t(s)':>7} {'conv':>5}")
    print("-" * 80)

    # Sort by gap ascending (smaller gap = better)
    sorted_results = sorted(results, key=lambda r: r.generalisation_gap)
    for r in sorted_results:
        conv_flag = "Y" if r.converged else "N"
        print(f"{r.rho:>6.2f} {r.lr:>8.5f} {r.wd:>8.4f} "
              f"{r.generalisation_gap:>8.4f} {r.val_r2:>8.4f} "
              f"{r.n_epochs:>7d} {r.runtime_s:>7.0f} {conv_flag:>5}")
    print("=" * 80)
    print(f"\nTotal runtime: {total_time/60:.1f} min across {len(grid)} configs.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 joint SAM hyperparameter sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print grid without running")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
