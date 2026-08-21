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

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.common.checkpoint import save_checkpoint, hash_split, peek_meta
from src.data.dataset import DeepCCDSDataset
from src.model.deepccds import DeepCCDS, compute_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "sam"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
MAX_EPOCHS = 500
PATIENCE = 30
MIN_CONSECUTIVE = 10
MIN_DELTA = 1e-4

SAM_RHO = 0.05
SAM_ADAPTIVE = False


@dataclass
class EpochMetrics:
  epoch: int
  train_loss: float
  val_loss: float
  generalisation_gap: float


def generalisation_gap(val_loss: float, train_loss: float) -> float:
  return val_loss - train_loss


def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
  ss_res = torch.sum((y_true - y_pred) ** 2)
  ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
  return (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
  return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


class StoppingTracker:
  """Same early-stopping-in-reverse tracker as baseline.py."""

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


class SAM(Optimizer):
  """Sharpness-Aware Minimization wrapper around a base optimizer."""

  def __init__(
    self,
    params,
    base_optimizer,
    rho: float = SAM_RHO,
    adaptive: bool = SAM_ADAPTIVE,
    **kwargs,
  ):
    if rho < 0:
      raise ValueError("rho must be non-negative")

    defaults = {"rho": rho, "adaptive": adaptive, **kwargs}
    super().__init__(params, defaults)

    self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
    self.param_groups = self.base_optimizer.param_groups

  @torch.no_grad()
  def first_step(self, zero_grad: bool = False) -> None:
    grad_norm = self._grad_norm()
    for group in self.param_groups:
      scale = group["rho"] / (grad_norm + 1e-12)
      adaptive = group["adaptive"]
      for p in group["params"]:
        if p.grad is None:
          continue
        self.state[p]["old_p"] = p.data.clone()
        e_w = ((torch.pow(p, 2) if adaptive else 1.0) * p.grad) * scale.to(p)
        p.add_(e_w)

    if zero_grad:
      self.zero_grad()

  @torch.no_grad()
  def second_step(self, zero_grad: bool = False) -> None:
    for group in self.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue
        p.data = self.state[p]["old_p"]
    self.base_optimizer.step()

    if zero_grad:
      self.zero_grad()

  @torch.no_grad()
  def step(self, closure=None):
    raise RuntimeError("SAM needs explicit first_step/second_step in the training loop")

  def _grad_norm(self) -> torch.Tensor:
    shared_device = self.param_groups[0]["params"][0].device
    norms = []
    for group in self.param_groups:
      adaptive = group["adaptive"]
      for p in group["params"]:
        if p.grad is None:
          continue
        grad = (torch.abs(p) if adaptive else 1.0) * p.grad
        norms.append(torch.norm(grad, p=2).to(shared_device))
    if not norms:
      return torch.tensor(0.0, device=shared_device)
    return torch.norm(torch.stack(norms), p=2)


def run_epoch(model: DeepCCDS, loader: DataLoader, optimizer=None, device: str = "cpu"
        ) -> tuple[float, torch.Tensor, torch.Tensor, np.ndarray]:
  """Runs one epoch. If `optimizer` is None, eval mode with no grad.
  Returns (avg_loss, y_true, y_pred, row_ids)."""
  is_train = optimizer is not None
  model.train() if is_train else model.eval()

  total_loss, n_samples = 0.0, 0
  all_true, all_pred, all_row_ids = [], [], []

  if is_train:
    for row_id, pathway, mutation, drug, y in loader:
      pathway, mutation, drug, y = pathway.to(device), mutation.to(device), drug.to(device), y.to(device)

      out = model(pathway, mutation=mutation, drug=drug)
      loss = compute_loss(out, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.first_step(zero_grad=True)

      out_perturbed = model(pathway, mutation=mutation, drug=drug)
      loss_perturbed = compute_loss(out_perturbed, y)
      loss_perturbed.backward()
      optimizer.second_step(zero_grad=True)

      total_loss += loss_perturbed.item() * pathway.size(0)
      n_samples += pathway.size(0)
      all_true.append(y.detach())
      all_pred.append(out_perturbed["prediction"].detach())
      all_row_ids.append(row_id.numpy() if torch.is_tensor(row_id) else np.asarray(row_id))
  else:
    with torch.no_grad():
      for row_id, pathway, mutation, drug, y in loader:
        pathway, mutation, drug, y = pathway.to(device), mutation.to(device), drug.to(device), y.to(device)
        out = model(pathway, mutation=mutation, drug=drug)
        loss = compute_loss(out, y)

        total_loss += loss.item() * pathway.size(0)
        n_samples += pathway.size(0)
        all_true.append(y.detach())
        all_pred.append(out["prediction"].detach())
        all_row_ids.append(row_id.numpy() if torch.is_tensor(row_id) else np.asarray(row_id))

  avg_loss = total_loss / n_samples
  return avg_loss, torch.cat(all_true), torch.cat(all_pred), np.concatenate(all_row_ids)


def _class_from_threshold(ln_ic50: np.ndarray, threshold: float, sensitive: str, resistant: str) -> np.ndarray:
  return np.where(ln_ic50 < threshold, sensitive, resistant)


def train_sam(device: str | None = None) -> dict:
  device = device or ("cuda" if torch.cuda.is_available() else "cpu")
  cfg = load_config()

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

  model = DeepCCDS(dropout=0.0).to(device)
  optimizer = SAM(
    model.parameters(),
    base_optimizer=torch.optim.Adam,
    lr=LEARNING_RATE,
    rho=SAM_RHO,
    adaptive=SAM_ADAPTIVE,
  )

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
      "Proceeding with the current (final-epoch) weights regardless.",
      MAX_EPOCHS,
    )

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
    "optimiser": "sam",
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "regularisation": "sam",
    "stopping_rule": f"patience={PATIENCE}, min_consecutive={MIN_CONSECUTIVE}, min_delta={MIN_DELTA}",
    "sam_rho": SAM_RHO,
    "sam_adaptive": SAM_ADAPTIVE,
    "data_source": "DeepCCDS authors' precomputed features (real, paper-verified — see precomputed_inputs.py)",
  }

  split_hash = hash_split(splits)
  baseline_ckpt = ARTIFACTS_DIR / "checkpoints" / "baseline"
  if (baseline_ckpt / "meta.json").exists():
    baseline_meta = peek_meta(baseline_ckpt)
    if baseline_meta.get("split_hash") != split_hash:
      raise ValueError(
        "Split hash mismatch against baseline checkpoint. "
        f"baseline={baseline_meta.get('split_hash')} sam={split_hash}"
      )

  save_checkpoint(
    model,
    CHECKPOINT_DIR,
    optimiser="sam",
    config_hash=cfg.config_hash,
    split_hash=split_hash,
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
      "rho": SAM_RHO,
      "adaptive": SAM_ADAPTIVE,
    },
  )

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
    preds_df.to_parquet(PREDICTIONS_DIR / f"sam_{split_name}.parquet")

  REPORTS_DIR.mkdir(parents=True, exist_ok=True)
  with open(REPORTS_DIR / "sam_gap_scorecard.json", "w", encoding="utf-8") as f:
    json.dump(scorecard, f, indent=2)
  with open(REPORTS_DIR / "sam_training_history.json", "w", encoding="utf-8") as f:
    json.dump([asdict(h) for h in history], f, indent=2)

  try:
    _plot_loss_curve(history, REPORTS_DIR / "sam_loss_curve.png")
  except Exception as exc:  # pragma: no cover
    logger.warning("Could not save loss-curve figure: %s", exc)

  logger.info("Stage 2 SAM complete. Scorecard: %s", json.dumps(scorecard, indent=2))

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
  ax.set_title("Stage 2 SAM (same architecture/data, SAM optimiser) — training curve")
  ax.legend()
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)


if __name__ == "__main__":
    train_sam()
