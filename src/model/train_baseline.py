"""
Stage 1: train the unregularised overfitting baseline (Section 4.2.3).

DeepCCDS is trained with plain Adam (lr=1e-3), no weight decay / dropout / other
regularisation. We deliberately do NOT stop at the point of best validation loss --
instead we use "early stopping in reverse": training keeps going with patience=30
purely to *confirm* the generalisation gap has stabilised (i.e. training loss keeps
falling while validation loss has plateaued or risen for >=10 consecutive epochs).
This intentionally-overfit checkpoint is the Fragility Baseline that Stage 2 (SAM,
feat/sam-intervention branch) is compared against.

UPDATED for the corrected DeepCCDS interface (see deepccds.py): the model now takes
a 38-dim `pathway_activity` tensor, not raw gene expression, per the paper. The real
pathway activity requires ssGSEA against the actual 38-KEGG-pathway gene sets, which
isn't built yet (see src/data/pathway_activity.py -- still a TODO). Until that's
ready, this script uses `temporary_pathway_proxy()` below -- a FIXED, seeded random
linear projection of the Z-scored gene expression down to 38 dims -- purely so the
full training loop is runnable and testable end-to-end. THIS IS NOT THE REAL
DEEPCCDS PATHWAY SIGNAL and results from it should not be reported as such. Swap in
real ssGSEA-derived pathway activity before this baseline is used for anything the
proposal will actually be evaluated on.

Loss follows Eq. 2 from the paper (MSE on the prediction, plus MSE reconstruction
terms for any active autoencoder branch) -- currently only the prediction term is
active since mutation_dim=drug_dim=0 until those branches are wired up.

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
from src.model.deepccds import DeepCCDS, N_PATHWAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")
LEARNING_RATE = 1e-3   # per the paper's grid search (Section: Parameter Setting of Feedforward NN)
BATCH_SIZE = 1024      # per the paper's grid search
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


class TemporaryPathwayProxy(nn.Module):
    """PLACEHOLDER ONLY -- NOT the real DeepCCDS pathway signal.

    Projects Z-scored gene expression down to N_PATHWAYS dims with a FIXED (seeded,
    non-trainable) random linear map, purely so the rest of the pipeline (model,
    training loop, overfitting diagnostics) is runnable and testable before the real
    ssGSEA-based pathway activity (src/data/pathway_activity.py) exists. Delete this
    class and feed real pathway activity into DeepCCDS once that's ready -- do not
    report generalisation-gap or fidelity results computed with this proxy as final.
    """

    def __init__(self, n_genes: int, n_pathways: int = N_PATHWAYS, seed: int = SEED):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        weight = torch.randn(n_genes, n_pathways, generator=g) / (n_genes ** 0.5)
        self.register_buffer("weight", weight)

    def forward(self, gene_expr: torch.Tensor) -> torch.Tensor:
        return gene_expr @ self.weight


def compute_loss(model_out: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """Eq. 2: Loss = MSE(S, S') + MSE(R_C, R'_C) + MSE(R_D, R'_D).
    The reconstruction terms are only included if the corresponding autoencoder
    branch is active (i.e. mutation/drug tensors were passed to the model)."""
    loss = nn.functional.mse_loss(model_out["prediction"], y)
    if "mutation_recon" in model_out:
        loss = loss + nn.functional.mse_loss(model_out["mutation_recon"], model_out["mutation_input"])
    if "drug_recon" in model_out:
        loss = loss + nn.functional.mse_loss(model_out["drug_recon"], model_out["drug_input"])
    return loss


def run_epoch(
    model: DeepCCDS,
    pathway_proxy: TemporaryPathwayProxy,
    loader: DataLoader,
    optimizer=None,
    device: str = "cpu",
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Runs one epoch. If `optimizer` is None, runs in eval mode with no grad."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_samples = 0.0, 0
    all_true, all_pred = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for gene_expr, y in loader:
            gene_expr, y = gene_expr.to(device), y.to(device)
            pathway_activity = pathway_proxy(gene_expr)  # TEMPORARY proxy -- see class docstring
            out = model(pathway_activity)  # mutation/drug branches not wired up yet -> not passed
            loss = compute_loss(out, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * gene_expr.size(0)
            n_samples += gene_expr.size(0)
            all_true.append(y.detach())
            all_pred.append(out["prediction"].detach())

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


def train_baseline_proxy(
    device: str = "cpu",
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict:
    """FALLBACK PATH -- uses TemporaryPathwayProxy (a random projection), NOT real
    pathway data. Kept for reference / for running the plumbing test without any
    precomputed data files. Prefer `train_baseline_real()` below, which uses the
    DeepCCDS authors' actual precomputed pathway/mutation/drug features and is a
    genuine (non-placeholder) Stage 1 baseline."""
    set_seed()

    train_ds = GDSCDataset("train")
    val_ds = GDSCDataset("val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    pathway_proxy = TemporaryPathwayProxy(n_genes=train_ds.n_features).to(device)
    model = DeepCCDS(mutation_dim=0, drug_dim=0, dropout=0.0).to(device)  # no regularisation for Stage 1

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # plain Adam, no weight decay

    history: list[EpochMetrics] = []
    val_loss_history: list[float] = []
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, pathway_proxy, train_loader, optimizer, device)
        val_loss, y_val_true, y_val_pred = run_epoch(model, pathway_proxy, val_loader, None, device)

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

    final_val_loss, y_val_true, y_val_pred = run_epoch(model, pathway_proxy, val_loader, None, device)
    final_train_loss, y_train_true, y_train_pred = run_epoch(model, pathway_proxy, train_loader, None, device)

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
        "WARNING": "pathway_activity is a TEMPORARY random-projection proxy, not real ssGSEA output",
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_dir / "baseline_adam.pt")
    with open(checkpoint_dir / "baseline_adam_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(checkpoint_dir / "baseline_adam_history.json", "w") as f:
        json.dump([asdict(h) for h in history], f, indent=2)

    logger.info("Stage 1 baseline complete. Summary: %s", json.dumps(summary, indent=2))
    return summary


def run_epoch_real(
    model: DeepCCDS,
    loader: DataLoader,
    optimizer=None,
    device: str = "cpu",
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Same as run_epoch(), but for DeepCCDSDataset's real three-branch batches
    (pathway, mutation, drug, y) instead of the TemporaryPathwayProxy path."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_samples = 0.0, 0
    all_true, all_pred = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for pathway, mutation, drug, y in loader:
            pathway, mutation, drug, y = (
                pathway.to(device), mutation.to(device), drug.to(device), y.to(device),
            )
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

    avg_loss = total_loss / n_samples
    return avg_loss, torch.cat(all_true), torch.cat(all_pred)


def train_baseline_real(
    device: str = "cpu",
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict:
    """REAL Stage 1 baseline training path (Section 4.2.3), using the DeepCCDS
    authors' own precomputed pathway/mutation/drug features (see
    src/data/precomputed_inputs.py) via DeepCCDSDataset -- NOT the
    TemporaryPathwayProxy placeholder. This is a genuine, defensible baseline:
    real 38-dim ssGSEA pathway activity, real 534-gene mutation calls, real
    1024-bit drug fingerprints, the full three-branch architecture with Eq. 2
    combined loss, all matching the paper's exact specs.

    TEAM DECISION FLAGGED: this uses the DeepCCDS authors' precomputed features
    rather than independently re-deriving them from raw GDSC files -- confirm
    with the course supervisors this is an acceptable scoping choice before
    reporting results from this path as final (see precomputed_inputs.py docstring).
    """
    from src.data.dataset import DeepCCDSDataset

    set_seed()

    train_ds = DeepCCDSDataset("train")
    val_ds = DeepCCDSDataset("val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepCCDS(mutation_dim=534, drug_dim=1024, dropout=0.0).to(device)  # full 3-branch, no regularisation
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history: list[EpochMetrics] = []
    val_loss_history: list[float] = []
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _ = run_epoch_real(model, train_loader, optimizer, device)
        val_loss, y_val_true, y_val_pred = run_epoch_real(model, val_loader, None, device)

        gap = generalisation_gap(val_loss, train_loss)
        history.append(EpochMetrics(epoch, train_loss, val_loss, gap))
        val_loss_history.append(val_loss)

        logger.info(
            "epoch %3d | train_loss=%.4f val_loss=%.4f gen_gap=%.4f",
            epoch, train_loss, val_loss, gap,
        )

        best_state = model.state_dict()

        if has_stabilised(val_loss_history):
            logger.info(
                "Generalisation gap stabilised after %d epochs. Stopping.", epoch,
            )
            break
    else:
        logger.warning("Reached MAX_EPOCHS=%d without a confirmed stabilised gap.", MAX_EPOCHS)

    final_val_loss, y_val_true, y_val_pred = run_epoch_real(model, val_loader, None, device)
    final_train_loss, y_train_true, y_train_pred = run_epoch_real(model, train_loader, None, device)

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
        "data_source": "DeepCCDS authors' precomputed features (real, not placeholder)",
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_dir / "baseline_adam_real.pt")
    with open(checkpoint_dir / "baseline_adam_real_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(checkpoint_dir / "baseline_adam_real_history.json", "w") as f:
        json.dump([asdict(h) for h in history], f, indent=2)

    logger.info("REAL Stage 1 baseline complete. Summary: %s", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train_baseline_real()
