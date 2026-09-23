"""
Stage 2: train DeepCCDS with SAM (Sharpness-Aware Minimization).

OWNER: Bahiya (feat/sam-intervention)

Methodology:
    §4.5   SAM regularisation, Stage 2
    §4.5.1 Comparison contract
    §4.5.2 SAM mechanics

CONTRACT:
    Arm 2 must use the identical:
        - architecture
        - data
        - preprocessing
        - fixed splits
        - loss
        - shared hyperparameters
        - stopping criteria

    compared with baseline.py.

    The ONLY methodological change is:
        Adam -> SAM-wrapped Adam

    SAM introduces one additional hyperparameter:
        rho

    Shared hyperparameters are imported from baseline.py instead of
    being redefined here.

Usage:
    python -m src.model.sam
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common.checkpoint import (
    hash_split,
    load_checkpoint,
    peek_meta,
    save_checkpoint,
)
from src.common.config import load_config
from src.data.dataset import DeepCCDSDataset
from src.model.deepccds import DeepCCDS, compute_loss

# --------------------------------------------------------------------------- #
# Shared baseline hyperparameters.
#
# IMPORTANT:
# These are imported from baseline.py rather than redefined here.
# This prevents the SAM and baseline arms from silently diverging.
# --------------------------------------------------------------------------- #
from src.model.baseline import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_CONSECUTIVE,
    MIN_DELTA,
    PATIENCE,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

DATA_DIR = ARTIFACTS_DIR / "data"

CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "sam"
BASELINE_CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "baseline"

PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


# --------------------------------------------------------------------------- #
# SAM-specific hyperparameter
#
# This is intentionally the ONLY new training hyperparameter introduced by
# the SAM arm.
# --------------------------------------------------------------------------- #

RHO = 0.05


# =========================================================================== #
# SAM OPTIMISER
# =========================================================================== #

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization optimizer wrapper.

    SAM performs two forward/backward passes per batch:

        1. Calculate gradient at theta.
        2. Perturb parameters to theta + epsilon.
        3. Calculate gradient at theta + epsilon.
        4. Restore theta.
        5. Apply the base optimizer update using the perturbed gradient.

    The base optimizer is Adam, preserving the baseline optimizer while
    adding SAM's neighbourhood-aware update.

    Usage:

        loss = ...
        loss.backward()

        optimizer.first_step(zero_grad=True)

        loss = ...
        loss.backward()

        optimizer.second_step(zero_grad=True)
    """

    def __init__(
        self,
        params,
        base_optimizer_cls=torch.optim.Adam,
        rho: float = RHO,
        **base_kwargs,
    ):
        if rho < 0:
            raise ValueError(
                f"SAM rho must be non-negative, got {rho}"
            )

        # Materialise parameters once so we can safely inspect them.
        params = list(params)

        if not params:
            raise ValueError(
                "SAM received no parameters to optimise."
            )

        # Create the parameter groups used by SAM.
        defaults = {
            "rho": rho,
            **base_kwargs,
        }

        super().__init__(params, defaults)

        if not self.param_groups:
            raise ValueError(
                "SAM received no parameter groups."
            )

        # The base optimizer operates on the same parameter groups.
        self.base_optimizer = base_optimizer_cls(
            self.param_groups,
            **base_kwargs,
        )

        # Keep the SAM optimizer's groups synchronised with the base
        # optimizer's groups.
        self.param_groups = self.base_optimizer.param_groups

        # Store rho explicitly on every parameter group.
        for group in self.param_groups:
            group["rho"] = rho

    # ------------------------------------------------------------------ #
    # First SAM step: move from theta -> theta + epsilon
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()

        if grad_norm.item() == 0.0:
            logger.warning(
                "SAM first_step encountered zero gradient; "
                "skipping perturbation."
            )

            if zero_grad:
                self.zero_grad()

            return

        for group in self.param_groups:
            rho = group["rho"]

            scale = rho / (grad_norm + 1e-12)

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                epsilon = parameter.grad * scale.to(parameter)

                parameter.add_(epsilon)

                self.state[parameter]["e_w"] = epsilon

        if zero_grad:
            self.zero_grad()

    # ------------------------------------------------------------------ #
    # Second SAM step: theta + epsilon -> theta -> Adam update
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        any_reverted = False

        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter)

                if state is None:
                    continue

                if "e_w" not in state:
                    continue

                epsilon = state.pop("e_w")

                parameter.sub_(epsilon)

                any_reverted = True

        if not any_reverted:
            logger.warning(
                "SAM second_step called without a pending perturbation."
            )

        # The gradient currently stored on each parameter was calculated
        # at theta + epsilon. The parameters have now been restored to
        # theta, so Adam applies that gradient at the original weights.
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    # ------------------------------------------------------------------ #
    # Calculate global L2 norm of all gradients
    # ------------------------------------------------------------------ #

    def _grad_norm(self) -> torch.Tensor:
        gradients = []

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradients.append(
                    parameter.grad.norm(p=2)
                )

        if not gradients:
            return torch.tensor(0.0)

        device = gradients[0].device

        return torch.norm(
            torch.stack(
                [
                    gradient.to(device)
                    for gradient in gradients
                ]
            ),
            p=2,
        )

    # ------------------------------------------------------------------ #
    # SAM does not use a normal single-step interface.
    # ------------------------------------------------------------------ #

    def step(self, closure=None):
        raise RuntimeError(
            "SAM requires first_step() and second_step(). "
            "Do not call optimizer.step() directly."
        )


# =========================================================================== #
# BATCH NORMALISATION HANDLING
# =========================================================================== #

@contextmanager
def _disabled_running_stats(model: nn.Module):
    """
    Prevent BatchNorm running statistics from being updated during SAM's
    second forward pass.

    First forward:
        theta
        BatchNorm running statistics update normally.

    Second forward:
        theta + epsilon
        BatchNorm running statistics are NOT updated.

    This prevents statistics generated by the perturbed weights from
    contaminating the model after those weights are restored.
    """

    batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.BatchNorm1d)
    ]

    previous_values = [
        layer.track_running_stats
        for layer in batch_norm_layers
    ]

    for layer in batch_norm_layers:
        layer.track_running_stats = False

    try:
        yield

    finally:
        for layer, previous_value in zip(
            batch_norm_layers,
            previous_values,
        ):
            layer.track_running_stats = previous_value


# =========================================================================== #
# EPOCH METRICS
# =========================================================================== #

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    generalisation_gap: float


# =========================================================================== #
# REPRODUCIBILITY
# =========================================================================== #

def set_seed(seed: int) -> None:
    """
    Set the same random seeds used by the baseline arm.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================== #
# METRICS
# =========================================================================== #

def generalisation_gap(
    val_loss: float,
    train_loss: float,
) -> float:
    """
    Generalisation gap:

        validation loss - training loss
    """

    return val_loss - train_loss


def r_squared(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """
    Calculate R².
    """

    ss_res = torch.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = torch.sum(
        (y_true - y_true.mean()) ** 2
    )

    if ss_tot.item() <= 0:
        return float("nan")

    return (
        1 - ss_res / ss_tot
    ).item()


def rmse(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """
    Calculate RMSE.
    """

    return torch.sqrt(
        torch.mean(
            (y_true - y_pred) ** 2
        )
    ).item()


# =========================================================================== #
# STOPPING RULE
# =========================================================================== #

class StoppingTracker:
    """
    Same stopping rule as baseline.py.

    IMPORTANT:
    SAM must not introduce a different stopping rule.
    """

    def __init__(
        self,
        patience: int = PATIENCE,
        min_consecutive: int = MIN_CONSECUTIVE,
        min_delta: float = MIN_DELTA,
    ):
        self.patience = patience
        self.min_consecutive = min_consecutive
        self.min_delta = min_delta

        self.best_val_loss = float("inf")

        self.no_improve_streak = 0

        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    def update(
        self,
        train_loss: float,
        val_loss: float,
    ) -> None:

        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.no_improve_streak = 0

        else:
            self.no_improve_streak += 1

    def stabilised(self) -> bool:
        """
        Same stopping criterion as baseline.py.

        Stop when:

        1. Validation loss has failed to improve for PATIENCE epochs.
        2. Training loss is still falling over MIN_CONSECUTIVE epochs.
        """

        if self.no_improve_streak < self.patience:
            return False

        if len(self.train_loss_history) < (
            self.min_consecutive + 1
        ):
            return False

        train_still_falling = (
            self.train_loss_history[-1]
            <
            self.train_loss_history[
                -(self.min_consecutive + 1)
            ]
        )

        return train_still_falling


# =========================================================================== #
# EVALUATION
# =========================================================================== #

def run_epoch_eval(
    model: DeepCCDS,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[
    float,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    """
    Evaluation pass used for validation and final train/validation/test
    scorecards.
    """

    model.eval()

    total_loss = 0.0
    n_samples = 0

    all_true = []
    all_pred = []
    all_row_ids = []

    with torch.no_grad():

        for (
            row_id,
            pathway,
            mutation,
            drug,
            y,
        ) in loader:

            pathway = pathway.to(device)
            mutation = mutation.to(device)
            drug = drug.to(device)
            y = y.to(device)

            output = model(
                pathway,
                mutation=mutation,
                drug=drug,
            )

            loss = compute_loss(
                output,
                y,
            )

            batch_size = pathway.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            n_samples += batch_size

            all_true.append(
                y.detach()
            )

            all_pred.append(
                output["prediction"].detach()
            )

            if torch.is_tensor(row_id):
                all_row_ids.append(
                    row_id.cpu().numpy()
                )
            else:
                all_row_ids.append(
                    np.asarray(row_id)
                )

    if n_samples == 0:
        raise RuntimeError(
            "Evaluation loader contained zero samples."
        )

    average_loss = total_loss / n_samples

    return (
        average_loss,
        torch.cat(all_true),
        torch.cat(all_pred),
        np.concatenate(all_row_ids),
    )


# =========================================================================== #
# SAM TRAINING EPOCH
# =========================================================================== #

def run_epoch_train_sam(
    model: DeepCCDS,
    loader: DataLoader,
    optimizer: SAM,
    device: str = "cpu",
) -> float:
    """
    Train for one epoch using SAM.

    For every batch:

        PASS 1
            theta
            forward
            backward
            theta -> theta + epsilon

        PASS 2
            theta + epsilon
            forward
            backward

        UPDATE
            theta + epsilon -> theta
            Adam update at theta using perturbed gradient
    """

    model.train()

    total_loss = 0.0
    n_samples = 0

    for (
        row_id,
        pathway,
        mutation,
        drug,
        y,
    ) in loader:

        pathway = pathway.to(device)
        mutation = mutation.to(device)
        drug = drug.to(device)
        y = y.to(device)

        # --------------------------------------------------------------- #
        # PASS 1: gradient at theta
        # --------------------------------------------------------------- #

        output = model(
            pathway,
            mutation=mutation,
            drug=drug,
        )

        loss = compute_loss(
            output,
            y,
        )

        loss.backward()

        optimizer.first_step(
            zero_grad=True
        )

        # --------------------------------------------------------------- #
        # PASS 2: gradient at theta + epsilon
        #
        # BatchNorm running statistics are disabled here.
        # --------------------------------------------------------------- #

        with _disabled_running_stats(model):

            output_perturbed = model(
                pathway,
                mutation=mutation,
                drug=drug,
            )

            perturbed_loss = compute_loss(
                output_perturbed,
                y,
            )

        perturbed_loss.backward()

        # --------------------------------------------------------------- #
        # Restore theta and perform Adam update
        # --------------------------------------------------------------- #

        optimizer.second_step(
            zero_grad=True
        )

        # Report the loss at the original theta.
        # This preserves the same training-loss semantics as baseline.py.
        batch_size = pathway.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        n_samples += batch_size

    if n_samples == 0:
        raise RuntimeError(
            "Training loader contained zero samples."
        )

    return total_loss / n_samples


# =========================================================================== #
# RESPONSE CLASSIFICATION
# =========================================================================== #

def _class_from_threshold(
    ln_ic50: np.ndarray,
    threshold: float,
    sensitive: str,
    resistant: str,
) -> np.ndarray:
    """
    Convert predicted ln(IC50) values to response classes using the exact
    threshold from response_classes.json.
    """

    return np.where(
        ln_ic50 < threshold,
        sensitive,
        resistant,
    )


# =========================================================================== #
# BASELINE COMPARISON
# =========================================================================== #

def _verify_against_baseline(
    cfg,
    split_hash: str,
) -> dict:
    """
    Verify that the SAM run is directly comparable with the baseline.

    Checks:

        - baseline checkpoint exists
        - config hash matches
        - split hash matches
        - baseline optimiser is Adam
        - baseline recorded architecture exists
    """

    if not BASELINE_CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"No baseline checkpoint found at "
            f"{BASELINE_CHECKPOINT_DIR}.\n\n"
            f"SAM must be trained against the existing baseline "
            f"checkpoint so the two arms are guaranteed comparable."
        )

    baseline_meta = peek_meta(
        BASELINE_CHECKPOINT_DIR
    )

    # --------------------------------------------------------------- #
    # Configuration
    # --------------------------------------------------------------- #

    if baseline_meta.get("config_hash") != cfg.config_hash:
        raise RuntimeError(
            "config_hash mismatch between SAM and baseline.\n"
            "The pipeline configuration differs from the configuration "
            "used to train the baseline.\n"
            "Aborting to protect the comparison contract."
        )

    # --------------------------------------------------------------- #
    # Splits
    # --------------------------------------------------------------- #

    if baseline_meta.get("split_hash") != split_hash:
        raise RuntimeError(
            "split_hash mismatch between SAM and baseline.\n"
            "The data splits do not match.\n"
            "Aborting to protect the comparison contract."
        )

    # --------------------------------------------------------------- #
    # Optimiser
    # --------------------------------------------------------------- #

    if baseline_meta.get("optimiser") != "adam":
        raise RuntimeError(
            "The baseline checkpoint does not contain an Adam run.\n"
            f"Found optimiser="
            f"{baseline_meta.get('optimiser')!r}.\n"
            "Expected optimiser='adam'."
        )

    # --------------------------------------------------------------- #
    # Architecture
    # --------------------------------------------------------------- #

    if "arch_config" not in baseline_meta:
        raise RuntimeError(
            "Baseline checkpoint meta.json does not contain "
            "'arch_config'. SAM cannot guarantee architectural "
            "identity without it."
        )

    return baseline_meta


# =========================================================================== #
# ARCHITECTURE ROUND-TRIP VALIDATION
# =========================================================================== #

def _verify_arch_round_trips(
    arch_config: dict,
    model: DeepCCDS,
) -> None:
    """
    Confirm that DeepCCDS reconstructs exactly the architecture recorded
    by the baseline checkpoint.
    """

    reconstructed = model.arch_config()

    if reconstructed != arch_config:
        raise RuntimeError(
            "Architecture round-trip mismatch.\n\n"
            f"Baseline arch_config:\n{arch_config}\n\n"
            f"Reconstructed arch_config:\n{reconstructed}\n\n"
            "The DeepCCDS constructor is not faithfully reproducing "
            "the baseline architecture. Aborting."
        )


# =========================================================================== #
# MAIN SAM TRAINING
# =========================================================================== #

def train_sam(
    device: str | None = None,
    rho: float = RHO,
) -> dict:
    """
    Complete Stage 2 SAM training pipeline.
    """

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #

    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        "Stage 2 SAM training using device: %s",
        device,
    )

    # ------------------------------------------------------------------ #
    # Configuration and seed
    # ------------------------------------------------------------------ #

    cfg = load_config()

    set_seed(
        cfg.seed
    )

    logger.info(
        "Random seed: %s",
        cfg.seed,
    )

    logger.info(
        "SAM rho: %.4f",
        rho,
    )

    logger.info(
        "Learning rate inherited from baseline: %s",
        LEARNING_RATE,
    )

    logger.info(
        "Batch size inherited from baseline: %s",
        BATCH_SIZE,
    )

    # ------------------------------------------------------------------ #
    # Validate data artifacts
    # ------------------------------------------------------------------ #

    processed_data = (
        DATA_DIR / "gdsc_processed.parquet"
    )

    splits_file = (
        DATA_DIR / "splits.json"
    )

    response_classes_file = (
        DATA_DIR / "response_classes.json"
    )

    if not processed_data.exists():
        raise FileNotFoundError(
            f"{processed_data} not found.\n"
            "Run the preprocessing stage first or pull the baseline "
            "data artifacts."
        )

    if not splits_file.exists():
        raise FileNotFoundError(
            f"{splits_file} not found."
        )

    if not response_classes_file.exists():
        raise FileNotFoundError(
            f"{response_classes_file} not found."
        )

    # ------------------------------------------------------------------ #
    # Load exact fixed splits
    # ------------------------------------------------------------------ #

    with open(
        splits_file,
        "r",
        encoding="utf-8",
    ) as file:
        splits = json.load(file)

    split_hash = hash_split(
        splits
    )

    logger.info(
        "Split hash: %s",
        split_hash,
    )

    # ------------------------------------------------------------------ #
    # Load response-class metadata
    # ------------------------------------------------------------------ #

    with open(
        response_classes_file,
        "r",
        encoding="utf-8",
    ) as file:
        response_classes = json.load(file)

    threshold = response_classes[
        "threshold"
    ]

    sensitive_label = response_classes[
        "classes"
    ][0]

    resistant_label = response_classes[
        "classes"
    ][1]

    # ------------------------------------------------------------------ #
    # Verify baseline compatibility
    # ------------------------------------------------------------------ #

    baseline_meta = _verify_against_baseline(
        cfg,
        split_hash,
    )

    arch_config = baseline_meta[
        "arch_config"
    ]

    logger.info(
        "Baseline architecture verified."
    )

    logger.info(
        "Using baseline arch_config: %s",
        arch_config,
    )

    # ------------------------------------------------------------------ #
    # Load datasets
    # ------------------------------------------------------------------ #

    train_dataset = DeepCCDSDataset(
        "train"
    )

    val_dataset = DeepCCDSDataset(
        "val"
    )

    test_dataset = DeepCCDSDataset(
        "test"
    )

    # ------------------------------------------------------------------ #
    # DataLoaders
    #
    # Same batch size and split structure as baseline.
    # ------------------------------------------------------------------ #

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    logger.info(
        "Dataset sizes: train=%d val=%d test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    # ------------------------------------------------------------------ #
    # Build exact baseline architecture
    # ------------------------------------------------------------------ #

    model = DeepCCDS(
        **arch_config
    )

    _verify_arch_round_trips(
        arch_config,
        model,
    )

    model = model.to(
        device
    )

    logger.info(
        "DeepCCDS architecture matches baseline checkpoint."
    )

    # ------------------------------------------------------------------ #
    # SAM-wrapped Adam
    #
    # LEARNING_RATE comes from baseline.py.
    # rho is the only new SAM-specific hyperparameter.
    # ------------------------------------------------------------------ #

    optimizer = SAM(
        model.parameters(),
        base_optimizer_cls=torch.optim.Adam,
        rho=rho,
        lr=LEARNING_RATE,
    )

    # ------------------------------------------------------------------ #
    # Training state
    # ------------------------------------------------------------------ #

    tracker = StoppingTracker()

    history: list[EpochMetrics] = []

    final_epoch = 0

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_loss = run_epoch_train_sam(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_loss, _, _, _ = run_epoch_eval(
            model,
            val_loader,
            device,
        )

        gap = generalisation_gap(
            val_loss,
            train_loss,
        )

        tracker.update(
            train_loss,
            val_loss,
        )

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                generalisation_gap=gap,
            )
        )

        final_epoch = epoch

        logger.info(
            "epoch %3d | train_loss=%.6f | "
            "val_loss=%.6f | gap=%.6f | "
            "no_improve=%d",
            epoch,
            train_loss,
            val_loss,
            gap,
            tracker.no_improve_streak,
        )

        # -------------------------------------------------------------- #
        # Same stopping rule as baseline
        # -------------------------------------------------------------- #

        if tracker.stabilised():

            logger.info(
                "Stopping rule fired at epoch %d.",
                epoch,
            )

            logger.info(
                "Validation loss has failed to improve for %d "
                "epochs while training loss is still declining.",
                tracker.no_improve_streak,
            )

            break

    else:

        logger.warning(
            "Reached MAX_EPOCHS=%d without the stopping criterion firing.",
            MAX_EPOCHS,
        )

    # ================================================================== #
    # FINAL SCORECARD
    # ================================================================== #

    logger.info(
        "Running final train/validation/test evaluation..."
    )

    train_loss, y_train_true, y_train_pred, train_row_ids = (
        run_epoch_eval(
            model,
            train_loader,
            device,
        )
    )

    val_loss, y_val_true, y_val_pred, val_row_ids = (
        run_epoch_eval(
            model,
            val_loader,
            device,
        )
    )

    test_loss, y_test_true, y_test_pred, test_row_ids = (
        run_epoch_eval(
            model,
            test_loader,
            device,
        )
    )

    gap = generalisation_gap(
        val_loss,
        train_loss,
    )

    scorecard = {
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "final_test_loss": test_loss,

        "generalisation_gap": gap,

        "train_r2": r_squared(
            y_train_true,
            y_train_pred,
        ),

        "val_r2": r_squared(
            y_val_true,
            y_val_pred,
        ),

        "test_r2": r_squared(
            y_test_true,
            y_test_pred,
        ),

        "train_rmse": rmse(
            y_train_true,
            y_train_pred,
        ),

        "val_rmse": rmse(
            y_val_true,
            y_val_pred,
        ),

        "test_rmse": rmse(
            y_test_true,
            y_test_pred,
        ),

        "n_epochs": final_epoch,

        # -------------------------------------------------------------- #
        # Optimisation metadata
        # -------------------------------------------------------------- #

        "optimiser": "sam",
        "base_optimiser": "adam",

        "rho": rho,

        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,

        "regularisation": "sam",

        "stopping_rule": (
            f"patience={PATIENCE}, "
            f"min_consecutive={MIN_CONSECUTIVE}, "
            f"min_delta={MIN_DELTA}"
        ),

        "data_source": (
            "DeepCCDS authors' precomputed features "
            "(real, paper-verified)"
        ),
    }

    # ================================================================== #
    # SAVE SAM CHECKPOINT
    # ================================================================== #

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

            "rho": rho,

            "train_r2": scorecard["train_r2"],
            "test_r2": scorecard["test_r2"],

            "train_rmse": scorecard["train_rmse"],
            "test_rmse": scorecard["test_rmse"],

            "response_threshold": threshold,

            "base_optimiser": "adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
        },
    )

    logger.info(
        "SAM checkpoint saved to %s",
        CHECKPOINT_DIR,
    )

    # ================================================================== #
    # SAVE PREDICTIONS
    # ================================================================== #

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_sets = (
        (
            "train",
            train_row_ids,
            y_train_pred,
        ),
        (
            "val",
            val_row_ids,
            y_val_pred,
        ),
        (
            "test",
            test_row_ids,
            y_test_pred,
        ),
    )

    for (
        split_name,
        row_ids,
        predictions,
    ) in prediction_sets:

        predictions_np = (
            predictions
            .detach()
            .cpu()
            .numpy()
        )

        predictions_df = pd.DataFrame(
            {
                "row_id": row_ids,

                "ln_ic50_pred": predictions_np,

                "class_pred": _class_from_threshold(
                    predictions_np,
                    threshold,
                    sensitive_label,
                    resistant_label,
                ),
            }
        )

        output_path = (
            PREDICTIONS_DIR
            / f"sam_{split_name}.parquet"
        )

        predictions_df.to_parquet(
            output_path,
            index=False,
        )

        logger.info(
            "Saved %s predictions to %s",
            split_name,
            output_path,
        )

    # ================================================================== #
    # SAVE REPORTS
    # ================================================================== #

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scorecard_path = (
        REPORTS_DIR
        / "sam_gap_scorecard.json"
    )

    with open(
        scorecard_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            scorecard,
            file,
            indent=2,
        )

    history_path = (
        REPORTS_DIR
        / "sam_training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [
                asdict(metrics)
                for metrics in history
            ],
            file,
            indent=2,
        )

    logger.info(
        "Saved SAM scorecard to %s",
        scorecard_path,
    )

    logger.info(
        "Saved SAM training history to %s",
        history_path,
    )

    # ================================================================== #
    # COMPARATIVE TRAINING CURVE
    # ================================================================== #

    try:

        _plot_comparative_training_curve(
            history,
            REPORTS_DIR
            / "baseline_training_history.json",
            REPORTS_DIR
            / "sam_vs_baseline_loss_curve.png",
        )

    except Exception as exc:

        logger.warning(
            "Could not generate comparative training curve: %s",
            exc,
        )

    # ================================================================== #
    # COMPARATIVE LOSS LANDSCAPE
    # ================================================================== #

    try:

        baseline_model = _load_baseline_model(
            arch_config,
            device,
        )

        _plot_comparative_loss_landscape(
            sam_model=model,
            baseline_model=baseline_model,
            loader=val_loader,
            device=device,
            out_path=(
                REPORTS_DIR
                / "sam_vs_baseline_loss_landscape.png"
            ),
        )

    except Exception as exc:

        logger.warning(
            "Could not generate comparative loss landscape: %s",
            exc,
        )

    # ================================================================== #
    # LOG FINAL RESULTS
    # ================================================================== #

    logger.info(
        "============================================================"
    )

    logger.info(
        "Stage 2 SAM training complete."
    )

    logger.info(
        "Final scorecard:\n%s",
        json.dumps(
            scorecard,
            indent=2,
        ),
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "Comparison with baseline:"
    )

    logger.info(
        "Validation R²: %.6f -> %.6f",
        baseline_meta.get(
            "r2",
            float("nan"),
        ),
        scorecard["val_r2"],
    )

    logger.info(
        "Generalisation gap: %.6f -> %.6f",
        baseline_meta.get(
            "generalisation_gap",
            float("nan"),
        ),
        gap,
    )

    # ================================================================== #
    # CLEVER HANS AUDIT
    # ================================================================== #

    try:

        from src.model.clever_hans_audit import (
            run_clever_hans_audit,
        )

        run_clever_hans_audit(
            model,
            device=device,
            condition="sam",
        )

    except Exception as exc:

        logger.warning(
            "Clever Hans audit failed: %s",
            exc,
        )

    return scorecard


# =========================================================================== #
# LOAD BASELINE MODEL
# =========================================================================== #

def _load_baseline_model(
    arch_config: dict,
    device: str,
) -> DeepCCDS:
    """
    Load the trained baseline model using the exact architecture recorded
    in the baseline checkpoint metadata.
    """

    model = DeepCCDS(
        **arch_config
    )

    _verify_arch_round_trips(
        arch_config,
        model,
    )

    load_checkpoint(
        model,
        BASELINE_CHECKPOINT_DIR,
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# =========================================================================== #
# COMPARATIVE TRAINING CURVE
# =========================================================================== #

def _plot_comparative_training_curve(
    sam_history: list[EpochMetrics],
    baseline_history_path: Path,
    out_path: Path,
) -> None:
    """
    Plot SAM and baseline train/validation losses on the same figure.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    sam_epochs = [
        metrics.epoch
        for metrics in sam_history
    ]

    sam_train = [
        metrics.train_loss
        for metrics in sam_history
    ]

    sam_val = [
        metrics.val_loss
        for metrics in sam_history
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    ax.plot(
        sam_epochs,
        sam_train,
        label="SAM train",
    )

    ax.plot(
        sam_epochs,
        sam_val,
        label="SAM validation",
    )

    # --------------------------------------------------------------- #
    # Baseline curve
    # --------------------------------------------------------------- #

    if baseline_history_path.exists():

        with open(
            baseline_history_path,
            "r",
            encoding="utf-8",
        ) as file:

            baseline_history = json.load(
                file
            )

        baseline_epochs = [
            item["epoch"]
            for item in baseline_history
        ]

        baseline_train = [
            item["train_loss"]
            for item in baseline_history
        ]

        baseline_val = [
            item["val_loss"]
            for item in baseline_history
        ]

        ax.plot(
            baseline_epochs,
            baseline_train,
            label="baseline train",
            linestyle="--",
            alpha=0.6,
        )

        ax.plot(
            baseline_epochs,
            baseline_val,
            label="baseline validation",
            linestyle="--",
            alpha=0.6,
        )

    else:

        logger.warning(
            "Baseline training history not found at %s",
            baseline_history_path,
        )

    ax.set_xlabel(
        "epoch"
    )

    ax.set_ylabel(
        "loss (Eq. 2, MSE-based)"
    )

    ax.set_title(
        "Stage 1 (Adam) vs Stage 2 (SAM) — training curves"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=150,
    )

    plt.close(fig)

    logger.info(
        "Saved comparative training curve to %s",
        out_path,
    )


# =========================================================================== #
# COMPARATIVE LOSS LANDSCAPE
# =========================================================================== #

def _plot_comparative_loss_landscape(
    sam_model: DeepCCDS,
    baseline_model: DeepCCDS,
    loader: DataLoader,
    device: str,
    out_path: Path,
    n_points: int = 21,
    span: float = 1.0,
) -> None:
    """
    Generate a 1-D filter-normalised loss landscape comparison.

    Both models use the SAME underlying random direction.

    The direction is then filter-normalised separately for each model,
    following the filter-normalisation idea used by Li et al. (2018).

    This provides a like-for-like comparison of the local loss geometry
    around the trained baseline and SAM solutions.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------ #
    # Evaluation helper
    # ------------------------------------------------------------------ #

    def eval_loss(
        model: DeepCCDS,
    ) -> float:

        model.eval()

        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():

            for (
                row_id,
                pathway,
                mutation,
                drug,
                y,
            ) in loader:

                pathway = pathway.to(device)
                mutation = mutation.to(device)
                drug = drug.to(device)
                y = y.to(device)

                output = model(
                    pathway,
                    mutation=mutation,
                    drug=drug,
                )

                loss = compute_loss(
                    output,
                    y,
                )

                batch_size = pathway.size(0)

                total_loss += (
                    loss.item()
                    * batch_size
                )

                n_samples += batch_size

        if n_samples == 0:
            raise RuntimeError(
                "Cannot calculate loss landscape: "
                "validation loader contains zero samples."
            )

        return total_loss / n_samples

    # ------------------------------------------------------------------ #
    # Filter-normalise direction
    # ------------------------------------------------------------------ #

    def filter_normalise(
        model: DeepCCDS,
        base_direction: list[torch.Tensor],
    ) -> list[torch.Tensor]:

        direction = []

        for parameter, base_d in zip(
            model.parameters(),
            base_direction,
        ):

            direction_tensor = base_d.clone()

            direction_tensor.mul_(
                parameter.norm()
                /
                (
                    direction_tensor.norm()
                    + 1e-12
                )
            )

            direction.append(
                direction_tensor
            )

        return direction

    # ------------------------------------------------------------------ #
    # Evaluate one model's landscape
    # ------------------------------------------------------------------ #

    def landscape(
        model: DeepCCDS,
        base_direction: list[torch.Tensor],
    ) -> tuple[
        list[float],
        list[float],
    ]:

        direction = filter_normalise(
            model,
            base_direction,
        )

        original_parameters = [
            parameter.detach().clone()
            for parameter in model.parameters()
        ]

        alphas = np.linspace(
            -span,
            span,
            n_points,
        )

        losses = []

        with torch.no_grad():

            for alpha in alphas:

                for (
                    parameter,
                    original,
                    direction_tensor,
                ) in zip(
                    model.parameters(),
                    original_parameters,
                    direction,
                ):

                    parameter.copy_(
                        original
                        + alpha * direction_tensor
                    )

                losses.append(
                    eval_loss(model)
                )

            # ---------------------------------------------------------- #
            # Always restore original trained weights.
            # ---------------------------------------------------------- #

            for (
                parameter,
                original,
            ) in zip(
                model.parameters(),
                original_parameters,
            ):

                parameter.copy_(
                    original
                )

        return (
            list(alphas),
            losses,
        )

    # ------------------------------------------------------------------ #
    # Generate ONE shared base direction
    # ------------------------------------------------------------------ #

    base_direction = [
        torch.randn_like(parameter)
        for parameter in sam_model.parameters()
    ]

    # ------------------------------------------------------------------ #
    # SAM landscape
    # ------------------------------------------------------------------ #

    sam_alphas, sam_losses = landscape(
        sam_model,
        base_direction,
    )

    # ------------------------------------------------------------------ #
    # Baseline landscape
    # ------------------------------------------------------------------ #

    baseline_alphas, baseline_losses = landscape(
        baseline_model,
        base_direction,
    )

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    ax.plot(
        baseline_alphas,
        baseline_losses,
        label="baseline (Adam)",
    )

    ax.plot(
        sam_alphas,
        sam_losses,
        label="SAM",
    )

    ax.set_xlabel(
        "perturbation magnitude "
        "(alpha, shared filter-normalised direction)"
    )

    ax.set_ylabel(
        "loss (Eq. 2, MSE-based)"
    )

    ax.set_title(
        "Loss landscape around trained weights — baseline vs SAM"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=150,
    )

    plt.close(fig)

    logger.info(
        "Saved comparative loss landscape to %s",
        out_path,
    )


# =========================================================================== #
# ENTRY POINT
# =========================================================================== #

if __name__ == "__main__":
    train_sam()