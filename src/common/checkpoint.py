"""
Save/load model checkpoints and their meta.json sidecars.

Contract (docs/CONTRACTS.md §0.1, Arm 1 & Arm 2):
    artifacts/checkpoints/{baseline|sam}/{model.pt, meta.json}

    meta.json is validated against src.common.schema before being
    written, so a checkpoint can never be saved without the fields
    everyone downstream depends on (optimiser, config_hash,
    split_hash, gap metrics).

    torch is imported lazily inside the functions that need it, so
    that arms which only need to *read* meta.json (e.g. Evaluation
    checking generalisation_gap) don't need torch installed to use
    this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.common.schema import load_checkpoint_meta, save_checkpoint_meta


def hash_split(splits: dict) -> str:
    """Deterministic hash of a splits.json dict, used as split_hash in
    checkpoint meta so two checkpoints can be confirmed to share the
    exact same train/val/test partition."""
    encoded = json.dumps(
        {k: sorted(v) for k, v in splits.items()}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def save_checkpoint(
    model: Any,
    out_dir: str | Path,
    *,
    optimiser: str,
    config_hash: str,
    split_hash: str,
    epoch: int,
    generalisation_gap: float,
    r2: float,
    rmse: float,
    extra_meta: dict | None = None,
) -> None:
    """
    Save a model checkpoint + validated meta.json.

    Args:
        model: a torch.nn.Module (or anything with .state_dict()).
        out_dir: e.g. artifacts/checkpoints/baseline/
        optimiser: "adam" or "sam" — see schema.VALID_OPTIMISERS.
        config_hash: from src.common.config.load_config().config_hash
        split_hash: from hash_split(splits) — must match across
            baseline/SAM checkpoints, since both are trained on the
            identical split (docs/CONTRACTS.md Arm 2).
        epoch: stopping epoch (per the early-stopping-in-reverse
            criterion, methodology §4.2.3).
        generalisation_gap, r2, rmse: Stage 1/2 scorecard values.
        extra_meta: optimiser-specific extras, e.g. {"rho": 0.05} for SAM.
    """
    import torch  # lazy import — only needed when actually saving

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")

    meta = {
        "optimiser": optimiser,
        "config_hash": config_hash,
        "split_hash": split_hash,
        "epoch": epoch,
        "generalisation_gap": generalisation_gap,
        "r2": r2,
        "rmse": rmse,
        **(extra_meta or {}),
    }
    save_checkpoint_meta(meta, out_dir / "meta.json")


def load_checkpoint(model: Any, in_dir: str | Path) -> dict:
    """
    Load a checkpoint's weights into `model` in-place and return its
    validated meta.json as a dict.

    Args:
        model: an already-constructed model instance (e.g. a fresh
            DeepCCDS()) whose state_dict will be overwritten.
        in_dir: e.g. artifacts/checkpoints/sam/

    Returns:
        the checkpoint's meta dict (optimiser, config_hash, split_hash,
        epoch, generalisation_gap, r2, rmse, ...).
    """
    import torch  # lazy import

    in_dir = Path(in_dir)
    state_dict = torch.load(in_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    return load_checkpoint_meta(in_dir / "meta.json")


def peek_meta(in_dir: str | Path) -> dict:
    """Read only meta.json without touching torch or model.pt — useful
    for arms (e.g. Evaluation) that just need the gap scorecard."""
    return load_checkpoint_meta(Path(in_dir) / "meta.json")
