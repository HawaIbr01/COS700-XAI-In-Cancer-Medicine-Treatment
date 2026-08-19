"""
src/common/checkpoint.py — Branch Contracts §0.3.

NSGA-II doesn't load the model itself (only predictions/*.parquet), but
checkpoint files must exist on disk for schema validation and for
Evaluation to find them later. torch is optional here: if it isn't
installed, model.pt is written as an empty placeholder file rather than a
real state dict — fine for this arm, since nothing here ever calls
load_checkpoint on it. Replace with a real torch.save once Arm 1/2 land.
"""

import json
from pathlib import Path


def save_checkpoint(model, meta: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        torch.save(model, out_dir / "model.pt")
    except ImportError:
        (out_dir / "model.pt").write_bytes(b"")

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_meta(checkpoint_dir) -> dict:
    with open(Path(checkpoint_dir) / "meta.json") as f:
        return json.load(f)


def load_checkpoint(checkpoint_dir):
    """Not used by NSGA-II — kept for contract completeness. Requires torch."""
    import torch
    return torch.load(Path(checkpoint_dir) / "model.pt", weights_only=False)
