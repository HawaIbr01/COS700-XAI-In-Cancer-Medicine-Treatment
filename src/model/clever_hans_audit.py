"""
Clever Hans attribution sanity check (§4.2.4).

OWNER: Angelique (feat/baseline-model)

Not a competing rule-extraction method — a lightweight, qualitative audit
(input-gradient attribution, the simplest LRP-adjacent technique per §4.2.4:
"a lightweight LRP implementation alongside vanilla input gradients") run on
the trained baseline to sanity-check whether the unregularised model's
reliance on the 38 pathway-activity features looks diffuse/inconsistent
between train and validation samples — the qualitative signature of Clever
Hans behaviour this project is diagnosing (§3.4.2, §4.2.4).

This audits the PATHWAY branch only (the 38 ssGSEA activity features), since
that is the branch with human-readable, per-feature names (pathway names) an
audit reader can reason about. The mutation (534-dim) and drug (1024-dim)
branches are not included in this particular check.

Produces:
    artifacts/reports/baseline_clever_hans_audit.json
    artifacts/reports/baseline_clever_hans_attribution.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import DeepCCDSDataset
from src.data.precomputed_inputs import load_pathway_activity, DEFAULT_PRECOMPUTED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DATA_DIR = ARTIFACTS_DIR / "data"

N_SAMPLE_ROWS = 256   # per split — enough for a stable mean/variance without being slow


def _pathway_names() -> list[str]:
    try:
        _, _, names = load_pathway_activity(DEFAULT_PRECOMPUTED_DIR)
        return names
    except FileNotFoundError:
        return [f"pathway_{i}" for i in range(38)]


def _input_gradients(model, loader, device: str, n_samples: int) -> np.ndarray:
    """Returns (n_used, 38) array of d(prediction)/d(pathway_activity), one
    row per sampled cell-drug pair. Mutation/drug branches are held fixed
    (gradients not required through them for this check)."""
    model.eval()
    grads = []
    seen = 0
    for row_id, pathway, mutation, drug, y in loader:
        pathway = pathway.to(device).clone().requires_grad_(True)
        mutation = mutation.to(device)
        drug = drug.to(device)

        out = model(pathway, mutation=mutation, drug=drug)
        prediction = out["prediction"].sum()
        grad = torch.autograd.grad(prediction, pathway, retain_graph=False)[0]
        grads.append(grad.detach().cpu().numpy())

        seen += pathway.shape[0]
        if seen >= n_samples:
            break

    return np.concatenate(grads, axis=0)[:n_samples]


def run_clever_hans_audit(model, device: str = "cpu", condition: str = "baseline") -> dict:
    """Runs the audit against the already-trained model passed in from
    baseline.py (avoids re-loading the checkpoint from disk). Can also be run
    standalone (`python -m src.model.clever_hans_audit`) against the saved
    checkpoint."""
    from torch.utils.data import DataLoader

    pathway_names = _pathway_names()

    train_loader = DataLoader(DeepCCDSDataset("train"), batch_size=64, shuffle=True)
    val_loader = DataLoader(DeepCCDSDataset("val"), batch_size=64, shuffle=True)

    train_grads = _input_gradients(model, train_loader, device, N_SAMPLE_ROWS)
    val_grads = _input_gradients(model, val_loader, device, N_SAMPLE_ROWS)

    def summarise(grads: np.ndarray) -> dict:
        mean_abs = np.abs(grads).mean(axis=0)          # per-pathway mean |gradient|
        std_abs = np.abs(grads).std(axis=0)
        # sample-to-sample inconsistency: coefficient of variation per pathway,
        # then averaged — high values flag pathways whose importance swings
        # wildly between individual predictions (§4.2.4's "high sample-to-sample
        # inconsistency" signature of Clever Hans behaviour).
        cv = std_abs / (mean_abs + 1e-8)
        ranked = sorted(zip(pathway_names, mean_abs.tolist()), key=lambda t: -t[1])
        return {
            "top_10_pathways_by_mean_abs_gradient": ranked[:10],
            "mean_coefficient_of_variation": float(cv.mean()),
            "per_pathway_mean_abs_gradient": dict(zip(pathway_names, mean_abs.tolist())),
        }

    train_summary = summarise(train_grads)
    val_summary = summarise(val_grads)

    audit = {
        "n_samples_per_split": N_SAMPLE_ROWS,
        "method": "vanilla input gradients (lightweight LRP-adjacent attribution, §4.2.4)",
        "train": train_summary,
        "val": val_summary,
        "interpretation_note": (
            "This is a QUALITATIVE sanity check, not a formal Clever Hans test. "
            "Read it alongside the generalisation-gap scorecard in "
            "baseline_gap_scorecard.json: a wide gap plus high "
            "mean_coefficient_of_variation (attribution swings a lot between "
            "individual val predictions) plus reliance on pathways with no "
            "obvious oncogenic relevance is the pattern §3.4.2 describes as "
            "evidence of shortcut learning. This does not by itself prove or "
            "disprove Clever Hans behaviour -- report it descriptively."
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{condition}_clever_hans"
    with open(REPORTS_DIR / f"{prefix}_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    try:
        _plot_attribution(
            train_summary,
            val_summary,
            REPORTS_DIR / f"{prefix}_attribution.png",
            condition=condition,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not save attribution figure: %s", exc)

    logger.info(
        "Clever Hans audit complete for %s. train CV=%.3f val CV=%.3f (see %s)",
        condition,
        train_summary["mean_coefficient_of_variation"], val_summary["mean_coefficient_of_variation"],
        f"{prefix}_audit.json",
    )
    return audit


def _plot_attribution(train_summary: dict, val_summary: dict, out_path: Path, condition: str = "baseline") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = [name for name, _ in train_summary["top_10_pathways_by_mean_abs_gradient"]]
    train_vals = [train_summary["per_pathway_mean_abs_gradient"][n] for n in top]
    val_vals = [val_summary["per_pathway_mean_abs_gradient"][n] for n in top]

    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y - 0.2, train_vals, height=0.4, label="train", color="#1f77b4")
    ax.barh(y + 0.2, val_vals, height=0.4, label="val", color="#d62728")
    ax.set_yticks(y)
    ax.set_yticklabels(top, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean |d(prediction)/d(pathway activity)|")
    ax.set_title(f"Top-10 pathway attributions — {condition}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from src.common.checkpoint import load_checkpoint
    from src.model.deepccds import DeepCCDS

    ckpt_dir = ARTIFACTS_DIR / "checkpoints" / "baseline"
    meta = None
    model = DeepCCDS(dropout=0.0)
    meta = load_checkpoint(model, ckpt_dir)
    logger.info("Loaded baseline checkpoint: %s", meta)
    run_clever_hans_audit(model, device="cpu")
