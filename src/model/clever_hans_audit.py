"""
Clever Hans sanity check for the trained baseline (Section 4.2.4).

Generates local feature-attribution maps (vanilla input gradients -- a lightweight
stand-in for full LRP, per the proposal's "lightweight LRP implementation alongside
vanilla input gradients") for a sample of training and validation predictions, then
reports which genes the model relies on most, propagated back through the
(currently temporary/proxy) pathway transform to the original gene-expression inputs.
This is a qualitative diagnostic, not a statistical test.

NOTE: attributions are computed w.r.t. the ORIGINAL gene expression input, gradient
flows through TemporaryPathwayProxy -> DeepCCDS end-to-end, so this still works
once the proxy is swapped for real ssGSEA-derived pathway activity (as long as that
step stays differentiable, or is precomputed per-sample so gradients start there
instead -- see the TODO at the bottom of this file).

Usage:
    python -m src.model.clever_hans_audit
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import GDSCDataset
from src.model.deepccds import DeepCCDS
from src.model.train_baseline import TemporaryPathwayProxy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path("checkpoints/baseline_adam.pt")
AUDIT_OUT_PATH = Path("checkpoints/clever_hans_audit.json")

# Genes with no documented oncogenic role that DeepCCDS/the baseline should NOT be
# relying heavily on. Placeholder list -- replace with the actual housekeeping-gene
# reference set the team settles on (e.g. from Eisenberg & Levanon, 2013).
HOUSEKEEPING_GENES = {"GENE_ACTB", "GENE_GAPDH", "GENE_TUBB", "GENE_B2M", "GENE_RPL13A"}


def input_gradients(
    model: torch.nn.Module,
    pathway_proxy: torch.nn.Module,
    X: torch.Tensor,
) -> torch.Tensor:
    """Vanilla input-gradient saliency: d(output)/d(gene_expression) for each sample,
    computed end-to-end through the pathway proxy and the DeepCCDS head."""
    X = X.clone().requires_grad_(True)
    model.eval()
    pathway_proxy.eval()
    pathway_activity = pathway_proxy(X)
    out = model(pathway_activity)
    out["prediction"].sum().backward()
    return X.grad.detach()


def attribution_summary(
    model: torch.nn.Module,
    pathway_proxy: torch.nn.Module,
    dataset: GDSCDataset,
    n_samples: int = 50,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    X = torch.stack([dataset[i][0] for i in idx])

    grads = input_gradients(model, pathway_proxy, X)  # (n_samples, n_genes)
    abs_grads = grads.abs()

    mean_importance = abs_grads.mean(dim=0)                # per-gene average |gradient|
    sample_to_sample_std = abs_grads.std(dim=0)             # instability across samples

    ranked = torch.argsort(mean_importance, descending=True)
    top_k = ranked[:20].tolist()

    top_genes = [dataset.feature_cols[i] for i in top_k]
    housekeeping_hits = [g for g in top_genes if g in HOUSEKEEPING_GENES]

    return {
        "top_20_genes_by_mean_abs_gradient": top_genes,
        "n_housekeeping_genes_in_top_20": len(housekeeping_hits),
        "housekeeping_genes_flagged": housekeeping_hits,
        "mean_sample_to_sample_std_top20": sample_to_sample_std[top_k].mean().item(),
        "n_samples_used": len(idx),
    }


def run_audit(checkpoint_path: Path = CHECKPOINT_PATH) -> dict:
    train_ds = GDSCDataset("train")
    val_ds = GDSCDataset("val")

    pathway_proxy = TemporaryPathwayProxy(n_genes=train_ds.n_features)
    model = DeepCCDS(mutation_dim=0, drug_dim=0, dropout=0.0)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    train_summary = attribution_summary(model, pathway_proxy, train_ds)
    val_summary = attribution_summary(model, pathway_proxy, val_ds)

    report = {
        "train_attribution": train_summary,
        "val_attribution": val_summary,
        "WARNING": (
            "Attributions flow through TemporaryPathwayProxy, not real ssGSEA "
            "pathway activity -- re-run this audit once the real pathway step "
            "(src/data/pathway_activity.py) is wired in before drawing conclusions."
        ),
        "interpretation_notes": (
            "Compare top genes across train/val: a large non-overlap, or a high "
            "housekeeping-gene count, or high sample-to-sample instability on "
            "validation, is qualitative evidence of Clever Hans behaviour "
            "(Section 4.2.4). This is a sanity check, not a statistical claim -- "
            "record it narratively in the baseline scorecard."
        ),
    }

    AUDIT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Clever Hans audit report:\n%s", json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_audit()
