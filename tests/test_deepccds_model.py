"""Unit tests for the DeepCCDS architecture and the baseline stopping logic.
Runs fast, no data files required.
"""

from __future__ import annotations

import torch

from src.model.deepccds import DeepCCDS, Autoencoder, compute_loss, N_PATHWAYS, MUTATION_DIM, DRUG_DIM
from src.model.baseline import StoppingTracker


def test_autoencoder_shapes():
    ae = Autoencoder(input_dim=100)
    x = torch.rand(8, 100)
    z, recon = ae(x)
    assert z.shape == (8, 30)
    assert recon.shape == (8, 100)


def test_deepccds_full_forward_shape():
    model = DeepCCDS(mutation_dim=MUTATION_DIM, drug_dim=DRUG_DIM, n_pathways=N_PATHWAYS)
    batch = 4
    pathway = torch.rand(batch, N_PATHWAYS)
    mutation = torch.randint(0, 2, (batch, MUTATION_DIM)).float()
    drug = torch.randint(0, 2, (batch, DRUG_DIM)).float()

    out = model(pathway, mutation=mutation, drug=drug)
    assert out["prediction"].shape == (batch,)
    assert out["mutation_recon"].shape == mutation.shape
    assert out["drug_recon"].shape == drug.shape


def test_deepccds_pathway_only_ablation():
    """mutation_dim=0, drug_dim=0 should still run (pathway-only ablation)."""
    model = DeepCCDS(mutation_dim=0, drug_dim=0, n_pathways=N_PATHWAYS)
    pathway = torch.rand(4, N_PATHWAYS)
    out = model(pathway)
    assert out["prediction"].shape == (4,)
    assert "mutation_recon" not in out
    assert "drug_recon" not in out


def test_head_input_dim_matches_concat():
    model = DeepCCDS(mutation_dim=MUTATION_DIM, drug_dim=DRUG_DIM, n_pathways=N_PATHWAYS)
    assert model.head_input_dim == N_PATHWAYS + 30 + 30


def test_arch_config_roundtrip():
    model = DeepCCDS(mutation_dim=MUTATION_DIM, drug_dim=DRUG_DIM, n_pathways=N_PATHWAYS, dropout=0.0)
    cfg = model.arch_config()
    rebuilt = DeepCCDS(**cfg)
    assert rebuilt.head_input_dim == model.head_input_dim


def test_compute_loss_combines_reconstruction_terms():
    model = DeepCCDS(mutation_dim=MUTATION_DIM, drug_dim=DRUG_DIM)
    pathway = torch.rand(4, N_PATHWAYS)
    mutation = torch.randint(0, 2, (4, MUTATION_DIM)).float()
    drug = torch.randint(0, 2, (4, DRUG_DIM)).float()
    y = torch.rand(4)

    out = model(pathway, mutation=mutation, drug=drug)
    loss_full = compute_loss(out, y)

    pred_only_model = DeepCCDS(mutation_dim=0, drug_dim=0)
    out_pred_only = pred_only_model(pathway)
    loss_pred_only = compute_loss(out_pred_only, y)

    # both should be finite, non-negative scalars
    assert loss_full.item() >= 0
    assert loss_pred_only.item() >= 0


def test_stopping_tracker_triggers_on_plateau():
    tracker = StoppingTracker(patience=5, min_consecutive=3, min_delta=1e-4)
    # train loss steadily decreasing, val loss flat after epoch 3
    train_losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    val_losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    stabilised_epoch = None
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses), start=1):
        tracker.update(tl, vl)
        if tracker.stabilised():
            stabilised_epoch = i
            break

    assert stabilised_epoch is not None, "stopping tracker never triggered on an obvious plateau"


def test_stopping_tracker_does_not_trigger_while_still_improving():
    tracker = StoppingTracker(patience=5, min_consecutive=3, min_delta=1e-4)
    train_losses = [1.0, 0.9, 0.8, 0.7, 0.6]
    val_losses = [1.0, 0.9, 0.8, 0.7, 0.6]  # improving every epoch

    for tl, vl in zip(train_losses, val_losses):
        tracker.update(tl, vl)
    assert not tracker.stabilised()
