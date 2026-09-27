"""
Lightweight unit tests for the feat/baseline-model branch. These don't need real
GDSC data -- they check the architecture and training-loop logic in isolation with
synthetic tensors, so they can run in CI before the real dataset is wired up.
"""

import torch

from src.model.deepccds import DeepCCDS, N_PATHWAYS, Autoencoder
from src.model.train_baseline import (
    TemporaryPathwayProxy, compute_loss, generalisation_gap, has_stabilised,
    r_squared, rmse,
)


def test_deepccds_forward_shape_pathway_only():
    model = DeepCCDS(mutation_dim=0, drug_dim=0)
    pathway_activity = torch.randn(16, N_PATHWAYS)
    out = model(pathway_activity)
    assert out["prediction"].shape == (16,)
    assert "mutation_recon" not in out
    assert "drug_recon" not in out


def test_deepccds_forward_shape_with_mutation_and_drug():
    model = DeepCCDS(mutation_dim=50, drug_dim=30)
    pathway_activity = torch.randn(8, N_PATHWAYS)
    mut = torch.rand(8, 50)   # rand (0-1) since mutation/fingerprint inputs are binary-ish
    drug = torch.rand(8, 30)
    out = model(pathway_activity, mutation=mut, drug=drug)
    assert out["prediction"].shape == (8,)
    assert out["mutation_recon"].shape == mut.shape
    assert out["drug_recon"].shape == drug.shape


def test_deepccds_raises_if_mutation_missing():
    model = DeepCCDS(mutation_dim=50, drug_dim=0)
    pathway_activity = torch.randn(4, N_PATHWAYS)
    try:
        model(pathway_activity)
        assert False, "expected ValueError when mutation tensor is missing"
    except ValueError:
        pass


def test_autoencoder_roundtrip_shapes():
    ae = Autoencoder(input_dim=100)
    x = torch.rand(8, 100)
    z, recon = ae(x)
    assert z.shape == (8, 30)  # AE_BOTTLENECK_DIM
    assert recon.shape == (8, 100)
    assert (recon >= 0).all() and (recon <= 1).all()  # sigmoid output


def test_temporary_pathway_proxy_shape():
    proxy = TemporaryPathwayProxy(n_genes=200)
    x = torch.randn(10, 200)
    out = proxy(x)
    assert out.shape == (10, N_PATHWAYS)


def test_compute_loss_prediction_only():
    out = {"prediction": torch.tensor([[1.0], [2.0]])}
    y = torch.tensor([[1.0], [2.0]])
    loss = compute_loss(out, y)
    assert abs(loss.item() - 0.0) < 1e-6


def test_compute_loss_includes_reconstruction_terms():
    out = {
        "prediction": torch.tensor([[1.0]]),
        "mutation_recon": torch.tensor([[0.5, 0.5]]),
        "mutation_input": torch.tensor([[1.0, 0.0]]),
    }
    y = torch.tensor([[1.0]])
    loss = compute_loss(out, y)
    assert loss.item() > 0.0  # reconstruction mismatch contributes to the loss


def test_generalisation_gap():
    assert generalisation_gap(val_loss=2.0, train_loss=0.5) == 1.5
    assert generalisation_gap(val_loss=0.5, train_loss=0.5) == 0.0


def test_has_stabilised_detects_plateau():
    decreasing = [5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert not has_stabilised(decreasing, min_consecutive=10)

    plateaued = [5.0] + [1.0] * 11
    assert has_stabilised(plateaued, min_consecutive=10)


def test_r2_and_rmse_perfect_prediction():
    y = torch.tensor([[1.0], [2.0], [3.0]])
    assert abs(r_squared(y, y) - 1.0) < 1e-6
    assert abs(rmse(y, y) - 0.0) < 1e-6
