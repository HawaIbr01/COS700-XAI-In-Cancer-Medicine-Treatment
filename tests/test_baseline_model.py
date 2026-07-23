"""
Lightweight unit tests for the feat/baseline-model branch. These don't need real
GDSC data -- they check the architecture and training-loop logic in isolation with
synthetic tensors, so they can run in CI before the real dataset is wired up.
"""

import torch

from src.model.deepccds import DeepCCDS
from src.model.train_baseline import generalisation_gap, has_stabilised, r_squared, rmse


def test_deepccds_forward_shape_no_extra_inputs():
    model = DeepCCDS(n_genes=100)
    x = torch.randn(16, 100)
    out = model(x)
    assert out.shape == (16, 1)


def test_deepccds_forward_shape_with_mutation_and_drug():
    model = DeepCCDS(n_genes=100, mutation_dim=50, drug_dim=30)
    x = torch.randn(8, 100)
    mut = torch.randn(8, 50)
    drug = torch.randn(8, 30)
    out = model(x, mutation=mut, drug=drug)
    assert out.shape == (8, 1)


def test_deepccds_raises_if_mutation_missing():
    model = DeepCCDS(n_genes=100, mutation_dim=50)
    x = torch.randn(4, 100)
    try:
        model(x)
        assert False, "expected ValueError when mutation tensor is missing"
    except ValueError:
        pass


def test_generalisation_gap():
    assert generalisation_gap(val_loss=2.0, train_loss=0.5) == 1.5
    assert generalisation_gap(val_loss=0.5, train_loss=0.5) == 0.0


def test_has_stabilised_detects_plateau():
    # strictly decreasing -> not stabilised
    decreasing = [5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert not has_stabilised(decreasing, min_consecutive=10)

    # flat/increasing tail -> stabilised
    plateaued = [5.0] + [1.0] * 11
    assert has_stabilised(plateaued, min_consecutive=10)


def test_r2_and_rmse_perfect_prediction():
    y = torch.tensor([[1.0], [2.0], [3.0]])
    assert abs(r_squared(y, y) - 1.0) < 1e-6
    assert abs(rmse(y, y) - 0.0) < 1e-6
