"""
DeepCCDS architecture (Wu et al., 2025, Advanced Science).

OWNER: Angelique (feat/baseline-model)
Methodology: proposal §3.4.1, §4.2.1; paper's "Design of DeepCCDS", "Parameter
Setting of Mutation and Drug Fingerprint Autoencoders", "Parameter Setting of
Feedforward Neural Network".

CONTRACT (docs/CONTRACTS.md Arm 1 & Arm 2):
    This module is imported UNCHANGED by feat/sam-intervention (Bahiya). She
    should be able to do:

        from src.model.deepccds import DeepCCDS
        model = DeepCCDS(**arch_config)

    and get the exact same architecture the baseline was trained with, then
    swap only the optimiser for SAM. If the architecture changes, that's a
    breaking change to Arm 2 too — announce it.

Architecture (exact, cross-checked against the paper's Experimental Section):

    gene expression --[ssGSEA, a PREPROCESSING step, NOT inside this module]-->
        38-dim pathway ACTIVITY score, fed directly into the head (no extra
        learned layer on this branch — ssGSEA is not learned).

    mutation (534-dim binary, COSMIC CGC driver genes) --[autoencoder]--> 30-dim
    drug (1024-dim Morgan fingerprint)                 --[autoencoder]--> 30-dim

    concat(38 + 30 + 30 = 98-dim) --> feedforward head --> LN_IC50 (regression)

Autoencoder structure (both mutation and drug branches, per the paper's grid
search result — "Parameter Setting of Mutation and Drug Fingerprint Autoencoders"):
    encoder: input -> Linear(300) -> BatchNorm -> ReLU
                    -> Linear(100) -> BatchNorm -> ReLU
                    -> Linear(30)  -> BatchNorm -> ReLU   [bottleneck / embedding]
    decoder: mirror of the above, final layer -> Sigmoid (inputs are binary
             vectors — mutation calls, fingerprint bits — hence BCE reconstruction
             loss during standalone pretraining, and Sigmoid keeps outputs in
             [0, 1] to match).

Feedforward head structure ("Parameter Setting of Feedforward Neural Network"):
    input layer: concatenated embedding (98-dim when all three branches are
                 present; fewer if mutation/drug are disabled, e.g. for a
                 pathway-only ablation — see mutation_dim=0 / drug_dim=0 below).
    two hidden layers, EACH THE SAME SIZE AS THE INPUT LAYER, BatchNorm + ReLU.
    output layer: 1 neuron, NO activation (unrestricted LN_IC50 regression).

Training loss (paper's Eq. 2 — used for the *complete* end-to-end training,
as opposed to the BCE used only during standalone autoencoder pretraining):
    Loss = MSE(S, S') + MSE(R_C, R'_C) + MSE(R_D, R'_D)
    i.e. prediction MSE + mutation-autoencoder reconstruction MSE +
    drug-autoencoder reconstruction MSE, jointly minimised end-to-end (NOT a
    frozen-encoder setup). See train_baseline() in baseline.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_PATHWAYS = 38               # per DeepCCDS (Wu et al., 2025)
MUTATION_DIM = 534            # COSMIC CGC driver genes used for the mutation matrix
DRUG_DIM = 1024                # Morgan fingerprint bit-vector length
AE_HIDDEN_DIMS = (300, 100)   # autoencoder encoder hidden layer sizes (paper's grid search)
AE_BOTTLENECK_DIM = 30        # autoencoder embedding size (paper's grid search)


class Autoencoder(nn.Module):
    """Mutation / drug-fingerprint autoencoder, matching the paper's selected
    architecture exactly: encoder hidden dims (300, 100), bottleneck 30,
    BatchNorm + ReLU on every encoder/decoder layer, Sigmoid on the final
    decoder output (inputs are binary vectors, so Sigmoid + BCE pretraining
    is the natural fit — see pretrain_autoencoder()).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = AE_HIDDEN_DIMS,
        bottleneck_dim: int = AE_BOTTLENECK_DIM,
    ):
        super().__init__()

        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev_dim = h
        encoder_layers += [
            nn.Linear(prev_dim, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev_dim = bottleneck_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev_dim = h
        decoder_layers += [nn.Linear(prev_dim, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (embedding, reconstruction) — both needed for the combined
        loss (Eq. 2) during full DeepCCDS training."""
        z = self.encode(x)
        return z, self.decode(z)


def pretrain_autoencoder(
    model: Autoencoder,
    loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Autoencoder:
    """Optional standalone BCE pretraining for an Autoencoder branch (paper's
    grid-search stage). NOT required for train_baseline() — the end-to-end
    run in baseline.py trains everything jointly from scratch via Eq. 2 —
    but kept here in case the team wants to replicate the paper's two-stage
    (pretrain, then joint fine-tune) procedure exactly.

    `loader` should yield single-tensor batches `(x,)` of the raw binary input.
    """
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for epoch in range(1, epochs + 1):
        total_loss, n = 0.0, 0
        for (x,) in loader:
            x = x.to(device)
            _, recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [autoencoder pretrain] epoch {epoch}/{epochs} BCE={total_loss / n:.4f}")
    return model


class DeepCCDS(nn.Module):
    """Full DeepCCDS regressor: concatenates pathway activity (precomputed via
    ssGSEA, passed in directly — NOT computed inside this module) with the
    mutation and drug autoencoder embeddings, then feeds a feedforward head
    matching the paper's exact structure (two hidden layers sized to match
    the input layer, BatchNorm + ReLU, unactivated single-neuron output).

    mutation_dim / drug_dim default to the real paper values (534 / 1024).
    Set either to 0 to disable that branch (e.g. for a pathway-only ablation
    — the head input dim adjusts automatically) but the Stage 1 Fragility
    Baseline reported by this branch always uses the full 3-branch model.
    """

    def __init__(
        self,
        mutation_dim: int = MUTATION_DIM,
        drug_dim: int = DRUG_DIM,
        n_pathways: int = N_PATHWAYS,
        dropout: float = 0.0,   # kept at 0 for the unregularised Stage 1 baseline (§4.2.3)
    ):
        super().__init__()
        self.n_pathways = n_pathways
        self.mutation_dim = mutation_dim
        self.drug_dim = drug_dim
        self.dropout = dropout

        self.mutation_ae = Autoencoder(mutation_dim) if mutation_dim > 0 else None
        self.drug_ae = Autoencoder(drug_dim) if drug_dim > 0 else None

        head_input_dim = n_pathways
        head_input_dim += AE_BOTTLENECK_DIM if self.mutation_ae else 0
        head_input_dim += AE_BOTTLENECK_DIM if self.drug_ae else 0
        self.head_input_dim = head_input_dim

        # "two hidden layers each with the same number of neurons as the input layer"
        def hidden_block(dim: int) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            return layers

        self.head = nn.Sequential(
            *hidden_block(head_input_dim),
            *hidden_block(head_input_dim),
            nn.Linear(head_input_dim, 1),  # output layer: no activation (unrestricted LN_IC50)
        )

    def forward(
        self,
        pathway_activity: torch.Tensor,
        mutation: torch.Tensor | None = None,
        drug: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Returns a dict with 'prediction' (LN_IC50) plus, when the
        respective branch is active, 'mutation_recon'/'mutation_input' and
        'drug_recon'/'drug_input' so the caller can compute the full combined
        loss (Eq. 2) without re-running the autoencoders separately."""
        parts = [pathway_activity]
        out: dict[str, torch.Tensor] = {}

        if self.mutation_ae is not None:
            if mutation is None:
                raise ValueError("Model was built with mutation_dim > 0 but no mutation tensor was passed.")
            mut_embed, mut_recon = self.mutation_ae(mutation)
            parts.append(mut_embed)
            out["mutation_recon"] = mut_recon
            out["mutation_input"] = mutation

        if self.drug_ae is not None:
            if drug is None:
                raise ValueError("Model was built with drug_dim > 0 but no drug tensor was passed.")
            drug_embed, drug_recon = self.drug_ae(drug)
            parts.append(drug_embed)
            out["drug_recon"] = drug_recon
            out["drug_input"] = drug

        combined = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        out["prediction"] = self.head(combined).squeeze(-1)  # -> (batch,) continuous LN_IC50
        return out

    def arch_config(self) -> dict:
        """Constructor args, recorded in checkpoint meta.json (extra_meta) so
        SAM's run (Bahiya) can be confirmed to use an identical architecture."""
        return {
            "mutation_dim": self.mutation_dim,
            "drug_dim": self.drug_dim,
            "n_pathways": self.n_pathways,
            "dropout": self.dropout,
        }


def compute_loss(model_out: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """Paper's Eq. 2: Loss = MSE(S, S') + MSE(R_C, R'_C) + MSE(R_D, R'_D).
    The reconstruction terms are only included if the corresponding
    autoencoder branch is active (i.e. mutation/drug tensors were passed).
    Shared by baseline.py (Adam) and sam.py (SAM) so both optimise the
    identical objective — only the optimiser step differs.
    """
    loss = nn.functional.mse_loss(model_out["prediction"], y)
    if "mutation_recon" in model_out:
        loss = loss + nn.functional.mse_loss(model_out["mutation_recon"], model_out["mutation_input"])
    if "drug_recon" in model_out:
        loss = loss + nn.functional.mse_loss(model_out["drug_recon"], model_out["drug_input"])
    return loss
