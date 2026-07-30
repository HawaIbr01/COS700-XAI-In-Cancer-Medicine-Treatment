"""
DeepCCDS architecture, rebuilt against the paper's Experimental Section (Wu et al.,
2025, "Parameter Setting of Mutation and Drug Fingerprint Autoencoders" / "Parameter
Setting of Feedforward Neural Network" / "Complete Training and Evaluation").

CORRECTION vs the first-pass version of this file: pathway activity is NOT computed
by a learnable layer inside the network. The paper computes it as a PREPROCESSING
step -- gene expression is transformed into a 38-dim pathway ACTIVITY SCORE vector
via single-sample GSEA (ssGSEA) *before* it ever reaches DeepCCDS. The network's
input for the expression branch is simply that 38-dim vector, used directly (no
extra layer). This file has been corrected to reflect that -- `DeepCCDS.forward()`
now takes a `pathway_activity` tensor of shape (batch, 38) rather than raw gene
expression. See `src/data/pathway_activity.py` for the ssGSEA computation (TODO --
still needs the real 38-pathway gene-set definition, see that file's docstring).

Architecture (exact, from the paper):
    gene expression --[ssGSEA, done in preprocessing, NOT in this file]--> 38-dim pathway activity
                                                                                          \\
    mutation (534-dim binary, 534 COSMIC CGC driver genes) --[autoencoder]--> 30-dim  --- concat (98-dim) --> feedforward head --> LN_IC50
                                                                                          /
    drug (1024-dim Morgan fingerprint) --[autoencoder]--> 30-dim                    ---/

Autoencoder structure (both mutation and drug, per the paper's grid search result):
    encoder: input -> Linear(300) -> BatchNorm -> ReLU -> Linear(100) -> BatchNorm -> ReLU
             -> Linear(30) -> BatchNorm -> ReLU  [bottleneck/embedding]
    decoder: mirror of the above, final layer -> Sigmoid (inputs are binary, hence
             BCE reconstruction loss during autoencoder pretraining; Sigmoid keeps
             outputs in [0,1] to match).

Feedforward head structure (per the paper):
    input layer: concatenated embedding (38 + 30 + 30 = 98-dim if all three branches
                 are present -- fewer if mutation/drug aren't wired up yet, see below)
    two hidden layers, EACH THE SAME SIZE AS THE INPUT LAYER, with BatchNorm + ReLU
    output layer: 1 neuron, NO activation (unrestricted LN_IC50 regression)

Training loss (Eq. 2 in the paper, used for the *complete* end-to-end training stage,
as opposed to BCE used only during autoencoder pretraining):
    Loss = MSE(S, S') + MSE(R_C, R'_C) + MSE(R_D, R'_D)
    i.e. prediction MSE + mutation-autoencoder reconstruction MSE + drug-autoencoder
    reconstruction MSE, all three jointly minimised -- NOT a frozen-encoder setup.
    See train_baseline.py for where this combined loss is implemented.

IMPORTANT GAP STILL OPEN (flagging honestly): mutation_dim defaults to 534 and
drug_dim to 1024 to match the paper, but the actual 534-driver-gene mutation matrix
and the 1024-bit Morgan fingerprint drug matrix are NOT YET WIRED UP on our data
side (see preprocessing.py docstring). Until then, DeepCCDS can still be constructed
and trained with mutation_dim=0, drug_dim=0 to run on the pathway-activity signal
alone -- the head input dim adjusts automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_PATHWAYS = 38          # per DeepCCDS (Wu et al., 2025)
MUTATION_DIM = 534       # number of COSMIC CGC driver genes used for the mutation matrix
DRUG_DIM = 1024          # Morgan fingerprint bit-vector length
AE_HIDDEN_DIMS = (300, 100)  # autoencoder encoder hidden layer sizes, per the paper's grid search
AE_BOTTLENECK_DIM = 30       # autoencoder embedding size, per the paper's grid search


class Autoencoder(nn.Module):
    """Mutation / drug-fingerprint autoencoder, matching the paper's selected
    architecture exactly: encoder hidden dims (300, 100), bottleneck 30, BatchNorm
    + ReLU on every encoder/decoder layer, Sigmoid on the final decoder output
    (inputs are binary vectors -- mutation calls and fingerprint bits -- so a
    Sigmoid + BCE pretraining loss is the natural fit; see `pretrain_autoencoder()`).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = AE_HIDDEN_DIMS,
        bottleneck_dim: int = AE_BOTTLENECK_DIM,
    ):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev_dim = h
        encoder_layers += [nn.Linear(prev_dim, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
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
        """Returns (embedding, reconstruction) -- both needed for the combined
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
    """Pretrains an Autoencoder with BCE reconstruction loss (per the paper's
    'Parameter Setting of Mutation and Drug Fingerprint Autoencoders' -- pretraining
    uses BCE; the *joint* end-to-end DeepCCDS training then switches to MSE for the
    reconstruction terms per Eq. 2 -- both are implemented, use whichever stage
    you're in). `loader` should yield batches of the raw binary input tensor only.
    """
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n = 0
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
    ssGSEA, passed in directly -- NOT computed inside this module) with optional
    mutation/drug autoencoder embeddings, then feeds a feedforward head matching
    the paper's exact structure (two hidden layers sized to match the input layer,
    BatchNorm + ReLU, unactivated single-neuron output).
    """

    def __init__(
        self,
        mutation_dim: int = 0,   # 0 = mutation branch not wired up yet; set to MUTATION_DIM (534) once it is
        drug_dim: int = 0,       # 0 = drug branch not wired up yet; set to DRUG_DIM (1024) once it is
        n_pathways: int = N_PATHWAYS,
        dropout: float = 0.0,    # kept at 0 for the unregularised Stage 1 baseline (Section 4.2.3)
    ):
        super().__init__()
        self.n_pathways = n_pathways
        self.mutation_ae = Autoencoder(mutation_dim) if mutation_dim > 0 else None
        self.drug_ae = Autoencoder(drug_dim) if drug_dim > 0 else None

        head_input_dim = n_pathways
        head_input_dim += AE_BOTTLENECK_DIM if self.mutation_ae else 0
        head_input_dim += AE_BOTTLENECK_DIM if self.drug_ae else 0

        # "two hidden layers each with the same number of neurons as the input layer"
        def hidden_block(dim: int) -> list[nn.Module]:
            layers = [nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU()]
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
        """Returns a dict with 'prediction' (LN_IC50) plus, when the respective
        branch is active, 'mutation_recon'/'mutation_input' and 'drug_recon'/
        'drug_input' so the caller can compute the full combined loss (Eq. 2)
        without re-running the autoencoders separately."""
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
        out["prediction"] = self.head(combined)
        return out
