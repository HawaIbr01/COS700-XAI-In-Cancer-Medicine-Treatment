"""
DeepCCDS-inspired architecture (Wu et al., 2025 -- see references.bib) for predicting
LN(IC50) drug sensitivity from gene expression, mutation, and drug structure inputs.

IMPORTANT: this is a first-pass implementation of the architecture as described in the
proposal (Section 3.4.1): a prior-knowledge pathway network (38 signalling pathways)
replacing raw high-dimensional gene expression, plus self-supervised autoencoders
compressing mutation and drug-structure inputs into low-dimensional embeddings.
The exact layer sizes / pathway gene membership from the original DeepCCDS paper
should be substituted once we've pulled the gene-pathway mapping (e.g. from KEGG/
Reactome) -- see `build_pathway_mask()` below for where that plugs in. Flagging this
so nobody assumes these numbers are final.

Architecture overview:
    gene expression (Z-scored) --[pathway mask]--> 38 pathway activations --MLP-->
                                                                                    \
    mutation vector            --[autoencoder encoder]--> mutation embedding ------ concat --> MLP head --> LN_IC50
                                                                                    /
    drug descriptor vector     --[autoencoder encoder]--> drug embedding    ------/
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_PATHWAYS = 38  # per DeepCCDS (Wu et al., 2025)


def build_pathway_mask(gene_to_pathway: dict[str, list[int]], gene_names: list[str]) -> torch.Tensor:
    """Build a (n_genes, N_PATHWAYS) binary mask from a gene -> pathway-index mapping,
    so the first layer of the pathway branch only connects each gene to the pathways
    it biologically belongs to, rather than being fully dense.

    `gene_to_pathway` example: {"EGFR": [2, 7], "TP53": [0, 5, 12], ...}
    Placeholder until we source the real KEGG/Reactome mapping used by DeepCCDS.
    """
    mask = torch.zeros(len(gene_names), N_PATHWAYS)
    for i, gene in enumerate(gene_names):
        for pathway_idx in gene_to_pathway.get(gene, []):
            mask[i, pathway_idx] = 1.0
    return mask


class PathwayLayer(nn.Module):
    """Masked linear layer: gene expression -> pathway activations. Only edges present
    in `mask` are learnable; everything else is hard-zeroed via the mask, encoding the
    prior biological knowledge that a gene only contributes to pathways it belongs to.
    """

    def __init__(self, n_genes: int, mask: torch.Tensor | None = None):
        super().__init__()
        self.linear = nn.Linear(n_genes, N_PATHWAYS)
        if mask is None:
            # Fallback: no prior-knowledge mapping supplied yet -> fully dense.
            # Replace with a real mask via build_pathway_mask() once available.
            mask = torch.ones(n_genes, N_PATHWAYS)
        self.register_buffer("mask", mask.T)  # (N_PATHWAYS, n_genes) to match weight shape
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.linear.weight * self.mask
        out = nn.functional.linear(x, masked_weight, self.linear.bias)
        return self.activation(out)


class SimpleAutoencoder(nn.Module):
    """Self-supervised autoencoder used to compress mutation / drug-structure inputs
    into a low-dimensional embedding (Section 3.4.1). Train with `.reconstruct()` +
    MSE loss during a pretraining pass, then use `.encode()` as a frozen feature
    extractor inside DeepCCDS.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


class DeepCCDS(nn.Module):
    """Full DeepCCDS-style regressor. Predicts LN(IC50) from gene expression
    (via the pathway layer) optionally combined with mutation / drug embeddings.

    For the Stage 1 baseline (this branch), `mutation_dim` and `drug_dim` can be
    left at 0 to train purely on the pathway-transformed expression signal if the
    mutation/drug tables aren't wired up yet -- the head gracefully handles a
    variable-size concatenated input.
    """

    def __init__(
        self,
        n_genes: int,
        pathway_mask: torch.Tensor | None = None,
        mutation_dim: int = 0,
        drug_dim: int = 0,
        mutation_embed_dim: int = 32,
        drug_embed_dim: int = 32,
        head_hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.0,  # kept at 0 for the unregularised Stage 1 baseline (Section 4.2.3)
    ):
        super().__init__()
        self.pathway_layer = PathwayLayer(n_genes, pathway_mask)

        self.mutation_ae = SimpleAutoencoder(mutation_dim, mutation_embed_dim) if mutation_dim > 0 else None
        self.drug_ae = SimpleAutoencoder(drug_dim, drug_embed_dim) if drug_dim > 0 else None

        head_input_dim = N_PATHWAYS
        head_input_dim += mutation_embed_dim if self.mutation_ae else 0
        head_input_dim += drug_embed_dim if self.drug_ae else 0

        layers = []
        prev_dim = head_input_dim
        for h in head_hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        gene_expr: torch.Tensor,
        mutation: torch.Tensor | None = None,
        drug: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pathway_act = self.pathway_layer(gene_expr)
        parts = [pathway_act]

        if self.mutation_ae is not None:
            if mutation is None:
                raise ValueError("Model was built with mutation_dim > 0 but no mutation tensor was passed.")
            parts.append(self.mutation_ae.encode(mutation))

        if self.drug_ae is not None:
            if drug is None:
                raise ValueError("Model was built with drug_dim > 0 but no drug tensor was passed.")
            parts.append(self.drug_ae.encode(drug))

        combined = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.head(combined)
