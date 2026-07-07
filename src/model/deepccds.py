"""
DeepCCDS architecture (Wu et al., 2025).

OWNER: Angelique (feat/baseline-model)
Methodology: §3.4.1, §4.2.1

CONTRACT (docs/CONTRACTS.md Arm 1 & Arm 2):
    This module is imported UNCHANGED by feat/sam-intervention.
    Bahiya's arm must be able to do:

        from src.model.deepccds import DeepCCDS
        model = DeepCCDS(**arch_config)

    and get the exact same architecture the baseline was trained
    with. If the architecture needs to change, that's a breaking
    change to Arm 2 as well — announce it.

TODO (Angelique):
    - Implement the pathway-informed prior-knowledge network (38
      signalling pathways, §3.4.1) in place of raw high-dimensional
      gene expression input.
    - Implement the self-supervised autoencoder branch for gene
      mutation / drug structure embeddings.
    - Regression head -> continuous LN_IC50 prediction.
    - Keep constructor args config-driven (no hardcoded dims) so
      arch_config can be recorded in checkpoint meta.json and
      reproduced exactly by SAM's training run.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepCCDS(nn.Module):
    """
    Pathway-informed drug-sensitivity regressor.

    Args:
        n_pathways: number of biologically meaningful signalling
            pathway features (38 per Wu et al., 2025).
        mutation_embed_dim: dimensionality of the compressed gene
            mutation autoencoder embedding.
        drug_embed_dim: dimensionality of the compressed drug
            structure autoencoder embedding.
        hidden_dims: hidden layer sizes for the prediction head.

    Forward:
        Input: pathway features + mutation embedding + drug embedding.
        Output: a single continuous LN_IC50 prediction per row.
    """

    def __init__(
        self,
        n_pathways: int = 38,
        mutation_embed_dim: int = 64,
        drug_embed_dim: int = 64,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()
        self.n_pathways = n_pathways
        self.mutation_embed_dim = mutation_embed_dim
        self.drug_embed_dim = drug_embed_dim
        self.hidden_dims = hidden_dims

        # TODO: replace this placeholder head with the real
        # pathway-network + autoencoder architecture from §3.4.1.
        input_dim = n_pathways + mutation_embed_dim + drug_embed_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, pathway_features: torch.Tensor,
                mutation_embedding: torch.Tensor,
                drug_embedding: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pathway_features, mutation_embedding, drug_embedding], dim=-1)
        return self.head(x).squeeze(-1)  # -> (batch,) continuous LN_IC50

    def arch_config(self) -> dict:
        """Returns constructor args, for recording in checkpoint meta.json
        so SAM's run can be confirmed to use an identical architecture."""
        return {
            "n_pathways": self.n_pathways,
            "mutation_embed_dim": self.mutation_embed_dim,
            "drug_embed_dim": self.drug_embed_dim,
            "hidden_dims": list(self.hidden_dims),
        }
