"""
PyTorch Dual-Branch GxE Neural Network Model
for Gene-Environment Interaction survival prediction.
"""

import torch
import torch.nn as nn


class GxEModel(nn.Module):
    """
    Multi-branch neural network for GxE interaction modeling.

    Architecture:
    - Branch 1 (Gene Encoder): gene expression features
    - Branch 2 (Environment Encoder): clinical/environmental features
    - Branch 3 (Methylation Encoder): DNA methylation features
    - Fusion Layer: concatenation + prediction
    """

    def __init__(self, n_genes, n_env, n_methyl):
        super(GxEModel, self).__init__()

        # Branch 1: Gene Encoder
        self.gene_encoder = nn.Sequential(
            nn.BatchNorm1d(n_genes),
            nn.Linear(n_genes, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Branch 2: Environment Encoder
        self.env_encoder = nn.Sequential(
            nn.BatchNorm1d(n_env),
            nn.Linear(n_env, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Branch 3: Methylation Encoder
        self.methyl_encoder = nn.Sequential(
            nn.Linear(n_methyl, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Interaction Layer (Gene + Env)
        self.interaction = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Final Fusion (Interaction + Methylation)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_gene, x_env, x_methyl):
        # Encode each branch
        gene_out = self.gene_encoder(x_gene)
        env_out = self.env_encoder(x_env)
        methyl_out = self.methyl_encoder(x_methyl)

        # Gene-Environment interaction
        interaction_input = torch.cat([gene_out, env_out], dim=1)
        interaction_out = self.interaction(interaction_input)

        # Final fusion with methylation
        fusion_input = torch.cat([interaction_out, methyl_out], dim=1)
        output = self.fusion(fusion_input)

        return output.squeeze(1)
