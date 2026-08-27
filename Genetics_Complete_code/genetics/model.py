import torch
import torch.nn as nn


class GeneticTransformer(nn.Module):

    def __init__(
        self,
        num_genes,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        feed_forward_dim=128,
        dropout=0.30,
        num_classes=2
    ):

        super().__init__()

        self.num_genes = num_genes

        self.embedding_dim = embedding_dim

        # --------------------------------------------------
        # Gene value -> token embedding
        # --------------------------------------------------

        self.gene_projection = nn.Linear(
            1,
            embedding_dim
        )

        # --------------------------------------------------
        # Learnable positional embeddings
        # --------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                num_genes,
                embedding_dim
            )
        )

        # --------------------------------------------------
        # Transformer Encoder
        # --------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # --------------------------------------------------
        # Normalization
        # --------------------------------------------------

        self.norm = nn.LayerNorm(
            embedding_dim
        )

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(
                embedding_dim,
                num_classes
            )
        )

    # ======================================================
    # Encode genetic data
    # ======================================================

    def encode(self, x):

        # x:
        # (batch, num_genes)

        x = x.unsqueeze(-1)

        # (batch, num_genes, 1)

        x = self.gene_projection(x)

        # (batch, num_genes, embedding_dim)

        x = x + self.position_embedding

        # Transformer

        x = self.encoder(x)

        # Normalize

        x = self.norm(x)

        # Global average pooling

        embedding = x.mean(dim=1)

        # (batch, embedding_dim)

        return embedding

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, x):

        embedding = self.encode(x)

        logits = self.classifier(
            embedding
        )

        return logits, embedding