import sys
import os

import torch

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from model import GeneticTransformer


# ==========================================================
# Configuration
# ==========================================================

BATCH_SIZE = 4

NUM_GENES = 1000

EMBEDDING_DIM = 64


# ==========================================================
# Create Model
# ==========================================================

model = GeneticTransformer(

    num_genes=NUM_GENES,

    embedding_dim=EMBEDDING_DIM,

    num_heads=4,

    num_layers=2,

    feed_forward_dim=128,

    dropout=0.30,

    num_classes=2
)


# ==========================================================
# Dummy Input
# ==========================================================

x = torch.randn(
    BATCH_SIZE,
    NUM_GENES
)


# ==========================================================
# Forward Pass
# ==========================================================

logits, embedding = model(x)


# ==========================================================
# Results
# ==========================================================

print("=" * 60)

print("GENETIC TRANSFORMER TEST")

print("=" * 60)

print()

print("Input Shape       :", tuple(x.shape))

print("Embedding Shape   :", tuple(embedding.shape))

print("Logits Shape      :", tuple(logits.shape))

print()

print("Model Parameters  :",
      sum(p.numel() for p in model.parameters()))

print()

print("Forward pass successful!")