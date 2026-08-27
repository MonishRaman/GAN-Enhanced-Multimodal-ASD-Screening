import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# STEP 27 — GENETIC 128-D FEATURE REPRESENTATION
# ============================================================

BASE = r"D:\org genetical data"

DATASET = os.path.join(
    BASE,
    "data",
    "processed",
    "top_features_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "genetic_embedding"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading genetic dataset...")

df = pd.read_csv(DATASET)

print("Done.")
print()


# ------------------------------------------------------------
# Identify metadata / target columns
# ------------------------------------------------------------

target_candidates = [
    "Target",
    "target",
    "Label",
    "label"
]

target_column = None

for col in target_candidates:
    if col in df.columns:
        target_column = col
        break


if target_column is None:
    raise ValueError(
        "Target column not found in dataset."
    )


# Remove non-feature columns
non_feature_columns = [
    "GSM_ID",
    "Source",
    "Subtype",
    "Target",
    "target",
    "Label",
    "label"
]

feature_columns = [
    col
    for col in df.columns
    if col not in non_feature_columns
]


# Keep only numeric features
X = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.fillna(0.0)

y = pd.to_numeric(
    df[target_column],
    errors="coerce"
)


# ============================================================
# CONVERT TO TORCH
# ============================================================

X_tensor = torch.tensor(
    X.values,
    dtype=torch.float32
).to(device)

y_tensor = torch.tensor(
    y.values,
    dtype=torch.long
).to(device)


print("=" * 60)
print("GENETIC DATASET")
print("=" * 60)

print(
    "Samples  :", X.shape[0]
)

print(
    "Features :", X.shape[1]
)

print(
    "ASD      :",
    int((y == 1).sum())
)

print(
    "Control  :",
    int((y == 0).sum())
)

print()


# ============================================================
# GENETIC TRANSFORMER ENCODER
# ============================================================

class GeneticTransformerEncoder(
    nn.Module
):

    def __init__(
        self,
        input_features,
        embedding_dim=128,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ):

        super().__init__()

        self.embedding_dim = embedding_dim

        # Each genetic feature becomes a token
        self.feature_embedding = nn.Linear(
            1,
            embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Pool all feature tokens into one vector
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.output_projection = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim
            ),
            nn.LayerNorm(
                embedding_dim
            )
        )

    def forward(self, x):

        # x:
        # [batch, features]

        x = x.unsqueeze(-1)

        # [batch, features, 1]
        x = self.feature_embedding(x)

        # [batch, features, 128]
        x = self.transformer(x)

        # Rearrange for pooling
        x = x.transpose(1, 2)

        # [batch, 128, 1]
        x = self.pool(x)

        # [batch, 128]
        x = x.squeeze(-1)

        # Final 128-D representation
        x = self.output_projection(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = GeneticTransformerEncoder(
    input_features=X.shape[1],
    embedding_dim=128,
    num_heads=8,
    num_layers=2
).to(device)


model.eval()


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("=" * 60)
print("GENERATING 128-D GENETIC EMBEDDINGS")
print("=" * 60)

with torch.no_grad():

    embeddings = model(
        X_tensor
    )


print(
    "Embedding Shape:",
    tuple(embeddings.shape)
)

print(
    "Expected Shape :",
    (X.shape[0], 128)
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

embedding_array = (
    embeddings
    .cpu()
    .numpy()
)


embedding_columns = [
    f"Genetic_Embedding_{i+1:03d}"
    for i in range(128)
]


embedding_df = pd.DataFrame(
    embedding_array,
    columns=embedding_columns
)


# Add sample information
result = pd.DataFrame()

if "GSM_ID" in df.columns:

    result["GSM_ID"] = df["GSM_ID"]

if "Target" in df.columns:

    result["Target"] = df["Target"]

result = pd.concat(
    [
        result.reset_index(drop=True),
        embedding_df.reset_index(drop=True)
    ],
    axis=1
)


output_file = os.path.join(
    OUTPUT_DIR,
    "genetic_128d_embeddings.csv"
)


result.to_csv(
    output_file,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = os.path.join(
    OUTPUT_DIR,
    "genetic_transformer_128d.pt"
)


torch.save(
    model.state_dict(),
    model_file
)


# ============================================================
# REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "genetic_128d_embedding_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 27 — GENETIC 128-D EMBEDDING REPORT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Samples: {X.shape[0]}\n"
    )

    f.write(
        f"Input Features: {X.shape[1]}\n"
    )

    f.write(
        "Transformer Embedding Dimension: 128\n"
    )

    f.write(
        f"Output Shape: {embeddings.shape}\n"
    )

    f.write(
        f"ASD Samples: {(y == 1).sum()}\n"
    )

    f.write(
        f"Control Samples: {(y == 0).sum()}\n"
    )

    f.write(
        "\nRepresentation:\n"
    )

    f.write(
        "Genetic expression features -> "
        "Transformer Encoder -> "
        "128-D Genetic Feature Vector\n"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("GENETIC 128-D EMBEDDING SUMMARY")
print("=" * 60)

print(
    "Input Shape     :",
    tuple(X.shape)
)

print(
    "Embedding Shape :",
    tuple(embeddings.shape)
)

print(
    "Expected        :",
    f"({X.shape[0]}, 128)"
)

print()
print(
    "Embedding saved:",
    output_file
)

print(
    "Model saved:",
    model_file
)

print(
    "Report saved:",
    report_file
)

print()
print(
    "Step 27 completed successfully."
)