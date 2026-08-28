import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score


# ============================================================
# STEP 28 — LIGHTWEIGHT TRAINED GENETIC TRANSFORMER
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

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading genetic dataset...")

df = pd.read_csv(DATASET)

print("Done.")
print()


# ============================================================
# TARGET
# ============================================================

target_column = None

for col in ["Target", "target", "Label", "label"]:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError("Target column not found.")


# ============================================================
# FEATURES
# ============================================================

metadata_columns = [
    "GSM_ID",
    "Source",
    "Subtype",
    "Target",
    "target",
    "Label",
    "label"
]

feature_columns = [
    c for c in df.columns
    if c not in metadata_columns
]

X = (
    df[feature_columns]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
    .values
    .astype(np.float32)
)

y = (
    pd.to_numeric(
        df[target_column],
        errors="coerce"
    )
    .astype(int)
    .values
)


print("=" * 60)
print("GENETIC DATASET")
print("=" * 60)

print("Samples  :", X.shape[0])
print("Features :", X.shape[1])
print("ASD      :", int((y == 1).sum()))
print("Control  :", int((y == 0).sum()))
print()


# ============================================================
# LIGHTWEIGHT TRANSFORMER
# ============================================================

class LightweightGeneticTransformer(nn.Module):

    def __init__(
        self,
        input_features=1000,
        embedding_dim=128,
        num_tokens=32,
        num_heads=4,
        num_layers=1
    ):
        super().__init__()

        self.input_features = input_features
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens

        # Compress 1000 genes into 32 genetic tokens.
        self.token_projection = nn.Linear(
            input_features,
            num_tokens * embedding_dim
        )

        # Learnable token positions.
        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                num_tokens,
                embedding_dim
            ) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Attention pooling.
        self.attention = nn.Linear(
            embedding_dim,
            1
        )

        self.norm = nn.LayerNorm(
            embedding_dim
        )

        # ASD / Control classifier.
        self.classifier = nn.Linear(
            embedding_dim,
            2
        )

    def encode(self, x):

        # ----------------------------------------------------
        # 1000 features → 32 tokens × 128
        # ----------------------------------------------------

        x = self.token_projection(x)

        x = x.view(
            -1,
            self.num_tokens,
            self.embedding_dim
        )

        x = (
            x +
            self.position_embedding
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        x = self.transformer(x)

        # ----------------------------------------------------
        # Attention pooling
        # ----------------------------------------------------

        scores = self.attention(x)

        weights = torch.softmax(
            scores,
            dim=1
        )

        embedding = torch.sum(
            x * weights,
            dim=1
        )

        embedding = self.norm(
            embedding
        )

        return embedding

    def forward(self, x):

        embedding = self.encode(x)

        logits = self.classifier(
            embedding
        )

        return embedding, logits


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    fold
):

    model = LightweightGeneticTransformer(
        input_features=X_train.shape[1],
        embedding_dim=128,
        num_tokens=32,
        num_heads=4,
        num_layers=1
    ).to(device)

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=device
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long,
        device=device
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    counts = np.bincount(
        y_train,
        minlength=2
    )

    weights = len(y_train) / (
        2.0 * counts
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4
    )

    best_auc = -np.inf
    best_state = None

    patience = 8
    patience_counter = 0

    MAX_EPOCHS = 40

    print()
    print("=" * 60)
    print(f"FOLD {fold}/5")
    print("=" * 60)

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()

        _, logits = model(
            X_train_tensor
        )

        loss = criterion(
            logits,
            y_train_tensor
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            _, val_logits = model(
                X_val_tensor
            )

            val_probability = torch.softmax(
                val_logits,
                dim=1
            )[:, 1]

        val_probability = (
            val_probability
            .cpu()
            .numpy()
        )

        val_prediction = (
            val_probability >= 0.5
        ).astype(int)

        auc = roc_auc_score(
            y_val,
            val_probability
        )

        accuracy = accuracy_score(
            y_val,
            val_prediction
        )

        if epoch == 1 or epoch % 5 == 0:

            print(
                f"Epoch {epoch:02d} | "
                f"Loss {loss.item():.4f} | "
                f"Val Acc {accuracy:.4f} | "
                f"Val AUC {auc:.4f}"
            )

        if auc > best_auc:

            best_auc = auc

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print(
                f"Early stopping at epoch {epoch}"
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():

        val_embedding, val_logits = model(
            X_val_tensor
        )

        probability = torch.softmax(
            val_logits,
            dim=1
        )[:, 1]

    embedding = (
        val_embedding
        .cpu()
        .numpy()
    )

    probability = (
        probability
        .cpu()
        .numpy()
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    final_accuracy = accuracy_score(
        y_val,
        prediction
    )

    final_auc = roc_auc_score(
        y_val,
        probability
    )

    print()
    print("Fold Results:")
    print(
        f"Accuracy       : {final_accuracy:.4f}"
    )
    print(
        f"ROC-AUC        : {final_auc:.4f}"
    )
    print(
        "Embedding Shape:",
        embedding.shape
    )

    return (
        embedding,
        probability,
        final_accuracy,
        final_auc,
        model
    )


# ============================================================
# 5-FOLD OOF TRAINING
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

oof_embeddings = np.zeros(
    (len(X), 128),
    dtype=np.float32
)

oof_probability = np.zeros(
    len(X),
    dtype=np.float32
)

fold_results = []


for fold, (
    train_idx,
    val_idx
) in enumerate(
    skf.split(X, y),
    start=1
):

    (
        embedding,
        probability,
        accuracy,
        auc,
        model
    ) = train_fold(
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        fold
    )

    oof_embeddings[
        val_idx
    ] = embedding

    oof_probability[
        val_idx
    ] = probability

    fold_results.append({
        "Fold": fold,
        "Accuracy": accuracy,
        "ROC_AUC": auc
    })


# ============================================================
# OVERALL OOF RESULTS
# ============================================================

oof_prediction = (
    oof_probability >= 0.5
).astype(int)

overall_accuracy = accuracy_score(
    y,
    oof_prediction
)

overall_auc = roc_auc_score(
    y,
    oof_probability
)


# ============================================================
# SAVE 128-D OOF EMBEDDINGS
# ============================================================

embedding_columns = [
    f"Genetic_Embedding_{i+1:03d}"
    for i in range(128)
]

embedding_df = pd.DataFrame(
    oof_embeddings,
    columns=embedding_columns
)

result = pd.DataFrame()

if "GSM_ID" in df.columns:
    result["GSM_ID"] = df[
        "GSM_ID"
    ].values

result["Target"] = y

result = pd.concat(
    [
        result,
        embedding_df
    ],
    axis=1
)

embedding_file = os.path.join(
    OUTPUT_DIR,
    "genetic_128d_embeddings_trained_oof.csv"
)

result.to_csv(
    embedding_file,
    index=False
)


# ============================================================
# SAVE CV RESULTS
# ============================================================

cv_file = os.path.join(
    OUTPUT_DIR,
    "genetic_transformer_lightweight_cv.csv"
)

pd.DataFrame(
    fold_results
).to_csv(
    cv_file,
    index=False
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model_file = os.path.join(
    OUTPUT_DIR,
    "genetic_transformer_128d_lightweight.pt"
)

torch.save(
    model.state_dict(),
    model_file
)


# ============================================================
# SAVE REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "genetic_transformer_128d_final_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 28 — TRAINED GENETIC TRANSFORMER\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Samples: {len(X)}\n"
    )

    f.write(
        f"Input Features: {X.shape[1]}\n"
    )

    f.write(
        "Transformer Tokens: 32\n"
    )

    f.write(
        "Transformer Layers: 1\n"
    )

    f.write(
        "Attention Heads: 4\n"
    )

    f.write(
        "Embedding Dimension: 128\n"
    )

    f.write(
        "Cross Validation: Stratified 5-Fold\n"
    )

    f.write(
        f"Overall OOF Accuracy: "
        f"{overall_accuracy:.4f}\n"
    )

    f.write(
        f"Overall OOF ROC-AUC: "
        f"{overall_auc:.4f}\n"
    )

    f.write(
        f"Embedding Shape: "
        f"{oof_embeddings.shape}\n"
    )

    f.write("\nPipeline:\n")

    f.write(
        "1000 genetic features -> "
        "32 compressed genetic tokens -> "
        "Transformer Encoder -> "
        "128-D genetic representation\n"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("STEP 28 — FINAL SUMMARY")
print("=" * 60)

print(
    "Input Shape              :",
    X.shape
)

print(
    "OOF Embedding Shape      :",
    oof_embeddings.shape
)

print(
    "Expected Embedding Shape:",
    (116, 128)
)

print()
print(
    f"Overall OOF Accuracy : "
    f"{overall_accuracy:.4f}"
)

print(
    f"Overall OOF ROC-AUC  : "
    f"{overall_auc:.4f}"
)

print()
print(
    "128-D trained embeddings:"
)

print(
    embedding_file
)

print()
print(
    "CV results:"
)

print(
    cv_file
)

print()
print(
    "Trained model:"
)

print(
    model_file
)

print()
print(
    "Report:"
)

print(
    report_file
)

print()
print(
    "Step 28 completed successfully."
)