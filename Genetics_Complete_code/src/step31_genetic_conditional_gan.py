import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 31 — GENETIC CONDITIONAL GAN
# Objective 2
#
# Input:
#   92 training samples
#   
#   69 ASD
#   23 Control
#
# Goal:
#   Generate 46 synthetic Control samples
#   to obtain a balanced 69 ASD / 69 Control training set.
#
# IMPORTANT:
#   The unseen 24-sample test set is NEVER loaded.
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\org genetical data"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "genetics",
    "outputs",
    "objective2_genetic",
    "train_test_split",
    "genetic_train_1000_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "genetics",
    "outputs",
    "objective2_genetic",
    "conditional_gan"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SYNTHETIC_FILE = os.path.join(
    OUTPUT_DIR,
    "synthetic_control_samples.csv"
)

BALANCED_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_balanced_training_dataset.csv"
)

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_conditional_gan.pt"
)

SCALER_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_gan_scaler.npz"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_conditional_gan_report.txt"
)

LOSS_FILE = os.path.join(
    OUTPUT_DIR,
    "gan_training_losses.csv"
)

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LATENT_DIM = 128
HIDDEN_DIM = 256

BATCH_SIZE = 8
EPOCHS = 800

LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# We need 69 Control samples total.
# There are 23 real Control samples.
# Therefore:
# 69 - 23 = 46 synthetic Control samples.
TARGET_CONTROL_COUNT = 69

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 31 — GENETIC CONDITIONAL GAN")
print("=" * 70)

print("\nUsing device:", DEVICE)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading genetic training data...")
df = pd.read_csv(INPUT_FILE)
print("Done.")

metadata_cols = [
    "GSM_ID",
    "Source",
    "Subtype",
    "Target"
]

feature_cols = [
    c for c in df.columns
    if c not in metadata_cols
]

if len(feature_cols) != 1000:
    raise ValueError(
        f"Expected 1000 genetic features, "
        f"found {len(feature_cols)}"
    )

print("\n" + "=" * 70)
print("GENETIC GAN TRAINING DATA")
print("=" * 70)

print("Samples  :", len(df))
print("Features :", len(feature_cols))

print("\nClass distribution:")
print(df["Target"].value_counts().sort_index())

# ============================================================
# VERIFY LABELS
# ============================================================

df["Target"] = pd.to_numeric(
    df["Target"],
    errors="raise"
).astype(int)

if not set(df["Target"].unique()).issubset({0, 1}):
    raise ValueError(
        "Target must contain only 0 and 1."
    )

# ============================================================
# IDENTIFY MINORITY CLASS
# ============================================================

class_counts = df["Target"].value_counts()

minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

print("\nMajority class :", majority_class)
print("Minority class :", minority_class)

if minority_class != 0:
    raise ValueError(
        "Expected Control (Target=0) to be the minority "
        "class for this genetic dataset."
    )

real_control_count = int(
    (df["Target"] == 0).sum()
)

real_asd_count = int(
    (df["Target"] == 1).sum()
)

synthetic_needed = (
    TARGET_CONTROL_COUNT - real_control_count
)

if synthetic_needed <= 0:
    raise ValueError(
        "No synthetic Control samples are required."
    )

print("\nReal ASD samples     :", real_asd_count)
print("Real Control samples :", real_control_count)
print("Synthetic required   :", synthetic_needed)

# ============================================================
# EXTRACT FEATURES
# ============================================================

X = df[feature_cols].values.astype(np.float32)
y = df["Target"].values.astype(np.int64)

# ============================================================
# SCALE FEATURES
#
# IMPORTANT:
# The scaler is fitted ONLY on the training dataset.
# The unseen test set is never accessed.
# ============================================================

print("\nFitting StandardScaler on training data...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X).astype(
    np.float32
)

print("Scaling completed.")

# ============================================================
# SAVE SCALER
# ============================================================

np.savez(
    SCALER_FILE,
    mean=scaler.mean_,
    scale=scaler.scale_
)

# ============================================================
# TORCH TENSORS
# ============================================================

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32,
    device=DEVICE
)

y_tensor = torch.tensor(
    y,
    dtype=torch.long,
    device=DEVICE
)

# ============================================================
# CONDITIONAL GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        latent_dim,
        hidden_dim,
        num_features
    ):
        super().__init__()

        # Class embedding allows the generator
        # to condition on ASD vs Control.
        self.label_embedding = nn.Embedding(
            2,
            32
        )

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim + 32,
                hidden_dim
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),

            nn.Linear(
                hidden_dim * 2,
                num_features
            )
        )

    def forward(
        self,
        noise,
        labels
    ):

        label_vector = self.label_embedding(
            labels
        )

        x = torch.cat(
            [noise, label_vector],
            dim=1
        )

        return self.network(x)


# ============================================================
# CONDITIONAL DISCRIMINATOR
# ============================================================

class Discriminator(nn.Module):

    def __init__(
        self,
        num_features,
        hidden_dim
    ):
        super().__init__()

        self.label_embedding = nn.Embedding(
            2,
            32
        )

        self.network = nn.Sequential(
            nn.Linear(
                num_features + 32,
                hidden_dim * 2
            ),
            nn.LeakyReLU(
                0.2
            ),
            nn.Dropout(0.2),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.LeakyReLU(
                0.2
            ),
            nn.Dropout(0.2),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        features,
        labels
    ):

        label_vector = self.label_embedding(
            labels
        )

        x = torch.cat(
            [features, label_vector],
            dim=1
        )

        return self.network(x)


# ============================================================
# CREATE MODELS
# ============================================================

num_features = len(feature_cols)

generator = Generator(
    LATENT_DIM,
    HIDDEN_DIM,
    num_features
).to(DEVICE)

discriminator = Discriminator(
    num_features,
    HIDDEN_DIM
).to(DEVICE)

# ============================================================
# OPTIMIZERS
# ============================================================

optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)

criterion = nn.BCEWithLogitsLoss()

# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("GAN TRAINING")
print("=" * 70)

print("\nTraining only on the 92-sample training set.")
print("Unseen test set is NOT loaded.")

loss_records = []

n_samples = len(X_tensor)

best_g_loss = float("inf")
best_state_G = None
best_state_D = None

# ------------------------------------------------------------
# Mini-batch training
# ------------------------------------------------------------

for epoch in range(1, EPOCHS + 1):

    # Random permutation
    permutation = torch.randperm(
        n_samples,
        device=DEVICE
    )

    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    batches = 0

    for start in range(
        0,
        n_samples,
        BATCH_SIZE
    ):

        idx = permutation[
            start:start + BATCH_SIZE
        ]

        real_features = X_tensor[idx]
        real_labels = y_tensor[idx]

        current_batch = len(idx)

        # ====================================================
        # TRAIN DISCRIMINATOR
        # ====================================================

        optimizer_D.zero_grad()

        real_logits = discriminator(
            real_features,
            real_labels
        )

        real_targets = torch.ones(
            current_batch,
            1,
            device=DEVICE
        )

        loss_real = criterion(
            real_logits,
            real_targets
        )

        # Generate fake samples using same labels
        noise = torch.randn(
            current_batch,
            LATENT_DIM,
            device=DEVICE
        )

        fake_features = generator(
            noise,
            real_labels
        )

        fake_logits = discriminator(
            fake_features.detach(),
            real_labels
        )

        fake_targets = torch.zeros(
            current_batch,
            1,
            device=DEVICE
        )

        loss_fake = criterion(
            fake_logits,
            fake_targets
        )

        loss_D = (
            loss_real +
            loss_fake
        ) / 2

        loss_D.backward()

        optimizer_D.step()

        # ====================================================
        # TRAIN GENERATOR
        # ====================================================

        optimizer_G.zero_grad()

        noise = torch.randn(
            current_batch,
            LATENT_DIM,
            device=DEVICE
        )

        fake_features = generator(
            noise,
            real_labels
        )

        fake_logits = discriminator(
            fake_features,
            real_labels
        )

        generator_targets = torch.ones(
            current_batch,
            1,
            device=DEVICE
        )

        loss_G = criterion(
            fake_logits,
            generator_targets
        )

        loss_G.backward()

        optimizer_G.step()

        epoch_g_loss += loss_G.item()
        epoch_d_loss += loss_D.item()

        batches += 1

    avg_g_loss = (
        epoch_g_loss / batches
    )

    avg_d_loss = (
        epoch_d_loss / batches
    )

    loss_records.append(
        {
            "Epoch": epoch,
            "Generator_Loss": avg_g_loss,
            "Discriminator_Loss": avg_d_loss
        }
    )

    # --------------------------------------------------------
    # Save best generator
    # --------------------------------------------------------

    if avg_g_loss < best_g_loss:

        best_g_loss = avg_g_loss

        best_state_G = {
            k: v.detach().cpu().clone()
            for k, v in generator.state_dict().items()
        }

        best_state_D = {
            k: v.detach().cpu().clone()
            for k, v in discriminator.state_dict().items()
        }

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 50 == 0
    ):

        print(
            f"Epoch {epoch:04d} | "
            f"G Loss {avg_g_loss:.4f} | "
            f"D Loss {avg_d_loss:.4f}"
        )

# ============================================================
# RESTORE BEST GENERATOR
# ============================================================

if best_state_G is not None:

    generator.load_state_dict(
        best_state_G
    )

if best_state_D is not None:

    discriminator.load_state_dict(
        best_state_D
    )

# ============================================================
# SAVE LOSS HISTORY
# ============================================================

loss_df = pd.DataFrame(
    loss_records
)

loss_df.to_csv(
    LOSS_FILE,
    index=False
)

# ============================================================
# GENERATE SYNTHETIC CONTROL SAMPLES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING SYNTHETIC CONTROL SAMPLES")
print("=" * 70)

generator.eval()

with torch.no_grad():

    noise = torch.randn(
        synthetic_needed,
        LATENT_DIM,
        device=DEVICE
    )

    control_labels = torch.zeros(
        synthetic_needed,
        dtype=torch.long,
        device=DEVICE
    )

    synthetic_scaled = generator(
        noise,
        control_labels
    ).cpu().numpy()

# ============================================================
# INVERSE SCALE
# ============================================================

synthetic_features = scaler.inverse_transform(
    synthetic_scaled
)

# ============================================================
# BUILD SYNTHETIC DATAFRAME
# ============================================================

synthetic_ids = [
    f"SYNTH_CONTROL_GAN_{i+1:03d}"
    for i in range(synthetic_needed)
]

synthetic_df = pd.DataFrame(
    synthetic_features,
    columns=feature_cols
)

synthetic_df.insert(
    0,
    "GSM_ID",
    synthetic_ids
)

synthetic_df.insert(
    1,
    "Source",
    "Conditional_GAN"
)

synthetic_df.insert(
    2,
    "Subtype",
    "Synthetic_Control"
)

synthetic_df.insert(
    3,
    "Target",
    0
)

# ============================================================
# CHECK SYNTHETIC VALUES
# ============================================================

nan_count = int(
    synthetic_df[feature_cols]
    .isna()
    .sum()
    .sum()
)

inf_count = int(
    np.isinf(
        synthetic_df[feature_cols].values
    ).sum()
)

print(
    "\nSynthetic samples generated:",
    len(synthetic_df)
)

print(
    "Synthetic feature shape:",
    synthetic_df[feature_cols].shape
)

print(
    "Synthetic NaN values:",
    nan_count
)

print(
    "Synthetic Inf values:",
    inf_count
)

if nan_count > 0:
    raise RuntimeError(
        "NaN values detected in synthetic samples."
    )

if inf_count > 0:
    raise RuntimeError(
        "Infinite values detected in synthetic samples."
    )

# ============================================================
# SAVE SYNTHETIC DATA
# ============================================================

synthetic_df.to_csv(
    SYNTHETIC_FILE,
    index=False
)

# ============================================================
# CREATE BALANCED TRAINING DATASET
# ============================================================

balanced_df = pd.concat(
    [
        df,
        synthetic_df
    ],
    ignore_index=True
)

# Shuffle balanced data
balanced_df = balanced_df.sample(
    frac=1.0,
    random_state=SEED
).reset_index(drop=True)

# ============================================================
# BALANCED CLASS CHECK
# ============================================================

balanced_counts = (
    balanced_df["Target"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print("BALANCED GENETIC TRAINING DATA")
print("=" * 70)

print(
    "\nTotal samples:",
    len(balanced_df)
)

for cls, count in balanced_counts.items():

    class_name = (
        "Control"
        if cls == 0
        else "ASD"
    )

    print(
        f"{class_name} ({cls}): {count}"
    )

if (
    balanced_counts.get(0, 0)
    != balanced_counts.get(1, 0)
):

    print(
        "\nWARNING: Classes are not perfectly balanced."
    )

else:

    print(
        "\nClass balancing: PASSED"
    )

# ============================================================
# SAVE BALANCED DATASET
# ============================================================

balanced_df.to_csv(
    BALANCED_FILE,
    index=False
)

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "generator_state_dict":
            generator.state_dict(),

        "discriminator_state_dict":
            discriminator.state_dict(),

        "latent_dim":
            LATENT_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "num_features":
            num_features,

        "feature_columns":
            feature_cols,

        "random_seed":
            SEED
    },
    MODEL_FILE
)

# ============================================================
# REPORT
# ============================================================

report = f"""
STEP 31 — GENETIC CONDITIONAL GAN
=================================

Objective
---------
Generate synthetic minority-class genetic samples to balance
the Objective-2 genetic training dataset.

Input
-----
{INPUT_FILE}

Training data
-------------
Samples: {len(df)}
Features: {num_features}

Real ASD samples:
{real_asd_count}

Real Control samples:
{real_control_count}

Minority class:
Control (Target=0)

Synthetic samples required:
{synthetic_needed}

GAN configuration
-----------------
Architecture:
Conditional GAN

Generator:
Latent dimension: {LATENT_DIM}
Hidden dimension: {HIDDEN_DIM}
Output dimension: {num_features}

Discriminator:
Input dimension: {num_features}
Conditional label embedding: 32

Optimizer:
Adam

Learning rate:
{LEARNING_RATE}

Loss:
Binary Cross Entropy with Logits

Epochs:
{EPOCHS}

Batch size:
{BATCH_SIZE}

Training device:
{DEVICE}

Data leakage protection
-----------------------
Only the 92 training samples were used.

The unseen 24-sample test dataset was NOT loaded,
accessed, or used during GAN training.

Synthetic data
--------------
Synthetic Control samples:
{len(synthetic_df)}

Synthetic NaN values:
{nan_count}

Synthetic Inf values:
{inf_count}

Balanced dataset
----------------
Real ASD:
{real_asd_count}

Real Control:
{real_control_count}

Synthetic Control:
{len(synthetic_df)}

Final ASD:
{balanced_counts.get(1, 0)}

Final Control:
{balanced_counts.get(0, 0)}

Output files
------------
Synthetic Control samples:
{SYNTHETIC_FILE}

Balanced genetic training dataset:
{BALANCED_FILE}

GAN model:
{MODEL_FILE}

Scaler:
{SCALER_FILE}

Training losses:
{LOSS_FILE}

Report:
{REPORT_FILE}
"""

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("STEP 31 FINAL SUMMARY")
print("=" * 70)

print(
    "\nReal training samples:",
    len(df)
)

print(
    "Real ASD:",
    real_asd_count
)

print(
    "Real Control:",
    real_control_count
)

print(
    "Synthetic Control:",
    len(synthetic_df)
)

print(
    "Balanced dataset:",
    len(balanced_df)
)

print(
    "\nFinal class distribution:"
)

print(
    balanced_df["Target"]
    .value_counts()
    .sort_index()
)

print("\nGAN model saved:")
print(MODEL_FILE)

print("\nSynthetic samples saved:")
print(SYNTHETIC_FILE)

print("\nBalanced dataset saved:")
print(BALANCED_FILE)

print("\nReport saved:")
print(REPORT_FILE)

print("\nStep 31 completed successfully.")