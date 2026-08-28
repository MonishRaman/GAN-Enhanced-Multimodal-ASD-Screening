import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 33 — EMBEDDING-SPACE GENETIC CONDITIONAL GAN
# ============================================================

BASE = r"D:\org genetical data"

INPUT_FILE = os.path.join(
    BASE,
    r"genetics\outputs\genetic_embedding",
    "genetic_128d_embeddings_trained_oof.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic",
    "embedding_gan"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "genetic_embedding_gan_train.csv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "genetic_embedding_gan_test_unseen.csv"
)

SYNTHETIC_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "synthetic_control_128d_embeddings.csv"
)

BALANCED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "genetic_balanced_128d_embedding_dataset.csv"
)

MODEL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "genetic_embedding_conditional_gan.pt"
)

REPORT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "genetic_embedding_conditional_gan_report.txt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading trained OOF genetic embeddings...")

df = pd.read_csv(INPUT_FILE)

print("Done.")

print("\n" + "=" * 70)
print("INPUT DATASET")
print("=" * 70)

print(f"Samples : {len(df)}")


embedding_cols = [
    c for c in df.columns
    if c.startswith("Genetic_Embedding_")
]

print(
    f"Embedding dimensions : "
    f"{len(embedding_cols)}"
)

print(
    "\nClass distribution:"
)

print(
    df["Target"].value_counts()
)


# ============================================================
# VALIDATE INPUT
# ============================================================

assert len(df) == 116, (
    f"Expected 116 samples, found {len(df)}"
)

assert len(embedding_cols) == 128, (
    f"Expected 128 embeddings, "
    f"found {len(embedding_cols)}"
)

assert df[embedding_cols].isna().sum().sum() == 0, \
    "NaN values detected."

assert np.isfinite(
    df[embedding_cols].to_numpy()
).all(), \
    "Infinite values detected."


# ============================================================
# STRATIFIED TRAIN / TEST SPLIT
# ============================================================

print("\nPerforming stratified 80/20 split...")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["Target"],
    random_state=SEED
)

train_df = train_df.sort_values(
    "GSM_ID"
).reset_index(drop=True)

test_df = test_df.sort_values(
    "GSM_ID"
).reset_index(drop=True)


train_ids = set(train_df["GSM_ID"])
test_ids = set(test_df["GSM_ID"])

overlap = train_ids.intersection(test_ids)

print(
    f"\nTrain samples : {len(train_df)}"
)

print(
    f"Test samples  : {len(test_df)}"
)

print(
    f"Sample overlap: {len(overlap)}"
)

assert len(overlap) == 0


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTraining distribution:")

print(
    train_df["Target"].value_counts()
)

print("\nTest distribution:")

print(
    test_df["Target"].value_counts()
)


# ============================================================
# SAVE TRAIN / TEST
# ============================================================

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False
)

test_df.to_csv(
    TEST_OUTPUT,
    index=False
)


# ============================================================
# EXTRACT TRAINING CONTROL EMBEDDINGS
# ============================================================

train_control = train_df[
    train_df["Target"] == 0
][embedding_cols].to_numpy(
    dtype=np.float32
)

train_asd = train_df[
    train_df["Target"] == 1
][embedding_cols].to_numpy(
    dtype=np.float32
)

print("\n" + "=" * 70)
print("GAN TRAINING DATA")
print("=" * 70)

print(
    f"Real ASD samples     : {len(train_asd)}"
)

print(
    f"Real Control samples : {len(train_control)}"
)


majority_count = len(train_asd)
minority_count = len(train_control)

synthetic_required = (
    majority_count - minority_count
)

print(
    f"Synthetic required   : "
    f"{synthetic_required}"
)


# ============================================================
# STANDARDIZATION
# ============================================================

print("\nFitting StandardScaler on training embeddings...")

scaler = StandardScaler()

X_train_all = train_df[
    embedding_cols
].to_numpy(
    dtype=np.float32
)

scaler.fit(X_train_all)

control_scaled = scaler.transform(
    train_control
).astype(np.float32)

print("Scaling completed.")


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        noise_dim=64,
        embedding_dim=128
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                noise_dim + 1,
                128
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Linear(
                128,
                256
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Linear(
                256,
                embedding_dim
            )
        )

    def forward(
        self,
        noise,
        labels
    ):

        labels = labels.float().unsqueeze(1)

        x = torch.cat(
            [noise, labels],
            dim=1
        )

        return self.network(x)


# ============================================================
# DISCRIMINATOR
# ============================================================

class Discriminator(nn.Module):

    def __init__(
        self,
        embedding_dim=128
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim + 1,
                256
            ),

            nn.LeakyReLU(
                0.2
            ),

            nn.Dropout(0.25),

            nn.Linear(
                256,
                128
            ),

            nn.LeakyReLU(
                0.2
            ),

            nn.Dropout(0.25),

            nn.Linear(
                128,
                1
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        embeddings,
        labels
    ):

        labels = labels.float().unsqueeze(1)

        x = torch.cat(
            [embeddings, labels],
            dim=1
        )

        return self.network(x)


# ============================================================
# INITIALIZE MODELS
# ============================================================

NOISE_DIM = 64
EMBEDDING_DIM = 128

generator = Generator(
    noise_dim=NOISE_DIM,
    embedding_dim=EMBEDDING_DIM
).to(device)

discriminator = Discriminator(
    embedding_dim=EMBEDDING_DIM
).to(device)


criterion = nn.BCELoss()

g_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

d_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)


# ============================================================
# PREPARE CONTROL DATA
# ============================================================

control_tensor = torch.tensor(
    control_scaled,
    dtype=torch.float32
).to(device)

control_labels = torch.zeros(
    len(control_tensor),
    dtype=torch.long,
    device=device
)


# ============================================================
# TRAIN GAN
# ============================================================

EPOCHS = 1000

print("\n" + "=" * 70)
print("TRAINING EMBEDDING-SPACE CONDITIONAL GAN")
print("=" * 70)

print(
    f"Training samples : {len(control_tensor)}"
)

print(
    f"Embedding size   : {EMBEDDING_DIM}"
)

print(
    f"Noise dimension  : {NOISE_DIM}"
)

print(
    f"Epochs           : {EPOCHS}"
)


loss_history = []


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Train discriminator
    # --------------------------------------------------------

    discriminator.train()
    generator.train()

    d_optimizer.zero_grad()

    real_labels = torch.ones(
        len(control_tensor),
        1,
        device=device
    )

    fake_labels = torch.zeros(
        len(control_tensor),
        1,
        device=device
    )

    real_pred = discriminator(
        control_tensor,
        control_labels
    )

    real_loss = criterion(
        real_pred,
        real_labels
    )

    noise = torch.randn(
        len(control_tensor),
        NOISE_DIM,
        device=device
    )

    fake_embeddings = generator(
        noise,
        control_labels
    )

    fake_pred = discriminator(
        fake_embeddings.detach(),
        control_labels
    )

    fake_loss = criterion(
        fake_pred,
        fake_labels
    )

    d_loss = (
        real_loss +
        fake_loss
    ) / 2

    d_loss.backward()

    d_optimizer.step()


    # --------------------------------------------------------
    # Train generator
    # --------------------------------------------------------

    g_optimizer.zero_grad()

    noise = torch.randn(
        len(control_tensor),
        NOISE_DIM,
        device=device
    )

    generated = generator(
        noise,
        control_labels
    )

    prediction = discriminator(
        generated,
        control_labels
    )

    g_loss = criterion(
        prediction,
        real_labels
    )

    g_loss.backward()

    g_optimizer.step()


    loss_history.append(
        {
            "Epoch": epoch,
            "Generator_Loss": g_loss.item(),
            "Discriminator_Loss": d_loss.item()
        }
    )


    if (
        epoch == 1
        or epoch % 50 == 0
    ):

        print(
            f"Epoch {epoch:04d} | "
            f"G Loss {g_loss.item():.4f} | "
            f"D Loss {d_loss.item():.4f}"
        )


# ============================================================
# GENERATE SYNTHETIC CONTROL EMBEDDINGS
# ============================================================

print("\nGenerating synthetic Control embeddings...")

generator.eval()

with torch.no_grad():

    noise = torch.randn(
        synthetic_required,
        NOISE_DIM,
        device=device
    )

    labels = torch.zeros(
        synthetic_required,
        dtype=torch.long,
        device=device
    )

    synthetic_scaled = generator(
        noise,
        labels
    ).cpu().numpy()


synthetic_embeddings = scaler.inverse_transform(
    synthetic_scaled
)


# ============================================================
# BASIC VALIDATION
# ============================================================

synthetic_nan = int(
    np.isnan(
        synthetic_embeddings
    ).sum()
)

synthetic_inf = int(
    np.isinf(
        synthetic_embeddings
    ).sum()
)

duplicate_count = int(
    pd.DataFrame(
        synthetic_embeddings
    ).duplicated().sum()
)


print("\n" + "=" * 70)
print("SYNTHETIC EMBEDDING VALIDATION")
print("=" * 70)

print(
    f"Synthetic samples : "
    f"{len(synthetic_embeddings)}"
)

print(
    f"Embedding shape   : "
    f"{synthetic_embeddings.shape}"
)

print(
    f"NaN values        : {synthetic_nan}"
)

print(
    f"Inf values        : {synthetic_inf}"
)

print(
    f"Duplicate samples : {duplicate_count}"
)


# ============================================================
# SAVE SYNTHETIC EMBEDDINGS
# ============================================================

synthetic_df = pd.DataFrame(
    synthetic_embeddings,
    columns=embedding_cols
)

synthetic_df.insert(
    0,
    "GSM_ID",
    [
        f"SYNTH_CONTROL_{i+1:03d}"
        for i in range(
            len(synthetic_df)
        )
    ]
)

synthetic_df.insert(
    1,
    "Target",
    0
)

synthetic_df.to_csv(
    SYNTHETIC_OUTPUT,
    index=False
)


# ============================================================
# CREATE BALANCED EMBEDDING DATASET
# ============================================================

real_train_embeddings = train_df[
    ["GSM_ID", "Target"] +
    embedding_cols
].copy()

balanced_df = pd.concat(
    [
        real_train_embeddings,
        synthetic_df
    ],
    ignore_index=True
)

balanced_df.to_csv(
    BALANCED_OUTPUT,
    index=False
)


# ============================================================
# FINAL CLASS BALANCE
# ============================================================

print("\n" + "=" * 70)
print("FINAL BALANCED DATASET")
print("=" * 70)

print(
    balanced_df["Target"].value_counts()
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

        "noise_dim":
            NOISE_DIM,

        "embedding_dim":
            EMBEDDING_DIM,

        "seed":
            SEED
    },
    MODEL_OUTPUT
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

loss_df = pd.DataFrame(
    loss_history
)

loss_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "embedding_gan_training_history.csv"
    ),
    index=False
)


# ============================================================
# REPORT
# ============================================================

with open(
    REPORT_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 33 — EMBEDDING-SPACE GENETIC CONDITIONAL GAN\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total samples : {len(df)}\n"
    )

    f.write(
        f"Training samples : {len(train_df)}\n"
    )

    f.write(
        f"Unseen test samples : {len(test_df)}\n"
    )

    f.write(
        f"Embedding dimensions : {len(embedding_cols)}\n\n"
    )

    f.write(
        "TRAINING CLASS DISTRIBUTION\n"
    )

    f.write(
        f"ASD : {len(train_asd)}\n"
    )

    f.write(
        f"Control : {len(train_control)}\n"
    )

    f.write(
        f"Synthetic Control : {synthetic_required}\n\n"
    )

    f.write(
        "FINAL BALANCE\n"
    )

    f.write(
        f"ASD : "
        f"{(balanced_df['Target'] == 1).sum()}\n"
    )

    f.write(
        f"Control : "
        f"{(balanced_df['Target'] == 0).sum()}\n\n"
    )

    f.write(
        "SYNTHETIC QUALITY\n"
    )

    f.write(
        f"NaN values : {synthetic_nan}\n"
    )

    f.write(
        f"Inf values : {synthetic_inf}\n"
    )

    f.write(
        f"Duplicate synthetic samples : "
        f"{duplicate_count}\n\n"
    )

    f.write(
        "IMPORTANT\n"
    )

    f.write(
        "This step generates synthetic genetic embeddings "
        "for the minority class. Synthetic quality must "
        "be independently validated before these samples "
        "are used in downstream multimodal training.\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 33 — FINAL SUMMARY")
print("=" * 70)

print(
    f"Training samples        : {len(train_df)}"
)

print(
    f"Unseen test samples     : {len(test_df)}"
)

print(
    f"Embedding dimensions    : {EMBEDDING_DIM}"
)

print(
    f"Real ASD                : {len(train_asd)}"
)

print(
    f"Real Control            : {len(train_control)}"
)

print(
    f"Synthetic Control       : {synthetic_required}"
)

print(
    f"Final ASD               : "
    f"{(balanced_df['Target'] == 1).sum()}"
)

print(
    f"Final Control           : "
    f"{(balanced_df['Target'] == 0).sum()}"
)

print(
    f"Synthetic NaN           : {synthetic_nan}"
)

print(
    f"Synthetic Inf           : {synthetic_inf}"
)

print(
    f"Synthetic duplicates    : {duplicate_count}"
)

print("\nFiles saved:")

print(TRAIN_OUTPUT)
print(TEST_OUTPUT)
print(SYNTHETIC_OUTPUT)
print(BALANCED_OUTPUT)
print(MODEL_OUTPUT)
print(REPORT_OUTPUT)

print("\nStep 33 completed.")