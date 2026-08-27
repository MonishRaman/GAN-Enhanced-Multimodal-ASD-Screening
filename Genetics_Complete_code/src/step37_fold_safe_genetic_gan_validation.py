import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# STEP 37
# FOLD-SAFE GENETIC GAN AUGMENTATION VALIDATION
# ============================================================

BASE = r"D:\org genetical data"

TRAIN_FILE = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\embedding_gan\genetic_embedding_gan_train.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\fold_safe_gan_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "fold_safe_gan_validation_results.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "fold_safe_gan_validation_summary.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "fold_safe_gan_validation_report.txt"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
N_SPLITS = 5

# Keep CPU runtime reasonable.
GAN_EPOCHS = 400

NOISE_DIM = 64
HIDDEN_DIM = 128

AUGMENTATION_LEVELS = [
    0,
    10,
    20,
    30
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

print("=" * 70)
print("STEP 37 — FOLD-SAFE GENETIC GAN VALIDATION")
print("=" * 70)

print("\nDevice:", DEVICE)
print("GAN epochs per fold:", GAN_EPOCHS)
print("CV folds:", N_SPLITS)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading genetic training embeddings...")

df = pd.read_csv(TRAIN_FILE)

print("Done.")


embedding_cols = [
    c for c in df.columns
    if c.startswith("Genetic_Embedding_")
]

if len(embedding_cols) != 128:
    raise ValueError(
        f"Expected 128 embedding dimensions, "
        f"found {len(embedding_cols)}"
    )


X = df[embedding_cols].astype(float).values
y = df["Target"].astype(int).values

print("\nDataset")
print("Samples     :", len(X))
print("Embeddings  :", X.shape[1])

print("\nClass distribution:")
print(pd.Series(y).value_counts().sort_index())


# ============================================================
# VALIDATION
# ============================================================

if np.isnan(X).any():
    raise ValueError("NaN values detected.")

if np.isinf(X).any():
    raise ValueError("Infinite values detected.")


# ============================================================
# CONDITIONAL GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        noise_dim,
        embedding_dim,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                embedding_dim
            )
        )

    def forward(self, z):

        return self.network(z)


# ============================================================
# DISCRIMINATOR
# ============================================================

class Discriminator(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.LeakyReLU(0.2),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.LeakyReLU(0.2),

            nn.Linear(
                hidden_dim,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# TRAIN GAN
# ============================================================

def train_gan(
    real_control,
    n_generate,
    fold_number
):

    embedding_dim = real_control.shape[1]

    scaler = StandardScaler()

    real_scaled = scaler.fit_transform(
        real_control
    )

    real_tensor = torch.tensor(
        real_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    generator = Generator(
        NOISE_DIM,
        embedding_dim,
        HIDDEN_DIM
    ).to(DEVICE)

    discriminator = Discriminator(
        embedding_dim,
        HIDDEN_DIM
    ).to(DEVICE)

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

    print(
        f"\n  Training GAN for Fold {fold_number}"
    )

    for epoch in range(1, GAN_EPOCHS + 1):

        # ----------------------------------------------------
        # Train discriminator
        # ----------------------------------------------------

        z = torch.randn(
            len(real_tensor),
            NOISE_DIM,
            device=DEVICE
        )

        fake = generator(z)

        real_labels = torch.ones(
            len(real_tensor),
            1,
            device=DEVICE
        )

        fake_labels = torch.zeros(
            len(real_tensor),
            1,
            device=DEVICE
        )

        d_optimizer.zero_grad()

        real_output = discriminator(
            real_tensor
        )

        fake_output = discriminator(
            fake.detach()
        )

        d_loss_real = criterion(
            real_output,
            real_labels
        )

        d_loss_fake = criterion(
            fake_output,
            fake_labels
        )

        d_loss = (
            d_loss_real +
            d_loss_fake
        ) / 2

        d_loss.backward()

        d_optimizer.step()

        # ----------------------------------------------------
        # Train generator
        # ----------------------------------------------------

        z = torch.randn(
            len(real_tensor),
            NOISE_DIM,
            device=DEVICE
        )

        g_optimizer.zero_grad()

        generated = generator(z)

        output = discriminator(
            generated
        )

        g_loss = criterion(
            output,
            real_labels
        )

        g_loss.backward()

        g_optimizer.step()

        if epoch in [1, 100, 200, 300, 400]:

            print(
                f"    Epoch {epoch:03d} | "
                f"G Loss {g_loss.item():.4f} | "
                f"D Loss {d_loss.item():.4f}"
            )

    # --------------------------------------------------------
    # Generate synthetic embeddings
    # --------------------------------------------------------

    generator.eval()

    with torch.no_grad():

        z = torch.randn(
            n_generate,
            NOISE_DIM,
            device=DEVICE
        )

        synthetic_scaled = (
            generator(z)
            .cpu()
            .numpy()
        )

    synthetic = scaler.inverse_transform(
        synthetic_scaled
    )

    return synthetic


# ============================================================
# CLASSIFIER EVALUATION
# ============================================================

def evaluate_classifier(
    X_train,
    y_train,
    X_val,
    y_val
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    model = LogisticRegression(
        max_iter=3000,
        random_state=SEED
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_val_scaled
    )

    probabilities = model.predict_proba(
        X_val_scaled
    )[:, 1]

    return {
        "Accuracy":
            accuracy_score(
                y_val,
                predictions
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_val,
                predictions
            ),

        "Precision":
            precision_score(
                y_val,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_val,
                predictions,
                zero_division=0
            ),

        "F1_Score":
            f1_score(
                y_val,
                predictions,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y_val,
                probabilities
            )
    }


# ============================================================
# STRATIFIED CV
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

fold_results = []


# ============================================================
# CV LOOP
# ============================================================

for fold_number, (
    train_idx,
    val_idx
) in enumerate(
    skf.split(X, y),
    start=1
):

    print("\n")
    print("=" * 70)
    print(
        f"FOLD {fold_number}/{N_SPLITS}"
    )
    print("=" * 70)

    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]

    X_fold_val = X[val_idx]
    y_fold_val = y[val_idx]

    print(
        "\nFold training samples:",
        len(X_fold_train)
    )

    print(
        "Fold validation samples:",
        len(X_fold_val)
    )

    print("\nTraining class distribution:")

    print(
        pd.Series(
            y_fold_train
        ).value_counts().sort_index()
    )

    print("\nValidation class distribution:")

    print(
        pd.Series(
            y_fold_val
        ).value_counts().sort_index()
    )

    # --------------------------------------------------------
    # Separate real Control samples
    # --------------------------------------------------------

    real_control = X_fold_train[
        y_fold_train == 0
    ]

    real_asd = X_fold_train[
        y_fold_train == 1
    ]

    print(
        "\nReal Control samples in fold:",
        len(real_control)
    )

    print(
        "Real ASD samples in fold:",
        len(real_asd)
    )

    # --------------------------------------------------------
    # Generate synthetic pool
    # --------------------------------------------------------

    max_synthetic = max(
        AUGMENTATION_LEVELS
    )

    synthetic_pool = train_gan(
        real_control,
        max_synthetic,
        fold_number
    )

    print(
        "\nSynthetic pool generated:",
        synthetic_pool.shape
    )

    # --------------------------------------------------------
    # Evaluate augmentation levels
    # --------------------------------------------------------

    for n_synthetic in AUGMENTATION_LEVELS:

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"Fold {fold_number} | "
            f"Synthetic Controls: "
            f"{n_synthetic}"
        )

        if n_synthetic == 0:

            X_aug = X_fold_train.copy()

            y_aug = y_fold_train.copy()

        else:

            synthetic_selected = (
                synthetic_pool[
                    :n_synthetic
                ]
            )

            X_aug = np.vstack([
                X_fold_train,
                synthetic_selected
            ])

            y_aug = np.concatenate([
                y_fold_train,
                np.zeros(
                    n_synthetic,
                    dtype=int
                )
            ])

        metrics = evaluate_classifier(
            X_aug,
            y_aug,
            X_fold_val,
            y_fold_val
        )

        row = {
            "Fold":
                fold_number,

            "Synthetic_Controls":
                n_synthetic,

            "Training_Samples":
                len(X_aug),

            "Validation_Samples":
                len(X_fold_val),

            "Training_ASD":
                int(
                    np.sum(
                        y_aug == 1
                    )
                ),

            "Training_Control":
                int(
                    np.sum(
                        y_aug == 0
                    )
                )
        }

        row.update(metrics)

        fold_results.append(row)

        print(
            f"Accuracy          : "
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{metrics['Balanced_Accuracy']:.4f}"
        )

        print(
            f"Precision         : "
            f"{metrics['Precision']:.4f}"
        )

        print(
            f"Recall            : "
            f"{metrics['Recall']:.4f}"
        )

        print(
            f"F1-score          : "
            f"{metrics['F1_Score']:.4f}"
        )

        print(
            f"ROC-AUC           : "
            f"{metrics['ROC_AUC']:.4f}"
        )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    fold_results
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

metric_columns = [
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

summary_rows = []

for n_synthetic in AUGMENTATION_LEVELS:

    subset = results_df[
        results_df[
            "Synthetic_Controls"
        ] == n_synthetic
    ]

    row = {
        "Synthetic_Controls":
            n_synthetic
    }

    for metric in metric_columns:

        row[
            metric + "_Mean"
        ] = subset[
            metric
        ].mean()

        row[
            metric + "_Std"
        ] = subset[
            metric
        ].std()

    summary_rows.append(row)


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STEP 37 — FINAL CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# IDENTIFY STABLE AUGMENTATION
# ============================================================

print("\n")
print("=" * 70)
print("AUGMENTATION INTERPRETATION")
print("=" * 70)

for _, row in summary_df.iterrows():

    print(
        f"\nSynthetic Controls: "
        f"{int(row['Synthetic_Controls'])}"
    )

    print(
        f"Balanced Accuracy: "
        f"{row['Balanced_Accuracy_Mean']:.4f} "
        f"+/- "
        f"{row['Balanced_Accuracy_Std']:.4f}"
    )

    print(
        f"F1: "
        f"{row['F1_Score_Mean']:.4f} "
        f"+/- "
        f"{row['F1_Score_Std']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{row['ROC_AUC_Mean']:.4f} "
        f"+/- "
        f"{row['ROC_AUC_Std']:.4f}"
    )


# ============================================================
# REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 37 — FOLD-SAFE GENETIC GAN VALIDATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Purpose:\n"
    )

    f.write(
        "Evaluate genetic GAN augmentation using fold-specific "
        "GAN training and stratified cross-validation.\n\n"
    )

    f.write(
        "Important leakage-control principle:\n"
    )

    f.write(
        "For every fold, the GAN is trained only using the "
        "real Control embeddings belonging to that fold's "
        "training partition. Synthetic samples are therefore "
        "not generated using the validation fold.\n\n"
    )

    f.write(
        f"Number of folds: {N_SPLITS}\n"
    )

    f.write(
        f"GAN epochs per fold: {GAN_EPOCHS}\n"
    )

    f.write(
        "Augmentation levels tested: "
        + ", ".join(
            str(x)
            for x in AUGMENTATION_LEVELS
        )
        + "\n\n"
    )

    f.write(
        "FOLD RESULTS\n"
    )

    f.write("-" * 70 + "\n\n")

    f.write(
        results_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "AGGREGATED RESULTS\n"
    )

    f.write("-" * 70 + "\n\n")

    f.write(
        summary_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "INTERPRETATION:\n"
    )

    f.write(
        "The purpose of this experiment is to determine whether "
        "minority-class genetic augmentation provides stable "
        "benefit across independent training/validation folds. "
        "The result should be interpreted jointly with the "
        "synthetic distribution validation and the independent "
        "unseen test evaluation performed previously.\n\n"
    )

    f.write(
        "The final 24-sample unseen test set from the original "
        "train/test split was not used in this cross-validation "
        "experiment.\n"
    )


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(RESULT_FILE)
print(SUMMARY_FILE)
print(REPORT_FILE)

print("\nStep 37 completed.")