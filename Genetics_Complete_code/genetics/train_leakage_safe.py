import os
import random

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from torch.utils.data import DataLoader

from config import *
from dataset import GeneticDataset
from model import GeneticTransformer


# ==========================================================
# CONFIGURATION
# ==========================================================

# IMPORTANT:
# Change this to the EXACT CSV produced by Step 3.
#
# It must be the dataset AFTER:
#     >75% missing probe filtering
#
# and BEFORE:
#     KNN imputation
#     Z-score normalization
#
INPUT_FILE = r"data/processed/filtered_expression_matrix.csv"

TOP_FEATURES = 1000

KNN_NEIGHBORS = 5

MI_RANDOM_STATE = 42


OUTPUT_DIR = "genetics/outputs/leakage_safe"

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading Step 3 filtered dataset...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")


# ==========================================================
# IDENTIFY SAMPLE COLUMNS
# ==========================================================

# The Step 3 expression dataset normally contains:
#
# Position | GSM386518 | GSM386519 | ...
#
# We transpose it so that:
#
# rows    = samples
# columns = probes
#
# ----------------------------------------------------------

if "Position" in df.columns:

    feature_table = df.set_index("Position")

else:

    feature_table = df.copy()


# ----------------------------------------------------------
# Load labels
# ----------------------------------------------------------

LABEL_FILE = r"data/processed/labels.csv"

labels_df = pd.read_csv(
    LABEL_FILE
)


# Create label dictionary

label_map = dict(
    zip(
        labels_df["GSM_ID"],
        labels_df["Target"]
    )
)


# ----------------------------------------------------------
# Keep only samples with labels
# ----------------------------------------------------------

available_samples = [

    sample

    for sample in feature_table.columns

    if sample in label_map
]


print(
    "Samples found:",
    len(available_samples)
)


# ----------------------------------------------------------
# Transpose
# ----------------------------------------------------------

X_df = feature_table[
    available_samples
].T


# ----------------------------------------------------------
# Labels
# ----------------------------------------------------------

y = np.array(

    [
        label_map[sample]
        for sample in available_samples
    ],

    dtype=np.int64
)


# ----------------------------------------------------------
# Remove non-numeric columns if any
# ----------------------------------------------------------

X_df = X_df.apply(
    pd.to_numeric,
    errors="coerce"
)


X = X_df.values.astype(
    np.float32
)


print()

print("=" * 60)

print("RAW CV DATASET")

print("=" * 60)

print(
    "Samples  :",
    X.shape[0]
)

print(
    "Features :",
    X.shape[1]
)

print(
    "ASD      :",
    np.sum(y == 1)
)

print(
    "Control  :",
    np.sum(y == 0)
)

print()


# ==========================================================
# CROSS VALIDATION
# ==========================================================

skf = StratifiedKFold(

    n_splits=NUM_FOLDS,

    shuffle=True,

    random_state=RANDOM_STATE
)


fold_results = []

oof_predictions = np.zeros(
    len(y),
    dtype=np.int64
)

oof_probabilities = np.zeros(
    len(y),
    dtype=np.float32
)

oof_embeddings = np.zeros(
    (
        len(y),
        EMBEDDING_DIM
    ),
    dtype=np.float32
)


# ==========================================================
# METRIC FUNCTION
# ==========================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0.0

    )


    metrics = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "specificity":
            specificity,

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob
            )

    }

    return metrics, cm


# ==========================================================
# TRAINING FUNCTION
# ==========================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0


    for X_batch, y_batch in loader:

        X_batch = X_batch.to(DEVICE)

        y_batch = y_batch.to(DEVICE)


        optimizer.zero_grad()


        logits, _ = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        total_loss += loss.item()


        predictions = torch.argmax(
            logits,
            dim=1
        )


        correct += (
            predictions == y_batch
        ).sum().item()


        total += len(
            y_batch
        )


    return (
        total_loss / len(loader),
        correct / total
    )


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_model(
    model,
    loader
):

    model.eval()

    y_true = []

    y_pred = []

    y_prob = []

    embeddings = []


    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )


            logits, embedding = model(
                X_batch
            )


            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]


            predictions = torch.argmax(
                logits,
                dim=1
            )


            y_true.extend(
                y_batch.numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

            y_prob.extend(
                probabilities.cpu().numpy()
            )

            embeddings.append(
                embedding.cpu().numpy()
            )


    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )

    y_prob = np.array(
        y_prob
    )

    embeddings = np.vstack(
        embeddings
    )


    return (
        y_true,
        y_pred,
        y_prob,
        embeddings
    )


# ==========================================================
# FOLD LOOP
# ==========================================================

for fold, (
    train_idx,
    val_idx
) in enumerate(

    skf.split(X, y),

    start=1

):

    print()

    print("=" * 60)

    print(
        f"FOLD {fold}/{NUM_FOLDS}"
    )

    print("=" * 60)


    # ------------------------------------------------------
    # Split raw data
    # ------------------------------------------------------

    X_train_raw = X[
        train_idx
    ]

    X_val_raw = X[
        val_idx
    ]

    y_train = y[
        train_idx
    ]

    y_val = y[
        val_idx
    ]


    print(
        "Training samples   :",
        len(train_idx)
    )

    print(
        "Validation samples :",
        len(val_idx)
    )


    # ======================================================
    # STEP A — KNN IMPUTATION
    # ======================================================

    print()

    print(
        "Fitting KNN imputer on training fold..."
    )


    imputer = KNNImputer(

        n_neighbors=KNN_NEIGHBORS,

        weights="uniform"
    )


    X_train_imputed = imputer.fit_transform(
        X_train_raw
    )


    X_val_imputed = imputer.transform(
        X_val_raw
    )


    print(
        "KNN imputation completed."
    )


    # ======================================================
    # STEP B — STANDARDIZATION
    # ======================================================

    print()

    print(
        "Fitting StandardScaler on training fold..."
    )


    scaler = StandardScaler()


    X_train_scaled = scaler.fit_transform(
        X_train_imputed
    )


    X_val_scaled = scaler.transform(
        X_val_imputed
    )


    print(
        "Standardization completed."
    )


    # ======================================================
    # STEP C — MUTUAL INFORMATION
    # ======================================================

    print()

    print(
        "Computing Mutual Information on training fold..."
    )


    mi_scores = mutual_info_classif(

        X_train_scaled,

        y_train,

        random_state=MI_RANDOM_STATE

    )


    feature_names = X_df.columns


    mi_df = pd.DataFrame({

        "Feature":
            feature_names,

        "MI_Score":
            mi_scores

    })


    mi_df = mi_df.sort_values(

        "MI_Score",

        ascending=False

    )


    selected_features = (

        mi_df

        .head(TOP_FEATURES)

        ["Feature"]

        .tolist()

    )


    selected_indices = [

        feature_names.get_loc(
            feature
        )

        for feature in selected_features

    ]


    print(
        "Selected features:",
        len(selected_indices)
    )


    # ======================================================
    # STEP D — APPLY FEATURE SELECTION
    # ======================================================

    X_train_selected = X_train_scaled[
        :,
        selected_indices
    ]


    X_val_selected = X_val_scaled[
        :,
        selected_indices
    ]


    # ======================================================
    # SAVE SELECTED FEATURES
    # ======================================================

    selected_path = os.path.join(

        OUTPUT_DIR,

        f"fold_{fold}_selected_features.csv"

    )


    mi_df.head(
        TOP_FEATURES
    ).to_csv(

        selected_path,

        index=False

    )


    # ======================================================
    # DATASETS
    # ======================================================

    train_dataset = GeneticDataset(

        X_train_selected,

        y_train

    )


    val_dataset = GeneticDataset(

        X_val_selected,

        y_val

    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False

    )


    # ======================================================
    # MODEL
    # ======================================================

    model = GeneticTransformer(

        num_genes=TOP_FEATURES,

        embedding_dim=EMBEDDING_DIM,

        num_heads=NUM_HEADS,

        num_layers=NUM_LAYERS,

        feed_forward_dim=FEED_FORWARD_DIM,

        dropout=DROPOUT,

        num_classes=NUM_CLASSES

    ).to(DEVICE)


    # ======================================================
    # CLASS WEIGHTS
    # ======================================================

    class_counts = np.bincount(

        y_train,

        minlength=NUM_CLASSES

    )


    class_weights = (

        len(y_train)

        /

        (
            NUM_CLASSES
            *
            class_counts
        )

    )


    class_weights = torch.tensor(

        class_weights,

        dtype=torch.float32,

        device=DEVICE

    )


    criterion = torch.nn.CrossEntropyLoss(

        weight=class_weights

    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=5

    )


    # ======================================================
    # TRAINING
    # ======================================================

    best_auc = -np.inf

    best_state = None

    patience_counter = 0


    for epoch in range(

        1,

        EPOCHS + 1

    ):

        train_loss, train_acc = train_epoch(

            model,

            train_loader,

            optimizer,

            criterion

        )


        (
            temp_true,
            temp_pred,
            temp_prob,
            temp_embeddings

        ) = evaluate_model(

            model,

            val_loader

        )


        temp_auc = roc_auc_score(

            temp_true,

            temp_prob

        )


        scheduler.step(
            temp_auc
        )


        if temp_auc > best_auc:

            best_auc = temp_auc

            best_state = {

                k: v.detach()
                .cpu()
                .clone()

                for k, v
                in model.state_dict().items()

            }

            patience_counter = 0

        else:

            patience_counter += 1


        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(

                f"Epoch {epoch:03d} | "

                f"Loss {train_loss:.4f} | "

                f"Train Acc {train_acc:.4f} | "

                f"Val AUC {temp_auc:.4f}"

            )


        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "Early stopping."
            )

            break


    # ======================================================
    # RESTORE BEST MODEL
    # ======================================================

    model.load_state_dict(
        best_state
    )


    # ======================================================
    # FINAL FOLD EVALUATION
    # ======================================================

    (
        fold_true,
        fold_pred,
        fold_prob,
        fold_embeddings

    ) = evaluate_model(

        model,

        val_loader

    )


    metrics, cm = calculate_metrics(

        fold_true,

        fold_pred,

        fold_prob

    )


    metrics["fold"] = fold


    fold_results.append(
        metrics
    )


    # ======================================================
    # SAVE OOF RESULTS
    # ======================================================

    oof_predictions[
        val_idx
    ] = fold_pred


    oof_probabilities[
        val_idx
    ] = fold_prob


    oof_embeddings[
        val_idx
    ] = fold_embeddings


    # ======================================================
    # SAVE MODEL
    # ======================================================

    checkpoint_path = os.path.join(

        CHECKPOINT_DIR,

        f"fold_{fold}.pth"

    )


    torch.save(

        model.state_dict(),

        checkpoint_path

    )


    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print()

    print(
        "Fold Results:"
    )


    for key, value in metrics.items():

        if key != "fold":

            print(

                f"{key:20s}: "
                f"{value:.4f}"

            )


    print()

    print(
        "Confusion Matrix:"
    )

    print(cm)


# ==========================================================
# SAVE FOLD METRICS
# ==========================================================

results_df = pd.DataFrame(
    fold_results
)


results_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "fold_metrics.csv"

    ),

    index=False

)


# ==========================================================
# OUT-OF-FOLD PREDICTIONS
# ==========================================================

oof_df = pd.DataFrame({

    "GSM_ID":
        available_samples,

    "Target":
        y,

    "Prediction":
        oof_predictions,

    "Probability_ASD":
        oof_probabilities

})


oof_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "out_of_fold_predictions.csv"

    ),

    index=False

)


# ==========================================================
# OUT-OF-FOLD EMBEDDINGS
# ==========================================================

np.save(

    os.path.join(

        OUTPUT_DIR,

        "out_of_fold_embeddings.npy"

    ),

    oof_embeddings

)


# ==========================================================
# OVERALL OOF METRICS
# ==========================================================

oof_metrics, oof_cm = calculate_metrics(

    y,

    oof_predictions,

    oof_probabilities

)


# ==========================================================
# SUMMARY
# ==========================================================

print()

print("=" * 60)

print(
    "LEAKAGE-SAFE 5-FOLD SUMMARY"
)

print("=" * 60)

print()


for metric in [

    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc"

]:

    mean = results_df[
        metric
    ].mean()

    std = results_df[
        metric
    ].std()


    print(

        f"{metric:20s}: "
        f"{mean:.4f} ± {std:.4f}"

    )


print()

print("=" * 60)

print(
    "OUT-OF-FOLD OVERALL RESULTS"
)

print("=" * 60)

print()


for metric, value in oof_metrics.items():

    print(

        f"{metric:20s}: "
        f"{value:.4f}"

    )


print()

print(
    "Out-of-Fold Confusion Matrix:"
)

print(oof_cm)


print()

print(
    "Leakage-safe training completed."
)

print(
    "Results saved to:",
    OUTPUT_DIR
)