import os
import random

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
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
# Reproducibility
# ==========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


# ==========================================================
# Directories
# ==========================================================

OUTPUT_DIR = "genetics/outputs"

CHECKPOINT_DIR = "genetics/checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==========================================================
# Load Data
# ==========================================================

print("Loading genetic dataset...")

df = pd.read_csv(DATA_PATH)

print("Done.\n")


# ==========================================================
# Feature Selection
# ==========================================================

feature_columns = [

    c for c in df.columns

    if c not in METADATA_COLUMNS

]

X = df[feature_columns].values.astype(
    np.float32
)

y = df[TARGET_COLUMN].values.astype(
    np.int64
)


print("=" * 60)

print("DATASET")

print("=" * 60)

print("Samples  :", X.shape[0])

print("Features :", X.shape[1])

print("ASD      :", np.sum(y == 1))

print("Control  :", np.sum(y == 0))

print()


# ==========================================================
# Cross Validation
# ==========================================================

skf = StratifiedKFold(

    n_splits=NUM_FOLDS,

    shuffle=True,

    random_state=RANDOM_STATE
)


fold_results = []


# ==========================================================
# Training Function
# ==========================================================

def train_one_fold(

    model,

    train_loader,

    optimizer,

    criterion

):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)

        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(X_batch)

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

        total += len(y_batch)

    return (

        total_loss / len(train_loader),

        correct / total

    )


# ==========================================================
# Evaluation Function
# ==========================================================

def evaluate(

    model,

    loader

):

    model.eval()

    y_true = []

    y_pred = []

    y_prob = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            logits, _ = model(X_batch)

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

    y_true = np.array(y_true)

    y_pred = np.array(y_pred)

    y_prob = np.array(y_prob)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0

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
# Fold Loop
# ==========================================================

for fold, (train_idx, val_idx) in enumerate(

    skf.split(X, y),

    start=1

):

    print("=" * 60)

    print(f"FOLD {fold}/{NUM_FOLDS}")

    print("=" * 60)


    X_train = X[train_idx]

    y_train = y[train_idx]

    X_val = X[val_idx]

    y_val = y[val_idx]


    train_dataset = GeneticDataset(
        X_train,
        y_train
    )

    val_dataset = GeneticDataset(
        X_val,
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


    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = GeneticTransformer(

        num_genes=X.shape[1],

        embedding_dim=EMBEDDING_DIM,

        num_heads=NUM_HEADS,

        num_layers=NUM_LAYERS,

        feed_forward_dim=FEED_FORWARD_DIM,

        dropout=DROPOUT,

        num_classes=NUM_CLASSES

    ).to(DEVICE)


    # ------------------------------------------------------
    # Class-weighted loss
    # ------------------------------------------------------

    train_counts = np.bincount(
        y_train,
        minlength=NUM_CLASSES
    )

    class_weights = (

        len(y_train)
        /
        (
            NUM_CLASSES
            *
            train_counts
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


    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    best_auc = -np.inf

    best_state = None

    patience_counter = 0


    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_fold(

            model,

            train_loader,

            optimizer,

            criterion

        )


        val_metrics, _ = evaluate(

            model,

            val_loader

        )


        val_auc = val_metrics["roc_auc"]


        scheduler.step(val_auc)


        if val_auc > best_auc:

            best_auc = val_auc

            best_state = {

                k: v.detach().cpu().clone()

                for k, v in model.state_dict().items()

            }

            patience_counter = 0

        else:

            patience_counter += 1


        if epoch == 1 or epoch % 10 == 0:

            print(

                f"Epoch {epoch:03d} | "

                f"Loss {train_loss:.4f} | "

                f"Train Acc {train_acc:.4f} | "

                f"Val AUC {val_auc:.4f}"

            )


        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print(

                f"Early stopping at epoch {epoch}"

            )

            break


    # ------------------------------------------------------
    # Restore Best Model
    # ------------------------------------------------------

    model.load_state_dict(
        best_state
    )


    # ------------------------------------------------------
    # Final Fold Evaluation
    # ------------------------------------------------------

    metrics, cm = evaluate(

        model,

        val_loader

    )


    metrics["fold"] = fold

    fold_results.append(metrics)


    print()

    print("Fold Results:")

    for key, value in metrics.items():

        if key != "fold":

            print(
                f"{key:20s}: {value:.4f}"
            )

    print()

    print("Confusion Matrix:")

    print(cm)


    # ------------------------------------------------------
    # Save Fold Model
    # ------------------------------------------------------

    checkpoint_path = os.path.join(

        CHECKPOINT_DIR,

        f"genetic_transformer_fold_{fold}.pth"

    )

    torch.save(

        model.state_dict(),

        checkpoint_path

    )


# ==========================================================
# Results
# ==========================================================

results_df = pd.DataFrame(
    fold_results
)

results_path = os.path.join(

    OUTPUT_DIR,

    "baseline_cv_results.csv"

)

results_df.to_csv(

    results_path,

    index=False

)


# ==========================================================
# Summary
# ==========================================================

print()

print("=" * 60)

print("5-FOLD CROSS-VALIDATION SUMMARY")

print("=" * 60)

print()

metric_columns = [

    "accuracy",

    "balanced_accuracy",

    "precision",

    "recall",

    "specificity",

    "f1",

    "roc_auc"

]

for metric in metric_columns:

    mean = results_df[metric].mean()

    std = results_df[metric].std()

    print(

        f"{metric:20s}: "

        f"{mean:.4f} ± {std:.4f}"

    )

print()

print(

    "Results saved to:",

    results_path

)