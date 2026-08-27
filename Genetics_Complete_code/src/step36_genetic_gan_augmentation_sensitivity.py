import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# STEP 36 — GENETIC GAN AUGMENTATION SENSITIVITY
# ============================================================

BASE = r"D:\org genetical data"

TRAIN_FILE = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\embedding_gan\genetic_embedding_gan_train.csv"
)

TEST_FILE = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\embedding_gan\genetic_embedding_gan_test_unseen.csv"
)

SYNTHETIC_FILE = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\embedding_gan\synthetic_control_128d_embeddings.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\augmentation_sensitivity"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_gan_augmentation_sensitivity_results.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_gan_augmentation_sensitivity_report.txt"
)


print("=" * 70)
print("STEP 36 — GENETIC GAN AUGMENTATION SENSITIVITY")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading original training embeddings...")
train_df = pd.read_csv(TRAIN_FILE)
print("Done.")

print("\nLoading unseen test embeddings...")
test_df = pd.read_csv(TEST_FILE)
print("Done.")

print("\nLoading synthetic Control embeddings...")
synthetic_df = pd.read_csv(SYNTHETIC_FILE)
print("Done.")


# ============================================================
# IDENTIFY EMBEDDING COLUMNS
# ============================================================

embedding_cols = [
    c for c in train_df.columns
    if c.startswith("Genetic_Embedding_")
]

synthetic_embedding_cols = [
    c for c in synthetic_df.columns
    if c.startswith("Genetic_Embedding_")
]

if len(embedding_cols) != 128:
    raise ValueError(
        f"Training embeddings: expected 128, found {len(embedding_cols)}"
    )

if len(synthetic_embedding_cols) != 128:
    raise ValueError(
        f"Synthetic embeddings: expected 128, found {len(synthetic_embedding_cols)}"
    )


# ============================================================
# PREPARE ARRAYS
# ============================================================

X_train_real = train_df[embedding_cols].astype(float).values
y_train_real = train_df["Target"].astype(int).values

X_test = test_df[embedding_cols].astype(float).values
y_test = test_df["Target"].astype(int).values

X_synthetic_all = synthetic_df[
    synthetic_embedding_cols
].astype(float).values

# All generated samples are Control = 0
y_synthetic_all = np.zeros(
    len(X_synthetic_all),
    dtype=int
)


# ============================================================
# BASIC CHECKS
# ============================================================

print("\n" + "=" * 70)
print("DATASET STRUCTURE")
print("=" * 70)

print("Real training samples :", len(X_train_real))
print("Synthetic samples     :", len(X_synthetic_all))
print("Unseen test samples   :", len(X_test))
print("Embedding dimensions  :", len(embedding_cols))

print("\nReal training class distribution:")
print(
    pd.Series(y_train_real)
    .value_counts()
    .sort_index()
)

print("\nUnseen test class distribution:")
print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)

print("\nSynthetic samples:", len(X_synthetic_all))
print("Synthetic class: Control")


# ============================================================
# NUMERIC VALIDATION
# ============================================================

for name, data in [
    ("Real training", X_train_real),
    ("Synthetic", X_synthetic_all),
    ("Test", X_test)
]:

    if np.isnan(data).any():
        raise ValueError(
            f"NaN values detected in {name} data."
        )

    if np.isinf(data).any():
        raise ValueError(
            f"Infinite values detected in {name} data."
        )

print("\nNumeric validation: PASSED")


# ============================================================
# FIXED RANDOM ORDER
# ============================================================
#
# We use a fixed random permutation so that:
#
# 10-sample experiment = first 10
# 20-sample experiment = first 20
# 30-sample experiment = first 30
# 46-sample experiment = all 46
#
# This makes the sensitivity experiment reproducible.
#
# ============================================================

rng = np.random.RandomState(42)

synthetic_order = rng.permutation(
    len(X_synthetic_all)
)

X_synthetic = X_synthetic_all[
    synthetic_order
]

y_synthetic = y_synthetic_all[
    synthetic_order
]


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    synthetic_count,
    experiment_name
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=3000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "-" * 70)
    print(experiment_name)
    print("-" * 70)

    print(
        f"Training samples   : {len(X_train)}"
    )

    print(
        f"Synthetic Controls : {synthetic_count}"
    )

    print(
        f"ASD                : {np.sum(y_train == 1)}"
    )

    print(
        f"Control            : {np.sum(y_train == 0)}"
    )

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy  : {balanced_accuracy:.4f}"
    )

    print(
        f"Precision          : {precision:.4f}"
    )

    print(
        f"Recall             : {recall:.4f}"
    )

    print(
        f"F1-score           : {f1:.4f}"
    )

    print(
        f"ROC-AUC            : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "Experiment": experiment_name,
        "Synthetic_Controls": synthetic_count,
        "Total_Training_Samples": len(X_train),
        "Training_ASD": int(np.sum(y_train == 1)),
        "Training_Control": int(np.sum(y_train == 0)),
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# ============================================================
# AUGMENTATION LEVELS
# ============================================================

augmentation_levels = [
    0,
    10,
    20,
    30,
    46
]

results = []


# ============================================================
# RUN EXPERIMENTS
# ============================================================

for n_synthetic in augmentation_levels:

    if n_synthetic == 0:

        X_aug = X_train_real.copy()
        y_aug = y_train_real.copy()

        name = "Baseline_No_GAN"

    else:

        X_selected = X_synthetic[
            :n_synthetic
        ]

        y_selected = y_synthetic[
            :n_synthetic
        ]

        X_aug = np.vstack([
            X_train_real,
            X_selected
        ])

        y_aug = np.concatenate([
            y_train_real,
            y_selected
        ])

        name = (
            f"GAN_Augmented_{n_synthetic}_Synthetic_Control"
        )

    result = evaluate_experiment(
        X_aug,
        y_aug,
        X_test,
        y_test,
        n_synthetic,
        name
    )

    results.append(result)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("STEP 36 — AUGMENTATION COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "Experiment",
            "Synthetic_Controls",
            "Total_Training_Samples",
            "Accuracy",
            "Balanced_Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC"
        ]
    ].to_string(index=False)
)


# ============================================================
# CHANGE FROM BASELINE
# ============================================================

baseline = results_df.iloc[0]

results_df["Accuracy_Change"] = (
    results_df["Accuracy"]
    - baseline["Accuracy"]
)

results_df["Balanced_Accuracy_Change"] = (
    results_df["Balanced_Accuracy"]
    - baseline["Balanced_Accuracy"]
)

results_df["Precision_Change"] = (
    results_df["Precision"]
    - baseline["Precision"]
)

results_df["Recall_Change"] = (
    results_df["Recall"]
    - baseline["Recall"]
)

results_df["F1_Change"] = (
    results_df["F1_Score"]
    - baseline["F1_Score"]
)

results_df["ROC_AUC_Change"] = (
    results_df["ROC_AUC"]
    - baseline["ROC_AUC"]
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
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
        "STEP 36 — GENETIC GAN AUGMENTATION SENSITIVITY\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Purpose:\n"
    )

    f.write(
        "Evaluate how different quantities of GAN-generated "
        "Control embeddings affect downstream genetic "
        "classification performance.\n\n"
    )

    f.write(
        "Original training data:\n"
    )

    f.write(
        "69 ASD + 23 Control = 92 samples.\n\n"
    )

    f.write(
        "Synthetic augmentation pool:\n"
    )

    f.write(
        f"{len(X_synthetic_all)} synthetic Control embeddings.\n\n"
    )

    f.write(
        "Unseen test data:\n"
    )

    f.write(
        "18 ASD + 6 Control = 24 real samples.\n\n"
    )

    f.write(
        "Important methodological statement:\n"
    )

    f.write(
        "The 24-sample test set was not used for GAN training. "
        "The GAN was trained using only the original training "
        "set. The test set remained unseen during synthetic "
        "sample generation.\n\n"
    )

    f.write(
        "A fixed random ordering of the synthetic samples was "
        "used so that the augmentation levels were reproducible.\n\n"
    )

    f.write(
        "RESULTS\n"
    )

    f.write("-" * 70 + "\n\n")

    for _, row in results_df.iterrows():

        f.write(
            f"{row['Experiment']}\n"
        )

        f.write(
            f"Synthetic Controls : "
            f"{row['Synthetic_Controls']}\n"
        )

        f.write(
            f"Training Samples   : "
            f"{row['Total_Training_Samples']}\n"
        )

        f.write(
            f"Accuracy           : "
            f"{row['Accuracy']:.4f}\n"
        )

        f.write(
            f"Balanced Accuracy  : "
            f"{row['Balanced_Accuracy']:.4f}\n"
        )

        f.write(
            f"Precision          : "
            f"{row['Precision']:.4f}\n"
        )

        f.write(
            f"Recall             : "
            f"{row['Recall']:.4f}\n"
        )

        f.write(
            f"F1-score           : "
            f"{row['F1_Score']:.4f}\n"
        )

        f.write(
            f"ROC-AUC            : "
            f"{row['ROC_AUC']:.4f}\n"
        )

        f.write(
            f"Accuracy Change    : "
            f"{row['Accuracy_Change']:+.4f}\n"
        )

        f.write(
            f"Balanced Acc Change: "
            f"{row['Balanced_Accuracy_Change']:+.4f}\n"
        )

        f.write(
            f"F1 Change          : "
            f"{row['F1_Change']:+.4f}\n"
        )

        f.write(
            f"ROC-AUC Change     : "
            f"{row['ROC_AUC_Change']:+.4f}\n\n"
        )


    f.write(
        "INTERPRETATION\n"
    )

    f.write("-" * 70 + "\n\n")

    f.write(
        "This experiment is a sensitivity analysis of the "
        "genetic GAN augmentation level. It should not be "
        "interpreted as test-set hyperparameter optimization. "
        "The unseen test set is retained as an independent "
        "evaluation set for reporting generalization behavior.\n\n"
    )

    f.write(
        "The results should be considered together with the "
        "synthetic-data distribution validation from Step 34 "
        "and the downstream utility validation from Step 35.\n"
    )


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(RESULT_FILE)
print(REPORT_FILE)

print("\nStep 36 completed.")

