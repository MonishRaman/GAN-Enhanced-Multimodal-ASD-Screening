import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix
)

# ============================================================
# PATHS
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
    r"genetics\outputs\objective2_genetic\embedding_gan_utility_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_gan_downstream_utility_results.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_gan_downstream_utility_report.txt"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 35 — GENETIC GAN DOWNSTREAM UTILITY VALIDATION")
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

if len(embedding_cols) != 128:
    raise ValueError(
        f"Expected 128 embedding columns, found {len(embedding_cols)}"
    )

print("\nEmbedding dimensions:", len(embedding_cols))


# ============================================================
# PREPARE ORIGINAL TRAINING DATA
# ============================================================

X_train_real = train_df[embedding_cols].astype(float).values
y_train_real = train_df["Target"].astype(int).values

X_test = test_df[embedding_cols].astype(float).values
y_test = test_df["Target"].astype(int).values


# ============================================================
# PREPARE SYNTHETIC DATA
# ============================================================

# Synthetic file may contain only embedding columns
# or may also contain Target / metadata.

synthetic_embedding_cols = [
    c for c in synthetic_df.columns
    if c.startswith("Genetic_Embedding_")
]

if len(synthetic_embedding_cols) != 128:
    raise ValueError(
        f"Expected 128 synthetic embedding columns, "
        f"found {len(synthetic_embedding_cols)}"
    )

X_synthetic = synthetic_df[
    synthetic_embedding_cols
].astype(float).values

# Synthetic samples are generated specifically as Control samples.
y_synthetic = np.zeros(len(X_synthetic), dtype=int)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET STRUCTURE")
print("=" * 70)

print("Original training samples :", len(X_train_real))
print("Synthetic samples         :", len(X_synthetic))
print("Unseen test samples       :", len(X_test))

print("\nOriginal training classes:")
print(pd.Series(y_train_real).value_counts().sort_index())

print("\nSynthetic class:")
print("Control:", len(y_synthetic))

print("\nUnseen test classes:")
print(pd.Series(y_test).value_counts().sort_index())


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

if (
    np.isnan(X_train_real).any()
    or np.isnan(X_synthetic).any()
    or np.isnan(X_test).any()
):
    raise ValueError("NaN values detected.")

if (
    np.isinf(X_train_real).any()
    or np.isinf(X_synthetic).any()
    or np.isinf(X_test).any()
):
    raise ValueError("Infinite values detected.")

print("\nNumeric validation: PASSED")


# ============================================================
# CLASSIFIER FUNCTION
# ============================================================

def evaluate_model(X_train, y_train, X_test, y_test, experiment_name):

    print("\n" + "-" * 70)
    print(experiment_name)
    print("-" * 70)

    # IMPORTANT:
    # scaler is fitted ONLY on training data.
    # Test data is never used for fitting.

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=3000,
        class_weight=None,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

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

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
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

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1-score          : {f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "Experiment": experiment_name,
        "Training_Samples": len(X_train),
        "Training_ASD": int(np.sum(y_train == 1)),
        "Training_Control": int(np.sum(y_train == 0)),
        "Test_Samples": len(X_test),
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
# EXPERIMENT 1 — ORIGINAL DATA
# ============================================================

baseline_result = evaluate_model(
    X_train_real,
    y_train_real,
    X_test,
    y_test,
    "Original Training Data"
)


# ============================================================
# EXPERIMENT 2 — GAN AUGMENTED DATA
# ============================================================

X_train_augmented = np.vstack([
    X_train_real,
    X_synthetic
])

y_train_augmented = np.concatenate([
    y_train_real,
    y_synthetic
])

print("\n" + "=" * 70)
print("GAN AUGMENTED DATASET")
print("=" * 70)

print("Total training samples :", len(X_train_augmented))
print("ASD                    :", np.sum(y_train_augmented == 1))
print("Control                :", np.sum(y_train_augmented == 0))


augmented_result = evaluate_model(
    X_train_augmented,
    y_train_augmented,
    X_test,
    y_test,
    "GAN Augmented Training Data"
)


# ============================================================
# COMPARISON
# ============================================================

results = pd.DataFrame([
    baseline_result,
    augmented_result
])

print("\n" + "=" * 70)
print("STEP 35 — MODEL COMPARISON")
print("=" * 70)

print(
    results[
        [
            "Experiment",
            "Training_Samples",
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
# IMPROVEMENT
# ============================================================

baseline = results.iloc[0]
augmented = results.iloc[1]

improvements = {
    "Accuracy_Change":
        augmented["Accuracy"] - baseline["Accuracy"],

    "Balanced_Accuracy_Change":
        augmented["Balanced_Accuracy"] - baseline["Balanced_Accuracy"],

    "Precision_Change":
        augmented["Precision"] - baseline["Precision"],

    "Recall_Change":
        augmented["Recall"] - baseline["Recall"],

    "F1_Change":
        augmented["F1_Score"] - baseline["F1_Score"],

    "ROC_AUC_Change":
        augmented["ROC_AUC"] - baseline["ROC_AUC"]
}

print("\nMetric changes after GAN augmentation:")

for metric, value in improvements.items():
    print(f"{metric:30s}: {value:+.4f}")


# ============================================================
# DECISION RULE
# ============================================================

auc_change = improvements["ROC_AUC_Change"]
f1_change = improvements["F1_Change"]
balanced_change = improvements["Balanced_Accuracy_Change"]

if (
    auc_change > 0.02
    and f1_change > 0
    and balanced_change > 0
):
    utility_status = "BENEFICIAL"

elif (
    auc_change >= -0.02
    and f1_change >= -0.02
    and balanced_change >= -0.02
):
    utility_status = "NEUTRAL_OR_ACCEPTABLE"

else:
    utility_status = "HARMFUL"


print("\n" + "=" * 70)
print("FINAL UTILITY ASSESSMENT")
print("=" * 70)

print("Status:", utility_status)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("STEP 35 — GENETIC GAN DOWNSTREAM UTILITY VALIDATION\n")
    f.write("=" * 70 + "\n\n")

    f.write("Objective:\n")
    f.write(
        "Evaluate whether GAN-generated synthetic Control genetic "
        "embeddings improve or preserve downstream ASD classification "
        "performance.\n\n"
    )

    f.write("Data split:\n")
    f.write(
        "The original training set contains 92 samples "
        "(69 ASD, 23 Control).\n"
    )

    f.write(
        "The unseen test set contains 24 samples "
        "(18 ASD, 6 Control).\n"
    )

    f.write(
        "The unseen test set was kept completely isolated from GAN "
        "training and augmentation.\n\n"
    )

    f.write("Synthetic augmentation:\n")
    f.write(
        f"{len(X_synthetic)} synthetic Control embeddings were added "
        "to the original training data.\n"
    )

    f.write(
        "Augmented training dataset: "
        f"{len(X_train_augmented)} samples "
        f"({np.sum(y_train_augmented == 1)} ASD, "
        f"{np.sum(y_train_augmented == 0)} Control).\n\n"
    )

    f.write("MODEL RESULTS\n")
    f.write("-" * 70 + "\n")

    for _, row in results.iterrows():

        f.write(f"\n{row['Experiment']}\n")
        f.write(
            f"Accuracy          : {row['Accuracy']:.4f}\n"
        )
        f.write(
            f"Balanced Accuracy : {row['Balanced_Accuracy']:.4f}\n"
        )
        f.write(
            f"Precision         : {row['Precision']:.4f}\n"
        )
        f.write(
            f"Recall            : {row['Recall']:.4f}\n"
        )
        f.write(
            f"F1-score          : {row['F1_Score']:.4f}\n"
        )
        f.write(
            f"ROC-AUC           : {row['ROC_AUC']:.4f}\n"
        )

        f.write(
            f"Confusion Matrix: "
            f"TN={row['TN']}, "
            f"FP={row['FP']}, "
            f"FN={row['FN']}, "
            f"TP={row['TP']}\n"
        )

    f.write("\n\nMETRIC CHANGES\n")
    f.write("-" * 70 + "\n")

    for metric, value in improvements.items():
        f.write(f"{metric}: {value:+.4f}\n")

    f.write("\n\nFINAL UTILITY ASSESSMENT\n")
    f.write("-" * 70 + "\n")
    f.write(f"Status: {utility_status}\n\n")

    if utility_status == "BENEFICIAL":
        f.write(
            "GAN augmentation improved downstream classification "
            "performance on the untouched unseen test set. "
            "The genetic synthetic augmentation is therefore "
            "supported as useful for Objective 2.\n"
        )

    elif utility_status == "NEUTRAL_OR_ACCEPTABLE":
        f.write(
            "GAN augmentation did not produce a substantial improvement, "
            "but downstream performance remained reasonably stable. "
            "The synthetic embeddings may therefore be retained as "
            "an acceptable augmentation strategy with appropriate caution.\n"
        )

    else:
        f.write(
            "GAN augmentation reduced downstream performance. "
            "The synthetic embeddings should not be considered "
            "validated for downstream augmentation without further "
            "investigation.\n"
        )

print("\nFiles saved:")
print(RESULT_FILE)
print(REPORT_FILE)

print("\nStep 35 completed.")