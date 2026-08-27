import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# ============================================================
# STEP 15
# Evaluate 32D Environmental Embedding
# ============================================================

TRAIN_FILE = "environmental_embedding_train_32d_with_target.csv"
TEST_FILE = "environmental_embedding_test_32d_with_target.csv"

RESULT_FILE = "environmental_32d_performance.csv"
THRESHOLD_FILE = "environmental_32d_threshold_analysis.csv"

ROC_PLOT = "environmental_32d_roc_curve.png"
PR_PLOT = "environmental_32d_precision_recall_curve.png"
CM_PLOT = "environmental_32d_confusion_matrix.png"


# ============================================================
# 1. Load data
# ============================================================

print("=" * 70)
print("ENVIRONMENTAL 32D MODEL PERFORMANCE EVALUATION")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nTraining shape:")
print(train_df.shape)

print("\nTesting shape:")
print(test_df.shape)


# ============================================================
# 2. Separate features and target
# ============================================================

FEATURE_COLUMNS = [
    f"env_emb_{i}"
    for i in range(1, 33)
]

TARGET = "k2q35a"

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET]


# ============================================================
# 3. Validate data
# ============================================================

print("\nFeature shape:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\nTarget shape:")
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nMissing values:")
print("Train:", X_train.isna().sum().sum())
print("Test :", X_test.isna().sum().sum())

print("\nTarget distribution - Training:")
print(y_train.value_counts())

print("\nTarget distribution - Testing:")
print(y_test.value_counts())


# ============================================================
# 4. Train Logistic Regression
# ============================================================

print("\n" + "=" * 70)
print("TRAINING 32D LOGISTIC REGRESSION")
print("=" * 70)

model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nModel training completed.")


# ============================================================
# 5. Prediction probabilities
# ============================================================

y_probability = model.predict_proba(
    X_test
)[:, 1]

# Default threshold
DEFAULT_THRESHOLD = 0.50

y_pred = (
    y_probability >= DEFAULT_THRESHOLD
).astype(int)


# ============================================================
# 6. Calculate performance
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


# ============================================================
# 7. Display results
# ============================================================

print("\n" + "=" * 70)
print("32D ENVIRONMENTAL MODEL TEST RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


# ============================================================
# 8. Classification report
# ============================================================

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 9. Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 10. Save confusion matrix plot
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Environmental 32D Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    ["Non-ASD", "ASD"]
)

plt.yticks(
    [0, 1],
    ["Non-ASD", "ASD"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    CM_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# 11. ROC Curve
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Environmental 32D ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ROC_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# 12. Precision-Recall Curve
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_probability
    )
)

plt.figure(figsize=(7, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Environmental 32D Precision-Recall Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PR_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# 13. Threshold analysis
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = np.arange(
    0.05,
    0.96,
    0.05
)

threshold_results = []

for threshold in thresholds:

    predictions = (
        y_probability >= threshold
    ).astype(int)

    p = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    r = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Precision": p,
        "Recall": r,
        "F1": f
    })

threshold_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# 14. Best threshold
# ============================================================

best_index = threshold_df["F1"].idxmax()

best_threshold = threshold_df.loc[
    best_index
]

print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(best_threshold)


# ============================================================
# 15. Save threshold results
# ============================================================

threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False
)


# ============================================================
# 16. Save performance summary
# ============================================================

performance = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
        "Best_Threshold",
        "Best_Threshold_Precision",
        "Best_Threshold_Recall",
        "Best_Threshold_F1"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc,
        best_threshold["Threshold"],
        best_threshold["Precision"],
        best_threshold["Recall"],
        best_threshold["F1"]
    ]
})

performance.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 17. Final output
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(RESULT_FILE)
print(THRESHOLD_FILE)
print(ROC_PLOT)
print(PR_PLOT)
print(CM_PLOT)

print("\n" + "=" * 70)
print("32D ENVIRONMENTAL MODEL EVALUATION COMPLETED")
print("=" * 70)