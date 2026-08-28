import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from tensorflow.keras.models import load_model


# ============================================================
# 1. Load test data
# ============================================================

X_test = pd.read_csv(
    "X_test_selected20.csv"
).values

y_test = pd.read_csv(
    "y_test_selected20.csv"
).iloc[:, 0].values


# ============================================================
# 2. Load trained environmental model
# ============================================================

model = load_model(
    "environmental_representation_model.keras"
)


# ============================================================
# 3. Generate probabilities
# ============================================================

probabilities = model.predict(
    X_test,
    verbose=0
).ravel()


# ============================================================
# 4. Overall ranking metrics
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

print("\n==========================================")
print("THRESHOLD ANALYSIS")
print("==========================================")

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")


# ============================================================
# 5. Test different thresholds
# ============================================================

results = []

for threshold in np.arange(
    0.05,
    0.96,
    0.05
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

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

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


results_df = pd.DataFrame(results)


# ============================================================
# 6. Print results
# ============================================================

print("\nThreshold results:")
print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 7. Best F1
# ============================================================

best = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n==========================================")
print("BEST F1 THRESHOLD")
print("==========================================")

print(best)


# ============================================================
# 8. Save
# ============================================================

results_df.to_csv(
    "environmental_threshold_analysis.csv",
    index=False
)

print(
    "\nSaved: "
    "environmental_threshold_analysis.csv"
)