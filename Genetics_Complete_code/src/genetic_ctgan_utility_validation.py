import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
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
# STEP 38 — CTGAN DOWNSTREAM UTILITY VALIDATION
# ============================================================

BASE = r"D:\org genetical data"


TRAIN_FILE = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "objective2_genetic",
    "train_test_split",
    "genetic_train_1000_features.csv"
)

TEST_FILE = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "objective2_genetic",
    "train_test_split",
    "genetic_test_1000_features_unseen.csv"
)

CTGAN_SYNTHETIC_FILE = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "objective2_genetic",
    "ctgan",
    "genetic_ctgan_synthetic_controls_46.csv"
)

CUSTOM_GAN_FILE = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "objective2_genetic",
    "embedding_gan",
    "synthetic_control_128d_embeddings.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "objective2_genetic",
    "ctgan_utility_validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("STEP 38 — CTGAN DOWNSTREAM UTILITY VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading training data...")
train_df = pd.read_csv(TRAIN_FILE)

print("Loading unseen test data...")
test_df = pd.read_csv(TEST_FILE)

print("Loading CTGAN synthetic data...")
ctgan_df = pd.read_csv(CTGAN_SYNTHETIC_FILE)


# ------------------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------------------

embedding_columns = [
    c for c in train_df.columns
    if c.startswith("Genetic_Embedding_")
]

if len(embedding_columns) != 128:

    raise ValueError(
        f"Expected 128 embeddings, found "
        f"{len(embedding_columns)}"
    )


# ------------------------------------------------------------
# CONVERT NUMERIC
# ------------------------------------------------------------

for df in [
    train_df,
    test_df,
    ctgan_df
]:

    for col in embedding_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

train_df["Target"] = pd.to_numeric(
    train_df["Target"]
)

test_df["Target"] = pd.to_numeric(
    test_df["Target"]
)

ctgan_df["Target"] = pd.to_numeric(
    ctgan_df["Target"]
)


# ------------------------------------------------------------
# BASIC CHECKS
# ------------------------------------------------------------

print("\nTraining samples :", len(train_df))
print("Test samples     :", len(test_df))
print("CTGAN samples    :", len(ctgan_df))

print("\nTraining classes:")
print(train_df["Target"].value_counts())

print("\nTest classes:")
print(test_df["Target"].value_counts())

print("\nCTGAN classes:")
print(ctgan_df["Target"].value_counts())


# ------------------------------------------------------------
# CREATE CLASSIFIER
# ------------------------------------------------------------

def create_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight=None,
                random_state=42
            )
        )
    ])


# ------------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate(
    name,
    training_data
):

    X_train = training_data[
        embedding_columns
    ]

    y_train = training_data[
        "Target"
    ]

    X_test = test_df[
        embedding_columns
    ]

    y_test = test_df[
        "Target"
    ]

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    results = {

        "Experiment":
            name,

        "Training_Samples":
            len(training_data),

        "ASD_Training":
            int((y_train == 1).sum()),

        "Control_Training":
            int((y_train == 0).sum()),

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "F1_Score":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y_test,
                probabilities
            )
    }

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    for key, value in results.items():

        if isinstance(value, float):

            print(
                f"{key:22s}: {value:.4f}"
            )

        else:

            print(
                f"{key:22s}: {value}"
            )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return results


# ------------------------------------------------------------
# 1. BASELINE
# ------------------------------------------------------------

baseline_result = evaluate(
    "Baseline_No_GAN",
    train_df
)


# ------------------------------------------------------------
# 2. CTGAN
# ------------------------------------------------------------

ctgan_training = pd.concat(
    [
        train_df[
            embedding_columns + ["Target"]
        ],

        ctgan_df[
            embedding_columns + ["Target"]
        ]
    ],
    ignore_index=True
)


ctgan_result = evaluate(
    "CTGAN_Augmented_46",
    ctgan_training
)


# ------------------------------------------------------------
# 3. CUSTOM GAN
# ------------------------------------------------------------

if os.path.exists(CUSTOM_GAN_FILE):

    custom_gan = pd.read_csv(
        CUSTOM_GAN_FILE
    )

    for col in embedding_columns:

        custom_gan[col] = pd.to_numeric(
            custom_gan[col],
            errors="coerce"
        )

    custom_gan["Target"] = 0

    custom_training = pd.concat(
        [
            train_df[
                embedding_columns + ["Target"]
            ],

            custom_gan[
                embedding_columns + ["Target"]
            ]
        ],
        ignore_index=True
    )

    custom_result = evaluate(
        "Custom_GAN_Augmented_46",
        custom_training
    )

else:

    print(
        "\nCustom GAN file not found."
    )

    custom_result = None


# ------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------

results = [
    baseline_result,
    ctgan_result
]

if custom_result is not None:

    results.append(
        custom_result
    )


comparison = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# METRIC CHANGES
# ------------------------------------------------------------

baseline = comparison[
    comparison["Experiment"]
    ==
    "Baseline_No_GAN"
].iloc[0]


for metric in [
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]:

    comparison[
        metric + "_Change_vs_Baseline"
    ] = (
        comparison[metric]
        -
        baseline[metric]
    )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    "genetic_ctgan_vs_custom_gan_utility_results.csv"
)

comparison.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# PRINT FINAL COMPARISON
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL CTGAN VS CUSTOM GAN COMPARISON")
print("=" * 70)

print(
    comparison[
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
    ].to_string(
        index=False
    )
)


print("\nResults saved:")
print(output_file)

print("\nStep 38 completed.")