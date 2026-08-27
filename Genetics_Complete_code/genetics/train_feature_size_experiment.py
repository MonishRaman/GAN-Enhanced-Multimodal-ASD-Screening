import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
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


# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

NUM_FOLDS = 5

KNN_NEIGHBORS = 5

FEATURE_SIZES = [
    25,
    50,
    100,
    250,
    500,
    1000
]

INPUT_FILE = r"data/processed/filtered_expression_matrix.csv"

LABEL_FILE = r"data/processed/labels.csv"

OUTPUT_DIR = "genetics/outputs/feature_size_experiment"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)


set_seed(RANDOM_STATE)


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading Step 3 filtered dataset...")

df = pd.read_csv(INPUT_FILE)

print("Done.")


# ==========================================================
# PREPARE EXPRESSION MATRIX
# ==========================================================

if "Position" in df.columns:

    feature_table = df.set_index("Position")

else:

    feature_table = df.copy()


# ==========================================================
# LOAD LABELS
# ==========================================================

print("Loading labels...")

labels_df = pd.read_csv(
    LABEL_FILE
)

print("Done.")


label_map = dict(
    zip(
        labels_df["GSM_ID"],
        labels_df["Target"]
    )
)


# ==========================================================
# MATCH SAMPLES
# ==========================================================

available_samples = [

    sample

    for sample in feature_table.columns

    if sample in label_map

]


X_df = feature_table[
    available_samples
].T


X_df = X_df.apply(
    pd.to_numeric,
    errors="coerce"
)


X = X_df.values.astype(
    np.float32
)


y = np.array(

    [
        label_map[sample]
        for sample in available_samples
    ],

    dtype=np.int64
)


feature_names = np.array(
    X_df.columns
)


# ==========================================================
# DATASET SUMMARY
# ==========================================================

print()

print("=" * 60)

print("FEATURE SIZE EXPERIMENT")

print("=" * 60)

print(
    "Samples  :",
    X.shape[0]
)

print(
    "Original Features :",
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

print(
    "Feature sizes:",
    FEATURE_SIZES
)

print()


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


    return {

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


# ==========================================================
# CROSS VALIDATION
# ==========================================================

skf = StratifiedKFold(

    n_splits=NUM_FOLDS,

    shuffle=True,

    random_state=RANDOM_STATE

)


all_results = []


# ==========================================================
# FEATURE SIZE LOOP
# ==========================================================

for feature_size in FEATURE_SIZES:

    print()

    print("=" * 60)

    print(
        f"TOP {feature_size} FEATURES"
    )

    print("=" * 60)


    fold_results = []


    # ======================================================
    # FOLD LOOP
    # ======================================================

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(

        skf.split(X, y),

        start=1

    ):

        print()

        print(
            f"Fold {fold}/{NUM_FOLDS}"
        )


        # --------------------------------------------------
        # RAW SPLIT
        # --------------------------------------------------

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


        # ==================================================
        # KNN IMPUTATION
        # ==================================================

        imputer = KNNImputer(

            n_neighbors=KNN_NEIGHBORS,

            weights="uniform"

        )


        X_train_imputed = (

            imputer.fit_transform(
                X_train_raw
            )

        )


        X_val_imputed = (

            imputer.transform(
                X_val_raw
            )

        )


        # ==================================================
        # STANDARDIZATION
        # ==================================================

        scaler = StandardScaler()


        X_train_scaled = (

            scaler.fit_transform(
                X_train_imputed
            )

        )


        X_val_scaled = (

            scaler.transform(
                X_val_imputed
            )

        )


        # ==================================================
        # MUTUAL INFORMATION
        # ==================================================

        mi_scores = mutual_info_classif(

            X_train_scaled,

            y_train,

            random_state=RANDOM_STATE

        )


        # ==================================================
        # SELECT TOP K
        # ==================================================

        top_indices = (

            np.argsort(
                mi_scores
            )[::-1][:feature_size]

        )


        X_train_selected = (

            X_train_scaled[
                :,
                top_indices
            ]

        )


        X_val_selected = (

            X_val_scaled[
                :,
                top_indices
            ]

        )


        # ==================================================
        # LOGISTIC REGRESSION
        # ==================================================

        model = LogisticRegression(

            C=1.0,

            class_weight="balanced",

            max_iter=3000,

            solver="liblinear",

            random_state=RANDOM_STATE

        )


        model.fit(

            X_train_selected,

            y_train

        )


        # ==================================================
        # PREDICTION
        # ==================================================

        predictions = model.predict(

            X_val_selected

        )


        probabilities = model.predict_proba(

            X_val_selected
        )[:, 1]


        # ==================================================
        # METRICS
        # ==================================================

        metrics = calculate_metrics(

            y_val,

            predictions,

            probabilities

        )


        metrics["feature_size"] = feature_size

        metrics["fold"] = fold


        fold_results.append(
            metrics
        )

        all_results.append(
            metrics
        )


        print(

            f"Accuracy       : "
            f"{metrics['accuracy']:.4f}"

        )

        print(

            f"Balanced Acc.  : "
            f"{metrics['balanced_accuracy']:.4f}"

        )

        print(

            f"F1             : "
            f"{metrics['f1']:.4f}"

        )

        print(

            f"ROC-AUC        : "
            f"{metrics['roc_auc']:.4f}"

        )


    # ======================================================
    # SIZE SUMMARY
    # ======================================================

    size_df = pd.DataFrame(
        fold_results
    )


    print()

    print(
        f"TOP {feature_size} SUMMARY"
    )

    print("-" * 60)


    for metric in [

        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc"

    ]:

        mean = size_df[
            metric
        ].mean()

        std = size_df[
            metric
        ].std()


        print(

            f"{metric:20s}: "
            f"{mean:.4f} ± {std:.4f}"

        )


# ==========================================================
# SAVE ALL RESULTS
# ==========================================================

all_results_df = pd.DataFrame(
    all_results
)


all_results_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "feature_size_fold_results.csv"

    ),

    index=False

)


# ==========================================================
# SUMMARY TABLE
# ==========================================================

summary_rows = []


for feature_size in FEATURE_SIZES:

    subset = all_results_df[

        all_results_df[
            "feature_size"
        ]
        == feature_size

    ]


    summary_rows.append({

        "Feature_Size":
            feature_size,

        "Accuracy_Mean":
            subset[
                "accuracy"
            ].mean(),

        "Accuracy_STD":
            subset[
                "accuracy"
            ].std(),

        "Balanced_Accuracy_Mean":
            subset[
                "balanced_accuracy"
            ].mean(),

        "Balanced_Accuracy_STD":
            subset[
                "balanced_accuracy"
            ].std(),

        "Precision_Mean":
            subset[
                "precision"
            ].mean(),

        "Recall_Mean":
            subset[
                "recall"
            ].mean(),

        "Specificity_Mean":
            subset[
                "specificity"
            ].mean(),

        "F1_Mean":
            subset[
                "f1"
            ].mean(),

        "F1_STD":
            subset[
                "f1"
            ].std(),

        "ROC_AUC_Mean":
            subset[
                "roc_auc"
            ].mean(),

        "ROC_AUC_STD":
            subset[
                "roc_auc"
            ].std()

    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "feature_size_summary.csv"

    ),

    index=False

)


# ==========================================================
# BEST FEATURE SIZE
# ==========================================================

best_row = summary_df.loc[

    summary_df[
        "ROC_AUC_Mean"
    ].idxmax()

]


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print()

print("=" * 60)

print(
    "FEATURE SIZE COMPARISON"
)

print("=" * 60)

print()

print(

    summary_df.to_string(
        index=False
    )

)


print()

print("=" * 60)

print(
    "BEST FEATURE SIZE BY MEAN ROC-AUC"
)

print("=" * 60)

print()

print(
    "Features :",
    int(
        best_row[
            "Feature_Size"
        ]
    )
)

print(

    "ROC-AUC  :",
    f"{best_row['ROC_AUC_Mean']:.4f}"

)

print(

    "STD      :",
    f"{best_row['ROC_AUC_STD']:.4f}"

)

print()

print(
    "Feature-size experiment completed."
)

print(
    "Results saved to:",
    OUTPUT_DIR
)