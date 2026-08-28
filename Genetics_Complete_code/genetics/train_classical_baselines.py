import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

TOP_FEATURES = 1000

KNN_NEIGHBORS = 5

INPUT_FILE = r"data/processed/filtered_expression_matrix.csv"

LABEL_FILE = r"data/processed/labels.csv"

OUTPUT_DIR = "genetics/outputs/classical_baselines"

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
# LOAD EXPRESSION DATA
# ==========================================================

print("Loading Step 3 filtered dataset...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")


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

print("Done.\n")


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


feature_names = X_df.columns


# ==========================================================
# DATASET SUMMARY
# ==========================================================

print("=" * 60)

print("CLASSICAL BASELINE DATASET")

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
# MODELS
# ==========================================================

def create_models():

    return {

        "Logistic_Regression":

            LogisticRegression(

                C=1.0,

                class_weight="balanced",

                max_iter=3000,

                solver="liblinear",

                random_state=RANDOM_STATE

            ),


        "Linear_SVM":

            SVC(

                C=1.0,

                kernel="linear",

                class_weight="balanced",

                probability=True,

                random_state=RANDOM_STATE

            ),


        "Random_Forest":

            RandomForestClassifier(

                n_estimators=300,

                max_depth=5,

                min_samples_leaf=2,

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1

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
# MODEL LOOP
# ==========================================================

for model_name in [

    "Logistic_Regression",

    "Linear_SVM",

    "Random_Forest"

]:

    print()

    print("=" * 60)

    print(
        f"MODEL: {model_name}"
    )

    print("=" * 60)


    fold_results = []


    oof_predictions = np.zeros(
        len(y),
        dtype=np.int64
    )


    oof_probabilities = np.zeros(
        len(y),
        dtype=np.float32
    )


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
        # Raw split
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


        top_indices = (

            np.argsort(
                mi_scores
            )[::-1][:TOP_FEATURES]

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
        # CREATE MODEL
        # ==================================================

        models = create_models()

        model = models[
            model_name
        ]


        # ==================================================
        # TRAIN
        # ==================================================

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


        metrics["fold"] = fold

        metrics["model"] = model_name


        fold_results.append(
            metrics
        )


        all_results.append(
            metrics
        )


        # ==================================================
        # OOF
        # ==================================================

        oof_predictions[
            val_idx
        ] = predictions


        oof_probabilities[
            val_idx
        ] = probabilities


        print()

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
    # MODEL RESULTS
    # ======================================================

    model_results = pd.DataFrame(
        fold_results
    )


    model_results.to_csv(

        os.path.join(

            OUTPUT_DIR,

            f"{model_name}_fold_results.csv"

        ),

        index=False

    )


    # ======================================================
    # OOF METRICS
    # ======================================================

    oof_metrics = calculate_metrics(

        y,

        oof_predictions,

        oof_probabilities

    )


    oof_metrics["model"] = model_name


    pd.DataFrame(
        [oof_metrics]
    ).to_csv(

        os.path.join(

            OUTPUT_DIR,

            f"{model_name}_oof_results.csv"

        ),

        index=False

    )


    # ======================================================
    # OOF PREDICTIONS
    # ======================================================

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

            f"{model_name}_oof_predictions.csv"

        ),

        index=False

    )


    # ======================================================
    # PRINT SUMMARY
    # ======================================================

    print()

    print(
        f"{model_name} SUMMARY"
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

        mean = model_results[
            metric
        ].mean()


        std = model_results[
            metric
        ].std()


        print(

            f"{metric:20s}: "
            f"{mean:.4f} ± {std:.4f}"

        )


# ==========================================================
# COMBINED RESULTS
# ==========================================================

all_results_df = pd.DataFrame(
    all_results
)


all_results_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "all_model_fold_results.csv"

    ),

    index=False

)


# ==========================================================
# FINAL COMPARISON
# ==========================================================

comparison = []


for model_name in all_results_df[
    "model"
].unique():

    subset = all_results_df[
        all_results_df["model"]
        == model_name
    ]


    row = {

        "Model":
            model_name,

        "Accuracy_Mean":
            subset["accuracy"].mean(),

        "Accuracy_STD":
            subset["accuracy"].std(),

        "Balanced_Accuracy_Mean":
            subset["balanced_accuracy"].mean(),

        "Balanced_Accuracy_STD":
            subset["balanced_accuracy"].std(),

        "F1_Mean":
            subset["f1"].mean(),

        "F1_STD":
            subset["f1"].std(),

        "ROC_AUC_Mean":
            subset["roc_auc"].mean(),

        "ROC_AUC_STD":
            subset["roc_auc"].std()

    }


    comparison.append(row)


comparison_df = pd.DataFrame(
    comparison
)


comparison_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "model_comparison.csv"

    ),

    index=False

)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print()

print("=" * 60)

print(
    "CLASSICAL MODEL COMPARISON"
)

print("=" * 60)

print()

print(
    comparison_df.to_string(
        index=False
    )
)

print()

print(
    "Results saved to:",
    OUTPUT_DIR
)