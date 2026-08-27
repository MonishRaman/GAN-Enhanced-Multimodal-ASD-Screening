import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif


# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42
NUM_FOLDS = 5

KNN_NEIGHBORS = 5
TOP_FEATURES = 100

INPUT_FILE = r"data/processed/filtered_expression_matrix.csv"
LABEL_FILE = r"data/processed/labels.csv"

OUTPUT_DIR = (
    "genetics/outputs/feature_stability"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


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

    feature_table = df.set_index(
        "Position"
    )

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

print(
    "TOP-100 FEATURE STABILITY EXTRACTION"
)

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


all_fold_features = []


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

    print("=" * 60)

    print(
        f"FOLD {fold}/{NUM_FOLDS}"
    )

    print("=" * 60)


    X_train_raw = X[
        train_idx
    ]


    y_train = y[
        train_idx
    ]


    # ======================================================
    # KNN IMPUTATION
    # ======================================================

    print(
        "Fitting KNN imputer..."
    )


    imputer = KNNImputer(

        n_neighbors=KNN_NEIGHBORS,

        weights="uniform"

    )


    X_train_imputed = (

        imputer.fit_transform(
            X_train_raw
        )

    )


    # ======================================================
    # STANDARDIZATION
    # ======================================================

    print(
        "Fitting StandardScaler..."
    )


    scaler = StandardScaler()


    X_train_scaled = (

        scaler.fit_transform(
            X_train_imputed
        )

    )


    # ======================================================
    # MUTUAL INFORMATION
    # ======================================================

    print(
        "Computing Mutual Information..."
    )


    mi_scores = mutual_info_classif(

        X_train_scaled,

        y_train,

        random_state=RANDOM_STATE

    )


    # ======================================================
    # CREATE MI TABLE
    # ======================================================

    mi_df = pd.DataFrame({

        "Feature":
            feature_names,

        "MI_Score":
            mi_scores

    })


    mi_df = mi_df.sort_values(

        "MI_Score",

        ascending=False

    ).reset_index(
        drop=True
    )


    # ======================================================
    # TOP 100
    # ======================================================

    top100 = mi_df.head(
        TOP_FEATURES
    ).copy()


    top100.insert(
        0,
        "Rank",
        range(
            1,
            TOP_FEATURES + 1
        )
    )


    top100.insert(
        0,
        "Fold",
        fold
    )


    # ======================================================
    # SAVE FOLD FEATURES
    # ======================================================

    output_file = os.path.join(

        OUTPUT_DIR,

        f"fold_{fold}_top100_features.csv"

    )


    top100.to_csv(

        output_file,

        index=False

    )


    # ======================================================
    # STORE FOR COMBINED ANALYSIS
    # ======================================================

    all_fold_features.append(
        top100
    )


    print()

    print(
        f"Selected {len(top100)} features."
    )

    print(
        "Saved:",
        output_file
    )

    print()


# ==========================================================
# COMBINE ALL FOLDS
# ==========================================================

combined_df = pd.concat(

    all_fold_features,

    ignore_index=True

)


combined_file = os.path.join(

    OUTPUT_DIR,

    "all_folds_top100_features.csv"

)


combined_df.to_csv(

    combined_file,

    index=False

)


# ==========================================================
# FEATURE SELECTION FREQUENCY
# ==========================================================

frequency_df = (

    combined_df

    .groupby("Feature")

    .agg(

        Selection_Frequency=(
            "Fold",
            "count"
        ),

        Mean_MI_Score=(
            "MI_Score",
            "mean"
        ),

        Min_MI_Score=(
            "MI_Score",
            "min"
        ),

        Max_MI_Score=(
            "MI_Score",
            "max"
        )

    )

    .reset_index()

)


# ==========================================================
# SORT BY STABILITY
# ==========================================================

frequency_df = frequency_df.sort_values(

    [
        "Selection_Frequency",
        "Mean_MI_Score"

    ],

    ascending=[
        False,
        False
    ]

).reset_index(
    drop=True
)


frequency_file = os.path.join(

    OUTPUT_DIR,

    "feature_selection_frequency.csv"

)


frequency_df.to_csv(

    frequency_file,

    index=False

)


# ==========================================================
# STABILITY SUMMARY
# ==========================================================

frequency_counts = (

    frequency_df[
        "Selection_Frequency"
    ]

    .value_counts()

    .sort_index(
        ascending=False
    )

)


print()

print("=" * 60)

print(
    "FEATURE STABILITY SUMMARY"
)

print("=" * 60)

print()


for frequency in range(
    NUM_FOLDS,
    0,
    -1
):

    count = frequency_counts.get(
        frequency,
        0
    )


    print(

        f"Selected in "
        f"{frequency}/{NUM_FOLDS} folds : "
        f"{count}"

    )


# ==========================================================
# CORE STABLE FEATURES
# ==========================================================

core_features = frequency_df[

    frequency_df[
        "Selection_Frequency"
    ]
    == NUM_FOLDS

].copy()


core_file = os.path.join(

    OUTPUT_DIR,

    "core_features_all_5_folds.csv"

)


core_features.to_csv(

    core_file,

    index=False

)


# ==========================================================
# FREQUENCY >= 4
# ==========================================================

stable_4plus = frequency_df[

    frequency_df[
        "Selection_Frequency"
    ]
    >= 4

].copy()


stable_4plus_file = os.path.join(

    OUTPUT_DIR,

    "stable_features_4plus_folds.csv"

)


stable_4plus.to_csv(

    stable_4plus_file,

    index=False

)


# ==========================================================
# FREQUENCY >= 3
# ==========================================================

stable_3plus = frequency_df[

    frequency_df[
        "Selection_Frequency"
    ]
    >= 3

].copy()


stable_3plus_file = os.path.join(

    OUTPUT_DIR,

    "stable_features_3plus_folds.csv"

)


stable_3plus.to_csv(

    stable_3plus_file,

    index=False

)


# ==========================================================
# FINAL REPORT
# ==========================================================

print()

print("=" * 60)

print(
    "STABILITY RESULTS"
)

print("=" * 60)

print()

print(
    "Unique features selected:",
    len(frequency_df)
)

print(

    "Features selected in "
    "all 5 folds:",

    len(core_features)

)

print(

    "Features selected in "
    "at least 4 folds:",

    len(stable_4plus)

)

print(

    "Features selected in "
    "at least 3 folds:",

    len(stable_3plus)

)


print()

print(
    "Top 20 most stable features:"
)

print()

print(

    frequency_df.head(
        20
    ).to_string(
        index=False
    )

)


print()

print("=" * 60)

print(
    "FILES SAVED"
)

print("=" * 60)

print()

print(
    "All fold features:",
    combined_file
)

print(
    "Frequency table:",
    frequency_file
)

print(
    "All-5-fold features:",
    core_file
)

print(
    "4+ fold features:",
    stable_4plus_file
)

print(
    "3+ fold features:",
    stable_3plus_file
)

print()

print(
    "Feature stability extraction completed."
)