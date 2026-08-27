import os
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = r"D:\org genetical data"

REAL_TRAIN = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\embedding_gan\genetic_embedding_gan_train.csv"
)

UNSEEN_TEST = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\embedding_gan\genetic_embedding_gan_test_unseen.csv"
)

CUSTOM_GAN = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\embedding_gan\synthetic_control_128d_embeddings.csv"
)

CTGAN = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\ctgan\genetic_ctgan_synthetic_controls_46.csv"
)

OUT_DIR = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\ctgan_validation"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 37 — CTGAN VS CUSTOM GAN VALIDATION")
print("=" * 70)

print("\nLoading real training embeddings...")
real_train = pd.read_csv(REAL_TRAIN)
print("Done.")

print("Loading unseen test embeddings...")
test = pd.read_csv(UNSEEN_TEST)
print("Done.")

print("Loading Custom GAN synthetic controls...")
custom = pd.read_csv(CUSTOM_GAN)
print("Done.")

print("Loading CTGAN synthetic controls...")
ctgan = pd.read_csv(CTGAN)
print("Done.")


# ============================================================
# DETECT EMBEDDING COLUMNS
# ============================================================

def get_embedding_columns(df):
    cols = [
        c for c in df.columns
        if str(c).startswith("Genetic_Embedding_")
    ]

    # Sort numerically
    cols = sorted(
        cols,
        key=lambda x: int(str(x).split("_")[-1])
    )

    return cols


real_emb_cols = get_embedding_columns(real_train)
test_emb_cols = get_embedding_columns(test)
custom_emb_cols = get_embedding_columns(custom)
ctgan_emb_cols = get_embedding_columns(ctgan)


print("\n" + "=" * 70)
print("EMBEDDING STRUCTURE")
print("=" * 70)

print("Real training embedding dimensions :", len(real_emb_cols))
print("Unseen test embedding dimensions  :", len(test_emb_cols))
print("Custom GAN embedding dimensions   :", len(custom_emb_cols))
print("CTGAN embedding dimensions        :", len(ctgan_emb_cols))


EXPECTED = 128

for name, cols in [
    ("Real training", real_emb_cols),
    ("Unseen test", test_emb_cols),
    ("Custom GAN", custom_emb_cols),
    ("CTGAN", ctgan_emb_cols),
]:
    if len(cols) != EXPECTED:
        raise ValueError(
            f"{name}: expected {EXPECTED} embeddings, found {len(cols)}"
        )


# ============================================================
# EXTRACT MATRICES
# ============================================================

X_real = real_train[real_emb_cols].astype(float).values
y_real = real_train["Target"].astype(int).values

X_test = test[test_emb_cols].astype(float).values
y_test = test["Target"].astype(int).values

X_custom = custom[custom_emb_cols].astype(float).values
X_ctgan = ctgan[ctgan_emb_cols].astype(float).values


# ============================================================
# BASIC VALIDATION
# ============================================================

def basic_quality(X):
    return {
        "Samples": X.shape[0],
        "Dimensions": X.shape[1],
        "NaN": int(np.isnan(X).sum()),
        "Inf": int(np.isinf(X).sum()),
        "Duplicate_Rows": int(
            pd.DataFrame(X).duplicated().sum()
        )
    }


print("\n" + "=" * 70)
print("BASIC QUALITY")
print("=" * 70)

quality = {
    "Real_Control": basic_quality(X_real[y_real == 0]),
    "Custom_GAN": basic_quality(X_custom),
    "CTGAN": basic_quality(X_ctgan)
}

for name, result in quality.items():
    print(f"\n{name}")
    for k, v in result.items():
        print(f"{k:20s}: {v}")


# ============================================================
# REAL CONTROL REFERENCE
# ============================================================

X_real_control = X_real[y_real == 0]

print("\nReal controls:", len(X_real_control))
print("Custom GAN:", len(X_custom))
print("CTGAN:", len(X_ctgan))


# ============================================================
# DISTRIBUTION SIMILARITY
# ============================================================

def distribution_metrics(real, synthetic):

    ks_values = []
    ks_pass = []

    mean_diffs = []
    std_diffs = []

    for i in range(real.shape[1]):

        r = real[:, i]
        s = synthetic[:, i]

        ks_stat, p_value = ks_2samp(r, s)

        ks_values.append(ks_stat)
        ks_pass.append(p_value >= 0.05)

        mean_diffs.append(
            abs(np.mean(r) - np.mean(s))
        )

        std_diffs.append(
            abs(np.std(r) - np.std(s))
        )

    return {
        "Mean_KS": np.mean(ks_values),
        "Median_KS": np.median(ks_values),
        "KS_p_ge_0.05": np.mean(ks_pass),
        "Mean_Absolute_Mean_Difference": np.mean(mean_diffs),
        "Mean_Absolute_STD_Difference": np.mean(std_diffs)
    }


print("\n" + "=" * 70)
print("DISTRIBUTION SIMILARITY")
print("=" * 70)

dist_custom = distribution_metrics(
    X_real_control,
    X_custom
)

dist_ctgan = distribution_metrics(
    X_real_control,
    X_ctgan
)

for name, result in [
    ("CUSTOM GAN", dist_custom),
    ("CTGAN", dist_ctgan)
]:

    print("\n" + name)

    for k, v in result.items():
        print(f"{k:35s}: {v:.6f}")


# ============================================================
# DISTANCE / DIVERSITY
# ============================================================

def distance_metrics(real, synthetic):

    distances = cdist(
        synthetic,
        real,
        metric="euclidean"
    )

    nearest = distances.min(axis=1)

    pairwise = pdist(
        synthetic,
        metric="euclidean"
    )

    return {
        "Mean_Nearest_Real_Distance":
            np.mean(nearest),

        "Median_Nearest_Real_Distance":
            np.median(nearest),

        "Mean_Synthetic_Pairwise_Distance":
            np.mean(pairwise),

        "Median_Synthetic_Pairwise_Distance":
            np.median(pairwise)
    }


print("\n" + "=" * 70)
print("DISTANCE AND DIVERSITY")
print("=" * 70)

dist_custom_distance = distance_metrics(
    X_real_control,
    X_custom
)

dist_ctgan_distance = distance_metrics(
    X_real_control,
    X_ctgan
)

for name, result in [
    ("CUSTOM GAN", dist_custom_distance),
    ("CTGAN", dist_ctgan_distance)
]:

    print("\n" + name)

    for k, v in result.items():
        print(f"{k:40s}: {v:.6f}")


# ============================================================
# DOWNSTREAM UTILITY
# ============================================================

def downstream_evaluation(
    real_training,
    synthetic_controls,
    test_X,
    test_y
):

    X_aug = np.vstack([
        real_training,
        synthetic_controls
    ])

    y_aug = np.concatenate([
        np.ones(np.sum(y_real == 1), dtype=int),
        np.zeros(
            np.sum(y_real == 0) +
            len(synthetic_controls),
            dtype=int
        )
    ])

    scaler = StandardScaler()

    X_aug_scaled = scaler.fit_transform(X_aug)
    X_test_scaled = scaler.transform(test_X)

    model = LogisticRegression(
        max_iter=5000,
        class_weight=None,
        random_state=42
    )

    model.fit(
        X_aug_scaled,
        y_aug
    )

    pred = model.predict(X_test_scaled)
    prob = model.predict_proba(X_test_scaled)[:, 1]

    return {
        "Training_Samples": len(X_aug),
        "Accuracy": accuracy_score(test_y, pred),
        "Balanced_Accuracy":
            balanced_accuracy_score(test_y, pred),
        "Precision":
            precision_score(
                test_y,
                pred,
                zero_division=0
            ),
        "Recall":
            recall_score(
                test_y,
                pred,
                zero_division=0
            ),
        "F1_Score":
            f1_score(
                test_y,
                pred,
                zero_division=0
            ),
        "ROC_AUC":
            roc_auc_score(
                test_y,
                prob
            ),
        "Confusion_Matrix":
            confusion_matrix(
                test_y,
                pred
            ).tolist()
    }


print("\n" + "=" * 70)
print("DOWNSTREAM UTILITY")
print("=" * 70)

baseline = downstream_evaluation(
    X_real,
    np.empty((0, EXPECTED)),
    X_test,
    y_test
)

custom_result = downstream_evaluation(
    X_real,
    X_custom,
    X_test,
    y_test
)

ctgan_result = downstream_evaluation(
    X_real,
    X_ctgan,
    X_test,
    y_test
)

for name, result in [
    ("BASELINE — NO GAN", baseline),
    ("CUSTOM GAN", custom_result),
    ("CTGAN", ctgan_result)
]:

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    for k, v in result.items():
        if k != "Confusion_Matrix":
            print(f"{k:25s}: {v:.6f}")

    print("Confusion Matrix:")
    print(np.array(result["Confusion_Matrix"]))


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame([
    {
        "Method": "Baseline_No_GAN",
        **{
            k: v
            for k, v in baseline.items()
            if k != "Confusion_Matrix"
        }
    },
    {
        "Method": "Custom_Conditional_GAN",
        **{
            k: v
            for k, v in custom_result.items()
            if k != "Confusion_Matrix"
        }
    },
    {
        "Method": "CTGAN",
        **{
            k: v
            for k, v in ctgan_result.items()
            if k != "Confusion_Matrix"
        }
    }
])

comparison_path = os.path.join(
    OUT_DIR,
    "ctgan_vs_custom_gan_downstream_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# DISTRIBUTION COMPARISON TABLE
# ============================================================

distribution_table = pd.DataFrame([
    {
        "Method": "Custom_GAN",
        **dist_custom,
        **dist_custom_distance
    },
    {
        "Method": "CTGAN",
        **dist_ctgan,
        **dist_ctgan_distance
    }
])

distribution_path = os.path.join(
    OUT_DIR,
    "ctgan_vs_custom_gan_distribution_comparison.csv"
)

distribution_table.to_csv(
    distribution_path,
    index=False
)


# ============================================================
# PCA
# ============================================================

all_data = np.vstack([
    X_real_control,
    X_custom,
    X_ctgan
])

labels = (
    ["Real_Control"] * len(X_real_control)
    + ["Custom_GAN"] * len(X_custom)
    + ["CTGAN"] * len(X_ctgan)
)

scaler = StandardScaler()

all_scaled = scaler.fit_transform(all_data)

pca = PCA(
    n_components=2,
    random_state=42
)

coords = pca.fit_transform(
    all_scaled
)

plt.figure(figsize=(9, 7))

for label in [
    "Real_Control",
    "Custom_GAN",
    "CTGAN"
]:

    mask = np.array(labels) == label

    plt.scatter(
        coords[mask, 0],
        coords[mask, 1],
        label=label,
        alpha=0.7
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(
    "Real Control vs Custom GAN vs CTGAN"
)

plt.legend()
plt.tight_layout()

pca_path = os.path.join(
    OUT_DIR,
    "real_custom_gan_ctgan_pca.png"
)

plt.savefig(
    pca_path,
    dpi=300
)

plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

report_path = os.path.join(
    OUT_DIR,
    "ctgan_vs_custom_gan_validation_report.txt"
)

with open(report_path, "w") as f:

    f.write(
        "STEP 37 — CTGAN VS CUSTOM GAN VALIDATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Real training samples: "
        f"{len(X_real)}\n"
    )

    f.write(
        "Real control samples: "
        f"{len(X_real_control)}\n"
    )

    f.write(
        "Custom GAN synthetic samples: "
        f"{len(X_custom)}\n"
    )

    f.write(
        "CTGAN synthetic samples: "
        f"{len(X_ctgan)}\n"
    )

    f.write(
        "Embedding dimensions: 128\n\n"
    )

    f.write(
        "DISTRIBUTION SIMILARITY\n"
    )

    f.write(
        distribution_table.to_string(
            index=False
        )
    )

    f.write("\n\nDOWNSTREAM UTILITY\n")

    f.write(
        comparison.to_string(
            index=False
        )
    )

    f.write("\n\nINTERPRETATION\n")

    f.write(
        "CTGAN and the custom conditional GAN "
        "were evaluated against the same real "
        "training control embeddings and the "
        "same unseen test set.\n"
    )

print("\n" + "=" * 70)
print("STEP 37 — FINAL SUMMARY")
print("=" * 70)

print("Custom GAN synthetic samples :", len(X_custom))
print("CTGAN synthetic samples      :", len(X_ctgan))
print("Embedding dimensions         :", EXPECTED)

print("\nFiles saved:")
print(comparison_path)
print(distribution_path)
print(pca_path)
print(report_path)

print("\nStep 37 completed successfully.")