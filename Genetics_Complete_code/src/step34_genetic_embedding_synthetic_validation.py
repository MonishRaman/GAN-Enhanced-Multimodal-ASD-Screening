import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# ============================================================
# STEP 34 — CORRECTED GENETIC EMBEDDING SYNTHETIC VALIDATION
# ============================================================

BASE = r"D:\org genetical data"

INPUT_DIR = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "objective2_genetic",
    "embedding_gan"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "objective2_genetic",
    "embedding_gan_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(
    INPUT_DIR,
    "genetic_embedding_gan_train.csv"
)

SYNTHETIC_FILE = os.path.join(
    INPUT_DIR,
    "synthetic_control_128d_embeddings.csv"
)


print("=" * 70)
print("STEP 34 — CORRECTED GENETIC EMBEDDING SYNTHETIC VALIDATION")
print("=" * 70)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading original training embeddings...")
train = pd.read_csv(TRAIN_FILE)
print("Done.")

print("Loading synthetic Control embeddings...")
synthetic = pd.read_csv(SYNTHETIC_FILE)
print("Done.")


# ============================================================
# FIND EMBEDDING COLUMNS
# ============================================================

embedding_cols = [
    c for c in train.columns
    if c.startswith("Genetic_Embedding_")
]

if len(embedding_cols) != 128:
    raise ValueError(
        f"Expected 128 embedding dimensions, "
        f"found {len(embedding_cols)}"
    )


# ============================================================
# REAL CONTROL ONLY
# ============================================================

real_control = train[
    train["Target"] == 0
][embedding_cols].copy()

synthetic_control = synthetic[
    embedding_cols
].copy()


print("\n" + "=" * 70)
print("DATASET STRUCTURE")
print("=" * 70)

print(f"Training samples             : {len(train)}")
print(f"Real Control samples         : {len(real_control)}")
print(f"Synthetic Control samples    : {len(synthetic_control)}")
print(f"Embedding dimensions         : {len(embedding_cols)}")


# ============================================================
# VERIFY EXPECTED COUNTS
# ============================================================

if len(real_control) != 23:
    print(
        f"\nWARNING: Expected 23 real Control samples, "
        f"found {len(real_control)}"
    )

if len(synthetic_control) != 46:
    print(
        f"\nWARNING: Expected 46 synthetic samples, "
        f"found {len(synthetic_control)}"
    )


# ============================================================
# BASIC QUALITY
# ============================================================

print("\n" + "=" * 70)
print("BASIC QUALITY CHECK")
print("=" * 70)

real_nan = int(real_control.isna().sum().sum())
synthetic_nan = int(synthetic_control.isna().sum().sum())

real_inf = int(
    np.isinf(
        real_control.to_numpy(dtype=float)
    ).sum()
)

synthetic_inf = int(
    np.isinf(
        synthetic_control.to_numpy(dtype=float)
    ).sum()
)

synthetic_duplicates = int(
    synthetic_control.duplicated().sum()
)

print(f"Real Control NaN       : {real_nan}")
print(f"Synthetic NaN          : {synthetic_nan}")
print(f"Real Control Inf       : {real_inf}")
print(f"Synthetic Inf          : {synthetic_inf}")
print(f"Synthetic duplicates   : {synthetic_duplicates}")


# ============================================================
# NUMPY
# ============================================================

real_X = real_control.to_numpy(dtype=float)
synthetic_X = synthetic_control.to_numpy(dtype=float)


# ============================================================
# DISTRIBUTION SIMILARITY
# ============================================================

print("\nCalculating embedding-wise distribution similarity...")

results = []

for i, feature in enumerate(embedding_cols):

    real_values = real_X[:, i]
    synthetic_values = synthetic_X[:, i]

    ks_stat, ks_p = ks_2samp(
        real_values,
        synthetic_values
    )

    real_mean = np.mean(real_values)
    synthetic_mean = np.mean(synthetic_values)

    real_std = np.std(real_values)
    synthetic_std = np.std(synthetic_values)

    results.append({
        "Embedding": feature,
        "KS_Statistic": ks_stat,
        "KS_P_Value": ks_p,
        "Real_Mean": real_mean,
        "Synthetic_Mean": synthetic_mean,
        "Absolute_Mean_Difference":
            abs(real_mean - synthetic_mean),
        "Real_STD": real_std,
        "Synthetic_STD": synthetic_std,
        "Absolute_STD_Difference":
            abs(real_std - synthetic_std)
    })


distribution_df = pd.DataFrame(results)

mean_ks = distribution_df[
    "KS_Statistic"
].mean()

median_ks = distribution_df[
    "KS_Statistic"
].median()

ks_pass_proportion = (
    distribution_df["KS_P_Value"] >= 0.05
).mean()

mean_abs_mean_diff = (
    distribution_df[
        "Absolute_Mean_Difference"
    ].mean()
)

mean_abs_std_diff = (
    distribution_df[
        "Absolute_STD_Difference"
    ].mean()
)


# ============================================================
# DISTANCE ANALYSIS
# ============================================================

print("\nCalculating real-synthetic distances...")

scaler = StandardScaler()

combined = np.vstack([
    real_X,
    synthetic_X
])

combined_scaled = scaler.fit_transform(
    combined
)

real_scaled = combined_scaled[
    :len(real_X)
]

synthetic_scaled = combined_scaled[
    len(real_X):
]


# Synthetic -> nearest REAL Control
distance_matrix = pairwise_distances(
    synthetic_scaled,
    real_scaled,
    metric="euclidean"
)

nearest_real_distance = (
    distance_matrix.min(axis=1)
)

mean_nearest_real = (
    nearest_real_distance.mean()
)

median_nearest_real = (
    np.median(nearest_real_distance)
)


# Synthetic diversity
synthetic_pairwise = pairwise_distances(
    synthetic_scaled,
    metric="euclidean"
)

upper_triangle = synthetic_pairwise[
    np.triu_indices_from(
        synthetic_pairwise,
        k=1
    )
]

synthetic_mean_pairwise = (
    upper_triangle.mean()
)

synthetic_median_pairwise = (
    np.median(upper_triangle)
)


# ============================================================
# PCA
# ============================================================

print("\nGenerating PCA visualization...")

pca = PCA(n_components=2)

pca_coordinates = pca.fit_transform(
    combined_scaled
)

real_pca = pca_coordinates[
    :len(real_X)
]

synthetic_pca = pca_coordinates[
    len(real_X):
]

plt.figure(figsize=(9, 7))

plt.scatter(
    real_pca[:, 0],
    real_pca[:, 1],
    label="Real Control",
    alpha=0.75
)

plt.scatter(
    synthetic_pca[:, 0],
    synthetic_pca[:, 1],
    label="Synthetic Control",
    alpha=0.75
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(
    "Real vs Synthetic Control Genetic Embeddings"
)

plt.legend()
plt.tight_layout()

pca_file = os.path.join(
    OUTPUT_DIR,
    "corrected_real_vs_synthetic_control_128d_pca.png"
)

plt.savefig(
    pca_file,
    dpi=300
)

plt.close()


# ============================================================
# BASIC VALIDATION
# ============================================================

basic_pass = (
    real_nan == 0
    and synthetic_nan == 0
    and real_inf == 0
    and synthetic_inf == 0
    and synthetic_duplicates == 0
)


# ============================================================
# STATUS
# ============================================================

if not basic_pass:

    status = "FAIL"

elif (
    mean_ks <= 0.20
    and ks_pass_proportion >= 0.50
):

    status = "GOOD"

elif (
    mean_ks <= 0.35
    and ks_pass_proportion >= 0.30
):

    status = "ACCEPTABLE_WITH_CAUTION"

else:

    status = "WEAK_DISTRIBUTIONAL_MATCH"


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 34 — CORRECTED FINAL SUMMARY")
print("=" * 70)

print(
    f"Real Control samples            : "
    f"{len(real_control)}"
)

print(
    f"Synthetic Control samples       : "
    f"{len(synthetic_control)}"
)

print(
    f"Embedding dimensions            : "
    f"{len(embedding_cols)}"
)

print(
    f"Basic validation                : "
    f"{'PASSED' if basic_pass else 'FAILED'}"
)

print(
    f"Mean KS statistic               : "
    f"{mean_ks:.6f}"
)

print(
    f"Median KS statistic             : "
    f"{median_ks:.6f}"
)

print(
    f"KS p >= 0.05 proportion         : "
    f"{ks_pass_proportion:.4f}"
)

print(
    f"Mean absolute mean difference   : "
    f"{mean_abs_mean_diff:.6f}"
)

print(
    f"Mean absolute STD difference    : "
    f"{mean_abs_std_diff:.6f}"
)

print(
    f"Mean nearest-real distance      : "
    f"{mean_nearest_real:.6f}"
)

print(
    f"Median nearest-real distance    : "
    f"{median_nearest_real:.6f}"
)

print(
    f"Synthetic mean pairwise distance: "
    f"{synthetic_mean_pairwise:.6f}"
)

print(
    f"Synthetic median pairwise dist. : "
    f"{synthetic_median_pairwise:.6f}"
)

print(
    f"\nFINAL VALIDATION STATUS         : "
    f"{status}"
)


# ============================================================
# SAVE DISTRIBUTION TABLE
# ============================================================

distribution_file = os.path.join(
    OUTPUT_DIR,
    "corrected_embedding_distribution_comparison.csv"
)

distribution_df.to_csv(
    distribution_file,
    index=False
)


# ============================================================
# SAVE REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "corrected_genetic_embedding_synthetic_validation_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 34 — CORRECTED GENETIC EMBEDDING "
        "SYNTHETIC VALIDATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write("IMPORTANT VALIDATION DESIGN\n")
    f.write("-" * 70 + "\n")

    f.write(
        "The comparison uses only the 23 genuine "
        "Control embeddings from the original "
        "training set as the real reference.\n"
    )

    f.write(
        "The 46 synthetic Control embeddings are "
        "evaluated separately.\n\n"
    )

    f.write("DATASET\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Real Control samples: "
        f"{len(real_control)}\n"
    )

    f.write(
        f"Synthetic Control samples: "
        f"{len(synthetic_control)}\n"
    )

    f.write(
        f"Embedding dimensions: "
        f"{len(embedding_cols)}\n\n"
    )

    f.write("QUALITY\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Real Control NaN: {real_nan}\n"
    )

    f.write(
        f"Synthetic NaN: {synthetic_nan}\n"
    )

    f.write(
        f"Real Control Inf: {real_inf}\n"
    )

    f.write(
        f"Synthetic Inf: {synthetic_inf}\n"
    )

    f.write(
        f"Synthetic duplicates: "
        f"{synthetic_duplicates}\n\n"
    )

    f.write("DISTRIBUTION SIMILARITY\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Mean KS statistic: "
        f"{mean_ks:.6f}\n"
    )

    f.write(
        f"Median KS statistic: "
        f"{median_ks:.6f}\n"
    )

    f.write(
        f"KS p >= 0.05 proportion: "
        f"{ks_pass_proportion:.4f}\n"
    )

    f.write(
        f"Mean absolute mean difference: "
        f"{mean_abs_mean_diff:.6f}\n"
    )

    f.write(
        f"Mean absolute STD difference: "
        f"{mean_abs_std_diff:.6f}\n\n"
    )

    f.write("DISTANCE AND DIVERSITY\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Mean nearest-real distance: "
        f"{mean_nearest_real:.6f}\n"
    )

    f.write(
        f"Median nearest-real distance: "
        f"{median_nearest_real:.6f}\n"
    )

    f.write(
        f"Synthetic mean pairwise distance: "
        f"{synthetic_mean_pairwise:.6f}\n"
    )

    f.write(
        f"Synthetic median pairwise distance: "
        f"{synthetic_median_pairwise:.6f}\n\n"
    )

    f.write("FINAL STATUS\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Basic validation: "
        f"{'PASSED' if basic_pass else 'FAILED'}\n"
    )

    f.write(
        f"Distributional status: "
        f"{status}\n\n"
    )

    f.write("INTERPRETATION\n")
    f.write("-" * 70 + "\n")

    if status == "GOOD":

        f.write(
            "The synthetic Control embeddings show "
            "a strong distributional similarity to "
            "the genuine Control embeddings and "
            "pass the basic quality checks.\n"
        )

    elif status == "ACCEPTABLE_WITH_CAUTION":

        f.write(
            "The synthetic Control embeddings show "
            "moderate similarity to genuine Control "
            "embeddings. They may be useful for "
            "augmentation but should undergo "
            "downstream validation.\n"
        )

    elif status == "WEAK_DISTRIBUTIONAL_MATCH":

        f.write(
            "The synthetic Control embeddings are "
            "numerically valid but show weak "
            "distributional similarity to genuine "
            "Control embeddings. GAN tuning should "
            "be considered.\n"
        )

    else:

        f.write(
            "The synthetic embeddings failed one "
            "or more basic quality checks and "
            "should not be used without correction.\n"
        )


print("\nFiles saved:")
print(distribution_file)
print(pca_file)
print(report_file)

print("\nStep 34 corrected validation completed.")