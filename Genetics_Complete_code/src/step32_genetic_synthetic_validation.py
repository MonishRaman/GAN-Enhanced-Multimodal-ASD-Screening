import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# ============================================================
# STEP 32 — GENETIC SYNTHETIC DATA VALIDATION
# ============================================================

BASE = r"D:\org genetical data"

TRAIN_FILE = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\train_test_split",
    "genetic_train_1000_features.csv"
)

SYNTHETIC_FILE = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\conditional_gan",
    "synthetic_control_samples.csv"
)

BALANCED_FILE = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\conditional_gan",
    "genetic_balanced_training_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\synthetic_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("STEP 32 — GENETIC SYNTHETIC DATA VALIDATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading real training data...")
train = pd.read_csv(TRAIN_FILE)
print("Done.")

print("Loading synthetic Control samples...")
synthetic = pd.read_csv(SYNTHETIC_FILE)
print("Done.")

print("Loading balanced dataset...")
balanced = pd.read_csv(BALANCED_FILE)
print("Done.")


# ============================================================
# 2. IDENTIFY GENETIC FEATURES
# ============================================================

metadata_cols = ["GSM_ID", "Source", "Subtype", "Target"]

feature_cols = [
    c for c in train.columns
    if c not in metadata_cols
]

# Keep only features common to both datasets
feature_cols = [
    c for c in feature_cols
    if c in synthetic.columns
]

print("\n" + "=" * 70)
print("DATASET STRUCTURE")
print("=" * 70)

print(f"Real training samples : {len(train)}")
print(f"Synthetic samples     : {len(synthetic)}")
print(f"Genetic features      : {len(feature_cols)}")


# ============================================================
# 3. EXTRACT REAL CONTROL AND SYNTHETIC CONTROL
# ============================================================

real_control = train[train["Target"] == 0][feature_cols].copy()

synthetic_control = synthetic[feature_cols].copy()

real_control = real_control.apply(pd.to_numeric, errors="coerce")
synthetic_control = synthetic_control.apply(pd.to_numeric, errors="coerce")


print(f"\nReal Control samples      : {len(real_control)}")
print(f"Synthetic Control samples : {len(synthetic_control)}")


# ============================================================
# 4. BASIC DATA QUALITY
# ============================================================

real_nan = int(real_control.isna().sum().sum())
synthetic_nan = int(synthetic_control.isna().sum().sum())

real_inf = int(
    np.isinf(real_control.to_numpy()).sum()
)

synthetic_inf = int(
    np.isinf(synthetic_control.to_numpy()).sum()
)

duplicate_synthetic = int(
    synthetic_control.duplicated().sum()
)

print("\n" + "=" * 70)
print("BASIC QUALITY CHECK")
print("=" * 70)

print(f"Real Control NaN       : {real_nan}")
print(f"Synthetic NaN          : {synthetic_nan}")
print(f"Real Control Inf       : {real_inf}")
print(f"Synthetic Inf          : {synthetic_inf}")
print(f"Synthetic duplicates   : {duplicate_synthetic}")


# ============================================================
# 5. FEATURE-WISE DISTRIBUTION COMPARISON
# ============================================================

print("\nCalculating feature-wise distribution similarity...")

ks_statistics = []
ks_pvalues = []

mean_differences = []
std_differences = []

for feature in feature_cols:

    real_values = real_control[feature].dropna().values
    synthetic_values = synthetic_control[feature].dropna().values

    if len(real_values) < 2 or len(synthetic_values) < 2:
        continue

    ks_stat, ks_p = ks_2samp(
        real_values,
        synthetic_values
    )

    ks_statistics.append(ks_stat)
    ks_pvalues.append(ks_p)

    real_mean = np.mean(real_values)
    synth_mean = np.mean(synthetic_values)

    real_std = np.std(real_values)
    synth_std = np.std(synthetic_values)

    mean_differences.append(
        abs(real_mean - synth_mean)
    )

    std_differences.append(
        abs(real_std - synth_std)
    )


mean_ks = np.mean(ks_statistics)
median_ks = np.median(ks_statistics)

ks_pass_rate = np.mean(
    np.array(ks_pvalues) >= 0.05
)

mean_difference = np.mean(mean_differences)
std_difference = np.mean(std_differences)


print("\n" + "=" * 70)
print("DISTRIBUTION SIMILARITY")
print("=" * 70)

print(f"Mean KS statistic        : {mean_ks:.6f}")
print(f"Median KS statistic      : {median_ks:.6f}")
print(f"KS p >= 0.05 proportion  : {ks_pass_rate:.4f}")
print(f"Mean absolute mean diff  : {mean_difference:.6f}")
print(f"Mean absolute std diff   : {std_difference:.6f}")


# ============================================================
# 6. REAL VS SYNTHETIC SUMMARY STATISTICS
# ============================================================

real_means = real_control.mean()
synthetic_means = synthetic_control.mean()

real_stds = real_control.std()
synthetic_stds = synthetic_control.std()

summary_df = pd.DataFrame({
    "Feature": feature_cols,
    "Real_Control_Mean": real_means.values,
    "Synthetic_Control_Mean": synthetic_means.values,
    "Mean_Absolute_Difference":
        np.abs(
            real_means.values -
            synthetic_means.values
        ),
    "Real_Control_STD": real_stds.values,
    "Synthetic_Control_STD": synthetic_stds.values,
    "STD_Absolute_Difference":
        np.abs(
            real_stds.values -
            synthetic_stds.values
        )
})

summary_path = os.path.join(
    OUTPUT_DIR,
    "feature_distribution_comparison.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 7. PCA VISUALIZATION
# ============================================================

print("\nGenerating PCA visualization...")

combined = pd.concat(
    [
        real_control,
        synthetic_control
    ],
    axis=0
)

combined = combined.replace(
    [np.inf, -np.inf],
    np.nan
)

combined = combined.fillna(
    combined.mean()
)

scaler = StandardScaler()

scaled = scaler.fit_transform(
    combined
)

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(
    scaled
)

n_real = len(real_control)

real_pca = pca_result[:n_real]
synthetic_pca = pca_result[n_real:]


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
    alpha=0.75,
    marker="x"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Real vs Synthetic Genetic Control Samples"
)

plt.legend()
plt.grid(alpha=0.25)

pca_path = os.path.join(
    OUTPUT_DIR,
    "real_vs_synthetic_control_pca.png"
)

plt.tight_layout()
plt.savefig(
    pca_path,
    dpi=300
)

plt.close()


explained_variance = (
    pca.explained_variance_ratio_
)


# ============================================================
# 8. DISTANCE ANALYSIS
# ============================================================

print("Calculating real-synthetic distances...")

real_array = real_control.fillna(
    real_control.mean()
).to_numpy()

synthetic_array = synthetic_control.fillna(
    real_control.mean()
).to_numpy()

distance_matrix = pairwise_distances(
    synthetic_array,
    real_array,
    metric="euclidean"
)

nearest_real_distance = (
    distance_matrix.min(axis=1)
)

mean_nearest_distance = (
    np.mean(nearest_real_distance)
)

median_nearest_distance = (
    np.median(nearest_real_distance)
)


# ============================================================
# 9. SYNTHETIC DIVERSITY
# ============================================================

synthetic_pairwise = pairwise_distances(
    synthetic_array,
    metric="euclidean"
)

upper_triangle = synthetic_pairwise[
    np.triu_indices(
        len(synthetic_array),
        k=1
    )
]

synthetic_diversity_mean = (
    np.mean(upper_triangle)
)

synthetic_diversity_median = (
    np.median(upper_triangle)
)


print("\n" + "=" * 70)
print("DISTANCE AND DIVERSITY")
print("=" * 70)

print(
    f"Mean nearest-real distance : "
    f"{mean_nearest_distance:.6f}"
)

print(
    f"Median nearest-real distance : "
    f"{median_nearest_distance:.6f}"
)

print(
    f"Synthetic mean pairwise distance : "
    f"{synthetic_diversity_mean:.6f}"
)

print(
    f"Synthetic median pairwise distance : "
    f"{synthetic_diversity_median:.6f}"
)


# ============================================================
# 10. VALIDATION STATUS
# ============================================================

quality_flags = []

if real_nan > 0:
    quality_flags.append(
        "Real Control data contains NaN values."
    )

if synthetic_nan > 0:
    quality_flags.append(
        "Synthetic data contains NaN values."
    )

if real_inf > 0:
    quality_flags.append(
        "Real Control data contains Inf values."
    )

if synthetic_inf > 0:
    quality_flags.append(
        "Synthetic data contains Inf values."
    )

if duplicate_synthetic > 0:
    quality_flags.append(
        "Duplicate synthetic samples detected."
    )


if len(quality_flags) == 0:
    basic_status = "PASSED"
else:
    basic_status = "FAILED"


# ============================================================
# 11. REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "genetic_synthetic_validation_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 32 — GENETIC SYNTHETIC DATA VALIDATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "DATASET\n"
    )

    f.write(
        f"Real training samples : {len(train)}\n"
    )

    f.write(
        f"Real Control samples : {len(real_control)}\n"
    )

    f.write(
        f"Synthetic Control samples : "
        f"{len(synthetic_control)}\n"
    )

    f.write(
        f"Genetic features : {len(feature_cols)}\n\n"
    )

    f.write(
        "BASIC QUALITY\n"
    )

    f.write(
        f"Real NaN : {real_nan}\n"
    )

    f.write(
        f"Synthetic NaN : {synthetic_nan}\n"
    )

    f.write(
        f"Real Inf : {real_inf}\n"
    )

    f.write(
        f"Synthetic Inf : {synthetic_inf}\n"
    )

    f.write(
        f"Synthetic duplicates : "
        f"{duplicate_synthetic}\n"
    )

    f.write(
        f"Basic validation : {basic_status}\n\n"
    )

    f.write(
        "DISTRIBUTION SIMILARITY\n"
    )

    f.write(
        f"Mean KS statistic : {mean_ks:.6f}\n"
    )

    f.write(
        f"Median KS statistic : {median_ks:.6f}\n"
    )

    f.write(
        f"KS p >= 0.05 proportion : "
        f"{ks_pass_rate:.6f}\n"
    )

    f.write(
        f"Mean absolute mean difference : "
        f"{mean_difference:.6f}\n"
    )

    f.write(
        f"Mean absolute STD difference : "
        f"{std_difference:.6f}\n\n"
    )

    f.write(
        "PCA\n"
    )

    f.write(
        f"PC1 explained variance : "
        f"{explained_variance[0]:.6f}\n"
    )

    f.write(
        f"PC2 explained variance : "
        f"{explained_variance[1]:.6f}\n\n"
    )

    f.write(
        "DISTANCE\n"
    )

    f.write(
        f"Mean nearest-real distance : "
        f"{mean_nearest_distance:.6f}\n"
    )

    f.write(
        f"Median nearest-real distance : "
        f"{median_nearest_distance:.6f}\n\n"
    )

    f.write(
        "SYNTHETIC DIVERSITY\n"
    )

    f.write(
        f"Mean pairwise distance : "
        f"{synthetic_diversity_mean:.6f}\n"
    )

    f.write(
        f"Median pairwise distance : "
        f"{synthetic_diversity_median:.6f}\n\n"
    )

    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "Synthetic samples passed basic numerical integrity "
        "checks. Distribution similarity and PCA results "
        "should be interpreted together with the GAN training "
        "behavior. No claim of biological validity is made "
        "solely from this validation.\n"
    )


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 32 — FINAL SUMMARY")
print("=" * 70)

print(
    f"Real Control samples       : {len(real_control)}"
)

print(
    f"Synthetic Control samples  : {len(synthetic_control)}"
)

print(
    f"Genetic features            : {len(feature_cols)}"
)

print(
    f"Basic validation            : {basic_status}"
)

print(
    f"Mean KS statistic           : {mean_ks:.6f}"
)

print(
    f"KS p >= 0.05 proportion     : {ks_pass_rate:.4f}"
)

print(
    f"Mean nearest-real distance  : "
    f"{mean_nearest_distance:.6f}"
)

print(
    f"Synthetic mean pairwise distance : "
    f"{synthetic_diversity_mean:.6f}"
)

print("\nFiles saved:")

print(summary_path)
print(pca_path)
print(report_path)

print("\nStep 32 completed.")