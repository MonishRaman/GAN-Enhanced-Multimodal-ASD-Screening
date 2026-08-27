import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==========================================================
# Paths
# ==========================================================

INPUT_FILE = r"data/processed/knn_imputed_expression_matrix.csv"
OUTPUT_FILE = r"data/processed/normalized_expression_matrix.csv"
REPORT_FILE = r"data/reports/zscore_normalization_report.txt"

os.makedirs("data/reports", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("Loading KNN imputed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")

position = df["Position"]

features = df.drop(columns=["Position"])

# ==========================================================
# Statistics Before
# ==========================================================

overall_mean_before = features.values.mean()
overall_std_before = features.values.std()

# ==========================================================
# Z-score Normalization
# ==========================================================

print("Performing Z-score Normalization...")

# IMPORTANT:
# Samples are rows for StandardScaler,
# so transpose -> scale each probe -> transpose back.

scaler = StandardScaler()

normalized = scaler.fit_transform(features.T).T

normalized_df = pd.DataFrame(
    normalized,
    columns=features.columns
)

# ==========================================================
# Statistics After
# ==========================================================

overall_mean_after = normalized_df.values.mean()
overall_std_after = normalized_df.values.std()

# ==========================================================
# Save Dataset
# ==========================================================

final_df = pd.concat([position, normalized_df], axis=1)

final_df.to_csv(OUTPUT_FILE, index=False)

# ==========================================================
# Report
# ==========================================================

with open(REPORT_FILE, "w") as f:

    f.write("="*60 + "\n")
    f.write("Z-SCORE NORMALIZATION REPORT\n")
    f.write("="*60 + "\n\n")

    f.write(f"Mean Before : {overall_mean_before:.6f}\n")
    f.write(f"Std Before  : {overall_std_before:.6f}\n\n")

    f.write(f"Mean After  : {overall_mean_after:.6f}\n")
    f.write(f"Std After   : {overall_std_after:.6f}\n")

print("="*60)
print("Z-SCORE NORMALIZATION SUMMARY")
print("="*60)

print(f"Mean Before : {overall_mean_before:.6f}")
print(f"Std Before  : {overall_std_before:.6f}")

print()

print(f"Mean After  : {overall_mean_after:.6f}")
print(f"Std After   : {overall_std_after:.6f}")

print("\nNormalized dataset saved successfully.")