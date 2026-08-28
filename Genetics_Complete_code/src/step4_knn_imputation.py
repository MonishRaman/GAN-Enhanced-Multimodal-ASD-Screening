import os
import pandas as pd
from sklearn.impute import KNNImputer

# ==========================================================
# CONFIGURATION
# ==========================================================

KNN_NEIGHBORS = 5
WEIGHTS = "uniform"

INPUT_FILE = r"data/processed/filtered_expression_matrix.csv"

OUTPUT_FILE = r"data/processed/knn_imputed_expression_matrix.csv"

REPORT_FILE = r"data/reports/knn_imputation_report.txt"

os.makedirs("data/reports", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("Loading filtered dataset...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")

position = df["Position"]

features = df.drop(columns=["Position"])

# ==========================================================
# Missing Before
# ==========================================================

missing_before = features.isna().sum().sum()

# ==========================================================
# KNN Imputation
# ==========================================================

print("Performing KNN Imputation...")

imputer = KNNImputer(
    n_neighbors=KNN_NEIGHBORS,
    weights=WEIGHTS
)

imputed = imputer.fit_transform(features)

imputed_df = pd.DataFrame(
    imputed,
    columns=features.columns
)

# ==========================================================
# Missing After
# ==========================================================

missing_after = imputed_df.isna().sum().sum()

# ==========================================================
# Save Dataset
# ==========================================================

final_df = pd.concat([position, imputed_df], axis=1)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==========================================================
# Report
# ==========================================================

with open(REPORT_FILE, "w") as f:

    f.write("="*60 + "\n")

    f.write("KNN IMPUTATION REPORT\n")

    f.write("="*60 + "\n\n")

    f.write(f"K Neighbors : {KNN_NEIGHBORS}\n")
    f.write(f"Weights     : {WEIGHTS}\n\n")

    f.write(f"Missing Before : {missing_before}\n")
    f.write(f"Missing After  : {missing_after}\n")

print("="*60)
print("KNN IMPUTATION SUMMARY")
print("="*60)

print(f"K = {KNN_NEIGHBORS}")
print(f"Missing Before : {missing_before}")
print(f"Missing After  : {missing_after}")

print("\nImputed dataset saved successfully.")