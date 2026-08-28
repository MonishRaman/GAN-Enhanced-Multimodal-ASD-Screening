import os
import numpy as np
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

CORRELATION_THRESHOLD = 0.95

INPUT_FILE = r"data/processed/top_features_dataset.csv"

OUTPUT_FILE = r"data/processed/final_genetic_features.csv"

REPORT_FILE = r"data/reports/correlation_filter_report.txt"

REMOVED_FILE = r"data/reports/removed_correlated_features.csv"

os.makedirs("data/reports", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")

metadata_cols = [
    "GSM_ID",
    "Source",
    "Subtype",
    "Target"
]

metadata = df[metadata_cols]

X = df.drop(columns=metadata_cols)

print(f"Features Before : {X.shape[1]}")

corr_matrix = X.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [
    column
    for column in upper.columns
    if any(upper[column] > CORRELATION_THRESHOLD)
]

print(f"Correlated Features Removed : {len(to_drop)}")

X_filtered = X.drop(columns=to_drop)

pd.DataFrame({
    "Removed_Feature": to_drop
}).to_csv(
    REMOVED_FILE,
    index=False
)

final_df = pd.concat(
    [metadata, X_filtered],
    axis=1
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

with open(REPORT_FILE, "w") as f:

    f.write("="*60+"\n")
    f.write("CORRELATION FILTER REPORT\n")
    f.write("="*60+"\n\n")

    f.write(f"Threshold : {CORRELATION_THRESHOLD}\n\n")

    f.write(f"Original Features : {X.shape[1]}\n")

    f.write(f"Remaining Features : {X_filtered.shape[1]}\n")

    f.write(f"Removed Features : {len(to_drop)}\n")

print("="*60)

print("CORRELATION FILTER SUMMARY")

print("="*60)

print(f"Original Features : {X.shape[1]}")

print(f"Remaining Features : {X_filtered.shape[1]}")

print(f"Removed Features : {len(to_drop)}")

print("\nCorrelation filtering completed.")