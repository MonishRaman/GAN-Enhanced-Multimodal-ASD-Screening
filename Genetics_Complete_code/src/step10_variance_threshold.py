import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# ==========================================================
# CONFIGURATION
# ==========================================================

VARIANCE_THRESHOLD = 0.01

INPUT_FILE = r"data/processed/genetic_dataset.csv"

OUTPUT_FILE = r"data/processed/variance_filtered_dataset.csv"

REPORT_FILE = r"data/reports/variance_threshold_report.txt"

REMOVED_FILE = r"data/reports/removed_low_variance_features.csv"

os.makedirs("data/reports", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")

# ----------------------------------------------------------
# Metadata
# ----------------------------------------------------------

metadata_cols = ["GSM_ID", "Source", "Subtype", "Target"]

metadata = df[metadata_cols]

X = df.drop(columns=metadata_cols)

# ----------------------------------------------------------
# Variance Threshold
# ----------------------------------------------------------

selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)

X_filtered = selector.fit_transform(X)

selected_features = X.columns[selector.get_support()]

removed_features = X.columns[~selector.get_support()]

# ----------------------------------------------------------
# Save Removed Features
# ----------------------------------------------------------

pd.DataFrame({
    "Removed_Feature": removed_features
}).to_csv(
    REMOVED_FILE,
    index=False
)

# ----------------------------------------------------------
# Build Dataset
# ----------------------------------------------------------

filtered_df = pd.DataFrame(
    X_filtered,
    columns=selected_features
)

final_df = pd.concat(
    [metadata.reset_index(drop=True),
     filtered_df.reset_index(drop=True)],
    axis=1
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ----------------------------------------------------------
# Report
# ----------------------------------------------------------

original = X.shape[1]
remaining = filtered_df.shape[1]
removed = original - remaining

with open(REPORT_FILE, "w") as f:

    f.write("="*60 + "\n")
    f.write("VARIANCE THRESHOLD REPORT\n")
    f.write("="*60 + "\n\n")

    f.write(f"Threshold : {VARIANCE_THRESHOLD}\n\n")
    f.write(f"Original Features : {original}\n")
    f.write(f"Remaining Features : {remaining}\n")
    f.write(f"Removed Features : {removed}\n")

print("="*60)
print("VARIANCE THRESHOLD SUMMARY")
print("="*60)

print(f"Threshold : {VARIANCE_THRESHOLD}")
print(f"Original Features : {original}")
print(f"Remaining Features : {remaining}")
print(f"Removed Features : {removed}")

print("\nVariance filtering completed.")