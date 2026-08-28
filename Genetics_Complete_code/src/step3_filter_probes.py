import os
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

MISSING_THRESHOLD = 75      # Change here if needed

INPUT_FILE = r"data/processed/expression_matrix.csv"

OUTPUT_FILE = r"data/processed/filtered_expression_matrix.csv"

REMOVED_FILE = r"data/removed/removed_probes.csv"

REPORT_FILE = r"data/reports/probe_filter_report.txt"

# ==========================================================
# Create folders
# ==========================================================

os.makedirs("data/reports", exist_ok=True)
os.makedirs("data/removed", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("Loading Expression Matrix...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")

num_samples = df.shape[1] - 1

# ==========================================================
# Calculate Missing %
# ==========================================================

missing_percentage = (

    df.iloc[:,1:].isna().sum(axis=1)

    / num_samples

) * 100

df["Missing_Percentage"] = missing_percentage

# ==========================================================
# Split Data
# ==========================================================

removed = df[

    df["Missing_Percentage"] > MISSING_THRESHOLD

].copy()

filtered = df[

    df["Missing_Percentage"] <= MISSING_THRESHOLD

].copy()

# Remove helper column

filtered = filtered.drop(columns=["Missing_Percentage"])

removed.to_csv(REMOVED_FILE, index=False)

filtered.to_csv(OUTPUT_FILE, index=False)

# ==========================================================
# Statistics
# ==========================================================

original = len(df)

removed_count = len(removed)

remaining = len(filtered)

removed_percent = (removed_count/original)*100

# ==========================================================
# Report
# ==========================================================

with open(REPORT_FILE,"w") as f:

    f.write("="*60+"\n")

    f.write("PROBE FILTER REPORT\n")

    f.write("="*60+"\n\n")

    f.write(f"Threshold Used : {MISSING_THRESHOLD}%\n\n")

    f.write(f"Original Probes : {original}\n")

    f.write(f"Removed Probes : {removed_count}\n")

    f.write(f"Remaining Probes : {remaining}\n")

    f.write(f"Removed Percentage : {removed_percent:.2f}%\n")

print("="*60)

print("PROBE FILTER SUMMARY")

print("="*60)

print(f"Threshold : {MISSING_THRESHOLD}%")

print(f"Original Probes : {original}")

print(f"Removed : {removed_count}")

print(f"Remaining : {remaining}")

print("\nFiltered dataset saved successfully.")