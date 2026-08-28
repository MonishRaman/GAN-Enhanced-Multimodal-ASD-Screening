import os
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
INPUT_FILE = r"data/processed/expression_matrix.csv"

OUTPUT_DIR = r"data/processed"

REPORT_FILE = os.path.join(OUTPUT_DIR, "data_quality_report.txt")

PROBE_REPORT = os.path.join(OUTPUT_DIR, "probe_missing_summary.csv")

SAMPLE_REPORT = os.path.join(OUTPUT_DIR, "sample_missing_summary.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading Expression Matrix...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")

# -----------------------------
# Basic Information
# -----------------------------
num_probes = df.shape[0]

num_samples = df.shape[1] - 1

total_cells = num_probes * num_samples

missing_values = df.iloc[:,1:].isna().sum().sum()

missing_percentage = (missing_values / total_cells) * 100

memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)

# -----------------------------
# Missing Values per Probe
# -----------------------------
probe_missing = pd.DataFrame({

    "Position": df["Position"],

    "Missing_Count": df.iloc[:,1:].isna().sum(axis=1)

})

probe_missing["Missing_Percentage"] = (

    probe_missing["Missing_Count"] / num_samples

) * 100

probe_missing.to_csv(PROBE_REPORT, index=False)

# -----------------------------
# Missing Values per Sample
# -----------------------------
sample_missing = pd.DataFrame({

    "Sample": df.columns[1:],

    "Missing_Count": df.iloc[:,1:].isna().sum()

})

sample_missing["Missing_Percentage"] = (

    sample_missing["Missing_Count"] / num_probes

) * 100

sample_missing.to_csv(SAMPLE_REPORT, index=False)

# -----------------------------
# Duplicate Samples
# -----------------------------
duplicates = df.iloc[:,1:].T.duplicated().sum()

# -----------------------------
# Constant Features
# -----------------------------
constant_features = (

    df.iloc[:,1:].nunique(dropna=True) <= 1

).sum()

# -----------------------------
# Save Report
# -----------------------------
with open(REPORT_FILE, "w") as f:

    f.write("="*60 + "\n")

    f.write("DATA QUALITY ASSESSMENT REPORT\n")

    f.write("="*60 + "\n\n")

    f.write(f"Number of Samples : {num_samples}\n")

    f.write(f"Number of Probes  : {num_probes}\n")

    f.write(f"Dataset Shape     : {df.shape}\n\n")

    f.write(f"Memory Usage (MB) : {memory_usage:.2f}\n\n")

    f.write(f"Missing Values    : {missing_values}\n")

    f.write(f"Missing %         : {missing_percentage:.2f}%\n\n")

    f.write(f"Duplicate Samples : {duplicates}\n")

    f.write(f"Constant Features : {constant_features}\n")

print("="*60)

print("DATA QUALITY SUMMARY")

print("="*60)

print(f"Samples           : {num_samples}")

print(f"Probes            : {num_probes}")

print(f"Missing Values    : {missing_values}")

print(f"Missing %         : {missing_percentage:.2f}%")

print(f"Duplicate Samples : {duplicates}")

print(f"Constant Features : {constant_features}")

print("\nReports Saved Successfully.")