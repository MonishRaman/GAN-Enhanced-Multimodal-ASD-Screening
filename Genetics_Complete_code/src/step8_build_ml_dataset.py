import os
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

EXPRESSION_FILE = r"data/processed/mapped_expression_matrix.csv"
LABEL_FILE = r"data/processed/labels.csv"

OUTPUT_FILE = r"data/processed/genetic_dataset.csv"

REPORT_FILE = r"data/reports/genetic_dataset_report.txt"

os.makedirs("data/reports", exist_ok=True)

# ==========================================================
# Load Files
# ==========================================================

print("Loading mapped expression matrix...")

expr = pd.read_csv(EXPRESSION_FILE)

print("Loading labels...")

labels = pd.read_csv(LABEL_FILE)

print("Done.\n")

# ==========================================================
# Build Expression Matrix
# ==========================================================

probe_names = expr["Position"].astype(str).tolist()

sample_names = expr.columns[1:].tolist()

expression_only = expr.iloc[:, 1:]

# Transpose: rows -> patients
expression_transposed = expression_only.T

expression_transposed.columns = probe_names

expression_transposed.insert(0, "GSM_ID", sample_names)

# ==========================================================
# Merge Labels
# ==========================================================

dataset = expression_transposed.merge(
    labels,
    on="GSM_ID",
    how="inner"
)

# ==========================================================
# Validation
# ==========================================================

missing_labels = dataset["Target"].isna().sum()

duplicates = dataset["GSM_ID"].duplicated().sum()

autism = (dataset["Target"] == 1).sum()

control = (dataset["Target"] == 0).sum()

# ==========================================================
# Save
# ==========================================================

dataset.to_csv(OUTPUT_FILE, index=False)

# ==========================================================
# Report
# ==========================================================

with open(REPORT_FILE, "w") as f:

    f.write("="*60 + "\n")
    f.write("FINAL GENETIC DATASET REPORT\n")
    f.write("="*60 + "\n\n")

    f.write(f"Samples            : {len(dataset)}\n")
    f.write(f"Genetic Features   : {len(probe_names)}\n")
    f.write(f"Dataset Shape      : {dataset.shape}\n\n")

    f.write(f"Autism             : {autism}\n")
    f.write(f"Control            : {control}\n\n")

    f.write(f"Missing Labels     : {missing_labels}\n")
    f.write(f"Duplicate Samples  : {duplicates}\n")

print("="*60)
print("FINAL DATASET SUMMARY")
print("="*60)

print(f"Samples          : {len(dataset)}")
print(f"Features         : {len(probe_names)}")
print(f"Shape            : {dataset.shape}")
print()
print(f"Autism           : {autism}")
print(f"Control          : {control}")
print()
print(f"Missing Labels   : {missing_labels}")
print(f"Duplicate Samples: {duplicates}")

print("\nFinal genetic dataset saved successfully.")