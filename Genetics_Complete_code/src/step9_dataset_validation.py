import os
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

INPUT_FILE = r"data/processed/genetic_dataset.csv"

REPORT_FILE = r"data/reports/dataset_validation_report.txt"

DUPLICATE_FILE = r"data/reports/duplicate_features.csv"

NON_NUMERIC_FILE = r"data/reports/non_numeric_columns.csv"

os.makedirs("data/reports", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")

# ==========================================================
# Metadata Columns
# ==========================================================

metadata = [

    "GSM_ID",

    "Source",

    "Subtype",

    "Target"

]

# ==========================================================
# Feature Columns
# ==========================================================

features = [

    c for c in df.columns

    if c not in metadata

]

# ==========================================================
# Duplicate Feature Names
# ==========================================================

duplicate_columns = pd.Series(features)

duplicate_columns = duplicate_columns[

    duplicate_columns.duplicated()

]

duplicate_columns.to_csv(

    DUPLICATE_FILE,

    index=False,

    header=["Duplicate_Feature"]

)

# ==========================================================
# Non Numeric Columns
# ==========================================================

non_numeric = []

for c in features:

    if not pd.api.types.is_numeric_dtype(df[c]):

        non_numeric.append(c)

pd.DataFrame(

    non_numeric,

    columns=["Column"]

).to_csv(

    NON_NUMERIC_FILE,

    index=False

)

# ==========================================================
# Missing Values
# ==========================================================

missing = df.isna().sum().sum()

# ==========================================================
# Duplicate Samples
# ==========================================================

duplicate_samples = df["GSM_ID"].duplicated().sum()

# ==========================================================
# Class Balance
# ==========================================================

autism = (df["Target"] == 1).sum()

control = (df["Target"] == 0).sum()

# ==========================================================
# Report
# ==========================================================

with open(REPORT_FILE, "w") as f:

    f.write("="*60+"\n")

    f.write("DATASET VALIDATION REPORT\n")

    f.write("="*60+"\n\n")

    f.write(f"Dataset Shape : {df.shape}\n")

    f.write(f"Samples : {len(df)}\n")

    f.write(f"Features : {len(features)}\n\n")

    f.write(f"Missing Values : {missing}\n")

    f.write(f"Duplicate Samples : {duplicate_samples}\n")

    f.write(f"Duplicate Features : {len(duplicate_columns)}\n")

    f.write(f"Non Numeric Features : {len(non_numeric)}\n\n")

    f.write(f"Autism : {autism}\n")

    f.write(f"Control : {control}\n")

print("="*60)

print("DATASET VALIDATION")

print("="*60)

print(f"Dataset Shape : {df.shape}")

print(f"Features : {len(features)}")

print(f"Missing : {missing}")

print(f"Duplicate Samples : {duplicate_samples}")

print(f"Duplicate Features : {len(duplicate_columns)}")

print(f"Non Numeric Features : {len(non_numeric)}")

print()

print(f"Autism : {autism}")

print(f"Control : {control}")

print("\nValidation completed successfully.")