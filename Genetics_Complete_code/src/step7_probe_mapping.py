import os
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

EXPRESSION_FILE = r"data/processed/normalized_expression_matrix.csv"
GPL_FILE = r"data/raw/GPL3427/GPL3427-tbl-1.txt"
OUTPUT_FILE = r"data/processed/mapped_expression_matrix.csv"
MAPPING_FILE = r"data/processed/probe_mapping.csv"
REPORT_FILE = r"data/reports/probe_mapping_report.txt"

os.makedirs("data/reports", exist_ok=True)

print("Loading expression matrix...")
expression_df = pd.read_csv(EXPRESSION_FILE)

print("Loading GPL annotation...")

# Read GPL file
gpl_df = pd.read_csv(
    GPL_FILE,
    sep="\t",
    header=None,
    names=["Position", "Clone_ID", "Extra"],
    engine="python"
)

print("Done.\n")

# ==========================================================
# Clean Mapping
# ==========================================================

gpl_df["Position"] = gpl_df["Position"].astype(str).str.strip()

gpl_df["Clone_ID"] = (
    gpl_df["Clone_ID"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove blank clone IDs
valid_mapping = gpl_df[gpl_df["Clone_ID"] != ""]

mapping_dict = dict(
    zip(
        valid_mapping["Position"],
        valid_mapping["Clone_ID"]
    )
)

# ==========================================================
# Create Mapping Table
# ==========================================================

mapping_table = pd.DataFrame({
    "Position": expression_df["Position"],
    "Clone_ID": expression_df["Position"].map(mapping_dict)
})

mapping_table["Clone_ID"] = mapping_table["Clone_ID"].fillna(mapping_table["Position"])

mapping_table.to_csv(MAPPING_FILE, index=False)

# ==========================================================
# Replace Probe Names
# ==========================================================

mapped_df = expression_df.copy()

mapped_df["Position"] = mapping_table["Clone_ID"]

mapped_df.to_csv(OUTPUT_FILE, index=False)

# ==========================================================
# Statistics
# ==========================================================

total = len(mapping_table)

mapped = mapping_table["Clone_ID"].isin(mapping_dict.values()).sum()

unmapped = total - mapped

duplicate_ids = mapping_table["Clone_ID"].duplicated().sum()

blank_ids = (
    mapping_table["Clone_ID"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

# ==========================================================
# Report
# ==========================================================

with open(REPORT_FILE, "w") as f:

    f.write("="*60 + "\n")
    f.write("PROBE MAPPING REPORT\n")
    f.write("="*60 + "\n\n")

    f.write(f"Total Probes        : {total}\n")
    f.write(f"Mapped Probes       : {mapped}\n")
    f.write(f"Unmapped Probes     : {unmapped}\n")
    f.write(f"Duplicate Clone IDs : {duplicate_ids}\n")
    f.write(f"Blank Clone IDs     : {blank_ids}\n")

print("="*60)
print("PROBE MAPPING SUMMARY")
print("="*60)

print(f"Total Probes        : {total}")
print(f"Mapped Probes       : {mapped}")
print(f"Unmapped Probes     : {unmapped}")
print(f"Duplicate Clone IDs : {duplicate_ids}")
print(f"Blank Clone IDs     : {blank_ids}")

print("\nMapped expression matrix saved successfully.")