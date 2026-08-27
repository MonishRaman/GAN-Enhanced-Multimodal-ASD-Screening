import os
import pandas as pd

# -------------------------------------------------
# Paths
# -------------------------------------------------

INPUT_FILE = r"data/processed/expression_matrix.csv"

OUTPUT_DIR = r"data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

print("Loading expression matrix...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")

num_samples = df.shape[1] - 1

# -------------------------------------------------
# Missing Percentage per Probe
# -------------------------------------------------

probe_missing = pd.DataFrame()

probe_missing["Position"] = df["Position"]

probe_missing["Missing_Count"] = df.iloc[:,1:].isna().sum(axis=1)

probe_missing["Missing_Percentage"] = (

    probe_missing["Missing_Count"] / num_samples

) * 100

# -------------------------------------------------
# Threshold Counts
# -------------------------------------------------

above50 = probe_missing[
    probe_missing["Missing_Percentage"] > 50
]

above75 = probe_missing[
    probe_missing["Missing_Percentage"] > 75
]

above90 = probe_missing[
    probe_missing["Missing_Percentage"] > 90
]

# -------------------------------------------------
# Save CSVs
# -------------------------------------------------

above50.to_csv(

    os.path.join(OUTPUT_DIR, "probes_above_50.csv"),

    index=False

)

above75.to_csv(

    os.path.join(OUTPUT_DIR, "probes_above_75.csv"),

    index=False

)

above90.to_csv(

    os.path.join(OUTPUT_DIR, "probes_above_90.csv"),

    index=False

)

# -------------------------------------------------
# Statistics
# -------------------------------------------------

stats = []

stats.append("="*60)

stats.append("MISSING VALUE DISTRIBUTION")

stats.append("="*60)

stats.append(f"Total Probes : {len(probe_missing)}")

stats.append(f"Average Missing % : {probe_missing['Missing_Percentage'].mean():.2f}")

stats.append(f"Median Missing % : {probe_missing['Missing_Percentage'].median():.2f}")

stats.append(f"Maximum Missing % : {probe_missing['Missing_Percentage'].max():.2f}")

stats.append("")

stats.append(f"Probes > 50% Missing : {len(above50)}")

stats.append(f"Probes > 75% Missing : {len(above75)}")

stats.append(f"Probes > 90% Missing : {len(above90)}")

stats.append("")

stats.append("Top 10 Most Missing Probes")

stats.append("")

top10 = probe_missing.sort_values(

    by="Missing_Percentage",

    ascending=False

).head(10)

stats.append(top10.to_string(index=False))

# -------------------------------------------------
# Save Report
# -------------------------------------------------

report_path = os.path.join(

    OUTPUT_DIR,

    "missing_value_statistics.txt"

)

with open(report_path, "w") as f:

    for line in stats:

        f.write(line + "\n")

# -------------------------------------------------
# Console Output
# -------------------------------------------------

print("="*60)

print("MISSING VALUE ANALYSIS")

print("="*60)

print(f"Average Missing % : {probe_missing['Missing_Percentage'].mean():.2f}")

print(f"Median Missing %  : {probe_missing['Missing_Percentage'].median():.2f}")

print(f"Maximum Missing % : {probe_missing['Missing_Percentage'].max():.2f}")

print()

print(f"Probes >50% Missing : {len(above50)}")

print(f"Probes >75% Missing : {len(above75)}")

print(f"Probes >90% Missing : {len(above90)}")

print("\nReports saved successfully.")