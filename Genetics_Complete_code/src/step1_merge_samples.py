import os
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
GSM_FOLDER = r"data/raw/GSM Sample Tables"
OUTPUT_FILE = r"data/processed/expression_matrix.csv"

# -----------------------------
# Read GSM files
# -----------------------------
expression_df = None

gsm_files = sorted(
    [f for f in os.listdir(GSM_FOLDER) if f.endswith(".txt")]
)

print(f"\nFound {len(gsm_files)} GSM files.\n")

for file in tqdm(gsm_files):

    sample_name = file.split("-")[0]      # GSM386518

    path = os.path.join(GSM_FOLDER, file)

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["Position", "Channel_A", "Channel_B"]
    )

    # Keep Row/Column position
    if expression_df is None:

        expression_df = df[["Position"]].copy()

    # Use Channel_A as gene expression
    expression_df[sample_name] = df["Channel_A"]

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

expression_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nExpression Matrix Saved!")

print(expression_df.head())

print("\nShape :", expression_df.shape)

print("\nMissing Values")

print(expression_df.isnull().sum().sum())