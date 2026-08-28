import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# STEP 38 — GENETIC 128-D TO 32-D CROSS-MODAL ALIGNMENT
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("STEP 38 — GENETIC 128-D TO 32-D ALIGNMENT")
print("=" * 70)

print(f"Using device: {DEVICE}")

# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"D:\org genetical data\genetics\outputs"
    r"\genetic_embedding\objective2_input"
    r"\genetic_objective2_input.csv"
)

OUTPUT_DIR = (
    r"D:\org genetical data\genetics\outputs"
    r"\objective2_genetic"
    r"\cross_modal_alignment"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_32d_aligned_representation.csv"
)

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_128_to_32_alignment.pt"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_128_to_32_alignment_report.txt"
)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading validated genetic 128-D embeddings...")
df = pd.read_csv(INPUT_FILE)
print("Done.")

print("\nInput shape:", df.shape)

# ============================================================
# IDENTIFY COLUMNS
# ============================================================

sample_id_col = "GSM_ID"
label_col = "Target"

embedding_cols = [
    c for c in df.columns
    if c.startswith("Genetic_Embedding_")
]

print("\nSample ID column:", sample_id_col)
print("Label column:", label_col)
print("Embedding dimensions:", len(embedding_cols))

# ============================================================
# VALIDATION
# ============================================================

assert sample_id_col in df.columns
assert label_col in df.columns
assert len(embedding_cols) == 128

X = df[embedding_cols].apply(pd.to_numeric, errors="coerce")
y = df[label_col]

print("\nChecking input values...")

nan_count = int(X.isna().sum().sum())
inf_count = int(np.isinf(X.values).sum())
duplicate_ids = int(df[sample_id_col].duplicated().sum())

print("NaN values      :", nan_count)
print("Inf values      :", inf_count)
print("Duplicate IDs   :", duplicate_ids)

if nan_count != 0:
    raise ValueError("Input contains NaN values.")

if inf_count != 0:
    raise ValueError("Input contains infinite values.")

if duplicate_ids != 0:
    raise ValueError("Duplicate sample IDs detected.")

# ============================================================
# MODEL
# ============================================================

class GeneticAlignmentNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.network(x)


model = GeneticAlignmentNetwork().to(DEVICE)

# ============================================================
# REPRESENTATION GENERATION
# ============================================================

model.eval()

with torch.no_grad():

    X_tensor = torch.tensor(
        X.values,
        dtype=torch.float32,
        device=DEVICE
    )

    aligned = model(X_tensor).cpu().numpy()

print("\nAlignment completed.")

print("Input shape :", X.shape)
print("Output shape:", aligned.shape)

# ============================================================
# CREATE OUTPUT DATASET
# ============================================================

aligned_columns = [
    f"Genetic_Aligned_{i:03d}"
    for i in range(1, 33)
]

aligned_df = pd.DataFrame(
    aligned,
    columns=aligned_columns
)

output_df = pd.concat(
    [
        df[[sample_id_col, label_col]].reset_index(drop=True),
        aligned_df.reset_index(drop=True)
    ],
    axis=1
)

# ============================================================
# OUTPUT VALIDATION
# ============================================================

output_values = output_df[aligned_columns].values

output_nan = int(np.isnan(output_values).sum())
output_inf = int(np.isinf(output_values).sum())

print("\n" + "=" * 70)
print("GENETIC 32-D VALIDATION")
print("=" * 70)

print("Samples              :", len(output_df))
print("Input dimensions     :", len(embedding_cols))
print("Output dimensions    :", len(aligned_columns))
print("NaN values           :", output_nan)
print("Inf values           :", output_inf)
print("Duplicate sample IDs :", output_df[sample_id_col].duplicated().sum())

print("\nClass distribution:")
print(output_df[label_col].value_counts())

# ============================================================
# SAVE
# ============================================================

output_df.to_csv(
    OUTPUT_FILE,
    index=False
)

torch.save(
    model.state_dict(),
    MODEL_FILE
)

# ============================================================
# REPORT
# ============================================================

report = f"""
STEP 38 — GENETIC 128-D TO 32-D ALIGNMENT REPORT

Input
-----
Input file:
{INPUT_FILE}

Samples:
{len(df)}

Input genetic embedding dimensions:
{len(embedding_cols)}

Classes:
ASD = {int((y == 1).sum())}
Control = {int((y == 0).sum())}

Alignment Architecture
-----------------------
Linear(128 -> 64)
BatchNorm1d(64)
ReLU
Linear(64 -> 32)

Output
------
Output dimensions:
{len(aligned_columns)}

Output shape:
{output_df.shape}

Quality Checks
--------------
Input NaN values:
{nan_count}

Input Inf values:
{inf_count}

Output NaN values:
{output_nan}

Output Inf values:
{output_inf}

Duplicate sample IDs:
{int(output_df[sample_id_col].duplicated().sum())}

Output File
-----------
{OUTPUT_FILE}

Model File
----------
{MODEL_FILE}

Status
------
Genetic 128-D representation successfully projected
to a 32-D representation for cross-modal alignment.
"""

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(report)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 38 — FINAL SUMMARY")
print("=" * 70)

print("Input Genetic Embedding :", "128-D")
print("Output Genetic Embedding:", "32-D")
print("Samples                 :", len(output_df))
print("Output Shape            :", output_df.shape)
print("NaN                     :", output_nan)
print("Inf                     :", output_inf)

print("\nFiles saved:")
print(OUTPUT_FILE)
print(MODEL_FILE)
print(REPORT_FILE)

print("\nStep 38 completed successfully.")