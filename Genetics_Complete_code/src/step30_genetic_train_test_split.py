import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# STEP 30 — GENETIC TRAIN / TEST SPLIT
# Objective 2
# ============================================================

BASE_DIR = r"D:\org genetical data"

# IMPORTANT:
# Objective-1 finalized 1000-feature genetic dataset
INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_genetic_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "genetics",
    "outputs",
    "objective2_genetic",
    "train_test_split"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_train_1000_features.csv"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_test_1000_features_unseen.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_train_test_split_1000_features_report.txt"
)

# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("STEP 30 — GENETIC TRAIN / TEST SPLIT")
print("=" * 70)

print("\nLoading Objective-1 finalized genetic features...")
df = pd.read_csv(INPUT_FILE)

print("Done.")

print("\nDataset shape:", df.shape)

# ============================================================
# REQUIRED COLUMNS
# ============================================================

metadata_cols = [
    "GSM_ID",
    "Source",
    "Subtype",
    "Target"
]

missing = [
    col for col in metadata_cols
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ============================================================
# IDENTIFY GENETIC FEATURES
# ============================================================

feature_cols = [
    col for col in df.columns
    if col not in metadata_cols
]

print("\nMetadata columns:")
print(metadata_cols)

print("\nGenetic feature count:", len(feature_cols))

if len(feature_cols) != 1000:
    raise ValueError(
        f"Expected exactly 1000 genetic features, "
        f"found {len(feature_cols)}"
    )

# ============================================================
# CHECK NUMERIC FEATURES
# ============================================================

print("\nChecking feature values...")

for col in feature_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

non_numeric = df[feature_cols].columns[
    df[feature_cols].isna().any()
].tolist()

if non_numeric:
    raise ValueError(
        f"Missing/non-numeric values found in features: "
        f"{non_numeric[:10]}"
    )

print("All 1000 genetic features are numeric.")

# ============================================================
# LABEL
# ============================================================

df["Target"] = pd.to_numeric(
    df["Target"],
    errors="coerce"
).astype(int)

print("\nClass distribution:")

class_counts = df["Target"].value_counts().sort_index()

for cls, count in class_counts.items():
    print(f"Class {cls}: {count}")

# ============================================================
# SAMPLE ID CHECK
# ============================================================

if df["GSM_ID"].duplicated().any():
    raise ValueError(
        "Duplicate GSM_ID values detected."
    )

print("\nDuplicate GSM_IDs:", df["GSM_ID"].duplicated().sum())

# ============================================================
# STRATIFIED SPLIT
# ============================================================

print("\nPerforming stratified 80/20 split...")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Target"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ============================================================
# LEAKAGE CHECK
# ============================================================

train_ids = set(
    train_df["GSM_ID"].astype(str)
)

test_ids = set(
    test_df["GSM_ID"].astype(str)
)

overlap = train_ids.intersection(test_ids)

print("\nTrain/Test sample overlap:", len(overlap))

if overlap:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED."
    )

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_counts = train_df["Target"].value_counts().sort_index()
test_counts = test_df["Target"].value_counts().sort_index()

print("\n" + "=" * 70)
print("TRAINING SET")
print("=" * 70)

print("Samples:", len(train_df))

for cls, count in train_counts.items():
    print(f"Class {cls}: {count}")

print("\n" + "=" * 70)
print("UNSEEN TEST SET")
print("=" * 70)

print("Samples:", len(test_df))

for cls, count in test_counts.items():
    print(f"Class {cls}: {count}")

# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

# ============================================================
# REPORT
# ============================================================

report = f"""
STEP 30 — GENETIC TRAIN / TEST SPLIT
====================================

Objective:
Prepare the finalized 1000-feature genetic representation
for Objective 2 Conditional GAN training.

Input
-----
{INPUT_FILE}

Input shape:
{df.shape}

Metadata columns
----------------
GSM_ID
Source
Subtype
Target

Genetic feature count
---------------------
1000

Total samples
-------------
{len(df)}

Class distribution
------------------
ASD (Target=1): {class_counts.get(1, 0)}
Control (Target=0): {class_counts.get(0, 0)}

Train/Test Split
----------------
Method: Stratified 80/20 split
Random state: 42

Training samples: {len(train_df)}
Testing samples : {len(test_df)}

Training distribution
---------------------
ASD: {train_counts.get(1, 0)}
Control: {train_counts.get(0, 0)}

Unseen test distribution
------------------------
ASD: {test_counts.get(1, 0)}
Control: {test_counts.get(0, 0)}

Leakage check
-------------
Train/Test GSM_ID overlap: {len(overlap)}

Status:
PASSED

Objective-2 GAN rule
--------------------
The Conditional GAN must be trained ONLY using the
training set.

The unseen test set must NOT be used during:
- GAN training
- Synthetic sample generation
- Encoder training
- Hyperparameter tuning

The test set is reserved for final evaluation.

Output
------
Training:
{TRAIN_FILE}

Unseen test:
{TEST_FILE}
"""

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 30 FINAL STATUS")
print("=" * 70)

print(f"Total samples        : {len(df)}")
print(f"Genetic features     : {len(feature_cols)}")
print(f"Training samples     : {len(train_df)}")
print(f"Unseen test samples  : {len(test_df)}")

print("\nTraining ASD     :", train_counts.get(1, 0))
print("Training Control :", train_counts.get(0, 0))

print("\nTest ASD         :", test_counts.get(1, 0))
print("Test Control     :", test_counts.get(0, 0))

print("\nTrain/Test leakage check: PASSED")

print("\nTraining file:")
print(TRAIN_FILE)

print("\nUnseen test file:")
print(TEST_FILE)

print("\nReport:")
print(REPORT_FILE)

print("\nStep 30 completed successfully.")