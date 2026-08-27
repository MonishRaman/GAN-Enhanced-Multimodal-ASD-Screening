import os
import numpy as np
import pandas as pd

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition


# ============================================================
# STEP 33 — GENETIC CTGAN AUGMENTATION
# ============================================================

BASE = r"D:\org genetical data"

EMBEDDING_FILE = os.path.join(
    BASE,
    r"genetics\outputs\genetic_embedding\genetic_128d_embeddings_trained_oof.csv"
)

TRAIN_FILE = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\train_test_split\genetic_train_1000_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    r"genetics\outputs\objective2_genetic\ctgan"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SYNTHETIC_FILE = os.path.join(
    OUTPUT_DIR,
    "synthetic_control_128d_ctgan.csv"
)

BALANCED_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_balanced_128d_ctgan.csv"
)

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_ctgan_128d.pkl"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_ctgan_128d_report.txt"
)


print("=" * 70)
print("STEP 33 — GENETIC CTGAN AUGMENTATION")
print("=" * 70)


# ============================================================
# 1. LOAD TRAINING EMBEDDINGS
# ============================================================

print("\nLoading trained genetic embeddings...")

embeddings = pd.read_csv(EMBEDDING_FILE)

print("Done.")
print("Embedding shape:", embeddings.shape)


# ============================================================
# 2. LOAD OBJECTIVE-2 TRAINING IDS
# ============================================================

print("\nLoading Objective-2 training sample IDs...")

train_df = pd.read_csv(TRAIN_FILE)

print("Done.")
print("Training dataset shape:", train_df.shape)


# ============================================================
# 3. VERIFY REQUIRED COLUMNS
# ============================================================

if "GSM_ID" not in embeddings.columns:
    raise ValueError("GSM_ID not found in embedding dataset.")

if "Target" not in embeddings.columns:
    raise ValueError("Target not found in embedding dataset.")

if "GSM_ID" not in train_df.columns:
    raise ValueError("GSM_ID not found in training dataset.")


# ============================================================
# 4. SELECT ONLY OBJECTIVE-2 TRAINING SAMPLES
# ============================================================

print("\nRestricting embeddings to Objective-2 training samples...")

train_ids = set(train_df["GSM_ID"].astype(str))

embeddings["GSM_ID"] = embeddings["GSM_ID"].astype(str)

train_embeddings = embeddings[
    embeddings["GSM_ID"].isin(train_ids)
].copy()

print("Training embeddings:", train_embeddings.shape)


# ============================================================
# 5. VERIFY NO TEST LEAKAGE
# ============================================================

if len(train_embeddings) != len(train_df):
    raise ValueError(
        f"Training ID mismatch: expected {len(train_df)}, "
        f"found {len(train_embeddings)}"
    )

duplicate_ids = train_embeddings["GSM_ID"].duplicated().sum()

if duplicate_ids > 0:
    raise ValueError(
        f"Duplicate GSM_ID values detected: {duplicate_ids}"
    )

print("Training sample count verified:", len(train_embeddings))
print("Duplicate IDs:", duplicate_ids)


# ============================================================
# 6. DETECT EMBEDDING COLUMNS
# ============================================================

embedding_columns = [
    col for col in train_embeddings.columns
    if col.startswith("Genetic_Embedding_")
]

print("\nGenetic embedding dimensions:", len(embedding_columns))

if len(embedding_columns) != 128:
    raise ValueError(
        f"Expected 128 embedding dimensions, "
        f"found {len(embedding_columns)}"
    )


# ============================================================
# 7. CREATE CTGAN DATASET
# ============================================================

ctgan_df = train_embeddings[
    embedding_columns + ["Target"]
].copy()

# Make target categorical/string.
# This is important because we want conditional sampling.
ctgan_df["Target"] = ctgan_df["Target"].astype(str)

# Check numeric embeddings
for col in embedding_columns:
    ctgan_df[col] = pd.to_numeric(
        ctgan_df[col],
        errors="coerce"
    )

nan_count = int(ctgan_df[embedding_columns].isna().sum().sum())
inf_count = int(
    np.isinf(
        ctgan_df[embedding_columns].to_numpy(dtype=float)
    ).sum()
)

print("\nEmbedding NaN values:", nan_count)
print("Embedding Inf values:", inf_count)

if nan_count > 0:
    raise ValueError("NaN values detected in embeddings.")

if inf_count > 0:
    raise ValueError("Infinite values detected in embeddings.")


# ============================================================
# 8. CLASS DISTRIBUTION
# ============================================================

class_counts = ctgan_df["Target"].value_counts()

print("\nClass distribution:")
print(class_counts)

majority_class = class_counts.idxmax()
minority_class = class_counts.idxmin()

majority_count = int(class_counts.max())
minority_count = int(class_counts.min())

synthetic_needed = majority_count - minority_count

print("\nMajority class :", majority_class)
print("Minority class :", minority_class)
print("Synthetic required:", synthetic_needed)


# ============================================================
# 9. DETECT SDV METADATA
# ============================================================

print("\nDetecting SDV metadata...")

metadata = Metadata.detect_from_dataframe(
    data=ctgan_df
)

# Explicitly mark Target as categorical.
metadata.update_column(
    column_name="Target",
    sdtype="categorical"
)

print("Metadata detected.")


# ============================================================
# 10. CREATE CTGAN
# ============================================================

print("\nCreating CTGAN...")

synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=300,
    batch_size=64,
    verbose=True
)

print("CTGAN created.")


# ============================================================
# 11. TRAIN CTGAN
# ============================================================

print("\nTraining CTGAN...")
print("IMPORTANT: only the 92 training samples are used.")

synthesizer.fit(ctgan_df)

print("\nCTGAN training completed.")


# ============================================================
# 12. SAVE MODEL
# ============================================================

synthesizer.save(MODEL_FILE)

print("\nCTGAN model saved:")
print(MODEL_FILE)


# ============================================================
# 13. GENERATE MINORITY CLASS
# ============================================================

print("\nGenerating synthetic Control samples...")

condition = Condition(
    num_rows=synthetic_needed,
    column_values={
        "Target": minority_class
    }
)

synthetic = synthesizer.sample_from_conditions(
    conditions=[condition]
)

print("Synthetic samples generated:", len(synthetic))


# ============================================================
# 14. VERIFY SYNTHETIC DATA
# ============================================================

synthetic_nan = int(
    synthetic[embedding_columns].isna().sum().sum()
)

synthetic_inf = int(
    np.isinf(
        synthetic[embedding_columns].to_numpy(dtype=float)
    ).sum()
)

print("\nSynthetic NaN values:", synthetic_nan)
print("Synthetic Inf values:", synthetic_inf)

if synthetic_nan > 0:
    raise ValueError("Synthetic data contains NaN values.")

if synthetic_inf > 0:
    raise ValueError("Synthetic data contains Inf values.")


# Force correct target
synthetic["Target"] = minority_class


# ============================================================
# 15. SAVE SYNTHETIC DATA
# ============================================================

synthetic.to_csv(
    SYNTHETIC_FILE,
    index=False
)

print("\nSynthetic Control dataset saved:")
print(SYNTHETIC_FILE)


# ============================================================
# 16. CREATE BALANCED DATASET
# ============================================================

real_ctgan = ctgan_df.copy()

balanced_df = pd.concat(
    [
        real_ctgan,
        synthetic[
            embedding_columns + ["Target"]
        ]
    ],
    ignore_index=True
)


# ============================================================
# 17. VERIFY BALANCE
# ============================================================

final_counts = balanced_df["Target"].value_counts()

print("\n" + "=" * 70)
print("FINAL BALANCED DATASET")
print("=" * 70)

print("Total samples:", len(balanced_df))
print("\nClass distribution:")
print(final_counts)

expected_total = majority_count * 2

if len(balanced_df) != expected_total:
    raise ValueError(
        f"Unexpected balanced size: {len(balanced_df)}"
    )

if final_counts.get(majority_class, 0) != majority_count:
    raise ValueError("Majority class count incorrect.")

if final_counts.get(minority_class, 0) != majority_count:
    raise ValueError("Minority class count incorrect.")

print("\nClass balancing: PASSED")


# ============================================================
# 18. SAVE BALANCED DATASET
# ============================================================

balanced_df.to_csv(
    BALANCED_FILE,
    index=False
)

print("\nBalanced dataset saved:")
print(BALANCED_FILE)


# ============================================================
# 19. REPORT
# ============================================================

report = f"""
STEP 33 — GENETIC CTGAN AUGMENTATION REPORT

Input trained embeddings:
{EMBEDDING_FILE}

Objective-2 training dataset:
{TRAIN_FILE}

Training samples:
{len(train_embeddings)}

Genetic embedding dimensions:
{len(embedding_columns)}

Training ASD samples:
{class_counts.get("1", 0)}

Training Control samples:
{class_counts.get("0", 0)}

Majority class:
{majority_class}

Minority class:
{minority_class}

Synthetic samples generated:
{synthetic_needed}

Final balanced samples:
{len(balanced_df)}

Final ASD samples:
{final_counts.get("1", 0)}

Final Control samples:
{final_counts.get("0", 0)}

Synthetic NaN values:
{synthetic_nan}

Synthetic Inf values:
{synthetic_inf}

CTGAN epochs:
300

CTGAN model:
{MODEL_FILE}

Synthetic Control samples:
{SYNTHETIC_FILE}

Balanced dataset:
{BALANCED_FILE}

STATUS:
CTGAN augmentation completed successfully.
"""

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("\nReport saved:")
print(REPORT_FILE)

print("\n" + "=" * 70)
print("STEP 33 COMPLETED")
print("=" * 70)