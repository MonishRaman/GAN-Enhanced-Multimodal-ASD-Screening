import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 29 — GENETIC EMBEDDING VALIDATION
# Objective 2 Genetic Branch
# ============================================================

BASE_DIR = r"D:\org genetical data"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "genetics",
    "outputs",
    "genetic_embedding",
    "genetic_128d_embeddings_trained_oof.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "genetics",
    "outputs",
    "genetic_embedding",
    "objective2_input"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_objective2_input.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_objective2_input_report.txt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 29 — GENETIC EMBEDDING VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("\nLoading trained OOF genetic embeddings...")

df = pd.read_csv(INPUT_FILE)

print("Done.")

print("\nInput shape:", df.shape)
print("\nInput columns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# IDENTIFY SAMPLE ID
# ------------------------------------------------------------

id_candidates = [
    "GSM_ID",
    "Sample_ID",
    "SampleID",
    "sample_id",
    "sample",
    "ID",
    "id"
]

id_column = None

for col in id_candidates:
    if col in df.columns:
        id_column = col
        break

if id_column is None:
    raise ValueError(
        "No valid sample ID column found."
    )

print("\nSample ID column:", id_column)

# ------------------------------------------------------------
# IDENTIFY LABEL
# ------------------------------------------------------------

label_candidates = [
    "Target",
    "Label",
    "label",
    "Class",
    "class",
    "Diagnosis",
    "diagnosis"
]

label_column = None

for col in label_candidates:
    if col in df.columns:
        label_column = col
        break

if label_column is None:
    raise ValueError(
        "No valid label column found."
    )

print("Label column:", label_column)

# ------------------------------------------------------------
# IDENTIFY EXACT GENETIC EMBEDDING COLUMNS
# ------------------------------------------------------------

embedding_columns = [
    col
    for col in df.columns
    if col.startswith("Genetic_Embedding_")
]

print(
    "\nGenetic embedding dimensions detected:",
    len(embedding_columns)
)

# ------------------------------------------------------------
# CHECK EMBEDDING NUMBERING
# ------------------------------------------------------------

expected_columns = [
    f"Genetic_Embedding_{i:03d}"
    for i in range(1, 129)
]

missing_embedding_columns = [
    col
    for col in expected_columns
    if col not in df.columns
]

extra_embedding_columns = [
    col
    for col in embedding_columns
    if col not in expected_columns
]

if missing_embedding_columns:
    print("\nMissing embedding columns:")
    print(missing_embedding_columns)

if extra_embedding_columns:
    print("\nUnexpected embedding columns:")
    print(extra_embedding_columns)

# ------------------------------------------------------------
# NUMERIC CHECK
# ------------------------------------------------------------

print("\nChecking embedding values...")

X = df[embedding_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

non_numeric_columns = [
    col
    for col in embedding_columns
    if not pd.api.types.is_numeric_dtype(df[col])
]

nan_count = X.isna().sum().sum()
inf_count = np.isinf(
    X.to_numpy(dtype=float)
).sum()

print(
    "Non-numeric embedding columns:",
    non_numeric_columns
)

print("NaN values:", nan_count)
print("Inf values:", inf_count)

# ------------------------------------------------------------
# SAMPLE ID VALIDATION
# ------------------------------------------------------------

duplicate_ids = df[id_column].duplicated().sum()

missing_ids = df[id_column].isna().sum()

print("\nDuplicate sample IDs:", duplicate_ids)
print("Missing sample IDs:", missing_ids)

# ------------------------------------------------------------
# LABEL VALIDATION
# ------------------------------------------------------------

print("\nLabel distribution:")
print(df[label_column].value_counts(dropna=False))

# ------------------------------------------------------------
# SAMPLE VALIDATION
# ------------------------------------------------------------

expected_samples = 116

print("\nExpected samples:", expected_samples)
print("Actual samples  :", len(df))

# ------------------------------------------------------------
# EMBEDDING SHAPE VALIDATION
# ------------------------------------------------------------

expected_dimensions = 128

print("\nExpected dimensions:", expected_dimensions)
print("Actual dimensions  :", len(embedding_columns))

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

problems = []

if len(df) != expected_samples:
    problems.append(
        f"Expected {expected_samples} samples, "
        f"found {len(df)}"
    )

if len(embedding_columns) != expected_dimensions:
    problems.append(
        f"Expected {expected_dimensions} embeddings, "
        f"found {len(embedding_columns)}"
    )

if missing_embedding_columns:
    problems.append(
        "Missing genetic embedding columns"
    )

if extra_embedding_columns:
    problems.append(
        "Unexpected genetic embedding columns"
    )

if non_numeric_columns:
    problems.append(
        "Non-numeric embedding columns detected"
    )

if nan_count > 0:
    problems.append(
        "NaN values detected in embeddings"
    )

if inf_count > 0:
    problems.append(
        "Infinite values detected in embeddings"
    )

if duplicate_ids > 0:
    problems.append(
        "Duplicate sample IDs detected"
    )

if missing_ids > 0:
    problems.append(
        "Missing sample IDs detected"
    )

# ------------------------------------------------------------
# CREATE CLEAN OBJECTIVE 2 DATASET
# ------------------------------------------------------------

clean_columns = [
    id_column,
    label_column
] + expected_columns

clean_df = df[clean_columns].copy()

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

clean_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 29 — GENETIC EMBEDDING VALIDATION REPORT\n"
    )
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Input file: {INPUT_FILE}\n"
    )

    f.write(
        f"Output file: {OUTPUT_FILE}\n\n"
    )

    f.write(
        f"Samples: {len(df)}\n"
    )

    f.write(
        f"Embedding dimensions: {len(embedding_columns)}\n"
    )

    f.write(
        f"Sample ID column: {id_column}\n"
    )

    f.write(
        f"Label column: {label_column}\n\n"
    )

    f.write(
        f"NaN values: {nan_count}\n"
    )

    f.write(
        f"Infinite values: {inf_count}\n"
    )

    f.write(
        f"Duplicate sample IDs: {duplicate_ids}\n"
    )

    f.write(
        f"Missing sample IDs: {missing_ids}\n\n"
    )

    f.write("Label distribution:\n")

    for label, count in df[
        label_column
    ].value_counts(dropna=False).items():

        f.write(
            f"  {label}: {count}\n"
        )

    f.write("\n")

    if not problems:

        f.write(
            "VALIDATION STATUS: PASSED\n\n"
        )

        f.write(
            "The 128-dimensional genetic embedding "
            "is ready for Objective 2 cross-modal "
            "representation alignment.\n"
        )

    else:

        f.write(
            "VALIDATION STATUS: FAILED\n\n"
        )

        f.write(
            "Problems detected:\n"
        )

        for problem in problems:
            f.write(
                f"- {problem}\n"
            )

# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 29 — FINAL VALIDATION SUMMARY")
print("=" * 70)

print("Samples              :", len(df))
print("Genetic Embeddings   :", len(embedding_columns))
print("Sample ID            :", id_column)
print("Label                :", label_column)

print("NaN Values           :", nan_count)
print("Infinite Values      :", inf_count)
print("Duplicate IDs        :", duplicate_ids)
print("Missing IDs          :", missing_ids)

if not problems:

    print("\nVALIDATION STATUS : PASSED")

    print(
        "\nGenetic embedding is READY for Objective 2."
    )

else:

    print("\nVALIDATION STATUS : FAILED")

    for problem in problems:
        print(" -", problem)

print("\nObjective-2 input:")
print(OUTPUT_FILE)

print("\nReport:")
print(REPORT_FILE)

print("\nStep 29 completed.")