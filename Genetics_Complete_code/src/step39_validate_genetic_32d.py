import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 39 — GENETIC 32-D REPRESENTATION VALIDATION
# ============================================================

print("=" * 70)
print("STEP 39 — GENETIC 32-D REPRESENTATION VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_128D = (
    r"D:\org genetical data\genetics\outputs"
    r"\genetic_embedding\objective2_input"
    r"\genetic_objective2_input.csv"
)

INPUT_32D = (
    r"D:\org genetical data\genetics\outputs"
    r"\objective2_genetic\cross_modal_alignment"
    r"\genetic_32d_aligned_representation.csv"
)

OUTPUT_DIR = (
    r"D:\org genetical data\genetics\outputs"
    r"\objective2_genetic\cross_modal_alignment"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "genetic_32d_validation_report.txt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("\nLoading original 128-D genetic representation...")
df128 = pd.read_csv(INPUT_128D)
print("Done.")

print("\nLoading genetic 32-D representation...")
df32 = pd.read_csv(INPUT_32D)
print("Done.")

# ------------------------------------------------------------
# IDENTIFY COLUMNS
# ------------------------------------------------------------

ID_COL = "GSM_ID"
LABEL_COL = "Target"

embedding128 = [
    c for c in df128.columns
    if c.startswith("Genetic_Embedding_")
]

embedding32 = [
    c for c in df32.columns
    if c.startswith("Genetic_Aligned_")
]

print("\n" + "=" * 70)
print("DATASET STRUCTURE")
print("=" * 70)

print("Original 128-D shape :", df128.shape)
print("Genetic 32-D shape   :", df32.shape)

print("128-D dimensions     :", len(embedding128))
print("32-D dimensions      :", len(embedding32))

# ------------------------------------------------------------
# BASIC STRUCTURE
# ------------------------------------------------------------

checks = []

checks.append(
    ("Expected 128-D input", len(embedding128) == 128)
)

checks.append(
    ("Expected 32-D output", len(embedding32) == 32)
)

checks.append(
    ("Same number of samples",
     len(df128) == len(df32))
)

checks.append(
    ("ID column preserved",
     ID_COL in df32.columns)
)

checks.append(
    ("Target column preserved",
     LABEL_COL in df32.columns)
)

# ------------------------------------------------------------
# SAMPLE ID CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE ID VALIDATION")
print("=" * 70)

duplicate_ids = int(df32[ID_COL].duplicated().sum())

ids_match = (
    df128[ID_COL].astype(str).tolist()
    ==
    df32[ID_COL].astype(str).tolist()
)

print("Duplicate IDs :", duplicate_ids)
print("ID order match:", ids_match)

checks.append(
    ("No duplicate sample IDs", duplicate_ids == 0)
)

checks.append(
    ("Sample ID order preserved", ids_match)
)

# ------------------------------------------------------------
# LABEL CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL VALIDATION")
print("=" * 70)

label_match = (
    df128[LABEL_COL].tolist()
    ==
    df32[LABEL_COL].tolist()
)

print("\nOriginal labels:")
print(df128[LABEL_COL].value_counts())

print("\n32-D labels:")
print(df32[LABEL_COL].value_counts())

print("\nLabels preserved:", label_match)

checks.append(
    ("Labels preserved", label_match)
)

# ------------------------------------------------------------
# NUMERIC CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERIC VALIDATION")
print("=" * 70)

X32 = df32[embedding32].apply(
    pd.to_numeric,
    errors="coerce"
)

nan_count = int(X32.isna().sum().sum())
inf_count = int(
    np.isinf(X32.to_numpy(dtype=float)).sum()
)

non_numeric = [
    c for c in embedding32
    if not pd.api.types.is_numeric_dtype(df32[c])
]

print("NaN values          :", nan_count)
print("Infinite values     :", inf_count)
print("Non-numeric columns :", non_numeric)

checks.append(
    ("No NaN values", nan_count == 0)
)

checks.append(
    ("No infinite values", inf_count == 0)
)

checks.append(
    ("All 32-D columns numeric",
     len(non_numeric) == 0)
)

# ------------------------------------------------------------
# SHAPE CHECK
# ------------------------------------------------------------

expected_rows = 116
expected_dimensions = 32

shape_ok = (
    len(df32) == expected_rows
    and len(embedding32) == expected_dimensions
)

print("\nExpected samples     :", expected_rows)
print("Actual samples       :", len(df32))

print("Expected dimensions  :", expected_dimensions)
print("Actual dimensions    :", len(embedding32))

checks.append(
    ("Expected sample count", len(df32) == expected_rows)
)

checks.append(
    ("Expected embedding dimensions",
     len(embedding32) == expected_dimensions)
)

# ------------------------------------------------------------
# CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = df32[LABEL_COL].value_counts().sort_index()

for cls, count in class_counts.items():
    print(f"Class {cls}: {count}")

class_distribution_ok = (
    int(df32[LABEL_COL].value_counts().get(0, 0)) == 29
    and
    int(df32[LABEL_COL].value_counts().get(1, 0)) == 87
)

checks.append(
    ("Expected class distribution",
     class_distribution_ok)
)

# ------------------------------------------------------------
# RANGE / SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("32-D REPRESENTATION SUMMARY")
print("=" * 70)

print(
    "\nMean absolute value:",
    float(np.abs(X32.to_numpy()).mean())
)

print(
    "Minimum value:",
    float(X32.to_numpy().min())
)

print(
    "Maximum value:",
    float(X32.to_numpy().max())
)

# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

failed = []

for name, status in checks:
    print(
        f"{name:<40}: "
        f"{'PASSED' if status else 'FAILED'}"
    )

    if not status:
        failed.append(name)

validation_passed = len(failed) == 0

print("\n" + "=" * 70)

if validation_passed:
    print("VALIDATION STATUS : PASSED")
else:
    print("VALIDATION STATUS : FAILED")

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

report_lines = []

report_lines.append(
    "STEP 39 — GENETIC 32-D REPRESENTATION VALIDATION"
)

report_lines.append("")
report_lines.append("Dataset")
report_lines.append("-------")
report_lines.append(
    f"Original 128-D samples : {len(df128)}"
)
report_lines.append(
    f"Genetic 32-D samples   : {len(df32)}"
)
report_lines.append(
    f"Input dimensions       : {len(embedding128)}"
)
report_lines.append(
    f"Output dimensions      : {len(embedding32)}"
)

report_lines.append("")
report_lines.append("Quality")
report_lines.append("-------")
report_lines.append(
    f"NaN values             : {nan_count}"
)
report_lines.append(
    f"Infinite values        : {inf_count}"
)
report_lines.append(
    f"Duplicate IDs          : {duplicate_ids}"
)
report_lines.append(
    f"Sample IDs preserved   : {ids_match}"
)
report_lines.append(
    f"Labels preserved       : {label_match}"
)

report_lines.append("")
report_lines.append("Class Distribution")
report_lines.append("------------------")
report_lines.append(
    f"ASD (1)                : "
    f"{int(df32[LABEL_COL].value_counts().get(1, 0))}"
)
report_lines.append(
    f"Control (0)            : "
    f"{int(df32[LABEL_COL].value_counts().get(0, 0))}"
)

report_lines.append("")
report_lines.append("Architecture")
report_lines.append("------------")
report_lines.append(
    "Linear(128 -> 64)"
)
report_lines.append(
    "BatchNorm1d(64)"
)
report_lines.append(
    "ReLU"
)
report_lines.append(
    "Linear(64 -> 32)"
)

report_lines.append("")
report_lines.append("Final Status")
report_lines.append("------------")
report_lines.append(
    "PASSED" if validation_passed else "FAILED"
)

if failed:
    report_lines.append("")
    report_lines.append("Failed Checks")
    report_lines.append("-------------")

    for item in failed:
        report_lines.append(f"- {item}")

report_lines.append("")
report_lines.append("Output")
report_lines.append("------")
report_lines.append(INPUT_32D)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(report_lines))

# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\nReport saved:")
print(REPORT_FILE)

if validation_passed:
    print("\nGenetic 32-D representation is READY")
    print("for the three-modality cross-modal alignment.")
else:
    print("\nGenetic 32-D representation requires correction.")

print("\nStep 39 completed.")