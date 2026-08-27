import os
import pandas as pd
import numpy as np


# ==========================================================
# CONFIGURATION
# ==========================================================

STATISTICAL_FILE = (
    "genetics/outputs/"
    "statistical_validation/"
    "stable_features_statistical_validation.csv"
)

ANNOTATION_FILE = (
    "genetics/outputs/"
    "gene_annotation/"
    "verified_stable_gene_annotations.csv"
)

OUTPUT_DIR = (
    "genetics/outputs/"
    "integrated_evidence"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "integrated_stable_feature_evidence.csv"
)

TIER_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "evidence_tier_summary.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading statistical validation results...")
stats = pd.read_csv(
    STATISTICAL_FILE
)
print("Done.")

print("Loading verified gene annotations...")
annotation = pd.read_csv(
    ANNOTATION_FILE
)
print("Done.")


# ==========================================================
# NORMALIZE COLUMN NAMES
# ==========================================================

stats.columns = [
    str(c).strip()
    for c in stats.columns
]

annotation.columns = [
    str(c).strip()
    for c in annotation.columns
]


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required_stats = [
    "Probe",
    "Selection_Frequency",
    "Mean_MI_Score",
    "ASD_Mean",
    "Control_Mean",
    "Mean_Difference",
    "Direction",
    "Welch_p_value",
    "FDR_p_value",
    "Mann_Whitney_p_value",
    "MW_FDR_p_value",
    "Cohens_d",
    "Effect_Size_Category"
]

missing_stats = [
    c for c in required_stats
    if c not in stats.columns
]

if missing_stats:
    raise ValueError(
        "Missing statistical columns: "
        + ", ".join(missing_stats)
    )


if "Probe" not in annotation.columns:
    raise ValueError(
        "Annotation file does not contain 'Probe'."
    )


# ==========================================================
# SELECT ANNOTATION COLUMNS
# ==========================================================

annotation_candidates = [
    "Probe",
    "Selection_Frequency",
    "Mean_MI_Score",
    "GenBank_Accession",
    "Final_Gene_Symbol",
    "Final_Gene_Name",
    "Annotation_Confidence",
    "Historical_ASD_Study_Overlap"
]

available_annotation = [
    c for c in annotation_candidates
    if c in annotation.columns
]

annotation_subset = annotation[
    available_annotation
].copy()


# ==========================================================
# AVOID DUPLICATE COLUMNS
# ==========================================================

duplicate_columns = [
    c for c in [
        "Selection_Frequency",
        "Mean_MI_Score"
    ]
    if c in annotation_subset.columns
    and c in stats.columns
]

annotation_subset = annotation_subset.drop(
    columns=duplicate_columns,
    errors="ignore"
)


# ==========================================================
# MERGE
# ==========================================================

evidence = stats.merge(
    annotation_subset,
    on="Probe",
    how="left"
)


# ==========================================================
# NORMALIZE NUMERIC COLUMNS
# ==========================================================

numeric_columns = [
    "Selection_Frequency",
    "Mean_MI_Score",
    "ASD_Mean",
    "Control_Mean",
    "Mean_Difference",
    "Welch_p_value",
    "FDR_p_value",
    "Mann_Whitney_p_value",
    "MW_FDR_p_value",
    "Cohens_d"
]

for column in numeric_columns:

    if column in evidence.columns:

        evidence[column] = pd.to_numeric(
            evidence[column],
            errors="coerce"
        )


# ==========================================================
# GENE ANNOTATION STATUS
# ==========================================================

def annotation_status(row):

    symbol = row.get(
        "Final_Gene_Symbol",
        np.nan
    )

    confidence = row.get(
        "Annotation_Confidence",
        np.nan
    )

    if (
        pd.notna(symbol)
        and
        str(symbol).strip() != ""
        and
        str(symbol).strip().lower()
        not in ["nan", "na", "not_assigned"]
    ):

        if (
            pd.notna(confidence)
            and
            str(confidence).strip().lower()
            == "high"
        ):
            return "Confirmed_gene_high_confidence"

        elif (
            pd.notna(confidence)
            and
            str(confidence).strip().lower()
            == "moderate"
        ):
            return "Gene_annotation_moderate_confidence"

        return "Gene_annotation_available"

    return "Unresolved_gene_annotation"


evidence[
    "Gene_Annotation_Status"
] = evidence.apply(
    annotation_status,
    axis=1
)


# ==========================================================
# STATISTICAL SIGNIFICANCE FLAGS
# ==========================================================

evidence[
    "Welch_Nominal_Significant"
] = (
    evidence["Welch_p_value"] < 0.05
)

evidence[
    "Welch_FDR_Significant"
] = (
    evidence["FDR_p_value"] < 0.05
)

evidence[
    "MW_FDR_Significant"
] = (
    evidence["MW_FDR_p_value"] < 0.05
)


# ==========================================================
# EFFECT SIZE FLAGS
# ==========================================================

evidence[
    "Meaningful_Effect"
] = (
    evidence["Cohens_d"].abs() >= 0.5
)


# ==========================================================
# ML STABILITY FLAGS
# ==========================================================

evidence[
    "Stable_3plus"
] = (
    evidence["Selection_Frequency"] >= 3
)

evidence[
    "Stable_4plus"
] = (
    evidence["Selection_Frequency"] >= 4
)


# ==========================================================
# EVIDENCE TIER
# ==========================================================

def assign_tier(row):

    frequency = row[
        "Selection_Frequency"
    ]

    fdr = row[
        "FDR_p_value"
    ]

    mw_fdr = row[
        "MW_FDR_p_value"
    ]

    effect = abs(
        row["Cohens_d"]
    )

    annotation_status_value = row[
        "Gene_Annotation_Status"
    ]


    # ------------------------------------------------------
    # TIER 1
    #
    # Stable >=3 folds
    # Welch FDR significant
    # Meaningful effect
    # Gene annotation available
    # ------------------------------------------------------

    if (
        frequency >= 3
        and
        fdr < 0.05
        and
        effect >= 0.5
        and
        annotation_status_value
        != "Unresolved_gene_annotation"
    ):

        return "Tier_1_Strongest_Evidence"


    # ------------------------------------------------------
    # TIER 2
    #
    # Strong statistical or ML support
    # ------------------------------------------------------

    if (
        frequency >= 3
        and
        (
            fdr < 0.05
            or
            mw_fdr < 0.05
            or
            effect >= 0.5
        )
    ):

        return "Tier_2_Supporting_Evidence"


    # ------------------------------------------------------
    # TIER 3
    #
    # Stable ML feature but weak statistical support
    # ------------------------------------------------------

    if frequency >= 3:

        return "Tier_3_ML_Stable_Weak_Statistical_Evidence"


    # ------------------------------------------------------
    # TIER 4
    #
    # Anything not meeting above criteria
    # ------------------------------------------------------

    return "Tier_4_Limited_Evidence"


evidence[
    "Evidence_Tier"
] = evidence.apply(
    assign_tier,
    axis=1
)


# ==========================================================
# EVIDENCE SCORE
# ==========================================================
#
# This is NOT a statistical score.
# It is simply a transparent ranking aid.
#
# Stability:
#   5/5 = 5 points
#   4/5 = 4 points
#   3/5 = 3 points
#
# FDR:
#   <0.001 = 4
#   <0.01  = 3
#   <0.05  = 2
#   otherwise = 0
#
# Effect:
#   >=0.8 = 3
#   >=0.5 = 2
#   >=0.2 = 1
#
# Annotation:
#   High = 2
#   Moderate = 1
# ==========================================================

def evidence_score(row):

    score = 0


    # Stability
    score += min(
        int(row["Selection_Frequency"]),
        5
    )


    # FDR
    fdr = row["FDR_p_value"]

    if fdr < 0.001:
        score += 4

    elif fdr < 0.01:
        score += 3

    elif fdr < 0.05:
        score += 2


    # Effect size
    effect = abs(
        row["Cohens_d"]
    )

    if effect >= 0.8:
        score += 3

    elif effect >= 0.5:
        score += 2

    elif effect >= 0.2:
        score += 1


    # Annotation
    confidence = str(
        row.get(
            "Annotation_Confidence",
            ""
        )
    ).strip().lower()

    if confidence == "high":
        score += 2

    elif confidence == "moderate":
        score += 1


    return score


evidence[
    "Evidence_Score"
] = evidence.apply(
    evidence_score,
    axis=1
)


# ==========================================================
# RANKING
# ==========================================================

evidence = evidence.sort_values(
    by=[
        "Evidence_Score",
        "Selection_Frequency",
        "FDR_p_value"
    ],
    ascending=[
        False,
        False,
        True
    ]
)


evidence[
    "Evidence_Rank"
] = range(
    1,
    len(evidence) + 1
)


# ==========================================================
# REORDER IMPORTANT COLUMNS
# ==========================================================

preferred_order = [
    "Evidence_Rank",
    "Evidence_Tier",
    "Evidence_Score",
    "Probe",
    "Selection_Frequency",
    "Mean_MI_Score",
    "GenBank_Accession",
    "Final_Gene_Symbol",
    "Final_Gene_Name",
    "Annotation_Confidence",
    "Historical_ASD_Study_Overlap",
    "Gene_Annotation_Status",
    "ASD_Mean",
    "Control_Mean",
    "Mean_Difference",
    "Direction",
    "Cohens_d",
    "Effect_Size_Category",
    "Welch_p_value",
    "FDR_p_value",
    "Mann_Whitney_p_value",
    "MW_FDR_p_value",
    "Welch_Nominal_Significant",
    "Welch_FDR_Significant",
    "MW_FDR_Significant",
    "Meaningful_Effect",
    "Stable_3plus",
    "Stable_4plus"
]


existing_order = [
    c
    for c in preferred_order
    if c in evidence.columns
]

remaining_columns = [
    c
    for c in evidence.columns
    if c not in existing_order
]

evidence = evidence[
    existing_order
    +
    remaining_columns
]


# ==========================================================
# SAVE MAIN TABLE
# ==========================================================

evidence.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================================
# TIER SUMMARY
# ==========================================================

tier_summary = (
    evidence
    .groupby(
        "Evidence_Tier"
    )
    .agg(
        Feature_Count=(
            "Probe",
            "count"
        ),
        Mean_Selection_Frequency=(
            "Selection_Frequency",
            "mean"
        ),
        Mean_Evidence_Score=(
            "Evidence_Score",
            "mean"
        ),
        FDR_Significant_Count=(
            "Welch_FDR_Significant",
            "sum"
        ),
        MW_FDR_Significant_Count=(
            "MW_FDR_Significant",
            "sum"
        )
    )
    .reset_index()
)


tier_summary.to_csv(
    TIER_SUMMARY_FILE,
    index=False
)


# ==========================================================
# CONSOLE REPORT
# ==========================================================

print()
print("=" * 60)
print("STEP 23 — INTEGRATED FEATURE EVIDENCE")
print("=" * 60)
print()

print(
    "Stable features evaluated:",
    len(evidence)
)

print()

print(
    "Evidence tier distribution:"
)

print(
    evidence[
        "Evidence_Tier"
    ].value_counts()
    .sort_index()
    .to_string()
)


# ==========================================================
# STRONGEST FEATURES
# ==========================================================

print()
print("=" * 60)
print("STRONGEST INTEGRATED EVIDENCE")
print("=" * 60)
print()

display_columns = [
    "Evidence_Rank",
    "Evidence_Tier",
    "Probe",
    "Selection_Frequency",
    "Final_Gene_Symbol",
    "Annotation_Confidence",
    "ASD_Mean",
    "Control_Mean",
    "Direction",
    "Cohens_d",
    "Welch_p_value",
    "FDR_p_value",
    "MW_FDR_p_value",
    "Evidence_Score"
]

display_columns = [
    c
    for c in display_columns
    if c in evidence.columns
]

print(
    evidence[
        display_columns
    ].to_string(
        index=False
    )
)


# ==========================================================
# TIER 1
# ==========================================================

tier1 = evidence[
    evidence[
        "Evidence_Tier"
    ]
    ==
    "Tier_1_Strongest_Evidence"
]


print()
print("=" * 60)
print("TIER 1 FEATURES")
print("=" * 60)
print()

if len(tier1) == 0:

    print(
        "No feature met all Tier 1 criteria."
    )

else:

    print(
        tier1[
            display_columns
        ].to_string(
            index=False
        )
    )


# ==========================================================
# TIER 2
# ==========================================================

tier2 = evidence[
    evidence[
        "Evidence_Tier"
    ]
    ==
    "Tier_2_Supporting_Evidence"
]


print()
print("=" * 60)
print("TIER 2 FEATURES")
print("=" * 60)
print()

if len(tier2) == 0:

    print(
        "No Tier 2 features."
    )

else:

    print(
        tier2[
            display_columns
        ].to_string(
            index=False
        )
    )


# ==========================================================
# OUTPUT
# ==========================================================

print()
print("=" * 60)
print("FILES SAVED")
print("=" * 60)
print()

print(
    "Integrated evidence:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "Evidence tier summary:"
)

print(
    TIER_SUMMARY_FILE
)

print()
print(
    "Step 23 completed successfully."
)