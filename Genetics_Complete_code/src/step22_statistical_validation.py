import os
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ==========================================================
# CONFIGURATION
# ==========================================================

# Change this only if your Step 5 output has a different name.
EXPRESSION_FILE = (
    "data/processed/"
    "normalized_expression_matrix.csv"
)

LABEL_FILE = (
    "data/processed/"
    "labels.csv"
)

ANNOTATION_FILE = (
    "genetics/outputs/gene_annotation/"
    "verified_stable_gene_annotations.csv"
)

OUTPUT_DIR = (
    "genetics/outputs/statistical_validation"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "stable_features_statistical_validation.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# HELPER: COHEN'S D
# ==========================================================

def cohens_d(
    group1,
    group2
):

    group1 = np.asarray(
        group1,
        dtype=float
    )

    group2 = np.asarray(
        group2,
        dtype=float
    )

    n1 = len(group1)
    n2 = len(group2)

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    var1 = np.var(
        group1,
        ddof=1
    )

    var2 = np.var(
        group2,
        ddof=1
    )

    pooled_sd = np.sqrt(
        (
            (n1 - 1) * var1
            +
            (n2 - 1) * var2
        )
        /
        (
            n1 + n2 - 2
        )
    )

    if pooled_sd == 0:
        return np.nan

    return (
        (mean1 - mean2)
        /
        pooled_sd
    )


# ==========================================================
# LOAD EXPRESSION
# ==========================================================

print(
    "Loading normalized expression matrix..."
)

expression = pd.read_csv(
    EXPRESSION_FILE,
    dtype=str
)

print("Done.")


# ==========================================================
# LOAD LABELS
# ==========================================================

print(
    "Loading labels..."
)

labels = pd.read_csv(
    LABEL_FILE,
    dtype=str
)

print("Done.")


# ==========================================================
# LOAD ANNOTATION
# ==========================================================

print(
    "Loading verified annotations..."
)

annotation = pd.read_csv(
    ANNOTATION_FILE,
    dtype=str
)

print("Done.")


# ==========================================================
# IDENTIFY SAMPLE COLUMN
# ==========================================================

# Expression matrix from the previous pipeline has:
#
# Position | GSM386518 | GSM386519 | ...
#
# Therefore the first column is the probe identifier.

probe_column = expression.columns[0]

expression = expression.rename(
    columns={
        probe_column: "Probe"
    }
)


# ==========================================================
# NUMERIC EXPRESSION
# ==========================================================

sample_columns = [
    c for c in expression.columns
    if c != "Probe"
]


for column in sample_columns:

    expression[column] = pd.to_numeric(
        expression[column],
        errors="coerce"
    )


# ==========================================================
# NORMALIZE LABEL TABLE
# ==========================================================

labels["Target"] = pd.to_numeric(
    labels["Target"],
    errors="coerce"
)


labels = labels[
    labels["Target"].isin(
        [0, 1]
    )
].copy()


# ==========================================================
# SAMPLE ALIGNMENT
# ==========================================================

available_samples = [
    s for s in sample_columns
    if s in set(
        labels["GSM_ID"]
    )
]


print()
print("=" * 60)
print("STEP 22 — STATISTICAL VALIDATION")
print("=" * 60)
print()

print(
    "Expression samples:",
    len(sample_columns)
)

print(
    "Labeled samples:",
    len(labels)
)

print(
    "Common samples:",
    len(available_samples)
)


# ==========================================================
# RESTRICT TO COMMON SAMPLES
# ==========================================================

labels_indexed = labels.set_index(
    "GSM_ID"
)

common_labels = labels_indexed.loc[
    available_samples
]


asd_samples = [
    s for s in available_samples
    if common_labels.loc[
        s,
        "Target"
    ] == 1
]

control_samples = [
    s for s in available_samples
    if common_labels.loc[
        s,
        "Target"
    ] == 0
]


print()
print(
    "ASD samples:",
    len(asd_samples)
)

print(
    "Control samples:",
    len(control_samples)
)


# ==========================================================
# STABLE FEATURES
# ==========================================================

stable_probes = (
    annotation[
        "Probe"
    ]
    .astype(str)
    .str.strip()
    .tolist()
)


print()
print(
    "Stable probes:",
    len(stable_probes)
)


# ==========================================================
# CREATE EXPRESSION LOOKUP
# ==========================================================

expression_lookup = expression.set_index(
    "Probe"
)


# ==========================================================
# STATISTICAL TESTING
# ==========================================================

results = []


for probe in stable_probes:

    if probe not in expression_lookup.index:
        continue

    row = expression_lookup.loc[
        probe
    ]

    asd_values = pd.to_numeric(
        row[
            asd_samples
        ],
        errors="coerce"
    ).dropna().values

    control_values = pd.to_numeric(
        row[
            control_samples
        ],
        errors="coerce"
    ).dropna().values


    if (
        len(asd_values) < 2
        or
        len(control_values) < 2
    ):
        continue


    # ------------------------------------------------------
    # MEANS
    # ------------------------------------------------------

    asd_mean = np.mean(
        asd_values
    )

    control_mean = np.mean(
        control_values
    )


    # ------------------------------------------------------
    # STANDARD DEVIATIONS
    # ------------------------------------------------------

    asd_std = np.std(
        asd_values,
        ddof=1
    )

    control_std = np.std(
        control_values,
        ddof=1
    )


    # ------------------------------------------------------
    # MEAN DIFFERENCE
    # ------------------------------------------------------

    mean_difference = (
        asd_mean
        -
        control_mean
    )


    # ------------------------------------------------------
    # WELCH T-TEST
    # ------------------------------------------------------

    t_stat, t_pvalue = ttest_ind(
        asd_values,
        control_values,
        equal_var=False
    )


    # ------------------------------------------------------
    # MANN-WHITNEY U
    # ------------------------------------------------------

    try:

        u_stat, mw_pvalue = mannwhitneyu(
            asd_values,
            control_values,
            alternative="two-sided"
        )

    except Exception:

        u_stat = np.nan
        mw_pvalue = np.nan


    # ------------------------------------------------------
    # COHEN'S D
    # ------------------------------------------------------

    effect_size = cohens_d(
        asd_values,
        control_values
    )


    # ------------------------------------------------------
    # DIRECTION
    # ------------------------------------------------------

    if mean_difference > 0:

        direction = "Higher_in_ASD"

    elif mean_difference < 0:

        direction = "Lower_in_ASD"

    else:

        direction = "No_difference"


    results.append({

        "Probe": probe,

        "ASD_N": len(asd_values),

        "Control_N": len(control_values),

        "ASD_Mean": asd_mean,

        "Control_Mean": control_mean,

        "ASD_SD": asd_std,

        "Control_SD": control_std,

        "Mean_Difference":
            mean_difference,

        "Direction":
            direction,

        "Welch_t_stat":
            t_stat,

        "Welch_p_value":
            t_pvalue,

        "Mann_Whitney_U":
            u_stat,

        "Mann_Whitney_p_value":
            mw_pvalue,

        "Cohens_d":
            effect_size

    })


results_df = pd.DataFrame(
    results
)


# ==========================================================
# MULTIPLE TESTING CORRECTION
# ==========================================================

if len(results_df) > 0:

    (
        reject,
        p_adjusted,
        _,
        _
    ) = multipletests(
        results_df[
            "Welch_p_value"
        ],
        method="fdr_bh"
    )

    results_df[
        "FDR_p_value"
    ] = p_adjusted

    results_df[
        "FDR_Significant"
    ] = reject


    (
        reject_mw,
        p_adjusted_mw,
        _,
        _
    ) = multipletests(
        results_df[
            "Mann_Whitney_p_value"
        ],
        method="fdr_bh"
    )

    results_df[
        "MW_FDR_p_value"
    ] = p_adjusted_mw

    results_df[
        "MW_FDR_Significant"
    ] = reject_mw


# ==========================================================
# MERGE ANNOTATION
# ==========================================================

annotation_columns = [
    "Probe",
    "Selection_Frequency",
    "Mean_MI_Score",
    "GenBank_Accession",
    "Final_Gene_Symbol",
    "Final_Gene_Name",
    "Annotation_Confidence",
    "Historical_ASD_Study_Overlap"
]


annotation_subset = annotation[
    [
        c for c in annotation_columns
        if c in annotation.columns
    ]
].copy()


results_df = results_df.merge(
    annotation_subset,
    on="Probe",
    how="left"
)


# ==========================================================
# EFFECT SIZE INTERPRETATION
# ==========================================================

def interpret_effect(
    d
):

    if pd.isna(d):
        return "Not_available"

    magnitude = abs(d)

    if magnitude < 0.2:
        return "Negligible"

    elif magnitude < 0.5:
        return "Small"

    elif magnitude < 0.8:
        return "Medium"

    else:
        return "Large"


results_df[
    "Effect_Size_Category"
] = results_df[
    "Cohens_d"
].apply(
    interpret_effect
)


# ==========================================================
# SORT
# ==========================================================

results_df = results_df.sort_values(
    by=[
        "FDR_p_value",
        "Welch_p_value"
    ],
    ascending=True
)


# ==========================================================
# SAVE
# ==========================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

print()
print("=" * 60)
print("STATISTICAL VALIDATION SUMMARY")
print("=" * 60)
print()

print(
    "Features tested:",
    len(results_df)
)

print(
    "Nominal p < 0.05:",
    (
        results_df[
            "Welch_p_value"
        ] < 0.05
    ).sum()
)

print(
    "FDR q < 0.05:",
    (
        results_df[
            "FDR_p_value"
        ] < 0.05
    ).sum()
)

print(
    "Mann-Whitney FDR < 0.05:",
    (
        results_df[
            "MW_FDR_p_value"
        ] < 0.05
    ).sum()
)

print()

print(
    "Statistically strongest features:"
)

display_columns = [
    "Probe",
    "Selection_Frequency",
    "GenBank_Accession",
    "Final_Gene_Symbol",
    "ASD_Mean",
    "Control_Mean",
    "Mean_Difference",
    "Direction",
    "Welch_p_value",
    "FDR_p_value",
    "Cohens_d",
    "Effect_Size_Category"
]


print(
    results_df[
        [
            c
            for c in display_columns
            if c in results_df.columns
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)

print()
print(
    "Step 22 completed successfully."
)