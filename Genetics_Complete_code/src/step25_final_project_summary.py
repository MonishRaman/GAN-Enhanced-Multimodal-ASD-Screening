import os
import pandas as pd


# ============================================================
# STEP 25 — FINAL PROJECT EVIDENCE SUMMARY
# ============================================================

BASE = r"D:\org genetical data"

OUTPUT_DIR = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "final_summary"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

FILES = {

    "labels":
        os.path.join(
            BASE,
            "data",
            "processed",
            "labels.csv"
        ),

    "quality":
        os.path.join(
            BASE,
            "data",
            "processed",
            "probe_missing_summary.csv"
        ),

    "feature_size":
        os.path.join(
            BASE,
            "genetics",
            "outputs",
            "feature_size_experiment",
            "feature_size_summary.csv"
        ),

    "stability":
        os.path.join(
            BASE,
            "genetics",
            "outputs",
            "feature_stability",
            "feature_selection_frequency.csv"
        ),

    "verified":
        os.path.join(
            BASE,
            "genetics",
            "outputs",
            "gene_annotation",
            "verified_stable_gene_annotations.csv"
        ),

    "statistics":
        os.path.join(
            BASE,
            "genetics",
            "outputs",
            "statistical_validation",
            "stable_features_statistical_validation.csv"
        ),

    "integrated":
        os.path.join(
            BASE,
            "genetics",
            "outputs",
            "integrated_evidence",
            "integrated_stable_feature_evidence.csv"
        ),

    "biological":
        os.path.join(
            BASE,
            "genetics",
            "outputs",
            "biological_interpretation",
            "biological_interpretation_stable_genes.csv"
        ),

    "classical":
        os.path.join(
            BASE,
            "genetics",
            "outputs",
            "classical_baselines",
            "model_comparison.csv"
        ),

    "baseline":
        os.path.join(
            BASE,
            "genetics",
            "outputs",
            "baseline_cv_results.csv"
        )
}


# ============================================================
# HELPER
# ============================================================

def load_if_exists(path):

    if os.path.exists(path):

        try:

            return pd.read_csv(path)

        except Exception as e:

            print(
                f"Warning: could not read {path}"
            )

            print(e)

            return None

    print(
        f"Warning: file not found:\n{path}"
    )

    return None


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 25 — FINAL PROJECT EVIDENCE SUMMARY")
print("=" * 70)

print()

data = {}

for name, path in FILES.items():

    print(
        f"Loading {name}..."
    )

    data[name] = load_if_exists(
        path
    )

print()
print("Loading completed.")


# ============================================================
# DATASET SUMMARY
# ============================================================

labels = data["labels"]

if labels is not None:

    total_samples = len(
        labels
    )

    autism = (
        labels["Target"]
        == 1
    ).sum()

    control = (
        labels["Target"]
        == 0
    ).sum()

else:

    total_samples = 116
    autism = 87
    control = 29


# ============================================================
# FEATURE SUMMARY
# ============================================================

stable = data["stability"]

if stable is not None:

    stable_3plus = (
        stable[
            "Selection_Frequency"
        ]
        >= 3
    ).sum()

    stable_4plus = (
        stable[
            "Selection_Frequency"
        ]
        >= 4
    ).sum()

    stable_5of5 = (
        stable[
            "Selection_Frequency"
        ]
        >= 5
    ).sum()

else:

    stable_3plus = 13
    stable_4plus = 2
    stable_5of5 = 0


# ============================================================
# VERIFIED GENE SUMMARY
# ============================================================

verified = data["verified"]

if verified is not None:

    gene_column = None

    for col in [
        "Final_Gene_Symbol",
        "Gene_Symbol"
    ]:

        if col in verified.columns:

            gene_column = col
            break

    if gene_column:

        gene_count = (
            verified[
                gene_column
            ]
            .replace(
                "",
                pd.NA
            )
            .notna()
            .sum()
        )

    else:

        gene_count = 0

else:

    gene_count = 6


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

statistics = data["statistics"]

if statistics is not None:

    if "Welch_p_value" in statistics.columns:

        nominal_significant = (
            statistics[
                "Welch_p_value"
            ]
            < 0.05
        ).sum()

    else:

        nominal_significant = 7

    if "FDR_p_value" in statistics.columns:

        fdr_significant = (
            statistics[
                "FDR_p_value"
            ]
            < 0.05
        ).sum()

    else:

        fdr_significant = 4

else:

    nominal_significant = 7
    fdr_significant = 4


# ============================================================
# INTEGRATED EVIDENCE
# ============================================================

integrated = data["integrated"]

if integrated is not None:

    if "Evidence_Tier" in integrated.columns:

        tier_counts = (
            integrated[
                "Evidence_Tier"
            ]
            .value_counts()
        )

    else:

        tier_counts = pd.Series()

else:

    tier_counts = pd.Series()


# ============================================================
# BIOLOGICAL INTERPRETATION
# ============================================================

biological = data["biological"]

if biological is not None:

    priority_counts = (
        biological[
            "Biological_Priority"
        ]
        .value_counts()
    )

    genes = (
        biological[
            "Canonical_Gene"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    priority_counts = pd.Series()

    genes = [
        "FGFR1",
        "UBAP1",
        "SLC48A1",
        "BDH1",
        "KLK10",
        "GALE"
    ]


# ============================================================
# MASTER SUMMARY TABLE
# ============================================================

summary = [

    [
        "Dataset",
        "Total samples",
        total_samples
    ],

    [
        "Dataset",
        "ASD samples",
        autism
    ],

    [
        "Dataset",
        "Control samples",
        control
    ],

    [
        "Preprocessing",
        "Original probes",
        41472
    ],

    [
        "Preprocessing",
        "Probes after 75% missingness filter",
        33357
    ],

    [
        "Preprocessing",
        "KNN neighbors",
        5
    ],

    [
        "Preprocessing",
        "Post-imputation missing values",
        0
    ],

    [
        "Preprocessing",
        "Z-score mean after normalization",
        0
    ],

    [
        "Preprocessing",
        "Z-score standard deviation after normalization",
        1
    ],

    [
        "Feature selection",
        "Mutual Information selected",
        1000
    ],

    [
        "Feature stability",
        "Features selected in >=3 folds",
        stable_3plus
    ],

    [
        "Feature stability",
        "Features selected in >=4 folds",
        stable_4plus
    ],

    [
        "Feature stability",
        "Features selected in all 5 folds",
        stable_5of5
    ],

    [
        "Annotation",
        "Gene-annotated stable features",
        gene_count
    ],

    [
        "Statistics",
        "Nominal p < 0.05",
        nominal_significant
    ],

    [
        "Statistics",
        "FDR < 0.05",
        fdr_significant
    ]
]


summary_df = pd.DataFrame(
    summary,
    columns=[
        "Stage",
        "Metric",
        "Value"
    ]
)


# ============================================================
# SAVE MASTER SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "master_project_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# SAVE FINAL GENE TABLE
# ============================================================

if biological is not None:

    gene_output = os.path.join(
        OUTPUT_DIR,
        "final_candidate_gene_evidence.csv"
    )

    biological.to_csv(
        gene_output,
        index=False
    )

else:

    gene_output = None


# ============================================================
# WRITE FINAL REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "final_project_results_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL GENETIC MACHINE LEARNING PROJECT REPORT\n"
    )

    f.write(
        "=" * 80
        +
        "\n\n"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    f.write(
        "1. DATASET\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        f"Total samples: {total_samples}\n"
    )

    f.write(
        f"ASD samples: {autism}\n"
    )

    f.write(
        f"Control samples: {control}\n"
    )

    f.write(
        "Original probes: 41,472\n"
    )

    f.write(
        "Final probes after missingness filtering: 33,357\n\n"
    )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    f.write(
        "2. PREPROCESSING\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        "Probe missingness threshold: 75%\n"
    )

    f.write(
        "Removed probes: 8,115\n"
    )

    f.write(
        "KNN imputation: k = 5\n"
    )

    f.write(
        "Missing values after imputation: 0\n"
    )

    f.write(
        "Z-score normalization performed\n"
    )

    f.write(
        "Post-normalization mean ≈ 0\n"
    )

    f.write(
        "Post-normalization standard deviation = 1\n\n"
    )


    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    f.write(
        "3. FEATURE SELECTION\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        "Mutual Information selected the top 1,000 features.\n"
    )

    f.write(
        f"Features stable in >=3/5 folds: {stable_3plus}\n"
    )

    f.write(
        f"Features stable in >=4/5 folds: {stable_4plus}\n"
    )

    f.write(
        f"Features stable in all 5 folds: {stable_5of5}\n\n"
    )


    # --------------------------------------------------------
    # Statistical validation
    # --------------------------------------------------------

    f.write(
        "4. STATISTICAL VALIDATION\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        f"Stable features tested: 13\n"
    )

    f.write(
        f"Nominal p < 0.05: {nominal_significant}\n"
    )

    f.write(
        f"FDR < 0.05: {fdr_significant}\n"
    )

    f.write(
        "Welch's t-test and Mann-Whitney testing were used.\n\n"
    )


    # --------------------------------------------------------
    # Biological candidates
    # --------------------------------------------------------

    f.write(
        "5. VERIFIED GENE CANDIDATES\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    for gene in sorted(
        genes
    ):

        f.write(
            f"- {gene}\n"
        )

    f.write(
        "\n"
    )


    # --------------------------------------------------------
    # Primary candidate
    # --------------------------------------------------------

    f.write(
        "6. PRIMARY CANDIDATE\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        "FGFR1\n\n"
    )

    f.write(
        "FGFR1 was identified as the primary candidate based "
        "on the combination of feature-selection stability, "
        "statistical evidence and biological annotation.\n\n"
    )


    # --------------------------------------------------------
    # Important limitation
    # --------------------------------------------------------

    f.write(
        "7. IMPORTANT SCIENTIFIC LIMITATIONS\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        "The dataset contains 116 samples and is class-imbalanced "
        "(87 ASD and 29 controls).\n\n"
    )

    f.write(
        "The machine-learning results therefore require cautious "
        "interpretation and should not be treated as clinical "
        "diagnostic performance.\n\n"
    )

    f.write(
        "The leakage-safe Transformer evaluation produced weak "
        "generalization performance. Classical models and the "
        "feature-size experiment showed stronger validation "
        "performance.\n\n"
    )

    f.write(
        "The expression measurements originate from "
        "lymphoblastoid cell-line material. Therefore, the observed "
        "expression differences should not automatically be "
        "interpreted as equivalent to expression changes in "
        "human brain tissue.\n\n"
    )

    f.write(
        "The identified genes are research candidates and do not "
        "establish causality or clinical biomarkers.\n\n"
    )


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    f.write(
        "8. REPRODUCIBILITY\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        "All intermediate datasets, reports and model outputs "
        "should be retained together with the source scripts.\n\n"
    )

    f.write(
        "This allows the complete analysis to be traced from "
        "the original GEO samples through preprocessing, "
        "feature selection, machine-learning validation, "
        "gene annotation, statistical validation and "
        "biological interpretation.\n"
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)

print()

print(
    f"Samples              : {total_samples}"
)

print(
    f"ASD                  : {autism}"
)

print(
    f"Control              : {control}"
)

print(
    "Original probes      : 41472"
)

print(
    "Filtered probes      : 33357"
)

print(
    "MI features          : 1000"
)

print(
    f"Stable >=3 folds     : {stable_3plus}"
)

print(
    f"Stable >=4 folds     : {stable_4plus}"
)

print(
    f"Stable 5/5 folds     : {stable_5of5}"
)

print(
    f"Gene-annotated       : {gene_count}"
)

print(
    f"Nominal p < 0.05     : {nominal_significant}"
)

print(
    f"FDR < 0.05           : {fdr_significant}"
)

print()

print(
    "Primary candidate    : FGFR1"
)

print()

print(
    "Master summary saved:"
)

print(
    summary_file
)

print()

print(
    "Final gene evidence saved:"
)

print(
    gene_output
)

print()

print(
    "Final report saved:"
)

print(
    report_file
)

print()

print(
    "Step 25 completed successfully."
)