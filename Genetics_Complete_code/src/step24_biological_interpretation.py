import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INTEGRATED_FILE = (
    "genetics/outputs/integrated_evidence/"
    "integrated_stable_feature_evidence.csv"
)

VERIFIED_ANNOTATION_FILE = (
    "genetics/outputs/gene_annotation/"
    "verified_stable_gene_annotations.csv"
)

OUTPUT_DIR = (
    "genetics/outputs/biological_interpretation"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "biological_interpretation_stable_genes.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "biological_interpretation_report.txt"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD STEP 23
# ============================================================

print("Loading integrated feature evidence...")

integrated = pd.read_csv(
    INTEGRATED_FILE
)

print("Done.")

print()
print("=" * 60)
print("STEP 23 COLUMNS")
print("=" * 60)

print(
    integrated.columns.tolist()
)


# ============================================================
# LOAD STEP 21
# ============================================================

print()
print("Loading verified gene annotations...")

verified = pd.read_csv(
    VERIFIED_ANNOTATION_FILE
)

print("Done.")

print()
print("=" * 60)
print("STEP 21 COLUMNS")
print("=" * 60)

print(
    verified.columns.tolist()
)


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "Probe" not in integrated.columns:
    raise ValueError(
        "Step 23 file does not contain 'Probe'."
    )

if "Probe" not in verified.columns:
    raise ValueError(
        "Step 21 annotation file does not contain 'Probe'."
    )


# ============================================================
# SELECT ANNOTATION COLUMNS
# ============================================================

annotation_columns = [
    "Probe"
]

possible_annotation_columns = [
    "Final_Gene_Symbol",
    "Gene_Symbol",
    "GenBank_Accession",
    "Selection_Frequency",
    "Mean_MI_Score",
    "Annotation_Confidence",
    "Historical_ASD_Study_Overlap"
]

for column in possible_annotation_columns:

    if column in verified.columns:
        annotation_columns.append(
            column
        )


verified_subset = verified[
    annotation_columns
].copy()


# ============================================================
# REMOVE DUPLICATE PROBES
# ============================================================

verified_subset = verified_subset.drop_duplicates(
    subset=["Probe"]
)


# ============================================================
# MERGE STEP 23 + STEP 21
# ============================================================

print()
print("Merging integrated evidence with verified annotations...")

df = integrated.merge(
    verified_subset,
    on="Probe",
    how="left",
    suffixes=(
        "",
        "_annotation"
    )
)

print("Done.")


# ============================================================
# DETERMINE GENE SYMBOL COLUMN
# ============================================================

# ============================================================
# DETERMINE VERIFIED GENE SYMBOL COLUMN
# ============================================================

# Step 21 is the authoritative source for verified
# gene annotations. After merging, pandas may create
# _x / _y suffixes.

if "Final_Gene_Symbol_y" in df.columns:

    gene_column = "Final_Gene_Symbol_y"

elif "Final_Gene_Symbol" in df.columns:

    gene_column = "Final_Gene_Symbol"

elif "Final_Gene_Symbol_x" in df.columns:

    gene_column = "Final_Gene_Symbol_x"

elif "Gene_Symbol" in df.columns:

    gene_column = "Gene_Symbol"

else:

    gene_column = None

print()
print("=" * 60)
print("GENE SYMBOL COLUMN")
print("=" * 60)

print(
    "Using:",
    gene_column
)


# ============================================================
# CANONICALIZE GENE SYMBOL
# ============================================================

def canonicalize_gene(symbol):

    # Safely handle pandas NA / NaN
    if pd.isna(symbol):
        return None

    symbol = str(symbol).strip()

    if symbol == "":
        return None

    # Historical alias correction
    if symbol.upper() == "FLJ20489":
        return "SLC48A1"

    return symbol


if gene_column is not None:

    df["Canonical_Gene"] = (
        df[gene_column]
        .apply(canonicalize_gene)
    )

else:

    df["Canonical_Gene"] = None


if gene_column is not None:

    df[
        "Canonical_Gene"
    ] = df[
        gene_column
    ].apply(
        canonicalize_gene
    )

else:

    df[
        "Canonical_Gene"
    ] = pd.NA


# ============================================================
# BIOLOGICAL ANNOTATIONS
# ============================================================

biological_annotations = {

    "FGFR1": {

        "Biological_Function":
            "Fibroblast growth factor receptor involved "
            "in FGF signaling and receptor tyrosine kinase "
            "signaling.",

        "Biological_Context":
            "Growth-factor signaling, development, "
            "cell proliferation and differentiation.",

        "ASD_Evidence":
            "Independent ASD candidate-gene evidence "
            "was identified during interpretation.",

        "ASD_Evidence_Level":
            "Strong candidate evidence",

        "Interpretation":
            "Highest-priority candidate because of high "
            "feature-selection stability, statistically "
            "significant expression difference and "
            "independent ASD evidence.",

        "Caution":
            "This study does not establish causality or "
            "prove that FGFR1 expression changes in brain tissue."
    },


    "UBAP1": {

        "Biological_Function":
            "Ubiquitin-associated protein involved in "
            "cellular trafficking and endosomal processes.",

        "Biological_Context":
            "Endosomal trafficking and ubiquitin-associated "
            "cellular biology.",

        "ASD_Evidence":
            "No strong ASD-specific evidence was established "
            "in this analysis.",

        "ASD_Evidence_Level":
            "Insufficient ASD-specific evidence",

        "Interpretation":
            "Potentially relevant to cellular biology but "
            "requires additional validation.",

        "Caution":
            "Should not be described as an established ASD gene "
            "from this dataset alone."
    },


    "SLC48A1": {

        "Biological_Function":
            "Intracellular heme transporter involved in "
            "heme availability and iron-related cellular processes.",

        "Biological_Context":
            "Heme transport, iron homeostasis and intracellular "
            "transport.",

        "ASD_Evidence":
            "No strong ASD-specific evidence was established "
            "in this analysis.",

        "ASD_Evidence_Level":
            "Insufficient ASD-specific evidence",

        "Interpretation":
            "Interesting metabolic and transport-related "
            "candidate requiring further investigation.",

        "Caution":
            "FLJ20489 is treated as a historical alias of "
            "SLC48A1 in this analysis."
    },


    "BDH1": {

        "Biological_Function":
            "Enzyme involved in beta-hydroxybutyrate and "
            "ketone-body metabolism.",

        "Biological_Context":
            "Ketone-body and cellular energy metabolism.",

        "ASD_Evidence":
            "No strong ASD-specific evidence was established "
            "in this analysis.",

        "ASD_Evidence_Level":
            "Insufficient ASD-specific evidence",

        "Interpretation":
            "Potential metabolic signal, but statistical "
            "evidence is limited.",

        "Caution":
            "Should be considered hypothesis-generating."
    },


    "KLK10": {

        "Biological_Function":
            "Secreted serine protease belonging to the "
            "kallikrein family.",

        "Biological_Context":
            "Extracellular proteolysis and secreted protein biology.",

        "ASD_Evidence":
            "No strong ASD-specific evidence was established "
            "in this analysis.",

        "ASD_Evidence_Level":
            "Insufficient ASD-specific evidence",

        "Interpretation":
            "Low-priority candidate because the observed "
            "expression difference was very small.",

        "Caution":
            "ML stability alone does not establish biological "
            "importance."
    },


    "GALE": {

        "Biological_Function":
            "UDP-galactose-4-epimerase involved in "
            "galactose and nucleotide-sugar metabolism.",

        "Biological_Context":
            "Galactose metabolism and nucleotide-sugar metabolism.",

        "ASD_Evidence":
            "No strong ASD-specific evidence was established "
            "in this analysis.",

        "ASD_Evidence_Level":
            "Insufficient ASD-specific evidence",

        "Interpretation":
            "Potential metabolic candidate that remains "
            "hypothesis-generating.",

        "Caution":
            "Nominal statistical significance should not be "
            "interpreted as confirmed ASD association."
    }
}


# ============================================================
# ADD BIOLOGICAL INFORMATION
# ============================================================

for column in [
    "Biological_Function",
    "Biological_Context",
    "ASD_Evidence",
    "ASD_Evidence_Level",
    "Interpretation",
    "Caution"
]:

    df[column] = pd.NA


for gene, annotation in biological_annotations.items():

    mask = (
        df["Canonical_Gene"]
        == gene
    )

    for column, value in annotation.items():

        df.loc[
            mask,
            column
        ] = value


# ============================================================
# BIOLOGICAL PRIORITY
# ============================================================

def calculate_priority(row):

    gene = row.get(
        "Canonical_Gene"
    )

    frequency = row.get(
        "Selection_Frequency",
        0
    )

    fdr = row.get(
        "FDR_p_value",
        1
    )

    effect = row.get(
        "Cohens_d",
        0
    )

    # --------------------------------------------------------
    # Safely handle missing values
    # --------------------------------------------------------

    if pd.isna(gene):
        gene = ""

    else:
        gene = str(gene).strip()

    if pd.isna(frequency):
        frequency = 0

    if pd.isna(fdr):
        fdr = 1

    if pd.isna(effect):
        effect = 0

    try:
        frequency = float(frequency)
    except:
        frequency = 0

    try:
        fdr = float(fdr)
    except:
        fdr = 1

    try:
        effect = abs(float(effect))
    except:
        effect = 0

    # --------------------------------------------------------
    # Highest priority
    # --------------------------------------------------------

    if (
        gene == "FGFR1"
        and frequency >= 4
        and fdr < 0.05
        and effect >= 0.5
    ):

        return "Highest_Priority"

    # --------------------------------------------------------
    # Strong statistical evidence
    # --------------------------------------------------------

    if (
        fdr < 0.05
        and effect >= 0.5
    ):

        return "High_Statistical_Priority"

    # --------------------------------------------------------
    # Exploratory candidates
    # --------------------------------------------------------

    if (
        frequency >= 3
        and (
            fdr < 0.10
            or effect >= 0.5
        )
    ):

        return "Exploratory_Priority"

    # --------------------------------------------------------
    # Low priority
    # --------------------------------------------------------

    return "Low_Priority"

df[
    "Biological_Priority"
] = df.apply(
    calculate_priority,
    axis=1
)


# ============================================================
# INTERPRETATION TYPE
# ============================================================

df[
    "Interpretation_Type"
] = (
    "Hypothesis_generating"
)

df.loc[
    df[
        "Biological_Priority"
    ]
    == "Highest_Priority",
    "Interpretation_Type"
] = (
    "High_priority_hypothesis"
)


# ============================================================
# SORT
# ============================================================

priority_order = {
    "Highest_Priority": 1,
    "High_Statistical_Priority": 2,
    "Exploratory_Priority": 3,
    "Low_Priority": 4
}

df[
    "_priority"
] = df[
    "Biological_Priority"
].map(
    priority_order
).fillna(99)


df = df.sort_values(
    by=[
        "_priority",
        "FDR_p_value"
    ],
    ascending=[
        True,
        True
    ]
)


df = df.drop(
    columns=[
        "_priority"
    ]
)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "STEP 24 — BIOLOGICAL INTERPRETATION\n"
    )

    report.write(
        "=" * 70
        +
        "\n\n"
    )

    report.write(
        "This step integrates the machine-learning feature "
        "stability and statistical evidence with verified "
        "gene annotations.\n\n"
    )

    report.write(
        "Important annotation correction:\n"
    )

    report.write(
        "FLJ20489 is treated as a historical alias of "
        "SLC48A1.\n\n"
    )

    report.write(
        "GENE INTERPRETATIONS\n"
    )

    report.write(
        "=" * 70
        +
        "\n\n"
    )

    for _, row in df.iterrows():

        gene = row[
            "Canonical_Gene"
        ]

        if pd.isna(gene):
            gene = "Not assigned"

        report.write(
            f"Gene: {gene}\n"
        )

        report.write(
            f"Probe: {row.get('Probe', '')}\n"
        )

        report.write(
            f"Selection frequency: "
            f"{row.get('Selection_Frequency', '')}/5\n"
        )

        report.write(
            f"Direction: "
            f"{row.get('Direction', '')}\n"
        )

        report.write(
            f"Cohen's d: "
            f"{row.get('Cohens_d', '')}\n"
        )

        report.write(
            f"FDR p-value: "
            f"{row.get('FDR_p_value', '')}\n"
        )

        report.write(
            f"Mann-Whitney FDR: "
            f"{row.get('MW_FDR_p_value', '')}\n"
        )

        report.write(
            f"Biological priority: "
            f"{row.get('Biological_Priority', '')}\n"
        )

        report.write(
            f"Function: "
            f"{row.get('Biological_Function', '')}\n"
        )

        report.write(
            f"Context: "
            f"{row.get('Biological_Context', '')}\n"
        )

        report.write(
            f"ASD evidence: "
            f"{row.get('ASD_Evidence', '')}\n"
        )

        report.write(
            f"Interpretation: "
            f"{row.get('Interpretation', '')}\n"
        )

        report.write(
            f"Caution: "
            f"{row.get('Caution', '')}\n"
        )

        report.write(
            "\n"
            +
            "-" * 70
            +
            "\n\n"
        )


    report.write(
        "OVERALL CONCLUSION\n"
    )

    report.write(
        "=" * 70
        +
        "\n\n"
    )

    report.write(
        "FGFR1 represents the highest-priority candidate in "
        "the integrated analysis because it combines strong "
        "machine-learning stability with statistically significant "
        "expression differences and independent ASD-related evidence.\n\n"
    )

    report.write(
        "The remaining genes are treated as exploratory candidates. "
        "The analysis does not establish causality, diagnostic utility, "
        "or a clinical biomarker.\n\n"
    )

    report.write(
        "The study uses lymphoblastoid cell-line expression data, "
        "therefore expression differences should not automatically "
        "be interpreted as equivalent to gene-expression changes "
        "in human brain tissue.\n"
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 60)
print("STEP 24 — BIOLOGICAL INTERPRETATION")
print("=" * 60)

print()

print(
    "Stable features evaluated:",
    len(df)
)

print(
    "Gene-annotated features:",
    df[
        "Canonical_Gene"
    ].notna().sum()
)

print()

print(
    "Biological priority distribution:"
)

print(
    df[
        "Biological_Priority"
    ].value_counts(
        dropna=False
    ).to_string()
)

print()

print(
    "Gene annotations identified:"
)

for gene in sorted(
    df[
        "Canonical_Gene"
    ]
    .dropna()
    .unique()
):

    print(
        " -",
        gene
    )

print()

print(
    "Primary candidate: FGFR1"
)

print(
    "Alias correction: FLJ20489 -> SLC48A1"
)

print()

print(
    "Files saved:"
)

print(
    OUTPUT_FILE
)

print(
    REPORT_FILE
)

print()

print(
    "Step 24 completed successfully."
)