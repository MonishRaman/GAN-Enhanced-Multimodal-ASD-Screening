import os
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

STABILITY_FILE = (
    "genetics/outputs/gene_annotation/"
    "stable_3plus_features_with_genbank.csv"
)

SEQUENCE_FILE = (
    "genetics/outputs/gene_annotation/"
    "genbank_sequence_annotations.csv"
)

OUTPUT_DIR = (
    "genetics/outputs/gene_annotation"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "verified_stable_gene_annotations.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading feature stability data...")
stability = pd.read_csv(
    STABILITY_FILE,
    dtype=str
)

print("Loading sequence annotations...")
sequence = pd.read_csv(
    SEQUENCE_FILE,
    dtype=str
)

print("Done.")


# ==========================================================
# NUMERIC CONVERSION
# ==========================================================

stability["Selection_Frequency"] = pd.to_numeric(
    stability["Selection_Frequency"],
    errors="coerce"
)

stability["Mean_MI_Score"] = pd.to_numeric(
    stability["Mean_MI_Score"],
    errors="coerce"
)

stability["Min_MI_Score"] = pd.to_numeric(
    stability["Min_MI_Score"],
    errors="coerce"
)

stability["Max_MI_Score"] = pd.to_numeric(
    stability["Max_MI_Score"],
    errors="coerce"
)


# ==========================================================
# KEEP >= 3/5 FEATURES
# ==========================================================

stability = stability[
    stability["Selection_Frequency"] >= 3
].copy()


print()
print("=" * 60)
print("STEP 21 — VERIFIED GENE ANNOTATION")
print("=" * 60)
print()

print(
    "Stable features:",
    len(stability)
)


# ==========================================================
# MERGE SEQUENCE INFORMATION
# ==========================================================

sequence_columns = [
    "GenBank_Accession",
    "Nucleotide_UID",
    "Definition",
    "DB_XREFS",
    "UniGene",
    "Mapping_Status"
]

sequence_subset = sequence[
    [
        c for c in sequence_columns
        if c in sequence.columns
    ]
].copy()


sequence_subset = sequence_subset.drop_duplicates(
    subset=["GenBank_Accession"]
)


annotated = stability.merge(
    sequence_subset,
    on="GenBank_Accession",
    how="left"
)


# ==========================================================
# HISTORICAL / VERIFIED MAPPINGS
# ==========================================================
#
# IMPORTANT:
# These mappings are explicitly recorded as evidence-backed
# assignments. We do NOT automatically assign unresolved
# probes.
#
# Evidence categories:
#
# HISTORICAL_STUDY
# HISTORICAL_ARRAY_ANNOTATION
# NCBI_GENE
# SEQUENCE_SIMILARITY
# UNRESOLVED
#
# ==========================================================

verified_mapping = {

    "R54846": {
        "Gene_Symbol": "FGFR1",
        "Gene_Name":
            "Fibroblast growth factor receptor 1",
        "Mapping_Status":
            "Confirmed_historical_annotation",
        "Confidence":
            "High",
        "Evidence_Source":
            "Historical ASD study classifier"
    },

    "T64881": {
        "Gene_Symbol": "UBAP1",
        "Gene_Name":
            "Ubiquitin associated protein 1",
        "Mapping_Status":
            "Confirmed_historical_annotation",
        "Confidence":
            "High",
        "Evidence_Source":
            "Historical ASD study classifier"
    },

    "T67058": {
        "Gene_Symbol": "BDH1",
        "Gene_Name":
            "3-hydroxybutyrate dehydrogenase 1",
        "Mapping_Status":
            "Supported_historical_annotation",
        "Confidence":
            "High",
        "Evidence_Source":
            "Historical cDNA array annotation"
    },

    "AA459401": {
        "Gene_Symbol": "KLK10",
        "Gene_Name":
            "Kallikrein related peptidase 10",
        "Mapping_Status":
            "Supported_historical_annotation",
        "Confidence":
            "High",
        "Evidence_Source":
            "Historical gene-array annotation"
    },

    "R16231": {
        "Gene_Symbol": "GALE",
        "Gene_Name":
            "UDP-galactose-4-epimerase",
        "Mapping_Status":
            "NCBI_gene_mapping",
        "Confidence":
            "High",
        "Evidence_Source":
            "NCBI nucleotide-to-Gene mapping"
    },

    "AA992015": {
        "Gene_Symbol":
            "FLJ20489",
        "Gene_Name":
            "Hypothetical protein / transcribed locus",
        "Mapping_Status":
            "Weak_historical_annotation",
        "Confidence":
            "Moderate",
        "Evidence_Source":
            "Historical cDNA annotation"
    }
}


# ==========================================================
# APPLY VERIFIED MAPPINGS
# ==========================================================

def get_value(
    accession,
    field,
    default=""
):

    if accession in verified_mapping:

        return verified_mapping[
            accession
        ].get(
            field,
            default
        )

    return default


annotated[
    "Verified_Gene_Symbol"
] = annotated[
    "GenBank_Accession"
].apply(
    lambda x:
        get_value(
            x,
            "Gene_Symbol"
        )
)


annotated[
    "Verified_Gene_Name"
] = annotated[
    "GenBank_Accession"
].apply(
    lambda x:
        get_value(
            x,
            "Gene_Name"
        )
)


annotated[
    "Verified_Mapping_Status"
] = annotated[
    "GenBank_Accession"
].apply(
    lambda x:
        get_value(
            x,
            "Mapping_Status",
            "Unresolved"
        )
)


annotated[
    "Annotation_Confidence"
] = annotated[
    "GenBank_Accession"
].apply(
    lambda x:
        get_value(
            x,
            "Confidence",
            "Not_assigned"
        )
)


annotated[
    "Evidence_Source"
] = annotated[
    "GenBank_Accession"
].apply(
    lambda x:
        get_value(
            x,
            "Evidence_Source",
            "No_secure_gene_annotation"
        )
)


# ==========================================================
# HISTORICAL STUDY OVERLAP
# ==========================================================

historical_overlap = {

    "R54846": "Yes",
    "T64881": "Yes"
}


annotated[
    "Historical_ASD_Study_Overlap"
] = annotated[
    "GenBank_Accession"
].apply(
    lambda x:
        historical_overlap.get(
            x,
            "No_confirmed_overlap"
        )
)


# ==========================================================
# FINAL GENE SYMBOL
# ==========================================================

annotated[
    "Final_Gene_Symbol"
] = annotated[
    "Verified_Gene_Symbol"
]

annotated[
    "Final_Gene_Name"
] = annotated[
    "Verified_Gene_Name"
]


# ==========================================================
# SORT
# ==========================================================

annotated = annotated.sort_values(
    by=[
        "Selection_Frequency",
        "Mean_MI_Score"
    ],
    ascending=[
        False,
        False
    ]
)


# ==========================================================
# SELECT FINAL COLUMNS
# ==========================================================

preferred_columns = [

    "Probe",

    "Selection_Frequency",

    "Mean_MI_Score",

    "Min_MI_Score",

    "Max_MI_Score",

    "GenBank_Accession",

    "Nucleotide_UID",

    "Definition",

    "DB_XREFS",

    "UniGene",

    "Final_Gene_Symbol",

    "Final_Gene_Name",

    "Verified_Mapping_Status",

    "Annotation_Confidence",

    "Evidence_Source",

    "Historical_ASD_Study_Overlap"
]


final_columns = [
    c for c in preferred_columns
    if c in annotated.columns
]


final_df = annotated[
    final_columns
].copy()


# ==========================================================
# SAVE
# ==========================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

mapped = final_df[
    final_df[
        "Final_Gene_Symbol"
    ].fillna("").str.strip() != ""
]

unresolved = final_df[
    final_df[
        "Final_Gene_Symbol"
    ].fillna("").str.strip() == ""
]


print()
print("=" * 60)
print("VERIFIED ANNOTATION SUMMARY")
print("=" * 60)
print()

print(
    "Stable features:",
    len(final_df)
)

print(
    "Gene-annotated:",
    len(mapped)
)

print(
    "Unresolved:",
    len(unresolved)
)

print()

print(
    "High-confidence:",
    (
        final_df[
            "Annotation_Confidence"
        ] == "High"
    ).sum()
)

print(
    "Moderate-confidence:",
    (
        final_df[
            "Annotation_Confidence"
        ] == "Moderate"
    ).sum()
)

print()

print(
    "Historical ASD-study overlap:",
    (
        final_df[
            "Historical_ASD_Study_Overlap"
        ] == "Yes"
    ).sum()
)

print()
print(
    "Verified gene annotations:"
)

print(
    final_df[
        [
            "Probe",
            "Selection_Frequency",
            "Mean_MI_Score",
            "GenBank_Accession",
            "Final_Gene_Symbol",
            "Annotation_Confidence",
            "Historical_ASD_Study_Overlap"
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "Output saved to:"
)

print(
    OUTPUT_FILE
)

print()
print(
    "Step 21 completed successfully."
)