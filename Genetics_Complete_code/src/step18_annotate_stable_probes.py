import os
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

GPL_FILE = r"data/raw/GPL3427/GPL3427-tbl-1.txt"

STABILITY_FILE = (
    "genetics/outputs/feature_stability/"
    "feature_selection_frequency.csv"
)

OUTPUT_DIR = (
    "genetics/outputs/gene_annotation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# LOAD GPL3427
# ==========================================================

print("Loading GPL3427 annotation...")

gpl = pd.read_csv(
    GPL_FILE,
    sep="\t",
    comment="#",
    dtype=str
)

print("Done.")

print()

print("=" * 60)
print("GPL3427 COLUMNS")
print("=" * 60)

print(gpl.columns.tolist())


# ==========================================================
# PREPARE GPL3427
# ==========================================================

# Actual file structure:
#
# 1,1        -> Probe position
# M86720     -> GenBank accession
# Unnamed: 2 -> additional/empty field


gpl = gpl.iloc[:, :3].copy()

gpl.columns = [
    "Probe",
    "GenBank_Accession",
    "SPOT_ID"
]


# Clean values

for column in gpl.columns:

    gpl[column] = (
        gpl[column]
        .astype(str)
        .str.strip()
    )


# Convert missing values

gpl["GenBank_Accession"] = (
    gpl["GenBank_Accession"]
    .replace(
        [
            "",
            "nan",
            "NaN",
            "None",
            "NULL",
            "null"
        ],
        pd.NA
    )
)


gpl["SPOT_ID"] = (
    gpl["SPOT_ID"]
    .replace(
        [
            "",
            "nan",
            "NaN",
            "None",
            "NULL",
            "null"
        ],
        pd.NA
    )
)


# ==========================================================
# LOAD FEATURE STABILITY
# ==========================================================

print()

print("Loading feature stability results...")

stability = pd.read_csv(
    STABILITY_FILE,
    dtype=str
)

print("Done.")


# ==========================================================
# INSPECT STABILITY COLUMNS
# ==========================================================

print()

print("=" * 60)
print("FEATURE STABILITY COLUMNS")
print("=" * 60)

print(
    stability.columns.tolist()
)


# ==========================================================
# CORRECT FEATURE COLUMN
# ==========================================================

# feature_selection_frequency.csv uses:
#
# Feature
#
# GPL3427 uses:
#
# Probe
#
# Rename Feature -> Probe so that the two
# tables can be joined.


if "Feature" in stability.columns:

    stability = stability.rename(
        columns={
            "Feature": "Probe"
        }
    )

elif "Probe" in stability.columns:

    pass

else:

    raise ValueError(
        "Could not find Feature or Probe column "
        "in feature_selection_frequency.csv"
    )


# ==========================================================
# CLEAN PROBE VALUES
# ==========================================================

stability["Probe"] = (
    stability["Probe"]
    .astype(str)
    .str.strip()
)


gpl["Probe"] = (
    gpl["Probe"]
    .astype(str)
    .str.strip()
)


# ==========================================================
# NUMERIC COLUMNS
# ==========================================================

for column in [

    "Selection_Frequency",
    "Mean_MI_Score",
    "Min_MI_Score",
    "Max_MI_Score"

]:

    if column in stability.columns:

        stability[column] = pd.to_numeric(
            stability[column],
            errors="coerce"
        )


# ==========================================================
# MAP FEATURES
# ==========================================================

print()

print(
    "Mapping stable features to GPL3427..."
)


annotated = stability.merge(

    gpl,

    on="Probe",

    how="left",

    validate="one_to_one"

)


# ==========================================================
# MAPPING SUMMARY
# ==========================================================

total = len(
    annotated
)

mapped = annotated[
    "GenBank_Accession"
].notna().sum()

unmapped = total - mapped


print()

print("=" * 60)

print(
    "PROBE ANNOTATION SUMMARY"
)

print("=" * 60)

print()

print(
    "Stable feature rows :",
    total
)

print(
    "Mapped to GenBank   :",
    mapped
)

print(
    "Unmapped            :",
    unmapped
)


# ==========================================================
# STABILITY CATEGORY
# ==========================================================

def classify_frequency(
    frequency
):

    if pd.isna(frequency):

        return "Unknown"

    frequency = int(
        frequency
    )

    if frequency == 5:

        return "5_of_5"

    elif frequency == 4:

        return "4_of_5"

    elif frequency == 3:

        return "3_of_5"

    elif frequency == 2:

        return "2_of_5"

    else:

        return "1_of_5"


annotated[
    "Stability_Category"
] = (

    annotated[
        "Selection_Frequency"
    ]

    .apply(
        classify_frequency
    )

)


# ==========================================================
# REORDER COLUMNS
# ==========================================================

desired_columns = [

    "Probe",

    "Selection_Frequency",

    "Stability_Category",

    "Mean_MI_Score",

    "Min_MI_Score",

    "Max_MI_Score",

    "GenBank_Accession",

    "SPOT_ID"

]


# Only retain columns that actually exist

column_order = [

    column

    for column in desired_columns

    if column in annotated.columns

]


annotated = annotated[
    column_order
]


# ==========================================================
# SORT
# ==========================================================

sort_columns = []

ascending = []


if "Selection_Frequency" in annotated.columns:

    sort_columns.append(
        "Selection_Frequency"
    )

    ascending.append(False)


if "Mean_MI_Score" in annotated.columns:

    sort_columns.append(
        "Mean_MI_Score"
    )

    ascending.append(False)


if sort_columns:

    annotated = annotated.sort_values(

        sort_columns,

        ascending=ascending

    )


# ==========================================================
# SAVE COMPLETE ANNOTATION
# ==========================================================

complete_file = os.path.join(

    OUTPUT_DIR,

    "stable_features_with_genbank.csv"

)


annotated.to_csv(

    complete_file,

    index=False

)


# ==========================================================
# SAVE 3+ FOLD FEATURES
# ==========================================================

stable_3plus = annotated[

    annotated[
        "Selection_Frequency"
    ] >= 3

].copy()


stable_3plus_file = os.path.join(

    OUTPUT_DIR,

    "stable_3plus_features_with_genbank.csv"

)


stable_3plus.to_csv(

    stable_3plus_file,

    index=False

)


# ==========================================================
# SAVE 4+ FOLD FEATURES
# ==========================================================

stable_4plus = annotated[

    annotated[
        "Selection_Frequency"
    ] >= 4

].copy()


stable_4plus_file = os.path.join(

    OUTPUT_DIR,

    "stable_4plus_features_with_genbank.csv"

)


stable_4plus.to_csv(

    stable_4plus_file,

    index=False

)


# ==========================================================
# SAVE 5/5 FEATURES
# ==========================================================

stable_5 = annotated[

    annotated[
        "Selection_Frequency"
    ] == 5

].copy()


stable_5_file = os.path.join(

    OUTPUT_DIR,

    "stable_5of5_features_with_genbank.csv"

)


stable_5.to_csv(

    stable_5_file,

    index=False

)


# ==========================================================
# MOST STABLE FEATURES
# ==========================================================

print()

print("=" * 60)

print(
    "MOST STABLE FEATURES"
)

print("=" * 60)

print()

print(

    annotated.head(
        20
    ).to_string(
        index=False
    )

)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print()

print("=" * 60)

print(
    "ANNOTATION FILES SAVED"
)

print("=" * 60)

print()

print(
    complete_file
)

print(
    stable_3plus_file
)

print(
    stable_4plus_file
)

print(
    stable_5_file
)

print()

print(
    "Probe-to-GenBank annotation completed."
)