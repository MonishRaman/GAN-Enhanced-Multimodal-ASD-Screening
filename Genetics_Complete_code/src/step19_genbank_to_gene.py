import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE = (
    "genetics/outputs/gene_annotation/"
    "stable_3plus_features_with_genbank.csv"
)

OUTPUT_DIR = (
    "genetics/outputs/gene_annotation"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "stable_3plus_features_gene_annotation.csv"
)

RAW_FILE = os.path.join(
    OUTPUT_DIR,
    "ncbi_gene_mapping_raw.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# NCBI E-utilities
EUTILS_BASE = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/"
)

# Keep requests slow enough for NCBI.
REQUEST_DELAY = 0.4


# ==========================================================
# NCBI REQUEST FUNCTION
# ==========================================================

def ncbi_request(
    endpoint,
    params
):

    query = urllib.parse.urlencode(
        params
    )

    url = (
        EUTILS_BASE
        + endpoint
        + "?"
        + query
    )

    request = urllib.request.Request(

        url,

        headers={
            "User-Agent":
                "GSE15402-genetic-analysis/1.0"
        }

    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        return response.read()


# ==========================================================
# SEARCH NCBI GENE
# ==========================================================

def search_gene_by_accession(
    accession
):

    try:

        data = ncbi_request(

            "esearch.fcgi",

            {
                "db": "gene",

                "term":
                    f'"{accession}"[ACCN]',

                "retmode": "xml",

                "retmax": 20

            }

        )


        root = ET.fromstring(
            data
        )


        ids = [

            element.text

            for element in root.findall(
                ".//Id"
            )

            if element.text

        ]


        return ids


    except Exception as e:

        print(
            f"NCBI Gene search error "
            f"for {accession}: {e}"
        )

        return []


# ==========================================================
# GET GENE SUMMARY
# ==========================================================

def get_gene_summary(
    gene_ids
):

    if not gene_ids:

        return []


    try:

        data = ncbi_request(

            "esummary.fcgi",

            {
                "db": "gene",

                "id":
                    ",".join(
                        gene_ids
                    ),

                "retmode": "xml"

            }

        )


        root = ET.fromstring(
            data
        )


        results = []


        for document in root.findall(
            ".//DocumentSummary"
        ):

            gene_id = (
                document.attrib.get(
                    "uid",
                    ""
                )
            )


            name_element = document.find(
                "Name"
            )


            description_element = (
                document.find(
                    "Description"
                )
            )


            organism_element = (
                document.find(
                    "Orgname"
                )
            )


            chromosome_element = (
                document.find(
                    "Chromosome"
                )
            )


            map_location_element = (
                document.find(
                    "MapLocation"
                )
            )


            results.append({

                "NCBI_Gene_ID":
                    gene_id,

                "Gene_Symbol":
                    (
                        name_element.text
                        if name_element is not None
                        else ""
                    ),

                "Gene_Name":
                    (
                        description_element.text
                        if description_element is not None
                        else ""
                    ),

                "Organism":
                    (
                        organism_element.text
                        if organism_element is not None
                        else ""
                    ),

                "Chromosome":
                    (
                        chromosome_element.text
                        if chromosome_element is not None
                        else ""
                    ),

                "Map_Location":
                    (
                        map_location_element.text
                        if map_location_element is not None
                        else ""
                    )

            })


        return results


    except Exception as e:

        print(
            f"NCBI Gene summary error: {e}"
        )

        return []


# ==========================================================
# LOAD STABLE FEATURES
# ==========================================================

print(
    "Loading stable genetic features..."
)

df = pd.read_csv(
    INPUT_FILE,
    dtype=str
)

print("Done.")


# ==========================================================
# PREPARE ACCESSIONS
# ==========================================================

df["Selection_Frequency"] = pd.to_numeric(

    df["Selection_Frequency"],

    errors="coerce"

)


stable_features = df[

    df[
        "Selection_Frequency"
    ] >= 3

].copy()


stable_features = stable_features.sort_values(

    [
        "Selection_Frequency",
        "Mean_MI_Score"

    ],

    ascending=[
        False,
        False
    ]

).reset_index(
    drop=True
)


print()

print("=" * 60)

print(
    "STABLE FEATURE SET"
)

print("=" * 60)

print()

print(
    "Features selected >= 3 folds:",
    len(stable_features)
)


# ==========================================================
# ACCESSION LIST
# ==========================================================

accessions = (

    stable_features[
        "GenBank_Accession"
    ]

    .dropna()

    .astype(str)

    .str.strip()

    .unique()

)


print(
    "Unique GenBank accessions:",
    len(accessions)
)

print()


# ==========================================================
# MAP EACH ACCESSION
# ==========================================================

mapping_rows = []


for index, accession in enumerate(
    accessions,
    start=1
):

    print(
        f"[{index}/{len(accessions)}] "
        f"Searching NCBI Gene: "
        f"{accession}"
    )


    gene_ids = search_gene_by_accession(
        accession
    )


    time.sleep(
        REQUEST_DELAY
    )


    if gene_ids:

        print(
            "  Gene IDs found:",
            ", ".join(gene_ids)
        )


        summaries = get_gene_summary(
            gene_ids
        )


        time.sleep(
            REQUEST_DELAY
        )


        if summaries:

            for summary in summaries:

                mapping_rows.append({

                    "GenBank_Accession":
                        accession,

                    "NCBI_Gene_ID":
                        summary[
                            "NCBI_Gene_ID"
                        ],

                    "Gene_Symbol":
                        summary[
                            "Gene_Symbol"
                        ],

                    "Gene_Name":
                        summary[
                            "Gene_Name"
                        ],

                    "Organism":
                        summary[
                            "Organism"
                        ],

                    "Chromosome":
                        summary[
                            "Chromosome"
                        ],

                    "Map_Location":
                        summary[
                            "Map_Location"
                        ],

                    "Mapping_Status":
                        "NCBI_Gene_Mapped"

                })


        else:

            mapping_rows.append({

                "GenBank_Accession":
                    accession,

                "NCBI_Gene_ID":
                    "",

                "Gene_Symbol":
                    "",

                "Gene_Name":
                    "",

                "Organism":
                    "",

                "Chromosome":
                    "",

                "Map_Location":
                    "",

                "Mapping_Status":
                    "Gene_ID_found_summary_failed"

            })


    else:

        print(
            "  No direct NCBI Gene record found."
        )


        mapping_rows.append({

            "GenBank_Accession":
                accession,

            "NCBI_Gene_ID":
                "",

            "Gene_Symbol":
                "",

            "Gene_Name":
                "",

            "Organism":
                "",

            "Chromosome":
                "",

            "Map_Location":
                "",

            "Mapping_Status":
                "No_direct_Gene_mapping"

        })


    print()


# ==========================================================
# SAVE RAW NCBI RESULTS
# ==========================================================

mapping_df = pd.DataFrame(
    mapping_rows
)


mapping_df.to_csv(

    RAW_FILE,

    index=False

)


# ==========================================================
# MERGE WITH ML RESULTS
# ==========================================================

final_df = stable_features.merge(

    mapping_df,

    on="GenBank_Accession",

    how="left"

)


# ==========================================================
# HANDLE MULTIPLE GENE MAPPINGS
# ==========================================================

final_df = final_df.fillna("")


# ==========================================================
# SAVE FINAL ANNOTATION
# ==========================================================

final_df.to_csv(

    OUTPUT_FILE,

    index=False

)


# ==========================================================
# SUMMARY
# ==========================================================

mapped_accessions = mapping_df[

    mapping_df[
        "Mapping_Status"
    ]
    == "NCBI_Gene_Mapped"

][
    "GenBank_Accession"
].nunique()


unmapped_accessions = (

    len(accessions)
    - mapped_accessions

)


print()

print("=" * 60)

print(
    "NCBI GENE ANNOTATION SUMMARY"
)

print("=" * 60)

print()

print(
    "Stable features >=3/5 :",
    len(stable_features)
)

print(
    "Unique GenBank accessions:",
    len(accessions)
)

print(
    "Mapped to NCBI Gene:",
    mapped_accessions
)

print(
    "No direct Gene mapping:",
    unmapped_accessions
)


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print()

print("=" * 60)

print(
    "GENE ANNOTATION RESULTS"
)

print("=" * 60)

print()

display_columns = [

    "Probe",

    "Selection_Frequency",

    "Mean_MI_Score",

    "GenBank_Accession",

    "NCBI_Gene_ID",

    "Gene_Symbol",

    "Gene_Name",

    "Mapping_Status"

]


print(

    final_df[
        [
            column

            for column in display_columns

            if column in final_df.columns
        ]

    ].to_string(
        index=False
    )

)


# ==========================================================
# FINISHED
# ==========================================================

print()

print("=" * 60)

print(
    "FILES SAVED"
)

print("=" * 60)

print()

print(
    "Raw NCBI mapping:",
    RAW_FILE
)

print(
    "Final gene annotation:",
    OUTPUT_FILE
)

print()

print(
    "NCBI GenBank-to-Gene mapping completed."
)