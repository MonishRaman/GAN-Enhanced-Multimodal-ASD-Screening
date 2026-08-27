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

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RAW_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "ncbi_sequence_gene_mapping_raw.csv"
)

FINAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stable_3plus_features_gene_annotation_v2.csv"
)

EUTILS_BASE = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/"
)

REQUEST_DELAY = 0.4


# ==========================================================
# NCBI REQUEST
# ==========================================================

def ncbi_request(endpoint, params):

    query = urllib.parse.urlencode(params)

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
# SEARCH NUCLEOTIDE DATABASE
# ==========================================================

def search_nucleotide(accession):

    try:

        data = ncbi_request(

            "esearch.fcgi",

            {
                "db": "nuccore",

                "term":
                    f'"{accession}"[ACCN]',

                "retmode": "xml",

                "retmax": 20
            }

        )

        root = ET.fromstring(data)

        ids = [

            element.text

            for element in root.findall(".//Id")

            if element.text

        ]

        return ids

    except Exception as e:

        print(
            f"  Nucleotide search error: {e}"
        )

        return []


# ==========================================================
# FETCH NUCLEOTIDE RECORD
# ==========================================================

def fetch_nucleotide(uid):

    try:

        data = ncbi_request(

            "efetch.fcgi",

            {
                "db": "nuccore",

                "id": uid,

                "rettype": "gb",

                "retmode": "text"
            }

        )

        return data.decode(
            "utf-8",
            errors="replace"
        )

    except Exception as e:

        print(
            f"  Nucleotide fetch error: {e}"
        )

        return ""


# ==========================================================
# SEARCH GENE USING ACCESSION
# ==========================================================

def search_gene_all_fields(accession):

    try:

        data = ncbi_request(

            "esearch.fcgi",

            {
                "db": "gene",

                "term":
                    f'"{accession}"',

                "retmode": "xml",

                "retmax": 20
            }

        )

        root = ET.fromstring(data)

        ids = [

            element.text

            for element in root.findall(".//Id")

            if element.text

        ]

        return ids

    except Exception:

        return []


# ==========================================================
# LINK NUCLEOTIDE → GENE
# ==========================================================

def link_nucleotide_to_gene(uid):

    try:

        data = ncbi_request(

            "elink.fcgi",

            {
                "dbfrom": "nuccore",

                "db": "gene",

                "id": uid,

                "retmode": "xml"
            }

        )

        root = ET.fromstring(data)

        ids = []

        for element in root.findall(
            ".//LinkSetDb/Link/Id"
        ):

            if element.text:

                ids.append(
                    element.text
                )

        return list(
            dict.fromkeys(ids)
        )

    except Exception as e:

        print(
            f"  ELink error: {e}"
        )

        return []


# ==========================================================
# GET GENE SUMMARY
# ==========================================================

def get_gene_summary(gene_ids):

    if not gene_ids:

        return []

    try:

        data = ncbi_request(

            "esummary.fcgi",

            {
                "db": "gene",

                "id":
                    ",".join(gene_ids),

                "retmode": "xml"
            }

        )

        root = ET.fromstring(data)

        results = []

        for document in root.findall(
            ".//DocumentSummary"
        ):

            gene_id = document.attrib.get(
                "uid",
                ""
            )

            def text(tag):

                element = document.find(
                    tag
                )

                if element is not None:

                    return (
                        element.text
                        or ""
                    )

                return ""

            results.append({

                "NCBI_Gene_ID":
                    gene_id,

                "Gene_Symbol":
                    text("Name"),

                "Gene_Name":
                    text("Description"),

                "Organism":
                    text("Orgname"),

                "Chromosome":
                    text("Chromosome"),

                "Map_Location":
                    text("MapLocation")

            })

        return results

    except Exception as e:

        print(
            f"  Gene summary error: {e}"
        )

        return []


# ==========================================================
# LOAD DATA
# ==========================================================

print(
    "Loading stable features..."
)

df = pd.read_csv(
    INPUT_FILE,
    dtype=str
)

print("Done.")


# ==========================================================
# SELECT >=3/5 FEATURES
# ==========================================================

df[
    "Selection_Frequency"
] = pd.to_numeric(

    df[
        "Selection_Frequency"
    ],

    errors="coerce"

)


stable = df[
    df[
        "Selection_Frequency"
    ] >= 3
].copy()


stable = stable.sort_values(

    [
        "Selection_Frequency",
        "Mean_MI_Score"
    ],

    ascending=[
        False,
        False
    ]

)


# ==========================================================
# UNIQUE ACCESSIONS
# ==========================================================

accessions = (

    stable[
        "GenBank_Accession"
    ]

    .dropna()

    .astype(str)

    .str.strip()

    .unique()

)


print()

print("=" * 60)

print(
    "GENBANK → NUCLEOTIDE → GENE MAPPING"
)

print("=" * 60)

print()

print(
    "Stable features:",
    len(stable)
)

print(
    "Unique accessions:",
    len(accessions)
)

print()


# ==========================================================
# PROCESS ACCESSIONS
# ==========================================================

mapping_rows = []


for number, accession in enumerate(
    accessions,
    start=1
):

    print(
        f"[{number}/{len(accessions)}] "
        f"{accession}"
    )


    # ------------------------------------------------------
    # STEP A: SEARCH NUCLEOTIDE
    # ------------------------------------------------------

    nucleotide_ids = (
        search_nucleotide(
            accession
        )
    )


    time.sleep(
        REQUEST_DELAY
    )


    print(
        "  Nucleotide IDs:",
        nucleotide_ids
    )


    # ------------------------------------------------------
    # STEP B: TRY GENERAL GENE SEARCH
    # ------------------------------------------------------

    gene_ids = (
        search_gene_all_fields(
            accession
        )
    )


    time.sleep(
        REQUEST_DELAY
    )


    # ------------------------------------------------------
    # STEP C: LINK NUCLEOTIDE → GENE
    # ------------------------------------------------------

    linked_gene_ids = []


    for uid in nucleotide_ids:

        ids = link_nucleotide_to_gene(
            uid
        )

        linked_gene_ids.extend(
            ids
        )

        time.sleep(
            REQUEST_DELAY
        )


    # Remove duplicates

    all_gene_ids = list(
        dict.fromkeys(

            gene_ids
            + linked_gene_ids

        )
    )


    print(
        "  Gene IDs:",
        all_gene_ids
    )


    # ------------------------------------------------------
    # STEP D: GET GENE DETAILS
    # ------------------------------------------------------

    if all_gene_ids:

        summaries = get_gene_summary(
            all_gene_ids
        )

        time.sleep(
            REQUEST_DELAY
        )


        if summaries:

            for summary in summaries:

                mapping_rows.append({

                    "GenBank_Accession":
                        accession,

                    "Nucleotide_UIDs":
                        ";".join(
                            nucleotide_ids
                        ),

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
                        "Mapped_via_NCBI"

                })

        else:

            mapping_rows.append({

                "GenBank_Accession":
                    accession,

                "Nucleotide_UIDs":
                    ";".join(
                        nucleotide_ids
                    ),

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
                    "Gene_ID_found_but_no_summary"

            })

    else:

        mapping_rows.append({

            "GenBank_Accession":
                accession,

            "Nucleotide_UIDs":
                ";".join(
                    nucleotide_ids
                ),

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
                "No_NCBI_Gene_mapping"

        })


# ==========================================================
# SAVE RAW RESULTS
# ==========================================================

mapping = pd.DataFrame(
    mapping_rows
)


mapping.to_csv(
    RAW_OUTPUT,
    index=False
)


# ==========================================================
# MERGE WITH ML FEATURES
# ==========================================================

final = stable.merge(

    mapping,

    on="GenBank_Accession",

    how="left"

)


# ==========================================================
# SAVE FINAL FILE
# ==========================================================

final.to_csv(

    FINAL_OUTPUT,

    index=False

)


# ==========================================================
# SUMMARY
# ==========================================================

mapped = mapping[
    mapping[
        "Mapping_Status"
    ] == "Mapped_via_NCBI"
][
    "GenBank_Accession"
].nunique()


print()

print("=" * 60)

print(
    "FINAL NCBI ANNOTATION SUMMARY"
)

print("=" * 60)

print()

print(
    "Stable features >=3/5 :",
    len(stable)
)

print(
    "Unique GenBank accessions:",
    len(accessions)
)

print(
    "Mapped through NCBI:",
    mapped
)

print(
    "Unresolved:",
    len(accessions) - mapped
)


# ==========================================================
# DISPLAY
# ==========================================================

print()

print("=" * 60)

print(
    "ANNOTATED FEATURES"
)

print("=" * 60)

print()

columns = [

    "Probe",
    "Selection_Frequency",
    "Mean_MI_Score",
    "GenBank_Accession",
    "NCBI_Gene_ID",
    "Gene_Symbol",
    "Gene_Name",
    "Mapping_Status"

]


available = [

    c

    for c in columns

    if c in final.columns

]


print(

    final[
        available
    ].to_string(
        index=False
    )

)


print()

print("=" * 60)

print(
    "FILES SAVED"
)

print("=" * 60)

print()

print(
    "Raw mapping:",
    RAW_OUTPUT
)

print(
    "Final annotation:",
    FINAL_OUTPUT
)

print()

print(
    "Step 19B completed."
)