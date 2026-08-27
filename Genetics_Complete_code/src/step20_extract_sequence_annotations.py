import os
import time
import urllib.parse
import urllib.request
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
    "genbank_sequence_annotations.csv"
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

REQUEST_DELAY = 0.5


# ==========================================================
# NCBI REQUEST
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
# SEARCH NUCLEOTIDE
# ==========================================================

def search_nucleotide(
    accession
):

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

    import xml.etree.ElementTree as ET

    root = ET.fromstring(
        data
    )

    return [

        element.text

        for element in root.findall(
            ".//Id"
        )

        if element.text

    ]


# ==========================================================
# FETCH GENBANK RECORD
# ==========================================================

def fetch_genbank(
    uid
):

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


# ==========================================================
# EXTRACT GENBANK FIELDS
# ==========================================================

def parse_genbank(
    text
):

    result = {

        "Definition": "",

        "Accession": "",

        "Version": "",

        "GI": "",

        "Source": "",

        "Organism": "",

        "DB_XREFS": "",

        "UniGene": "",

        "Gene_Xrefs": "",

        "References": ""

    }


    lines = text.splitlines()


    # ------------------------------------------------------
    # DEFINITION
    # ------------------------------------------------------

    for i, line in enumerate(lines):

        if line.startswith(
            "DEFINITION"
        ):

            value = (
                line[12:].strip()
            )

            j = i + 1

            while (
                j < len(lines)
                and lines[j].startswith(
                    "            "
                )
            ):

                value += " " + (
                    lines[j].strip()
                )

                j += 1

            result[
                "Definition"
            ] = value

            break


    # ------------------------------------------------------
    # ACCESSION
    # ------------------------------------------------------

    for line in lines:

        if line.startswith(
            "ACCESSION"
        ):

            result[
                "Accession"
            ] = line[
                12:
            ].strip()

            break


    # ------------------------------------------------------
    # VERSION
    # ------------------------------------------------------

    for line in lines:

        if line.startswith(
            "VERSION"
        ):

            result[
                "Version"
            ] = line[
                12:
            ].strip()

            break


    # ------------------------------------------------------
    # SOURCE
    # ------------------------------------------------------

    for line in lines:

        if line.startswith(
            "SOURCE"
        ):

            result[
                "Source"
            ] = line[
                12:
            ].strip()

            break


    # ------------------------------------------------------
    # ORGANISM
    # ------------------------------------------------------

    for i, line in enumerate(lines):

        if line.strip() == "ORGANISM":

            if (
                i + 1
                < len(lines)
            ):

                result[
                    "Organism"
                ] = lines[
                    i + 1
                ].strip()

            break


    # ------------------------------------------------------
    # DB_XREF
    # ------------------------------------------------------

    db_xrefs = []

    for line in lines:

        if (
            "db_xref="
            in line
        ):

            value = line.split(
                "db_xref="
            )[1]

            value = value.strip(
                '"'
            )

            db_xrefs.append(
                value
            )


    result[
        "DB_XREFS"
    ] = ";".join(
        sorted(
            set(
                db_xrefs
            )
        )
    )


    # ------------------------------------------------------
    # UNIGENE
    # ------------------------------------------------------

    unigene = []

    for line in lines:

        if (
            "UniGene"
            in line
        ):

            unigene.append(
                line.strip()
            )


    result[
        "UniGene"
    ] = ";".join(
        sorted(
            set(
                unigene
            )
        )
    )


    # ------------------------------------------------------
    # GENE REFERENCES
    # ------------------------------------------------------

    gene_refs = []

    for line in lines:

        lower = line.lower()

        if (
            "gene" in lower
            or "locus" in lower
        ):

            gene_refs.append(
                line.strip()
            )


    result[
        "Gene_Xrefs"
    ] = ";".join(
        sorted(
            set(
                gene_refs
            )
        )
    )


    # ------------------------------------------------------
    # REFERENCES
    # ------------------------------------------------------

    references = []

    for line in lines:

        if (
            line.startswith(
                "REFERENCE"
            )
        ):

            references.append(
                line.strip()
            )


    result[
        "References"
    ] = ";".join(
        references
    )


    return result


# ==========================================================
# LOAD STABLE FEATURES
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
# SELECT >=3/5
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
    "GENBANK SEQUENCE ANNOTATION"
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
# PROCESS
# ==========================================================

rows = []


for number, accession in enumerate(
    accessions,
    start=1
):

    print(
        f"[{number}/{len(accessions)}] "
        f"{accession}"
    )


    try:

        uids = search_nucleotide(
            accession
        )

        time.sleep(
            REQUEST_DELAY
        )


        if not uids:

            print(
                "  No nucleotide record."
            )

            rows.append({

                "GenBank_Accession":
                    accession,

                "Nucleotide_UID": "",

                "Mapping_Status":
                    "No_nucleotide_record"

            })

            continue


        for uid in uids:

            print(
                "  Nucleotide UID:",
                uid
            )


            record = fetch_genbank(
                uid
            )

            time.sleep(
                REQUEST_DELAY
            )


            annotation = parse_genbank(
                record
            )


            rows.append({

                "GenBank_Accession":
                    accession,

                "Nucleotide_UID":
                    uid,

                **annotation,

                "Mapping_Status":
                    "Nucleotide_record_found"

            })


            # Print the most useful field immediately

            print(
                "  Definition:",
                annotation[
                    "Definition"
                ]
            )

            print(
                "  DB_XREFS:",
                annotation[
                    "DB_XREFS"
                ]
            )

            print(
                "  UniGene:",
                annotation[
                    "UniGene"
                ]
            )


    except Exception as e:

        print(
            "  ERROR:",
            e
        )


        rows.append({

            "GenBank_Accession":
                accession,

            "Nucleotide_UID": "",

            "Mapping_Status":
                "Error"

        })


    print()


# ==========================================================
# SAVE
# ==========================================================

annotation_df = pd.DataFrame(
    rows
)


annotation_df.to_csv(

    OUTPUT_FILE,

    index=False

)


# ==========================================================
# SUMMARY
# ==========================================================

print()

print("=" * 60)

print(
    "SEQUENCE ANNOTATION SUMMARY"
)

print("=" * 60)

print()

print(
    "Accessions requested:",
    len(accessions)
)

print(
    "Records retrieved:",
    len(annotation_df)
)

print(
    "Unique accessions retrieved:",
    annotation_df[
        "GenBank_Accession"
    ].nunique()
)

print()

print(
    "Output:",
    OUTPUT_FILE
)

print()

print(
    "Step 20 completed."
)