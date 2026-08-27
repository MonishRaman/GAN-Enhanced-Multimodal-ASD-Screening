import os
import pandas as pd
import xml.etree.ElementTree as ET

# ==========================================================
# Paths
# ==========================================================

XML_FILE = r"data/raw/GSE15402_family.xml"

OUTPUT_LABELS = r"data/processed/labels.csv"

REPORT_FILE = r"data/reports/label_report.txt"

os.makedirs("data/reports", exist_ok=True)

print("Reading XML...")

tree = ET.parse(XML_FILE)

root = tree.getroot()

print("Done.\n")

ns = {

    "geo":"http://www.ncbi.nlm.nih.gov/geo/info/MINiML"

}

labels=[]

# ==========================================================
# Read Samples
# ==========================================================

for sample in root.findall("geo:Sample",ns):

    gsm = sample.attrib.get("iid")

    channel = sample.find("geo:Channel[@position='1']",ns)

    source=""

    if channel is not None:

        source_node = channel.find("geo:Source",ns)

        if source_node is not None:

            source = source_node.text.strip().lower()

    target=None

    subtype="Unknown"

    if "nonautistic control" in source:

        target=0

        subtype="Control"

    elif "autistic" in source:

        target=1

        if "mild" in source:

            subtype="Mild"

        elif "savant" in source:

            subtype="Savant"

        elif "severe" in source:

            subtype="Severe Language Impaired"

        else:

            subtype="Autism"

    labels.append([

        gsm,

        source,

        subtype,

        target

    ])

labels_df=pd.DataFrame(

    labels,

    columns=[

        "GSM_ID",

        "Source",

        "Subtype",

        "Target"

    ]

)

labels_df.to_csv(

    OUTPUT_LABELS,

    index=False

)

autism=(labels_df["Target"]==1).sum()

control=(labels_df["Target"]==0).sum()

unknown=labels_df["Target"].isna().sum()

with open(REPORT_FILE,"w") as f:

    f.write("="*60+"\n")

    f.write("LABEL EXTRACTION REPORT\n")

    f.write("="*60+"\n\n")

    f.write(f"Total Samples : {len(labels_df)}\n")

    f.write(f"Autism : {autism}\n")

    f.write(f"Control : {control}\n")

    f.write(f"Unknown : {unknown}\n\n")

    f.write(labels_df["Subtype"].value_counts().to_string())

print("="*60)

print("LABEL EXTRACTION SUMMARY")

print("="*60)

print(f"Total Samples : {len(labels_df)}")

print(f"Autism : {autism}")

print(f"Control : {control}")

print(f"Unknown : {unknown}")

print()

print(labels_df["Subtype"].value_counts())

print("\nlabels.csv saved successfully.")