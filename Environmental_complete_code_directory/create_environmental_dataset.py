import pandas as pd

# Read the original NSCH dataset
df = pd.read_csv(
    "nsch_2023e_topical.csv",
    encoding="latin1"
)

# Environmental variables + ASD target
columns = [
    "fpl_i1",
    "higrade",
    "family_r",
    "house_gen",
    "metro_yn",
    "mpc_yn",
    "currins",
    "insgap",
    "foodsit",
    "k11q61",
    "k9q41",
    "k10q40_r",
    "k10q30",
    "k10q23",
    "k10q22",
    "k10q20",
    "k10q11",
    "k10q12",
    "k10q13",
    "k10q14",
    "ace1",
    "ace3",
    "ace4",
    "ace5",
    "ace6",
    "ace7",
    "ace8",
    "ace9",
    "ace10",
    "hoursleep",
    "physactiv",
    "screentime",
    "k2q35a"
]

# Check that all columns exist
missing = [col for col in columns if col not in df.columns]

if missing:
    print("These columns are missing:")
    print(missing)
else:
    print("All required columns found.")

    # Create environmental dataset
    environmental = df[columns].copy()

    # Save as genuine UTF-8 CSV
    environmental.to_csv(
        "environmental_dataset.csv",
        index=False,
        encoding="utf-8"
    )

    print("Environmental dataset created successfully!")
    print("Shape:", environmental.shape)