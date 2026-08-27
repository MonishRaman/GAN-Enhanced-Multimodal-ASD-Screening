import pandas as pd

df = pd.read_csv("environmental_step2_target_clean.csv")

features = [
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
    "screentime"
]

for col in features:

    print("\n" + "=" * 60)
    print(col)

    print("Data type:", df[col].dtype)
    print("Missing:", df[col].isna().sum())

    print("\nValue counts:")
    print(
        df[col]
        .value_counts(dropna=False)
        .sort_index()
        .head(30)
    )