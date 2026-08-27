import pandas as pd
import numpy as np

# -----------------------------------------
# 1. Load environmental dataset
# -----------------------------------------

df = pd.read_csv("environmental_dataset.csv")

print("Original shape:", df.shape)


# -----------------------------------------
# 2. Remove records with missing ASD target
# -----------------------------------------

df = df[df["k2q35a"].notna()].copy()

print("\nAfter removing missing ASD target:")
print(df.shape)


# -----------------------------------------
# 3. Convert ASD target
# -----------------------------------------

# NSCH coding:
# 1 = Yes / ASD
# 2 = No / Non-ASD

df["k2q35a"] = df["k2q35a"].map({
    1: 1,
    2: 0
})


# -----------------------------------------
# 4. Check target
# -----------------------------------------

print("\nTarget distribution:")
print(df["k2q35a"].value_counts())

print("\nTarget percentages:")
print(
    df["k2q35a"]
    .value_counts(normalize=True)
    .mul(100)
)


# -----------------------------------------
# 5. Inspect missing values
# -----------------------------------------

missing = df.isna().sum()

missing_percent = (
    df.isna().mean() * 100
)

missing_table = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percent": missing_percent
})

print("\nMissing values:")
print(
    missing_table[
        missing_table["Missing_Count"] > 0
    ].sort_values(
        "Missing_Percent",
        ascending=False
    )
)


# -----------------------------------------
# 6. Save intermediate dataset
# -----------------------------------------

df.to_csv(
    "environmental_step2_target_clean.csv",
    index=False
)

print("\nSaved:")
print("environmental_step2_target_clean.csv")