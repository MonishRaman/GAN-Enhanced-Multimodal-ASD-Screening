import pandas as pd

# ============================================================
# Selected environmental variables
# ============================================================

selected_features = [
    "ace8",
    "ace1",
    "foodsit",
    "k11q61",
    "family_r",
    "physactiv",
    "fpl_i1",
    "ace10",
    "house_gen",
    "hoursleep",
    "ace6",
    "k10q23",
    "k10q40_r",
    "ace4",
    "k10q12",
    "k10q11",
    "screentime",
    "ace9",
    "ace3",
    "ace5"
]

target = "k2q35a"


# ============================================================
# Load Step 2 dataset
# ============================================================

df = pd.read_csv(
    "environmental_step2_target_clean.csv"
)


# ============================================================
# Create selected dataset
# ============================================================

selected_df = df[
    selected_features + [target]
].copy()


# ============================================================
# Save
# ============================================================

selected_df.to_csv(
    "environmental_selected_20.csv",
    index=False
)


print("Selected environmental dataset created.")
print("Shape:", selected_df.shape)

print("\nFeatures:")
for feature in selected_features:
    print(feature)

print("\nTarget:")
print(target)