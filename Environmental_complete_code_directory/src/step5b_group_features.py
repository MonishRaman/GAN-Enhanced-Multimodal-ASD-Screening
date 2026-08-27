import pandas as pd


# ============================================================
# 1. Load MI results
# ============================================================

mi = pd.read_csv(
    "environmental_feature_importance.csv"
)

print("Encoded features:", len(mi))


# ============================================================
# 2. Original environmental variables
# ============================================================

original_features = [
    "fpl_i1",
    "hoursleep",
    "higrade",
    "family_r",
    "house_gen",
    "metro_yn",
    "mpc_yn",
    "currins",
    "insgap",
    "foodsit",
    "k11q61",
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
    "physactiv",
    "screentime"
]


# ============================================================
# 3. Group encoded features by original variable
# ============================================================

results = []

for original in original_features:

    # Find encoded columns belonging to this original variable
    matching = mi[
        (mi["Feature"] == original) |
        (mi["Feature"].str.startswith(original + "_"))
    ]

    if len(matching) == 0:
        continue

    results.append({
        "Original_Feature": original,
        "Number_of_Encoded_Features": len(matching),
        "Maximum_MI": matching["Mutual_Information"].max(),
        "Mean_MI": matching["Mutual_Information"].mean(),
        "Total_MI": matching["Mutual_Information"].sum()
    })


# ============================================================
# 4. Create ranking
# ============================================================

grouped = pd.DataFrame(results)

grouped = grouped.sort_values(
    by="Maximum_MI",
    ascending=False
)


# ============================================================
# 5. Display
# ============================================================

print("\n==========================================")
print("ORIGINAL ENVIRONMENTAL FEATURE RANKING")
print("==========================================")

print(
    grouped.to_string(index=False)
)


# ============================================================
# 6. Save
# ============================================================

grouped.to_csv(
    "environmental_grouped_feature_importance.csv",
    index=False
)

print("\nSaved:")
print("environmental_grouped_feature_importance.csv")