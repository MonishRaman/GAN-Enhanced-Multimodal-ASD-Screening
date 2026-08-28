import pandas as pd


# ============================================================
# 1. Load Random Forest importance
# ============================================================

rf = pd.read_csv(
    "environmental_rf_feature_importance.csv"
)

print("Encoded features:", len(rf))


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
# 3. Aggregate encoded features
# ============================================================

results = []

for original in original_features:

    matching = rf[
        (rf["Feature"] == original) |
        (rf["Feature"].str.startswith(original + "_"))
    ]

    if len(matching) == 0:
        continue

    results.append({
        "Original_Feature": original,
        "Number_of_Encoded_Features": len(matching),
        "Maximum_RF_Importance": matching["Importance"].max(),
        "Mean_RF_Importance": matching["Importance"].mean(),
        "Total_RF_Importance": matching["Importance"].sum()
    })


# ============================================================
# 4. Ranking
# ============================================================

grouped = pd.DataFrame(results)

grouped = grouped.sort_values(
    "Total_RF_Importance",
    ascending=False
)


# ============================================================
# 5. Display
# ============================================================

print("\n==============================================")
print("ORIGINAL ENVIRONMENTAL RF FEATURE RANKING")
print("==============================================")

print(
    grouped.to_string(index=False)
)


# ============================================================
# 6. Save
# ============================================================

grouped.to_csv(
    "environmental_grouped_rf_importance.csv",
    index=False
)

print("\nSaved:")
print("environmental_grouped_rf_importance.csv")