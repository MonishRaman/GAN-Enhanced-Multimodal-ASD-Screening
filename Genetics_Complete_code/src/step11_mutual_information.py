import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# ==========================================================
# CONFIGURATION
# ==========================================================

TOP_FEATURES = 1000
RANDOM_STATE = 42

INPUT_FILE = r"data/processed/variance_filtered_dataset.csv"

OUTPUT_FILE = r"data/processed/top_features_dataset.csv"

SCORES_FILE = r"data/reports/mutual_information_scores.csv"

REPORT_FILE = r"data/reports/mutual_information_report.txt"

os.makedirs("data/reports", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Done.\n")

# ==========================================================
# Metadata
# ==========================================================

metadata_cols = [
    "GSM_ID",
    "Source",
    "Subtype",
    "Target"
]

metadata = df[metadata_cols]

X = df.drop(columns=metadata_cols)

y = df["Target"]

print("Computing Mutual Information scores...")

scores = mutual_info_classif(
    X,
    y,
    random_state=RANDOM_STATE
)

score_df = pd.DataFrame({
    "Feature": X.columns,
    "MI_Score": scores
})

score_df = score_df.sort_values(
    by="MI_Score",
    ascending=False
)

score_df.to_csv(
    SCORES_FILE,
    index=False
)

top_features = score_df.head(TOP_FEATURES)["Feature"].tolist()

X_selected = X[top_features]

final_df = pd.concat(
    [metadata.reset_index(drop=True),
     X_selected.reset_index(drop=True)],
    axis=1
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

with open(REPORT_FILE, "w") as f:

    f.write("="*60 + "\n")
    f.write("MUTUAL INFORMATION REPORT\n")
    f.write("="*60 + "\n\n")

    f.write(f"Original Features : {X.shape[1]}\n")
    f.write(f"Selected Features : {len(top_features)}\n")
    f.write(f"Random State      : {RANDOM_STATE}\n")

print("="*60)
print("MUTUAL INFORMATION SUMMARY")
print("="*60)

print(f"Original Features : {X.shape[1]}")
print(f"Selected Features : {len(top_features)}")

print("\nTop-feature dataset saved successfully.")