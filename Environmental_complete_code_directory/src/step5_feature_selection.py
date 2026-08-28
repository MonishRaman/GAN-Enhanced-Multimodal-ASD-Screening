import pandas as pd

from sklearn.feature_selection import mutual_info_classif


# ============================================================
# 1. Load processed data
# ============================================================

X_train = pd.read_csv(
    "X_train_environmental.csv"
)

y_train = pd.read_csv(
    "y_train_environmental.csv"
).iloc[:, 0]


print("Training shape:", X_train.shape)
print("Target shape:", y_train.shape)


# ============================================================
# 2. Calculate Mutual Information
# ============================================================

print("\nCalculating mutual information...")

mi_scores = mutual_info_classif(
    X_train,
    y_train,
    random_state=42
)


# ============================================================
# 3. Create named feature ranking
# ============================================================

feature_scores = pd.DataFrame({
    "Feature": X_train.columns,
    "Mutual_Information": mi_scores
})


feature_scores = feature_scores.sort_values(
    by="Mutual_Information",
    ascending=False
)


# ============================================================
# 4. Display top 20
# ============================================================

print("\nTop 20 encoded features:")

print(
    feature_scores.head(20).to_string(index=False)
)


# ============================================================
# 5. Save ranking
# ============================================================

feature_scores.to_csv(
    "environmental_feature_importance.csv",
    index=False
)

print("\nSaved:")
print("environmental_feature_importance.csv")