import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. Load training data
# ============================================================

X_train = pd.read_csv(
    "X_train_environmental.csv"
)

y_train = pd.read_csv(
    "y_train_environmental.csv"
).iloc[:, 0]


print("Training shape:", X_train.shape)
print("Target distribution:")
print(y_train.value_counts())


# ============================================================
# 2. Train Random Forest
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 3. Extract feature importance
# ============================================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


# ============================================================
# 4. Display top 30
# ============================================================

print("\n======================================")
print("TOP 30 MODEL-BASED FEATURES")
print("======================================")

print(
    importance.head(30).to_string(index=False)
)


# ============================================================
# 5. Save results
# ============================================================

importance.to_csv(
    "environmental_rf_feature_importance.csv",
    index=False
)


print("\nSaved:")
print("environmental_rf_feature_importance.csv")