import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Load selected environmental dataset
# ============================================================

df = pd.read_csv(
    "environmental_selected_20.csv"
)

print("Dataset shape:", df.shape)


# ============================================================
# 2. Separate X and y
# ============================================================

X = df.drop(columns=["k2q35a"])
y = df["k2q35a"]


# ============================================================
# 3. Identify categorical variables
# ============================================================

categorical_features = [
    "ace8",
    "ace1",
    "foodsit",
    "k11q61",
    "family_r",
    "physactiv",
    "ace10",
    "house_gen",
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


# ============================================================
# 4. Numeric variables
# ============================================================

numeric_features = [
    "fpl_i1",
    "hoursleep"
]


# ============================================================
# 5. Convert categorical columns to object
# ============================================================

for col in categorical_features:
    X[col] = X[col].astype("object")


# ============================================================
# 6. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. Numeric preprocessing
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ============================================================
# 8. Categorical preprocessing
# ============================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )
    )
])


# ============================================================
# 9. Combine preprocessing
# ============================================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 10. Fit ONLY on training data
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# 11. Get feature names
# ============================================================

try:
    feature_names = preprocessor.get_feature_names_out()

except AttributeError:

    numeric_names = numeric_features

    encoder = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )

    categorical_names = encoder.get_feature_names_out(
        categorical_features
    )

    feature_names = list(numeric_names) + list(
        categorical_names
    )


# ============================================================
# 12. Convert to DataFrames
# ============================================================

X_train_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


# ============================================================
# 13. Save
# ============================================================

X_train_df.to_csv(
    "X_train_selected20.csv",
    index=False
)

X_test_df.to_csv(
    "X_test_selected20.csv",
    index=False
)

y_train.to_csv(
    "y_train_selected20.csv",
    index=False
)

y_test.to_csv(
    "y_test_selected20.csv",
    index=False
)


print("\nProcessed training shape:")
print(X_train_df.shape)

print("\nProcessed testing shape:")
print(X_test_df.shape)

print("\nSelected-feature preprocessing completed.")