import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv(
    "environmental_step2_target_clean.csv"
)

print("Initial shape:", df.shape)


# ============================================================
# 2. Remove extremely sparse feature
# ============================================================

df = df.drop(columns=["k9q41"])

print("After removing k9q41:")
print(df.shape)


# ============================================================
# 3. Separate X and y
# ============================================================

X = df.drop(columns=["k2q35a"])
y = df["k2q35a"]


# ============================================================
# 4. Categorical variables
# ============================================================

categorical_features = [
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
# 5. Numeric variables
# ============================================================

numeric_features = [
    "fpl_i1",
    "hoursleep"
]


# ============================================================
# 6. Force categorical variables to object type
# ============================================================

for col in categorical_features:
    X[col] = X[col].astype("object")


# ============================================================
# 7. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. Numeric preprocessing
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


# ============================================================
# 9. Categorical preprocessing
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
# 10. Combine preprocessing
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
# 11. Fit only on training data
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)
# Get the names of the 83 processed features
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    # For older scikit-learn versions
    numeric_names = numeric_features

    encoder = preprocessor.named_transformers_["categorical"] \
        .named_steps["encoder"]

    categorical_names = encoder.get_feature_names_out(
        categorical_features
    )

    feature_names = list(numeric_names) + list(categorical_names)

print("\nNumber of feature names:", len(feature_names))

print("\nFirst 20 feature names:")
for i, name in enumerate(feature_names[:20]):
    print(i, name)


# ============================================================
# 12. Results
# ============================================================

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)


# ============================================================
# 13. Save processed datasets
# ============================================================

X_train_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

X_train_df.to_csv(
    "X_train_environmental.csv",
    index=False
)

X_test_df.to_csv(
    "X_test_environmental.csv",
    index=False
)

pd.DataFrame(
    X_test_processed
).to_csv(
    "X_test_environmental.csv",
    index=False
)

pd.DataFrame(
    y_train
).to_csv(
    "y_train_environmental.csv",
    index=False
)

pd.DataFrame(
    y_test
).to_csv(
    "y_test_environmental.csv",
    index=False
)

print("\nPreprocessing completed successfully!")