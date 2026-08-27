import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score


# ============================================================
# 1. Load embeddings
# ============================================================

train_embedding = pd.read_csv(
    "environmental_embedding_train.csv"
)

test_embedding = pd.read_csv(
    "environmental_embedding_test.csv"
)

y_train = pd.read_csv(
    "y_train_selected20.csv"
).iloc[:, 0]

y_test = pd.read_csv(
    "y_test_selected20.csv"
).iloc[:, 0]


# ============================================================
# 2. Basic dimensions
# ============================================================

print("==========================================")
print("ENVIRONMENTAL EMBEDDING VALIDATION")
print("==========================================")

print("\nTrain embedding shape:")
print(train_embedding.shape)

print("\nTest embedding shape:")
print(test_embedding.shape)


# ============================================================
# 3. Check NaN
# ============================================================

print("\nNaN values:")
print(
    "Train:",
    train_embedding.isna().sum().sum()
)

print(
    "Test :",
    test_embedding.isna().sum().sum()
)


# ============================================================
# 4. Check infinity
# ============================================================

print("\nInfinite values:")

print(
    "Train:",
    np.isinf(train_embedding.values).sum()
)

print(
    "Test :",
    np.isinf(test_embedding.values).sum()
)


# ============================================================
# 5. Check constant dimensions
# ============================================================

train_std = train_embedding.std()

constant_dimensions = (
    train_std == 0
).sum()

print("\nConstant dimensions:")
print(constant_dimensions)


# ============================================================
# 6. Embedding statistics
# ============================================================

print("\nEmbedding statistics:")

print(
    train_embedding.describe().loc[
        ["mean", "std", "min", "max"]
    ].iloc[:, :10]
)


# ============================================================
# 7. Target relationship
# ============================================================

# Calculate simple univariate AUC for every
# embedding dimension.

auc_results = []

for column in train_embedding.columns:

    values = train_embedding[column]

    try:

        auc = roc_auc_score(
            y_train,
            values
        )

        # AUC below 0.5 is equivalent to
        # reversing the direction.

        auc = max(
            auc,
            1 - auc
        )

        auc_results.append({
            "Dimension": column,
            "AUC": auc
        })

    except Exception:

        auc_results.append({
            "Dimension": column,
            "AUC": np.nan
        })


auc_df = pd.DataFrame(
    auc_results
)

auc_df = auc_df.sort_values(
    "AUC",
    ascending=False
)


print("\nTop embedding dimensions:")
print(
    auc_df.head(15).to_string(
        index=False
    )
)


# ============================================================
# 8. Save diagnostic results
# ============================================================

auc_df.to_csv(
    "environmental_embedding_dimension_auc.csv",
    index=False
)


print("\nValidation completed.")

print(
    "\nSaved:"
    "\nenvironmental_embedding_dimension_auc.csv"
)