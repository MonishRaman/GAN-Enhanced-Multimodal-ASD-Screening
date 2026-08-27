import pandas as pd

# ============================================================
# STEP 14
# Add ASD target to final 32D environmental embedding
# ============================================================

# Load 32D embeddings
train_emb = pd.read_csv(
    "environmental_embedding_train_32d.csv"
)

test_emb = pd.read_csv(
    "environmental_embedding_test_32d.csv"
)

# Load targets
y_train = pd.read_csv(
    "y_train_selected20.csv"
)

y_test = pd.read_csv(
    "y_test_selected20.csv"
)

print("Embedding shapes:")
print("Train:", train_emb.shape)
print("Test :", test_emb.shape)

print("\nTarget shapes:")
print("Train:", y_train.shape)
print("Test :", y_test.shape)


# ============================================================
# Validate row counts
# ============================================================

if len(train_emb) != len(y_train):
    raise ValueError("Training embedding and target row counts do not match.")

if len(test_emb) != len(y_test):
    raise ValueError("Testing embedding and target row counts do not match.")


# ============================================================
# Ensure target column is named k2q35a
# ============================================================

y_train.columns = ["k2q35a"]
y_test.columns = ["k2q35a"]


# ============================================================
# Combine embedding + target
# ============================================================

train_final = pd.concat(
    [
        train_emb.reset_index(drop=True),
        y_train.reset_index(drop=True)
    ],
    axis=1
)

test_final = pd.concat(
    [
        test_emb.reset_index(drop=True),
        y_test.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# Validation
# ============================================================

print("\nFinal shapes:")
print("Training:", train_final.shape)
print("Testing :", test_final.shape)

print("\nColumns:")
print(train_final.columns.tolist())

print("\nTraining target distribution:")
print(train_final["k2q35a"].value_counts())

print("\nTesting target distribution:")
print(test_final["k2q35a"].value_counts())


# ============================================================
# Save
# ============================================================

train_final.to_csv(
    "environmental_embedding_train_32d_with_target.csv",
    index=False
)

test_final.to_csv(
    "environmental_embedding_test_32d_with_target.csv",
    index=False
)


print("\nSaved:")
print("environmental_embedding_train_32d_with_target.csv")
print("environmental_embedding_test_32d_with_target.csv")

print("\nFinal format: 32 embedding features + 1 target = 33 columns")