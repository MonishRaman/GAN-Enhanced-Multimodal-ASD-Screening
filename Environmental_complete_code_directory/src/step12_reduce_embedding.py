import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib
import os

# ============================================================
# STEP 12: REDUCE ENVIRONMENTAL EMBEDDING
# 128 dimensions -> 32 dimensions using PCA
# ============================================================

TRAIN_FILE = "environmental_embedding_train.csv"
TEST_FILE = "environmental_embedding_test.csv"

OUTPUT_TRAIN = "environmental_embedding_train_32d.csv"
OUTPUT_TEST = "environmental_embedding_test_32d.csv"
PCA_MODEL = "environmental_pca_128_to_32.joblib"

# ------------------------------------------------------------
# 1. Load embeddings
# ------------------------------------------------------------

print("Loading environmental embeddings...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Original training shape:", train_df.shape)
print("Original testing shape :", test_df.shape)

# Convert to numpy
X_train = train_df.values.astype(np.float32)
X_test = test_df.values.astype(np.float32)

# ------------------------------------------------------------
# 2. Check for invalid values
# ------------------------------------------------------------

print("\nChecking embeddings...")

print("Training NaN:", np.isnan(X_train).sum())
print("Testing NaN :", np.isnan(X_test).sum())

print("Training Inf:", np.isinf(X_train).sum())
print("Testing Inf :", np.isinf(X_test).sum())

if np.isnan(X_train).any() or np.isnan(X_test).any():
    raise ValueError("NaN values found in embeddings.")

if np.isinf(X_train).any() or np.isinf(X_test).any():
    raise ValueError("Infinite values found in embeddings.")

# ------------------------------------------------------------
# 3. Fit PCA ONLY on training data
# ------------------------------------------------------------

print("\nFitting PCA...")
print("Input dimensions: 128")
print("Output dimensions: 32")

pca = PCA(
    n_components=32,
    random_state=42
)

X_train_32 = pca.fit_transform(X_train)

# ------------------------------------------------------------
# 4. Transform test data using SAME PCA
# ------------------------------------------------------------

X_test_32 = pca.transform(X_test)

# ------------------------------------------------------------
# 5. Calculate explained variance
# ------------------------------------------------------------

explained_variance = pca.explained_variance_ratio_

total_variance = explained_variance.sum()

print("\nPCA RESULTS")
print("=" * 50)

print("Original dimensions :", X_train.shape[1])
print("Reduced dimensions  :", X_train_32.shape[1])

print(
    "Explained variance by 32 components:",
    round(total_variance, 6)
)

print(
    "Explained variance percentage:",
    round(total_variance * 100, 2),
    "%"
)

# ------------------------------------------------------------
# 6. Save 32D embeddings
# ------------------------------------------------------------

train_32_df = pd.DataFrame(
    X_train_32,
    columns=[f"env_emb_{i+1}" for i in range(32)]
)

test_32_df = pd.DataFrame(
    X_test_32,
    columns=[f"env_emb_{i+1}" for i in range(32)]
)

train_32_df.to_csv(
    OUTPUT_TRAIN,
    index=False
)

test_32_df.to_csv(
    OUTPUT_TEST,
    index=False
)

# ------------------------------------------------------------
# 7. Save PCA model
# ------------------------------------------------------------

joblib.dump(
    pca,
    PCA_MODEL
)

# ------------------------------------------------------------
# 8. Final validation
# ------------------------------------------------------------

print("\nFINAL SHAPES")
print("=" * 50)

print("32D training embedding:", train_32_df.shape)
print("32D testing embedding :", test_32_df.shape)

print("\nSaved files:")
print(OUTPUT_TRAIN)
print(OUTPUT_TEST)
print(PCA_MODEL)

print("\n128D -> 32D environmental embedding completed successfully!")