import pandas as pd
import numpy as np

TRAIN_FILE = "environmental_embedding_train_32d.csv"
TEST_FILE = "environmental_embedding_test_32d.csv"

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("=" * 60)
print("FINAL ENVIRONMENTAL 32D EMBEDDING VALIDATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. Shape
# ------------------------------------------------------------

print("\nShapes:")
print("Training:", train.shape)
print("Testing :", test.shape)

# ------------------------------------------------------------
# 2. Number of dimensions
# ------------------------------------------------------------

print("\nNumber of dimensions:")
print("Training features:", train.shape[1])
print("Testing features :", test.shape[1])

if train.shape[1] != 32 or test.shape[1] != 32:
    raise ValueError("Expected exactly 32 embedding dimensions.")

# ------------------------------------------------------------
# 3. Missing values
# ------------------------------------------------------------

print("\nMissing values:")
print("Training:", train.isna().sum().sum())
print("Testing :", test.isna().sum().sum())

# ------------------------------------------------------------
# 4. Infinite values
# ------------------------------------------------------------

print("\nInfinite values:")
print("Training:", np.isinf(train.values).sum())
print("Testing :", np.isinf(test.values).sum())

# ------------------------------------------------------------
# 5. Constant dimensions
# ------------------------------------------------------------

constant_train = (train.std() == 0).sum()
constant_test = (test.std() == 0).sum()

print("\nConstant dimensions:")
print("Training:", constant_train)
print("Testing :", constant_test)

# ------------------------------------------------------------
# 6. Basic statistics
# ------------------------------------------------------------

print("\nFirst 10 dimensions:")
print(train.iloc[:, :10].describe().loc[["mean", "std", "min", "max"]])

# ------------------------------------------------------------
# 7. Column names
# ------------------------------------------------------------

print("\nEmbedding columns:")
print(list(train.columns))

# ------------------------------------------------------------
# 8. Final validation
# ------------------------------------------------------------

checks = {
    "Correct training shape": train.shape == (43892, 32),
    "Correct testing shape": test.shape == (10974, 32),
    "No training NaN": not train.isna().any().any(),
    "No testing NaN": not test.isna().any().any(),
    "No training infinity": not np.isinf(train.values).any(),
    "No testing infinity": not np.isinf(test.values).any(),
    "No constant training dimensions": constant_train == 0,
    "No constant testing dimensions": constant_test == 0
}

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

all_passed = True

for check, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"{status:5} - {check}")

    if not result:
        all_passed = False

print("\n" + "=" * 60)

if all_passed:
    print("ENVIRONMENTAL 32D EMBEDDING: READY FOR FUSION")
else:
    print("ENVIRONMENTAL 32D EMBEDDING: CHECK REQUIRED")

print("=" * 60)