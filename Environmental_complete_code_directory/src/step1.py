import pandas as pd

df = pd.read_csv("environmental_dataset.csv")

print("================================")
print("ENVIRONMENTAL DATASET")
print("================================")

print("\nShape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print(column)

print("\nFirst 5 rows:")
print(df.head())

print("\nASD target values:")
print(df["k2q35a"].value_counts(dropna=False))

print("\nASD target data type:")
print(df["k2q35a"].dtype)