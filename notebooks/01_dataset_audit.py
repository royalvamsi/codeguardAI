import sys
import io
import pandas as pd
from datasets import load_dataset
from collections import Counter

# Set standard output encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading CircleCat/pymeta dataset...")
ds = load_dataset('CircleCat/pymeta')

# Convert train split to pandas
train_df = ds["train"].to_pandas()
validation_df = ds["validation"].to_pandas()
test_df = ds["test"].to_pandas()

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)
print(f"Train       : {len(train_df):,}")
print(f"Validation  : {len(validation_df):,}")
print(f"Test        : {len(test_df):,}")
print(f"Total       : {len(train_df) + len(validation_df) + len(test_df):,}")

print("\n" + "=" * 70)
print("ERROR CATEGORY DISTRIBUTION - TRAIN")
print("=" * 70)
distribution = train_df["error_category"].value_counts()
print(distribution)

print("\nPercentage distribution:")
print((train_df["error_category"].value_counts(normalize=True) * 100).round(2))

print("\n" + "=" * 70)
print("ALL ERROR CATEGORIES")
print("=" * 70)
all_categories = sorted(
    set(train_df["error_category"].dropna())
    | set(validation_df["error_category"].dropna())
    | set(test_df["error_category"].dropna())
)
for category in all_categories:
    print(category)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(train_df[
    ["studentAnswer", "error_category", "R_errortype", "R_traceback"]
].isnull().sum())

print("\n" + "=" * 70)
print("DUPLICATE CODE")
print("=" * 70)
duplicates = train_df["studentAnswer"].duplicated().sum()
print(f"Duplicate studentAnswer records: {duplicates:,}")

print("\n" + "=" * 70)
print("EMPTY CODE")
print("=" * 70)
empty_code = (
    train_df["studentAnswer"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)
print(f"Empty code records: {empty_code:,}")

print("\n" + "=" * 70)
print("ERROR CATEGORY vs R_ERRORTYPE")
print("=" * 70)
comparison = pd.crosstab(
    train_df["error_category"],
    train_df["R_errortype"]
)
print(comparison.to_string())

print("\n" + "=" * 70)
print("ERROR TYPE DISTRIBUTION ACROSS ALL SPLITS")
print("=" * 70)

for split_name, dataframe in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df)
]:
    print(f"\n--- {split_name} ---")
    print(dataframe["R_errortype"].value_counts().to_string())

print("\n" + "=" * 70)
print("CROSS-SPLIT DUPLICATE ANALYSIS")
print("=" * 70)

train_codes = set(
    train_df["studentAnswer"]
    .fillna("")
    .astype(str)
)

validation_codes = set(
    validation_df["studentAnswer"]
    .fillna("")
    .astype(str)
)

test_codes = set(
    test_df["studentAnswer"]
    .fillna("")
    .astype(str)
)

train_validation_overlap = train_codes & validation_codes
train_test_overlap = train_codes & test_codes
validation_test_overlap = validation_codes & test_codes

print(
    f"Train ↔ Validation overlap: "
    f"{len(train_validation_overlap):,}"
)

print(
    f"Train ↔ Test overlap: "
    f"{len(train_test_overlap):,}"
)

print(
    f"Validation ↔ Test overlap: "
    f"{len(validation_test_overlap):,}"
)

print("\n" + "=" * 70)
print("UNIQUE CODE COUNTS")
print("=" * 70)

print(f"Train unique codes      : {train_df['studentAnswer'].nunique():,}")
print(f"Validation unique codes : {validation_df['studentAnswer'].nunique():,}")
print(f"Test unique codes       : {test_df['studentAnswer'].nunique():,}")

print("\n" + "=" * 70)
print("ERROR TYPE / NO ERROR")
print("=" * 70)

print(
    train_df.groupby("R_errortype")
    .size()
    .sort_values(ascending=False)
    .to_string()
)
