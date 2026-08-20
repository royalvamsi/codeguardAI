"""
Data Preparation Pipeline for Code Error Classification and Bug Detection System.
Loads PyMETA dataset, filters target classes, removes duplicates and empty code,
performs leakage-controlled stratified train/test split, and saves artifacts.
"""

import os
import sys
import io
from pathlib import Path
from typing import Tuple
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Ensure UTF-8 standard output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# Configuration & Constants
# ============================================================

DATA_DIR = Path("dataset")
CLEAN_DATASET_PATH = DATA_DIR / "clean_dataset.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET_CLASSES = [
    "No error",
    "NameError",
    "TypeError",
    "UnboundLocalError",
    "IndexError",
    "KeyError",
    "RecursionError",
    "EOFError",
    "ValueError",
    "AttributeError"
]

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# Data Loading & Preprocessing Functions
# ============================================================

def load_pymeta() -> pd.DataFrame:
    """Load CircleCat/pymeta dataset and combine train, validation, and test splits."""
    print("\nLoading CircleCat/pymeta dataset from HuggingFace Hub...")
    ds = load_dataset("CircleCat/pymeta")
    
    train_df = ds["train"].to_pandas()
    val_df = ds["validation"].to_pandas()
    test_df = ds["test"].to_pandas()
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"Loaded {len(combined_df):,} total records across all splits.")
    return combined_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    1. Keep relevant columns (studentAnswer, R_errortype).
    2. Filter out rare classes (keep only TARGET_CLASSES).
    3. Remove empty or whitespace-only code.
    4. Remove duplicate code to prevent data leakage.
    """
    print("\nCleaning data...")
    # Select relevant columns
    df_clean = df[["studentAnswer", "R_errortype"]].copy()
    
    # 1. Filter target classes
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean["R_errortype"].isin(TARGET_CLASSES)]
    print(f"Filtered target classes: {len(df_clean):,} / {initial_count:,} records retained.")
    
    # 2. Remove empty code
    df_clean["studentAnswer"] = df_clean["studentAnswer"].fillna("").astype(str)
    df_clean = df_clean[df_clean["studentAnswer"].str.strip() != ""]
    print(f"Removed empty code records: {len(df_clean):,} records remaining.")
    
    # 3. Remove duplicate studentAnswer records
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["studentAnswer"]).reset_index(drop=True)
    print(f"Deduplicated code: removed {before_dedup - len(df_clean):,} duplicates, {len(df_clean):,} unique records remaining.")
    
    return df_clean


def print_class_distribution(df: pd.DataFrame, title: str) -> None:
    """Print class counts and percentage distribution for a given DataFrame."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    counts = df["R_errortype"].value_counts()
    pcts = (df["R_errortype"].value_counts(normalize=True) * 100).round(2)
    summary_table = pd.DataFrame({"Count": counts, "Percentage (%)": pcts})
    print(summary_table.to_string())
    print(f"\nTotal: {len(df):,} samples")


def split_dataset(clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a stratified train/test split on cleaned data."""
    print(f"\nSplitting dataset into train ({(1-TEST_SIZE)*100:.0f}%) and test ({TEST_SIZE*100:.0f}%) with stratification...")
    train_df, test_df = train_test_split(
        clean_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=clean_df["R_errortype"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def verify_no_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Verify that there is zero code overlap between train and test splits."""
    print("\nVerifying cross-split code isolation (leakage check)...")
    train_codes = set(train_df["studentAnswer"])
    test_codes = set(test_df["studentAnswer"])
    overlap = train_codes & test_codes
    
    if len(overlap) == 0:
        print("✓ Zero leakage verified: Train and Test splits share 0 identical code snippets.")
    else:
        raise ValueError(f"Data leakage detected! {len(overlap)} overlapping records found between train and test splits.")


def save_datasets(clean_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save clean dataset, train split, and test split to CSV files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    clean_df.to_csv(
        CLEAN_DATASET_PATH,
        index=False,
        encoding="utf-8"
    )
    
    train_df.to_csv(
        TRAIN_PATH,
        index=False,
        encoding="utf-8"
    )
    
    test_df.to_csv(
        TEST_PATH,
        index=False,
        encoding="utf-8"
    )

    print(f"\n✓ {CLEAN_DATASET_PATH}")
    print(f"✓ {TRAIN_PATH}")
    print(f"✓ {TEST_PATH}")


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:

    print("\n")
    print("=" * 70)
    print(" CODE ERROR CLASSIFICATION - DATA PREPARATION")
    print("=" * 70)

    # 1. Load
    df = load_pymeta()

    # 2. Clean
    clean_df = clean_data(df)

    # 3. Display cleaned distribution
    print_class_distribution(
        clean_df,
        "CLEAN DATASET CLASS DISTRIBUTION"
    )

    # 4. Split
    train_df, test_df = split_dataset(clean_df)

    # 5. Verify leakage
    verify_no_overlap(
        train_df,
        test_df
    )

    # 6. Display split distributions
    print_class_distribution(
        train_df,
        "TRAINING SET DISTRIBUTION"
    )

    print_class_distribution(
        test_df,
        "TEST SET DISTRIBUTION"
    )

    # 7. Save
    save_datasets(
        clean_df,
        train_df,
        test_df
    )

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nClean dataset : {len(clean_df):,} rows"
    )

    print(
        f"Training set  : {len(train_df):,} rows"
    )

    print(
        f"Test set      : {len(test_df):,} rows"
    )


if __name__ == "__main__":
    main()
