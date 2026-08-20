"""
Train and compare machine-learning models for
Python code error classification.

Feature:
    studentAnswer

Target:
    R_errortype

Models:
    1. Logistic Regression
    2. Multinomial Naive Bayes
    3. Linear SVM
    4. Random Forest

The best model is selected using Macro F1.
"""

import sys
import io
from pathlib import Path
import json
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "dataset" / "train.csv"
TEST_PATH = PROJECT_ROOT / "dataset" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

MAX_FEATURES = 30000
NGRAM_RANGE = (1, 2)

MIN_DF = 2
MAX_DF = 0.98


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("CODE ERROR CLASSIFICATION - MODEL TRAINING")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training samples : {len(train_df):,}")
print(f"Test samples     : {len(test_df):,}")


X_train = train_df["studentAnswer"].fillna("")
y_train = train_df["R_errortype"]

X_test = test_df["studentAnswer"].fillna("")
y_test = test_df["R_errortype"]


# ============================================================
# TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: TF-IDF FEATURE EXTRACTION")
print("=" * 70)

vectorizer = TfidfVectorizer(
    analyzer="word",
    lowercase=False,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_df=MAX_DF,
    max_features=MAX_FEATURES,
    sublinear_tf=True,
)

start = time.time()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

elapsed = time.time() - start

print(
    f"\nTF-IDF vocabulary size: "
    f"{len(vectorizer.vocabulary_):,}"
)

print(
    f"Training matrix shape: "
    f"{X_train_tfidf.shape}"
)

print(
    f"Test matrix shape: "
    f"{X_test_tfidf.shape}"
)

print(
    f"Feature extraction time: "
    f"{elapsed:.2f} seconds"
)


# ============================================================
# Models
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Naive Bayes": MultinomialNB(
        alpha=0.1
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ============================================================
# Train and evaluate
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: MODEL TRAINING & EVALUATION")
print("=" * 70)

results = []

trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)

    start = time.time()

    model.fit(
        X_train_tfidf,
        y_train
    )

    training_time = time.time() - start

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"Training Time  : {training_time:.2f}s")

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Training Time (sec)": training_time,
    })

    trained_models[model_name] = model


# ============================================================
# Model comparison
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# Select best model
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"\nSelected model: {best_model_name}"
)

print(
    f"Macro F1: "
    f"{results_df.iloc[0]['Macro F1']:.4f}"
)


# ============================================================
# Detailed classification report
# ============================================================

best_predictions = best_model.predict(
    X_test_tfidf
)

report = classification_report(
    y_test,
    best_predictions,
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT - BEST MODEL")
print("=" * 70)

print(report)


# ============================================================
# Save artifacts
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL ARTIFACTS")
print("=" * 70)

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)

results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"Best Model: {best_model_name}\n\n"
    )

    file.write(report)


metadata = {
    "best_model": best_model_name,
    "random_state": RANDOM_STATE,
    "max_features": MAX_FEATURES,
    "ngram_range": list(NGRAM_RANGE),
    "min_df": MIN_DF,
    "max_df": MAX_DF,
    "training_samples": len(train_df),
    "test_samples": len(test_df),
    "feature_count": len(vectorizer.vocabulary_),
}

with open(
    RESULTS_DIR / "training_metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print("\n✓ tfidf_vectorizer.pkl")
print("✓ best_model.pkl")
print("✓ model_comparison.csv")
print("✓ classification_report.txt")
print("✓ training_metadata.json")

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)
