"""
Advanced Feature Engineering and Model Comparison for Code Error Classification.

Evaluates and compares:
1. Word-level TF-IDF (Baseline)
2. Character-level TF-IDF (Syntactic & structural patterns)
3. Hybrid Word + Character TF-IDF (Feature Union)

Generates comprehensive evaluation metrics, confusion matrix visualizations,
and persists the top-performing production model and pipeline.
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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# Paths & Directories
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "dataset" / "train.csv"
TEST_PATH = PROJECT_ROOT / "dataset" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Ensure baseline comparison is backed up if present
baseline_csv = RESULTS_DIR / "model_comparison.csv"
baseline_backup = RESULTS_DIR / "baseline_model_comparison.csv"
if baseline_csv.exists() and not baseline_backup.exists():
    import shutil
    shutil.copy(baseline_csv, baseline_backup)


# ============================================================
# Load Datasets
# ============================================================

print("=" * 75)
print(" ADVANCED CODE ERROR CLASSIFICATION - FEATURE ENGINEERING & TRAINING")
print("=" * 75)

print("\nLoading datasets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training samples : {len(train_df):,}")
print(f"Test samples     : {len(test_df):,}")

X_train = train_df["studentAnswer"].fillna("")
y_train = train_df["R_errortype"]

X_test = test_df["studentAnswer"].fillna("")
y_test = test_df["R_errortype"]

classes = sorted(y_train.unique())


# ============================================================
# Define Feature Extractors & Pipelines
# ============================================================

print("\n" + "=" * 75)
print("CONFIGURING FEATURE REPRESENTATIONS & EXPERIMENTAL PIPELINES")
print("=" * 75)

# 1. Word-level TF-IDF (Lexical)
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    lowercase=False,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=25000,
    sublinear_tf=True,
)

# 2. Character-level TF-IDF (Syntactic & Operator Patterns)
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    lowercase=False,
    ngram_range=(2, 5),
    min_df=3,
    max_df=0.98,
    max_features=40000,
    sublinear_tf=True,
)

# 3. Hybrid Feature Union (Word + Character)
hybrid_features = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(
        analyzer="word",
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=25000,
        sublinear_tf=True,
    )),
    ("char_tfidf", TfidfVectorizer(
        analyzer="char",
        lowercase=False,
        ngram_range=(2, 5),
        min_df=3,
        max_df=0.98,
        max_features=40000,
        sublinear_tf=True,
    )),
])

experiments = {
    "Baseline (Word TF-IDF + Linear SVM)": Pipeline([
        ("tfidf", word_vectorizer),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)),
    ]),

    "Experiment A (Char TF-IDF + Linear SVM)": Pipeline([
        ("tfidf", char_vectorizer),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)),
    ]),

    "Experiment B (Hybrid TF-IDF + Linear SVM)": Pipeline([
        ("features", hybrid_features),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)),
    ]),

    "Experiment C (Hybrid TF-IDF + Calibrated Linear SVM)": Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(analyzer="word", lowercase=False, ngram_range=(1, 2), min_df=2, max_df=0.98, max_features=25000, sublinear_tf=True)),
            ("char_tfidf", TfidfVectorizer(analyzer="char", lowercase=False, ngram_range=(2, 5), min_df=3, max_df=0.98, max_features=40000, sublinear_tf=True)),
        ])),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
            cv=3
        )),
    ]),

    "Experiment D (Hybrid TF-IDF + Logistic Regression)": Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(analyzer="word", lowercase=False, ngram_range=(1, 2), min_df=2, max_df=0.98, max_features=25000, sublinear_tf=True)),
            ("char_tfidf", TfidfVectorizer(analyzer="char", lowercase=False, ngram_range=(2, 5), min_df=3, max_df=0.98, max_features=40000, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ]),
}


# ============================================================
# Execute Experiments & Track Metrics
# ============================================================

print("\n" + "=" * 75)
print("TRAINING & EVALUATING ADVANCED EXPERIMENTS")
print("=" * 75)

comparison_results = []
trained_pipelines = {}
predictions_dict = {}

for exp_name, pipeline in experiments.items():
    print("\n" + "-" * 75)
    print(f"Executing: {exp_name}")
    print("-" * 75)
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - start_time
    
    start_pred = time.time()
    preds = pipeline.predict(X_test)
    pred_duration = time.time() - start_pred
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    
    print(f"Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro Precision  : {prec:.4f}")
    print(f"Macro Recall     : {rec:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Weighted F1      : {weighted_f1:.4f}")
    print(f"Training Time    : {train_duration:.2f}s")
    print(f"Inference Time   : {pred_duration:.2f}s")
    
    comparison_results.append({
        "Experiment": exp_name,
        "Accuracy": acc,
        "Macro Precision": prec,
        "Macro Recall": rec,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Training Time (s)": train_duration,
        "Inference Time (s)": pred_duration,
    })
    
    trained_pipelines[exp_name] = pipeline
    predictions_dict[exp_name] = preds


# ============================================================
# Comparison Table & Model Selection
# ============================================================

results_df = pd.DataFrame(comparison_results)
results_df = results_df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)

print("\n" + "=" * 75)
print("ADVANCED EXPERIMENTAL COMPARISON TABLE")
print("=" * 75)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best_exp_name = results_df.iloc[0]["Experiment"]
best_pipeline = trained_pipelines[best_exp_name]
best_preds = predictions_dict[best_exp_name]

print("\n" + "=" * 75)
print("SELECTED PRODUCTION PIPELINE")
print("=" * 75)
print(f"Selected Top Model : {best_exp_name}")
print(f"Top Macro F1 Score : {results_df.iloc[0]['Macro F1']:.4f}")
print(f"Top Accuracy       : {results_df.iloc[0]['Accuracy']*100:.2f}%")


# ============================================================
# Detailed Classification Report
# ============================================================

report = classification_report(y_test, best_preds, zero_division=0)

print("\n" + "=" * 75)
print(f"CLASSIFICATION REPORT - {best_exp_name}")
print("=" * 75)
print(report)


# ============================================================
# Confusion Matrix Visualization
# ============================================================

print("\nGenerating confusion matrix visualization...")

cm = confusion_matrix(y_test, best_preds, labels=classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    cbar=True,
    square=True,
    linewidths=0.5,
    linecolor="#e0e0e0",
    annot_kws={"size": 9}
)

plt.title(f"Normalized Confusion Matrix\n{best_exp_name}", fontsize=14, pad=15, weight='bold')
plt.xlabel("Predicted Label", fontsize=11, labelpad=10, weight='bold')
plt.ylabel("True Label", fontsize=11, labelpad=10, weight='bold')
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

cm_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved confusion matrix heatmap to: {cm_path}")


# ============================================================
# Save Artifacts & Metadata
# ============================================================

print("\n" + "=" * 75)
print("SAVING ADVANCED PIPELINE ARTIFACTS")
print("=" * 75)

# Save best full pipeline
joblib.dump(best_pipeline, MODEL_DIR / "best_pipeline.pkl")
joblib.dump(best_pipeline, MODEL_DIR / "best_model.pkl")

# Save comparison table
results_df.to_csv(RESULTS_DIR / "advanced_model_comparison.csv", index=False)

# Save text report
with open(RESULTS_DIR / "advanced_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Best Production Pipeline: {best_exp_name}\n")
    f.write(f"Macro F1: {results_df.iloc[0]['Macro F1']:.4f}\n")
    f.write(f"Accuracy: {results_df.iloc[0]['Accuracy']:.4f}\n\n")
    f.write("=" * 60 + "\n")
    f.write("Detailed Classification Report:\n")
    f.write("=" * 60 + "\n")
    f.write(report)

# Save run metadata
metadata = {
    "selected_model": best_exp_name,
    "macro_f1": float(results_df.iloc[0]["Macro F1"]),
    "weighted_f1": float(results_df.iloc[0]["Weighted F1"]),
    "accuracy": float(results_df.iloc[0]["Accuracy"]),
    "training_samples": len(train_df),
    "test_samples": len(test_df),
    "num_classes": len(classes),
    "classes": classes,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

with open(RESULTS_DIR / "advanced_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("✓ models/best_pipeline.pkl")
print("✓ models/best_model.pkl")
print("✓ results/advanced_model_comparison.csv")
print("✓ results/advanced_classification_report.txt")
print("✓ results/confusion_matrix.png")
print("✓ results/advanced_metadata.json")

print("\n" + "=" * 75)
print("ADVANCED FEATURE ENGINEERING & TRAINING COMPLETED SUCCESSFULLY")
print("=" * 75)
