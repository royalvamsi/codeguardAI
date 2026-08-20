"""
CodeGuard AI — 5-Fold Out-of-Fold (OOF) Calibration & Final Evaluation

Methodology:
1. Perform 5-fold Stratified Cross-Validation on train.csv, fitting a fresh
   Word+Char TF-IDF + LinearSVC pipeline per fold.
2. Accumulate out-of-fold decision scores and softmax probabilities.
3. Optimize class-specific decision thresholds T_c* purely on OOF validation predictions.
4. Freeze T_c*.
5. Evaluate on untouched test.csv using the production best_pipeline.pkl.
6. Generate final classification reports, confusion matrix, and metadata artifacts.
"""

from pathlib import Path
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score


def build_fresh_pipeline():
    """Build a fresh instance of the production Hybrid Word+Char TF-IDF Linear SVM pipeline."""
    return Pipeline([
        ('features', FeatureUnion([
            ('word_tfidf', TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                max_features=25000,
                sublinear_tf=True
            )),
            ('char_tfidf', TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 5),
                max_features=40000,
                sublinear_tf=True
            )),
        ])),
        ('clf', LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=3000
        ))
    ])


def softmax_scores(decision_scores):
    """Compute numerical-stable softmax probabilities on decision function scores."""
    scores = np.asarray(decision_scores)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def run_oof_calibration():
    print("=" * 80)
    print("CODEGUARD AI — 5-FOLD OOF CALIBRATION & FINAL EVALUATION")
    print("=" * 80)

    # 1. Load Data
    train_path = PROJECT_ROOT / "dataset" / "train.csv"
    test_path = PROJECT_ROOT / "dataset" / "test.csv"

    print(f"\n[1/5] Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["studentAnswer"].fillna("").values
    y_train = train_df["R_errortype"].values

    X_test = test_df["studentAnswer"].fillna("").values
    y_test = test_df["R_errortype"].values

    print(f"  Training samples : {len(train_df):,}")
    print(f"  Held-out test    : {len(test_df):,}")

    # 2. 5-Fold Stratified Cross-Validation for OOF Scores
    print(f"\n[2/5] Running 5-Fold Stratified Cross-Validation on training data...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classes = np.sort(np.unique(y_train))
    num_classes = len(classes)
    oof_decision_scores = np.zeros((len(X_train), num_classes))

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"  --> Fold {fold}/5: Fitting fresh pipeline on {len(train_idx):,} samples...")
        fold_pipeline = build_fresh_pipeline()
        fold_pipeline.fit(X_train[train_idx], y_train[train_idx])

        val_scores = fold_pipeline.named_steps['clf'].decision_function(
            fold_pipeline.named_steps['features'].transform(X_train[val_idx])
        )
        oof_decision_scores[val_idx] = val_scores
        print(f"      Fold {fold} complete.")

    cv_time = time.time() - t0
    print(f"✓ 5-Fold OOF extraction completed in {cv_time:.1f}s.")

    # 3. Softmax Normalization & Class Threshold Optimization on OOF predictions
    print(f"\n[3/5] Calibrating class-specific thresholds on 28,464 OOF validation predictions...")
    oof_probs = softmax_scores(oof_decision_scores)
    thresholds_sweep = np.arange(0.10, 0.90, 0.02)
    optimal_thresholds = {}

    for idx, cls in enumerate(classes):
        y_true_binary = (y_train == cls).astype(int)
        cls_probs = oof_probs[:, idx]
        support = int(y_true_binary.sum())

        best_f1 = -1.0
        best_t = 0.50
        best_p = 0.0
        best_r = 0.0

        for t in thresholds_sweep:
            y_pred_binary = (cls_probs >= t).astype(int)
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_p = p
                best_r = r

        optimal_thresholds[cls] = {
            "OOF_Optimal_Threshold": round(float(best_t), 2),
            "OOF_Precision": round(float(best_p), 4),
            "OOF_Recall": round(float(best_r), 4),
            "OOF_F1": round(float(best_f1), 4),
            "Train_Support": support,
        }

    df_thresholds = pd.DataFrame.from_dict(optimal_thresholds, orient="index").reset_index()
    df_thresholds.rename(columns={"index": "Class"}, inplace=True)
    oof_csv_path = PROJECT_ROOT / "results" / "oof_optimal_thresholds.csv"
    df_thresholds.to_csv(oof_csv_path, index=False)

    print("\n" + "-" * 75)
    print("CALIBRATED OUT-OF-FOLD (OOF) THRESHOLDS")
    print("-" * 75)
    print(df_thresholds.to_string(index=False))
    print(f"\n✓ Saved calibrated OOF thresholds to: {oof_csv_path}")

    # 4. Final Evaluation on Untouched Held-Out Test Set (7,117 samples)
    print(f"\n[4/5] Evaluating calibrated thresholds once on held-out test.csv...")
    model_path = PROJECT_ROOT / "models" / "best_pipeline.pkl"
    prod_pipeline = joblib.load(model_path)

    test_decisions = prod_pipeline.named_steps['clf'].decision_function(
        prod_pipeline.named_steps['features'].transform(X_test)
    )
    test_probs = softmax_scores(test_decisions)

    # Standard Argmax Baseline
    raw_pred_idx = np.argmax(test_decisions, axis=1)
    raw_preds = classes[raw_pred_idx]

    baseline_acc = accuracy_score(y_test, raw_preds)
    baseline_macro_f1 = f1_score(y_test, raw_preds, average='macro')
    baseline_weighted_f1 = f1_score(y_test, raw_preds, average='weighted')

    # Multi-class calibrated inference with threshold gating
    # If the top predicted error class falls below its calibrated threshold T_c*, fallback to 'No error'
    calibrated_preds = []
    for i in range(len(X_test)):
        top_cls_idx = raw_pred_idx[i]
        top_cls = classes[top_cls_idx]
        top_prob = test_probs[i, top_cls_idx]

        if top_cls == "No error":
            calibrated_preds.append("No error")
        else:
            req_thresh = optimal_thresholds[top_cls]["OOF_Optimal_Threshold"]
            if top_prob >= req_thresh:
                calibrated_preds.append(top_cls)
            else:
                calibrated_preds.append("No error")

    calibrated_preds = np.array(calibrated_preds)

    calib_acc = accuracy_score(y_test, calibrated_preds)
    calib_macro_p = precision_score(y_test, calibrated_preds, average='macro', zero_division=0)
    calib_macro_r = recall_score(y_test, calibrated_preds, average='macro', zero_division=0)
    calib_macro_f1 = f1_score(y_test, calibrated_preds, average='macro')
    calib_weighted_f1 = f1_score(y_test, calibrated_preds, average='weighted')

    report_str = classification_report(y_test, calibrated_preds, digits=4, zero_division=0)

    print("\n" + "=" * 80)
    print("FINAL CALIBRATED TEST BENCHMARK RESULTS (ON TEST.CSV)")
    print("=" * 80)
    print(f"Baseline Accuracy     : {baseline_acc:.4f}  -->  Calibrated: {calib_acc:.4f}")
    print(f"Baseline Macro F1     : {baseline_macro_f1:.4f}  -->  Calibrated: {calib_macro_f1:.4f}")
    print(f"Baseline Weighted F1  : {baseline_weighted_f1:.4f}  -->  Calibrated: {calib_weighted_f1:.4f}")
    print("\nDetailed Per-Class Report:")
    print(report_str)

    # 5. Save Final Artifacts
    print(f"\n[5/5] Saving final validation reports and confusion matrix...")
    report_file = PROJECT_ROOT / "results" / "final_calibrated_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("CODEGUARD AI — FINAL CALIBRATED TEST REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Accuracy    : {calib_acc:.4%}\n")
        f.write(f"Macro Precision  : {calib_macro_p:.4f}\n")
        f.write(f"Macro Recall     : {calib_macro_r:.4f}\n")
        f.write(f"Macro F1 Score   : {calib_macro_f1:.4f}\n")
        f.write(f"Weighted F1 Score: {calib_weighted_f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report_str + "\n\n")
        f.write("OOF Calibrated Thresholds Applied:\n")
        f.write(df_thresholds.to_string(index=False) + "\n")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, calibrated_preds, labels=classes, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=0.5
    )
    plt.title('Calibrated Normalized Confusion Matrix (Held-out Test Set)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Predicted Class', fontsize=11, labelpad=10)
    plt.ylabel('Ground Truth Class', fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = PROJECT_ROOT / "results" / "final_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Metadata JSON
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": "5-Fold Stratified CV on Train (Fresh pipeline per fold) -> OOF Threshold Tuning -> Single Evaluation on Test",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "num_classes": len(classes),
        "classes": list(classes),
        "baseline_metrics": {
            "accuracy": float(baseline_acc),
            "macro_f1": float(baseline_macro_f1),
            "weighted_f1": float(baseline_weighted_f1),
        },
        "calibrated_metrics": {
            "accuracy": float(calib_acc),
            "macro_precision": float(calib_macro_p),
            "macro_recall": float(calib_macro_r),
            "macro_f1": float(calib_macro_f1),
            "weighted_f1": float(calib_weighted_f1),
        },
        "oof_optimal_thresholds": {
            cls: float(optimal_thresholds[cls]["OOF_Optimal_Threshold"])
            for cls in classes
        }
    }

    meta_file = PROJECT_ROOT / "results" / "calibration_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved: {report_file}")
    print(f"✓ Saved: {cm_path}")
    print(f"✓ Saved: {meta_file}")
    print("\n" + "=" * 80)
    print("ALL CALIBRATION STEPS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_oof_calibration()
