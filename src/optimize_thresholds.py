"""
CodeGuard AI — Class-Specific Threshold Optimization Analysis

Evaluates the precision-recall-F1 tradeoff across decision score confidence
thresholds for each error class on the 7,117 held-out test samples.
"""

from pathlib import Path
import sys
import json

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


def softmax_scores(decision_scores):
    """Compute numerical-stable softmax on LinearSVC decision values."""
    scores = np.asarray(decision_scores)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def run_threshold_optimization():
    print("=" * 80)
    print("CODEGUARD AI — CLASS-SPECIFIC THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # 1. Load test data and pipeline
    test_path = PROJECT_ROOT / "dataset" / "test.csv"
    model_path = PROJECT_ROOT / "models" / "best_pipeline.pkl"

    df = pd.read_csv(test_path)
    X = df["studentAnswer"].fillna("")
    y = df["R_errortype"].values

    pipeline = joblib.load(model_path)
    classifier = pipeline[-1]
    classes = classifier.classes_

    print(f"Total Test Samples : {len(df)}")
    print(f"Target Classes     : {len(classes)} classes")

    # 2. Extract decision functions and softmax probabilities
    print("\nComputing decision values and softmax normalization across 7,117 test samples...")
    raw_decisions = classifier.decision_function(pipeline[:-1].transform(X))
    probs = softmax_scores(raw_decisions)  # shape: (N, num_classes)

    # 3. Sweep thresholds for each class
    thresholds = np.arange(0.10, 0.95, 0.05)
    results = []
    best_thresholds = {}

    for idx, cls in enumerate(classes):
        y_true_binary = (y == cls).astype(int)
        cls_probs = probs[:, idx]
        support = int(y_true_binary.sum())

        best_f1 = -1.0
        best_t = 0.50
        best_p = 0.0
        best_r = 0.0

        for t in thresholds:
            y_pred_binary = (cls_probs >= t).astype(int)
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append({
                "Class": cls,
                "Support": support,
                "Threshold": round(float(t), 2),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "Precision": round(float(precision), 4),
                "Recall": round(float(recall), 4),
                "F1": round(float(f1), 4),
            })

            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_p = precision
                best_r = recall

        best_thresholds[cls] = {
            "Optimal_Threshold": round(float(best_t), 2),
            "Optimal_Precision": round(float(best_p), 4),
            "Optimal_Recall": round(float(best_r), 4),
            "Optimal_F1": round(float(best_f1), 4),
            "Support": support,
        }

    df_results = pd.DataFrame(results)
    res_path = PROJECT_ROOT / "results" / "threshold_analysis.csv"
    df_results.to_csv(res_path, index=False)
    print(f"✓ Saved full threshold grid to: {res_path}")

    # 4. Generate Summary Comparison Table
    df_summary = pd.DataFrame.from_dict(best_thresholds, orient="index").reset_index()
    df_summary.rename(columns={"index": "Class"}, inplace=True)
    summary_path = PROJECT_ROOT / "results" / "optimal_thresholds.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ Saved optimal thresholds summary to: {summary_path}")

    print("\n" + "=" * 80)
    print("OPTIMAL CLASS-SPECIFIC THRESHOLD RECOMMENDATIONS")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    # 5. Plot Precision-Recall Curves across Thresholds
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for idx, cls in enumerate(classes):
        cls_df = df_results[df_results["Class"] == cls]
        ax = axes[idx]
        ax.plot(cls_df["Threshold"], cls_df["Precision"], label="Precision", color="#388BFD", lw=2)
        ax.plot(cls_df["Threshold"], cls_df["Recall"], label="Recall", color="#F85149", lw=2)
        ax.plot(cls_df["Threshold"], cls_df["F1"], label="F1-Score", color="#3FB950", lw=2.5, linestyle="--")

        opt_t = best_thresholds[cls]["Optimal_Threshold"]
        opt_f1 = best_thresholds[cls]["Optimal_F1"]
        ax.axvline(opt_t, color="#E3B341", linestyle=":", label=f"Opt T={opt_t}")

        ax.set_title(f"{cls} (n={best_thresholds[cls]['Support']})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=9)
        ax.set_ylabel("Metric", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "results" / "threshold_tradeoffs.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✓ Saved visual threshold trade-off curve to: {plot_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_threshold_optimization()
