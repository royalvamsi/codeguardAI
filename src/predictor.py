"""
Unified Prediction Engine for CodeGuard AI.

Integrates:
1. Layer 1: Deterministic Static Code Analysis (AST & Pattern Checking)
2. Layer 2: Hybrid ML Classification (Word + Char TF-IDF + Linear SVM)
3. Layer 3: Bug Suggestion & Diagnostic Engine
"""

from pathlib import Path
import sys
from typing import Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.static_analyzer import analyze_code
from src.suggestion_engine import get_suggestion

MODEL_PATH = PROJECT_ROOT / "models" / "best_pipeline.pkl"

ML_CONFIDENCE_THRESHOLD = 0.50

_pipeline = None


def load_pipeline():
    """Load the trained hybrid production pipeline."""
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_code(code: str) -> Dict[str, Any]:
    """
    Predict error type for given Python source code using a two-stage architecture:
    Stage 1: High-confidence static/AST analysis
    Stage 2: Hybrid Word+Char TF-IDF Linear SVM classification
    Stage 3: Contextual bug suggestion retrieval

    Returns:
        {
            "error_type": str,
            "confidence": float,
            "source": "static_analysis" | "machine_learning",
            "method_label": str,
            "confidence_label": str,
            "message": str,
            "suggestion": str,
            "decision_scores": dict,
            "explanation": dict
        }
    """
    if not isinstance(code, str):
        raise TypeError("Code must be provided as a string.")

    code_clean = code.strip()
    if not code_clean:
        raise ValueError("Python code cannot be empty.")

    # ========================================================
    # Stage 1: Deterministic Static Analysis (AST & Heuristics)
    # ========================================================
    static_res = analyze_code(code)

    if static_res["error_type"] is not None and static_res["confidence"] >= 0.85:
        error_type = static_res["error_type"]
        confidence = static_res["confidence"]
        message = static_res["message"]
        suggestion = static_res["suggestion"]
        source = "static_analysis"
        method_label = "Static Analysis (AST & Rules)"
        confidence_label = "High (Deterministic)"

        explanation = get_suggestion(error_type, custom_message=message)

        # Construct deterministic decision score profile
        classes = [
            "No error", "NameError", "TypeError", "UnboundLocalError",
            "IndexError", "KeyError", "RecursionError", "EOFError",
            "ValueError", "AttributeError", "SyntaxError"
        ]
        decision_scores = {cls_name: (1.0 if cls_name == error_type else 0.0) for cls_name in classes}

        return {
            "error_type": error_type,
            "confidence": confidence,
            "source": source,
            "method_label": method_label,
            "confidence_label": confidence_label,
            "message": message,
            "suggestion": suggestion,
            "decision_scores": decision_scores,
            "explanation": explanation,
        }

    # ========================================================
    # Stage 2: Hybrid ML Classification
    # ========================================================
    pipeline = load_pipeline()

    prediction = pipeline.predict([code])[0]
    classifier = pipeline[-1]

    # Compute softmax-normalized relative decision scores
    scores = classifier.decision_function(pipeline[:-1].transform([code]))
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    scores = scores[0]
    classes = classifier.classes_

    exp_scores = np.exp(scores - np.max(scores))
    normalized = exp_scores / exp_scores.sum()

    pred_idx = np.where(classes == prediction)[0][0]
    confidence = float(normalized[pred_idx])

    decision_scores = {
        str(label): float(score)
        for label, score in zip(classes, normalized)
    }

    # Sort decision scores
    decision_scores = dict(sorted(decision_scores.items(), key=lambda item: item[1], reverse=True))

    # ========================================================
    # ML Confidence Gate
    # ========================================================

    if str(prediction) == "No error" or confidence < ML_CONFIDENCE_THRESHOLD:
        return {
            "error_type": None,
            "confidence": confidence,
            "source": "machine_learning",
            "method_label": "Hybrid ML (Word + Char TF-IDF + Linear SVM)",
            "confidence_label": f"{confidence * 100:.1f}% (Below Error Threshold)",
            "message": "No high-confidence error was detected in the submitted code.",
            "suggestion": "The code appears valid based on the available static and machine-learning evidence.",
            "decision_scores": decision_scores,
            "explanation": {
                "title": "No High-Confidence Error Detected",
                "category": "Clean / Low-Confidence",
                "description": "The machine-learning classifier did not produce a sufficiently confident error prediction.",
                "common_causes": [],
                "suggestions": [
                    "Review runtime inputs and edge cases if the code depends on external data.",
                    "Add unit tests for boundary conditions."
                ]
            },
        }

    source = "machine_learning"
    method_label = "Hybrid ML (Word + Char TF-IDF + Linear SVM)"
    confidence_label = f"{confidence * 100:.1f}% (Relative Decision Score)"

    explanation = get_suggestion(str(prediction))
    message = explanation["description"]
    suggestion = explanation["suggestions"][0] if explanation.get("suggestions") else "Review code logic."

    return {
        "error_type": str(prediction),
        "confidence": confidence,
        "source": source,
        "method_label": method_label,
        "confidence_label": confidence_label,
        "message": message,
        "suggestion": suggestion,
        "decision_scores": decision_scores,
        "explanation": explanation,
    }


# ============================================================
# Quick CLI verification
# ============================================================

if __name__ == "__main__":
    test_snippets = {
        "Syntax Check": "if True\n    print('hello')",
        "Static Name Check": "print(undefined_identifier)",
        "Static Index Check": "numbers = [1, 2]\nprint(numbers[10])",
        "Clean Code": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    }

    print("=" * 70)
    print("CODEGUARD AI - UNIFIED PREDICTOR TEST")
    print("=" * 70)

    for test_name, snippet in test_snippets.items():
        res = predict_code(snippet)
        print(f"\n--- {test_name} ---")
        print(f"Error Type       : {res['error_type']}")
        print(f"Detection Method : {res['method_label']}")
        print(f"Confidence       : {res['confidence_label']}")
        print(f"Diagnostic Msg   : {res['message']}")
        print(f"Suggestion       : {res['suggestion']}")
