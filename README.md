# 🛡️ CodeGuard AI — Python Code Error Classification & Bug Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-PyMETA-green.svg)](https://huggingface.co/datasets/CircleCat/pymeta)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An intelligent, multi-layered Python code analysis and error classification system developed for the **NIELIT AI/ML Capstone Project**. CodeGuard AI combines deterministic Abstract Syntax Tree (AST) parsing with a hybrid machine learning classifier (Word + Character n-gram TF-IDF + Linear SVM) to detect syntax violations and predict runtime error classes directly from raw source code.

---

## 🏛️ System Architecture

```text
                             USER PYTHON CODE
                                    │
                                    ▼
                         ┌────────────────────┐
                         │  STATIC ANALYZER   │
                         │   (AST & Rules)    │
                         └──────────┬─────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
          High-Confidence Error              No Static Violation
        (Syntax, Name, Index, Key, etc.)              │
                  │                                   ▼
                  │                         ┌───────────────────┐
                  │                         │  HYBRID ML MODEL  │
                  │                         │Word + Char TF-IDF │
                  │                         │   + Linear SVM    │
                  │                         └─────────┬─────────┘
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  SUGGESTION & DIAGNOSIS │
                       │         ENGINE          │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  STREAMLIT DASHBOARD    │
                       │       (app/app.py)      │
                       └─────────────────────────┘
```

---

## 📊 Empirical Benchmarks

Trained and evaluated on **35,581 unique student submissions** from the **PyMETA** dataset using a **leakage-controlled stratified 80/20 train/test split** (28,464 training rows, 7,117 held-out test rows).

### Model Comparison Table

| Model & Representation | Test Accuracy | Macro F1 | Weighted F1 | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid (Word + Char) TF-IDF + Linear SVM** 🏆 | **83.84%** | **0.5750** | **0.8443** | 22.40s |
| Calibrated Hybrid Linear SVM | 85.46% | 0.4490 | 0.8219 | 33.25s |
| Character-level TF-IDF + Linear SVM | 80.86% | 0.5119 | 0.8193 | 19.42s |
| Baseline Word TF-IDF + Linear SVM | 75.80% | 0.4204 | 0.7781 | 4.20s |
| Hybrid TF-IDF + Logistic Regression | 56.05% | 0.3174 | 0.6316 | 28.55s |

### Class-Wise F1 Breakdown (Production Model)

| Target Error Class | Baseline F1 | Hybrid Model F1 | Relative Gain |
| :--- | :---: | :---: | :---: |
| **No error** (Clean Code) | 0.86 | **0.91** | +5.8% |
| **KeyError** | 0.50 | **0.62** | +24.0% |
| **EOFError** | 0.40 | **0.59** | +47.5% |
| **RecursionError** | 0.34 | **0.59** | +73.5% |
| **IndexError** | 0.37 | **0.57** | +54.1% |
| **NameError** | 0.41 | **0.53** | +29.3% |
| **TypeError** | 0.34 | **0.50** | +47.1% |
| **ValueError** | 0.19 | **0.50** | **+163.2%** |
| **UnboundLocalError** | 0.39 | **0.49** | +25.6% |
| **AttributeError** | 0.39 | **0.44** | +12.8% |

---

## 📁 Repository Structure

```text
Code_Error_Classification_Bug_Detection_System/
│
├── dataset/                    # Cleaned and split dataset files
│   ├── clean_dataset.csv       # 35,581 deduplicated samples
│   ├── train.csv               # 28,464 training records (80%)
│   └── test.csv                # 7,117 held-out test records (20%)
│
├── models/                     # Serialized machine learning models
│   ├── best_pipeline.pkl       # Production Hybrid TF-IDF + SVM Pipeline
│   ├── best_model.pkl          # Trained classifier weights
│   └── tfidf_vectorizer.pkl    # TF-IDF vocabulary extractor
│
├── results/                    # Empirical evaluation reports & plots
│   ├── advanced_model_comparison.csv
│   ├── advanced_classification_report.txt
│   ├── confusion_matrix.png    # High-resolution heatmap
│   └── advanced_metadata.json
│
├── notebooks/                  # Exploratory & audit scripts
│   └── 01_dataset_audit.py     # Dataset audit & leakage inspection
│
├── src/                        # Core application & ML modules
│   ├── prepare_dataset.py      # Data loading, deduplication & splitting
│   ├── train_models.py         # Baseline ML training script
│   ├── train_advanced_models.py# Advanced feature engineering experiments
│   ├── static_analyzer.py      # AST parser & heuristic bug detector
│   ├── suggestion_engine.py    # Structured diagnosis & fix knowledge base
│   ├── predictor.py            # Unified 3-stage prediction pipeline
│   ├── test_predictor.py       # Diagnostic sanity unit test suite
│   └── test_real_samples.py    # Test split validation script
│
├── app/                        # Web user interface
│   └── app.py                  # Interactive Streamlit dashboard
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Virtual Environment

Clone the repository and initialize a Python 3.10+ virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. (Optional) Re-run Data Prep & Training

To recreate the dataset and retrain models from scratch:

```powershell
# Prepare datasets from PyMETA
python src/prepare_dataset.py

# Run advanced feature engineering experiments
python src/train_advanced_models.py
```

### 4. Launch the Web Application

```powershell
streamlit run app/app.py
```

Open your browser at `http://localhost:8501`.

---

## 🔬 Methodology Highlights for Academic Evaluation

1. **Zero Data Leakage**: Standard public benchmarks often suffer from cross-split duplicates. We performed complete deduplication across 48,646 PyMETA submissions before generating an 80/20 stratified train/test split.
2. **Rare-Class Filtering**: Categories with fewer than 10 total instances (`RuntimeError`, `MemoryError`, `ZeroDivisionError`, `ModuleNotFoundError`) were systematically removed to prevent unrepresentative evaluations.
3. **Hybrid Representation**: Combining word n-grams $(1, 2)$ and character n-grams $(2, 5)$ allows the classifier to capture both high-level semantic tokens (e.g. `import`, `range`) and fine-grained syntactic structures (e.g. `[]`, `()`, `.append()`, `+=`).
4. **Transparent Attribution**: The web interface clearly distinguishes deterministic AST findings (`Static Analysis`) from statistical predictions (`Hybrid ML`), presenting relative decision scores without misrepresenting them as calibrated probabilities.

---

## 🎓 Academic Credits

- **Institution**: National Institute of Electronics & Information Technology (NIELIT)
- **Domain**: Artificial Intelligence & Machine Learning (AI/ML)
- **Project**: Code Error Classification & Bug Detection System (CodeGuard AI)
