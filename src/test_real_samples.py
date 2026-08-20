import pandas as pd
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import predict_code


TEST_PATH = PROJECT_ROOT / "dataset" / "test.csv"

df = pd.read_csv(TEST_PATH)


TARGET_CLASSES = [
    "NameError",
    "TypeError",
    "UnboundLocalError",
    "IndexError",
    "KeyError",
    "RecursionError",
    "EOFError",
    "ValueError",
    "AttributeError",
]


print("=" * 80)
print("CODEGUARD AI — REAL DATASET PREDICTION TEST")
print("=" * 80)


for error_type in TARGET_CLASSES:

    samples = df[
        df["R_errortype"] == error_type
    ]

    if len(samples) == 0:
        continue

    sample = samples.iloc[0]

    code = sample["studentAnswer"]

    result = predict_code(code)

    print("\n" + "-" * 80)
    print(f"Expected   : {error_type}")
    print(f"Predicted  : {result['error_type']}")
    print(
        f"Confidence : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print("\nCode:")
    print(code[:1000])

    print("\nTop predictions:")

    top_predictions = sorted(
        result["decision_scores"].items(),
        key=lambda item: item[1],
        reverse=True
    )[:5]

    for label, score in top_predictions:
        print(
            f"  {label:25} "
            f"{score * 100:.2f}%"
        )
