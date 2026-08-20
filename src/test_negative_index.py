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

INDEX_TESTS = {
    "Test A (valid -1)": ("numbers = [10, 20, 30]\nprint(numbers[-1])", None),
    "Test B (valid -3)": ("numbers = [10, 20, 30]\nprint(numbers[-3])", None),
    "Test C (invalid -4)": ("numbers = [10, 20, 30]\nprint(numbers[-4])", "IndexError"),
    "Test D (invalid -5)": ("numbers = [10, 20, 30]\nprint(numbers[-5])", "IndexError"),
}


def main():
    print("=" * 80)
    print("NEGATIVE INDEX BOUNDARY DIAGNOSTIC TEST")
    print("=" * 80)

    for name, (code, expected_err) in INDEX_TESTS.items():
        res = predict_code(code)
        err = res["error_type"]
        is_ok = (err == expected_err)
        status = "✓" if is_ok else "✗"
        print(f"\n{status} {name}")
        print(f"  Code      : {code.strip()}")
        print(f"  Error     : {err}")
        print(f"  Method    : {res['method_label']}")
        print(f"  Confidence: {res['confidence_label']}")
        print(f"  Message   : {res['message']}")


if __name__ == "__main__":
    main()
