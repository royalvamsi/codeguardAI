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

TESTS = [
    (
        "Test 1 — Clean loop & append",
        """def process(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result

numbers = [1, 2, 3, 4]
print(process(numbers))""",
        None
    ),
    (
        "Test 2 — Genuine UnboundLocalError",
        """def calculate():
    print(total)
    total = 100

calculate()""",
        "UnboundLocalError"
    ),
    (
        "Test 3 — Another genuine case (conditional assign)",
        """def update():
    print(value)
    if True:
        value = 10

update()""",
        "UnboundLocalError"
    ),
    (
        "Test 4 — Clean accumulator loop",
        """def calculate(numbers):
    total = 0
    for number in numbers:
        total += number
    return total

print(calculate([10, 20, 30]))""",
        None
    ),
]


def main():
    print("=" * 80)
    print("UNBOUND LOCAL ERROR CONTROL-FLOW TEST")
    print("=" * 80)

    for name, code, expected_err in TESTS:
        res = predict_code(code)
        err = res["error_type"]
        is_ok = (err == expected_err)
        status = "✓ PASS" if is_ok else "✗ FAIL"
        print(f"\n{status} | {name}")
        print(f"       Expected : {expected_err}")
        print(f"       Actual   : {err} | {res['method_label']}")
        print(f"       Message  : {res['message']}")


if __name__ == "__main__":
    main()
