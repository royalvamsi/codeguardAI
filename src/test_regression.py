"""
CodeGuard AI — Complete Static + ML Regression Test Suite
Validates all 12 core diagnostic error cases.
"""

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

REGRESSION_SUITE = [
    {
        "name": "1. SyntaxError",
        "expected_error": "SyntaxError",
        "expected_source": "static_analysis",
        "code": """def calculate(a, b)
    return a + b
""",
    },
    {
        "name": "2. NameError",
        "expected_error": "NameError",
        "expected_source": "static_analysis",
        "code": """print(undefined_variable)
""",
    },
    {
        "name": "3. TypeError",
        "expected_error": "TypeError",
        "expected_source": "static_analysis",
        "code": """number = 10
text = "hello"
result = number + text
print(result)
""",
    },
    {
        "name": "4. ZeroDivisionError",
        "expected_error": "ZeroDivisionError",
        "expected_source": "static_analysis",
        "code": """a = 100
b = 0
result = a / b
print(result)
""",
    },
    {
        "name": "5. ValueError",
        "expected_error": "ValueError",
        "expected_source": "static_analysis",
        "code": """value = "hello"
number = int(value)
print(number)
""",
    },
    {
        "name": "6. IndexError (+10)",
        "expected_error": "IndexError",
        "expected_source": "static_analysis",
        "code": """numbers = [10, 20, 30]
print(numbers[10])
""",
    },
    {
        "name": "7. IndexError (-5)",
        "expected_error": "IndexError",
        "expected_source": "static_analysis",
        "code": """numbers = [10, 20, 30]
print(numbers[-5])
""",
    },
    {
        "name": "8. KeyError",
        "expected_error": "KeyError",
        "expected_source": "static_analysis",
        "code": """user_profile = {"username": "coder_99", "role": "developer"}
print(user_profile["subscription_tier"])
""",
    },
    {
        "name": "9. AttributeError",
        "expected_error": "AttributeError",
        "expected_source": "static_analysis",
        "code": """log_message = "System ready"
log_message.append(" [OK]")
print(log_message)
""",
    },
    {
        "name": "10. UnboundLocalError",
        "expected_error": "UnboundLocalError",
        "expected_source": "static_analysis",
        "code": """def calculate():
    print(total)
    total = 100

calculate()
""",
    },
    {
        "name": "11. RecursionError",
        "expected_error": "RecursionError",
        "expected_source": "static_analysis",
        "code": """def recurse():
    recurse()

recurse()
""",
    },
    {
        "name": "12. Clean Code",
        "expected_error": None,
        "expected_source": "machine_learning",
        "code": """def compute_factorial(n):
    if n <= 1:
        return 1
    return n * compute_factorial(n - 1)

print("5! =", compute_factorial(5))
""",
    },
]


def run_regression_suite():
    print("=" * 80)
    print("CODEGUARD AI — FULL 12-POINT REGRESSION TEST SUITE")
    print("=" * 80)

    passed = 0
    total = len(REGRESSION_SUITE)

    for item in REGRESSION_SUITE:
        res = predict_code(item["code"])
        pred_err = res["error_type"]
        pred_src = res["source"]

        err_match = (pred_err == item["expected_error"])
        src_match = (pred_src == item["expected_source"])

        success = err_match and src_match
        if success:
            passed += 1

        status = "✓ PASS" if success else "✗ FAIL"
        expected_str = f"{item['expected_error']} ({item['expected_source']})"
        predicted_str = f"{pred_err} ({pred_src})"

        print(f"\n{status} | {item['name']}")
        print(f"       Expected : {expected_str}")
        print(f"       Actual   : {predicted_str} | {res['confidence_label']}")
        print(f"       Message  : {res['message']}")

    print("\n" + "=" * 80)
    print(f"FINAL RESULT: {passed}/{total} Tests Passed ({passed / total * 100:.1f}%)")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    all_ok = run_regression_suite()
    sys.exit(0 if all_ok else 1)
