"""
CodeGuard AI — Focused False-Positive Diagnostic Suite

Tests 10 clean and genuine error cases while logging the decision margin:
Margin = Top Candidate Probability - 'No error' Probability
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

FP_TEST_CASES = {
    "1. Clean class (generic)": (
        """class Account:\n    def __init__(self, balance):\n        self.balance = balance\n\n    def deposit(self, amount):\n        self.balance += amount\n        return self.balance\n\nacc = Account(100)\nprint(acc.deposit(50))""",
        None
    ),
    "2. Clean recursion with base case": (
        """def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))""",
        None
    ),
    "3. Clean Student class": (
        """class Student:\n    def __init__(self, name):\n        self.name = name\n\n    def greet(self):\n        return f"Hello {self.name}"\n\nstudent = Student("Royal")\nprint(student.greet())""",
        None
    ),
    "4. Clean fib function": (
        """def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\nprint(fib(6))""",
        None
    ),
    "5. Clean accumulator loop": (
        """total = 0\nfor number in range(1, 6):\n    total += number\nprint(total)""",
        None
    ),
    "6. Clean dictionary processing": (
        """scores = {"math": 90, "science": 85}\ntotal = sum(scores.values())\nprint("Average:", total / len(scores))""",
        None
    ),
    "7. Clean list comprehension": (
        """numbers = [1, 2, 3, 4, 5]\nsquares = [x * x for x in numbers if x % 2 == 0]\nprint(squares)""",
        None
    ),
    "8. Genuine NameError": (
        """def compute():\n    return non_existent_variable + 10\n\nprint(compute())""",
        "NameError"
    ),
    "9. Genuine TypeError": (
        """count = 10\nlabel = "items"\nprint(count + label)""",
        "TypeError"
    ),
    "10. Genuine ValueError": (
        """data = "invalid_number"\nnum = int(data)\nprint(num)""",
        "ValueError"
    ),
}


def run_fp_diagnostic():
    print("=" * 90)
    print("CODEGUARD AI — FOCUSED FALSE-POSITIVE DIAGNOSTIC SUITE")
    print("=" * 90)

    for name, (code, expected_err) in FP_TEST_CASES.items():
        res = predict_code(code)
        pred_err = res["error_type"]
        scores = res.get("decision_scores", {})

        no_error_score = scores.get("No error", 0.0)
        top_err_candidate = None
        top_err_score = 0.0

        for k, v in scores.items():
            if k != "No error":
                top_err_candidate = k
                top_err_score = v
                break

        margin = top_err_score - no_error_score

        is_match = (pred_err == expected_err)
        status = "✓ PASS" if is_match else "✗ FAIL"

        print(f"\n{status} | {name}")
        print(f"       Expected   : {expected_err}")
        print(f"       Prediction : {pred_err} [{res['method_label']}]")
        print(f"       Confidence : {res['confidence_label']}")
        if res["source"] == "machine_learning":
            print(f"       Top ML Err : {top_err_candidate} ({top_err_score * 100:.1f}%) vs No-error ({no_error_score * 100:.1f}%) | Margin: {margin * 100:+.1f}%")


if __name__ == "__main__":
    run_fp_diagnostic()
