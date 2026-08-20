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

TEST_CASES = {
    "ZeroDivisionError": """
a = 100
b = 0
result = a / b
print(result)
""",

    "TypeError": """
number = 10
text = "hello"
result = number + text
print(result)
""",

    "ValueError": """
value = "hello"
number = int(value)
print(number)
""",

    "Dynamic IndexError": """
numbers = [1, 2, 3]
for i in range(len(numbers) + 1):
    print(numbers[i])
""",

    "NameError": """
print(undefined_variable)
""",

    "KeyError": """
data = {"name": "Royal"}
print(data["age"])
""",

    "AttributeError": """
text = "hello"
text.append("world")
""",

    "SyntaxError": """
def calculate(a, b)
    return a + b
""",

    "No error": """
def add(a, b):
    return a + b

result = add(10, 20)
print(result)
""",
}


def main():
    print("=" * 80)
    print("CODEGUARD AI — COMPREHENSIVE DIAGNOSTIC TEST")
    print("=" * 80)

    correct = 0
    total = len(TEST_CASES)

    for expected, code in TEST_CASES.items():
        res = predict_code(code)
        predicted = res["error_type"]
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"\n{status} Expected  : {expected}")
        print(f"  Predicted : {predicted}")
        print(f"  Method    : {res['method_label']}")
        print(f"  Reason    : {res['message']}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {correct}/{total} Tests Passed ({correct/total*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
