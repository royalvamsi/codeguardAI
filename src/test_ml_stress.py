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
    "EOFError (input loop)": """
while True:
    line = input()
    print(line)
""",

    "Dynamic RecursionError": """
def countdown(n):
    return countdown(n - 1)


print(countdown(5))
""",

    "Dynamic TypeError": """
def combine(a, b):
    return a + b


print(combine("hello", 10))
""",

    "Dynamic ValueError": """
def parse_user_input(text):
    return int(text)


print(parse_user_input("not_a_number"))
""",

    "Dynamic ZeroDivisionError": """
def divide(a, b):
    return a / b


print(divide(10, 0))
""",

    "Dynamic KeyError": """
def get_email(user):
    return user["email"]


data = {"name": "Royal"}
print(get_email(data))
""",

    "Dynamic AttributeError": """
class User:
    def __init__(self):
        self.name = "Royal"


user = User()
print(user.age.upper())
""",

    "Dynamic IndexError": """
def get_item(items, position):
    return items[position]


values = [10, 20, 30]
print(get_item(values, 10))
""",

    "UnboundLocal in branch": """
def compute(flag):
    if flag:
        val = 42
    return val + 1


print(compute(False))
""",

    "Clean function": """
def calculate(a, b):
    return (a + b) * 2


print(calculate(10, 20))
""",

    "Clean class": """
class Student:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello {self.name}"


student = Student("Royal")
print(student.greet())
""",

    "Clean loop": """
total = 0

for number in range(1, 6):
    total += number

print(total)
""",

    "Clean dictionary processing": """
scores = {"math": 90, "science": 85}
total = sum(scores.values())
print("Average:", total / len(scores))
""",

    "Clean list comprehension": """
numbers = [1, 2, 3, 4, 5]
squares = [x * x for x in numbers if x % 2 == 0]
print(squares)
""",

    "Clean recursion with base case": """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


print(fib(6))
""",
}


def main():
    print("=" * 90)
    print("CODEGUARD AI — ML / UNSEEN STRESS TEST")
    print("=" * 90)

    for name, code in TEST_CASES.items():
        result = predict_code(code)

        print(f"\n{name}")
        print("-" * 70)
        print(f"Prediction : {result['error_type']}")
        print(f"Confidence : {result['confidence_label']}")
        print(f"Method     : {result['method_label']}")
        print(f"Message    : {result['message']}")


if __name__ == "__main__":
    main()
