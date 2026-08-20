"""
CodeGuard AI - Bug Suggestion & Explanation Engine

Provides structured, explainable diagnostics, root-cause analyses,
corrective suggestions, and contextual examples for detected Python errors.
"""

from typing import Dict, Any, Optional

# ============================================================
# Error Knowledge Base
# ============================================================

ERROR_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "SyntaxError": {
        "title": "Syntax Error (Invalid Python Syntax)",
        "category": "Parsing / Syntax",
        "description": "Python's parser encountered invalid syntax that violates the language grammar rules before execution could begin.",
        "common_causes": [
            "Missing colon (':') at the end of if, for, while, def, or class headers.",
            "Unclosed quotes (strings), parentheses '()', square brackets '[]', or curly braces '{}'.",
            "Mismatched or invalid indentation levels.",
            "Using reserved Python keywords as variable names (e.g., class = 5).",
            "Using assignment '=' instead of equality '==' in conditional statements.",
        ],
        "suggestions": [
            "Inspect the line and column number reported by the parser.",
            "Verify all opening brackets and quotation marks have matching closing symbols.",
            "Ensure compound statement headers end with a colon (':').",
            "Check for proper, consistent 4-space indentation across all blocks."
        ],
        "buggy_example": "def calculate_sum(a, b)\n    return a + b",
        "fixed_example": "def calculate_sum(a, b):\n    return a + b",
    },

    "NameError": {
        "title": "Name Error (Unresolved Identifier)",
        "category": "Runtime / Scope",
        "description": "An identifier (variable, function, or module name) was referenced in a local or global scope where it has not been defined or imported.",
        "common_causes": [
            "Typo or misspelling in a variable, function, or module name.",
            "Using a variable before its assignment statement.",
            "Missing 'import' statement for a standard library or third-party module (e.g., using math.sqrt without import math).",
            "Variable defined inside a function scope being accessed outside that scope without return/global.",
        ],
        "suggestions": [
            "Verify the exact spelling and casing of the identifier.",
            "Check that the variable is initialized before the line referencing it.",
            "Ensure all required modules are imported at the top of the file.",
            "Review variable scope to ensure local variables are returned or passed as parameters."
        ],
        "buggy_example": "total = price + tax  # 'tax' was never assigned\nprint(total)",
        "fixed_example": "price = 100\ntax = 18\ntotal = price + tax\nprint(total)",
    },

    "IndexError": {
        "title": "Index Error (Sequence Subscript Out of Range)",
        "category": "Runtime / Sequence Access",
        "description": "A subscript (index) was used on a sequence (list, tuple, string) that is outside the valid range [0, len(seq) - 1] or [-len(seq), -1].",
        "common_causes": [
            "Attempting to access an element at index == len(list) (off-by-one error; Python uses 0-based indexing).",
            "Accessing an index on an empty sequence (len == 0).",
            "Loop counter exceeding sequence bounds.",
            "Assuming a multi-dimensional array exists before initializing nested dimensions."
        ],
        "suggestions": [
            "Check the length using len(sequence) before indexing.",
            "Remember that 0-based indexing means the last valid index is len(sequence) - 1.",
            "Iterate directly over elements ('for item in my_list:') instead of manual index counters.",
            "Use slice syntax or safe bounds checking."
        ],
        "buggy_example": "items = [10, 20, 30]\nprint(items[3])  # Valid indices are 0, 1, 2",
        "fixed_example": "items = [10, 20, 30]\nif len(items) > 3:\n    print(items[3])\nelse:\n    print(items[-1])  # Or handle default",
    },

    "KeyError": {
        "title": "Key Error (Missing Dictionary Key)",
        "category": "Runtime / Mapping Access",
        "description": "A dictionary key was accessed using subscript notation dict[key], but the specified key does not exist in the mapping.",
        "common_causes": [
            "Accessing a key that has not been inserted into the dictionary.",
            "Typo in key string name or mismatched data type (e.g., string '1' vs integer 1).",
            "Parsing external JSON/data where optional keys may be omitted.",
        ],
        "suggestions": [
            "Use dict.get(key, default) instead of direct indexing dict[key].",
            "Verify key existence using 'if key in my_dict:' before access.",
            "Use collections.defaultdict if default values are needed for missing keys."
        ],
        "buggy_example": "user = {'name': 'Royal'}\nprint(user['age'])",
        "fixed_example": "user = {'name': 'Royal'}\nprint(user.get('age', 'N/A'))",
    },

    "TypeError": {
        "title": "Type Error (Incompatible Data Type / Argument Mismatch)",
        "category": "Runtime / Type Incompatibility",
        "description": "An operation or function was applied to an object of an inappropriate or incompatible data type.",
        "common_causes": [
            "Concatenating a string and an integer directly (e.g., 'Score: ' + 10).",
            "Calling a non-callable object (e.g., int_val() or list[item] vs list(item)).",
            "Passing wrong number or types of arguments to a function.",
            "Performing arithmetic operations on NoneType or unsupported types."
        ],
        "suggestions": [
            "Explicitly cast data types when performing operations (e.g., str(number) or int(text)).",
            "Use f-strings for string interpolation (f'Score: {score}').",
            "Check function signatures and ensure required parameters match.",
            "Verify objects are not None before accessing their methods."
        ],
        "buggy_example": "age = 21\nmessage = 'Age: ' + age",
        "fixed_example": "age = 21\nmessage = f'Age: {age}'",
    },

    "AttributeError": {
        "title": "Attribute Error (Invalid Object Attribute / Method)",
        "category": "Runtime / Object Model",
        "description": "An attribute reference or method invocation failed because the target object does not have the specified attribute or method.",
        "common_causes": [
            "Calling a list method on a string or vice versa (e.g., str.append() instead of string concatenation).",
            "Calling a method on None (e.g., when a function returns None and you call a method on the result).",
            "Typo in method or property name.",
            "Confusing mutable in-place method return values (e.g., list.sort() returns None, so x = my_list.sort() makes x None)."
        ],
        "suggestions": [
            "Verify the datatype and available methods using type(obj) and dir(obj).",
            "Remember that in-place list methods (append, sort, reverse) return None.",
            "Check for None returns before chaining method calls.",
            "Inspect method spelling."
        ],
        "buggy_example": "text = 'hello'\ntext.append(' world')  # Strings don't have .append()",
        "fixed_example": "text = 'hello'\ntext = text + ' world'  # Or text += ' world'",
    },

    "ZeroDivisionError": {
        "title": "Zero Division Error (Division by Zero)",
        "category": "Runtime / Arithmetic",
        "description": "An arithmetic operation attempted to divide, floor-divide, or take the modulo of a value by zero.",
        "common_causes": [
            "A denominator variable contains zero.",
            "A literal zero was used as the divisor.",
            "A calculation produced zero before a division operation."
        ],
        "suggestions": [
            "Validate that the denominator is non-zero before performing the operation.",
            "Handle zero-valued input explicitly with a conditional or try-except ZeroDivisionError block.",
            "Check calculations that produce the denominator before division."
        ],
        "buggy_example": "denominator = 0\nresult = 100 / denominator",
        "fixed_example": "denominator = 0\nif denominator != 0:\n    result = 100 / denominator\nelse:\n    result = 0  # Handle the zero case explicitly",
    },

    "ValueError": {
        "title": "Value Error (Invalid Value with Correct Type)",
        "category": "Runtime / Value Domain",
        "description": "A function or operation received an argument that has the correct data type but an inappropriate or invalid value.",
        "common_causes": [
            "Passing a non-numeric string to int() or float() (e.g., int('abc')).",
            "Unpacking a sequence with mismatched number of variables (e.g., a, b = [1, 2, 3]).",
            "Using math functions with domain errors (e.g., math.sqrt(-1)).",
            "Calling list.remove(x) or list.index(x) when x is not in the list."
        ],
        "suggestions": [
            "Validate string format using .isdigit() or regex before type conversion.",
            "Wrap conversions in try-except ValueError blocks when handling user input.",
            "Ensure the number of unpacking targets matches sequence length exactly.",
            "Check values before performing domain-restricted operations."
        ],
        "buggy_example": "user_input = 'twenty'\nage = int(user_input)",
        "fixed_example": "user_input = 'twenty'\ntry:\n    age = int(user_input)\nexcept ValueError:\n    age = 0  # Fallback",
    },

    "UnboundLocalError": {
        "title": "Unbound Local Error (Local Variable Referenced Before Assignment)",
        "category": "Runtime / Scope Resolution",
        "description": "A local variable was referenced inside a function or scope before it had a value bound to it.",
        "common_causes": [
            "Reading a global variable and assigning to it in the same function without declaring 'global'.",
            "Referencing a variable in an inner block before its assignment in a conditional branch.",
            "Shadowing a variable inside a loop or function.",
        ],
        "suggestions": [
            "Ensure variables are assigned before being read inside function scopes.",
            "If intending to modify an outer/global variable, declare 'global var_name' or 'nonlocal var_name'.",
            "Pass values as explicit function arguments and return updated values."
        ],
        "buggy_example": "counter = 0\ndef increment():\n    counter += 1  # Treats counter as local but reads before assignment\nincrement()",
        "fixed_example": "counter = 0\ndef increment():\n    global counter\n    counter += 1\nincrement()",
    },

    "RecursionError": {
        "title": "Recursion Error (Maximum Recursion Depth Exceeded)",
        "category": "Runtime / Call Stack",
        "description": "A function called itself recursively until the Python interpreter call stack exceeded its maximum depth limit.",
        "common_causes": [
            "Missing base case in recursive function definition.",
            "Base case condition is never satisfied due to incorrect state updates (e.g., n + 1 instead of n - 1).",
            "Mutual infinite recursion between two functions calling each other."
        ],
        "suggestions": [
            "Verify that every recursive function has at least one reachable terminating base case.",
            "Ensure recursive arguments monotonically progress towards the base condition.",
            "Consider converting deeply recursive algorithms to iterative loops with explicit data structures."
        ],
        "buggy_example": "def countdown(n):\n    print(n)\n    return countdown(n)  # Missing base case",
        "fixed_example": "def countdown(n):\n    if n <= 0:\n        return\n    print(n)\n    return countdown(n - 1)",
    },

    "EOFError": {
        "title": "EOF Error (Unexpected End of File / Input Stream)",
        "category": "Runtime / I/O Stream",
        "description": "The input() function reached the end of the input stream (EOF) without reading any data.",
        "common_causes": [
            "Calling input() repeatedly when the test runner or standard input stream has no more lines to provide.",
            "Reading from closed pipes or redirected input files.",
            "Mismatched number of input() calls compared to provided input lines."
        ],
        "suggestions": [
            "Wrap input() calls inside a try-except EOFError block when reading variable-length input.",
            "Use sys.stdin.read().split() to read all available tokens at once.",
            "Check whether input lines are available before prompting for input."
        ],
        "buggy_example": "while True:\n    line = input()  # Will raise EOFError when stream ends",
        "fixed_example": "import sys\nfor line in sys.stdin:\n    process(line.strip())",
    },

    "No error": {
        "title": "Clean Code (No Error Detected)",
        "category": "Validated",
        "description": "No syntax violations or common runtime exceptions were identified in the submitted code.",
        "common_causes": [
            "Code syntax is valid and follows standard execution patterns.",
        ],
        "suggestions": [
            "Ensure all edge cases and boundary conditions are covered with unit tests.",
            "Maintain clean code practices (PEP 8 style, meaningful variable names, docstrings).",
            "Consider time and space complexity optimizations for larger inputs."
        ],
        "buggy_example": "# N/A - Code is clean",
        "fixed_example": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))",
    },
}


# ============================================================
# Suggestion Retrieval Helper
# ============================================================

def get_suggestion(error_type: str, custom_message: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve structured suggestions, root cause, and examples for a given error type.
    """
    error_info = ERROR_KNOWLEDGE_BASE.get(
        error_type,
        {
            "title": f"{error_type} Detected",
            "category": "General Error",
            "description": f"An issue associated with {error_type} was identified in the code.",
            "common_causes": [f"Potential logical or runtime exception matching {error_type}."],
            "suggestions": ["Review the code logic, variable assignments, and type compatibility."],
            "buggy_example": "# Example not available",
            "fixed_example": "# Example not available",
        }
    )

    result = dict(error_info)
    if custom_message:
        result["specific_message"] = custom_message

    return result


if __name__ == "__main__":
    for err in ["SyntaxError", "NameError", "IndexError", "No error"]:
        info = get_suggestion(err)
        print("=" * 60)
        print(f"Error: {err}")
        print(f"Title: {info['title']}")
        print(f"Category: {info['category']}")
        print(f"Description: {info['description']}")
        print(f"Sample Suggestion: {info['suggestions'][0]}")