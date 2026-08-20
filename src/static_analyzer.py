"""
CodeGuard AI - Static Python Code Analyzer

High-confidence deterministic checks for common Python errors.
This layer intentionally handles obvious cases before the ML classifier:
SyntaxError, NameError, TypeError, ValueError, ZeroDivisionError,
IndexError, KeyError, and AttributeError.
"""

import ast
import builtins
import re
import sys
from typing import Any, Dict, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def result(
    error_type: Optional[str] = None,
    message: str = "",
    suggestion: str = "",
    confidence: float = 0.0,
    source: str = "static_analysis",
) -> Dict[str, Any]:
    return {
        "error_type": error_type,
        "message": message,
        "suggestion": suggestion,
        "confidence": confidence,
        "source": source,
    }


# ============================================================
# 1. Syntax
# ============================================================

def check_syntax(code: str):
    try:
        return ast.parse(code), None
    except SyntaxError as exc:
        return None, result(
            "SyntaxError",
            f"Syntax error at line {exc.lineno}, column {exc.offset}: {exc.msg}",
            "Check the indicated line for missing colons, brackets, parentheses, quotes, commas, or invalid indentation.",
            1.0,
        )


# ============================================================
# 2. NameError
# ============================================================

def check_name_errors(tree):
    defined_names = set(dir(builtins))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined_names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_names.add(node.name)
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                defined_names.add(arg.arg)
            if node.args.vararg:
                defined_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined_names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            defined_names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    defined_names.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined_names:
                return result(
                    "NameError",
                    f"Identifier '{node.id}' is referenced but is not defined or imported.",
                    f"Define or initialize '{node.id}' before referencing it and verify its spelling.",
                    0.95,
                )
    return None


# ============================================================
# Helpers for literal/type/value inference
# ============================================================

def _literal_type(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "str"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "NoneType"
    return None


def _const_value(node):
    return node.value if isinstance(node, ast.Constant) else None


def _infer_literal_type(node, var_types):
    if isinstance(node, ast.Constant):
        return _literal_type(node.value)
    if isinstance(node, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.Set):
        return "set"
    if isinstance(node, ast.Name):
        return var_types.get(node.id)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return {
                "str": "str", "int": "int", "float": "float",
                "list": "list", "tuple": "tuple",
                "dict": "dict", "set": "set"
            }.get(node.func.id)
    return None


def _collect_var_types_and_values(tree):
    var_types = {}
    var_values = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            inferred_type = _infer_literal_type(node.value, var_types)
            value = _const_value(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if inferred_type:
                        var_types[target.id] = inferred_type
                    if isinstance(node.value, ast.Constant):
                        var_values[target.id] = value

    return var_types, var_values


# ============================================================
# 3. ZeroDivisionError
# ============================================================

def check_zero_division_errors(tree):
    _, var_values = _collect_var_types_and_values(tree)

    def zero_value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value == 0
        if isinstance(node, ast.Name) and node.id in var_values:
            value = var_values[node.id]
            return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if zero_value(node.right):
                op_name = {
                    ast.Div: "/",
                    ast.FloorDiv: "//",
                    ast.Mod: "%",
                }.get(type(node.op), "division")
                return result(
                    "ZeroDivisionError",
                    f"The right-hand operand of '{op_name}' evaluates to zero.",
                    "Ensure the denominator/divisor is non-zero before performing the operation.",
                    0.99,
                )
    return None


# ============================================================
# 4. ValueError
# ============================================================

def check_obvious_value_errors(tree):
    # Explicit conversion of a known invalid literal.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = node.func.id
            if func in {"int", "float"} and node.args:
                arg = node.args[0]
                string_value = None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    string_value = arg.value
                elif isinstance(arg, ast.Name):
                    for assignment in ast.walk(tree):
                        if (
                            isinstance(assignment, ast.Assign)
                            and isinstance(assignment.value, ast.Constant)
                            and isinstance(assignment.value.value, str)
                            and any(
                                isinstance(target, ast.Name) and target.id == arg.id
                                for target in assignment.targets
                            )
                        ):
                            string_value = assignment.value.value
                            break

                if string_value is not None:
                    text = string_value.strip()
                    valid = False
                    try:
                        if func == "int":
                            int(text)
                        else:
                            float(text)
                        valid = True
                    except (ValueError, TypeError):
                        valid = False

                    if not valid:
                        return result(
                            "ValueError",
                            f"{func}() cannot convert the string value {string_value!r} to a numeric value.",
                            f"Validate the input before calling {func}() or provide a valid numeric string.",
                            0.99,
                        )

        # list.index(x) / list.remove(x) with a known absent literal
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"index", "remove"} and node.args:
                receiver = node.func.value
                needle = node.args[0]
                if isinstance(receiver, ast.List):
                    values = [
                        x.value for x in receiver.elts
                        if isinstance(x, ast.Constant)
                    ]
                    if isinstance(needle, ast.Constant) and needle.value not in values:
                        return result(
                            "ValueError",
                            f"Value {needle.value!r} is not present in the list.",
                            f"Check membership before calling .{node.func.attr}() or handle ValueError with try/except.",
                            0.98,
                        )
                elif isinstance(receiver, ast.Name):
                    # Find simple list assignments for this variable.
                    for parent in ast.walk(tree):
                        if (
                            isinstance(parent, ast.Assign)
                            and isinstance(parent.value, ast.List)
                            and any(isinstance(t, ast.Name) and t.id == receiver.id for t in parent.targets)
                            and isinstance(needle, ast.Constant)
                        ):
                            values = [
                                x.value for x in parent.value.elts
                                if isinstance(x, ast.Constant)
                            ]
                            if needle.value not in values:
                                return result(
                                    "ValueError",
                                    f"Value {needle.value!r} is not present in list '{receiver.id}'.",
                                    f"Check membership before calling .{node.func.attr}() or handle ValueError with try/except.",
                                    0.97,
                                )

        # Unpacking a literal sequence into the wrong number of variables.
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List)):
                    if len(target.elts) != len(node.value.elts):
                        return result(
                            "ValueError",
                            f"Cannot unpack {len(node.value.elts)} values into {len(target.elts)} variables.",
                            "Make the number of unpacking targets match the number of values, or use starred unpacking.",
                            0.99,
                        )

    return None


# ============================================================
# 5. TypeError
# ============================================================

def check_obvious_type_errors(tree):
    var_types, _ = _collect_var_types_and_values(tree)

    def get_type(node):
        return _infer_literal_type(node, var_types)

    for node in ast.walk(tree):
        # Binary operations
        if isinstance(node, ast.BinOp):
            left_type = get_type(node.left)
            right_type = get_type(node.right)

            if left_type and right_type:
                if isinstance(node.op, ast.Add):
                    incompatible = (
                        (left_type == "str" and right_type in {"int", "float", "list", "dict", "set"})
                        or (right_type == "str" and left_type in {"int", "float", "list", "dict", "set"})
                        or (left_type == "list" and right_type not in {"list"})
                        or (right_type == "list" and left_type not in {"list"})
                    )
                    if incompatible:
                        return result(
                            "TypeError",
                            f"Unsupported '+' operation between '{left_type}' and '{right_type}'.",
                            "Convert the operands to compatible types before using '+'.",
                            0.98,
                        )

                elif isinstance(node.op, (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                    if left_type == "str" or right_type == "str":
                        return result(
                            "TypeError",
                            f"Unsupported arithmetic operation between '{left_type}' and '{right_type}'.",
                            "Ensure arithmetic operands are numeric (int/float).",
                            0.98,
                        )

        # Calling an object that is statically known to be non-callable.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            t = var_types.get(node.func.id)
            if t in {"int", "float", "str", "list", "tuple", "dict", "set", "bool", "NoneType"}:
                return result(
                    "TypeError",
                    f"Object '{node.func.id}' is a {t} value and cannot be called like a function.",
                    f"Remove the parentheses after '{node.func.id}' or replace it with a callable function.",
                    0.97,
                )

    return None


# ============================================================
# 6. IndexError - literal and dynamic range(len(seq) + k)
# ============================================================

def check_literal_index_errors(tree):
    """Detect out-of-range literal indexes, including negative indexes."""

    literal_lengths = {}

    # Collect lengths of literal lists/tuples
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    literal_lengths[target.id] = len(node.value.elts)

    def get_integer_index(node):
        """Extract a literal integer, including -5 and +5."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                return node.value

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                if (
                    isinstance(node.operand, ast.Constant)
                    and isinstance(node.operand.value, int)
                    and not isinstance(node.operand.value, bool)
                ):
                    return -node.operand.value

            if isinstance(node.op, ast.UAdd):
                if (
                    isinstance(node.operand, ast.Constant)
                    and isinstance(node.operand.value, int)
                    and not isinstance(node.operand.value, bool)
                ):
                    return node.operand.value

        return None

    # Check literal indexes
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue

        index = get_integer_index(node.slice)

        if index is None:
            continue

        # Direct literal list/tuple
        if isinstance(node.value, (ast.List, ast.Tuple)):
            length = len(node.value.elts)

            if not (-length <= index < length):
                return result(
                    "IndexError",
                    f"Index {index} is outside the valid range for literal sequence of length {length}.",
                    "Use a valid index between -len(sequence) and len(sequence)-1.",
                    0.98,
                )

        # Named list/tuple previously detected
        elif isinstance(node.value, ast.Name):
            sequence_name = node.value.id

            if sequence_name in literal_lengths:
                length = literal_lengths[sequence_name]

                if not (-length <= index < length):
                    return result(
                        "IndexError",
                        f"Index {index} is outside valid range for "
                        f"'{sequence_name}' (length {length}).",
                        "Check the sequence length before accessing this index.",
                        0.98,
                    )

    return None


def check_dynamic_index_errors(tree):
    """
    Detect a high-confidence off-by-one pattern such as:

        for i in range(len(numbers) + 1):
            total += numbers[i]

    The loop can reach i == len(numbers), while the last valid index
    is len(numbers) - 1.
    """
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For):
            continue

        if not isinstance(loop.target, ast.Name):
            continue

        index_name = loop.target.id
        iterator = loop.iter

        if not (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range"
            and len(iterator.args) == 1
        ):
            continue

        arg = iterator.args[0]
        sequence_name = None

        if (
            isinstance(arg, ast.BinOp)
            and isinstance(arg.op, ast.Add)
            and isinstance(arg.left, ast.Call)
            and isinstance(arg.left.func, ast.Name)
            and arg.left.func.id == "len"
            and len(arg.left.args) == 1
            and isinstance(arg.left.args[0], ast.Name)
            and isinstance(arg.right, ast.Constant)
            and isinstance(arg.right.value, int)
            and arg.right.value > 0
        ):
            sequence_name = arg.left.args[0].id

        if sequence_name is None:
            continue

        for subscript in ast.walk(loop):
            if not isinstance(subscript, ast.Subscript):
                continue
            if not (
                isinstance(subscript.value, ast.Name)
                and subscript.value.id == sequence_name
                and isinstance(subscript.slice, ast.Name)
                and subscript.slice.id == index_name
            ):
                continue

            return result(
                "IndexError",
                f"Loop index '{index_name}' can reach len({sequence_name}), which is outside the valid index range.",
                f"Use range(len({sequence_name})) or iterate directly over the elements to avoid the off-by-one access.",
                0.99,
            )

    return None


# ============================================================
# 7. KeyError
# ============================================================

def check_literal_key_errors(tree):
    literal_dicts = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
                    keys = {
                        k.value for k in node.value.keys
                        if isinstance(k, ast.Constant)
                    }
                    literal_dicts[target.id] = keys

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = node.slice.value

            if isinstance(node.value, ast.Dict):
                keys = [
                    k.value for k in node.value.keys
                    if isinstance(k, ast.Constant)
                ]
                if key not in keys:
                    return result(
                        "KeyError",
                        f"Key {key!r} does not exist in the literal dictionary.",
                        "Check whether the key exists before accessing it. Consider using dict.get().",
                        0.98,
                    )

            elif isinstance(node.value, ast.Name) and node.value.id in literal_dicts:
                if key not in literal_dicts[node.value.id]:
                    return result(
                        "KeyError",
                        f"Key {key!r} does not exist in dictionary '{node.value.id}'.",
                        f"Use '{node.value.id}.get({key!r})' or check membership before direct access.",
                        0.95,
                    )

    return None


# ============================================================
# 8. AttributeError
# ============================================================

COMMON_ATTRIBUTE_ERRORS = {
    "str": {"append", "extend", "remove", "pop", "sort", "reverse"},
    "int": {"append", "extend", "lower", "upper", "split", "strip"},
    "float": {"append", "extend", "lower", "upper", "split", "strip"},
    "list": {"lower", "upper", "split", "strip", "capitalize", "startswith", "endswith"},
    "dict": {"append", "extend", "lower", "upper", "split", "sort"},
    "set": {"append", "extend", "lower", "upper", "split"},
}


def check_obvious_attribute_errors(tree):
    var_types, _ = _collect_var_types_and_values(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue

        receiver_type = _infer_literal_type(node.func.value, var_types)
        method = node.func.attr

        if receiver_type in COMMON_ATTRIBUTE_ERRORS and method in COMMON_ATTRIBUTE_ERRORS[receiver_type]:
            return result(
                "AttributeError",
                f"'{receiver_type}' object does not provide the '{method}()' method.",
                f"Check the datatype of the object before calling '.{method}()' and use a method supported by {receiver_type}.",
                0.97,
            )

    return None


# ============================================================
# 9. UnboundLocalError
# ============================================================

def check_unbound_local_errors(tree):
    """
    Detect local variables in functions that are referenced before assignment.
    In Python, if a variable is assigned anywhere in a function, it is treated as
    local. If referenced before assignment, it raises UnboundLocalError.
    """
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Find all variables assigned locally in this function
        local_assigned = set()
        global_or_nonlocal = set()

        for child in ast.walk(function):
            if isinstance(child, (ast.Global, ast.Nonlocal)):
                global_or_nonlocal.update(child.names)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                local_assigned.add(child.id)

        # Collect parameter names
        param_names = set()
        for arg in function.args.posonlyargs + function.args.args + function.args.kwonlyargs:
            param_names.add(arg.arg)
        if function.args.vararg:
            param_names.add(function.args.vararg.arg)
        if function.args.kwarg:
            param_names.add(function.args.kwarg.arg)

        local_names = (local_assigned - global_or_nonlocal) - param_names
        if not local_names:
            continue

        initialized = set(param_names)

        def check_reads(node, initialized_names):
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if child.id in local_names and child.id not in initialized_names:
                        return result(
                            "UnboundLocalError",
                            f"Local variable '{child.id}' is referenced before assignment.",
                            f"Assign or initialize '{child.id}' before referencing it, or declare 'global {child.id}'.",
                            0.98,
                        )
            return None

        def process_statements(stmts, initialized_names):
            for stmt in stmts:
                # ================================================
                # SIMPLE ASSIGNMENT (Assign, AnnAssign)
                # ================================================
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    # Check RHS expression first
                    if stmt.value:
                        found = check_reads(stmt.value, initialized_names)
                        if found:
                            return found

                    if isinstance(stmt, ast.Assign):
                        targets = stmt.targets
                    else:
                        targets = [stmt.target]

                    for target in targets:
                        for child in ast.walk(target):
                            if isinstance(child, ast.Name):
                                initialized_names.add(child.id)

                    continue

                # ================================================
                # AUGMENTED ASSIGNMENT (AugAssign: x += 1)
                # ================================================
                if isinstance(stmt, ast.AugAssign):
                    # x += 1 requires x to already exist
                    if isinstance(stmt.target, ast.Name):
                        if (
                            stmt.target.id in local_names
                            and stmt.target.id not in initialized_names
                        ):
                            return result(
                                "UnboundLocalError",
                                f"Local variable '{stmt.target.id}' is referenced before assignment.",
                                f"Initialize '{stmt.target.id}' before using '+=' or another augmented assignment.",
                                0.97,
                            )

                    found = check_reads(stmt.value, initialized_names)
                    if found:
                        return found

                    if isinstance(stmt.target, ast.Name):
                        initialized_names.add(stmt.target.id)

                    continue

                # ================================================
                # FOR LOOPS
                # ================================================
                if isinstance(stmt, (ast.For, ast.AsyncFor)):
                    # Check the iterable first
                    found = check_reads(stmt.iter, initialized_names)
                    if found:
                        return found

                    # Target variables become initialized in loop
                    for child in ast.walk(stmt.target):
                        if isinstance(child, ast.Name):
                            initialized_names.add(child.id)

                    # Process loop body
                    found = process_statements(stmt.body, initialized_names)
                    if found:
                        return found
                    continue

                # ================================================
                # IF STATEMENTS
                # ================================================
                if isinstance(stmt, ast.If):
                    found = check_reads(stmt.test, initialized_names)
                    if found:
                        return found

                    body_init = set(initialized_names)
                    found = process_statements(stmt.body, body_init)
                    if found:
                        return found

                    if stmt.orelse:
                        orelse_init = set(initialized_names)
                        found = process_statements(stmt.orelse, orelse_init)
                        if found:
                            return found
                    continue

                # ================================================
                # OTHER STATEMENTS (Calls, Return, Expr, etc.)
                # ================================================
                found = check_reads(stmt, initialized_names)
                if found:
                    return found

                # Register ordinary assignments contained in the
                # statement after its reads have been checked.
                for child in ast.walk(stmt):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        initialized_names.add(child.id)

            return None

        # Analyze the function body
        found = process_statements(function.body, initialized)
        if found:
            return found

    return None


# ============================================================
# 10. RecursionError
# ============================================================

def check_obvious_recursion_errors(tree):
    """
    Detect simple unconditional self-recursion with no obvious
    terminating condition.

    Example:
        def recurse():
            recurse()

        recurse()
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        function_name = node.name

        # Look for a direct call to the same function.
        self_calls = [
            child
            for child in ast.walk(node)
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == function_name
            )
        ]

        if not self_calls:
            continue

        # If the function contains a return/conditional structure,
        # avoid claiming definite infinite recursion.
        has_conditional = any(
            isinstance(child, (ast.If, ast.IfExp, ast.Match))
            for child in ast.walk(node)
        )

        if has_conditional:
            continue

        # A direct self-call with no conditional/base case is
        # a high-confidence infinite recursion pattern.
        return result(
            "RecursionError",
            f"Function '{function_name}()' calls itself without an apparent base case.",
            f"Add a terminating/base condition to '{function_name}()' so the recursive calls eventually stop.",
            0.95,
        )

    return None


# ============================================================
# Main static analysis pipeline
# ============================================================

def analyze_code(code: str):
    if not isinstance(code, str):
        raise TypeError("Code must be a string.")
    if not code.strip():
        raise ValueError("Code cannot be empty.")

    tree, syntax_result = check_syntax(code)
    if syntax_result:
        return syntax_result

    checks = (
        check_name_errors,
        check_zero_division_errors,
        check_obvious_value_errors,
        check_obvious_type_errors,
        check_literal_index_errors,
        check_dynamic_index_errors,
        check_literal_key_errors,
        check_obvious_attribute_errors,
        check_unbound_local_errors,
        check_obvious_recursion_errors,
    )

    for check in checks:
        analysis = check(tree)
        if analysis:
            return analysis

    return result(
        error_type=None,
        message="No high-confidence static error was detected.",
        suggestion="The code will be evaluated by the Machine Learning classification pipeline.",
        confidence=0.0,
    )


if __name__ == "__main__":
    test_cases = {
        "ZeroDivisionError": "x = 0\nresult = 100 / x\nprint(result)",
        "TypeError": 'number = 10\ntext = "hello"\nresult = number + text',
        "ValueError": 'value = "hello"\nnumber = int(value)',
        "Dynamic IndexError": "numbers = [1, 2, 3]\nfor i in range(len(numbers) + 1):\n    print(numbers[i])",
        "NameError": "print(undefined_variable)",
        "KeyError": 'data = {"name": "Royal"}\nprint(data["age"])',
        "AttributeError": 'text = "hello"\ntext.append("world")',
        "Clean": "def add(a, b):\n    return a + b\nprint(add(10, 20))",
    }

    print("=" * 70)
    print("CODEGUARD AI - STATIC ANALYZER TEST")
    print("=" * 70)

    for name, code in test_cases.items():
        res = analyze_code(code)
        print(f"\n{name}: {res['error_type']}")
        print(res["message"])
        print("-" * 70)