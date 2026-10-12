"""Safe parser for custom mathematical expressions."""

import ast
import math
import sympy as sp


x = sp.Symbol("x")

ALLOWED_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}

ALLOWED_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}

MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 60
MAX_EXPONENT = 10


def convert_node(node):
    """Convert only approved AST nodes into symbolic expressions."""

    if isinstance(node, ast.Constant):
        value = node.value

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Only numeric constants are allowed.")

        if not math.isfinite(value):
            raise ValueError("Constants must be finite.")

        return sp.sympify(value)

    if isinstance(node, ast.Name):
        if node.id == "x":
            return x
        raise ValueError(f"Unknown name: {node.id}")

    if isinstance(node, ast.UnaryOp):
        value = convert_node(node.operand)

        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value

        raise ValueError("Unsupported unary operator.")

    if isinstance(node, ast.BinOp):
        operation = ALLOWED_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported mathematical operator.")

        left = convert_node(node.left)
        right = convert_node(node.right)

        if isinstance(node.op, ast.Pow):
            if right.is_number is not True:
                raise ValueError("Exponents must be numeric constants.")

            exponent = float(right)

            if not math.isfinite(exponent) or abs(exponent) > MAX_EXPONENT:
                raise ValueError("Absolute exponent must not exceed 10.")

        return operation(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function call.")

        function = ALLOWED_FUNCTIONS.get(node.func.id)

        if function is None:
            raise ValueError(f"Function '{node.func.id}' is not allowed.")

        if len(node.args) != 1 or node.keywords:
            raise ValueError("Functions must have exactly one argument.")

        return function(convert_node(node.args[0]))

    raise ValueError("Unsupported expression element.")


def build_custom_function(expression):
    """Return a numerical function and its symbolic derivative."""

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Enter a mathematical expression.")

    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid mathematical syntax.") from error

    if len(list(ast.walk(tree))) > MAX_AST_NODES:
        raise ValueError("Expression is too complex.")

    parsed = convert_node(tree.body)
    derivative = sp.diff(parsed, x)

    function_impl = sp.lambdify(x, parsed, modules=["math"])
    gradient_impl = sp.lambdify(x, derivative, modules=["math"])

    def function(value):
        try:
            result = float(function_impl(float(value)))
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            raise ValueError("Function is undefined at this x value.") from error

        if not math.isfinite(result):
            raise ValueError("Function returned a non-finite result.")

        return result

    def gradient(value):
        try:
            result = float(gradient_impl(float(value)))
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            raise ValueError("Gradient is undefined at this x value.") from error

        if not math.isfinite(result):
            raise ValueError("Gradient returned a non-finite result.")

        return result

    return function, gradient