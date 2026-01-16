"""Safe parsing of numeric expressions."""

from __future__ import annotations

import ast
import math
import operator


_BIN_OPS: dict[type[ast.operator], callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def parse_number_expr(text: str) -> float:
    """Parse and evaluate a numeric expression safely."""
    if text is None or not str(text).strip():
        raise ValueError("Eingabe ist leer.")
    try:
        expr = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Ungültiger numerischer Ausdruck.") from exc

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Nur Zahlen sind erlaubt.")
            return float(value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            return _BIN_OPS[type(node.op)](left, right)
        raise ValueError(
            "Nur Zahlen und die Operatoren + - * / ( ) sind erlaubt."
        )

    try:
        result = _eval(expr)
    except ZeroDivisionError as exc:
        raise ValueError("Division durch 0 ist nicht erlaubt.") from exc

    if not math.isfinite(result):
        raise ValueError("Ergebnis ist nicht endlich.")
    return float(result)
