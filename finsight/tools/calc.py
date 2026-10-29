"""Safe arithmetic. The model writes the formula; Python does the maths.

    calculate("(a - b) / b * 100", {"a": 51217, "b": 46710})  -> 9.649...
"""
from __future__ import annotations

import ast
import re
import math
import operator as op

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
        ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
_UN = {ast.USub: op.neg, ast.UAdd: op.pos}
_FUN = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log}


class CalcError(ValueError):
    pass


def calculate(expression: str, variables: dict[str, float] | None = None) -> float:
    variables = {k: float(v) for k, v in (variables or {}).items()}
    expr = re.sub(r"(?<=\d),(?=\d{3})", "", expression.replace("^", "**"))
    if len(expr) > 300:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"invalid expression: {e.msg}") from e

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id not in variables:
                raise CalcError(f"unknown variable '{n.id}'; provided: {sorted(variables)}")
            return variables[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 10:
                raise CalcError("exponent too large")
            try:
                return _BIN[type(n.op)](ev(n.left), ev(n.right))
            except ZeroDivisionError as e:
                raise CalcError("division by zero") from e
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN and not n.keywords:
            args = [ev(a) for a in n.args]
            if n.func.id == "round" and len(args) == 2:
                args[1] = int(args[1])
            return float(_FUN[n.func.id](*args))
        raise CalcError(f"unsupported syntax: {ast.dump(n)[:60]}")

    return ev(tree)
