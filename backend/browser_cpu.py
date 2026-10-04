"""Lower the existing conservative CPU analyzer into a small browser interpreter IR.

Parse a single function with the CPU analyzer's AST read/write checks. Accept
independent numeric maps/comprehensions and sum/min/max/append reductions; refuse
loop-carried dependencies, unknown calls and effects. Lower each iteration to
numeric IR, cut the input into score-weighted contiguous ranges, and return
float64 values for ordered assembly/reduction. Literals, inputs and arithmetic
are bounded to the browser's supported numeric range. General Python objects,
bit operations and inter-region dependencies are unsupported. Uploaded Python
is never executed, and the interpreter cannot access imports, globals, files,
network operations or dynamic evaluation.
"""

import ast
import math
from typing import Any

from .cpu_analyzer import analyze
from .models import TypedArray

OPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.FloorDiv: "floor_div",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}
CALLS = {"abs", "min", "max", "int", "float", "bool", "pow"}


def expression(node: ast.AST) -> list[Any]:
    """Lower a checked numeric expression to the browser instruction tree."""
    if (
        isinstance(node, ast.Constant)
        and type(node.value) in (int, float, bool)
        and math.isfinite(node.value)
        and abs(node.value) <= 2**53 - 1
    ):
        return ["constant", node.value]
    if isinstance(node, ast.Name):
        return ["name", node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return [OPS[type(node.op)], expression(node.left), expression(node.right)]
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Not)
    ):
        return [
            {ast.UAdd: "positive", ast.USub: "negative", ast.Not: "not"}[type(node.op)],
            expression(node.operand),
        ]
    if isinstance(node, ast.IfExp):
        return [
            "if",
            expression(node.test),
            expression(node.body),
            expression(node.orelse),
        ]
    if isinstance(node, ast.BoolOp):
        return [
            "and" if isinstance(node.op, ast.And) else "or",
            *map(expression, node.values),
        ]
    if isinstance(node, ast.Compare) and all(type(op) in OPS for op in node.ops):
        return [
            "compare",
            [OPS[type(op)] for op in node.ops],
            *map(expression, [node.left, *node.comparators]),
        ]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in CALLS
        and not node.keywords
        and 1 <= len(node.args) <= 8
    ):
        name, count = node.func.id, len(node.args)
        if (
            (name in {"abs", "int", "float", "bool"} and count != 1)
            or (name == "pow" and count != 2)
            or (name in {"min", "max"} and count < 2)
        ):
            raise ValueError(f"Unsupported numeric call signature: {name}")
        return ["call", name, *map(expression, node.args)]
    raise ValueError(f"Browser CPU execution does not support {ast.unparse(node)}")


def lower(source: str) -> dict[str, Any]:
    """Validate one independent function and lower its per-item numeric work."""
    report = analyze(source)
    if (
        not report.splittable
        or not report.generated_source
        or report.reduction not in ("append", "sum", "min", "max")
    ):
        raise ValueError(
            "; ".join(report.reasons)
            or "Select an independent numeric loop or comprehension"
        )
    if report.iterable_source not in (
        report.parameter,
        f"range(len({report.parameter}))",
        f"range({report.parameter})",
    ):
        raise ValueError(
            "CPU loop must iterate over the input array, range(len(input)), or range(count)"
        )
    if (
        report.initial is not None
        and not isinstance(report.initial, list)
        and (not math.isfinite(report.initial) or abs(report.initial) > 2**53 - 1)
    ):
        raise ValueError(
            "CPU initial value must be a finite, exactly representable number"
        )
    fn = ast.parse(report.generated_source).body[0]
    steps = []
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            steps.append(["set", stmt.targets[0].id, expression(stmt.value)])
        elif isinstance(stmt, ast.Return):
            steps.append(["return", expression(stmt.value)])
        else:
            raise ValueError("Unsupported CPU statement")
    return {
        "uses_count": report.iterable_source == f"range({report.parameter})",
        "arguments": [a.arg for a in fn.args.args],
        "indexed": report.iterable_source != report.parameter,
        "steps": steps,
        "reduction": report.reduction,
        "initial": report.initial,
    }


def plan(
    program: dict[str, Any],
    values: list[float] | None,
    count: int,
    partitions: list[tuple[int, int, str | None]],
) -> dict[str, Any]:
    """Package weighted ranges and ordered float64 output for browser workers."""
    total = len(values) if values is not None else count
    if values is not None and any(
        not math.isfinite(v) or abs(v) > 2**53 - 1 for v in values
    ):
        raise ValueError("CPU input exceeds the supported numeric range")
    if len(partitions) > 2048:
        raise ValueError("Maximum 2048 CPU chunks")
    chunks = []
    for offset, size, worker in partitions:
        chunks.append(
            {
                "chunk_id": f"cpu-{offset}",
                "offset": offset,
                "count": size,
                "byte_length": size * 8,
                "preferred_worker": worker,
                "assignment": {
                    "kind": "cpu",
                    "cpu_program": program,
                    "offset": offset,
                    "count": size,
                    "output_format": "f64",
                    "input": TypedArray.encode(
                        values[offset : offset + size]
                    ).model_dump()
                    if values is not None
                    else None,
                },
            }
        )
    return {
        "chunks": chunks,
        "size": total * 8,
        "assets": {},
        "output_format": "f64",
        "output_shape": [total],
        "reduction": program["reduction"],
        "initial": program["initial"],
    }
