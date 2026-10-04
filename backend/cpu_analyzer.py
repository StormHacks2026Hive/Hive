"""Conservative Python AST read/write analysis and weighted Ray execution.

Parse one function containing an initialized sum/min/max/list accumulator, one
for loop, and its return, or a comprehension over pure scalar expressions.
Each iteration may assign local scalars and read its own input element; only
recognized final sum/min/max/append updates may escape the iteration. Captured
mutable state, globals/nonlocals, neighbor reads, unknown calls and I/O are
refused with reasons. Pure helper functions must be explicitly supplied and
independently checked from their AST. Separate if statements with pure calls
can be reported independent; elif alternatives remain mutually exclusive and
are never all executed. Generated iteration functions run in Ray tasks over
node_ranker weighted ranges, pinning remote nodes when a Ray node ID exists.
Ordered partials preserve append order; integer sums are exact, floating sums
can change rounding and require allclose. Analysis never executes source;
execution APIs are only for trusted local code, not uploaded browser jobs.
"""

from __future__ import annotations
import ast
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from .common.nodes import ComputeNode

log = logging.getLogger(__name__)
PURE = {
    "abs",
    "min",
    "max",
    "int",
    "float",
    "bool",
    "len",
    "range",
    "sum",
    "pow",
    "round",
}
IO = {"print", "open", "input", "exec", "eval", "compile", "__import__"}


@dataclass
class CPUAnalysis:
    """Decision and generated pure per-iteration source, with refusal reasons."""

    splittable: bool
    reasons: list[str]
    reduction: str | None = None
    generated_source: str | None = None
    function_name: str | None = None
    parameter: str | None = None
    initial: Any = None
    iterable_source: str | None = None
    independent_branches: list[int] = field(default_factory=list)
    branch_expressions: list[tuple[str, str]] = field(default_factory=list)


def _pure_expression(node: ast.AST, allowed: set[str], helpers: set[str]) -> list[str]:
    reasons = []
    for part in ast.walk(node):
        if isinstance(part, ast.Call):
            if isinstance(part.func, ast.Name) and part.func.id in allowed:
                reasons.append('Iteration variable called as a function: ' + part.func.id)
            if (
                not isinstance(part.func, ast.Name)
                or part.func.id not in PURE | helpers
            ):
                reasons.append(
                    "Unknown or order-dependent call/I/O: " + ast.unparse(part.func)
                )
        if (
            isinstance(part, ast.Name)
            and isinstance(part.ctx, ast.Load)
            and part.id not in allowed | PURE | helpers
        ):
            reasons.append("Captured/global read: " + part.id)
        if isinstance(part, (ast.NamedExpr, ast.Await, ast.Yield, ast.Attribute)):
            reasons.append("Effectful or unproven expression: " + type(part).__name__)
    return reasons


def analyze(source: str, pure_helpers: set[str] | None = None) -> CPUAnalysis:
    """Analyze a complete function; user declarations do not bypass AST hazards."""
    helpers = pure_helpers or set()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return CPUAnalysis(False, [str(exc)])
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return CPUAnalysis(
            False, ["Expected one function, without top-level side effects"]
        )
    fn = tree.body[0]
    if (
        fn.decorator_list
        or fn.args.vararg
        or fn.args.kwarg
        or fn.args.defaults
        or fn.args.kwonlyargs
        or len(fn.args.args) != 1
    ):
        return CPUAnalysis(
            False, ["Use one positional argument without decorators or defaults"]
        )
    parameter = fn.args.args[0].arg
    if parameter in PURE | helpers:
        return CPUAnalysis(False, ["Parameter shadows a known pure function"])
    for node in ast.walk(fn):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return CPUAnalysis(
                False, ["Writes to a global/nonlocal are not independent"]
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in IO
        ):
            return CPUAnalysis(
                False, ["Order-dependent I/O or dynamic execution: " + node.func.id]
            )
    body = [
        n
        for n in fn.body
        if not isinstance(n, ast.Expr)
        or not isinstance(n.value, ast.Constant)
        or not isinstance(n.value.value, str)
    ]
    if (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.ListComp)
    ):
        comp = body[0].value
        if (
            len(comp.generators) != 1
            or comp.generators[0].ifs
            or comp.generators[0].is_async
        ):
            return CPUAnalysis(
                False, ["Only one unfiltered comprehension generator is supported"]
            )
        generator = comp.generators[0]
        if not isinstance(generator.target, ast.Name):
            return CPUAnalysis(False, ["Comprehension unpacking unsupported"])
        target = generator.target.id
        reasons = []
        expr = copy.deepcopy(comp.elt)
        if ast.unparse(generator.iter) != parameter:
            expr = _replace_current_reads(expr, parameter, target, reasons)
        reasons += _pure_expression(expr, {target, 'hive_value'}, helpers) + _pure_expression(
            generator.iter, {parameter}, helpers
        )
        if reasons:
            return CPUAnalysis(False, reasons)
        args = target + ', hive_value' if any(isinstance(n, ast.Name) and n.id == 'hive_value' for n in ast.walk(expr)) else target
        generated = f"def hive_iteration({args}):\n    return {ast.unparse(expr)}\n"
        return CPUAnalysis(
            True,
            [],
            "append",
            generated,
            fn.name,
            parameter,
            [],
            ast.unparse(generator.iter),
        )
    branches = [n for n in body if isinstance(n, ast.If)]
    if (
        len(branches) > 1
        and len(body) == len(branches) + 2
        and isinstance(body[0], ast.Assign)
        and len(body[0].targets) == 1
        and isinstance(body[0].targets[0], ast.Name)
        and isinstance(body[0].value, ast.List)
        and not body[0].value.elts
        and isinstance(body[-1], ast.Return)
    ):
        result_name = body[0].targets[0].id
        if result_name == parameter or result_name in PURE | helpers:
            return CPUAnalysis(
                False, ["Branch accumulator shadows input or pure helper"]
            )
        expressions = []
        reasons = []
        if not isinstance(body[-1].value, ast.Name) or body[-1].value.id != result_name:
            reasons.append("Branches must return their ordered append results")
        for branch in branches:
            if (
                branch.orelse
                or len(branch.body) != 1
                or not isinstance(branch.body[0], ast.Expr)
                or not isinstance(branch.body[0].value, ast.Call)
            ):
                reasons.append("elif/else or unproven branch effects")
                continue
            call = branch.body[0].value
            if (
                not isinstance(call.func, ast.Attribute)
                or not isinstance(call.func.value, ast.Name)
                or call.func.value.id != result_name
                or call.func.attr != "append"
                or len(call.args) != 1
                or call.keywords
            ):
                reasons.append("Branches must append one pure result")
                continue
            reasons.extend(_pure_expression(branch.test, {parameter}, helpers))
            reasons.extend(_pure_expression(call.args[0], {parameter}, helpers))
            expressions.append((ast.unparse(branch.test), ast.unparse(call.args[0])))
        if not reasons:
            return CPUAnalysis(
                True,
                [],
                reduction="branches",
                function_name=fn.name,
                parameter=parameter,
                independent_branches=[b.lineno for b in branches],
                branch_expressions=expressions,
            )
        return CPUAnalysis(False, reasons)
    if len(branches) > 1:
        independent = []
        for branch in branches:
            if (
                not branch.orelse
                and all(
                    isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    for n in branch.body
                )
                and not _pure_expression(branch, {parameter}, helpers)
            ):
                independent.append(branch.lineno)
        if independent:
            return CPUAnalysis(
                False,
                [
                    "Independent if calls found; use run_calls to preserve predicate selection"
                ],
                independent_branches=independent,
            )
    if (
        len(body) != 3
        or not isinstance(body[0], ast.Assign)
        or not isinstance(body[1], ast.For)
        or not isinstance(body[2], ast.Return)
    ):
        return CPUAnalysis(
            False,
            [
                "Expected accumulator initialization, one for loop, and return; unknown control flow refused"
            ],
        )
    init, loop, ret = body
    if (
        len(init.targets) != 1
        or not isinstance(init.targets[0], ast.Name)
        or not isinstance(loop.target, ast.Name)
        or loop.orelse
    ):
        return CPUAnalysis(
            False, ["Only scalar/list accumulator and simple loop targets supported"]
        )
    accumulator = init.targets[0].id
    target = loop.target.id
    if (
        accumulator == target
        or accumulator == parameter
        or not isinstance(ret.value, ast.Name)
        or ret.value.id != accumulator
    ):
        return CPUAnalysis(
            False, ["Return must be the accumulator with a distinct loop variable"]
        )
    # A fresh dense output with exactly one current-index write is an ordered
    # map. Normalize its full-coverage write set to append reduction.
    if (
        loop.body
        and isinstance(loop.body[-1], ast.Assign)
        and len(loop.body[-1].targets) == 1
        and isinstance(loop.body[-1].targets[0], ast.Subscript)
    ):
        write = loop.body[-1].targets[0]
        canonical_iter = f"range(len({parameter}))"
        fresh = (
            isinstance(init.value, ast.BinOp)
            and isinstance(init.value.op, ast.Mult)
            and isinstance(init.value.left, ast.List)
            and len(init.value.left.elts) == 1
            and isinstance(init.value.left.elts[0], ast.Constant)
            and ast.unparse(init.value.right) == f"len({parameter})"
        )
        own_write = (
            isinstance(write.value, ast.Name)
            and write.value.id == accumulator
            and isinstance(write.slice, ast.Name)
            and write.slice.id == target
        )
        if not fresh or not own_write or ast.unparse(loop.iter) != canonical_iter:
            return CPUAnalysis(
                False,
                [
                    "Array write set is not a fresh full-coverage current-index map; loop-carried dependencies are possible"
                ],
            )
        init.value = ast.List(elts=[], ctx=ast.Load())
        loop.body[-1] = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(accumulator, ast.Load()),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[loop.body[-1].value],
                keywords=[],
            )
        )
        ast.fix_missing_locations(tree)
        return analyze(ast.unparse(tree), helpers)
    try:
        initial = ast.literal_eval(init.value)
    except (ValueError, TypeError):
        return CPUAnalysis(False, ["Accumulator initializer must be a literal"])
    if not loop.body:
        return CPUAnalysis(False, ["Empty loop"])
    final = loop.body[-1]
    reduction = None
    expression = None
    if (
        isinstance(final, ast.AugAssign)
        and isinstance(final.target, ast.Name)
        and final.target.id == accumulator
        and isinstance(final.op, ast.Add)
    ):
        reduction = "sum"
        expression = final.value
    elif isinstance(final, ast.Expr) and isinstance(final.value, ast.Call):
        call = final.value
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == accumulator
            and call.func.attr == "append"
            and len(call.args) == 1
            and not call.keywords
        ):
            reduction = "append"
            expression = call.args[0]
    elif (
        isinstance(final, ast.Assign)
        and len(final.targets) == 1
        and isinstance(final.targets[0], ast.Name)
        and final.targets[0].id == accumulator
    ):
        call = final.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in {"min", "max"}
            and len(call.args) == 2
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == accumulator
        ):
            reduction = call.func.id
            expression = call.args[1]
    if expression is None:
        return CPUAnalysis(False, ["Unrecognized reduction or loop-carried dependency"])
    if reduction == "append" and initial != []:
        return CPUAnalysis(False, ["Append accumulator must start empty"])
    if reduction != "append" and type(initial) not in (int, float):
        return CPUAnalysis(
            False, ["Numeric reduction requires a numeric initial value"]
        )
    allowed = {target}
    reasons = []
    iteration = []
    indexed = not isinstance(loop.iter, ast.Name) or loop.iter.id != parameter
    for statement in loop.body[:-1]:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            reasons.append(
                "Mutation, I/O, nested control flow or loop-carried write unsupported"
            )
            continue
        local = statement.targets[0].id
        if local in {accumulator, target, parameter}:
            reasons.append("Loop-carried dependency or input mutation: " + local)
        expr = copy.deepcopy(statement.value)
        if indexed:
            expr = _replace_current_reads(expr, parameter, target, reasons)
        reasons.extend(_pure_expression(expr, allowed | {"hive_value"}, helpers))
        allowed.add(local)
        iteration.append(ast.Assign(targets=[ast.Name(local, ast.Store())], value=expr))
    expression = copy.deepcopy(expression)
    if indexed:
        expression = _replace_current_reads(expression, parameter, target, reasons)
    reasons.extend(_pure_expression(expression, allowed | {"hive_value"}, helpers))
    reasons.extend(_pure_expression(loop.iter, {parameter}, helpers))
    if reasons:
        return CPUAnalysis(False, list(dict.fromkeys(reasons)))
    args = [ast.arg(arg=target)]
    if any(
        isinstance(n, ast.Name) and n.id == "hive_value"
        for expr in [expression, *iteration]
        for n in ast.walk(expr)
    ):
        args.append(ast.arg(arg="hive_value"))
    generated = ast.FunctionDef(
        name="hive_iteration",
        args=ast.arguments(
            posonlyargs=[], args=args, kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[*iteration, ast.Return(expression)],
        decorator_list=[],
    )
    ast.fix_missing_locations(generated)
    return CPUAnalysis(
        True,
        [],
        reduction,
        ast.unparse(generated),
        fn.name,
        parameter,
        initial,
        ast.unparse(loop.iter),
    )


def _replace_current_reads(
    expression: ast.AST, parameter: str, target: str, reasons: list[str]
) -> ast.AST:
    class Replace(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == parameter
                and isinstance(node.slice, ast.Name)
                and node.slice.id == target
            ):
                return ast.copy_location(ast.Name("hive_value", ast.Load()), node)
            reasons.append("Neighbor read or unproven subscript: " + ast.unparse(node))
            return node

    return Replace().visit(expression)


def _run_chunk(
    generated: str,
    items: list[Any],
    indexed: bool,
    values: Any,
    helpers: dict[str, Any],
    reduction: str,
) -> Any:
    namespace = dict(helpers)
    exec(compile(generated, "<hive-iteration>", "exec"), namespace)
    fn = namespace["hive_iteration"]
    results = (fn(item, values[item]) if indexed else fn(item) for item in items)
    if reduction == "sum":
        return sum(results)
    if reduction == "min":
        return min(results, default=None)
    if reduction == "max":
        return max(results, default=None)
    return list(results)


def run(
    source: str,
    values: Any,
    nodes: list[ComputeNode],
    helpers: dict[str, Any] | None = None,
    return_metrics: bool = False,
) -> Any:
    """Run trusted analyzed function using weighted Ray chunks and ordered merge."""
    import inspect

    helpers = helpers or {}
    for name, helper_fn in helpers.items():
        tree = ast.parse(inspect.getsource(helper_fn))
        fn = tree.body[0]
        args = {a.arg for a in fn.args.args}
        locals_ = {
            n.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        if _pure_expression(fn, args | locals_, set()) or any(
            isinstance(n, (ast.Global, ast.Nonlocal, ast.Subscript))
            for n in ast.walk(fn)
        ):
            raise ValueError("Helper purity could not be established: " + name)
    report = analyze(source, set(helpers))
    if not report.splittable:
        raise ValueError("; ".join(report.reasons))
    if report.reduction == "branches":
        namespace = dict(helpers)
        namespace[report.parameter] = values
        calls = []
        for predicate, expression in report.branch_expressions:
            if eval(
                compile(ast.parse(predicate, mode="eval"), "<hive-predicate>", "eval"),
                namespace,
            ):
                fn = eval(
                    compile(
                        ast.parse(
                            f"lambda {report.parameter}: {expression}", mode="eval"
                        ),
                        "<hive-branch>",
                        "eval",
                    ),
                    dict(helpers),
                )
                calls.append((fn, (values,)))
        return run_calls(calls, nodes, return_metrics=return_metrics)
    started = time.perf_counter()
    namespace = {report.parameter: values}
    items = eval(
        compile(
            ast.parse(report.iterable_source, mode="eval"), "<hive-iterable>", "eval"
        ),
        {"__builtins__": {"range": range, "len": len}},
        namespace,
    )
    if not isinstance(items, (range, list, tuple)):
        items = list(items)
    indexed = len(ast.parse(report.generated_source).body[0].args.args) == 2
    from .node_ranker import ranges

    plan = ranges(len(items), nodes)
    finishes = {}
    if len(plan) < 2:
        log.warning("Fewer than two CPU workers; executing analyzed iteration serially")
        parts = [
            _run_chunk(
                report.generated_source,
                items,
                indexed,
                values,
                helpers,
                report.reduction,
            )
        ]
        finishes = {p.node.node_id: time.perf_counter() - started for p in plan}
    else:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        if not ray.is_initialized():
            ray.init(
                num_cpus=min(len(plan), __import__("os").cpu_count() or 1),
                include_dashboard=False,
            )
        remote = ray.remote(_run_chunk)
        refs = []
        for chunk in plan:
            task = (
                remote.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        chunk.node.ray_node_id, soft=False
                    )
                )
                if chunk.node.ray_node_id
                else remote
            )
            refs.append(
                task.remote(
                    report.generated_source,
                    items[chunk.start : chunk.stop],
                    indexed,
                    values,
                    helpers,
                    report.reduction,
                )
            )
        pending = list(refs)
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            finishes[plan[refs.index(ready[0])].node.node_id] = (
                time.perf_counter() - started
            )
        parts = ray.get(refs)
    if report.reduction == "append":
        result = [v for part in parts for v in part]
    elif report.reduction == "sum":
        result = report.initial + sum(parts)
    elif report.reduction == "min":
        result = min([report.initial, *[v for v in parts if v is not None]])
    else:
        result = max([report.initial, *[v for v in parts if v is not None]])
    elapsed = time.perf_counter() - started
    if return_metrics:
        return (
            result,
            plan,
            elapsed,
            finishes,
            {node: max(0.0, elapsed - finish) for node, finish in finishes.items()},
        )
    return result


def run_calls(
    calls: list[tuple[Any, tuple[Any, ...]]],
    nodes: list[ComputeNode],
    return_metrics: bool = False,
) -> Any:
    """Execute explicitly selected trusted independent calls concurrently in Ray."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if not ray.is_initialized():
        ray.init(include_dashboard=False)
    active = [n for n in nodes if n.online and n.score > 0]
    if not active:
        raise ValueError("No online CPU nodes")

    def invoke(fn: Any, args: tuple[Any, ...]) -> Any:
        return fn(*args)

    started = time.perf_counter()
    remote = ray.remote(invoke)
    loads = {n: 0.0 for n in active}
    refs = []
    owners = []
    shares = {n.node_id: 0 for n in active}
    for fn, args in calls:
        node = min(active, key=lambda n: loads[n] + 1 / n.score)
        loads[node] += 1 / node.score
        task = (
            remote.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node.ray_node_id, soft=False
                )
            )
            if node.ray_node_id
            else remote
        )
        refs.append(task.remote(fn, args))
        owners.append(node.node_id)
        shares[node.node_id] += 1
    pending = list(refs)
    finishes = {}
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        finishes[owners[refs.index(ready[0])]] = time.perf_counter() - started
    result = ray.get(refs)
    elapsed = time.perf_counter() - started
    if return_metrics:
        return (
            result,
            elapsed,
            shares,
            finishes,
            {node: max(0.0, elapsed - finish) for node, finish in finishes.items()},
        )
    return result
