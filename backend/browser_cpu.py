"""Lower the existing conservative CPU analyzer into a small browser interpreter IR.

Uploaded Python is parsed, never executed. The interpreter has no access to
imports, globals, attributes, files, the network, or dynamic evaluation.
"""
import ast
import math
from .cpu_analyzer import analyze
from .models import TypedArray

OPS = {ast.Add:'add', ast.Sub:'sub', ast.Mult:'mul', ast.Div:'div', ast.FloorDiv:'floor_div', ast.Mod:'mod', ast.Pow:'pow', ast.BitAnd:'bit_and', ast.BitOr:'bit_or', ast.BitXor:'bit_xor', ast.LShift:'shift_left', ast.RShift:'shift_right', ast.Eq:'eq', ast.NotEq:'ne', ast.Lt:'lt', ast.LtE:'le', ast.Gt:'gt', ast.GtE:'ge'}
CALLS = {'abs', 'min', 'max', 'int', 'float', 'bool', 'round', 'pow'}


def expression(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool) and math.isfinite(node.value) and abs(node.value) <= 1e30:
        return ['constant', node.value]
    if isinstance(node, ast.Name):
        return ['name', node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return [OPS[type(node.op)], expression(node.left), expression(node.right)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        return [{ast.UAdd:'positive', ast.USub:'negative', ast.Not:'not'}[type(node.op)], expression(node.operand)]
    if isinstance(node, ast.IfExp):
        return ['if', expression(node.test), expression(node.body), expression(node.orelse)]
    if isinstance(node, ast.BoolOp):
        return ['and' if isinstance(node.op, ast.And) else 'or', *map(expression, node.values)]
    if isinstance(node, ast.Compare) and all(type(op) in OPS for op in node.ops):
        return ['compare', [OPS[type(op)] for op in node.ops], *map(expression, [node.left, *node.comparators])]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in CALLS and not node.keywords and 1 <= len(node.args) <= 8:
        return ['call', node.func.id, *map(expression, node.args)]
    raise ValueError(f'Browser CPU execution does not support {ast.unparse(node)}')


def lower(source):
    report = analyze(source)
    if not report.splittable or not report.generated_source or report.reduction not in ('append', 'sum', 'min', 'max'):
        raise ValueError('; '.join(report.reasons) or 'Select an independent numeric loop or comprehension')
    if report.iterable_source not in (report.parameter, f'range(len({report.parameter}))', f'range({report.parameter})'):
        raise ValueError('CPU loop must iterate over the input array, range(len(input)), or range(count)')
    fn = ast.parse(report.generated_source).body[0]
    steps = []
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            steps.append(['set', stmt.targets[0].id, expression(stmt.value)])
        elif isinstance(stmt, ast.Return):
            steps.append(['return', expression(stmt.value)])
        else:
            raise ValueError('Unsupported CPU statement')
    return {'arguments': [a.arg for a in fn.args.args], 'indexed': report.iterable_source != report.parameter, 'steps': steps,
            'reduction': report.reduction, 'initial': report.initial}


def plan(program, values, count, partitions):
    total = len(values) if values is not None else count
    if len(partitions) > 2048:
        raise ValueError('Maximum 2048 CPU chunks')
    chunks = []
    for offset, size, worker in partitions:
        chunks.append({'chunk_id': f'cpu-{offset}', 'offset': offset, 'count': size, 'byte_length': size*4, 'preferred_worker': worker,
            'assignment': {'kind': 'cpu', 'cpu_program': program, 'offset': offset, 'count': size, 'output_format': 'f32',
                'input': TypedArray.encode(values[offset:offset+size]).model_dump() if values is not None else None}})
    return {'chunks': chunks, 'size': total*4, 'assets': {}, 'output_format': 'f32', 'output_shape': [total],
            'reduction': program['reduction'], 'initial': program['initial']}
