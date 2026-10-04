"""Find independent array loops and translate only explicitly selected regions.

Comments select extraction, never execution of the surrounding Python program.
"""
import ast
import io
import math
import tokenize
from typing import Literal
from pydantic import Field
from .models import Model
from .compiler import compile_source

BEGIN = '# hive:parallel begin'
END = '# hive:parallel end'

class AnalysisRequest(Model):
    source: str = Field(min_length=1, max_length=32000)
    auto_mark: bool = False
    candidate_line: int | None = Field(default=None, ge=1)

class Candidate(Model):
    line: int
    end_line: int
    input_name: str
    output_name: str
    pattern: str
    kernel: str

class Analysis(Model):
    status: Literal['ready', 'selection_required', 'unsupported']
    candidates: list[Candidate] = Field(default_factory=list)
    marked_source: str | None = None
    kernel: str | None = None
    wgsl: str | None = None
    findings: list[str] = Field(default_factory=list)


def convert_loop(fn, loop):
    if fn.decorator_list or loop.orelse or not isinstance(loop.target, ast.Name):
        raise ValueError('Decorated functions, destructuring, and for/else need manual conversion')
    index = loop.target.id
    indexed = isinstance(loop.iter, ast.Call)
    if indexed:
        call = loop.iter
        if not (isinstance(call.func, ast.Name) and call.func.id == 'range' and len(call.args) == 1 and not call.keywords):
            raise ValueError('Use range(len(input)) for indexed loops')
        length = call.args[0]
        if not (isinstance(length, ast.Call) and isinstance(length.func, ast.Name) and length.func.id == 'len' and len(length.args) == 1 and not length.keywords and isinstance(length.args[0], ast.Name)):
            raise ValueError('Indexed bounds must be len(input)')
        input_name = length.args[0].id
    elif isinstance(loop.iter, ast.Name):
        input_name = loop.iter.id
    else:
        raise ValueError('Use for value in input or for i in range(len(input))')
    if input_name not in {a.arg for a in fn.args.args} or index == input_name:
        raise ValueError('Input must be a function argument and distinct from the loop variable')
    locals_map = {}
    body = []
    output_name = None

    def expression(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float) and math.isfinite(n.value) and abs(n.value) <= 1e30:
            return repr(float(n.value))
        if isinstance(n, ast.Name):
            if not indexed and n.id == index:
                return 'values[i]'
            if n.id in locals_map:
                return locals_map[n.id]
        if indexed and isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == input_name and isinstance(n.slice, ast.Name) and n.slice.id == index:
            return 'values[i]'
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)):
            op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*'}[type(n.op)]
            return f'({expression(n.left)} {op} {expression(n.right)})'
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            value = expression(n.operand)
            return value if isinstance(n.op, ast.UAdd) else f'(-{value})'
        raise ValueError('Only current-element reads, finite constants, iteration-local variables, and + - * are supported; dependencies and calls are rejected')

    for stmt in loop.body:
        if output_name is not None:
            raise ValueError('The output write must be the last statement')
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            expr = expression(stmt.value)
            if isinstance(target, ast.Name) and target.id not in (input_name, index):
                if target.id in locals_map:
                    raise ValueError('Assign each iteration-local variable once')
                locals_map[target.id] = f'local{len(locals_map)}'
                body.append(f'{locals_map[target.id]} = {expr}')
                continue
            if indexed and isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and isinstance(target.slice, ast.Name) and target.slice.id == index:
                output_name = target.value.id
                body.append(f'result[i] = {expr}')
                continue
        if not indexed and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.attr == 'append' and len(call.args) == 1 and not call.keywords:
                output_name = call.func.value.id
                body.append(f'result[i] = {expression(call.args[0])}')
                continue
        raise ValueError('Use local assignments followed by one out[i] write or append; mutations, reductions, and side effects are unsupported')
    if not output_name or output_name in {input_name, index, *locals_map} or output_name in {a.arg for a in fn.args.args}:
        raise ValueError('Write to a fresh output array, separate from input arguments')
    # Require a fresh result list immediately before the selected loop.
    statements = fn.body
    position = next((i for i, n in enumerate(statements) if n is loop), -1)
    init = statements[position-1] if position > 0 else None
    if not (isinstance(init, ast.Assign) and len(init.targets) == 1 and isinstance(init.targets[0], ast.Name) and init.targets[0].id == output_name):
        raise ValueError('Declare the fresh output list immediately before the loop')
    fresh = isinstance(init.value, ast.List) and not init.value.elts
    if indexed:
        v = init.value
        fresh = (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult) and isinstance(v.left, ast.List) and len(v.left.elts) == 1 and isinstance(v.left.elts[0], ast.Constant) and type(v.left.elts[0].value) in (int, float) and ast.dump(v.right) == ast.dump(loop.iter.args[0]))
    if not fresh:
        raise ValueError('Initialize output with [] for append, or [0.0] * len(input) for indexing')
    kernel = 'def transform(offset: u32, count: u32, values: Array[f32], result: Array[f32]):\n    i = global_id()\n    if i < count:\n' + ''.join(f'        {s}\n' for s in body)
    compile_source(kernel)
    return Candidate(line=loop.lineno, end_line=loop.end_lineno, input_name=input_name, output_name=output_name, pattern='indexed array' if indexed else 'append map', kernel=kernel)


def analyze_marked(request: AnalysisRequest):
    try:
        tree = ast.parse(request.source)
        if sum(1 for _ in ast.walk(tree)) > 4000:
            raise ValueError('Source exceeds 4000 AST nodes')
        comments = [(t.start[0], t.string.strip()) for t in tokenize.generate_tokens(io.StringIO(request.source).readline) if t.type == tokenize.COMMENT]
    except (ValueError, SyntaxError, tokenize.TokenError, RecursionError) as exc:
        return Analysis(status='unsupported', findings=[str(exc)])
    candidates, findings = [], []
    # Only direct function-body loops; nested/control-flow extraction is ambiguous.
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for loop in fn.body:
                if isinstance(loop, ast.For):
                    try:
                        candidates.append(convert_loop(fn, loop))
                    except (ValueError, OverflowError, RecursionError) as exc:
                        findings.append(f'Line {loop.lineno}: {exc}')
    candidates = candidates[:40]
    begins = [line for line, comment in comments if comment == BEGIN]
    ends = [line for line, comment in comments if comment == END]
    source = request.source
    selected = None
    if begins or ends:
        if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0]:
            return Analysis(status='unsupported', candidates=candidates, findings=['Use exactly one ordered pair of hive:parallel begin/end comment lines'])
        lines = source.splitlines()
        inside = [i+1 for i, text in enumerate(lines) if begins[0] < i+1 < ends[0] and text.strip() and not text.lstrip().startswith('#')]
        selected = next((c for c in candidates if inside and c.line == inside[0] and c.end_line == inside[-1]), None)
        if selected is None:
            return Analysis(status='unsupported', candidates=candidates, findings=['Markers must enclose exactly one supported independent for loop', *findings[:40]])
    elif request.auto_mark:
        selected = next((c for c in candidates if c.line == request.candidate_line), None) if request.candidate_line else candidates[0] if len(candidates) == 1 else None
        if selected:
            lines = source.splitlines(keepends=True)
            indent = lines[selected.line-1][:len(lines[selected.line-1])-len(lines[selected.line-1].lstrip())]
            if not lines[selected.end_line-1].endswith('\n'):
                lines[selected.end_line-1] += '\n'
            lines.insert(selected.end_line, indent + END + '\n')
            lines.insert(selected.line-1, indent + BEGIN + '\n')
            source = ''.join(lines)
    if selected:
        return Analysis(status='ready', candidates=candidates, marked_source=source, kernel=selected.kernel, wgsl=compile_source(selected.kernel).wgsl, findings=['Only the marked region is extracted; surrounding Python is never executed. Supply the selected input array separately.', 'GPU arithmetic uses float32; review precision differences from Python.', *findings[:38]])
    return Analysis(status='selection_required' if candidates else 'unsupported', candidates=candidates, findings=['Select an independent loop and add hive:parallel begin/end comments, or use automatic marking.', *findings[:39]])
