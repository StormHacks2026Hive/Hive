"""Conservative source-to-source suggestions. Uploaded Python is never executed."""
import ast
from typing import Literal
from pydantic import Field
from .models import Model, Binding, Uniform
from .compiler import compile_source

class SourceRequest(Model):
    source: str = Field(min_length=1, max_length=32000)

class Finding(Model):
    severity: Literal['info', 'warning', 'error']
    message: str
    line: int | None = None

class CompatibilityReport(Model):
    status: Literal['compatible', 'conversion_available', 'manual_conversion_required']
    findings: list[Finding]
    kernel: str | None = None
    mode: Literal['data-slice', 'parameter-only'] | None = None
    parameters: dict[str, int | float] = Field(default_factory=dict)

class KernelValidation(Model):
    valid: bool
    error: str | None = None
    bindings: list[Binding] = Field(default_factory=list)
    uniforms: list[Uniform] = Field(default_factory=list)
    mode: Literal['data-slice', 'parameter-only'] | None = None


def validate_preview(source):
    try:
        k = compile_source(source)
        uniforms = {u.name: u.type for u in k.uniforms}
        mode = 'data-slice' if any(b.access == 'read' for b in k.buffers) else 'parameter-only'
        required = {'offset', 'count'} | ({'seed'} if mode == 'parameter-only' else set())
        if not required <= uniforms.keys() or any(uniforms[n] != 'u32' for n in {'offset', 'count', 'seed'} & uniforms.keys()):
            raise ValueError('Declare offset: u32, count: u32, and seed: u32 for parameter-only kernels')
        return KernelValidation(valid=True, bindings=[Binding(**vars(b)) for b in k.buffers],
            uniforms=[Uniform(**vars(u)) for u in k.uniforms], mode=mode)
    except ValueError as exc:
        return KernelValidation(valid=False, error=str(exc))


def convert_elementwise(tree):
    """Match only a complete three-statement, one-input list transformation."""
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    fn = tree.body[0]
    args = fn.args
    if fn.decorator_list or args.defaults or args.kw_defaults or args.vararg or args.kwarg or args.posonlyargs or args.kwonlyargs or len(args.args) != 1 or len(fn.body) != 3:
        return None
    init, loop, ret = fn.body
    if not isinstance(init, ast.Assign) or len(init.targets) != 1 or not isinstance(init.targets[0], ast.Name) or not isinstance(init.value, ast.List) or init.value.elts:
        return None
    output = init.targets[0].id
    input_name = args.args[0].arg
    if output == input_name or not isinstance(loop, ast.For) or loop.orelse or not isinstance(loop.target, ast.Name) or not isinstance(loop.iter, ast.Name) or loop.iter.id != input_name or len(loop.body) != 1:
        return None
    variable = loop.target.id
    if variable in (output, input_name):
        return None
    stmt = loop.body[0]
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    call = stmt.value
    if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name) or call.func.value.id != output or call.func.attr != 'append' or len(call.args) != 1 or call.keywords:
        return None
    if not isinstance(ret, ast.Return) or not isinstance(ret.value, ast.Name) or ret.value.id != output:
        return None
    # Float arithmetic only. No calls, indexing, division/modulo, powers, or captured variables.
    def expression(node):
        if isinstance(node, ast.Name) and node.id == variable:
            return 'values[i]'
        if isinstance(node, ast.Constant) and type(node.value) in (int, float) and abs(node.value) <= 1e30:
            return repr(float(node.value))
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            op = {ast.Add:'+', ast.Sub:'-', ast.Mult:'*'}[type(node.op)]
            return f'({expression(node.left)} {op} {expression(node.right)})'
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return f"({'-' if isinstance(node.op, ast.USub) else '+'}{expression(node.operand)})"
        raise ValueError('Expression requires manual conversion')
    try:
        expr = expression(call.args[0])
    except (ValueError, OverflowError, RecursionError):
        return None
    return ('def transform(offset: u32, count: u32, values: Array[f32], result: Array[f32]):\n'
            '    i = global_id()\n    if i < count:\n        result[i] = ' + expr + '\n')


def analyze(source):
    try:
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 4000:
            raise ValueError('Source exceeds 4000 AST nodes')
    except (SyntaxError, ValueError, RecursionError) as exc:
        return CompatibilityReport(status='manual_conversion_required', findings=[Finding(severity='error', message=str(exc), line=getattr(exc, 'lineno', None))])
    preview = validate_preview(source)
    if preview.valid:
        return CompatibilityReport(status='compatible', kernel=source, mode=preview.mode,
            findings=[Finding(severity='info', message='This file already compiles as a kernel. Review its indexing and supply inputs before submitting.')],
            parameters={u.name: 1 if u.type != 'f32' else 1.0 for u in preview.uniforms if u.name not in {'offset','count','seed'}})
    candidate = convert_elementwise(tree)
    if candidate and validate_preview(candidate).valid:
        return CompatibilityReport(status='conversion_available', kernel=candidate, mode='data-slice', findings=[
            Finding(severity='info', message='Converted a one-input append loop with independent arithmetic into an elementwise kernel.'),
            Finding(severity='warning', message='The proposed kernel uses float32 arithmetic, which can differ from Python integers and float64. Review it and provide the input array separately.')])
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            findings.append(Finding(severity='warning', line=node.lineno, message='Imports cannot run on GPU nodes; rewrite library operations as supported scalar arithmetic.'))
        elif isinstance(node, ast.While):
            findings.append(Finding(severity='warning', line=node.lineno, message='While loops require a bounded literal range and an explicit escape condition.'))
        elif isinstance(node, ast.Constant) and isinstance(node.value, complex):
            findings.append(Finding(severity='warning', line=node.lineno, message='Complex values require separate real and imaginary scalar variables.'))
        elif isinstance(node, ast.ClassDef):
            findings.append(Finding(severity='warning', line=node.lineno, message='Classes need manual extraction of independent computations.'))
        elif isinstance(node, ast.Call):
            findings.append(Finding(severity='warning', line=node.lineno, message=f'Call {ast.unparse(node.func)} needs review; arbitrary Python calls are not automatically converted.'))
    findings = findings[:40]
    findings.append(Finding(severity='info', message='Automatic conversion currently supports exactly one function with one input argument, an empty result list, a for-each append loop using +, -, or *, and return of that list. Other files need manual conversion. No code was executed.'))
    return CompatibilityReport(status='manual_conversion_required', findings=findings)
