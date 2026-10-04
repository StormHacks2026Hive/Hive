"""Only AST parsing and py2wgsl source translation; never execute user code."""
import ast
import math
from functools import lru_cache
from py2wgsl import compile_kernel_source, WGSLCompileError

@lru_cache(maxsize=128)
def compile_source(source: str):
    try:
        tree = ast.parse(source)
        if len(source) > 32000 or sum(1 for _ in ast.walk(tree)) > 4000:
            raise ValueError('Kernel too large')
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError('Submit exactly one kernel function, without imports or top-level statements')
        fn = tree.body[0]
        if fn.decorator_list or fn.args.defaults or fn.args.kw_defaults or fn.args.vararg or fn.args.kwarg:
            raise ValueError('Decorators, defaults and variadic arguments are unsupported')
        # Distributed MVP permits elementwise kernels and strictly bounded constant loops.
        for node in ast.walk(fn):
            if isinstance(node, (ast.While, ast.AsyncFor)):
                raise ValueError('Unbounded loops are unsupported; use range with literal bounds')
            if isinstance(node, ast.For):
                call = node.iter
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != 'range' or not all(isinstance(a, ast.Constant) and type(a.value) is int for a in call.args):
                    raise ValueError('Loop range bounds must be integer literals')
                if not 1 <= len(call.args) <= 3 or len(range(*(a.value for a in call.args))) > 1024:
                    raise ValueError('Loop limit is 1024 iterations')
        loops = [len(range(*(a.value for a in n.iter.args))) for n in ast.walk(fn) if isinstance(n, ast.For)]
        if math.prod(loops) > 1024:
            raise ValueError('Combined loop limit is 1024 iterations per element')
        kernel = compile_kernel_source(source, workgroup_size=64)
        writes = [b for b in kernel.buffers if b.access == 'read_write']
        reads = [b for b in kernel.buffers if b.access == 'read']
        if len(writes) != 1 or len(reads) > 1:
            raise ValueError('Use exactly one output array and at most one input array')
        return kernel
    except (SyntaxError, WGSLCompileError, ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise ValueError(f'Unsupported kernel: {exc}') from exc

def validate_job(request, kernel):
    reads = [b for b in kernel.buffers if b.access == 'read']
    if (request.mode == 'data-slice') != bool(reads):
        raise ValueError('data-slice needs one read-only array; parameter-only needs none')
    if reads and reads[0].element_type != request.input.dtype:
        raise ValueError('Input dtype does not match kernel annotation')
    reserved = {'offset', 'count', 'seed'}
    uniforms = {u.name: u.type for u in kernel.uniforms}
    if not {'offset', 'count'} <= uniforms.keys() or (request.mode == 'parameter-only' and 'seed' not in uniforms):
        raise ValueError('Declare offset: u32, count: u32, and for parameter-only seed: u32')
    if any(uniforms[n] != 'u32' for n in reserved & uniforms.keys()):
        raise ValueError('offset, count and seed must be u32')
    if set(request.parameters) != uniforms.keys() - reserved:
        raise ValueError('parameters must match the non-reserved scalar kernel arguments')
    for name, value in request.parameters.items():
        kind = uniforms[name]
        if not math.isfinite(value) or (kind == 'f32' and abs(value) > 3.402823e38):
            raise ValueError(f'{name} is outside f32 range')
        if kind != 'f32' and (type(value) is not int or not (-2**31 <= value < 2**31 if kind == 'i32' else 0 <= value < 2**32)):
            raise ValueError(f'{name} must fit {kind}')
