"""Bounded job descriptions and planning; no submitted computations execute here."""
import base64
import hashlib
import math
import re
from typing import Annotated, Literal, Union
from pydantic import Field, model_validator, BeforeValidator
from ..models import Model, TypedArray
from ..compiler import compile_source
from ..marked_python import AnalysisRequest, analyze_marked
from .models import JobRequest, Parameters
from .image_workloads import WGSLImageRequest

class AnimationRequest(Model):
    kind: Literal['animation']
    width: int = Field(default=512, ge=1, le=4096)
    height: int = Field(default=512, ge=1, le=4096)
    tile_size: Literal[64] = 64
    frames: list[Parameters] = Field(min_length=1, max_length=32)
    fps: int = Field(default=12, ge=1, le=60)
    distribution: Literal['tiles', 'frames'] = 'tiles'

    @model_validator(mode='after')
    def bounded_output(self):
        if self.width * self.height * 4 * len(self.frames) > 100_663_296:
            raise ValueError('Animation output must fit 96 MiB')
        return self

class ComputeRequest(Model):
    count: int = Field(default=1, ge=1, le=2_000_000)
    chunk_size: int = Field(default=4096, ge=1, le=4096)
    input: TypedArray | None = None
    seed: int = Field(default=1, ge=0, le=0xffffffff)

class WGSLRequest(ComputeRequest):
    kind: Literal['wgsl']
    wgsl: str = Field(min_length=1, max_length=32000)
    splitting: Literal['auto', 'declared'] = 'auto'
    independent: bool = False
    output_dtype: Literal['f32', 'u32', 'i32'] = 'f32'

class PythonRequest(ComputeRequest):
    kind: Literal['python']
    source: str = Field(min_length=1, max_length=32000)

class OnnxRequest(Model):
    kind: Literal['onnx']
    model: str = Field(min_length=1, max_length=5_592_408)
    input: TypedArray
    input_shape: list[int] = Field(min_length=2, max_length=4)
    batch_size: int = Field(default=32, ge=1, le=256)
    independent: Literal[True]

    @model_validator(mode='after')
    def shape(self):
        if any(d < 1 for d in self.input_shape) or math.prod(self.input_shape) > 500_000:
            raise ValueError('Input shape must contain positive dimensions and at most 500,000 elements')
        if self.input.dtype != 'f32':
            raise ValueError('ONNX inputs must use f32')
        return self

class OnnxAnalysisRequest(Model):
    model: str = Field(min_length=1, max_length=5_592_408)
    input_shape: list[int] | None = Field(default=None, min_length=2, max_length=4)
    samples: int = Field(default=256, ge=1, le=500_000)
    batch_size: int | None = Field(default=None, ge=1, le=256)

JobSubmission = Annotated[Union[JobRequest, AnimationRequest, WGSLRequest, PythonRequest, OnnxRequest, WGSLImageRequest], Field(discriminator='kind'), BeforeValidator(lambda v: {'kind':'mandelbrot', **v} if isinstance(v, dict) else v)]


def asset(data):
    return hashlib.sha256(data).hexdigest(), data


def compute_plan(request, partitions=None):
    values = request.input.decode() if request.input else None
    total = len(values) if values is not None else request.count
    if total < 1:
        raise ValueError('Input array must not be empty')
    if math.ceil(total/request.chunk_size) > 2048:
        raise ValueError('Maximum 2048 chunks; increase chunk_size')
    if isinstance(request, PythonRequest):
        if request.input is None or request.input.dtype != 'f32':
            raise ValueError('Marked Python requires a separate f32 input array')
        report = analyze_marked(AnalysisRequest(source=request.source))
        if report.status != 'ready':
            raise ValueError('; '.join(report.findings))
        kernel = compile_source(report.kernel)
        code = kernel.wgsl
        bindings = [vars(b) for b in kernel.buffers]
        uniforms = [vars(u) for u in kernel.uniforms]
        dtype = 'f32'
    else:
        from ..wgsl_analyzer import WGSLAnalysisRequest, analyze_wgsl
        if request.splitting == 'auto':
            report = analyze_wgsl(WGSLAnalysisRequest(source=request.wgsl, count=total, chunk_size=request.chunk_size))
            if report.status != 'ready':
                raise ValueError('; '.join(report.findings))
            if (request.input.dtype if request.input else None) != report.input_dtype:
                raise ValueError('Input array must match the analyzed shader input dtype')
            if request.output_dtype != report.output_dtype:
                raise ValueError('Output dtype must match the analyzed shader')
            code = report.wgsl
        else:
            if not request.independent:
                raise ValueError('Declared WGSL requires independent=true and the Hive binding contract')
            code = request.wgsl
        # The worker performs actual WGSL parsing and compilation before dispatch.
        plain = re.sub(r'/\*.*?\*/|//[^\n]*', '', code, flags=re.S)
        if re.findall(r'@workgroup_size\s*\(([^)]*)\)', plain) != ['64'] or not re.search(r'@compute\b', plain) or not re.search(r'\bfn\s+main\s*\(', plain):
            raise ValueError('Custom WGSL must declare one @compute @workgroup_size(64) fn main entry point')
        dtype = request.output_dtype
        bindings = [{'binding':2, 'access':'read_write', 'element_type':dtype}]
        if request.input:
            bindings.insert(0, {'binding':1, 'access':'read', 'element_type':request.input.dtype})
        uniforms = [{'name':n, 'type':'u32'} for n in ('offset','count','seed')]
    shader_id, data = asset(code.encode())
    chunks = []
    pieces = partitions or [(offset, min(request.chunk_size, total-offset), None) for offset in range(0, total, request.chunk_size)]
    if len(pieces) > 2048:
        raise ValueError('Maximum 2048 chunks; increase chunk_size')
    for offset, count, preferred_worker in pieces:
        chunks.append({'chunk_id':f'batch-{offset}', 'offset':offset, 'count':count, 'byte_length':count*4,
            'preferred_worker': preferred_worker,
            'assignment':{'kind':'compute', 'shader_id':shader_id, 'output_format':dtype, 'offset':offset, 'count':count,
                'bindings':bindings, 'uniforms':uniforms, 'compute_parameters':{'offset':offset, 'count':count, 'seed':request.seed},
                'input':TypedArray.encode(values[offset:offset+count], request.input.dtype).model_dump() if values is not None else None}})
    return {'chunks':chunks, 'size':total*4, 'assets':{shader_id:data}, 'output_format':dtype, 'output_shape':[total]}


def onnx_plan(request, inspect_only=False):
    import onnx
    from onnx import TensorProto
    try:
        raw = base64.b64decode(request.model, validate=True)
        if not 1 <= len(raw) <= 4_194_304:
            raise ValueError('Embedded ONNX model limit is 4 MiB')
        model = onnx.load_model_from_string(raw)
        if model.functions or model.training_info or model.ir_version > 10:
            raise ValueError('Use an inference model with IR version <= 10 and no local functions')
        if len(model.graph.node) > 128 or len(model.graph.initializer) > 128 or model.graph.sparse_initializer:
            raise ValueError('Model limit is 128 nodes/initializers; sparse tensors are unsupported')
        for tensor in model.graph.initializer:
            if tensor.external_data or tensor.data_location == TensorProto.EXTERNAL or math.prod(tensor.dims) > 2_000_000:
                raise ValueError('Weights must be bounded embedded tensors; external data is unsupported')
        if any(o.domain not in ('', 'ai.onnx') or not 13 <= o.version <= 21 for o in model.opset_import):
            raise ValueError('Use standard ONNX operators with opset 13–21')
        # Reject nested graphs before calling the checker (which can resolve external data).
        allowed = {'Identity', 'Relu', 'Sigmoid', 'Tanh', 'Add', 'Sub', 'Mul', 'Div', 'MatMul', 'Gemm'}
        for node in model.graph.node:
            if node.domain not in ('','ai.onnx') or node.op_type not in allowed or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS, onnx.AttributeProto.TENSOR, onnx.AttributeProto.TENSORS) for a in node.attribute):
                raise ValueError(f'Unsupported independent-batch operator: {node.op_type}')
        onnx.checker.check_model(model, full_check=True)
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        g = model.graph
        weights = {t.name: list(t.dims) for t in g.initializer}
        inputs = [v for v in g.input if v.name not in weights]
        if len(inputs) != 1 or len(g.output) != 1:
            raise ValueError('Use exactly one float32 input and one output')
        def shape(v):
            t = v.type.tensor_type
            if t.elem_type != TensorProto.FLOAT:
                raise ValueError('Runtime tensors must be float32')
            dims = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in t.shape.dim]
            if not 2 <= len(dims) <= 4 or any(type(d) is not int or d < 1 for d in dims[1:]) or not dims[0]:
                raise ValueError('Only the first (batch) dimension may be dynamic')
            return dims
        input_shape, output_shape = shape(inputs[0]), shape(g.output[0])
        requested_shape = request.input_shape or [request.samples, *input_shape[1:]]
        batch_size = request.batch_size or (input_shape[0] if isinstance(input_shape[0], int) else 32)
        if not 1 <= batch_size <= 256:
            raise ValueError('Supported model batch sizes are 1–256')
        if math.prod(requested_shape) > 500_000 or any(d < 1 for d in requested_shape):
            raise ValueError('Input limit is 500,000 positive-shaped elements')
        if input_shape[1:] != requested_shape[1:] or input_shape[0] != output_shape[0]:
            raise ValueError('Model sample shapes must match input; output must preserve the batch dimension')
        if isinstance(input_shape[0], int) and (batch_size != input_shape[0] or requested_shape[0] % batch_size):
            raise ValueError('Static batch models require matching batch_size and a divisible sample count')
        shapes = {v.name:shape(v) for v in [*inputs, *g.value_info, *g.output] if v.name not in weights}
        if any(batch_size*math.prod(dims[1:])*4 > 1_048_576 for dims in shapes.values()):
            raise ValueError('Every intermediate batch tensor must fit 1 MiB; reduce batch_size')
        dependent = {inputs[0].name}
        for node in g.node:
            variable = [n for n in node.input if n in dependent]
            if not variable or any(n and n not in dependent and n not in weights for n in node.input):
                raise ValueError('Every operator must process the independent input or embedded weights')
            if node.op_type in {'MatMul', 'Gemm'}:
                if len(shapes[node.input[0]]) != 2 or node.input[0] not in dependent or node.input[1] not in weights or len(weights[node.input[1]]) != 2:
                    raise ValueError('Matrix operations require rank-2 samples multiplied by constant rank-2 weights')
                if any(a.name == 'transA' and a.i != 0 for a in node.attribute):
                    raise ValueError('Gemm transA would mix samples across the batch axis')
            for n in node.input:
                if n in weights and not (node.op_type in {'MatMul','Gemm'} and n == node.input[1]):
                    dims = weights[n]
                    rank = len(shapes[variable[0]])
                    if len(dims) > rank or (len(dims) == rank and dims[0] != 1):
                        raise ValueError('Broadcast weights may not vary along the batch axis')
            for n in node.output:
                if n not in shapes or shapes[n][0] != input_shape[0]:
                    raise ValueError('Every intermediate tensor must preserve the independent batch axis')
                dependent.add(n)
        samples = requested_shape[0]
        sample_in, sample_out = math.prod(input_shape[1:]), math.prod(output_shape[1:])
        if samples*sample_out > 2_000_000 or batch_size*max(sample_in,sample_out)*4 > 1_048_576:
            raise ValueError('Output limit is 2 million elements; each batch buffer must fit 1 MiB')
        if math.ceil(samples/batch_size) > 2048:
            raise ValueError('Maximum 2048 inference batches')
        if inspect_only:
            return {'status':'ready', 'input_name':inputs[0].name, 'output_name':g.output[0].name,
                'model_input_shape':input_shape, 'model_output_shape':output_shape,
                'input_shape':requested_shape, 'output_shape':[samples,*output_shape[1:]],
                'batch_size':batch_size, 'chunk_count':math.ceil(samples/batch_size),
                'operators':sorted({n.op_type for n in g.node}),
                'findings':['The graph preserves the sample axis. Each worker caches the complete model and processes a separate input batch.', 'Results are reassembled in input order. This partitions inference inputs, not model weights or layers.']}
        values = request.input.decode()
        if len(values) != math.prod(requested_shape):
            raise ValueError('Input array length does not match input_shape')
        model_id, data = asset(raw)
        chunks = []
        for offset in range(0, samples, batch_size):
            count = min(batch_size, samples-offset)
            chunks.append({'chunk_id':f'batch-{offset}', 'offset':offset*sample_out, 'count':count*sample_out, 'byte_length':count*sample_out*4,
                'assignment':{'kind':'onnx_batch', 'output_format':'f32', 'model_id':model_id, 'offset':offset, 'count':count,
                    'input_name':inputs[0].name, 'output_name':g.output[0].name, 'input_shape':[count,*input_shape[1:]],
                    'output_shape':[count,*output_shape[1:]], 'input':TypedArray.encode(values[offset*sample_in:(offset+count)*sample_in]).model_dump()}})
        return {'chunks':chunks, 'size':samples*sample_out*4, 'assets':{model_id:data}, 'output_format':'f32', 'output_shape':[samples,*output_shape[1:]]}
    except Exception as exc:
        raise ValueError(f'Invalid ONNX model or batch: {exc}') from exc
