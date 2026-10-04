import asyncio
import base64
import struct
import pytest
import onnx
from onnx import helper, TensorProto
from fastapi.testclient import TestClient
from backend.marked_python import AnalysisRequest, analyze_marked
from backend.wgsl_analyzer import WGSLAnalysisRequest, analyze_wgsl
from backend.models import TypedArray
from backend.pool.workloads import WGSLRequest, PythonRequest, AnimationRequest, OnnxRequest, OnnxAnalysisRequest, onnx_plan
from backend.pool.coordinator import Coordinator
from backend.pool.models import ResultHeader
from tests.test_pool import add_worker, CAPABILITIES

PYTHON = '''import os

def transform(values):
    result = [0.0] * len(values)
    for i in range(len(values)):
        scaled = values[i] * 2
        result[i] = scaled + 1
    return result
'''
WGSL = '''@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    result[i] = values[i] * 2.0 + 1.0;
}'''


def model(op='Relu', shape=None, **attributes):
    shape = shape or ['batch', 2]
    graph = helper.make_graph([helper.make_node(op, ['X'], ['Y'], **attributes)], 'test',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, shape)])
    return base64.b64encode(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10).SerializeToString()).decode()


def reply(a, value=2):
    size = (a.output_shape[0]*a.output_shape[1] if a.kind == 'onnx_batch' else a.count) if a.kind != 'image_tile' else a.tile.width*a.tile.height
    payload = struct.pack('<f', value)*size if a.output_format == 'f32' else bytes([value,0,0,255])*size
    return ResultHeader(type='chunk_result', job_id=a.job_id, chunk_id=a.chunk_id, attempt_id=a.attempt_id,
        output_format=a.output_format, byte_length=len(payload), elapsed_ms=2), payload


def test_python_mark_suggest_compile_and_never_execute(tmp_path):
    marker = tmp_path/'executed'
    source = f"open({str(marker)!r}, 'w').write('bad')\n" + PYTHON
    report = analyze_marked(AnalysisRequest(source=source, auto_mark=True))
    assert report.status == 'ready' and report.wgsl and '# hive:parallel begin' in report.marked_source
    assert analyze_marked(AnalysisRequest(source=report.marked_source)).status == 'ready'
    assert not marker.exists()
    assert analyze_marked(AnalysisRequest(source=source)).status == 'selection_required'


@pytest.mark.parametrize('expr', ['result[i-1] + values[i]', 'values[i-1]', 'sum(values)', 'scaled + values[i]', 'values[i] + captured'])
def test_python_rejects_dependencies(expr):
    source = PYTHON.replace('scaled = values[i] * 2', f'scaled = {expr}')
    assert analyze_marked(AnalysisRequest(source=source, auto_mark=True)).status == 'unsupported'


def test_marker_bounds_and_comments_in_strings():
    marked = analyze_marked(AnalysisRequest(source=PYTHON, auto_mark=True)).marked_source
    assert analyze_marked(AnalysisRequest(source=marked.replace('    # hive:parallel end', '    return result\n    # hive:parallel end'))).status == 'unsupported'
    assert analyze_marked(AnalysisRequest(source='x = "# hive:parallel begin"\n'+PYTHON, auto_mark=True)).status == 'ready'
    assert analyze_marked(AnalysisRequest(source=PYTHON+PYTHON.replace('transform', 'other'), auto_mark=True)).status == 'selection_required'


def test_wgsl_rewrite_and_partition_ranges():
    report = analyze_wgsl(WGSLAnalysisRequest(source=WGSL, count=10, chunk_size=4))
    assert report.status == 'ready' and report.chunk_count == 3
    assert report.ranges == [{'offset':0,'count':4},{'offset':4,'count':4},{'offset':8,'count':2}]
    assert '@binding(2)' in report.wgsl and 'values[hive_local_index]' in report.wgsl
    assert '@workgroup_size(64)' in report.wgsl
    generated = WGSL.replace('values[i] * 2.0 + 1.0', 'values[i] + f32(i)')
    assert '(hive_params.offset + hive_local_index)' in analyze_wgsl(WGSLAnalysisRequest(source=generated)).wgsl


@pytest.mark.parametrize('source', [WGSL.replace('values[i]', 'values[i-1]'), WGSL.replace('values[i]', 'result[i-1]'), WGSL.replace('result[i] =', 'result[i+1] ='), WGSL.replace('values[i]', 'atomicAdd(&values[i], 1)'), WGSL.replace('result[i] =', 'storageBarrier(); result[i] =')])
def test_wgsl_rejects_dependent_or_shared_work(source):
    assert analyze_wgsl(WGSLAnalysisRequest(source=source)).status != 'ready'


def test_compute_ordered_assembly_and_stale_attempt():
    pool = Coordinator()
    job = pool.create(WGSLRequest(kind='wgsl', wgsl=WGSL, input=TypedArray.encode(range(10)), chunk_size=4))
    a, b = add_worker(pool), add_worker(pool)
    first, second = pool.pull(a), pool.pull(b)
    pool.accept(b, *reply(second, 9)); pool.disconnect(a.worker_id)
    retry = pool.pull(b)
    assert retry.offset == first.offset and retry.attempt_id != first.attempt_id
    assert pool.accept(a, *reply(first)).disposition == 'stale'
    pool.accept(b, *reply(retry, 3)); pool.accept(b, *reply(pool.pull(b), 7))
    assert job.status == 'done'
    assert list(struct.unpack('<10f', job.image)) == [3]*4+[9]*4+[7]*2


def test_python_pool_uses_compiled_bindings():
    source = analyze_marked(AnalysisRequest(source=PYTHON, auto_mark=True)).marked_source
    pool=Coordinator(); pool.create(PythonRequest(kind='python', source=source, input=TypedArray.encode([1,2])))
    assignment = pool.pull(add_worker(pool))
    assert assignment.kind == 'compute' and len(assignment.bindings) == 2
    assert set(assignment.compute_parameters) == {'offset','count','seed'}


@pytest.mark.parametrize('distribution, total', [('tiles',128), ('frames',2)])
def test_animation_frames_assemble_separately(distribution, total):
    pool = Coordinator(); job = pool.create(AnimationRequest(kind='animation', frames=[{}, {}], distribution=distribution))
    assert len(job.chunks) == total
    w=add_worker(pool)
    for _ in job.chunks:
        a=pool.pull(w); pool.accept(w,*reply(a, a.frame_index+1))
    assert job.status == 'done' and len(job.image)==512*512*4*2
    assert job.image[:4] == bytes([1,0,0,255]) and job.image[512*512*4:512*512*4+4] == bytes([2,0,0,255])


def test_onnx_metadata_partial_batch_and_capability():
    report=onnx_plan(OnnxAnalysisRequest(model=model(), samples=5, batch_size=2), True)
    assert report['input_shape']==[5,2] and report['chunk_count']==3
    request=OnnxRequest(kind='onnx', model=model(), input=TypedArray.encode(range(10)), input_shape=[5,2], batch_size=2, independent=True)
    pool=Coordinator(); job=pool.create(request)
    worker=add_worker(pool)
    assert pool.pull(worker).type=='no_work'
    worker.capabilities.onnx=True
    assignments=[]
    for _ in range(3):
        a=pool.pull(worker); assignments.append(a); pool.accept(worker,*reply(a))
    assert [a.input_shape for a in assignments]==[[2,2],[2,2],[1,2]] and job.status=='done'


def test_onnx_rejects_batch_mixing_wrong_shape_external_and_static_remainder():
    with pytest.raises(ValueError): onnx_plan(OnnxAnalysisRequest(model=model('Softmax', axis=0)),True)
    with pytest.raises(ValueError): onnx_plan(OnnxAnalysisRequest(model=model(),input_shape=[8,3]),True)
    with pytest.raises(ValueError): onnx_plan(OnnxAnalysisRequest(model=model(shape=[4,2]),samples=5,batch_size=4),True)
    m=onnx.load_model_from_string(base64.b64decode(model()))
    weight=helper.make_tensor('weights',TensorProto.FLOAT,[2],[1,2]); weight.data_location=TensorProto.EXTERNAL
    m.graph.initializer.append(weight)
    with pytest.raises(ValueError,match='external'): onnx_plan(OnnxAnalysisRequest(model=base64.b64encode(m.SerializeToString()).decode()),True)


def test_extended_http_apis_and_format_validation():
    from backend.main import app
    from backend.pool.routes import pool
    pool.jobs.clear(); pool.workers.clear()
    with TestClient(app) as client:
        report=client.post('/pool/python/analyze',json={'source':PYTHON,'auto_mark':True}).json()
        assert report['status']=='ready'
        assert client.post('/pool/wgsl/analyze',json={'source':WGSL,'count':10}).json()['status']=='ready'
        assert client.post('/pool/onnx/analyze',json={'model':model()}).json()['status']=='ready'
        assert client.post('/pool/onnx/analyze',json={'model':'bad'}).json()['status']=='unsupported'
        submitted=client.post('/pool/jobs',json={'kind':'python','source':report['marked_source'],'input':TypedArray.encode([1,2]).model_dump()})
        assert submitted.status_code==202
        job=pool.jobs[submitted.json()['job_id']]; worker=add_worker(pool); a=pool.pull(worker)
        header,payload=reply(a); header.output_format='u32'
        with pytest.raises(ValueError): pool.accept(worker,header,payload)
        assert job.image==bytes(8)
        assert client.get('/pool/assets/'+next(iter(job.assets))).status_code==200
        assert client.get('/pool/runtime/../../auth.py').status_code==404


def test_wgsl_guard_no_input_and_chunk_limits():
    source=WGSL.replace('@group(0) @binding(0) var<storage, read> values: array<f32>;','').replace('values[i] * 2.0 + 1.0','f32(i)')
    report=analyze_wgsl(WGSLAnalysisRequest(source=source, count=100, chunk_size=2))
    assert report.status=='ready' and report.input_dtype is None
    pool=Coordinator(); job=pool.create(WGSLRequest(kind='wgsl', wgsl=source, count=100, chunk_size=2))
    assert len(job.chunks)==50
    guarded=WGSL.replace('result[i] = values[i] * 2.0 + 1.0;', 'if (i < arrayLength(&result)) { result[i] = values[i] * 2.0 + 1.0; }')
    assert analyze_wgsl(WGSLAnalysisRequest(source=guarded)).status=='ready'
    assert analyze_wgsl(WGSLAnalysisRequest(source=source,count=2000000,chunk_size=1)).status=='unsupported'


def test_declared_shader_requires_independence_assertion():
    normalized=analyze_wgsl(WGSLAnalysisRequest(source=WGSL)).wgsl
    with pytest.raises(ValueError,match='independent=true'):
        Coordinator().create(WGSLRequest(kind='wgsl',wgsl=normalized,splitting='declared',input=TypedArray.encode([1,2])))
    job=Coordinator().create(WGSLRequest(kind='wgsl',wgsl=normalized,splitting='declared',independent=True,input=TypedArray.encode([1,2])))
    assert len(job.chunks)==1


def test_dense_onnx_with_changed_output_width_and_batch_mixing_rejection():
    graph=helper.make_graph([helper.make_node('MatMul',['X','W'],['Y'])], 'dense',
        [helper.make_tensor_value_info('X',TensorProto.FLOAT,['batch',2])],
        [helper.make_tensor_value_info('Y',TensorProto.FLOAT,['batch',3])],
        [helper.make_tensor('W',TensorProto.FLOAT,[2,3],[1,2,3,4,5,6])])
    encoded=base64.b64encode(helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10).SerializeToString()).decode()
    plan=onnx_plan(OnnxRequest(kind='onnx',model=encoded,input=TypedArray.encode(range(10)),input_shape=[5,2],batch_size=2,independent=True))
    assert plan['output_shape']==[5,3] and [c['byte_length'] for c in plan['chunks']]==[24,24,12]
    # A broadcast tensor with a per-sample row cannot be repeated in independently sized batches.
    graph=helper.make_graph([helper.make_node('Add',['X','W'],['Y'])], 'batch-bias',
        [helper.make_tensor_value_info('X',TensorProto.FLOAT,[2,2])],
        [helper.make_tensor_value_info('Y',TensorProto.FLOAT,[2,2])],
        [helper.make_tensor('W',TensorProto.FLOAT,[2,2],[1,2,3,4])])
    encoded=base64.b64encode(helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10).SerializeToString()).decode()
    with pytest.raises(ValueError,match='batch axis'):
        onnx_plan(OnnxAnalysisRequest(model=encoded,samples=4,batch_size=2),True)


def test_nonfinite_results_and_chunk_budgets():
    pool=Coordinator(); pool.create(WGSLRequest(kind='wgsl',wgsl=WGSL,input=TypedArray.encode([1,2])))
    worker=add_worker(pool); a=pool.pull(worker)
    with pytest.raises(ValueError,match='Non-finite'):
        pool.accept(worker,*reply(a,float('nan')))
    assert worker.busy is None
    with pytest.raises(ValueError,match='2048'):
        pool.create(WGSLRequest(kind='wgsl',wgsl=WGSL,count=2000000,chunk_size=1))
