import asyncio
import time
import pytest
from fastapi.testclient import TestClient
from backend.compiler import compile_source, validate_job
from backend.models import JobRequest, TypedArray, Limits, Register, ChunkResult, Summary
from backend.chunker import make_chunks, capacity
from backend.store import MemoryStore, Job
from backend.scheduler import Scheduler, Node
from backend.aggregator import aggregate
from demos.elementwise import KERNEL as ELEMENTWISE
from demos.monte_carlo import KERNEL as PI

LIMITS = Limits(max_buffer_size=1048576, max_storage_buffer_binding_size=1048576,
    max_compute_workgroup_size_x=256, max_compute_invocations_per_workgroup=256,
    max_compute_workgroups_per_dimension=65535)

class Socket:
    def __init__(self): self.messages = []
    async def send_json(self, msg): self.messages.append(msg)
    async def close(self, code=1000): pass

def request(mode='parameter-only', **kw):
    return JobRequest(kernel=PI if mode == 'parameter-only' else ELEMENTWISE, mode=mode, **kw)

def job_for(req):
    return Job('job', req, compile_source(req.kernel), req.input.decode() if req.input else None)

def nodes(n=2):
    return [Node(str(i), Socket(), LIMITS, time.monotonic()) for i in range(n)]

def test_source_translation_and_cache():
    k = compile_source(PI)
    assert k is compile_source(PI)
    assert 'var<uniform>' in k.wgsl
    assert k.buffers[0].element_type == 'u32'
    validate_job(request(), k)

@pytest.mark.parametrize('source', [
    'import os\ndef bad():\n    pass',
    '@danger()\ndef bad():\n    pass',
    'def bad():\n    __import__("os")',
    'def bad():\n    while True:\n        pass',
    'def bad(n: u32):\n    for i in range(n):\n        pass',
    'def bad():\n    for i in range(1025):\n        pass',
    'def bad():\n    for i in range(999999999999999999999999999):\n        pass',
    'def bad():\n    for i in range(0, 10, 0):\n        pass',
])
def test_reject_unsafe_or_unsupported(source):
    with pytest.raises(ValueError): compile_source(source)

def test_slice_offsets_and_capability():
    values = list(range(10000))
    req = request('data-slice', input=TypedArray.encode(values), parameters={'scale':2.,'bias':1.})
    job = job_for(req)
    chunks = make_chunks('job', req, job.kernel, values, nodes())
    assert len(chunks) == 2
    assert [v for c in chunks for v in c.message.input.decode()] == values
    assert [(c.message.offset, c.message.count) for c in chunks] == [(0,5000),(5000,5000)]
    assert all(c.message.count <= capacity(LIMITS, req.mode) for c in chunks)

def test_parameter_ranges_seeds():
    req = request(count=10000)
    chunks = make_chunks('job', req, compile_source(PI), None, nodes())
    assert len({c.message.seed for c in chunks}) == len(chunks)
    assert all(c.message.input is None for c in chunks)
    assert [i for c in chunks for i in range(c.message.offset, c.message.offset+c.message.count)] == list(range(10000))

def test_tiny_single_chunk():
    req = request(count=100)
    assert len(make_chunks('job', req, compile_source(PI), None, nodes())) == 1

@pytest.mark.parametrize('op,expected', [('sum',12),('count',12),('min',4),('max',8),('mean',.0012)])
def test_reduce_merging(op, expected):
    req = request(count=10000, reduce=op)
    chunks = make_chunks('job', req, compile_source(PI), None, nodes())
    results = {c.message.chunk_id: ChunkResult(type='chunk_result', chunk_id=c.message.chunk_id, attempt_id='a', summary=Summary(value=v,count=c.message.count)) for c,v in zip(chunks,[4,8])}
    assert aggregate(chunks, results, op, 'u32') == pytest.approx(expected)

def test_aggregate_order():
    req = request(count=10000)
    chunks = make_chunks('job', req, compile_source(PI), None, nodes())
    results = {c.message.chunk_id: ChunkResult(type='chunk_result', chunk_id=c.message.chunk_id, attempt_id='a', output=TypedArray.encode([c.message.offset]*c.message.count, 'u32')) for c in reversed(chunks)}
    assert aggregate(list(reversed(chunks)), results, None, 'u32').decode() == [0]*5000+[5000]*5000

@pytest.mark.parametrize('failure', ['disconnect','timeout','device_lost'])
def test_reassign_and_stale_result(failure):
    async def run():
        store = MemoryStore(); scheduler = Scheduler(store)
        job = job_for(request(count=100, reduce='sum')); store.add(job)
        a,b = nodes(); scheduler.nodes = {a.node_id:a,b.node_id:b}
        await scheduler.tick()
        chunk = job.chunks[0]; old = chunk.message.attempt_id
        if failure == 'timeout':
            chunk.deadline = 0
        elif failure == 'device_lost':
            scheduler.fail_attempt(a, chunk, 'device lost'); scheduler.disconnect(a.node_id)
        else: scheduler.disconnect(a.node_id)
        await scheduler.tick()
        assert chunk.assigned == b.node_id and chunk.message.attempt_id != old
        scheduler.result(a, ChunkResult(type='chunk_result', chunk_id=chunk.message.chunk_id, attempt_id=old, summary=Summary(value=3,count=100)))
        assert not job.results and b.busy
        scheduler.result(b, ChunkResult(type='chunk_result', chunk_id=chunk.message.chunk_id, attempt_id=chunk.message.attempt_id, summary=Summary(value=3,count=100)))
        assert job.status == 'done' and job.result == 3
    asyncio.run(run())

def test_retry_exhaustion():
    async def run():
        store=MemoryStore(); scheduler=Scheduler(store,max_attempts=1)
        job=job_for(request()); store.add(job)
        a,b=nodes(); scheduler.nodes={a.node_id:a,b.node_id:b}
        await scheduler.tick(); scheduler.disconnect(a.node_id); await scheduler.tick()
        assert job.status == 'failed'
    asyncio.run(run())

@pytest.mark.parametrize('second, expected', [(3.000001,'done'),(4,'failed')])
def test_two_node_verification(second, expected):
    async def run():
        store=MemoryStore(); scheduler=Scheduler(store)
        job=job_for(request(count=100,reduce='sum',verify=True)); store.add(job)
        a,b=nodes(); scheduler.nodes={a.node_id:a,b.node_id:b}
        await scheduler.tick()
        c=job.chunks[0]
        scheduler.result(a,ChunkResult(type='chunk_result',chunk_id=c.message.chunk_id,attempt_id=c.message.attempt_id,summary=Summary(value=3,count=100)))
        await scheduler.tick()
        assert c.assigned == b.node_id
        scheduler.result(b,ChunkResult(type='chunk_result',chunk_id=c.message.chunk_id,attempt_id=c.message.attempt_id,summary=Summary(value=second,count=100)))
        assert job.status == expected
    asyncio.run(run())

def test_http_and_native_websocket():
    from backend.main import app, store, scheduler
    store.jobs.clear(); scheduler.nodes.clear()
    with TestClient(app) as client:
        assert client.post('/jobs',json={'kernel':'import os','mode':'parameter-only'}).status_code==422
        created=client.post('/jobs',json=request(count=100,reduce='sum').model_dump())
        assert created.status_code==202
        job_id=created.json()['job_id']
        with client.websocket_connect('/nodes') as socket:
            socket.send_json(Register(type='register',webgpu=True,limits=LIMITS).model_dump())
            assert socket.receive_json()['type']=='registered'
            assignment=socket.receive_json()
            socket.send_json({'type':'chunk_result','chunk_id':assignment['chunk_id'],'attempt_id':assignment['attempt_id'],'summary':{'value':78,'count':100}})
            for _ in range(50):
                status=client.get('/jobs/'+job_id).json()
                if status['status']=='done': break
                time.sleep(.01)
            assert status['result']==78
        assert client.get('/node/').status_code==200

def test_invalid_result_retried_not_accepted():
    async def run():
        store=MemoryStore(); scheduler=Scheduler(store)
        job=job_for(request(count=100)); store.add(job)
        a=nodes(1)[0]; scheduler.nodes={a.node_id:a}
        await scheduler.tick(); c=job.chunks[0]
        scheduler.result(a,ChunkResult(type='chunk_result',chunk_id=c.message.chunk_id,attempt_id=c.message.attempt_id,output=TypedArray.encode([1], 'u32')))
        assert not c.done and c.assigned is None and not job.results
        await scheduler.tick()
        assert c.attempts == 2
    asyncio.run(run())

def test_http_payload_limit_and_array_encoding():
    from backend.main import app, store, scheduler
    store.jobs.clear(); scheduler.nodes.clear()
    with TestClient(app) as client:
        assert client.post('/jobs',content=b'x'*12_000_001,headers={'content-type':'application/json'}).status_code == 413
        payload=request('data-slice',input=TypedArray(dtype='f32',data='invalid'),parameters={'scale':2.,'bias':1.}).model_dump()
        assert client.post('/jobs',json=payload).status_code == 422
        assert client.get('/jobs/missing').status_code == 404
