import asyncio
import json
import struct
import time
import pytest
from fastapi.testclient import TestClient
from backend.pool.coordinator import CapacityError, Coordinator
from backend.pool.models import JobRequest, Register, ResultHeader
from backend.pool.protocol import decode_result

CAPABILITIES = {
    'webgpu': True, 'adapter': {'description':'Test GPU'}, 'features': [],
    'limits': {'maxBufferSize':268435456, 'maxStorageBufferBindingSize':134217728,
        'maxUniformBufferBindingSize':65536,'maxComputeWorkgroupSizeX':256,
        'maxComputeWorkgroupSizeY':256, 'maxComputeInvocationsPerWorkgroup':256,
        'maxComputeWorkgroupsPerDimension':65535},
    'benchmark': {'version':'mandelbrot-v1','pixels':4096,'elapsed_ms':1.5},
}
class Socket:
    def __init__(self): self.closed=False
    async def close(self,code=1000): self.closed=True

def add_worker(pool,label='test', capabilities=None):
    return pool.register(Socket(),Register(type='register',label=label,capabilities=capabilities or CAPABILITIES))

def result(assignment, pixels=None):
    pixels = pixels if pixels is not None else bytes([5,10,15,255])*4096
    return ResultHeader(type='chunk_result',job_id=assignment.job_id,
        chunk_id=assignment.chunk_id,attempt_id=assignment.attempt_id,
        output_format='rgba8',byte_length=len(pixels),elapsed_ms=2),pixels

def frame(header,payload):
    raw=header.model_dump_json().encode()
    return struct.pack('<I',len(raw))+raw+payload


def test_exactly_64_nonoverlapping_tiles():
    pool=Coordinator(); job=pool.create(JobRequest())
    assert len(job.chunks)==64
    covered=set()
    for c in job.chunks:
        indices={(y,x) for y in range(c.tile.y,c.tile.y+64) for x in range(c.tile.x,c.tile.x+64)}
        assert not covered & indices
        covered |= indices
    assert len(covered)==512*512


def test_out_of_order_byte_assembly_and_duplicate():
    pool=Coordinator(); job=pool.create(JobRequest()); a=add_worker(pool,'A'); b=add_worker(pool,'B')
    first=pool.pull(a); second=pool.pull(b)
    assert first.chunk_id != second.chunk_id
    assert pool.accept(b,*result(second,bytes([9,8,7,255])*4096)).disposition=='accepted'
    assert pool.accept(a,*result(first)).disposition=='accepted'
    assert pool.accept(a,*result(first)).disposition=='duplicate'
    assert job.image[:4]==bytes([5,10,15,255])
    assert job.image[64*4:64*4+4]==bytes([9,8,7,255])
    assert pool.status(job).completed_chunks==2
    assert sum(c.chunks for c in job.contributions.values())==2


@pytest.mark.parametrize('reason',['disconnect','timeout','heartbeat','stop','device_lost'])
def test_reassign_stale_result_and_lease(reason):
    pool=Coordinator(); job=pool.create(JobRequest()); a=add_worker(pool,'A'); b=add_worker(pool,'B')
    old=pool.pull(a)
    if reason=='timeout': job.chunks[0].deadline=0; asyncio.run(pool.sweep())
    elif reason=='heartbeat': a.last_heartbeat=0; asyncio.run(pool.sweep())
    else: pool.disconnect(a.worker_id,reason)
    new=pool.pull(b)
    assert old.chunk_id==new.chunk_id and old.attempt_id!=new.attempt_id
    assert pool.accept(a,*result(old)).disposition=='stale'
    assert b.busy and pool.status(job).completed_chunks==0
    assert pool.accept(b,*result(new)).disposition=='accepted'
    assert pool.status(job).retries==1


def test_expired_result_not_accepted_without_sweep():
    pool=Coordinator(); job=pool.create(JobRequest()); a=add_worker(pool)
    assignment=pool.pull(a); job.chunks[0].deadline=0
    assert pool.accept(a,*result(assignment)).disposition=='stale'
    assert pool.status(job).completed_chunks==0 and not a.active


def test_four_attempts_fail_job():
    pool=Coordinator(); job=pool.create(JobRequest())
    for _ in range(4):
        worker=add_worker(pool); pool.pull(worker); pool.disconnect(worker.worker_id)
    assert job.status=='failed' and 'retry limit' in job.error


def test_visibility_pause_capability_and_pull_busy():
    pool=Coordinator(); job=pool.create(JobRequest()); worker=add_worker(pool)
    worker.visible=False; assert pool.pull(worker).type=='no_work'
    worker.visible=True; worker.active=False; assert pool.pull(worker).type=='no_work'
    worker.active=True; worker.capabilities.limits.maxComputeWorkgroupSizeX=4
    assert pool.pull(worker).type=='no_work'
    worker.capabilities.limits.maxComputeWorkgroupSizeX=256
    assert pool.pull(worker).type=='assign_chunk'
    assert pool.pull(worker).type=='no_work'
    assert sum(c.attempts for c in job.chunks)==1


def test_invalid_result_requeues():
    pool=Coordinator(); job=pool.create(JobRequest()); worker=add_worker(pool); assignment=pool.pull(worker)
    with pytest.raises(ValueError): pool.accept(worker,*result(assignment,b'abcd'))
    assert worker.busy is None and pool.status(job).completed_chunks==0


def test_complete_image_and_watch_snapshot():
    pool=Coordinator(); job=pool.create(JobRequest()); worker=add_worker(pool)
    queue=asyncio.Queue(maxsize=4); pool.watchers[job.job_id]={queue}
    for _ in range(64): pool.accept(worker,*result(pool.pull(worker)))
    status=pool.status(job)
    assert job.status=='done' and status.progress==1 and status.result_url
    assert len(job.image)==512*512*4 and worker.completed==64
    updates=[]
    while not queue.empty(): updates.append(queue.get_nowait())
    assert updates[-1]['type']=='job_done'
    assert len(updates[-1]['job']['tiles'])==64


def test_binary_frame_validation():
    pool=Coordinator(); pool.create(JobRequest()); assignment=pool.pull(add_worker(pool))
    header,payload=result(assignment)
    assert decode_result(frame(header,payload))==(header,payload)
    for invalid in [b'abc',b'x'*32769,struct.pack('<I',5000),frame(header,payload)[:-1]]:
        with pytest.raises(ValueError): decode_result(invalid)


def test_job_limits_and_expiry():
    with pytest.raises(ValueError): JobRequest(width=8192)
    with pytest.raises(ValueError): JobRequest(parameters={'max_iterations':5000})
    with pytest.raises(ValueError): JobRequest(parameters={'xmin':2,'xmax':1})
    pool=Coordinator()
    for _ in range(16): pool.create(JobRequest())
    with pytest.raises(ValueError): pool.create(JobRequest())
    job=next(iter(pool.jobs.values())); pool.finish(job,'cancelled'); job.finished_at=time.monotonic()-1801
    asyncio.run(pool.sweep()); assert len(pool.jobs)==15


def test_terminal_results_are_reclaimed_instead_of_blocking_new_jobs():
    pool = Coordinator(max_retained_jobs=2)
    oldest = pool.create(JobRequest())
    pool.finish(oldest, 'done')
    newer = pool.create(JobRequest())
    pool.finish(newer, 'cancelled')
    latest = pool.create(JobRequest())
    assert oldest.job_id not in pool.jobs
    assert newer.job_id in pool.jobs and latest.job_id in pool.jobs
    for _ in range(25):
        pool.finish(latest, 'done')
        latest = pool.create(JobRequest())
    assert len(pool.jobs) == 2


def test_capacity_preserves_subscribed_results_and_cancelled_live_leases():
    pool = Coordinator(max_retained_jobs=1)
    job = pool.create(JobRequest())
    pool.finish(job, 'done')
    pool.watchers[job.job_id] = {asyncio.Queue()}
    with pytest.raises(CapacityError): pool.create(JobRequest())
    assert job.job_id in pool.jobs
    pool.watchers.clear()
    running = pool.create(JobRequest())
    worker = add_worker(pool)
    pool.pull(worker)
    pool.finish(running, 'cancelled')
    with pytest.raises(CapacityError): pool.create(JobRequest())
    asyncio.run(pool.sweep())
    pool.create(JobRequest())
    assert running.job_id not in pool.jobs


def test_result_memory_pressure_reclaims_only_when_submission_can_fit():
    pool = Coordinator(max_retained_bytes=3 * 1048576)
    old = pool.create(JobRequest())  # Image plus reserved chunk copies = 2 MiB.
    pool.finish(old, 'cancelled')  # Uncomputed chunk copies are no longer reserved.
    active = pool.create(JobRequest())
    with pytest.raises(CapacityError): pool.create(JobRequest())
    assert set(pool.jobs) == {old.job_id, active.job_id}
    pool.finish(active, 'cancelled')
    incoming = pool.create(JobRequest())
    assert set(pool.jobs) == {active.job_id, incoming.job_id}


def test_http_and_binary_websocket(monkeypatch):
    monkeypatch.setenv("HIVE_ALLOW_LEGACY_POOL", "true")
    from backend.main import app
    from backend.pool.routes import pool
    pool.jobs.clear(); pool.workers.clear(); pool.watchers.clear()
    with TestClient(app) as client:
        assert client.get('/pool/manifest').status_code==200
        assert client.post('/pool/jobs',json={'parameters':{'max_iterations':1025}}).status_code==422
        job_id=client.post('/pool/jobs',json={}).json()['job_id']
        with client.websocket_connect('/pool/events') as events, client.websocket_connect('/pool/nodes') as ws:
            events.send_json({'v':1,'type':'subscribe','job_id':job_id})
            assert events.receive_json()['type']=='job_snapshot'
            ws.send_json(Register(type='register',label='Test laptop',capabilities=CAPABILITIES).model_dump())
            assert ws.receive_json()['type']=='registered'
            ws.send_json({'v':1,'type':'request_chunk'})
            from backend.pool.models import Assignment
            assignment=Assignment(**ws.receive_json())
            ws.send_bytes(frame(*result(assignment)))
            assert ws.receive_json()['disposition']=='accepted'
            for _ in range(10):
                update=events.receive_json()
                if update['type']=='tile_ready': break
            assert update['type']=='tile_ready'
            assert len(client.get(update['tile']['url']).content)==16384
            assert client.get('/pool/jobs/'+job_id).json()['completed_chunks']==1
            assert client.post('/pool/jobs/'+job_id+'/cancel').json()['status']=='cancelled'
        assert client.get('/pool/jobs/'+job_id+'/result').status_code==409
        assert client.get('/pool/jobs/missing').status_code==404
