"""Pull scheduling with short leases. All state changes happen on one event loop."""
import asyncio
import hashlib
import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
from .models import (Assignment, Tile, AcceptedTile, WorkerStatus,
    Contribution, JobStatus, Snapshot, TileReady, NoWork, ResultAck)

SHADER = Path(__file__).with_name('mandelbrot.wgsl').read_bytes()
SHADER_ID = hashlib.sha256(SHADER).hexdigest()

@dataclass
class Chunk:
    chunk_id: str
    tile: Tile | None
    assignment: dict = field(default_factory=dict)
    byte_length: int = 16384
    offset: int = 0
    count: int = 4096
    frame_index: int = 0
    attempts: int = 0
    worker_id: str | None = None
    previous_worker: str | None = None
    attempt_id: str | None = None
    deadline: float = 0
    output: bytes | None = None
    accepted: AcceptedTile | None = None

@dataclass
class Job:
    job_id: str
    request: object
    chunks: list[Chunk]
    image: bytearray
    assets: dict = field(default_factory=dict)
    output_format: str = 'rgba8'
    output_shape: list = field(default_factory=list)
    status: str = 'queued'
    error: str | None = None
    contributions: dict = field(default_factory=dict)
    finished_at: float | None = None

@dataclass
class Worker:
    worker_id: str
    socket: object
    label: str
    capabilities: object
    last_heartbeat: float
    active: bool = True
    visible: bool = True
    busy: tuple | None = None
    completed: int = 0
    compute_ms: float = 0

class Coordinator:
    def __init__(self, lease=15, heartbeat_timeout=20, max_attempts=4):
        self.jobs = {}
        self.workers = {}
        self.watchers = {}
        self.lease = lease
        self.heartbeat_timeout = heartbeat_timeout
        self.max_attempts = max_attempts

    def create(self, request, plan=None):
        if len(self.jobs) >= 16:
            raise ValueError('16 retained jobs maximum; wait for 30-minute result expiry')
        from .workloads import AnimationRequest, WGSLRequest, PythonRequest, compute_plan, onnx_plan
        if request.kind in ('mandelbrot', 'animation'):
            frames = request.frames if isinstance(request, AnimationRequest) else [request.parameters]
            step = 512 if isinstance(request, AnimationRequest) and request.distribution == 'frames' else 64
            chunks = []
            for frame_index, parameters in enumerate(frames):
                for y in range(0, request.height, step):
                    for x in range(0, request.width, step):
                        chunk_id = f'tile-{y//64:02d}-{x//64:02d}'
                        if request.kind == 'animation':
                            chunk_id = f'frame-{frame_index:03d}-' + chunk_id
                        tile = Tile(x=x, y=y, width=step, height=step)
                        chunks.append(Chunk(chunk_id, tile, frame_index=frame_index,
                            byte_length=step*step*4, count=step*step,
                            assignment={'shader_id':SHADER_ID, 'tile':tile.model_dump(),
                                'image':{'width':request.width,'height':request.height},
                                'parameters':parameters.model_dump(), 'frame_index':frame_index}))
            job = Job(uuid4().hex, request, chunks, bytearray(request.width * request.height * 4 * len(frames)),
                output_shape=[len(frames),request.height,request.width,4])
        else:
            plan = plan or (compute_plan(request) if isinstance(request, (WGSLRequest, PythonRequest)) else onnx_plan(request))
            chunks = [Chunk(tile=None, **c) for c in plan['chunks']]
            job = Job(uuid4().hex, request, chunks, bytearray(plan['size']), assets=plan['assets'],
                output_format=plan['output_format'], output_shape=plan['output_shape'])
        if sum(len(j.image) + sum(len(a) for a in j.assets.values()) for j in self.jobs.values()) + len(job.image) + sum(len(a) for a in job.assets.values()) > 134_217_728:
            raise ValueError('128 MiB retained output/asset capacity reached; wait for result expiry')
        self.jobs[job.job_id] = job
        return job

    def worker_status(self):
        return [WorkerStatus(worker_id=w.worker_id, label=w.label,
            state='working' if w.busy else 'idle' if w.active and w.visible else 'paused',
            visible=w.visible, capabilities=w.capabilities, completed_chunks=w.completed,
            compute_ms=w.compute_ms,
            pixels_per_second=w.completed*4096/(w.compute_ms/1000) if w.compute_ms else 0)
            for w in self.workers.values()]

    def status(self, job):
        accepted = [c.accepted for c in job.chunks if c.accepted]
        return JobStatus(job_id=job.job_id, status=job.status, progress=len(accepted)/len(job.chunks),
            completed_chunks=len(accepted), total_chunks=len(job.chunks),
            kind=job.request.kind, frame_count=len(getattr(job.request, 'frames', [None])),
            fps=getattr(job.request, 'fps', 12), output_format=job.output_format, output_shape=job.output_shape,
            width=getattr(job.request, 'width', 0), height=getattr(job.request, 'height', 0), parameters=getattr(job.request, 'parameters', None),
            tiles=accepted, contributions=list(job.contributions.values()),
            retries=sum(max(0, c.attempts-1) for c in job.chunks), error=job.error,
            result_url=f'/pool/jobs/{job.job_id}/result' if job.status=='done' else None)

    def snapshot(self, job, kind='job_update'):
        return Snapshot(type=kind, job=self.status(job), workers=self.worker_status())

    def emit(self, job_id, message):
        for queue in self.watchers.get(job_id, set()):
            # Snapshots include all accepted tiles, so clients recover dropped updates.
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(message.model_dump())

    def publish(self, job):
        kind = 'job_done' if job.status=='done' else 'job_failed' if job.status=='failed' else 'job_update'
        self.emit(job.job_id, self.snapshot(job, kind))

    def publish_workers(self):
        for job in self.jobs.values():
            if job.job_id in self.watchers:
                self.publish(job)

    def register(self, socket, message):
        if len(self.workers) >= 64:
            raise ValueError('Worker capacity reached')
        w = Worker(uuid4().hex, socket, message.label, message.capabilities, time.monotonic())
        self.workers[w.worker_id] = w
        self.publish_workers()
        return w

    def compatible(self, worker, chunk=None):
        l = worker.capabilities.limits
        if chunk is None or chunk.tile is not None:
            size = chunk.byte_length if chunk else 16384
            groups = math.ceil((chunk.tile.width if chunk else 64)/8)
            return (l.maxBufferSize >= size and l.maxStorageBufferBindingSize >= size
                and l.maxUniformBufferBindingSize >= 48 and l.maxComputeWorkgroupSizeX >= 8
                and l.maxComputeWorkgroupSizeY >= 8 and l.maxComputeInvocationsPerWorkgroup >= 64
                and l.maxComputeWorkgroupsPerDimension >= groups)
        if chunk.assignment['kind'] == 'onnx_batch':
            input_size = math.prod(chunk.assignment['input_shape'])*4
            return worker.capabilities.onnx and min(l.maxBufferSize, l.maxStorageBufferBindingSize) >= max(input_size,chunk.byte_length)
        return (min(l.maxBufferSize,l.maxStorageBufferBindingSize) >= chunk.byte_length
            and l.maxUniformBufferBindingSize >= 16 and l.maxComputeWorkgroupSizeX >= 64
            and l.maxComputeInvocationsPerWorkgroup >= 64 and l.maxComputeWorkgroupsPerDimension >= math.ceil(chunk.count/64))

    def find(self, worker, chunk_id, attempt_id):
        if not worker.busy:
            return None, None
        job_id, current_chunk = worker.busy
        job = self.jobs.get(job_id)
        if not job or job.status not in ('queued','running'):
            return None, None
        chunk = next((c for c in job.chunks if c.chunk_id==current_chunk), None)
        if chunk and chunk.chunk_id==chunk_id and chunk.attempt_id==attempt_id and chunk.worker_id==worker.worker_id:
            return job, chunk
        return None, None

    def finish(self, job, status, error=None):
        job.status, job.error, job.finished_at = status, error, time.monotonic()

    def release(self, worker, error):
        if not worker.busy:
            return
        job_id, chunk_id = worker.busy
        job = self.jobs.get(job_id)
        worker.busy = None
        if not job:
            return
        chunk = next(c for c in job.chunks if c.chunk_id==chunk_id)
        chunk.previous_worker = worker.worker_id
        chunk.worker_id, chunk.attempt_id = None, None
        if chunk.attempts >= self.max_attempts and job.status in ('queued','running'):
            self.finish(job, 'failed', f'{chunk_id}: retry limit reached ({error})')
        self.publish(job)

    def disconnect(self, worker_id, error='Worker disconnected'):
        worker = self.workers.pop(worker_id, None)
        if worker:
            self.release(worker, error)
            self.publish_workers()

    def pull(self, worker):
        if worker.worker_id not in self.workers or worker.busy or not worker.active or not worker.visible:
            return NoWork(reason='Worker is paused or already working')
        for job in self.jobs.values():
            if job.status not in ('queued','running'):
                continue
            for chunk in job.chunks:
                if chunk.output is not None or chunk.worker_id is not None or not self.compatible(worker, chunk):
                    continue
                if chunk.previous_worker == worker.worker_id and any(w.worker_id != worker.worker_id and w.active and w.visible and not w.busy and self.compatible(w, chunk) for w in self.workers.values()):
                    continue
                chunk.attempts += 1
                chunk.worker_id, chunk.attempt_id = worker.worker_id, uuid4().hex
                duration = max(60, self.lease) if job.request.kind == 'onnx' else self.lease
                chunk.deadline = time.monotonic() + duration
                worker.busy = (job.job_id, chunk.chunk_id)
                job.status = 'running'
                self.publish(job)
                return Assignment(job_id=job.job_id, chunk_id=chunk.chunk_id,
                    attempt_id=chunk.attempt_id, timeout_ms=int(duration*1000), **chunk.assignment)
        return NoWork()

    def accept(self, worker, header, payload):
        job, chunk = self.find(worker, header.chunk_id, header.attempt_id)
        if job is None or header.job_id != job.job_id:
            previous_job = self.jobs.get(header.job_id)
            previous = next((c for c in previous_job.chunks if c.chunk_id == header.chunk_id), None) if previous_job else None
            if previous and previous.accepted and previous.attempt_id == header.attempt_id and previous.accepted.worker_id == worker.worker_id:
                return ResultAck(chunk_id=header.chunk_id, attempt_id=header.attempt_id, disposition='duplicate')
            return ResultAck(chunk_id=header.chunk_id, attempt_id=header.attempt_id, disposition='stale')
        if chunk.deadline <= time.monotonic():
            self.release(worker, 'Result arrived after lease expiry')
            worker.active = False
            return ResultAck(chunk_id=header.chunk_id, attempt_id=header.attempt_id, disposition='stale')
        if len(payload) != chunk.byte_length or header.byte_length != len(payload) or header.output_format != job.output_format:
            self.release(worker, 'Invalid output length or format')
            raise ValueError('Invalid output length or format')
        if job.output_format == 'f32' and any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f', payload)):
            self.release(worker, 'Non-finite float output')
            raise ValueError('Non-finite float output')
        chunk.output = payload
        chunk.accepted = AcceptedTile(chunk_id=chunk.chunk_id, tile=chunk.tile, frame_index=chunk.frame_index, offset=chunk.offset, count=chunk.count,
            url=f'/pool/jobs/{job.job_id}/chunks/{chunk.chunk_id}', worker_id=worker.worker_id,
            worker_label=worker.label, elapsed_ms=header.elapsed_ms)
        # Only byte placement occurs on the coordinator.
        if chunk.tile is not None:
            stride = chunk.tile.width * 4
            frame_start = chunk.frame_index * job.request.width * job.request.height * 4
            for row in range(chunk.tile.height):
                start = frame_start + ((chunk.tile.y+row)*job.request.width+chunk.tile.x)*4
                job.image[start:start+stride] = payload[row*stride:(row+1)*stride]
        else:
            start = chunk.offset*4
            job.image[start:start+len(payload)] = payload
        worker.busy = None
        worker.completed += 1
        worker.compute_ms += header.elapsed_ms
        contribution = job.contributions.setdefault(worker.worker_id, Contribution(worker_id=worker.worker_id,label=worker.label,chunks=0,elapsed_ms=0))
        contribution.chunks += 1
        contribution.elapsed_ms += header.elapsed_ms
        if all(c.output is not None for c in job.chunks):
            self.finish(job, 'done')
        self.emit(job.job_id, TileReady(job_id=job.job_id,tile=chunk.accepted))
        self.publish(job)
        return ResultAck(chunk_id=chunk.chunk_id,attempt_id=header.attempt_id,disposition='accepted')

    async def sweep(self):
        now = time.monotonic()
        for worker in list(self.workers.values()):
            expired = False
            if worker.busy:
                job = self.jobs.get(worker.busy[0])
                c = next((c for c in job.chunks if c.chunk_id==worker.busy[1]), None) if job else None
                expired = c is not None and c.deadline <= now
                # Job completion/cancellation invalidates all outstanding leases.
                if job and job.status in ('failed','cancelled'):
                    expired = True
            if expired or now-worker.last_heartbeat > self.heartbeat_timeout:
                self.disconnect(worker.worker_id, 'Lease or heartbeat timeout')
                try:
                    await asyncio.wait_for(worker.socket.close(code=1013), 1)
                except Exception:
                    pass
        for job_id, job in list(self.jobs.items()):
            if job.finished_at and now-job.finished_at > 1800 and not self.watchers.get(job_id):
                self.jobs.pop(job_id)

    async def run(self):
        while True:
            await self.sweep()
            await asyncio.sleep(.25)
