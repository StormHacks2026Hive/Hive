"""Pull scheduling with short leases. All state changes happen on one event loop."""
import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
from .models import (Assignment, Image, Tile, AcceptedTile, WorkerStatus,
    Contribution, JobStatus, Snapshot, TileReady, NoWork, ResultAck)

SHADER = Path(__file__).with_name('mandelbrot.wgsl').read_bytes()
SHADER_ID = hashlib.sha256(SHADER).hexdigest()

@dataclass
class Chunk:
    chunk_id: str
    tile: Tile
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

    def create(self, request):
        if len(self.jobs) >= 16:
            raise ValueError('16 retained image jobs maximum; cancel or wait for 30-minute result expiry')
        chunks = [Chunk(f'tile-{y//64:02d}-{x//64:02d}', Tile(x=x, y=y))
            for y in range(0, request.height, 64) for x in range(0, request.width, 64)]
        job = Job(uuid4().hex, request, chunks, bytearray(request.width * request.height * 4))
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
            width=job.request.width, height=job.request.height, parameters=job.request.parameters,
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

    def compatible(self, worker):
        l = worker.capabilities.limits
        return (l.maxBufferSize >= 16384 and l.maxStorageBufferBindingSize >= 16384
            and l.maxUniformBufferBindingSize >= 48 and l.maxComputeWorkgroupSizeX >= 8
            and l.maxComputeWorkgroupSizeY >= 8 and l.maxComputeInvocationsPerWorkgroup >= 64
            and l.maxComputeWorkgroupsPerDimension >= 8)

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
        if not self.compatible(worker):
            return NoWork(reason='Device limits do not support this shader')
        for job in self.jobs.values():
            if job.status not in ('queued','running'):
                continue
            for chunk in job.chunks:
                if chunk.output is not None or chunk.worker_id is not None:
                    continue
                if chunk.previous_worker == worker.worker_id and any(w.worker_id != worker.worker_id and w.active and w.visible and not w.busy and self.compatible(w) for w in self.workers.values()):
                    continue
                chunk.attempts += 1
                chunk.worker_id, chunk.attempt_id = worker.worker_id, uuid4().hex
                chunk.deadline = time.monotonic() + self.lease
                worker.busy = (job.job_id, chunk.chunk_id)
                job.status = 'running'
                self.publish(job)
                return Assignment(job_id=job.job_id, chunk_id=chunk.chunk_id,
                    attempt_id=chunk.attempt_id, shader_id=SHADER_ID, tile=chunk.tile,
                    image=Image(width=job.request.width,height=job.request.height),
                    parameters=job.request.parameters, timeout_ms=int(self.lease*1000))
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
        if len(payload) != chunk.tile.width * chunk.tile.height * 4 or header.byte_length != len(payload):
            self.release(worker, 'Invalid RGBA tile length')
            raise ValueError('Invalid RGBA tile length')
        chunk.output = payload
        chunk.accepted = AcceptedTile(chunk_id=chunk.chunk_id, tile=chunk.tile,
            url=f'/pool/jobs/{job.job_id}/chunks/{chunk.chunk_id}', worker_id=worker.worker_id,
            worker_label=worker.label, elapsed_ms=header.elapsed_ms)
        # Only byte placement occurs on the server, never fractal computation.
        stride = chunk.tile.width * 4
        for row in range(chunk.tile.height):
            start = ((chunk.tile.y+row)*job.request.width+chunk.tile.x)*4
            job.image[start:start+stride] = payload[row*stride:(row+1)*stride]
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
