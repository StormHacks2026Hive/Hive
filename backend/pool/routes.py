import asyncio
import contextlib
import time
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from ..models import JobCreated
from .models import (Incoming, Register, Registered, RequestChunk, Heartbeat,
    Control, ChunkError, ChunkStarted, Subscribe, JobRequest, JobStatus, WorkerStatus, Manifest)
from .coordinator import Coordinator, SHADER, SHADER_ID
from .protocol import decode_result, MAX_FRAME

router = APIRouter(prefix='/pool')
pool = Coordinator()

@router.get('/manifest', response_model=Manifest)
async def manifest():
    return Manifest(shader_id=SHADER_ID)

def get_job(job_id):
    job = pool.jobs.get(job_id)
    if not job:
        raise HTTPException(404, 'Unknown or expired image job')
    return job

@router.post('/jobs', response_model=JobCreated, status_code=202)
async def create_job(request: JobRequest):
    try:
        return JobCreated(job_id=pool.create(request).job_id)
    except ValueError as exc:
        raise HTTPException(429, str(exc)) from exc

@router.get('/jobs/{job_id}', response_model=JobStatus)
async def status(job_id: str):
    return pool.status(get_job(job_id))

@router.post('/jobs/{job_id}/cancel', response_model=JobStatus)
async def cancel(job_id: str):
    job = get_job(job_id)
    if job.status in ('queued','running'):
        pool.finish(job, 'cancelled')
        pool.publish(job)
        # Closing affected sockets destroys their devices before reconnecting.
        await pool.sweep()
    return pool.status(job)

@router.get('/workers', response_model=list[WorkerStatus])
async def workers():
    return pool.worker_status()

@router.get('/assets/{shader_id}')
async def asset(shader_id: str):
    if shader_id != SHADER_ID:
        raise HTTPException(404, 'Unknown shader')
    return Response(SHADER, media_type='text/plain',headers={'Cache-Control':'public, max-age=31536000, immutable'})

@router.get('/jobs/{job_id}/chunks/{chunk_id}')
async def chunk_result(job_id: str, chunk_id: str):
    job = get_job(job_id)
    chunk = next((c for c in job.chunks if c.chunk_id==chunk_id), None)
    if not chunk or chunk.output is None:
        raise HTTPException(404, 'Tile not ready')
    return Response(chunk.output, media_type='application/octet-stream')

@router.get('/jobs/{job_id}/result')
async def result(job_id: str):
    job = get_job(job_id)
    if job.status != 'done':
        raise HTTPException(409, 'Image is not complete')
    return Response(bytes(job.image), media_type='application/octet-stream', headers={
        'X-Image-Width': str(job.request.width), 'X-Image-Height': str(job.request.height), 'X-Pixel-Format':'rgba8'})

@router.websocket('/nodes')
async def nodes(socket: WebSocket):
    await socket.accept()
    worker = None
    try:
        first = await asyncio.wait_for(socket.receive_text(), 30)
        if len(first.encode()) > MAX_FRAME:
            raise ValueError('Registration too large')
        registration = Incoming.validate_json(first)
        if not isinstance(registration, Register):
            raise ValueError('Register first')
        worker = pool.register(socket, registration)
        await socket.send_json(Registered(worker_id=worker.worker_id).model_dump())
        while worker.worker_id in pool.workers:
            event = await asyncio.wait_for(socket.receive(), 25)
            if event['type']=='websocket.disconnect':
                break
            if event.get('bytes') is not None:
                header, payload = decode_result(event['bytes'])
                ack = pool.accept(worker, header, payload)
                await socket.send_json(ack.model_dump())
                if ack.disposition == 'stale' and not worker.active:
                    await socket.close(code=1013)
                    break
                continue
            text = event.get('text', '')
            if len(text.encode()) > MAX_FRAME:
                raise ValueError('Message too large')
            message = Incoming.validate_json(text)
            if isinstance(message, Heartbeat):
                worker.last_heartbeat = time.monotonic()
                worker.visible = message.visible
            elif isinstance(message, RequestChunk):
                await socket.send_json(pool.pull(worker).model_dump())
            elif isinstance(message, Control):
                if message.type=='stop':
                    break
                worker.active = message.type=='resume'
                pool.publish_workers()
            elif isinstance(message, ChunkError):
                job, chunk = pool.find(worker, message.chunk_id, message.attempt_id)
                if chunk:
                    pool.release(worker, message.error)
                if message.code in ('device_lost','timeout'):
                    break
            elif isinstance(message, ChunkStarted):
                pass  # Lease already includes bounded shader preparation.
            else:
                raise ValueError('Already registered')
    except WebSocketDisconnect:
        pass
    except (ValueError, asyncio.TimeoutError):
        with contextlib.suppress(Exception):
            await socket.close(code=1008)
    finally:
        if worker:
            pool.disconnect(worker.worker_id)
        with contextlib.suppress(Exception):
            await socket.close()

@router.websocket('/events')
async def events(socket: WebSocket):
    await socket.accept()
    job_id = None
    queue = asyncio.Queue(maxsize=64)
    try:
        raw = await asyncio.wait_for(socket.receive_text(), 15)
        if len(raw) > 4096:
            raise ValueError('Subscription too large')
        message = Subscribe.model_validate_json(raw)
        job = pool.jobs.get(message.job_id)
        if not job:
            raise ValueError('Unknown job')
        job_id = job.job_id
        pool.watchers.setdefault(job_id, set()).add(queue)
        await socket.send_json(pool.snapshot(job,'job_snapshot').model_dump())
        while True:
            try:
                update = await asyncio.wait_for(queue.get(), 10)
            except asyncio.TimeoutError:
                update = pool.snapshot(job).model_dump()
            await asyncio.wait_for(socket.send_json(update), 5)
    except (ValueError, WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
        with contextlib.suppress(Exception):
            await socket.close(code=1008)
    finally:
        if job_id:
            pool.watchers[job_id].discard(queue)
            if not pool.watchers[job_id]:
                pool.watchers.pop(job_id, None)
