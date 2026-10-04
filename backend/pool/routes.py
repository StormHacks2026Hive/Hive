import asyncio
import contextlib
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from ..models import JobCreated
from .models import (Incoming, Register, Registered, RequestChunk, Heartbeat,
    Control, ChunkError, ChunkStarted, Subscribe, JobStatus, WorkerStatus, Manifest)
from .coordinator import Coordinator, SHADER, SHADER_ID
from .workloads import JobSubmission, OnnxAnalysisRequest, onnx_plan
from .image_workloads import ImageAnalysisRequest, ImageAnalysis, analyze_image
from ..marked_python import AnalysisRequest, Analysis, analyze_marked
from ..wgsl_analyzer import WGSLAnalysisRequest, WGSLAnalysis, analyze_wgsl
from .protocol import decode_result, MAX_FRAME
from .access import network_context, job_access, legacy_enabled
from .. import database as db

RUNTIME_DIR = Path(__file__).resolve().parents[2] / 'HiveFrontend/node_modules/onnxruntime-web/dist'
if not RUNTIME_DIR.is_dir():
    RUNTIME_DIR = Path(__file__).resolve().parents[2] / 'Frontend/HiveFrontend/node_modules/onnxruntime-web/dist'

router = APIRouter(prefix='/pool')
pool = Coordinator()

@router.get('/manifest', response_model=Manifest)
async def manifest():
    return Manifest(shader_id=SHADER_ID, onnx_runtime_ready=(RUNTIME_DIR / 'ort.webgpu.bundle.min.mjs').is_file())

def get_job(job_id):
    job = pool.jobs.get(job_id)
    if not job:
        raise HTTPException(404, 'Unknown or expired job')
    return job

@router.post('/jobs', response_model=JobCreated, status_code=202)
async def create_job(request: JobSubmission, http: Request):
    user, network_id = network_context(http)
    try:
        if request.kind == 'onnx':
            plan = await asyncio.to_thread(onnx_plan, request)
            return JobCreated(job_id=pool.create(request, plan=plan, network_id=network_id, owner_id=user.id if user else None).job_id)
        return JobCreated(job_id=pool.create(request, network_id=network_id, owner_id=user.id if user else None).job_id)
    except ValueError as exc:
        raise HTTPException(429 if 'retained' in str(exc) else 422, str(exc)) from exc

@router.get('/jobs/{job_id}', response_model=JobStatus)
async def status(job_id: str, request: Request):
    job_access(request, get_job(job_id))
    return pool.status(get_job(job_id))

@router.post('/jobs/{job_id}/cancel', response_model=JobStatus)
async def cancel(job_id: str, request: Request):
    job = get_job(job_id)
    job_access(request, job, control=True)
    if job.status in ('queued','running'):
        pool.finish(job, 'cancelled')
        pool.publish(job)
        # Closing affected sockets destroys their devices before reconnecting.
        await pool.sweep()
    return pool.status(job)

@router.get('/workers', response_model=list[WorkerStatus])
async def workers(request: Request):
    _, network_id = network_context(request)
    return pool.worker_status(network_id)

@router.get('/assets/{shader_id}')
async def asset(shader_id: str, request: Request):
    data = SHADER if shader_id == SHADER_ID else next((job.assets[shader_id] for job in pool.jobs.values() if shader_id in job.assets), None)
    if data is None:
        raise HTTPException(404, 'Unknown or expired asset')
    if shader_id != SHADER_ID and not legacy_enabled():
        from ..auth import require_user
        user = require_user(request)
        if not any(shader_id in j.assets and j.network_id and db.member(j.network_id, user.id) for j in pool.jobs.values()):
            raise HTTPException(404, 'Unknown asset')
    return Response(data, media_type='application/octet-stream', headers={'Cache-Control':'private, no-store'})

@router.get('/jobs/{job_id}/chunks/{chunk_id}')
async def chunk_result(job_id: str, chunk_id: str, request: Request):
    job = get_job(job_id)
    job_access(request, job)
    chunk = next((c for c in job.chunks if c.chunk_id==chunk_id), None)
    if not chunk or chunk.output is None:
        raise HTTPException(404, 'Tile not ready')
    return Response(chunk.output, media_type='application/octet-stream')

@router.get('/jobs/{job_id}/result')
async def result(job_id: str, request: Request):
    job = get_job(job_id)
    job_access(request, job)
    if job.status != 'done':
        raise HTTPException(409, 'Job is not complete')
    return Response(bytes(job.image), media_type='application/octet-stream', headers={
        'X-Image-Width': str(getattr(job.request, 'width', 0)), 'X-Image-Height': str(getattr(job.request, 'height', 0)),
        'X-Pixel-Format':job.output_format, 'X-Output-Shape':','.join(map(str,job.output_shape))})

@router.post('/python/analyze', response_model=Analysis)
async def python_analysis(request: AnalysisRequest):
    return analyze_marked(request)

@router.post('/wgsl/analyze', response_model=WGSLAnalysis)
async def wgsl_analysis(request: WGSLAnalysisRequest):
    return analyze_wgsl(request)

@router.post('/wgsl/image/analyze', response_model=ImageAnalysis)
async def image_analysis(request: ImageAnalysisRequest):
    return analyze_image(request)

@router.post('/onnx/analyze')
async def onnx_analysis(request: OnnxAnalysisRequest):
    try:
        return await asyncio.to_thread(onnx_plan, request, True)
    except ValueError as exc:
        return {'status':'unsupported', 'findings':[str(exc)]}

@router.get('/runtime/{filename}')
async def runtime_asset(filename: str):
    # Serve only the pinned runtime's public entry and WASM companions.
    allowed = {'ort.webgpu.bundle.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm', 'ort-wasm-simd-threaded.jspi.mjs', 'ort-wasm-simd-threaded.jspi.wasm'}
    if filename not in allowed or not (RUNTIME_DIR / filename).is_file():
        raise HTTPException(404, 'ONNX runtime asset unavailable; install frontend dependencies')
    from fastapi.responses import FileResponse
    return FileResponse(RUNTIME_DIR / filename, media_type='application/wasm' if filename.endswith('.wasm') else 'text/javascript')

@router.get('/jobs/{job_id}/frames/{frame_index}')
async def animation_frame(job_id: str, frame_index: int, request: Request):
    job = get_job(job_id)
    job_access(request, job)
    if job.request.kind not in ('animation', 'mandelbrot', 'wgsl_image'):
        raise HTTPException(422, 'This job does not produce images')
    chunks = [c for c in job.chunks if c.frame_index == frame_index]
    if not chunks:
        raise HTTPException(404, 'Unknown frame')
    if any(c.output is None for c in chunks):
        raise HTTPException(409, 'Frame is not complete')
    size = job.request.width * job.request.height * 4
    return Response(bytes(job.image[frame_index*size:(frame_index+1)*size]), media_type='application/octet-stream')

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
        user, _ = network_context(socket, registration.network_id)
        if user:
            node = db.query('SELECT * FROM nodes WHERE id=? AND network_id=? AND user_id=?', (registration.node_id, registration.network_id, user.id), one=True)
            if not node or node['mode'] == 'off':
                raise ValueError('Node must be enrolled and started first')
            if any(w.node_id == node['id'] for w in pool.workers.values()):
                raise ValueError('This node is already connected in another tab')
            registration = registration.model_copy(update={'label': node['label']})
        worker = pool.register(socket, registration, user_id=user.id if user else None)
        if user:
            worker.active = node['mode'] == 'running'
        await socket.send_json(Registered(worker_id=worker.worker_id, mode='running' if worker.active else 'paused').model_dump())
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
                worker.phase = message.phase
                db.save_node(worker)
            elif isinstance(message, RequestChunk):
                await socket.send_json(pool.pull(worker).model_dump())
            elif isinstance(message, Control):
                if message.type=='stop':
                    break
                persisted = db.query('SELECT mode FROM nodes WHERE id=?', (worker.node_id,), one=True) if worker.node_id else None
                worker.active = message.type=='resume' and (not persisted or persisted['mode'] == 'running')
                pool.publish_workers()
            elif isinstance(message, ChunkError):
                job, chunk = pool.find(worker, message.chunk_id, message.attempt_id)
                if chunk:
                    pool.release(worker, message.error)
                if message.code in ('device_lost','timeout'):
                    break
            elif isinstance(message, ChunkStarted):
                worker.phase = 'working'
            else:
                raise ValueError('Already registered')
    except WebSocketDisconnect:
        pass
    except (ValueError, asyncio.TimeoutError, HTTPException):
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
        job_access(socket, job)
        job_id = job.job_id
        pool.watchers.setdefault(job_id, set()).add(queue)
        await socket.send_json(pool.snapshot(job,'job_snapshot').model_dump())
        while True:
            try:
                update = await asyncio.wait_for(queue.get(), 10)
            except asyncio.TimeoutError:
                update = pool.snapshot(job).model_dump()
            await asyncio.wait_for(socket.send_json(update), 5)
    except (ValueError, WebSocketDisconnect, asyncio.TimeoutError, RuntimeError, HTTPException):
        with contextlib.suppress(Exception):
            await socket.close(code=1008)
    finally:
        if job_id:
            pool.watchers[job_id].discard(queue)
            if not pool.watchers[job_id]:
                pool.watchers.pop(job_id, None)
