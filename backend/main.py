import asyncio
from . import config  # Load .env before routers and middleware read settings.
import contextlib
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError, TypeAdapter
from typing import Annotated, Union
from pydantic import Field
from .models import JobRequest, JobCreated, JobStatus, Register, Heartbeat, ChunkResult, ChunkError, MAX_FRAME
from .compiler import compile_source, validate_job
from .store import MemoryStore, Job
from .scheduler import Scheduler
from .converter import SourceRequest, CompatibilityReport, KernelValidation, analyze, validate_preview
from .pool.routes import router as pool_router, pool
from .auth import router as auth_router
from .networks import router as network_router
from .programs import router as program_router

store = MemoryStore()
scheduler = Scheduler(store)
Incoming = TypeAdapter(Annotated[Union[Register, Heartbeat, ChunkResult, ChunkError], Field(discriminator='type')])

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(scheduler.run())
    pool_task = asyncio.create_task(pool.run())
    yield
    task.cancel()
    pool_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    with contextlib.suppress(asyncio.CancelledError):
        await pool_task

app = FastAPI(lifespan=lifespan)


@app.middleware('http')
async def google_popup_policy(request: Request, call_next):
    """Keep Google sign-in popups able to communicate with their opener."""
    response = await call_next(request)
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin-allow-popups'
    return response


app.include_router(pool_router)
app.include_router(auth_router)
app.include_router(network_router)
app.include_router(program_router)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(','), allow_credentials=True, allow_methods=['GET','POST'], allow_headers=['Content-Type','X-CSRF-Token'])

@app.post('/kernels/analyze', response_model=CompatibilityReport)
async def analyze_kernel(request: SourceRequest):
    return analyze(request.source)

@app.post('/kernels/validate', response_model=KernelValidation)
async def validate_kernel(request: SourceRequest):
    return validate_preview(request.source)

@app.middleware('http')
async def body_limit(request, call_next):
    if request.method == 'POST':
        size = 0
        parts = []
        async for part in request.stream():
            size += len(part)
            if size > MAX_FRAME:
                from fastapi.responses import JSONResponse
                return JSONResponse({'detail':'Payload exceeds 12 MB'}, status_code=413)
            parts.append(part)
        request._body = b''.join(parts)
    return await call_next(request)

@app.post('/jobs', response_model=JobCreated, status_code=202)
async def create_job(request: JobRequest):
    if len(store.jobs) >= 32:
        raise HTTPException(429, 'In-memory job capacity reached; restart server to clear retained jobs')
    try:
        kernel = compile_source(request.kernel)
        validate_job(request, kernel)
        values = request.input.decode() if request.input else None
        if values is not None and not values:
            raise ValueError('Input array must not be empty')
        total = len(values) if values is not None else request.count
        # Cap total shader loop work, in addition to per-element compiler limits.
        import ast
        loops = [len(range(*(a.value for a in n.iter.args))) for n in ast.walk(ast.parse(request.kernel)) if isinstance(n, ast.For)]
        import math
        if total * max(1, math.prod(loops)) > 20_000_000:
            raise ValueError('Job exceeds 20 million iteration budget')
        job = Job(uuid4().hex, request, kernel, values)
        store.add(job)
        return JobCreated(job_id=job.job_id)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(422, str(exc)) from exc

@app.get('/jobs/{job_id}', response_model=JobStatus)
async def get_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, 'Unknown job')
    return store.status(job)

@app.websocket('/nodes')
async def nodes(socket: WebSocket):
    await socket.accept()
    node = None
    try:
        while True:
            raw = await asyncio.wait_for(socket.receive_text(), 35)
            if len(raw.encode()) > MAX_FRAME:
                await socket.close(code=1009)
                break
            message = Incoming.validate_json(raw)
            if node is None:
                if not isinstance(message, Register):
                    raise ValueError('First message must be register')
                node = await scheduler.register(socket, message)
            elif node.node_id not in scheduler.nodes:
                break
            elif isinstance(message, Heartbeat):
                node.last_heartbeat = time.monotonic()
            elif isinstance(message, ChunkResult):
                scheduler.result(node, message)
            elif isinstance(message, ChunkError):
                job, chunk = scheduler.find(node, message)
                if chunk:
                    scheduler.fail_attempt(node, chunk, message.error)
                if message.device_lost:
                    break
            else:
                raise ValueError('Already registered')
    except WebSocketDisconnect:
        pass
    except (ValueError, ValidationError, asyncio.TimeoutError, KeyError):
        with contextlib.suppress(Exception):
            await socket.close(code=1008)
    finally:
        if node:
            scheduler.disconnect(node.node_id)

app.mount('/node', StaticFiles(directory=Path(__file__).resolve().parents[1] / 'node-web', html=True), name='node')
app.mount('/legacy-node', StaticFiles(directory=Path(__file__).resolve().parents[1] / 'legacy-node-web', html=True), name='legacy-node')
app.mount('/shared', StaticFiles(directory=Path(__file__).resolve().parents[1] / 'shared'), name='shared')
# Building the frontend before starting uvicorn enables a single-origin deployment.
frontend_dist = Path(__file__).resolve().parents[1] / 'HiveFrontend/dist'
if not frontend_dist.is_dir():
    frontend_dist = Path(__file__).resolve().parents[1] / 'Frontend/HiveFrontend/dist'
if frontend_dist.is_dir():
    app.mount('/', StaticFiles(directory=frontend_dist, html=True), name='submitter')
