"""Network API connections and persisted, server-driven whole-task timers."""
import asyncio
import base64
import json
import logging
import secrets
import struct
import time
from types import SimpleNamespace
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import Field, TypeAdapter

from . import database as db
from .auth import User
from .models import Model
from .networks import access
from .pool import routes
from .pool.coordinator import CapacityError
from .pool.workloads import JobSubmission, OnnxRequest, onnx_plan
from .programs import ProgramRequest, create_run

router = APIRouter(prefix='/api/networks/{network_id}', tags=['automation'])
external = APIRouter(prefix='/api/v1/networks/{network_id}', tags=['external API'])
logger = logging.getLogger(__name__)
timer_inflight = set()


class KeyCreate(Model):
    name: str = Field(min_length=1, max_length=64)
    permission: str = Field(default='read', pattern='^(read|run)$')
    expires_days: int = Field(default=30, ge=1, le=365)


class TimerCreate(Model):
    run_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=80)
    interval_seconds: int = Field(default=300, ge=0, le=31_536_000)
    first_run: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class TimerControl(Model):
    action: str = Field(pattern='^(pause|resume|delete|run)$')


def key_user(request, network_id, write=False):
    authorization = request.headers.get('Authorization', '')
    scheme, _, token = authorization.partition(' ')
    if scheme.lower() != 'bearer' or not token:
        raise HTTPException(401, 'Supply Authorization: Bearer YOUR_API_KEY', headers={'WWW-Authenticate': 'Bearer'})
    key = db.query('SELECT * FROM api_keys WHERE token_hash=? AND network_id=? AND revoked=0 AND expires>?',
                   (db.session_hash(token), network_id, time.time()), one=True)
    if not key or not db.member(network_id, key['user_id']):
        raise HTTPException(401, 'API key is invalid, expired, or revoked.')
    if write and key['permission'] != 'run':
        raise HTTPException(403, 'This key is read-only. Create a key with submit access.')
    user = db.query('SELECT id,name,email FROM users WHERE id=?', (key['user_id'],), one=True)
    return User(**user)


@router.get('/api-keys')
def keys(network_id: str, request: Request):
    user, _ = access(request, network_id)
    return db.query('SELECT id,name,permission,created,expires FROM api_keys WHERE network_id=? AND user_id=? AND revoked=0 ORDER BY created DESC',
                    (network_id, user.id))


@router.post('/api-keys', status_code=201)
def new_key(network_id: str, payload: KeyCreate, request: Request, response: Response):
    user, _ = access(request, network_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(422, 'Enter a connection name.')
    token, key_id, now = 'hive_' + secrets.token_urlsafe(32), uuid4().hex, time.time()
    expires = now + payload.expires_days * 86400
    db.execute('INSERT INTO api_keys(id,token_hash,network_id,user_id,name,permission,created,expires) VALUES(?,?,?,?,?,?,?,?)',
               (key_id, db.session_hash(token), network_id, user.id, name, payload.permission, now, expires))
    response.headers['Cache-Control'] = 'no-store'
    return {'id': key_id, 'name': name, 'key': token, 'permission': payload.permission, 'expires': expires}


@router.post('/api-keys/{key_id}/revoke')
def revoke_key(network_id: str, key_id: str, request: Request):
    user, _ = access(request, network_id)
    db.execute('UPDATE api_keys SET revoked=1 WHERE id=? AND network_id=? AND user_id=?', (key_id, network_id, user.id))
    return {'revoked': True}


def saved_run(network_id, run_id):
    row = db.query('SELECT r.*,i.payload FROM runs r LEFT JOIN run_inputs i ON r.id=i.run_id WHERE r.id=? AND r.network_id=?',
                   (run_id, network_id), one=True)
    if not row:
        raise HTTPException(404, 'Unknown task.')
    row['jobs'] = json.loads(row['jobs'])
    if not row['payload']:
        # Recover older tasks while their jobs are still retained. CPU plans
        # contain the validated interpreter program; host Python is never run.
        jobs = [routes.pool.jobs.get(j['job_id']) for j in row['jobs']]
        if jobs and all(j and j.network_id == network_id for j in jobs):
            segments = []
            for meta, job in zip(row['jobs'], jobs):
                if job.request.kind == 'cpu':
                    plan = {'chunks': [{'chunk_id': c.chunk_id, 'offset': c.offset, 'count': c.count,
                                        'byte_length': c.byte_length, 'assignment': c.assignment} for c in job.chunks],
                            'size': len(job.image), 'output_format': job.output_format, 'output_shape': job.output_shape,
                            'assets': {k: base64.b64encode(v).decode() for k, v in job.assets.items()},
                            'reduction': job.reduction, 'initial': job.initial}
                    segments.append({'request': {'kind': 'cpu'}, 'plan': plan, 'name': meta['name'], 'target': meta['target']})
                else:
                    segments.append({'request': job.request.model_dump(mode='json'), 'name': meta['name'], 'target': meta['target']})
            row['payload'] = json.dumps({'kind': 'replay', 'segments': segments})
            db.execute('INSERT OR IGNORE INTO run_inputs VALUES(?,?)', (run_id, row['payload']))
    return row


def task_summary(row):
    payload = json.loads(row['payload']) if row['payload'] else None
    detail = payload.get('request', {}) if payload else {}
    mode = 'ONNX inference' if payload and payload['kind'] == 'onnx' else 'Animation' if detail.get('mode') == 'animation' else 'Compute'
    input_count = None
    if detail.get('input'):
        value = detail['input']
        input_count = len(base64.b64decode(value['data'])) // (8 if value['dtype'] == 'f64' else 4)
    jobs = []
    for meta in row['jobs']:
        job = routes.pool.jobs.get(meta['job_id'])
        description = f"Saved {meta['target'].upper()} segment."
        if job:
            if job.output_format == 'rgba8':
                description = f"Renders {len(getattr(job.request, 'frames', [None]))} frames at {job.request.width} × {job.request.height}."
            elif job.request.kind == 'onnx':
                description = f"Runs ONNX inference in {len(job.chunks)} batches."
            elif job.reduction:
                description = f"Computes independent CPU chunks and combines them using {job.reduction}."
            else:
                description = f"Computes {job.output_shape[0]:,} output values across {len(job.chunks)} chunks."
        jobs.append(meta | {'status': job.status if job else 'expired', 'kind': job.request.kind if job else None,
                            'output_shape': job.output_shape if job else [], 'progress': routes.pool.status(job).progress if job else 0,
                            'description': description})
    suffix = f"{detail.get('frames', 1)} frames · {detail.get('width', 0)} × {detail.get('height', 0)}" if mode == 'Animation' else f"{input_count or detail.get('count', '')} elements" if input_count or detail.get('count') else 'Saved task'
    return {k: row[k] for k in ('id', 'name', 'created', 'user_id')} | {
        'jobs': jobs, 'repeatable': bool(payload), 'description': f"{mode} · {suffix} · {len(jobs)} segment{'s' if len(jobs) != 1 else ''}"}


def task_list(network_id):
    rows = db.query('SELECT id FROM runs WHERE network_id=? ORDER BY created DESC LIMIT 50', (network_id,))
    return [task_summary(saved_run(network_id, row['id'])) for row in rows]


@router.get('/tasks')
def tasks(network_id: str, request: Request):
    access(request, network_id)
    return task_list(network_id)


async def repeat_task(network_id, run_id, user_id):
    row = saved_run(network_id, run_id)
    if not row['payload']:
        raise HTTPException(410, 'This older task has expired. Send it again from Compute to save its inputs.')
    payload = json.loads(row['payload'])
    if payload['kind'] == 'program':
        return create_run(network_id, ProgramRequest.model_validate(payload['request']), user_id)
    created = []
    try:
        segments = payload['segments'] if payload['kind'] == 'replay' else [{'request': payload['request'], 'name': 'ONNX inference', 'target': 'gpu'}]
        for segment in segments:
            spec = segment['request']
            if spec['kind'] == 'cpu':
                job_request = SimpleNamespace(kind='cpu')
                plan = segment['plan'] | {'assets': {k: base64.b64decode(v) for k, v in segment['plan']['assets'].items()}}
            else:
                job_request = TypeAdapter(JobSubmission).validate_python(spec)
                plan = await asyncio.to_thread(onnx_plan, job_request) if spec['kind'] == 'onnx' else None
            job = routes.pool.create(job_request, plan=plan, network_id=network_id, owner_id=user_id)
            created.append({'job_id': job.job_id, 'name': segment['name'], 'target': segment['target']})
        return {'id': db.save_run(network_id, user_id, row['name'], created, payload), 'jobs': created}
    except Exception as exc:
        for job in created:
            routes.pool.jobs.pop(job['job_id'], None)
        if isinstance(exc, ValueError):
            raise HTTPException(429 if isinstance(exc, CapacityError) else 422, str(exc)) from exc
        raise


@router.post('/tasks/{run_id}/repeat', status_code=202)
async def repeat(network_id: str, run_id: str, request: Request):
    user, _ = access(request, network_id)
    return await repeat_task(network_id, run_id, user.id)


@router.get('/timers')
def timers(network_id: str, request: Request):
    user, network = access(request, network_id)
    rows = db.query('SELECT t.*,r.name AS task_name FROM task_timers t JOIN runs r ON t.run_id=r.id WHERE t.network_id=? ORDER BY t.created DESC', (network_id,))
    for row in rows:
        row['can_control'] = user.id in (row['user_id'], network['owner_id'])
        last = db.query('SELECT jobs FROM runs WHERE id=?', (row['last_run_id'],), one=True) if row['last_run_id'] else None
        row['jobs'] = [j | {'status': routes.pool.jobs[j['job_id']].status if j['job_id'] in routes.pool.jobs else 'expired'} for j in json.loads(last['jobs'])] if last else []
    return rows


@router.post('/timers', status_code=201)
def new_timer(network_id: str, payload: TimerCreate, request: Request):
    user, _ = access(request, network_id)
    if 0 < payload.interval_seconds < 10 or not payload.name.strip():
        raise HTTPException(422, 'Enter a name and use an interval of at least 10 seconds, or zero for once.')
    if not saved_run(network_id, payload.run_id)['payload']:
        raise HTTPException(410, 'Send this task again from Compute before scheduling it.')
    timer_id, now = uuid4().hex, time.time()
    first = payload.first_run if payload.first_run is not None else now + (payload.interval_seconds or 60)
    db.execute('INSERT INTO task_timers(id,network_id,user_id,run_id,name,interval_seconds,next_run,created) VALUES(?,?,?,?,?,?,?,?)',
               (timer_id, network_id, user.id, payload.run_id, payload.name.strip(), payload.interval_seconds, first, now))
    return {'id': timer_id}


def timer_active(row):
    last = db.query('SELECT jobs FROM runs WHERE id=?', (row['last_run_id'],), one=True) if row['last_run_id'] else None
    return bool(last and any(j['job_id'] in routes.pool.jobs and routes.pool.jobs[j['job_id']].status in ('queued', 'running') for j in json.loads(last['jobs'])))


@router.post('/timers/{timer_id}/control')
async def control_timer(network_id: str, timer_id: str, payload: TimerControl, request: Request):
    user, network = access(request, network_id)
    row = db.query('SELECT * FROM task_timers WHERE id=? AND network_id=?', (timer_id, network_id), one=True)
    if not row:
        raise HTTPException(404, 'Unknown timer.')
    if user.id not in (row['user_id'], network['owner_id']):
        raise HTTPException(403, 'Only the timer owner or network owner can change it.')
    if payload.action == 'delete':
        db.execute('DELETE FROM task_timers WHERE id=?', (timer_id,))
    elif payload.action == 'pause':
        db.execute('UPDATE task_timers SET enabled=0 WHERE id=?', (timer_id,))
    elif payload.action == 'resume':
        db.execute('UPDATE task_timers SET enabled=1,next_run=?,last_error=\'\' WHERE id=?', (time.time() + (row['interval_seconds'] or 60), timer_id))
    else:
        if timer_id in timer_inflight or timer_active(row):
            raise HTTPException(409, 'The previous whole task is still running.')
        timer_inflight.add(timer_id)
        try:
            run = await repeat_task(network_id, row['run_id'], row['user_id'])
            db.execute('UPDATE task_timers SET last_run_id=?,last_fired=?,last_error=\'\' WHERE id=?', (run['id'], time.time(), timer_id))
        finally:
            timer_inflight.discard(timer_id)
        return run
    return {'ok': True}


async def tick(now=None):
    now = time.time() if now is None else now
    for row in db.query('SELECT * FROM task_timers WHERE enabled=1 AND next_run<=? ORDER BY next_run LIMIT 20', (now,)):
        # Claim before yielding, preventing duplicate firings in concurrent ticks.
        with db.connection() as conn:
            claimed = conn.execute('UPDATE task_timers SET next_run=? WHERE id=? AND enabled=1 AND next_run=?',
                                   (now + (row['interval_seconds'] or 60), row['id'], row['next_run'])).rowcount
        if not claimed:
            continue
        if not db.member(row['network_id'], row['user_id']):
            db.execute('UPDATE task_timers SET enabled=0,last_error=? WHERE id=?', ('Network membership ended.', row['id']))
            continue
        active = row['id'] in timer_inflight or timer_active(row)
        if active or not any(w.network_id == row['network_id'] and w.active and w.visible for w in routes.pool.workers.values()):
            reason = 'Waiting for the previous task to finish.' if active else 'Waiting for a connected contributor.'
            db.execute('UPDATE task_timers SET next_run=?,last_error=? WHERE id=?', (now + min(30, row['interval_seconds'] or 30), reason, row['id']))
            continue
        timer_inflight.add(row['id'])
        try:
            run = await repeat_task(row['network_id'], row['run_id'], row['user_id'])
            db.execute('UPDATE task_timers SET last_run_id=?,last_fired=?,last_error=\'\',enabled=?,next_run=CASE WHEN interval_seconds=0 THEN 0 ELSE next_run END WHERE id=?',
                       (run['id'], now, int(bool(row['interval_seconds'])), row['id']))
        except (ValueError, HTTPException) as exc:
            db.execute('UPDATE task_timers SET last_error=?,enabled=? WHERE id=?',
                       (str(exc.detail if isinstance(exc, HTTPException) else exc)[:500], int(bool(row['interval_seconds'])), row['id']))
        finally:
            timer_inflight.discard(row['id'])


async def run_timers():
    while True:
        try:
            await tick()
        except Exception:
            logger.exception('Timer tick failed')
        await asyncio.sleep(2)


@external.get('/runs')
def external_tasks(network_id: str, request: Request):
    key_user(request, network_id)
    return task_list(network_id)


@external.post('/runs', status_code=202)
async def external_submit(network_id: str, payload: ProgramRequest, request: Request):
    user = key_user(request, network_id, write=True)
    return create_run(network_id, payload, user.id)


@external.post('/runs/{run_id}/repeat', status_code=202)
async def external_repeat(network_id: str, run_id: str, request: Request):
    user = key_user(request, network_id, write=True)
    try:
        return await repeat_task(network_id, run_id, user.id)
    except ValueError as exc:
        raise HTTPException(429 if isinstance(exc, CapacityError) else 422, str(exc)) from exc


@external.post('/onnx', status_code=202)
async def external_onnx(network_id: str, payload: OnnxRequest, request: Request):
    user = key_user(request, network_id, write=True)
    job = None
    try:
        plan = await asyncio.to_thread(onnx_plan, payload)
        job = routes.pool.create(payload, plan=plan, network_id=network_id, owner_id=user.id)
        jobs = [{'job_id': job.job_id, 'name': 'ONNX inference', 'target': 'gpu'}]
        run_id = db.save_run(network_id, user.id, 'ONNX model', jobs, {'kind': 'onnx', 'request': payload.model_dump(mode='json')})
        return {'id': run_id, 'jobs': jobs}
    except Exception as exc:
        if job:
            routes.pool.jobs.pop(job.job_id, None)
        if isinstance(exc, ValueError):
            raise HTTPException(429 if isinstance(exc, CapacityError) else 422, str(exc)) from exc
        raise


def external_job(network_id, job_id, request, control=False):
    user = key_user(request, network_id, write=control)
    job = routes.pool.jobs.get(job_id)
    if not job or job.network_id != network_id:
        raise HTTPException(404, 'Unknown or expired job.')
    if control and user.id not in (job.owner_id, db.member(network_id, user.id)['owner_id']):
        raise HTTPException(403, 'Only the sender or network owner can cancel this job.')
    return job


@external.get('/jobs/{job_id}')
def external_status(network_id: str, job_id: str, request: Request):
    return routes.pool.status(external_job(network_id, job_id, request))


@external.get('/jobs/{job_id}/result')
def external_result(network_id: str, job_id: str, request: Request, format: str = 'json'):
    job = external_job(network_id, job_id, request)
    if job.status != 'done':
        raise HTTPException(409, 'Job is not complete.')
    if format == 'binary':
        return Response(bytes(job.image), media_type='application/octet-stream', headers={
            'X-Pixel-Format': job.output_format, 'X-Output-Shape': ','.join(map(str, job.output_shape)), 'Cache-Control': 'no-store'})
    if format != 'json' or job.output_format == 'rgba8':
        raise HTTPException(422, 'Use ?format=binary for rendered RGBA frames.')
    scalar = {'f32': 'f', 'f64': 'd', 'i32': 'i', 'u32': 'I'}[job.output_format]
    import math
    values = [item[0] for item in struct.iter_unpack('<' + scalar, job.image)]
    # GPU float outputs may contain NaN/Infinity; JSON has no such numbers.
    values = [v if not isinstance(v, float) or math.isfinite(v) else None for v in values]
    return JSONResponse({'job_id': job_id, 'format': job.output_format, 'shape': job.output_shape, 'values': values}, headers={'Cache-Control': 'no-store'})


@external.post('/jobs/{job_id}/cancel')
async def external_cancel(network_id: str, job_id: str, request: Request):
    job = external_job(network_id, job_id, request, control=True)
    if job.status in ('queued', 'running'):
        routes.pool.finish(job, 'cancelled')
        routes.pool.publish(job)
        await routes.pool.sweep()
    return routes.pool.status(job)
