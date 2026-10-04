"""API credentials, complete task replay, and durable timer execution."""
import asyncio
import base64
import json
import struct
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import auth, automation, database as db, networks, programs
from backend.models import TypedArray
from backend.pool import routes
from backend.pool.coordinator import Coordinator
from backend.pool.models import Register
from tests.test_pool import CAPABILITIES, Socket
from tests.test_programs import WGSL, CPU


@pytest.fixture
def setup(monkeypatch):
    pool = Coordinator()
    monkeypatch.setattr(routes, 'pool', pool)
    app = FastAPI()
    for router in (auth.router, networks.router, programs.router, routes.router, automation.router, automation.external):
        app.include_router(router)
    with TestClient(app) as client:
        csrf = client.get('/auth/config').json()['csrf_token']
        headers = {'X-CSRF-Token': csrf}
        assert client.post('/auth/guest', json={}, headers=headers).status_code == 200
        network = client.post('/api/networks', json={'name': 'Timer hive', 'password': 'test-password'}, headers=headers).json()
        yield client, network['id'], headers, pool


def new_key(client, network, headers, permission='run'):
    response = client.post(f'/api/networks/{network}/api-keys', json={'name': 'test connection', 'permission': permission}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def whole_task(client, network, headers, source=None):
    # Two independent shaders are one whole task, not a single job.
    source = source or f'first = {WGSL!r}\nsecond = {WGSL!r}\n'
    response = client.post(f'/api/networks/{network}/runs', json={
        'source': source, 'filename': 'whole.py', 'input': TypedArray.encode([1, 2, 3, 4]).model_dump()}, headers=headers)
    assert response.status_code == 202, response.text
    return response.json()


def worker(pool, network):
    return pool.register(Socket(), Register(type='register', label='test', capabilities=CAPABILITIES, network_id=network))


def test_key_hash_scope_read_permissions_revocation_and_expiry(setup):
    client, network, csrf, pool = setup
    run = whole_task(client, network, csrf)
    key = new_key(client, network, csrf, 'read')
    bearer = {'Authorization': 'Bearer ' + key['key']}
    stored = db.query('SELECT * FROM api_keys WHERE id=?', (key['id'],), one=True)
    assert stored['token_hash'] == db.session_hash(key['key']) and key['key'] not in json.dumps(stored)
    listed = client.get(f'/api/networks/{network}/api-keys').json()
    assert 'key' not in listed[0] and 'token_hash' not in listed[0]
    assert client.get(f'/api/v1/networks/{network}/runs').status_code == 401
    assert client.get(f'/api/v1/networks/{network}/runs', headers=bearer).status_code == 200
    assert client.post(f'/api/v1/networks/{network}/runs/{run["id"]}/repeat', headers=bearer).status_code == 403
    assert client.get('/api/v1/networks/another/runs', headers=bearer).status_code == 401
    job_url = f'/api/v1/networks/{network}/jobs/{run["jobs"][0]["job_id"]}'
    assert client.get(job_url, headers=bearer).status_code == 200
    assert client.get(job_url + '/result', headers=bearer).status_code == 409
    assert client.post(f'/api/networks/{network}/api-keys', json={'name': 'no csrf'}).status_code == 403
    db.execute('UPDATE api_keys SET expires=0 WHERE id=?', (key['id'],))
    assert client.get(job_url, headers=bearer).status_code == 401
    key = new_key(client, network, csrf)
    bearer = {'Authorization': 'Bearer ' + key['key']}
    assert client.post(f'/api/networks/{network}/api-keys/{key["id"]}/revoke', json={}, headers=csrf).status_code == 200
    assert client.get(job_url, headers=bearer).status_code == 401


def test_external_submit_complete_replay_and_full_json_after_restart(setup, monkeypatch):
    client, network, csrf, pool = setup
    key = new_key(client, network, csrf)
    bearer = {'Authorization': 'Bearer ' + key['key']}
    response = client.post(f'/api/v1/networks/{network}/runs', json={
        'source': f'first = {WGSL!r}\nsecond = {WGSL!r}\n',
        'filename': 'two.py', 'input': TypedArray.encode(range(8205)).model_dump()}, headers=bearer)
    assert response.status_code == 202, response.text
    run = response.json()
    assert len(run['jobs']) == 2
    job = pool.jobs[run['jobs'][0]['job_id']]
    job.image[:] = struct.pack('<' + 'f' * 8205, *range(8205))
    job.status = 'done'
    url = f'/api/v1/networks/{network}/jobs/{job.job_id}/result'
    result = client.get(url, headers=bearer)
    assert result.json()['values'] == list(range(8205))
    assert result.json()['shape'] == [8205]
    assert client.get(url + '?format=binary', headers=bearer).content == bytes(job.image)
    restarted = Coordinator()
    monkeypatch.setattr(routes, 'pool', restarted)
    repeated = client.post(f'/api/v1/networks/{network}/runs/{run["id"]}/repeat', headers=bearer)
    assert repeated.status_code == 202, repeated.text
    assert len(repeated.json()['jobs']) == 2
    assert all(j.request.input.decode() == list(range(8205)) for j in restarted.jobs.values())
    assert all(j.owner_id == db.query('SELECT user_id FROM api_keys WHERE id=?', (key['id'],), one=True)['user_id'] for j in restarted.jobs.values())


def test_timer_whole_task_no_overlap_pause_resume_and_once(setup):
    client, network, csrf, pool = setup
    run = whole_task(client, network, csrf)
    base = f'/api/networks/{network}'
    timer = client.post(base + '/timers', json={'run_id': run['id'], 'name': 'Every minute', 'interval_seconds': 60, 'first_run': 100}, headers=csrf).json()
    worker(pool, network)
    asyncio.run(automation.tick(101))
    row = db.query('SELECT * FROM task_timers WHERE id=?', (timer['id'],), one=True)
    last = db.query('SELECT * FROM runs WHERE id=?', (row['last_run_id'],), one=True)
    assert len(json.loads(last['jobs'])) == 2
    assert row['next_run'] == 161
    count = len(pool.jobs)
    asyncio.run(automation.tick(162))
    assert len(pool.jobs) == count
    assert 'previous task' in db.query('SELECT last_error FROM task_timers WHERE id=?', (timer['id'],), one=True)['last_error']
    assert client.post(base + f'/timers/{timer["id"]}/control', json={'action': 'run'}, headers=csrf).status_code == 409
    for job in pool.jobs.values():
        pool.finish(job, 'cancelled')
    asyncio.run(automation.tick(193))
    assert len(pool.jobs) == count + 2
    assert client.post(base + f'/timers/{timer["id"]}/control', json={'action': 'pause'}, headers=csrf).status_code == 200
    asyncio.run(automation.tick(1000))
    assert len(pool.jobs) == count + 2
    assert client.post(base + f'/timers/{timer["id"]}/control', json={'action': 'resume'}, headers=csrf).status_code == 200
    once = client.post(base + '/timers', json={'run_id': run['id'], 'name': 'Once', 'interval_seconds': 0, 'first_run': 100}, headers=csrf).json()
    asyncio.run(automation.tick(101))
    assert db.query('SELECT enabled FROM task_timers WHERE id=?', (once['id'],), one=True)['enabled'] == 0
    assert len(pool.jobs) == count + 4
    assert client.get(base + '/timers').json()[0]['jobs']


def test_timer_waits_for_nodes_and_survives_expired_original_jobs(setup, monkeypatch):
    client, network, csrf, pool = setup
    run = whole_task(client, network, csrf)
    timer = client.post(f'/api/networks/{network}/timers', json={'run_id': run['id'], 'name': 'Offline', 'interval_seconds': 0, 'first_run': 1}, headers=csrf).json()
    restarted = Coordinator()
    monkeypatch.setattr(routes, 'pool', restarted)
    asyncio.run(automation.tick(2))
    row = db.query('SELECT * FROM task_timers WHERE id=?', (timer['id'],), one=True)
    assert row['enabled'] == 1 and 'contributor' in row['last_error']
    assert len(restarted.jobs) == 0
    worker(restarted, network)
    asyncio.run(automation.tick(33))
    assert len(restarted.jobs) == 2
    assert db.query('SELECT enabled FROM task_timers WHERE id=?', (timer['id'],), one=True)['enabled'] == 0


def test_old_live_cpu_task_is_recovered_and_unrecoverable_history_is_clear(setup):
    client, network, csrf, pool = setup
    run = whole_task(client, network, csrf, CPU)
    db.execute('DELETE FROM run_inputs WHERE run_id=?', (run['id'],))
    tasks = client.get(f'/api/networks/{network}/tasks').json()
    assert tasks[0]['repeatable']
    result = client.post(f'/api/networks/{network}/tasks/{run["id"]}/repeat', headers=csrf)
    assert result.status_code == 202, result.text
    new = pool.jobs[result.json()['jobs'][0]['job_id']]
    assert new.request.kind == 'cpu' and new.reduction == 'sum' and new.initial == 10
    assert sum(c.count for c in new.chunks) == 4
    db.execute('DELETE FROM run_inputs WHERE run_id=?', (run['id'],))
    pool.jobs.pop(run['jobs'][0]['job_id'])
    assert not next(t for t in client.get(f'/api/networks/{network}/tasks').json() if t['id'] == run['id'])['repeatable']
    assert client.post(f'/api/networks/{network}/timers', json={'run_id': run['id'], 'name': 'Expired'}, headers=csrf).status_code == 410


def test_timer_controls_are_owned_and_foreign_jobs_are_hidden(setup):
    client, network, csrf, pool = setup
    run = whole_task(client, network, csrf)
    timer = client.post(f'/api/networks/{network}/timers', json={'run_id': run['id'], 'name': 'Owned timer'}, headers=csrf).json()
    assert client.post('/auth/logout', headers=csrf).status_code == 200
    csrf = {'X-CSRF-Token': client.get('/auth/config').json()['csrf_token']}
    assert client.post('/auth/guest', json={}, headers=csrf).status_code == 200
    csrf = {'X-CSRF-Token': client.get('/auth/config').json()['csrf_token']}
    assert client.post('/api/networks/join', json={'network_id': network, 'password': 'test-password'}, headers=csrf).status_code == 200
    assert client.post(f'/api/networks/{network}/timers/{timer["id"]}/control', json={'action': 'pause'}, headers=csrf).status_code == 403
    key = new_key(client, network, csrf)
    bearer = {'Authorization': 'Bearer ' + key['key']}
    assert client.post(f'/api/v1/networks/{network}/jobs/{run["jobs"][0]["job_id"]}/cancel', headers=bearer).status_code == 403
    pool.jobs[run['jobs'][0]['job_id']].network_id = 'another'
    assert client.get(f'/api/v1/networks/{network}/jobs/{run["jobs"][0]["job_id"]}', headers=bearer).status_code == 404


def test_onnx_submission_is_saved_and_can_repeat_with_original_inputs(setup, monkeypatch):
    client, network, csrf, pool = setup
    model = base64.b64encode(Path('HiveFrontend/public/examples/dense.onnx').read_bytes()).decode()
    payload = {'kind': 'onnx', 'model': model, 'input': TypedArray.encode([2, 3, 4, 5, 6, 7, 8, 9]).model_dump(),
               'input_shape': [4, 2], 'batch_size': 2, 'independent': True}
    ui = client.post('/pool/jobs?network_id=' + network, json=payload, headers=csrf)
    assert ui.status_code == 202, ui.text
    saved = client.get(f'/api/networks/{network}/tasks').json()[0]
    assert saved['repeatable'] and saved['jobs'][0]['job_id'] == ui.json()['job_id']
    restarted = Coordinator()
    monkeypatch.setattr(routes, 'pool', restarted)
    repeated = client.post(f'/api/networks/{network}/tasks/{saved["id"]}/repeat', headers=csrf)
    assert repeated.status_code == 202, repeated.text
    job = restarted.jobs[repeated.json()['jobs'][0]['job_id']]
    assert job.request.input.decode() == [2, 3, 4, 5, 6, 7, 8, 9]
    assert job.output_shape == [4, 3] and len(job.chunks) == 2
    key = new_key(client, network, csrf)
    response = client.post(f'/api/v1/networks/{network}/onnx', json=payload, headers={'Authorization': 'Bearer ' + key['key']})
    assert response.status_code == 202, response.text
    assert db.query('SELECT * FROM run_inputs WHERE run_id=?', (response.json()['id'],), one=True)


def test_invalid_timer_intervals_and_disabled_timers_do_not_run(setup):
    client, network, csrf, pool = setup
    run = whole_task(client, network, csrf)
    base = f'/api/networks/{network}'
    for interval in [-1, 1, 9]:
        assert client.post(base + '/timers', json={'name': 'Invalid', 'run_id': run['id'], 'interval_seconds': interval}, headers=csrf).status_code == 422
    timer = client.post(base + '/timers', json={'name': 'Paused', 'run_id': run['id'], 'interval_seconds': 60, 'first_run': 1}, headers=csrf).json()
    client.post(base + f'/timers/{timer["id"]}/control', json={'action': 'pause'}, headers=csrf)
    worker(pool, network)
    asyncio.run(automation.tick(100))
    assert len(pool.jobs) == 2


def test_program_persistence_failure_rolls_back_all_created_jobs(setup, monkeypatch):
    client, network, csrf, pool = setup
    def fail(*args, **kwargs):
        raise RuntimeError('simulated database failure')
    monkeypatch.setattr(db, 'save_run', fail)
    with pytest.raises(RuntimeError, match='database failure'):
        whole_task(client, network, csrf)
    assert not pool.jobs
