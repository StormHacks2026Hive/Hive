"""Authenticated networks, whole-file analysis, weighted CPU/GPU work and controls."""

import asyncio
import base64
import json
import struct
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import auth
from backend import database as db
from backend.browser_cpu import lower
from backend.browser_cpu import plan as cpu_plan
from backend.models import TypedArray
from backend.pool import routes
from backend.pool.coordinator import Chunk, Coordinator
from backend.pool.models import Register, ResultHeader, Tile
from backend.programs import ProgramRequest, inspect_program
from tests.test_pool import CAPABILITIES, Socket

ROOT = Path(__file__).resolve().parents[1]
PYTHON = """def transform(values):
    result = [0.0] * len(values)
    for i in range(len(values)):
        result[i] = values[i] * 2 + 1
    return result
"""
CPU = """def total(values):
    result = 10
    # hive:cpu begin
    for value in values:
        result += value * value
    # hive:cpu end
    return result
"""
WGSL = """@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    result[gid.x] = values[gid.x] * 2.0 + 1.0;
}"""


def test_gpu_specs_influence_estimates_but_cpu_stays_single_worker():
    coordinator = Coordinator()
    workers = []
    for cores, memory, texture_limit in [(4, 4, 8192), (32, 64, 64)]:
        caps = dict(CAPABILITIES)
        caps['hardware'] = {'logical_cores': cores, 'memory_gib': memory}
        caps['limits'] = dict(CAPABILITIES['limits'], maxTextureDimension2D=texture_limit)
        workers.append(coordinator.register(Socket(), Register(type='register', label='test', capabilities=caps)))
    assert coordinator.weights(None)[workers[1].worker_id] > coordinator.weights(None)[workers[0].worker_id]
    assert list(coordinator.weights(None, cpu=True).values()) == [0.5, 0.5]
    tile = Chunk('tile', Tile(x=0, y=0, width=128, height=64), assignment={'kind': 'texture_tile'}, byte_length=32768)
    assert coordinator.compatible(workers[0], tile)
    assert not coordinator.compatible(workers[1], tile)
    old = Register(type='register', label='old', capabilities=CAPABILITIES)
    assert old.capabilities.hardware.logical_cores is None


def test_laptop_gpu_share_beats_phone_even_with_noisy_faster_phone_probe(pool):
    workers = []
    for kind, elapsed in [('phone', .01), ('laptop', 10)]:
        caps = dict(CAPABILITIES, device_type=kind)
        caps['benchmark'] = dict(CAPABILITIES['benchmark'], elapsed_ms=elapsed)
        workers.append(pool.register(Socket(), Register(type='register', label=kind, network_id='network', capabilities=caps)))
    phone, laptop = workers
    weights = pool.weights('network')
    assert weights[laptop.worker_id] > weights[phone.worker_id]
    parts = pool.partitions(8205, 'network')
    assert sum(n for _, n, owner in parts if owner == laptop.worker_id) > sum(n for _, n, owner in parts if owner == phone.worker_id)
    assert sum(n for _, n, _ in parts) == 8205
    assert all(n <= 4096 for _, n, _ in parts)
    assert [offset for offset, _, _ in parts] == [sum(n for _, n, _ in parts[:i]) for i in range(len(parts))]
    laptop.active = False
    assert pool.weights('network') == {phone.worker_id: 1.0}


def test_reported_apple_pro_tier_has_higher_gpu_allocation(pool):
    workers = []
    for name in ['Apple M1', 'Apple M1 Pro']:
        caps = dict(CAPABILITIES, device_type='laptop', adapter={'vendor':'apple', 'description':name})
        workers.append(pool.register(Socket(), Register(type='register', label=name, network_id='network', capabilities=caps)))
    weights = pool.weights('network')
    assert weights[workers[1].worker_id] > weights[workers[0].worker_id]


@pytest.mark.parametrize('source', [
    PYTHON.replace('transform', 'arbitrary_function_name'),
    'def other_name(values):\n    return [abs(v) / 2 for v in values]\n',
    'def offset_map(values):\n    return [values[i] + i for i in range(len(values))]\n',
])
def test_explicit_gpu_and_cpu_targets_preserve_full_chunked_array(source, pool):
    values = TypedArray.encode(range(100))
    for target in ('gpu', 'cpu'):
        report, plans = inspect_program(ProgramRequest(source=source, input=values, target=target, chunk_size=16), 'network')
        assert report['status'] == 'ready', report
        assert report['segments'][0]['target'] == target
        plan = plans[0][2]
        assert plan['output_shape'] == [100]
        assert sum(c['count'] for c in plan['chunks']) == 100
        assert len(plan['chunks']) > 1
        assert all(c['count'] <= 16 for c in plan['chunks'])
        if target == 'gpu': assert report['segments'][0]['wgsl']


def test_packaged_mandelbulb_with_python_wrapper_and_wgsl_extension_animates(pool):
    source = (ROOT / 'HiveFrontend/public/examples/Mandelbulb.wgsl').read_text()
    report, plans = inspect_program(ProgramRequest(source=source, filename='Mandelbulb.wgsl', mode='animation', frames=3, width=129, height=65), 'network')
    assert report['status'] == 'ready', report
    _, request, plan = plans[0]
    assert plan['output_shape'] == [3, 65, 129, 4]
    assert len({chunk['assignment']['uniform_data'] for chunk in plan['chunks']}) == 3
    job = pool.create(request, plan=plan, network_id='network')
    assert pool.status(job).frame_count == 3


@pytest.mark.parametrize('source', [
    'def f(values):\n    return [values[i-1] for i in range(len(values))]\n',
    'def f(values):\n    return [unknown(v) for v in values]\n',
    'def f(values):\n    return [abs(abs) for abs in values]\n',
])
def test_array_conversion_does_not_claim_unsupported_python_works(source, pool):
    for target in ('gpu', 'cpu'):
        report, _ = inspect_program(ProgramRequest(source=source, input=TypedArray.encode([1,2,3]), target=target), 'network')
        assert report['status'] == 'unsupported'



@pytest.fixture
def pool(monkeypatch):
    value = Coordinator()
    monkeypatch.setattr(routes, "pool", value)
    return value


def login(client, name="alice"):
    user = auth.User(id=name, name=name.title(), email=f"{name}@example.com")
    token = f"test-session-{name}"
    db.save_session(token, user, time.time() + 3600)
    client.cookies.set(auth.SESSION_COOKIE, token)
    return {"X-CSRF-Token": client.get("/auth/config").json()["csrf_token"]}


def register(pool, network="network", cpu_score=1, gpu_ms=1, label="test"):
    caps = dict(CAPABILITIES, cpu_score=cpu_score)
    caps["benchmark"] = dict(CAPABILITIES["benchmark"], elapsed_ms=gpu_ms)
    return pool.register(
        Socket(),
        Register(type="register", label=label, network_id=network, capabilities=caps),
    )


def test_mark_edit_targets_and_weighted_allocations(pool):
    fast = register(pool, cpu_score=3, gpu_ms=1)
    slow = register(pool, cpu_score=1, gpu_ms=3)
    values = TypedArray.encode(range(16))
    report, plans = inspect_program(
        ProgramRequest(source=PYTHON, input=values, mark=True), "network"
    )
    assert report["status"] == "ready"
    assert "# hive:gpu begin" in report["marked_source"]
    shares = report["segments"][0]["allocations"]
    assert shares[fast.worker_id] > shares[slow.worker_id] and sum(shares.values()) == 16
    edited = report["marked_source"].replace("hive:gpu", "hive:cpu")
    cpu, plans = inspect_program(
        ProgramRequest(source=edited, input=values, segmentation="manual"), "network"
    )
    assert cpu["status"] == "ready" and cpu["segments"][0]["target"] == "cpu"
    assert cpu["segments"][0]["allocations"] == {fast.worker_id: 12, slow.worker_id: 4}
    assert plans[0][2]["output_format"] == "f64"
    pool.disconnect(slow.worker_id)
    assert sum(n for _, n, _ in pool.partitions(16, "network")) == 16
    assert all(
        owner == fast.worker_id for _, _, owner in pool.partitions(16, "network")
    )


def test_entire_renderer_upload_and_two_animation_frames(pool):
    source = (ROOT / "demos/test_files/Mandelbulb_wgsl.py").read_text()
    report, plans = inspect_program(
        ProgramRequest(
            source=source,
            filename="Mandelbulb_wgsl.py",
            mode="animation",
            width=129,
            height=65,
            frames=2,
        ),
        "network",
    )
    assert report["status"] == "ready", report
    assert len(plans) == 1
    segment, request, plan = plans[0]
    assert segment["kind"] == "image" and plan["output_shape"] == [2, 65, 129, 4]
    assert len(plan["chunks"]) == 4
    assert (
        plan["chunks"][0]["assignment"]["uniform_data"]
        != plan["chunks"][2]["assignment"]["uniform_data"]
    )
    assert any("host Python" in finding for finding in report["findings"])


def test_all_embedded_shaders_are_scanned_and_never_execute_host_python(pool, tmp_path):
    marker = tmp_path / "executed"
    source = f"open({str(marker)!r}, 'w').write('bad')\nshader1 = {WGSL!r}\ndef host():\n    shader2 = {WGSL!r}\n"
    report, plans = inspect_program(
        ProgramRequest(source=source, input=TypedArray.encode([1, 2])), "network"
    )
    assert report["status"] == "ready" and len(plans) == 2
    assert not marker.exists()
    bad = (
        source
        + "\nshader3 = "
        + repr(WGSL.replace("values[gid.x]", "values[gid.x-1u]"))
    )
    assert (
        inspect_program(
            ProgramRequest(source=bad, input=TypedArray.encode([1, 2])), "network"
        )[0]["status"]
        == "unsupported"
    )


@pytest.mark.parametrize(
    "source",
    [
        CPU.replace("hive:cpu end", "hive:gpu end"),
        CPU.replace("value * value", "print(value)"),
        PYTHON.replace("values[i] * 2", "values[i-1] * 2"),
        "global shared\ndef bad(values):\n    global shared\n    shared = values\n",
    ],
)
def test_unsafe_or_mismatched_regions_are_refused(pool, source):
    try:
        report, _ = inspect_program(
            ProgramRequest(source=source, input=TypedArray.encode([1, 2])), "network"
        )
        assert report["status"] == "unsupported"
    except ValueError:
        pass


def cpu_execute(assignment):
    script = "import {executeCPU} from './node-web/cpu.js';let text='';for await(const data of process.stdin)text+=data;const result=executeCPU(JSON.parse(text));process.stdout.write(Buffer.from(result.pixels).toString('base64'));"
    result = subprocess.run(
        [
            "node",
            "--experimental-default-type=module",
            "--input-type=module",
            "-e",
            script,
        ],
        input=json.dumps(assignment),
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    return base64.b64decode(result.stdout)


@pytest.mark.parametrize(
    "source,values,expected",
    [
        (PYTHON, [1, 2, 3, 4], [3, 5, 7, 9]),
        (CPU, [1, 2, 3, 4], [40]),
        (
            CPU.replace("result = 10", "result = 100").replace(
                "result += value * value", "result = min(result, value)"
            ),
            [7, -3, 12],
            [-3],
        ),
        (
            CPU.replace("result = 10", "result = -100").replace(
                "result += value * value", "result = max(result, value)"
            ),
            [7, -3, 12],
            [12],
        ),
        (
            "def mapping(values):\n    return [value / 3 for value in values]\n",
            [1, 2, 3, 4],
            [1 / 3, 2 / 3, 1, 4 / 3],
        ),
        (
            "def total(count):\n    result = 0\n    for i in range(count):\n        result += i * i\n    return result\n",
            None,
            [sum(i * i for i in range(12))],
        ),
        (
            PYTHON.replace("values[i] * 2 + 1", "values[i] * 10000001 + 1"),
            [2, 3],
            [20000003, 30000004],
        ),
    ],
)
def test_real_browser_cpu_interpreter_and_ordered_merge(pool, source, values, expected):
    program = lower(source)
    count = len(values) if values is not None else 12
    plan = cpu_plan(
        program, values, count, pool.partitions(count, "network", maximum=2, cpu=True)
    )
    job = pool.create(SimpleNamespace(kind="cpu"), plan=plan, network_id="network")
    worker = register(pool, cpu_score=2)
    while job.status != "done":
        assignment = pool.pull(worker)
        payload = cpu_execute(assignment.model_dump())
        pool.accept(
            worker,
            ResultHeader(
                type="chunk_result",
                job_id=job.job_id,
                chunk_id=assignment.chunk_id,
                attempt_id=assignment.attempt_id,
                output_format="f64",
                byte_length=len(payload),
                elapsed_ms=1,
            ),
            payload,
        )
    assert list(struct.unpack("<" + "d" * len(expected), job.image)) == pytest.approx(
        expected, rel=1e-12
    )


def test_network_password_isolation_persistence_and_controls(pool):
    from backend.main import app

    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as stranger:
        ah = login(alice)
        bh = login(bob, "bob")
        sh = login(stranger, "stranger")
        network = alice.post(
            "/api/networks",
            json={"name": "Shared hive", "password": "test-password"},
            headers=ah,
        ).json()
        network_id = network["id"]
        assert stranger.get(f"/api/networks/{network_id}").status_code == 403
        assert (
            bob.post(
                "/api/networks/join",
                json={"network_id": network_id, "password": "wrong-password"},
                headers=bh,
            ).status_code
            == 403
        )
        assert (
            bob.post(
                "/api/networks/join",
                json={"network_id": network_id, "password": "test-password"},
                headers=bh,
            ).status_code
            == 200
        )
        stored = db.query("SELECT * FROM networks WHERE id=?", (network_id,), one=True)
        assert "test-password" not in stored["password_hash"] and stored[
            "password_hash"
        ].startswith("scrypt$")
        node = bob.post(
            f"/api/networks/{network_id}/nodes",
            json={"device_key": "persistent-device-123", "label": "Bob laptop"},
            headers=bh,
        ).json()
        repeat = bob.post(
            f"/api/networks/{network_id}/nodes",
            json={"device_key": "persistent-device-123", "label": "Bob laptop"},
            headers=bh,
        ).json()
        assert repeat["id"] == node["id"]
        registration = Register(
            type="register",
            node_id=node["id"],
            network_id=network_id,
            label="Bob laptop",
            capabilities=CAPABILITIES,
        )
        worker = pool.register(Socket(), registration, user_id="bob")
        path = f"/api/networks/{network_id}/nodes/{node['id']}/control"
        assert (
            stranger.post(path, json={"action": "kill"}, headers=sh).status_code == 403
        )
        # Owner pauses/resumes an idle node, then stops it persistently.
        worker.socket.send_json = lambda msg: asyncio.sleep(0)
        assert alice.post(path, json={"action": "pause"}, headers=ah).status_code == 200
        assert not worker.active
        assert (
            alice.post(path, json={"action": "resume"}, headers=ah).status_code == 200
        )
        assert worker.active
        assert alice.post(path, json={"action": "kill"}, headers=ah).status_code == 200
        assert worker.worker_id not in pool.workers
        assert (
            db.query("SELECT mode FROM nodes WHERE id=?", (node["id"],), one=True)[
                "mode"
            ]
            == "off"
        )
        assert (
            alice.get(f"/api/networks/{network_id}").json()["nodes"][0]["status"]
            == "offline"
        )
        auth.sessions.clear()  # SQLite session and memberships survive a process cache reset.
        assert bob.get("/auth/me").json()["user"]["id"] == "bob"
        assert bob.get("/api/networks").json()[0]["id"] == network_id


def test_network_run_results_and_asset_authorization(pool):
    from backend.main import app

    with TestClient(app) as client, TestClient(app) as outsider:
        headers = login(client)
        login(outsider, "outsider")
        network_id = client.post(
            "/api/networks",
            json={"name": "Only us", "password": "test-password"},
            headers=headers,
        ).json()["id"]
        request = {
            "source": CPU,
            "filename": "cpu.py",
            "input": TypedArray.encode([1, 2, 3, 4]).model_dump(),
        }
        run = client.post(
            f"/api/networks/{network_id}/runs", json=request, headers=headers
        )
        assert run.status_code == 202, run.text
        job_id = run.json()["jobs"][0]["job_id"]
        assert outsider.get(f"/pool/jobs/{job_id}").status_code == 404
        assert (
            outsider.post(
                f"/pool/jobs/{job_id}/cancel",
                json={},
                headers={
                    "X-CSRF-Token": outsider.get("/auth/config").json()["csrf_token"]
                },
            ).status_code
            == 404
        )
        assert (
            client.post(f"/pool/jobs/{job_id}/cancel", json={}, headers=headers).json()[
                "status"
            ]
            == "cancelled"
        )
        assert (
            client.get(f"/api/networks/{network_id}/runs").json()[0]["jobs"][0][
                "status"
            ]
            == "cancelled"
        )
        created = client.post(
            f"/api/networks/{network_id}/runs",
            json={
                "source": WGSL,
                "filename": "test.wgsl",
                "input": TypedArray.encode([1, 2]).model_dump(),
            },
            headers=headers,
        )
        assert created.status_code == 202, created.text
        job = pool.jobs[created.json()["jobs"][0]["job_id"]]
        asset = next(iter(job.assets))
        assert client.get(f"/pool/assets/{asset}").status_code == 200
        assert outsider.get(f"/pool/assets/{asset}").status_code == 404
        register(pool, "other-network")
        assert pool.pull(next(iter(pool.workers.values()))).type == "no_work"


def test_guests_create_join_submit_and_keep_network_isolation(pool):
    from backend.main import app

    with TestClient(app) as owner, TestClient(app) as peer:
        headers = {"X-CSRF-Token": owner.get("/auth/config").json()["csrf_token"]}
        peer_headers = {"X-CSRF-Token": peer.get("/auth/config").json()["csrf_token"]}
        owner_user = owner.post("/auth/guest", json={}, headers=headers).json()["user"]
        peer_user = peer.post("/auth/guest", json={}, headers=peer_headers).json()["user"]
        assert owner_user["id"] != peer_user["id"]
        network = owner.post("/api/networks", json={"name": "Guest hive", "password": "test-password"}, headers=headers).json()
        network_id = network["id"]
        assert peer.get(f"/api/networks/{network_id}").status_code == 403
        joined = peer.post("/api/networks/join", json={"network_id": network_id, "password": "test-password"}, headers=peer_headers)
        assert joined.status_code == 200
        node = owner.post(f"/api/networks/{network_id}/nodes", json={"device_key": "guest-device-123", "label": "Guest laptop"}, headers=headers).json()
        path = f"/api/networks/{network_id}/nodes/{node['id']}/control"
        assert peer.post(path, json={"action": "kill"}, headers=peer_headers).status_code == 403
        submitted = owner.post(f"/api/networks/{network_id}/runs", json={"source": CPU, "filename": "cpu.py", "input": TypedArray.encode([1, 2, 3]).model_dump()}, headers=headers)
        assert submitted.status_code == 202, submitted.text
        job_id = submitted.json()["jobs"][0]["job_id"]
        assert owner.post(f"/pool/jobs/{job_id}/cancel", json={}, headers=headers).json()["status"] == "cancelled"
        auth.sessions.clear()
        assert owner.get("/api/networks").json()[0]["id"] == network_id
        assert owner.get(f"/api/networks/{network_id}").json()["nodes"][0]["id"] == node["id"]


def test_submission_capacity_returns_429_and_reclaims_cancelled_results(pool):
    from backend.main import app

    pool.max_retained_jobs = 1
    with TestClient(app) as client:
        headers = login(client)
        network_id = client.post('/api/networks', json={'name': 'Capacity hive', 'password': 'test-password'}, headers=headers).json()['id']
        path = f'/api/networks/{network_id}/runs'
        payload = {'source': CPU, 'filename': 'cpu.py', 'input': TypedArray.encode([1, 2]).model_dump()}
        first = client.post(path, json=payload, headers=headers)
        assert first.status_code == 202
        job_id = first.json()['jobs'][0]['job_id']
        blocked = client.post(path, json=payload, headers=headers)
        assert blocked.status_code == 429
        assert job_id in pool.jobs
        assert client.post(f'/pool/jobs/{job_id}/cancel', json={}, headers=headers).status_code == 200
        replacement = client.post(path, json=payload, headers=headers)
        assert replacement.status_code == 202
        assert job_id not in pool.jobs


def test_popular_gpu_bonus_is_small_and_cpu_scores_are_separate(pool):
    caps = dict(CAPABILITIES, cpu_score=3, adapter={"vendor": "popular"})
    a = pool.register(
        Socket(),
        Register(type="register", label="A", network_id="network", capabilities=caps),
    )
    b = pool.register(
        Socket(),
        Register(type="register", label="B", network_id="network", capabilities=caps),
    )
    other = dict(caps, cpu_score=1, adapter={"vendor": "other"})
    c = pool.register(
        Socket(),
        Register(type="register", label="C", network_id="network", capabilities=other),
    )
    gpu = pool.weights("network", normalized=False)
    assert gpu[a.worker_id] / gpu[c.worker_id] == pytest.approx(1.08)
    cpu = pool.weights("network", cpu=True, normalized=False)
    assert cpu[a.worker_id] / cpu[c.worker_id] == 3
    b.visible = False
    assert b.worker_id not in pool.weights("network")
    assert (
        pool.weights("network", normalized=False)[a.worker_id]
        == pool.weights("network", normalized=False)[c.worker_id]
    )


def test_killed_worker_chunk_is_reassigned_and_old_result_is_stale(pool):
    report, plans = inspect_program(
        ProgramRequest(source=CPU, input=TypedArray.encode([1, 2, 3, 4])), "network"
    )
    _, request, plan = plans[0]
    job = pool.create(request, plan=plan, network_id="network")
    a = register(pool, label="A")
    old = pool.pull(a)
    pool.disconnect(a.worker_id, "Killed by owner")
    b = register(pool, label="B")
    retry = pool.pull(b)
    assert old.chunk_id == retry.chunk_id and old.attempt_id != retry.attempt_id
    payload = cpu_execute(old.model_dump())
    header = ResultHeader(
        type="chunk_result",
        job_id=job.job_id,
        chunk_id=old.chunk_id,
        attempt_id=old.attempt_id,
        output_format="f64",
        byte_length=len(payload),
        elapsed_ms=1,
    )
    assert pool.accept(a, header, payload).disposition == "stale"
    header = header.model_copy(update={"attempt_id": retry.attempt_id})
    assert pool.accept(b, header, payload).disposition == "accepted"
    assert job.status == "done" and struct.unpack("<d", job.image) == (40,)


@pytest.mark.parametrize("width,height", [(129, 65), (256, 128)])
def test_builtin_animation_honors_dimensions_and_clipped_tiles(pool, width, height):
    report, plans = inspect_program(
        ProgramRequest(
            source="", mode="animation", width=width, height=height, frames=2
        ),
        "network",
    )
    _, request, _ = plans[0]
    job = pool.create(request, network_id="network")
    assert job.output_shape == [2, height, width, 4]
    assert len(job.chunks) == report["segments"][0]["chunk_count"]
    for index in (0, 1):
        assert (
            sum(c.count for c in job.chunks if c.frame_index == index) == width * height
        )


def test_cpu_floor_division_and_large_integer_modulo_preserve_python_semantics(pool):
    for expression, value, expected in [
        ("value // 0.1", 7, 69),
        ("value % 9007199254740991", 2, 2),
        ("value % -3", 5, -1),
    ]:
        source = (
            f"def mapping(values):\n    return [{expression} for value in values]\n"
        )
        program = lower(source)
        plan = cpu_plan(program, [value], 1, [(0, 1, None)])
        output = cpu_execute(plan["chunks"][0]["assignment"])
        assert struct.unpack("<d", output) == (expected,)
