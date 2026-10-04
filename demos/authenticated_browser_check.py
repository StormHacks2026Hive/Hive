"""Exercise the real merged UI, two users, GPU/CPU work and node controls.

Includes API key connections and server-driven task timers. Starts an isolated
server/database. Test sessions are seeded locally; no login
bypass is exposed by the production app. Google token verification has separate
signed-token tests. Both browser contexts share one physical GPU, so this is a
correctness check, not a scaling benchmark.
"""

import base64
import io
import json
import logging
import os
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import expect, sync_playwright

from backend import auth
from backend import database as db
from backend.pool.image_workloads import ImageAnalysisRequest, extract, uniform_data
from demos.mandelbulb_upload_check import REFERENCE

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
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


def main():
    with tempfile.TemporaryDirectory(prefix="hive-browser-") as temp:
        os.environ["HIVE_DB_PATH"] = str(Path(temp) / "hive.sqlite3")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        tokens = ["browser-session-alice", "browser-session-bob"]
        for name, token in zip(["Alice", "Bob"], tokens):
            db.save_session(
                token,
                auth.User(
                    id=name.lower(), name=name, email=f"{name.lower()}@example.com"
                ),
                time.time() + 3600,
            )
        env = dict(os.environ, HIVE_ALLOW_LEGACY_POOL="false", COOKIE_SECURE="false")
        with (Path(temp) / "server.log").open("w") as log:
            server = subprocess.Popen(
                [
                    str(ROOT / "backend/.venv/bin/python"),
                    "-m",
                    "uvicorn",
                    "backend.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--ws-max-size",
                    "12000000",
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=log,
            )
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        channel="chrome", headless=True, args=["--enable-unsafe-webgpu"]
                    )
                    contexts = [browser.new_context(reduced_motion="reduce") for _ in range(2)]
                    errors = []
                    for context, token in zip(contexts, tokens):
                        context.add_cookies(
                            [
                                {
                                    "name": "hive_session",
                                    "value": token,
                                    "url": base,
                                    "httpOnly": True,
                                    "sameSite": "Lax",
                                }
                            ]
                        )

                    # Only alter the GPU module; the actual worker and transport run unchanged.
                    def cpu_only(route):
                        response = route.fetch()
                        route.fulfill(
                            response=response,
                            body=response.text().replace(
                                "  async init(onLost) {",
                                "  async init(onLost) { throw new Error('CPU-only test device');",
                            ),
                        )

                    contexts[1].route("**/node/gpu.js", cpu_only)
                    pages = [c.new_page() for c in contexts]
                    for page in pages:
                        page.on("pageerror", lambda e: errors.append(str(e)))
                        page.on(
                            "console",
                            lambda message: (
                                LOG.info("Browser %s: %s", message.type, message.text)
                                if message.type == "error"
                                else None
                            ),
                        )
                    for _ in range(100):
                        try:
                            pages[0].goto(base, timeout=3000)
                            break
                        except Exception:
                            if server.poll() is not None:
                                raise RuntimeError(
                                    (Path(temp) / "server.log").read_text()
                                )
                            time.sleep(0.1)
                    alice, bob = pages
                    alice.get_by_role(
                        "button", name="Create a network", exact=True
                    ).click()
                    alice.get_by_label("Network name", exact=True).fill("Browser hive")
                    alice.get_by_label("Network password", exact=True).fill(
                        "browser-password"
                    )
                    alice.locator("form").get_by_role(
                        "button", name="Create network", exact=True
                    ).click()
                    alice.wait_for_function(
                        "document.querySelector('.network-id') !== null", timeout=30000
                    )
                    network_id = (
                        contexts[0].request.get(base + "/api/networks").json()[0]["id"]
                    )
                    bob.goto(base)
                    bob.get_by_label("Network ID", exact=True).fill(network_id)
                    bob.get_by_label("Network password", exact=True).fill(
                        "browser-password"
                    )
                    bob.locator("form").get_by_role(
                        "button", name="Connect to network", exact=True
                    ).click()
                    deadline = time.monotonic() + 45
                    while time.monotonic() < deadline:
                        snapshot = (
                            contexts[0]
                            .request.get(f"{base}/api/networks/{network_id}")
                            .json()
                        )
                        if len(snapshot["nodes"]) == 2 and all(
                            n["status"] == "idle" for n in snapshot["nodes"]
                        ):
                            break
                        time.sleep(0.2)
                    assert len(snapshot["nodes"]) == 2 and all(
                        n["status"] == "idle" for n in snapshot["nodes"]
                    ), (
                        snapshot,
                        [page.locator(".node-work").inner_text() for page in pages],
                        errors,
                        (Path(temp) / "server.log").read_text()[:5000],
                    )
                    assert any(n["capabilities"]["webgpu"] for n in snapshot["nodes"])
                    assert any(
                        not n["capabilities"]["webgpu"] and n["capabilities"]["cpu"]
                        for n in snapshot["nodes"]
                    )
                    alice.get_by_role("button", name="Compute", exact=True).click()

                    screenshot_dir = os.getenv("HIVE_BROWSER_SCREENSHOTS_DIR")
                    if screenshot_dir:
                        destination = Path(screenshot_dir)
                        destination.mkdir(parents=True, exist_ok=True)
                        expect(alice.locator(".node-card-heading .status-pill")).to_have_text(["idle", "idle"], timeout=30000)
                        for label, width in [("desktop", 1440), ("mobile", 390)]:
                            alice.set_viewport_size({"width": width, "height": 1000})
                            alice.locator(".work-editor").wait_for()
                            alice.screenshot(path=str(destination / f"compute-{label}.png"), full_page=True, animations="disabled")
                            assert alice.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), f"Compute overflow at {width}px"
                        alice.set_viewport_size({"width": 1280, "height": 720})

                    def run_ui(
                        source,
                        mode="compute",
                        values="[1,2,3,4]",
                        filename=None,
                        mark=False,
                        target='auto',
                    ):
                        alice.get_by_label("Work mode").select_option(mode)
                        expect(alice.get_by_role('button', name='Analyze', exact=True)).to_be_enabled(timeout=30000)
                        if filename:
                            alice.get_by_role(
                                "button", name="Upload file", exact=True
                            ).click()
                            alice.locator("input[type=file]").set_input_files(
                                {
                                    "name": filename,
                                    "mimeType": "text/plain",
                                    "buffer": source.encode(),
                                }
                            )
                            expect(alice.locator('.source-caption span')).to_have_text(filename)
                            expect(alice.get_by_role('button', name='Analyze', exact=True)).to_be_enabled(timeout=30000)
                        else:
                            alice.locator("#source-code").fill(source)
                        expect(alice.locator('#source-code')).to_have_value(source)
                        if mode == "compute":
                            alice.get_by_label("Input values", exact=True).fill(values)
                            if filename and filename.endswith('.py'):
                                expect(alice.get_by_role('combobox', name='Python target', exact=True)).to_be_visible()
                                alice.get_by_role('combobox', name='Python target', exact=True).select_option(target)
                        alice.get_by_role(
                            "button",
                            name="Mark & analyze" if mark else "Analyze",
                            exact=True,
                        ).click()
                        expect(
                            alice.get_by_role("button", name="Send", exact=True)
                        ).to_be_enabled(timeout=30000)
                        with alice.expect_response(
                            lambda r: (
                                r.url.endswith("/runs") and r.request.method == "POST"
                            )
                        ) as created:
                            alice.get_by_role("button", name="Send", exact=True).click()
                        assert created.value.status == 202, created.value.text()
                        jobs = created.value.json()["jobs"]
                        deadline = time.monotonic() + 60
                        for job in jobs:
                            while time.monotonic() < deadline:
                                status = (
                                    contexts[0]
                                    .request.get(base + "/pool/jobs/" + job["job_id"])
                                    .json()
                                )
                                if status["status"] in ("done", "failed", "cancelled"):
                                    break
                                time.sleep(0.1)
                            assert status["status"] == "done", status
                        return status, contexts[0].request.get(
                            base + status["result_url"]
                        ).body()

                    status, data = run_ui(PYTHON, mark=True)
                    assert struct.unpack("<4f", data) == (3, 5, 7, 9)
                    assert (
                        "# hive:gpu begin"
                        in alice.locator("#source-code").input_value()
                    )
                    edited = (
                        alice.locator("#source-code")
                        .input_value()
                        .replace("hive:gpu", "hive:cpu")
                    )
                    status, data = run_ui(edited)
                    assert struct.unpack("<4d", data) == (3, 5, 7, 9)
                    expect(alice.get_by_label("Output array")).to_contain_text(
                        "[3,5,7,9]", timeout=30000
                    )
                    csrf = (
                        contexts[0]
                        .request.get(base + "/auth/config")
                        .json()["csrf_token"]
                    )
                    alice_id = next(
                        n["id"] for n in snapshot["nodes"] if n["user_id"] == "alice"
                    )
                    alice_control = (
                        f"{base}/api/networks/{network_id}/nodes/{alice_id}/control"
                    )
                    assert (
                        contexts[0]
                        .request.post(
                            alice_control,
                            data={"action": "pause"},
                            headers={"X-CSRF-Token": csrf},
                        )
                        .ok
                    )
                    status, data = run_ui(CPU)
                    assert struct.unpack("<d", data) == (40,)
                    assert all("Bob" in c["label"] for c in status["contributions"]), (
                        status
                    )
                    assert (
                        contexts[0]
                        .request.post(
                            alice_control,
                            data={"action": "resume"},
                            headers={"X-CSRF-Token": csrf},
                        )
                        .ok
                    )
                    status, data = run_ui(WGSL, filename="compute.wgsl")
                    assert struct.unpack("<4f", data) == (3, 5, 7, 9)
                    renderer = (
                        ROOT / "demos/test_files/Mandelbulb_wgsl.py"
                    ).read_text()
                    # Upload original file; render two different camera frames and a clipped edge.
                    alice.get_by_label("Work mode").select_option("animation")
                    alice.get_by_label("Width", exact=True).fill("129")
                    alice.get_by_label("Height", exact=True).fill("65")
                    alice.get_by_label("Frames", exact=True).fill("2")
                    status, data = run_ui(
                        renderer, mode="animation", filename="Mandelbulb_wgsl.py"
                    )
                    assert (
                        len(data) == 129 * 65 * 4 * 2
                        and data[: len(data) // 2] != data[len(data) // 2 :]
                    )
                    shader, config = extract(
                        ImageAnalysisRequest(source=renderer, width=129, height=65)
                    )
                    reference = base64.b64decode(
                        alice.evaluate(
                            REFERENCE,
                            {
                                "source": shader,
                                "uniform": base64.b64encode(
                                    uniform_data(config)
                                ).decode(),
                                "width": 129,
                                "height": 65,
                            },
                        )
                    )
                    assert (
                        np.abs(
                            np.frombuffer(reference, np.uint8).astype(int)
                            - np.frombuffer(data[: len(reference)], np.uint8)
                        ).max()
                        <= 1
                    )
                    expect(
                        alice.get_by_role("button", name="Download PNG", exact=True)
                    ).to_be_enabled(timeout=30000)
                    with alice.expect_download() as download:
                        alice.get_by_role(
                            "button", name="Download PNG", exact=True
                        ).click()
                    png = Image.open(
                        io.BytesIO(Path(download.value.path()).read_bytes())
                    ).convert("RGBA")
                    assert png.size == (129, 65) and np.array_equal(
                        np.asarray(png).reshape(-1), np.frombuffer(reference, np.uint8)
                    )

                    # Full arrays across multiple chunks, with global-index reads.
                    array_source = 'def any_function_name(values):\n    return [values[i] + i + 1 for i in range(len(values))]\n'
                    array_input = json.dumps(list(range(8205)))
                    expected_array = [i * 2 + 1 for i in range(8205)]
                    for target, scalar in [('gpu', 'f'), ('cpu', 'd')]:
                        status, data = run_ui(array_source, values=array_input, filename='array.py', target=target)
                        assert status['total_chunks'] > 1
                        assert status['output_format'] == ('f32' if target == 'gpu' else 'f64'), (target, status['output_format'], status['output_shape'], len(data), alice.locator('#source-code').input_value())
                        assert list(struct.unpack('<' + scalar * 8205, data)) == expected_array
                        output = alice.get_by_label('Output array', exact=True)
                        expect(output).to_contain_text('16409]', timeout=30000)
                        assert json.loads(output.inner_text()) == expected_array
                    with alice.expect_download() as download:
                        alice.get_by_role('button', name='Download JSON', exact=True).click()
                    assert json.loads(Path(download.value.path()).read_text()) == expected_array

                    # The bundled user file has a Python string wrapper and a .wgsl extension.
                    alice.get_by_role('combobox', name='Example', exact=True).select_option('python')
                    alice.get_by_role('button', name='Upload file', exact=True).click()
                    # A settings edit must not discard an in-flight file read.
                    alice.evaluate('''() => {
                        window.originalFileText = File.prototype.text;
                        File.prototype.text = function() {
                            const file = this;
                            return new Promise(resolve => {
                                window.finishFileUpload = () => window.originalFileText.call(file).then(resolve);
                            });
                        };
                    }''')
                    uploaded_renderer = 'WGSL_SHADER = r"""' + shader + '"""'
                    alice.locator('input[type=file]').set_input_files({'name': 'Mandelbulb.wgsl', 'mimeType': 'text/plain', 'buffer': uploaded_renderer.encode()})
                    alice.wait_for_function('typeof window.finishFileUpload === "function"')
                    alice.get_by_label('Input values', exact=True).fill('[2,3,4]')
                    alice.evaluate('window.finishFileUpload()')
                    alice.evaluate('() => { File.prototype.text = window.originalFileText; }')
                    expect(alice.get_by_label('Work mode')).to_have_value('animation')
                    expect(alice.locator('#source-code')).to_have_value(uploaded_renderer)
                    assert alice.locator('input[type=file]').input_value() == ''
                    # Re-selecting the same file should trigger a new read.
                    alice.get_by_label('Work mode').select_option('compute')
                    alice.locator('input[type=file]').set_input_files({'name': 'Mandelbulb.wgsl', 'mimeType': 'text/plain', 'buffer': uploaded_renderer.encode()})
                    expect(alice.get_by_label('Work mode')).to_have_value('animation')
                    for pasted in (shader, 'WGSL_SHADER = r"""' + shader + '"""'):
                        alice.get_by_role('combobox', name='Example', exact=True).select_option('python')
                        alice.locator('#source-code').fill(pasted)
                        expect(alice.get_by_label('Work mode')).to_have_value('animation')
                        alice.get_by_label('Width', exact=True).fill('65')
                        alice.get_by_label('Height', exact=True).fill('65')
                        alice.get_by_label('Frames', exact=True).fill('2')
                        with alice.expect_response(lambda r: r.url.endswith('/analyze') and r.request.method == 'POST') as analyzed:
                            alice.get_by_role('button', name='Analyze', exact=True).click()
                        report = analyzed.value.json()
                        assert report['status'] == 'ready', report
                        assert report['segments'][0]['kind'] == 'image'
                        assert report['segments'][0]['output_shape'] == [2, 65, 65, 4]
                    alice.get_by_role('combobox', name='Example', exact=True).select_option('mandelbulb')
                    expect(alice.get_by_role('button', name='Analyze', exact=True)).to_be_enabled(timeout=30000)
                    assert alice.get_by_label('Frames', exact=True).input_value() == '8'
                    alice.get_by_label('Width', exact=True).fill('129')
                    alice.get_by_label('Height', exact=True).fill('65')
                    alice.get_by_label('Frames', exact=True).fill('3')
                    status, data = run_ui(alice.locator('#source-code').input_value(), mode='animation', filename='Mandelbulb.wgsl')
                    expect(alice.get_by_role('button', name='Download PNG', exact=True)).to_be_enabled(timeout=30000)
                    frame_size = 129 * 65 * 4
                    assert len(data) == frame_size * 3
                    assert data[:frame_size] != data[frame_size:frame_size * 2]
                    slider = alice.get_by_label('Animation frame', exact=True)
                    for index in range(3):
                        slider.press('Home')
                        for _ in range(index): slider.press('ArrowRight')
                        expected_pixels = list(data[index * frame_size:(index + 1) * frame_size])
                        alice.wait_for_function('(expected) => { const c = document.querySelector(".job-results canvas"); return c && c.getContext("2d").getImageData(0, 0, c.width, c.height).data.every((v, i) => v === expected[i]); }', arg=expected_pixels)
                    tile_requests = []
                    listener = lambda req: tile_requests.append(req.url) if '/chunks/' in req.url else None
                    alice.on('request', listener)
                    alice.get_by_role('button', name='Play', exact=True).click()
                    alice.wait_for_timeout(500)
                    alice.get_by_role('button', name='Pause', exact=True).click()
                    assert not tile_requests, tile_requests
                    alice.remove_listener('request', listener)

                    # Selectable ONNX preset includes a compatible input and batch shape.
                    alice.get_by_role('combobox', name='Example', exact=True).select_option('onnx')
                    expect(alice.get_by_role('button', name='Analyze', exact=True)).to_be_enabled(timeout=30000)
                    alice.get_by_role('button', name='Analyze', exact=True).click()
                    expect(alice.get_by_role('button', name='Send', exact=True)).to_be_enabled(timeout=30000)
                    with alice.expect_response(lambda r: r.url.split('?')[0].endswith('/pool/jobs') and r.request.method == 'POST') as created:
                        alice.get_by_role('button', name='Send', exact=True).click()
                    assert created.value.status == 202, created.value.text()
                    output = alice.get_by_label('Output array', exact=True)
                    expect(output).to_contain_text('[14,19,24', timeout=60000)
                    assert json.loads(output.inner_text()) == [14,19,24,24,33,42,34,47,60,44,61,78]

                    # API connections use scoped bearer keys; timer runs happen
                    # on the server, even when the Timer page is not open.
                    alice.get_by_role('button', name='API', exact=True).click()
                    alice.get_by_role('combobox', name='Access', exact=True).select_option('run')
                    with alice.expect_response(lambda r: r.url.endswith('/api-keys') and r.request.method == 'POST') as created:
                        alice.get_by_role('button', name='Create API key', exact=True).click()
                    assert created.value.status == 201
                    connection = created.value.json()
                    bearer = {'Authorization': 'Bearer ' + connection['key']}
                    expect(alice.get_by_label('New API key', exact=True)).to_have_value(connection['key'])
                    alice.get_by_role('button', name='Test connection', exact=True).click()
                    expect(alice.get_by_role('status')).to_have_text('Connected to this hive')
                    assert '/api/v1/networks/' + network_id in alice.locator('.api-calls').inner_text()
                    alice.get_by_text('Send a task through API', exact=True).click()
                    with alice.expect_response(lambda r: '/api/v1/' in r.url and r.url.endswith('/runs') and r.request.method == 'POST') as created:
                        alice.get_by_role('button', name='Send through API', exact=True).click()
                    assert created.value.status == 202, created.value.text()
                    api_run = created.value.json()
                    api_job_url = base + '/api/v1/networks/' + network_id + '/jobs/' + api_run['jobs'][0]['job_id']
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        result = contexts[0].request.get(api_job_url + '/result', headers=bearer)
                        if result.status == 200:
                            break
                        assert result.status == 409, result.text()
                        alice.wait_for_timeout(100)
                    assert result.json()['values'] == [285]
                    expect(alice.get_by_role('combobox', name='Task', exact=True)).to_have_value(api_run['id'])

                    alice.get_by_role('button', name='Timer', exact=True).click()
                    alice.get_by_label('Timer name', exact=True).fill('Browser timer')
                    alice.get_by_role('combobox', name='Run', exact=True).select_option('once')
                    import datetime
                    past = (datetime.datetime.now() - datetime.timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M')
                    alice.get_by_label('First run · your local time', exact=True).fill(past)
                    with alice.expect_response(lambda r: r.url.endswith('/timers') and r.request.method == 'POST') as created:
                        alice.get_by_role('button', name='Save timer', exact=True).click()
                    assert created.value.status == 201, created.value.text()
                    timer_id = created.value.json()['id']
                    alice.get_by_role('button', name='Mapping', exact=True).click()
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        timers = contexts[0].request.get(base + '/api/networks/' + network_id + '/timers').json()
                        scheduled = next(t for t in timers if t['id'] == timer_id)
                        if scheduled['last_run_id'] and scheduled['jobs'][0]['status'] == 'done':
                            break
                        alice.wait_for_timeout(100)
                    assert scheduled['enabled'] == 0 and scheduled['jobs'][0]['status'] == 'done'
                    scheduled_job = scheduled['jobs'][0]['job_id']
                    assert contexts[0].request.get(base + '/api/v1/networks/' + network_id + '/jobs/' + scheduled_job + '/result', headers=bearer).json()['values'] == [285]
                    alice.get_by_role('button', name='Timer', exact=True).click()
                    expect(alice.locator('.timer-cards')).to_contain_text('Finished')
                    alice.reload()
                    expect(alice.locator('.timer-cards')).to_contain_text('Browser timer', timeout=30000)
                    alice.get_by_role('button', name='Resume', exact=True).click()
                    expect(alice.get_by_role('button', name='Pause', exact=True)).to_be_visible()
                    alice.get_by_role('button', name='Pause', exact=True).click()
                    expect(alice.locator('.timer-cards')).to_contain_text('Paused')
                    alice.get_by_role('button', name='Remove', exact=True).click()
                    expect(alice.locator('.timer-cards')).not_to_contain_text('Browser timer')
                    alice.get_by_role('button', name='API', exact=True).click()
                    alice.get_by_role('button', name='Revoke', exact=True).click()
                    expect(alice.get_by_role('button', name='Revoke', exact=True)).not_to_be_visible()
                    assert contexts[0].request.get(api_job_url, headers=bearer).status == 401
                    alice.get_by_role('button', name='Compute', exact=True).click()
                    # Controls persist and the remote browser obeys them.
                    csrf = (
                        contexts[0]
                        .request.get(base + "/auth/config")
                        .json()["csrf_token"]
                    )
                    bob_id = next(
                        n["id"] for n in snapshot["nodes"] if n["user_id"] == "bob"
                    )
                    control = f"{base}/api/networks/{network_id}/nodes/{bob_id}/control"
                    for action, expected in [("pause", "paused"), ("resume", "idle")]:
                        assert (
                            contexts[0]
                            .request.post(
                                control,
                                data={"action": action},
                                headers={"X-CSRF-Token": csrf},
                            )
                            .ok
                        )
                        deadline = time.monotonic() + 15
                        while time.monotonic() < deadline:
                            snap = (
                                contexts[0]
                                .request.get(f"{base}/api/networks/{network_id}")
                                .json()
                            )
                            if (
                                next(
                                    n["status"]
                                    for n in snap["nodes"]
                                    if n["id"] == bob_id
                                )
                                == expected
                            ):
                                break
                            time.sleep(0.1)
                        assert (
                            next(
                                n["status"] for n in snap["nodes"] if n["id"] == bob_id
                            )
                            == expected
                        )
                    alice.locator(".contributors-panel li").filter(
                        has_text="Bob's laptop"
                    ).get_by_role("button", name="Kill node", exact=True).click()
                    alice.get_by_role("dialog").get_by_role(
                        "button", name="Kill node", exact=True
                    ).click()
                    bob.reload()
                    time.sleep(2)
                    snap = (
                        contexts[0]
                        .request.get(f"{base}/api/networks/{network_id}")
                        .json()
                    )
                    assert (
                        next(n["mode"] for n in snap["nodes"] if n["id"] == bob_id)
                        == "off"
                    )
                    assert not errors, errors
                    browser.close()
                    LOG.info(
                        "Authenticated UI: GPU/CPU arrays, Mandelbulb/ONNX presets, cached playback, API key connections/submission/results/revocation, server timers/refresh/controls, node controls: passed"
                    )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
