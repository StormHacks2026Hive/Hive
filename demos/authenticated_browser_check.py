"""Exercise the real merged UI, two users, GPU/CPU work and node controls.

Starts an isolated server/database. Test sessions are seeded locally; no login
bypass is exposed by the production app. Google token verification has separate
signed-token tests. Both browser contexts share one physical GPU, so this is a
correctness check, not a scaling benchmark.
"""

import base64
import io
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
                    ):
                        alice.get_by_label("Work mode").select_option(mode)
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
                        else:
                            alice.locator("#source-code").fill(source)
                        if mode == "compute":
                            alice.get_by_label("Input values", exact=True).fill(values)
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
                    expect(alice.get_by_label("Output preview")).to_contain_text(
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
                        "Authenticated UI: network create/join, GPU, CPU-only fallback, editable markers, CPU sum, raw WGSL, two Mandelbulb frames, reference pixels, PNG, node controls and persisted stop: passed"
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
