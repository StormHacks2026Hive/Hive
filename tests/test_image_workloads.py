"""Fast upload, bounds, wire format and tile assembly tests for Mandelbulb."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.pool.coordinator import Coordinator
from backend.pool.image_workloads import (
    SHADER,
    ImageAnalysisRequest,
    analyze_image,
    extract,
    image_plan,
    uniform_data,
)
from tests.test_pool import add_worker, result

RENDERER = Path(__file__).resolve().parents[1] / "demos/test_files/Mandelbulb_wgsl.py"


def test_original_upload_and_no_python_execution(tmp_path):
    marker = tmp_path / "executed"
    source = f"open({str(marker)!r}, 'w').write('bad')\n" + RENDERER.read_text()
    request = ImageAnalysisRequest(source=source)
    shader, settings = extract(request)
    assert shader == SHADER
    assert (settings.width, settings.height) == (1280, 1024)
    assert analyze_image(request).chunk_count == 80
    assert not marker.exists()
    # Execute ONLY the known packing function AST to check the original ABI.
    tree = ast.parse(RENDERER.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "create_uniform_data"
    )
    namespace = {"np": np, "math": math, "RenderConfig": object}
    exec(  # noqa: S102 - only the trusted repository packing function, never upload data
        compile(ast.Module(body=[function], type_ignores=[]), str(RENDERER), "exec"),
        namespace,
    )
    assert uniform_data(settings) == namespace["create_uniform_data"](settings)


@pytest.mark.parametrize(
    "source",
    [
        SHADER.replace("params.power - 1.0", "params.power - 2.0"),
        "WGSL_SHADER = str(123)",
        'print("no shader")',
    ],
    ids=["altered-shader", "nonliteral", "missing-shader"],
)
def test_unsupported_upload_explains_why(source):
    report = analyze_image(ImageAnalysisRequest(source=source))
    assert report.status == "unsupported" and report.findings


def test_uneven_edges_cover_image_once():
    plan = image_plan(ImageAnalysisRequest(source=SHADER, width=259, height=131))
    cover = np.zeros((131, 259), dtype=np.uint8)
    for chunk in plan["chunks"]:
        tile = chunk["tile"]
        cover[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += 1
    assert np.all(cover == 1) and len(plan["chunks"]) == 6
    shader = next(iter(plan["assets"].values())).decode()
    assert "let id = hive_image_raw + hive_image_tile.origin.xyz;" in shader
    assert "i32(hive_image_raw.x)" in shader


def test_pool_assembles_texture_tiles_and_http_frame(monkeypatch):
    monkeypatch.setenv("HIVE_ALLOW_LEGACY_POOL", "true")
    from backend.main import app
    from backend.pool import routes

    pool = Coordinator()
    monkeypatch.setattr(routes, "pool", pool)
    with TestClient(app) as client:
        source = RENDERER.read_text()
        analysis = client.post(
            "/pool/wgsl/image/analyze",
            json={"source": source, "width": 129, "height": 65},
        )
        assert analysis.json()["status"] == "ready"
        response = client.post(
            "/pool/jobs",
            json={"kind": "wgsl_image", "source": source, "width": 129, "height": 65},
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        job = pool.jobs[job_id]
        workers = [add_worker(pool, "A"), add_worker(pool, "B")]
        assignments = [pool.pull(worker) for worker in workers]
        for worker, assignment, value in reversed(
            list(zip(workers, assignments, [5, 9]))
        ):
            assert assignment.kind == "texture_tile"
            pixels = bytes([value, 10, 15, 255]) * (
                assignment.tile.width * assignment.tile.height
            )
            assert (
                pool.accept(worker, *result(assignment, pixels)).disposition
                == "accepted"
            )
        assert job.status == "done"
        pixels = np.frombuffer(
            client.get(f"/pool/jobs/{job_id}/frames/0").content, np.uint8
        ).reshape(65, 129, 4)
        assert np.all(pixels[:, :128, 0] == 5)
        assert np.all(pixels[:, 128:, 0] == 9)
        assert client.get(f"/pool/jobs/{job_id}").json()["width"] == 129


def test_nonfinite_or_oversized_config_rejected():
    source = RENDERER.read_text().replace("power: float = 8.0", "power: float = 1e309")
    assert analyze_image(ImageAnalysisRequest(source=source)).status == "unsupported"
    assert (
        analyze_image(
            ImageAnalysisRequest(source=SHADER, width=4096, height=4096)
        ).status
        == "unsupported"
    )
