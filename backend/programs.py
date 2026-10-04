"""Inspect an entire upload, mark independent regions, and submit weighted work."""

import ast
import io
import json
import logging
import math
import re
import time
import tokenize
from types import SimpleNamespace
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from . import database as db
from .browser_cpu import lower as lower_cpu
from .browser_cpu import plan as cpu_plan
from .compiler import compile_source
from .marked_python import convert_comprehension, convert_loop
from .models import Model, TypedArray
from .networks import access
from .pool import routes as pool_routes
from .pool.coordinator import CapacityError
from .pool.image_workloads import (
    ImageAnalysisRequest,
    WGSLImageRequest,
    extract,
    image_plan,
)
from .pool.workloads import AnimationRequest, WGSLRequest, compute_plan
from .wgsl_analyzer import WGSLAnalysisRequest, analyze_wgsl

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/networks/{network_id}", tags=["programs"])


class ProgramRequest(Model):
    source: str = Field(default="", max_length=65536)
    filename: str = Field(default="program.py", max_length=128)
    mode: Literal["compute", "animation"] = "compute"
    segmentation: Literal["auto", "manual"] = "auto"
    target: Literal["auto", "gpu", "cpu"] = "auto"
    mark: bool = False
    input: TypedArray | None = None
    count: int = Field(default=8192, ge=1, le=2_000_000)
    chunk_size: int = Field(default=4096, ge=1, le=4096)
    frames: int = Field(default=1, ge=1, le=32)
    fps: int = Field(default=12, ge=1, le=60)
    width: int = Field(default=256, ge=1, le=4096)
    height: int = Field(default=256, ge=1, le=4096)
    settings: dict = Field(default_factory=dict)
    camera_turn: float = Field(default=30, ge=-360, le=360)
    frame_values: list[dict] | None = Field(default=None, min_length=1, max_length=32)


def marked_regions(source: str) -> list[tuple[int, int, str]]:
    """Match standalone CPU/GPU marker pairs without interpreting string contents."""
    regions, opened = [], None
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = re.fullmatch(
            r"#\s*hive:(gpu|cpu|parallel)\s+(begin|end)\s*", token.string, re.I
        )
        if not match:
            continue
        target, edge = match.groups()
        target = "gpu" if target.lower() == "parallel" else target.lower()
        if edge.lower() == "begin":
            if opened:
                raise ValueError("Markers cannot overlap or nest")
            opened = (token.start[0], target)
        else:
            if not opened or opened[1] != target:
                raise ValueError("Each marker needs a matching CPU/GPU end comment")
            regions.append((opened[0], token.start[0], target))
            opened = None
    if opened:
        raise ValueError("Missing end marker")
    return regions


def mark_source(source: str, regions: list[dict[str, Any]]) -> str:
    """Insert editable target comments around validated source regions."""
    lines = source.splitlines(keepends=True)
    for segment in sorted(regions, key=lambda s: s["line"], reverse=True):
        begin, end = segment["line"], segment["end_line"]
        indent = re.match(r"\s*", lines[begin - 1]).group().replace("\n", "")
        if not lines[end - 1].endswith("\n"):
            lines[end - 1] += "\n"
        target = segment["target"]
        lines.insert(end, f"{indent}# hive:{target} end\n")
        lines.insert(begin - 1, f"{indent}# hive:{target} begin\n")
    return "".join(lines)


def inspect_program(
    request: ProgramRequest, network_id: str
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Any, dict[str, Any] | None]]]:
    """Inspect all embedded shaders and selected independent numeric regions."""
    values = request.input.decode() if request.input else None
    total = len(values) if values is not None else request.count
    if total < 1:
        raise ValueError("Input cannot be empty")
    segments, findings, plans = [], [], []
    shader_ranges = []
    weights = pool_routes.pool.weights(network_id)
    contributors = [
        {
            "worker_id": w.worker_id,
            "node_id": w.node_id,
            "name": w.label,
            "share": weights.get(w.worker_id, 0),
            "cpu_share": pool_routes.pool.weights(network_id, cpu=True).get(
                w.worker_id, 0
            ),
            "gpu": w.capabilities.adapter.description or w.capabilities.adapter.vendor,
            "device": w.capabilities.device_type,
        }
        for w in pool_routes.pool.workers.values()
        if w.network_id == network_id and w.active and w.visible
    ]
    if not contributors:
        findings.append(
            "No active nodes yet. Work will queue until a compatible node connects."
        )

    def add(name, line, end_line, target, kind, payload=None, kernel=None, error=None):
        segment = {
            "id": f"{kind}-{line}-{len(segments)}",
            "name": name,
            "line": line,
            "end_line": end_line,
            "target": target,
            "kind": kind,
            "status": "unsupported" if error else "ready",
            "findings": [error] if error else [],
            "wgsl": kernel,
        }
        if payload:
            job_request, plan = payload
            segment["chunk_count"] = len(plan["chunks"])
            segment["output_shape"] = plan["output_shape"]
            allocations = {}
            for c in plan["chunks"]:
                worker = c.get("preferred_worker")
                if worker:
                    allocations[worker] = allocations.get(worker, 0) + c["count"]
            segment["allocations"] = allocations
            if not allocations:
                from .pool.coordinator import Chunk, Job

                preview = Job(
                    "preview",
                    job_request,
                    [Chunk(**({"tile": None} | c)) for c in plan["chunks"]],
                    bytearray(),
                )
                preview.network_id = network_id
                pool_routes.pool.rank_pending(preview)
                for chunk in preview.chunks:
                    if chunk.preferred_worker:
                        allocations[chunk.preferred_worker] = (
                            allocations.get(chunk.preferred_worker, 0) + chunk.count
                        )
            plans.append((segment, job_request, plan))
        segments.append(segment)

    def shader_segment(shader, line, end_line, name, full_source=None):
        try:
            if "texture_storage_2d" in shader:
                extract(
                    ImageAnalysisRequest(
                        source=shader,
                        width=request.width,
                        height=request.height,
                        settings=request.settings,
                    )
                )
                image_source = (
                    full_source
                    if full_source and "WGSL_SHADER" in full_source
                    else shader
                )
                base = ImageAnalysisRequest(
                    source=image_source,
                    width=request.width,
                    height=request.height,
                    settings=request.settings,
                )
                _, config = extract(base)
                frames = [{}]
                if request.mode == "animation":
                    frames = request.frame_values or []
                    if not frames:
                        x, y, z = config.camera_position
                        tx, ty, tz = config.camera_target
                        for index in range(request.frames):
                            a = (
                                math.radians(request.camera_turn)
                                * index
                                / max(1, request.frames - 1)
                            )
                            frames.append(
                                {
                                    "camera_position": [
                                        tx
                                        + (x - tx) * math.cos(a)
                                        - (z - tz) * math.sin(a),
                                        y,
                                        tz
                                        + (x - tx) * math.sin(a)
                                        + (z - tz) * math.cos(a),
                                    ]
                                }
                            )
                payload = WGSLImageRequest(
                    **base.model_dump(),
                    kind="wgsl_image",
                    frames=frames,
                    fps=request.fps,
                )
                plan = image_plan(payload)
                add(
                    name,
                    line,
                    end_line,
                    "gpu",
                    "image",
                    (payload, plan),
                    next(iter(plan["assets"].values())).decode(),
                )
            else:
                if request.mode == "animation":
                    raise ValueError(
                        "Animation needs an image renderer; this shader produces a numeric array"
                    )
                report = analyze_wgsl(
                    WGSLAnalysisRequest(
                        source=shader, count=total, chunk_size=request.chunk_size
                    )
                )
                if report.status != "ready":
                    raise ValueError("; ".join(report.findings))
                payload = WGSLRequest(
                    kind="wgsl",
                    wgsl=shader,
                    input=request.input,
                    count=total,
                    chunk_size=request.chunk_size,
                    output_dtype=report.output_dtype,
                )
                plan = compute_plan(
                    payload,
                    pool_routes.pool.partitions(total, network_id, request.chunk_size),
                )
                add(name, line, end_line, "gpu", "wgsl", (payload, plan), report.wgsl)
        except (ValueError, SyntaxError, TypeError) as exc:
            add(name, line, end_line, "gpu", "wgsl", error=str(exc))

    if not request.source.strip() and request.mode == "animation":
        frames = request.frame_values or [
            {
                "xmin": -0.75 - 1.5 / (1 + i * 0.04),
                "xmax": -0.75 + 1.5 / (1 + i * 0.04),
                "ymin": -1.5 / (1 + i * 0.04),
                "ymax": 1.5 / (1 + i * 0.04),
                "max_iterations": 128,
            }
            for i in range(request.frames)
        ]
        payload = AnimationRequest(
            kind="animation",
            frames=frames,
            fps=request.fps,
            width=request.width,
            height=request.height,
        )
        # The existing coordinator builds Mandelbrot tiles on creation.
        add("Mandelbrot animation", 1, 1, "gpu", "animation")
        segments[-1]["chunk_count"] = (
            len(frames) * math.ceil(request.width / 64) * math.ceil(request.height / 64)
        )
        plans.append((segments[-1], payload, None))
    elif request.filename.lower().endswith(".wgsl") and not re.search(r'\bWGSL_SHADER\s*=', request.source):
        shader_segment(
            request.source, 1, len(request.source.splitlines()), request.filename
        )
    else:
        tree = ast.parse(request.source)
        if sum(1 for _ in ast.walk(tree)) > 12000:
            raise ValueError("Source exceeds 12,000 AST nodes")
        markers = marked_regions(request.source)
        used = set()
        # Every literal shader is inspected, including shaders embedded in functions.
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and "@compute" in n.value
            ):
                shader_ranges.append((n.lineno, n.end_lineno))
                shader_segment(
                    n.value,
                    n.lineno,
                    n.end_lineno,
                    f"WGSL at line {n.lineno}",
                    request.source if "texture_storage_2d" in n.value else None,
                )
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for loop in fn.body:
                if not (
                    isinstance(loop, ast.For)
                    or isinstance(loop, ast.Return)
                    and isinstance(loop.value, ast.ListComp)
                ):
                    continue
                covering = [
                    (i, t)
                    for i, (start, end, t) in enumerate(markers)
                    if start < loop.lineno and loop.end_lineno < end
                ]
                if (
                    request.segmentation == "manual"
                    and not covering
                    and not request.mark
                ):
                    continue
                if shader_ranges and not covering:
                    findings.append(
                        f"Line {loop.lineno}: host Python is not executed; only embedded shaders and explicitly marked loops are submitted."
                    )
                    continue
                gpu, gpu_error = None, None
                try:
                    gpu = convert_loop(fn, loop) if isinstance(loop, ast.For) else convert_comprehension(fn, loop)
                except (ValueError, OverflowError, RecursionError) as exc:
                    gpu_error = str(exc)
                target = request.target if request.target != 'auto' else covering[0][1] if covering else "gpu" if gpu else "cpu"
                if covering:
                    marker_index = covering[0][0]
                    start, end, _ = markers[marker_index]
                    inside = [
                        i + 1
                        for i, s in enumerate(request.source.splitlines())
                        if start < i + 1 < end
                        and s.strip()
                        and not s.lstrip().startswith("#")
                    ]
                    if (
                        not inside
                        or inside[0] != loop.lineno
                        or inside[-1] != loop.end_lineno
                        or marker_index in used
                    ):
                        raise ValueError(
                            "Each CPU/GPU marker pair must enclose exactly one independent loop or comprehension"
                        )
                    used.add(marker_index)
                try:
                    if request.mode == "animation":
                        continue
                    if target == "gpu":
                        if not gpu:
                            raise ValueError(gpu_error)
                        if request.input is None or request.input.dtype != "f32":
                            raise ValueError(
                                "Python array loops require a float32 input array"
                            )
                        code = compile_source(gpu.kernel).wgsl
                        payload = WGSLRequest(
                            kind="wgsl",
                            wgsl=code,
                            splitting="declared",
                            independent=True,
                            input=request.input,
                            count=total,
                            chunk_size=request.chunk_size,
                        )
                        plan = compute_plan(
                            payload,
                            pool_routes.pool.partitions(
                                total, network_id, request.chunk_size
                            ),
                        )
                        add(
                            fn.name,
                            loop.lineno,
                            loop.end_lineno,
                            target,
                            "python",
                            (payload, plan),
                            code,
                        )
                    else:
                        if request.input and request.input.dtype != "f32":
                            raise ValueError("CPU input must be float32")
                        program = lower_cpu(ast.unparse(fn))
                        if program["uses_count"] and values is not None:
                            raise ValueError(
                                "This loop uses a count; turn off Input array and supply an element count"
                            )
                        if values is None and program["indexed"] is False:
                            raise ValueError("This CPU loop requires an input array")
                        plan = cpu_plan(
                            program,
                            values,
                            total,
                            pool_routes.pool.partitions(
                                total, network_id, request.chunk_size, cpu=True
                            ),
                        )
                        add(
                            fn.name,
                            loop.lineno,
                            loop.end_lineno,
                            target,
                            "python",
                            (SimpleNamespace(kind="cpu"), plan),
                        )
                except (ValueError, OverflowError, RecursionError) as exc:
                    add(
                        fn.name,
                        loop.lineno,
                        loop.end_lineno,
                        target,
                        "python",
                        error=str(exc),
                    )
        if len(used) != len(markers) and request.mode != "animation":
            raise ValueError(
                "A marker pair does not match one supported loop or comprehension"
            )
        findings.append(
            "Only the listed independent regions run. Imports, surrounding Python, and dependencies between regions are not executed; each region receives the supplied input."
        )
        if request.mark and not markers:
            marked = mark_source(
                request.source,
                [
                    s
                    for s in segments
                    if s["kind"] == "python" and s["status"] == "ready"
                ],
            )
        else:
            marked = request.source
    if len(plans) > 16:
        raise ValueError("At most 16 independent regions per submission")
    ready = bool(plans) and not any(s["status"] != "ready" for s in segments)
    return {
        "status": "ready" if ready else "unsupported",
        "segments": segments,
        "marked_source": locals().get("marked", request.source),
        "findings": findings
        or [
            "Work is partitioned by current device benchmarks; the most common GPU family receives at most an 8% bonus."
        ],
        "contributors": contributors,
    }, plans


@router.post("/analyze")
async def analyze(network_id: str, payload: ProgramRequest, request: Request):
    access(request, network_id)
    try:
        report, _ = inspect_program(payload, network_id)
        return report
    except (
        ValueError,
        TypeError,
        SyntaxError,
        tokenize.TokenError,
        RecursionError,
        OverflowError,
    ) as exc:
        return {
            "status": "unsupported",
            "segments": [],
            "findings": [str(exc)],
            "contributors": [],
        }


@router.post("/runs", status_code=202)
async def submit(network_id: str, payload: ProgramRequest, request: Request):
    user, _ = access(request, network_id)
    created = []
    try:
        report, plans = inspect_program(payload, network_id)
        if report["status"] != "ready":
            raise ValueError(
                "; ".join(
                    report["findings"]
                    + [f for s in report["segments"] for f in s["findings"]]
                )
                or "No runnable regions"
            )
        for segment, job_request, plan in plans:
            job = pool_routes.pool.create(
                job_request, plan=plan, network_id=network_id, owner_id=user.id
            )
            created.append(
                {
                    "job_id": job.job_id,
                    "name": segment["name"],
                    "target": segment["target"],
                    "line": segment["line"],
                }
            )
        run_id = uuid4().hex
        db.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?,?)",
            (
                run_id,
                network_id,
                user.id,
                payload.filename,
                json.dumps(created),
                time.time(),
            ),
        )
        logger.info(
            "Created run %s in network %s with %d independent jobs",
            run_id,
            network_id,
            len(created),
        )
        return {"id": run_id, "jobs": created, "analysis": report}
    except (
        ValueError,
        TypeError,
        SyntaxError,
        tokenize.TokenError,
        RecursionError,
        OverflowError,
    ) as exc:
        for job in created:
            pool_routes.pool.jobs.pop(job["job_id"], None)
        raise HTTPException(429 if isinstance(exc, CapacityError) else 422, str(exc)) from exc


@router.get("/runs")
def history(network_id: str, request: Request):
    access(request, network_id)
    rows = db.query(
        "SELECT * FROM runs WHERE network_id=? ORDER BY created DESC LIMIT 30",
        (network_id,),
    )
    for row in rows:
        row["jobs"] = [
            j
            | {
                "status": pool_routes.pool.jobs[j["job_id"]].status
                if j["job_id"] in pool_routes.pool.jobs
                else "expired"
            }
            for j in json.loads(row["jobs"])
        ]
    return rows
