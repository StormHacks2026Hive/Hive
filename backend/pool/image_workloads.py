"""Analyze the registered Mandelbulb renderer and plan independent image tiles.

Parse Python only with ast: extract the literal WGSL_SHADER and literal defaults
from RenderConfig, never import or execute an uploaded program. Raw WGSL is also
accepted with renderer defaults. Require the shader to match the registered
Mandelbulb module after comment/whitespace normalization, rather than claiming
that arbitrary texture shaders or helper functions are safe. This known shader
reads only uniforms, runs invocation-local distance/ray calculations, and writes
one pixel at its global ID. Add a tile offset/extent uniform and rewrite only the
textureStore coordinates to tile-local IDs. Camera coordinates remain global,
so independently rendered tiles match the full texture render. Dynamic loop
bounds are validated renderer settings; camera animation and Python code outside
the literal definitions are not executed. Pixel tiles are transport units;
Coordinator reservations weight their assignment by measured node scores.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import math
import re
import struct
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from ..models import Model
from .models import Tile

SHADER = Path(__file__).with_name("mandelbulb.wgsl").read_text()


class RenderSettings(Model):
    """Validated values matching the renderer's 176-byte uniform layout."""

    width: int = Field(default=1280, ge=1, le=4096)
    height: int = Field(default=1024, ge=1, le=4096)
    max_steps: int = Field(default=400, ge=1, le=2048)
    fractal_iterations: int = Field(default=12, ge=1, le=64)
    power: float = Field(default=8.0, gt=1, le=32)
    bailout: float = Field(default=4.0, gt=0, le=100)
    max_distance: float = Field(default=20.0, gt=0, le=1000)
    hit_epsilon: float = Field(default=0.0005, gt=0, le=1)
    normal_epsilon: float = Field(default=0.001, gt=0, le=1)
    fov_degrees: float = Field(default=45.0, gt=0, lt=179)
    ambient_strength: float = Field(default=0.15, ge=0, le=10)
    diffuse_strength: float = Field(default=1.0, ge=0, le=10)
    specular_strength: float = Field(default=0.35, ge=0, le=10)
    shininess: float = Field(default=32.0, gt=0, le=1024)
    shadow_softness: float = Field(default=12.0, gt=0, le=1000)
    ao_strength: float = Field(default=1.0, ge=0, le=10)
    camera_position: tuple[float, float, float] = (3.2, 2.2, 3.2)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    light_direction: tuple[float, float, float] = (-0.5, 0.8, -0.6)
    object_color: tuple[float, float, float] = (0.95, 0.55, 0.12)
    background_top: tuple[float, float, float] = (0.04, 0.06, 0.10)
    background_bottom: tuple[float, float, float] = (0.005, 0.008, 0.015)
    enable_shadows: bool = True
    enable_ao: bool = True

    @model_validator(mode="after")
    def bounded(self) -> RenderSettings:
        if self.width * self.height > 8_847_360:
            raise ValueError("Image limit is 8,847,360 pixels")
        for name in (
            "camera_position",
            "camera_target",
            "light_direction",
            "object_color",
            "background_top",
            "background_bottom",
        ):
            if any(
                not math.isfinite(value) or abs(value) > 1e4
                for value in getattr(self, name)
            ):
                raise ValueError("Vector values must be finite and bounded")
        if (
            self.camera_position[0] == self.camera_target[0]
            and self.camera_position[2] == self.camera_target[2]
        ):
            raise ValueError("Camera direction must not be parallel to world-up")
        if not any(self.light_direction):
            raise ValueError("Light direction must be nonzero")
        return self


class ImageAnalysisRequest(Model):
    source: str = Field(min_length=1, max_length=65536)
    width: int | None = Field(default=None, ge=1, le=4096)
    height: int | None = Field(default=None, ge=1, le=4096)


class WGSLImageRequest(ImageAnalysisRequest):
    kind: Literal["wgsl_image"]


class ImageAnalysis(Model):
    status: Literal["ready", "unsupported"]
    kind: Literal["wgsl_image"] = "wgsl_image"
    findings: list[str]
    width: int = 0
    height: int = 0
    chunk_count: int = 0
    output_shape: list[int] = Field(default_factory=list)
    wgsl: str | None = None


def normalized(shader: str) -> str:
    """Canonical comparison permits only comments/formatting differences."""
    return re.sub(
        r"\s+", "", re.sub(r"/\*.*?\*/|//[^\n]*", "", shader, flags=re.DOTALL)
    )


def extract(request: ImageAnalysisRequest) -> tuple[str, RenderSettings]:
    """Read literal renderer data without running imports, functions or __main__."""
    shader = request.source
    settings: dict[str, Any] = {}
    if (
        not request.source.lstrip().startswith(("struct", "//", "/*", "@"))
        or "WGSL_SHADER" in request.source
    ):
        tree = ast.parse(request.source)
        if sum(1 for _ in ast.walk(tree)) > 12000:
            raise ValueError("Renderer exceeds AST size limit")
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "WGSL_SHADER" for t in node.targets
            )
        ]
        if len(assignments) != 1:
            raise ValueError(
                "Python renderer must contain one literal WGSL_SHADER assignment"
            )
        shader = ast.literal_eval(assignments[0].value)
        if not isinstance(shader, str):
            raise ValueError("WGSL_SHADER must be a literal string")
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RenderConfig"
        ]
        if len(classes) > 1:
            raise ValueError("Expected at most one RenderConfig class")
        for node in classes[0].body if classes else []:
            target = (
                node.target
                if isinstance(node, ast.AnnAssign)
                else node.targets[0]
                if isinstance(node, ast.Assign) and len(node.targets) == 1
                else None
            )
            if (
                isinstance(target, ast.Name)
                and target.id in RenderSettings.model_fields
            ):
                settings[target.id] = ast.literal_eval(node.value)
    if normalized(shader) != normalized(SHADER):
        raise ValueError(
            "This image upload supports the registered Mandelbulb shader; altered or arbitrary texture shaders need a separate independence analysis"
        )
    if request.width is not None:
        settings["width"] = request.width
    if request.height is not None:
        settings["height"] = request.height
    return shader, RenderSettings(**settings)


def tiled_shader(shader: str) -> str:
    """Keep global ray coordinates; write pixels to each node's local texture."""
    header = """struct HiveImageTile { origin: vec4<u32>, extent: vec4<u32>, }
@group(0) @binding(2) var<uniform> hive_image_tile: HiveImageTile;
"""
    rewritten = re.sub(
        r"(@builtin\(global_invocation_id\)\s*)id\b",
        r"\1hive_image_raw",
        shader,
        count=1,
    )
    entry = re.search(r"\bfn\s+main\s*\(", rewritten)
    start = rewritten.index("{", entry.start()) + 1
    rewritten = (
        rewritten[:start]
        + """\n    if (any(hive_image_raw.xy >= hive_image_tile.extent.xy)) { return; }
    let id = hive_image_raw + hive_image_tile.origin.xyz;
"""
        + rewritten[start:]
    )
    rewritten, count = re.subn(
        r"(textureStore\(\s*output_texture\s*,\s*vec2<i32>\(\s*)i32\(id.x\)(\s*,\s*)i32\(id.y\)",
        r"\1i32(hive_image_raw.x)\2i32(hive_image_raw.y)",
        rewritten,
    )
    if count != 1:
        raise ValueError("Expected exactly one registered pixel write")
    return header + rewritten


def uniform_data(config: RenderSettings) -> bytes:
    """Pack the exact original renderer layout in little-endian form."""
    floats = [
        config.power,
        config.bailout,
        config.max_distance,
        config.hit_epsilon,
        config.normal_epsilon,
        math.radians(config.fov_degrees),
        config.ambient_strength,
        config.diffuse_strength,
        config.specular_strength,
        config.shininess,
        config.shadow_softness,
        config.ao_strength,
    ]
    vectors = []
    for name, padding in [
        ("camera_position", 0.0),
        ("camera_target", 0.0),
        ("light_direction", 0.0),
        ("object_color", 1.0),
        ("background_top", 1.0),
        ("background_bottom", 1.0),
    ]:
        vectors.extend([*getattr(config, name), padding])
    return struct.pack(
        "<4I36f4I",
        config.width,
        config.height,
        config.max_steps,
        config.fractal_iterations,
        *floats,
        *vectors,
        int(config.enable_shadows),
        int(config.enable_ao),
        0,
        0,
    )


def image_plan(request: ImageAnalysisRequest) -> dict[str, Any]:
    """Plan bounded 128x128 texture tiles with clipped right/bottom edges."""
    _shader, config = extract(request)
    code = tiled_shader(SHADER).encode()
    shader_id = hashlib.sha256(code).hexdigest()
    packed = base64.b64encode(uniform_data(config)).decode()
    chunks = []
    for y in range(0, config.height, 128):
        for x in range(0, config.width, 128):
            tile = Tile(
                x=x,
                y=y,
                width=min(128, config.width - x),
                height=min(128, config.height - y),
            )
            count = tile.width * tile.height
            chunks.append(
                {
                    "chunk_id": f"image-{y}-{x}",
                    "tile": tile,
                    "count": count,
                    "byte_length": count * 4,
                    "assignment": {
                        "kind": "texture_tile",
                        "shader_id": shader_id,
                        "tile": tile.model_dump(),
                        "image": {"width": config.width, "height": config.height},
                        "uniform_data": packed,
                        "output_format": "rgba8",
                    },
                }
            )
    return {
        "chunks": chunks,
        "size": config.width * config.height * 4,
        "assets": {shader_id: code},
        "output_format": "rgba8",
        "output_shape": [1, config.height, config.width, 4],
        "width": config.width,
        "height": config.height,
    }


def analyze_image(request: ImageAnalysisRequest) -> ImageAnalysis:
    """Report supported upload data and explain still-frame execution."""
    try:
        plan = image_plan(request)
        return ImageAnalysis(
            status="ready",
            width=plan["width"],
            height=plan["height"],
            chunk_count=len(plan["chunks"]),
            output_shape=plan["output_shape"],
            wgsl=next(iter(plan["assets"].values())).decode(),
            findings=[
                "Mandelbulb pixels are independent; camera coordinates remain global while each tile writes its own texture.",
                "Shader and config defaults were read without executing Python. One still frame is rendered; the Python rotation loop is not executed.",
                "Tiles are assigned using contributor throughput scores and assembled into an RGBA image.",
            ],
        )
    except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
        return ImageAnalysis(status="unsupported", findings=[str(exc)])
