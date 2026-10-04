"""Conservative WGSL parsing and score/cost-weighted domain splitting.

The browser compatibility analyzer parses a complete restricted 1D elementwise
module and rewrites storage indices to local ranges plus a global offset. The
new domain analyzer extracts compute entry points, workgroup dimensions and
bindings; it accepts canonical gid-based output indexing, literal bounded loops,
and local arithmetic. Atomics, barriers, pointer escapes, output reads and
noncanonical array accesses are refused. Offset injection renames the builtin
parameter and introduces a bounds-guarded logical gid from an added uniform.
Full-domain buffers preserve original indexing; gather copies only owned regions.
Cost maps come from caller-provided low-resolution invocation-cost probes and
are aggregated into row bands with score-proportional cumulative cost. This is
not a complete WGSL compiler or an independence proof for arbitrary functions:
helper functions and unrecognized memory operations require a manual contract.
"""

import re
import math
from typing import Literal
from pydantic import Field
from .models import Model


class WGSLAnalysisRequest(Model):
    source: str = Field(min_length=1, max_length=32000)
    count: int = Field(default=1, ge=1, le=2_000_000)
    chunk_size: int = Field(default=4096, ge=1, le=4096)


class WGSLAnalysis(Model):
    status: Literal["ready", "manual_contract_required", "unsupported"]
    findings: list[str]
    wgsl: str | None = None
    input_dtype: str | None = None
    output_dtype: str | None = None
    chunk_count: int = 0
    ranges: list[dict[str, int]] = Field(default_factory=list)


HEADER = """struct HiveParams { offset: u32, count: u32, seed: u32, padding: u32, }
@group(0) @binding(0) var<uniform> hive_params: HiveParams;
"""


def analyze_wgsl(request: WGSLAnalysisRequest) -> WGSLAnalysis:
    if math.ceil(request.count / request.chunk_size) > 2048:
        return WGSLAnalysis(
            status="unsupported", findings=["Maximum 2048 chunks; increase chunk_size"]
        )
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", request.source, flags=re.S).strip()
    findings = []
    if re.search(
        r"\b(atomic\w*|workgroupBarrier|storageBarrier|texture\w*|workgroup)\b", code
    ):
        return WGSLAnalysis(
            status="unsupported",
            findings=[
                "Shared memory, atomics, textures, and barriers require a manual algorithm rewrite before range splitting"
            ],
        )
    # Only recognize complete modules consisting of 0/1 inputs, one output,
    # and a straight-line compute function. Every token must be accounted for.
    declaration = r"@group\(\s*0\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var<\s*storage\s*,\s*(read|read_write)\s*>\s*(\w+)\s*:\s*array<\s*(f32|u32|i32)\s*>\s*;"
    buffers = re.findall(declaration, code)
    remainder = re.sub(declaration, "", code).strip()
    main = re.fullmatch(
        r"@compute\s*@workgroup_size\(\s*(\d+)(?:\s*,\s*1\s*,\s*1)?\s*\)\s*fn\s+main\s*\(\s*@builtin\(global_invocation_id\)\s*(\w+)\s*:\s*vec3<u32>\s*\)\s*\{(.*)\}",
        remainder,
        flags=re.S,
    )
    reads = [b for b in buffers if b[1] == "read"]
    writes = [b for b in buffers if b[1] == "read_write"]
    if (
        not main
        or len(reads) > 1
        or len(writes) != 1
        or len({b[0] for b in buffers}) != len(buffers)
        or len({b[2] for b in buffers}) != len(buffers)
    ):
        return WGSLAnalysis(
            status="manual_contract_required",
            findings=[
                "Automatic analysis accepts a complete shader with one writable array, at most one read-only array, and a straight-line 1D main function. For other independent algorithms use the declared Hive binding contract."
            ],
        )
    gid, body = main[2], main[3].strip()
    reserved = {
        gid,
        *(b[2] for b in buffers),
        "hive_params",
        "HiveParams",
        "hive_local_index",
        "hive_gid",
        "hive_input_value",
        "hive_global_index",
    }
    if len(reserved) != len(buffers) + 7:
        return WGSLAnalysis(
            status="unsupported",
            findings=["Rename identifiers that collide with HiveParams or hive_params"],
        )
    index = re.match(
        r"(?:let|var)\s+(\w+)(?:\s*:\s*u32)?\s*=\s*" + re.escape(gid) + r"\.x\s*;", body
    )
    local_index = index[1] if index else None
    if index:
        body = body[index.end() :].strip()
        if local_index in reserved:
            return WGSLAnalysis(
                status="unsupported",
                findings=["The local index must have a distinct name"],
            )
    # Strip one conventional bounds guard; add the actual per-chunk count guard.
    guard = re.fullmatch(r"if\s*\(?\s*(.*?)\s*\)?\s*\{(.*)\}", body, flags=re.S)
    if guard:
        expected = (
            (local_index or gid + ".x")
            + r"\s*<\s*arrayLength\(\s*&\s*"
            + re.escape(writes[0][2])
            + r"\s*\)"
        )
        if not re.fullmatch(expected, guard[1].strip()):
            return WGSLAnalysis(
                status="unsupported",
                findings=[
                    "Only an output arrayLength guard can be rewritten automatically"
                ],
            )
        body = guard[2].strip()
    statements = [s.strip() for s in body.split(";") if s.strip()]
    allowed_names = set()
    builtins = {
        "abs",
        "min",
        "max",
        "clamp",
        "sqrt",
        "sin",
        "cos",
        "exp",
        "log",
        "f32",
        "u32",
        "i32",
    }
    out_name, out_dtype = writes[0][2:]
    normalized = []
    input_used = False
    for number, statement in enumerate(statements):
        expr = None
        local = re.fullmatch(
            r"let\s+(\w+)(?:\s*:\s*(f32|u32|i32))?\s*=\s*(.*)", statement, flags=re.S
        )
        target = re.fullmatch(
            re.escape(out_name) + r"\[\s*(.*?)\s*\]\s*=\s*(.*)", statement, flags=re.S
        )
        if (
            local
            and number < len(statements) - 1
            and local[1] not in reserved | allowed_names | builtins | {local_index}
        ):
            expr = local[3]
        elif (
            target
            and number == len(statements) - 1
            and target[1] in (local_index, gid + ".x")
        ):
            expr = target[2]
        else:
            findings.append(
                "Use iteration-local let assignments followed by exactly one output[index] write"
            )
            break
        if re.search(
            r"\bhive_(?:input_value|global_index|local_index|gid|params)\b", expr
        ):
            findings.append("Rename identifiers reserved for Hive shader rewriting")
            break
        # Replace only exact current-element reads. No neighboring reads,
        # output reads, pointer operations, assignment expressions, or captures.
        if reads:
            input_name = reads[0][2]
            pattern = (
                re.escape(input_name)
                + r"\[\s*("
                + re.escape(local_index or gid + ".x")
                + r")\s*\]"
            )
            expr, replacements = re.subn(pattern, "hive_input_value", expr)
            input_used = input_used or replacements > 0
        expr = re.sub(r"\b" + re.escape(gid) + r"\.x\b", "hive_global_index", expr)
        if local_index:
            expr = re.sub(
                r"\b" + re.escape(local_index) + r"\b", "hive_global_index", expr
            )
        tokens = re.findall(
            r"(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[fui]?)|(?:[A-Za-z_]\w*)|[()+*/%,\-]",
            expr,
        )
        if "".join(tokens) != re.sub(r"\s+", "", expr) or any(
            re.match(r"[A-Za-z_]", t)
            and t
            not in allowed_names | builtins | {"hive_input_value", "hive_global_index"}
            for t in tokens
        ):
            findings.append(
                "Expressions may read only the current input element, global index, numeric constants, scalar math, and previously assigned local values"
            )
            break
        expr = expr.replace(
            "hive_global_index", "(hive_params.offset + hive_local_index)"
        )
        if reads:
            expr = expr.replace("hive_input_value", f"{reads[0][2]}[hive_local_index]")
        if local:
            normalized.append(
                f"let {local[1]}"
                + (f": {local[2]}" if local[2] else "")
                + f" = {expr};"
            )
            allowed_names.add(local[1])
        else:
            normalized.append(f"{out_name}[hive_local_index] = {expr};")
    if reads and not input_used:
        findings.append(
            "Remove the unused input array declaration; it would not appear in the WebGPU binding layout"
        )
    if findings or not statements:
        return WGSLAnalysis(
            status="unsupported",
            findings=findings or ["The shader must write one output per invocation"],
        )
    declaration_code = ""
    if reads:
        declaration_code += f"@group(0) @binding(1) var<storage, read> {reads[0][2]}: array<{reads[0][3]}>;\n"
    declaration_code += f"@group(0) @binding(2) var<storage, read_write> {out_name}: array<{out_dtype}>;\n"
    rewritten = (
        HEADER
        + declaration_code
        + "@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) hive_gid: vec3<u32>) {\n    let hive_local_index = hive_gid.x;\n    if (hive_local_index < hive_params.count) {\n"
        + "".join(f"        {s}\n" for s in normalized)
        + "    }\n}\n"
    )
    ranges = [
        {"offset": o, "count": min(request.chunk_size, request.count - o)}
        for o in range(
            0, min(request.count, 64 * request.chunk_size), request.chunk_size
        )
    ]
    return WGSLAnalysis(
        status="ready",
        findings=[
            "Each invocation writes one distinct output element. Input reads are local to the same element; no inter-chunk dependencies were found.",
            "The shader was rewritten with local buffer indexing, global offsets, a count guard, and workgroups of 64. Browser WebGPU compilation is still required.",
        ],
        wgsl=rewritten,
        input_dtype=reads[0][3] if reads else None,
        output_dtype=out_dtype,
        chunk_count=math.ceil(request.count / request.chunk_size),
        ranges=ranges,
    )


from dataclasses import dataclass
from typing import Callable, Any
import logging
import numpy as np
from .common.nodes import ComputeNode

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShaderAnalysis:
    """Static shader metadata, explicit hazards and heuristic instruction cost."""

    entry_point: str
    workgroup_size: tuple[int, int, int]
    dimensions: int
    bindings: tuple[tuple[int, int, str, str], ...]
    gid: str
    cost: float
    hazards: tuple[str, ...]

    @property
    def splittable(self) -> bool:
        """Unknown semantics and known hazards both prevent automatic splitting."""
        return not self.hazards


@dataclass(frozen=True)
class TilePlan:
    """One full-width row band, or contiguous cells of a 1D domain."""

    node: ComputeNode
    start: int
    stop: int
    estimated_cost: float


def inspect_shader(source: str) -> ShaderAnalysis:
    """Extract metadata and reject constructs outside canonical independent WGSL."""
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    hazards: list[str] = []
    entries = re.findall(
        r"@compute\s*@workgroup_size\(([^)]+)\)\s*fn\s+(\w+)\s*\(\s*@builtin\(global_invocation_id\)\s*(\w+)\s*:\s*vec3<u32>\s*\)",
        code,
    )
    if len(entries) != 1:
        return ShaderAnalysis(
            "",
            (1, 1, 1),
            0,
            (),
            "",
            0,
            ("Need exactly one canonical compute entry with global_invocation_id",),
        )
    size, entry, gid = entries[0]
    try:
        sizes = tuple(int(v.strip().rstrip("u")) for v in size.split(","))
        if not 1 <= len(sizes) <= 3 or any(v <= 0 for v in sizes):
            raise ValueError
        group = sizes + (1,) * (3 - len(sizes))
    except ValueError:
        group = (1, 1, 1)
        hazards.append("Nonliteral workgroup_size")
    bindings = tuple(
        (int(g), int(b), access.strip(), name)
        for g, b, access, name in re.findall(
            r"@group\(\s*(\d+)\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var<([^>]+)>\s*(\w+)\s*:",
            code,
        )
    )
    if re.search(r"\batomic\w*\b", code):
        hazards.append("Atomics may share state across invocations")
    if re.search(r"\b(?:workgroupBarrier|storageBarrier|textureBarrier)\b", code):
        hazards.append("Barrier synchronization across a split axis is unsupported")
    if re.search(r"\bvar<\s*workgroup\b", code):
        hazards.append("Shared workgroup memory")
    if re.search(r"\b(?:texture\w*|ptr)\b|&", code):
        hazards.append("Texture/pointer memory semantics are not proven independent")
    if len(re.findall(r"\bfn\s+", code)) != 1:
        hazards.append("Helper functions require a manual independence contract")
    if re.search(r"\b(?:while|loop)\b", code):
        hazards.append("Unbounded loop cost")
    if "hive_split" in code:
        hazards.append("Reserved hive_split identifier collision")
    aliases = {gid + ".x"}
    for alias, expr in re.findall(r"let\s+(\w+)\s*=\s*([^;]+);", code):
        if expr.strip() == gid + ".x":
            aliases.add(alias)
    for alias in aliases - {gid + ".x"}:
        if (
            len(re.findall(r"\b(?:let|var|const)\s+" + re.escape(alias) + r"\b", code))
            != 1
        ):
            hazards.append("Shadowed invocation index: " + alias)
    dims = 1
    if re.search(r"\b" + re.escape(gid) + r"\.z\b", code):
        dims = 3
    elif re.search(r"\b" + re.escape(gid) + r"\.y\b", code):
        dims = 2
    outputs = [
        name
        for _, _, access, name in bindings
        if re.fullmatch(r"storage\s*,\s*read_write", access)
    ]
    storage = [name for _, _, access, name in bindings if access.startswith("storage")]
    if len(outputs) != 1:
        hazards.append("Need exactly one writable storage output")
    for name in storage:
        accesses = list(re.finditer(r"\b" + re.escape(name) + r"\s*\[([^]]+)\]", code))
        if not accesses:
            hazards.append(f"Unused storage binding {name}")
        for access in accesses:
            index = re.sub(r"\s+", "", access[1])
            canonical = index in aliases
            # The only accepted multidimensional index is row-major with an
            # explicit uniform width/depth stride and direct logical gid axes.
            if dims == 2:
                canonical = bool(
                    re.fullmatch(
                        re.escape(gid) + r"\.y\*\w+\.width\+" + re.escape(gid) + r"\.x",
                        index,
                    )
                )
            if dims == 3:
                canonical = bool(
                    re.fullmatch(
                        r"\("
                        + re.escape(gid)
                        + r"\.z\*\w+\.height\+"
                        + re.escape(gid)
                        + r"\.y\)\*\w+\.width\+"
                        + re.escape(gid)
                        + r"\.x",
                        index,
                    )
                )
            if not canonical:
                hazards.append(
                    f"Overlapping writes or neighbor reads: {name}[{access[1]}]"
                )
            tail = code[access.end() :]
            if name in outputs and not re.match(r"\s*=(?!=)", tail):
                hazards.append("Output reads or compound writes introduce dependencies")
        # Disallow bare references passed to functions or aliased as pointers.
        scrub = re.sub(r"\b" + re.escape(name) + r"\s*\[[^]]+\]", "", code)
        scrub = re.sub(r"var<[^>]+>\s*" + re.escape(name) + r"\s*:[^;]+;", "", scrub)
        if re.search(r"\b" + re.escape(name) + r"\b", scrub):
            hazards.append(f"Storage alias/escape: {name}")
    bounds = re.findall(r"for\s*\([^;]*;\s*\w+\s*<\s*(\d+)u?\s*;", code)
    if len(bounds) != len(re.findall(r"\bfor\s*\(", code)):
        hazards.append("Dynamic loop bounds require a cost contract")
    instructions = len(
        re.findall(r"[+*/%^-]|\b(?:sin|cos|sqrt|pow|exp|log)\s*\(", code)
    )
    cost = max(1, instructions) * math.prod(int(b) for b in bounds)
    return ShaderAnalysis(
        entry, group, dims, bindings, gid, float(cost), tuple(dict.fromkeys(hazards))
    )


def inject_offset(source: str, domain: tuple[int, int, int]) -> tuple[str, int]:
    """Add a 32-byte uniform: offset.xyz, pad, extent.xyz, pad; preserve bindings."""
    report = inspect_shader(source)
    if not report.splittable:
        raise ValueError("; ".join(report.hazards))
    if len(domain) != 3 or any(v <= 0 for v in domain):
        raise ValueError("Positive xyz domain required")
    binding = max((b for g, b, _, _ in report.bindings if g == 0), default=-1) + 1
    code = re.sub(
        r"(@builtin\(global_invocation_id\)\s*)" + re.escape(report.gid) + r"\b",
        r"\1hive_split_raw",
        source,
        count=1,
    )
    # builtin annotation contains parentheses; locate body from function name.
    fn_start = re.search(
        r"\bfn\s+" + re.escape(report.entry_point) + r"\s*\(", code
    ).start()
    body_start = code.index("{", fn_start) + 1
    prefix = f"\nif (any(hive_split_raw >= hive_split.extent.xyz)) {{ return; }}\nlet {report.gid} = hive_split_raw + hive_split.offset.xyz;\n"
    code = code[:body_start] + prefix + code[body_start:]
    header = f"struct HiveSplit {{ offset: vec4<u32>, extent: vec4<u32>, }}\n@group(0) @binding({binding}) var<uniform> hive_split: HiveSplit;\n"
    return header + code, binding


def probe_cost_map(
    probe: Callable[[tuple[int, int]], Any],
    domain: tuple[int, int],
    resolution: tuple[int, int] = (64, 36),
) -> np.ndarray:
    """Upsample a cheap cost probe; probe must return positive invocation costs."""
    width, height = domain
    rw, rh = min(resolution[0], width), min(resolution[1], height)
    costs = np.asarray(probe((rw, rh)), dtype=float)
    if costs.shape != (rh, rw) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Probe must return a finite positive (height,width) cost map")
    log.info("Cost probe %sx%s for domain %sx%s", rw, rh, width, height)
    return costs[
        np.minimum(np.arange(height) * rh // height, rh - 1)[:, None],
        np.minimum(np.arange(width) * rw // width, rw - 1),
    ]


def split_cost_map(costs: Any, nodes: list[ComputeNode]) -> list[TilePlan]:
    """Cut cumulative row cost according to measured score and transfer overhead."""
    from .node_ranker import allocate

    array = np.asarray(costs, dtype=float)
    if array.ndim not in (1, 2) or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError("Costs must be positive finite 1D or 2D values")
    rows = array if array.ndim == 1 else array.sum(axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(rows)))
    shares = allocate(1_000_000, nodes)
    total_shares = sum(shares.values())
    if not total_shares:
        raise ValueError("No online nodes")
    plans = []
    start, target = 0, 0.0
    for node, share in shares.items():
        if not share:
            continue
        target += share / total_shares * cumulative[-1]
        boundary = int(np.searchsorted(cumulative, target))
        candidates = [
            min(len(rows), max(start, boundary - 1)),
            min(len(rows), max(start, boundary)),
        ]
        stop = min(candidates, key=lambda c: abs(cumulative[c] - target))
        if stop > start:
            plans.append(
                TilePlan(node, start, stop, float(cumulative[stop] - cumulative[start]))
            )
        start = stop
    if start < len(rows):
        last = plans[-1]
        plans[-1] = TilePlan(
            last.node,
            last.start,
            len(rows),
            float(cumulative[-1] - cumulative[last.start]),
        )
    return plans


def dispatch_local(
    source: str,
    domain: tuple[int, int, int],
    buffers: dict[int, bytes],
    output_binding: int,
    output_bytes: int,
    device: Any,
    offset: tuple[int, int, int] = (0, 0, 0),
    extent: tuple[int, int, int] | None = None,
) -> bytes:
    """Execute injected shader on wgpu; full-domain storage retains global indices."""
    import struct
    import wgpu

    report = inspect_shader(source)
    code, split_binding = inject_offset(source, domain)
    region = extent or domain
    if any(o < 0 or e <= 0 or o + e > d for o, e, d in zip(offset, region, domain)):
        raise ValueError("Split region must lie within the logical domain")
    if report.dimensions > 1:
        # A canonical row-major write is injective only if the uniform stride
        # agrees with the logical domain. Limit this ABI to scalar u32 fields.
        for _, binding, access, name in report.bindings:
            if access != "uniform":
                continue
            declaration = re.search(
                r"var<\s*uniform\s*>\s*" + re.escape(name) + r"\s*:\s*(\w+)", source
            )
            structure = (
                re.search(
                    r"\bstruct\s+" + re.escape(declaration[1]) + r"\s*\{([^}]+)\}",
                    source,
                )
                if declaration
                else None
            )
            if structure is None:
                raise ValueError("Need a scalar dimension uniform struct")
            fields = [part.strip() for part in structure[1].split(",") if part.strip()]
            for axis, field_name in [(0, "width"), (1, "height")]:
                indices = [
                    i
                    for i, field in enumerate(fields)
                    if re.fullmatch(field_name + r"\s*:\s*u32", field)
                ]
                if indices:
                    index = indices[0]
                    if any(
                        not re.fullmatch(r"\w+\s*:\s*(?:u32|i32|f32)", f)
                        for f in fields[: index + 1]
                    ):
                        raise ValueError(
                            "Only scalar dimension uniform layouts are supported"
                        )
                    if (
                        binding not in buffers
                        or len(buffers[binding]) < (index + 1) * 4
                        or struct.unpack_from("<I", buffers[binding], index * 4)[0]
                        != domain[axis]
                    ):
                        raise ValueError(
                            "Uniform dimension must match the logical domain"
                        )
    uniform = struct.pack("<8I", *offset, 0, *region, 0)
    resources = []
    entries = []
    try:
        all_buffers = dict(buffers)
        all_buffers[split_binding] = uniform
        for binding, data in all_buffers.items():
            is_uniform = binding == split_binding or any(
                b == binding and access == "uniform"
                for g, b, access, _ in report.bindings
                if g == 0
            )
            usage = wgpu.BufferUsage.UNIFORM if is_uniform else wgpu.BufferUsage.STORAGE
            buffer = device.create_buffer_with_data(
                data=data,
                usage=usage | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
            )
            resources.append(buffer)
            entries.append({"binding": binding, "resource": {"buffer": buffer}})
        if output_binding not in all_buffers:
            buffer = device.create_buffer(
                size=output_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            )
            resources.append(buffer)
            entries.append({"binding": output_binding, "resource": {"buffer": buffer}})
        output = next(
            e["resource"]["buffer"] for e in entries if e["binding"] == output_binding
        )
        if any(g != 0 for g, _, _, _ in report.bindings):
            raise ValueError("Local dispatch currently requires group 0 bindings")
        module = device.create_shader_module(code=code)
        pipeline = device.create_compute_pipeline(
            layout="auto", compute={"module": module, "entry_point": report.entry_point}
        )
        bind = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0), entries=entries
        )
        encoder = device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, bind)
        compute.dispatch_workgroups(
            *(math.ceil(e / g) for e, g in zip(region, report.workgroup_size))
        )
        compute.end()
        device.queue.submit([encoder.finish()])
        return bytes(device.queue.read_buffer(output, 0, output_bytes))
    finally:
        for resource in resources:
            resource.destroy()
