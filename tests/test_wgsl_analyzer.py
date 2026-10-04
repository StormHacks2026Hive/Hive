"""WGSL hazards and actual offset-injected GPU execution."""

import struct
import pytest
from backend.wgsl_analyzer import inspect_shader, inject_offset, dispatch_local

SHADER = """@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i = gid.x;
 var v = i;
 for (var j = 0u; j < 100u; j++) { v = v * 1664525u + 1013904223u; }
 out[i] = v;
}"""


def test_metadata_and_injection():
    report = inspect_shader(SHADER)
    assert report.splittable and report.workgroup_size == (64, 1, 1)
    assert report.cost >= 100
    code, binding = inject_offset(SHADER, (256, 1, 1))
    assert binding == 1 and "let gid = hive_split_raw + hive_split.offset.xyz" in code


@pytest.mark.parametrize(
    "change,reason",
    [
        ("atomicAdd(&out[i],1u);", "Atomics"),
        ("workgroupBarrier();", "Barrier"),
        ("out[i] = out[i-1u];", "neighbor"),
        ("out[0] = v;", "Overlapping"),
    ],
)
def test_hazards(change, reason):
    report = inspect_shader(SHADER.replace("out[i] = v;", change))
    assert not report.splittable and reason in " ".join(report.hazards)


def test_four_stitched_tiles_identical():
    wgpu = pytest.importorskip("wgpu")
    try:
        device = wgpu.gpu.request_adapter_sync().request_device_sync()
    except Exception as exc:
        pytest.skip(f"No accessible wgpu device: {exc}")
    try:
        full = dispatch_local(SHADER, (256, 1, 1), {}, 0, 1024, device)
        pieces = []
        for start in range(0, 256, 64):
            result = dispatch_local(
                SHADER, (256, 1, 1), {}, 0, 1024, device, (start, 0, 0), (64, 1, 1)
            )
            pieces.append(result[start * 4 : (start + 64) * 4])
        assert b"".join(pieces) == full
        expected = []
        for i in range(256):
            for _ in range(100):
                i = (i * 1664525 + 1013904223) & 0xFFFFFFFF
            expected.append(i)
        assert list(struct.unpack("<256I", full)) == expected
    finally:
        device.destroy()


@pytest.mark.slow
def test_heavy_hash_benchmark():
    """Calibrate serial batches of bounded GPU work, on one physical Metal GPU."""
    import wgpu
    import numpy as np
    from backend.node_ranker import benchmark_gpu
    from backend.common.nodes import ComputeNode
    from backend.wgsl_analyzer import split_cost_map
    from backend.scheduler import ComputeScheduler
    from tests.benchmark_support import calibrate, report, check_speedup

    try:
        device = wgpu.gpu.request_adapter_sync().request_device_sync()
    except Exception as exc:
        pytest.skip(f"No accessible wgpu GPU: {exc}")
    count = 65536
    try:
        measured = benchmark_gpu(device)
        source = SHADER.replace("j < 100u", "j < 65536u")
        reference = dispatch_local(source, (count, 1, 1), {}, 0, count * 4, device)

        def serial(batches):
            for _ in range(batches):
                actual = dispatch_local(source, (count, 1, 1), {}, 0, count * 4, device)
                assert actual == reference
            return reference

        size, reference, baseline = calibrate(serial, 1)
        simulated = [
            ComputeNode(f"gpu-lane-{i}", measured.node.score * score, device=device)
            for i, score in enumerate([3, 1])
        ]
        plan = split_cost_map(np.ones(count), simulated)
        import time

        start = time.perf_counter()
        finishes = {}
        busy = {n.node_id: 0.0 for n in simulated}
        for _ in range(size):
            actual, timing = ComputeScheduler().render_wgsl(
                source, (count, 1, 1), {}, 0, 4, plan
            )
            assert actual == reference
            for node, finish in timing.finish_times.items():
                finishes[node] = time.perf_counter() - start - timing.elapsed + finish
                busy[node] += finish
        elapsed = time.perf_counter() - start
        result = report(
            "wgsl_hash_one_physical_gpu",
            baseline,
            elapsed,
            {p.node.node_id: (p.stop - p.start) * size for p in plan},
            finishes,
            {node: max(0.0, elapsed - work) for node, work in busy.items()},
            batches=size,
            invocations_per_batch=count,
            iterations_per_invocation=65536,
            physical_gpus=1,
            simulated_scores=[3, 1],
            gpu_breakdown=measured.breakdown,
        )
        check_speedup(result, 1)
    finally:
        device.destroy()


def test_two_dimensional_row_bands():
    import numpy as np
    import wgpu
    from backend.common.nodes import ComputeNode
    from backend.wgsl_analyzer import split_cost_map
    from backend.scheduler import ComputeScheduler

    shader = """struct Dimensions { width: u32, height: u32, pad0: u32, pad1: u32, }
@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[gid.y * dims.width + gid.x] = gid.y * 100u + gid.x;
}"""
    assert inspect_shader(shader).splittable
    try:
        device = wgpu.gpu.request_adapter_sync().request_device_sync()
    except Exception as exc:
        pytest.skip(f"No accessible wgpu device: {exc}")
    try:
        buffers = {0: struct.pack("<4I", 64, 16, 0, 0)}
        nodes = [ComputeNode("a", 3, device=device), ComputeNode("b", 1, device=device)]
        plans = split_cost_map(np.ones((16, 64)), nodes)
        reference = dispatch_local(shader, (64, 16, 1), buffers, 1, 4096, device)
        actual, _ = ComputeScheduler().render_wgsl(
            shader, (64, 16, 1), buffers, 1, 4, plans
        )
        assert actual == reference
        with pytest.raises(ValueError, match="dimension"):
            dispatch_local(
                shader,
                (64, 16, 1),
                {0: struct.pack("<4I", 32, 16, 0, 0)},
                1,
                4096,
                device,
            )
    finally:
        device.destroy()


def test_shadowed_index_refused():
    shader = SHADER.replace("out[i] = v;", "{ let i = 0u; out[i] = v; }")
    assert not inspect_shader(shader).splittable
