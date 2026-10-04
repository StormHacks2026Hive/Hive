"""Measure node compute and allocate work to minimize predicted makespan.

CPU score is integer ALU iterations/second times usable cores. GPU score is
ALU operations/second multiplied by the geometric mean of bandwidth and transfer
ratios relative to fixed reference rates (100 GB/s and 10 GB/s). Breakdown retains
all raw rates: CPU/GPU scores are workload-specific proxies, not interchangeable
runtime guarantees. Remote nodes must run the same probe through their transport;
no invented network measurements. Cache measurements by node ID with a monotonic
TTL. Allocation solves sum(max(0,(T-latency)/(1/score+bytes_per_work/bandwidth)))
= total_work, then rounds integer shares with largest remainders. Slow links and
high startup latency reduce allocation, possibly to zero; offline nodes are
excluded on every call, including cached ranking.
"""

from __future__ import annotations
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Any
from .common.nodes import ComputeNode, WorkRange

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Measurement:
    """Comparable workload score plus independently measured raw quantities."""

    node: ComputeNode
    breakdown: dict[str, float]
    measured_at: float


def allocate(total_work: int, nodes: list[ComputeNode]) -> dict[ComputeNode, int]:
    """Allocate integer work minimizing affine completion time, not equal shares."""
    if (
        isinstance(total_work, bool)
        or not isinstance(total_work, int)
        or total_work < 0
    ):
        raise ValueError("total_work must be a nonnegative integer")
    if len({n.node_id for n in nodes}) != len(nodes):
        raise ValueError("Duplicate node IDs")
    active = [n for n in nodes if n.online and math.isfinite(n.score) and n.score > 0]
    for n in active:
        if (
            not math.isfinite(n.latency)
            or n.latency < 0
            or n.bandwidth <= 0
            or math.isnan(n.bandwidth)
            or not math.isfinite(n.bytes_per_work)
            or n.bytes_per_work < 0
        ):
            raise ValueError("Invalid node transfer measurements")
    if not active:
        if total_work:
            raise ValueError("No online ranked nodes")
        return {}
    if total_work == 0:
        return {n: 0 for n in active}
    slopes = {n: 1 / n.score + n.bytes_per_work / n.bandwidth for n in active}
    lo = 0.0
    hi = min(n.latency + total_work * slopes[n] for n in active)
    for _ in range(100):
        mid = (lo + hi) / 2
        if sum(max(0.0, (mid - n.latency) / slopes[n]) for n in active) < total_work:
            lo = mid
        else:
            hi = mid
    real = {n: max(0.0, (hi - n.latency) / slopes[n]) for n in active}
    # Normalize small bisection roundoff before largest-remainder rounding.
    factor = total_work / sum(real.values())
    real = {n: v * factor for n, v in real.items()}
    shares = {n: int(v) for n, v in real.items()}
    for n in sorted(active, key=lambda n: real[n] - shares[n], reverse=True)[
        : total_work - sum(shares.values())
    ]:
        shares[n] += 1
    return shares


def ranges(total_work: int, nodes: list[ComputeNode]) -> list[WorkRange]:
    """Return ordered half-open ranges with no overlap or gaps."""

    start = 0
    result = []
    for node, count in allocate(total_work, nodes).items():
        if count:
            result.append(WorkRange(node, start, start + count))
            start += count
    return result


def _cpu_loop(iterations: int) -> int:
    value = 1
    for _ in range(iterations):
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    return value


def benchmark_cpu(
    node_id: str = "local-cpu", cores: int | None = None, repeats: int = 5
) -> Measurement:
    """Warm an integer ALU microbenchmark; use median of timed ~100ms samples."""
    iterations = 100_000
    _cpu_loop(iterations)
    begin = time.perf_counter()
    _cpu_loop(iterations)
    elapsed = time.perf_counter() - begin
    iterations = max(iterations, int(iterations * 0.1 / max(elapsed, 1e-9)))
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter()
        _cpu_loop(iterations)
        samples.append(iterations / (time.perf_counter() - begin))
    usable = cores or os.cpu_count() or 1
    rate = statistics.median(samples)
    cv = statistics.pstdev(samples) / statistics.mean(samples)
    log.info("CPU probe iterations=%d cores=%d CV=%.3f", iterations, usable, cv)
    return Measurement(
        ComputeNode(node_id, rate * usable),
        {"single_thread": rate, "cores": float(usable), "sample_cv": cv},
        time.monotonic(),
    )


def benchmark_gpu(
    device: Any, node_id: str = "local-gpu", repeats: int = 5
) -> Measurement:
    """Measure ALU, storage bandwidth and host/device copy rates using wgpu."""
    import wgpu

    count = 262144
    byte_count = count * 4
    data = bytes(byte_count)
    buffer = device.create_buffer_with_data(
        data=data,
        usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST,
    )

    def kernel(loop: bool) -> float:
        body = (
            "var v = f32(i)*0.001; for (var j=0u;j<256u;j++) { v = fract(v*1.00001+0.123); } out[i]=v;"
            if loop
            else "out[i]=out[i]*1.01+1.0;"
        )
        shader = f"@group(0) @binding(0) var<storage, read_write> out: array<f32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ let i=gid.x; {body} }}"
        pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={
                "module": device.create_shader_module(code=shader),
                "entry_point": "main",
            },
        )
        bind = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": buffer}}],
        )

        def dispatch() -> float:
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, bind)
            for _ in range(8):
                compute.dispatch_workgroups(count // 64)
            compute.end()
            begin = time.perf_counter()
            device.queue.submit([encoder.finish()])
            device.queue.read_buffer(buffer, 0, 4)
            return time.perf_counter() - begin

        dispatch()
        return statistics.median(dispatch() for _ in range(repeats))

    try:
        alu = count * 256 * 3 * 8 / kernel(True)
        bandwidth = byte_count * 2 * 8 / kernel(False)
        transfer = []
        for _ in range(repeats):
            begin = time.perf_counter()
            device.queue.write_buffer(buffer, 0, data)
            device.queue.read_buffer(buffer)
            transfer.append(byte_count * 2 / (time.perf_counter() - begin))
        copy_rate = statistics.median(transfer)
    finally:
        buffer.destroy()
    score = alu * math.sqrt(bandwidth / 1e11 * copy_rate / 1e10)
    return Measurement(
        ComputeNode(node_id, score, device=device),
        {
            "gpu_alu": alu,
            "memory_bandwidth": bandwidth,
            "host_device_transfer": copy_rate,
        },
        time.monotonic(),
    )


class NodeRanker:
    """TTL cache; probes are injected so remote transport remains explicit."""

    def __init__(self, ttl: float = 300.0) -> None:
        if ttl < 0:
            raise ValueError("TTL must be nonnegative")
        self.ttl = ttl
        self.cache: dict[str, Measurement] = {}

    def rank(
        self,
        nodes: list[ComputeNode],
        probe: Callable[[ComputeNode], Measurement],
        refresh: bool = False,
    ) -> list[Measurement]:
        """Exclude offline nodes immediately and remeasure expired live nodes."""
        result = []
        for node in nodes:
            if not node.online:
                self.cache.pop(node.node_id, None)
                continue
            measured = self.cache.get(node.node_id)
            if (
                refresh
                or measured is None
                or time.monotonic() - measured.measured_at >= self.ttl
            ):
                measured = probe(node)
                if measured.node.node_id != node.node_id:
                    raise ValueError("Probe returned wrong node")
                self.cache[node.node_id] = measured
            result.append(measured)
        return sorted(result, key=lambda m: m.node.score, reverse=True)


def benchmark_network(
    roundtrip: Callable[[bytes], bytes],
    payload_bytes: int = 1_048_576,
    repeats: int = 5,
) -> dict[str, float]:
    """Measure explicit remote echo transport latency and bidirectional bandwidth."""
    latency = []
    bandwidth = []
    payload = bytes(payload_bytes)
    for _ in range(repeats):
        begin = time.perf_counter()
        reply = roundtrip(b"x")
        elapsed = time.perf_counter() - begin
        if reply != b"x":
            raise ValueError("Invalid latency echo")
        latency.append(elapsed)
        begin = time.perf_counter()
        reply = roundtrip(payload)
        elapsed = time.perf_counter() - begin
        if reply != payload:
            raise ValueError("Invalid bandwidth echo")
        bandwidth.append(2 * payload_bytes / max(elapsed, 1e-9))
    return {
        "network_latency": statistics.median(latency),
        "network_bandwidth": statistics.median(bandwidth),
    }
