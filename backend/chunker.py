"""Weighted legacy kernel ranges bounded by device and transport capacity.

Without compute measurements, use one node and warn instead of assuming equal
throughput. The legacy per-chunk seed contract is retained for old clients.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from uuid import uuid4
from .models import AssignChunk, Binding, Uniform, TypedArray


@dataclass
class Chunk:
    message: AssignChunk
    attempts: int = 0
    assigned: str | None = None
    deadline: float = 0
    done: bool = False
    candidates: list = field(default_factory=list)
    verifier_nodes: set = field(default_factory=set)
    planned_for: str | None = None


def capacity(limits: Any, mode: str, workgroup_size: int = 64) -> int:
    if (
        min(
            limits.max_compute_workgroup_size_x,
            limits.max_compute_invocations_per_workgroup,
        )
        < workgroup_size
    ):
        return 0
    gpu = min(limits.max_buffer_size, limits.max_storage_buffer_binding_size) // 4
    dispatch = limits.max_compute_workgroups_per_dimension * workgroup_size
    network = (
        max(1, int(limits.bandwidth_mbps * 1_000_000 / 8 / 12))
        if mode == "data-slice"
        else 262144
    )
    return min(gpu, dispatch, network, 262144)


def make_chunks(
    job_id: str, request: Any, kernel: Any, values: list | None, nodes: list[Any]
) -> list[Chunk]:
    total = len(values) if values is not None else request.count
    from .common.nodes import ComputeNode
    from .node_ranker import allocate

    eligible = [n for n in nodes if capacity(n.limits, request.mode) > 0]
    if not eligible:
        return []
    measured = [n for n in eligible if n.score is not None and n.score > 0]
    if not measured:
        logging.getLogger(__name__).warning(
            "No legacy compute scores; planning for one node instead of assuming equal throughput"
        )
        measured = [max(eligible, key=lambda n: capacity(n.limits, request.mode))]
        shares = {measured[0].node_id: total}
    else:
        ranked = [
            ComputeNode(
                n.node_id,
                n.score,
                bandwidth=n.limits.bandwidth_mbps * 1_000_000 / 8,
                bytes_per_work=8 if values is not None else 4,
            )
            for n in measured
        ]
        if total <= 4096:
            ranked = [max(ranked, key=lambda n: n.score)]
        shares = {n.node_id: count for n, count in allocate(total, ranked).items()}
    chunks = []
    partitions = []
    start = 0
    for node in measured:
        share = shares.get(node.node_id, 0)
        size = capacity(node.limits, request.mode)
        for offset in range(start, start + share, size):
            partitions.append((node.node_id, offset, min(size, start + share - offset)))
        start += share
    for owner, offset, count in partitions:
        seed = (request.seed + len(chunks)) & 0xFFFFFFFF
        parameters = dict(request.parameters, offset=offset, count=count, seed=seed)
        parameters = {u.name: parameters[u.name] for u in kernel.uniforms}
        chunks.append(
            Chunk(
                AssignChunk(
                    job_id=job_id,
                    chunk_id=uuid4().hex,
                    attempt_id="",
                    chunk_type="data_slice" if values is not None else "parameters",
                    wgsl=kernel.wgsl,
                    bindings=[Binding(**vars(b)) for b in kernel.buffers],
                    uniforms=[Uniform(**vars(u)) for u in kernel.uniforms],
                    parameters=parameters,
                    input=TypedArray.encode(
                        values[offset : offset + count], request.input.dtype
                    )
                    if values is not None
                    else None,
                    offset=offset,
                    count=count,
                    seed=seed,
                    reduce=request.reduce,
                ),
                planned_for=owner,
            )
        )
    return chunks
