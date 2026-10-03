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


def capacity(limits, mode, workgroup_size=64):
    if min(limits.max_compute_workgroup_size_x, limits.max_compute_invocations_per_workgroup) < workgroup_size:
        return 0
    gpu = min(limits.max_buffer_size, limits.max_storage_buffer_binding_size) // 4
    dispatch = limits.max_compute_workgroups_per_dimension * workgroup_size
    network = max(1, int(limits.bandwidth_mbps * 1_000_000 / 8 / 12)) if mode == 'data-slice' else 262144
    return min(gpu, dispatch, network, 262144)


def make_chunks(job_id, request, kernel, values, nodes):
    total = len(values) if values is not None else request.count
    caps = [capacity(n.limits, request.mode) for n in nodes]
    caps = [c for c in caps if c]
    if not caps:
        return []
    size = min(max(caps), max(1, (total + len(caps) - 1) // len(caps)))
    if total <= 4096:
        size = min(total, max(caps))
    chunks = []
    for offset in range(0, total, size):
        count = min(size, total - offset)
        seed = (request.seed + len(chunks)) & 0xffffffff
        parameters = dict(request.parameters, offset=offset, count=count, seed=seed)
        parameters = {u.name: parameters[u.name] for u in kernel.uniforms}
        chunks.append(Chunk(AssignChunk(job_id=job_id, chunk_id=uuid4().hex, attempt_id='',
            chunk_type='data_slice' if values is not None else 'parameters', wgsl=kernel.wgsl,
            bindings=[Binding(**vars(b)) for b in kernel.buffers], uniforms=[Uniform(**vars(u)) for u in kernel.uniforms],
            parameters=parameters, input=TypedArray.encode(values[offset:offset+count], request.input.dtype) if values is not None else None,
            offset=offset, count=count, seed=seed, reduce=request.reduce)))
    return chunks
