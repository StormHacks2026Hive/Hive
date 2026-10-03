"""Wire schema: all arrays are little-endian base64 typed arrays."""
import base64
import math
import struct
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_ELEMENTS = 2_000_000
MAX_FRAME = 12_000_000
Scalar = Literal['f32', 'u32', 'i32']
Reduce = Literal['sum', 'count', 'min', 'max', 'mean']

class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class TypedArray(Model):
    dtype: Scalar = 'f32'
    data: str = Field(max_length=10_666_668)

    def decode(self):
        raw = base64.b64decode(self.data, validate=True)
        if len(raw) % 4 or len(raw) > MAX_ELEMENTS * 4:
            raise ValueError('Invalid or oversized typed array')
        values = [v[0] for v in struct.iter_unpack('<' + {'f32':'f','u32':'I','i32':'i'}[self.dtype], raw)]
        if any(not math.isfinite(v) for v in values):
            raise ValueError('Array values must be finite')
        return values

    @classmethod
    def encode(cls, values, dtype='f32'):
        fmt = '<' + {'f32':'f','u32':'I','i32':'i'}[dtype]
        return cls(dtype=dtype, data=base64.b64encode(b''.join(struct.pack(fmt, v) for v in values)).decode())

class JobRequest(Model):
    kernel: str = Field(min_length=1, max_length=32_000)
    mode: Literal['data-slice','parameter-only']
    reduce: Reduce | None = None
    input: TypedArray | None = None
    count: int = Field(default=1, ge=1, le=MAX_ELEMENTS)
    parameters: dict[str, int | float] = Field(default_factory=dict, max_length=32)
    seed: int = Field(default=1, ge=0, le=0xffffffff)
    verify: bool = False

    @model_validator(mode='after')
    def check_mode(self):
        if (self.mode == 'data-slice') != (self.input is not None):
            raise ValueError('data-slice requires input; parameter-only forbids input')
        return self

class Limits(Model):
    max_buffer_size: int = Field(ge=4, le=2**53)
    max_storage_buffer_binding_size: int = Field(ge=4, le=2**53)
    max_compute_workgroup_size_x: int = Field(ge=1, le=65535)
    max_compute_invocations_per_workgroup: int = Field(ge=1, le=65535)
    max_compute_workgroups_per_dimension: int = Field(ge=1, le=65535)
    bandwidth_mbps: float = Field(default=20, gt=0, le=100000)

class Register(Model):
    type: Literal['register']
    webgpu: bool
    limits: Limits

class Registered(Model):
    type: Literal['registered'] = 'registered'
    node_id: str

class Heartbeat(Model):
    type: Literal['heartbeat']

class Binding(Model):
    name: str
    element_type: Scalar
    access: Literal['read','read_write']
    binding: int

class Uniform(Model):
    name: str
    type: Scalar

class AssignChunk(Model):
    type: Literal['assign_chunk'] = 'assign_chunk'
    job_id: str
    chunk_id: str
    attempt_id: str
    chunk_type: Literal['data_slice','parameters']
    wgsl: str
    bindings: list[Binding]
    uniforms: list[Uniform]
    parameters: dict[str, int | float]
    input: TypedArray | None = None
    offset: int
    count: int
    seed: int
    workgroup_size: int = 64
    timeout_ms: int = 15000
    reduce: Reduce | None = None

class Summary(Model):
    value: float | int
    count: int = Field(ge=0, le=MAX_ELEMENTS)

class ChunkResult(Model):
    type: Literal['chunk_result']
    chunk_id: str
    attempt_id: str
    output: TypedArray | None = None
    summary: Summary | None = None

    @model_validator(mode='after')
    def one_result(self):
        if (self.output is None) == (self.summary is None):
            raise ValueError('Return exactly one of output or summary')
        return self

class ChunkError(Model):
    type: Literal['chunk_error']
    chunk_id: str
    attempt_id: str
    error: str = Field(max_length=2000)
    device_lost: bool = False

class JobCreated(Model):
    job_id: str

class JobStatus(Model):
    job_id: str
    status: Literal['queued','running','done','failed']
    progress: float
    completed_chunks: int
    total_chunks: int
    result: TypedArray | float | int | None = None
    error: str | None = None
