from typing import Annotated, Literal, Union
from pydantic import Field, TypeAdapter, model_validator
from ..models import Model

class Parameters(Model):
    xmin: float = Field(default=-2, ge=-1000000, le=1000000)
    xmax: float = Field(default=1, ge=-1000000, le=1000000)
    ymin: float = Field(default=-1.5, ge=-1000000, le=1000000)
    ymax: float = Field(default=1.5, ge=-1000000, le=1000000)
    max_iterations: int = Field(default=256, ge=1, le=1024)

    @model_validator(mode='after')
    def ordered(self):
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError('Coordinate minima must be smaller than maxima')
        return self

class JobRequest(Model):
    kind: Literal['mandelbrot'] = 'mandelbrot'
    width: Literal[512] = 512
    height: Literal[512] = 512
    tile_size: Literal[64] = 64
    parameters: Parameters = Field(default_factory=Parameters)

class Manifest(Model):
    shader_id: str
    width: int = 512
    height: int = 512
    tile_size: int = 64
    workgroup_size: list[int] = [8, 8]

class Limits(Model):
    maxBufferSize: int = Field(ge=4, le=2**53)
    maxStorageBufferBindingSize: int = Field(ge=4, le=2**53)
    maxUniformBufferBindingSize: int = Field(ge=16, le=2**53)
    maxComputeWorkgroupSizeX: int = Field(ge=1, le=65535)
    maxComputeWorkgroupSizeY: int = Field(ge=1, le=65535)
    maxComputeInvocationsPerWorkgroup: int = Field(ge=1, le=65535)
    maxComputeWorkgroupsPerDimension: int = Field(ge=1, le=65535)

class Adapter(Model):
    vendor: str = Field(default='', max_length=256)
    architecture: str = Field(default='', max_length=256)
    description: str = Field(default='', max_length=256)
    device: str = Field(default='', max_length=256)

class Benchmark(Model):
    version: Literal['mandelbrot-v1'] = 'mandelbrot-v1'
    pixels: Literal[4096] = 4096
    elapsed_ms: float = Field(gt=0, le=120000)

class Capabilities(Model):
    webgpu: Literal[True]
    adapter: Adapter
    limits: Limits
    features: list[Literal['shader-f16']] = Field(default_factory=list, max_length=1)
    benchmark: Benchmark

class Wire(Model):
    v: Literal[1] = 1

class Register(Wire):
    type: Literal['register']
    label: str = Field(min_length=1, max_length=80)
    capabilities: Capabilities

class Registered(Wire):
    type: Literal['registered'] = 'registered'
    worker_id: str
    heartbeat_ms: int = 5000

class Heartbeat(Wire):
    type: Literal['heartbeat']
    visible: bool
    attempt_id: str | None = None

class RequestChunk(Wire):
    type: Literal['request_chunk']

class Control(Wire):
    type: Literal['pause', 'resume', 'stop']

class ChunkStarted(Wire):
    type: Literal['chunk_started']
    chunk_id: str
    attempt_id: str

class ChunkError(Wire):
    type: Literal['chunk_error']
    chunk_id: str
    attempt_id: str
    code: Literal['validation', 'device_lost', 'timeout', 'execution']
    error: str = Field(max_length=2000)

class Tile(Model):
    x: int
    y: int
    width: int = 64
    height: int = 64

class Image(Model):
    width: int
    height: int

class Assignment(Wire):
    type: Literal['assign_chunk'] = 'assign_chunk'
    job_id: str
    chunk_id: str
    attempt_id: str
    kind: Literal['image_tile'] = 'image_tile'
    shader_id: str
    tile: Tile
    image: Image
    parameters: Parameters
    output_format: Literal['rgba8'] = 'rgba8'
    timeout_ms: int

class ResultHeader(Wire):
    type: Literal['chunk_result']
    job_id: str
    chunk_id: str
    attempt_id: str
    output_format: Literal['rgba8']
    byte_length: int = Field(ge=1, le=16384)
    elapsed_ms: float = Field(gt=0, le=120000)

class ResultAck(Wire):
    type: Literal['result_ack'] = 'result_ack'
    chunk_id: str
    attempt_id: str
    disposition: Literal['accepted', 'stale', 'duplicate']

class NoWork(Wire):
    type: Literal['no_work'] = 'no_work'
    retry_after_ms: int = 1000
    reason: str = 'No compatible work available'

class CancelAttempt(Wire):
    type: Literal['cancel_attempt'] = 'cancel_attempt'
    attempt_id: str
    reason: str

class Subscribe(Wire):
    type: Literal['subscribe']
    job_id: str

class WorkerStatus(Model):
    worker_id: str
    label: str
    state: Literal['idle', 'working', 'paused']
    visible: bool
    capabilities: Capabilities
    completed_chunks: int
    compute_ms: float
    pixels_per_second: float

class AcceptedTile(Model):
    chunk_id: str
    tile: Tile
    url: str
    worker_id: str
    worker_label: str
    elapsed_ms: float

class Contribution(Model):
    worker_id: str
    label: str
    chunks: int
    elapsed_ms: float

class JobStatus(Model):
    job_id: str
    status: Literal['queued', 'running', 'done', 'failed', 'cancelled']
    progress: float
    completed_chunks: int
    total_chunks: int
    width: int
    height: int
    parameters: Parameters
    tiles: list[AcceptedTile]
    contributions: list[Contribution]
    retries: int
    error: str | None
    result_url: str | None

class Snapshot(Wire):
    type: Literal['job_snapshot', 'job_update', 'job_done', 'job_failed']
    job: JobStatus
    workers: list[WorkerStatus]

class TileReady(Wire):
    type: Literal['tile_ready'] = 'tile_ready'
    job_id: str
    tile: AcceptedTile

Incoming = TypeAdapter(Annotated[Union[Register, Heartbeat, RequestChunk, Control, ChunkStarted, ChunkError], Field(discriminator='type')])
