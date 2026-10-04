# Distributed workloads

The FastAPI coordinator accepts animation, ONNX inference, custom WGSL, and
marked Python through **POST /pool/jobs**. All use the same `/pool/nodes`
browser contributors, pull scheduler, attempt IDs, retries, cancellation, and
ordered binary result assembly. Computation runs in WebGPU workers; the server
parses/validates submissions and places returned bytes. It does not execute
submitted Python or run ONNX inference.

The active React source is **HiveFrontend/**. `Frontend/HiveFrontend/` remains
the previous frontend. The server prefers `HiveFrontend/dist` and falls back to
the previous build when that directory is absent.

```sh
backend/.venv/bin/pip install -r requirements.txt
cd HiveFrontend
npm ci
npm run build
cd ..
backend/.venv/bin/python -m uvicorn backend.main:app --port 8000 --ws-max-size 12000000
```

Open `/` for submissions and `/node/` on each contributor. For React development,
`cd HiveFrontend && npm run dev` proxies the same APIs. Node is only required for
building the UI and installing the pinned ONNX Runtime Web assets; contributors
need only a WebGPU browser on localhost or HTTPS.

## Analyze before submitting

Analysis endpoints return `status: ready` when supported. Unsupported source
returns a report rather than a job. Syntax/schema errors in the request return
422. Analysis does not dispatch GPU work.

| Endpoint | JSON request | Result |
| --- | --- | --- |
| POST /pool/wgsl/analyze | `source`, optional `count`, `chunk_size` | Rewritten WGSL, input/output dtypes, chunk count, first 64 ranges, findings |
| POST /pool/python/analyze | `source`, optional `auto_mark`, `candidate_line` | Candidate loops, marked source, restricted Python kernel, WGSL, findings |
| POST /pool/onnx/analyze | base64 `model`, optional `input_shape`, `samples`, `batch_size` | Operators, model shapes, job shapes, batch size, chunk count, findings |

ONNX analysis defaults to 256 samples when no input shape is supplied. The React
UI detects the sample count from its flat input array after inspecting the model.
Specify a compatible shape and batch size for a model with a fixed batch axis.

## Custom WGSL

Automatic analysis accepts a complete module with one compute `main`, zero or
one read-only storage array and one writable storage array. Arrays use f32,
u32, or i32. Main accepts `@builtin(global_invocation_id)` as `vec3<u32>` and
writes exactly one output at the current X index. Expressions may use current
input values, the global index, numeric scalar literals, local `let` variables,
arithmetic, and a small set of scalar math functions. A conventional output
`arrayLength` guard is recognized. Synchronization, shared memory, textures,
neighboring input reads, dependent output reads/writes, and arbitrary helper
functions are not automatically split.

Example source:

```wgsl
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    result[i] = values[i] * 2.0 + f32(i);
}
```

Submit the **original** source; automatic rewriting happens again on submission:

```json
{
  "kind": "wgsl",
  "wgsl": "<source above>",
  "splitting": "auto",
  "input": {"dtype":"f32","data":"<base64 little-endian floats>"},
  "output_dtype": "f32",
  "chunk_size": 1024
}
```

Input length determines output count; without input, set `count`. Generated WGSL
uses binding 0 for a 16-byte uniform `{offset:u32,count:u32,seed:u32,padding:u32}`,
binding 1 for the optional input array, binding 2 for output, and workgroups of
64. Local buffer index is `gid.x`; original index values become `offset + gid.x`.
Each chunk receives sliced input and writes `count` output elements.

Other independent computations can be submitted using `splitting: declared`,
`independent: true`, and WGSL already obeying that exact binding contract.
Declare only bindings used by the shader. Use `main`, `@workgroup_size(64)`, and
a local index/count guard. Declared mode is an author assertion: the server
does not prove independence or validate WGSL syntax. The browser validates and
compiles every shader; errors retry and eventually fail the job. It cannot turn
a dependent algorithm into an independent one.

## Marked Python

Use comments to select exactly one complete supported loop:

```python
def transform(values):
    result = [0.0] * len(values)
    # hive:parallel begin
    for i in range(len(values)):
        scaled = values[i] * 2
        result[i] = scaled + 1
    # hive:parallel end
    return result
```

Or use `for value in values: result.append(value * 2 + 1)` after a fresh
`result = []`. Only iteration-local assignments and `+`, `-`, `*`, unary signs,
and finite numeric constants are translated. Indexed reads must be exactly
`values[i]`, and writes exactly `result[i]`. Input is a function argument;
output is a fresh list declared immediately before the loop. Captures, output
reads, neighbor reads, reductions, calls, in-place mutations, and control-flow
extraction require a manual rewrite. Float32 arithmetic can differ from Python.

`auto_mark: true` inserts comments when exactly one safe candidate exists. With
multiple candidates, select its starting line using `candidate_line`. Candidate
reports include line bounds and input/output names. Existing explicit markers
remain authoritative. Surrounding source is never executed, including imports
and top-level statements; only the marked region is extracted.

```json
{
  "kind":"python",
  "source":"<marked Python>",
  "input":{"dtype":"f32","data":"<base64>"},
  "chunk_size":1024
}
```

Command-line analysis:

```sh
backend/.venv/bin/python -m scripts.analyze_parallel program.py --mark
# Multiple candidates:
backend/.venv/bin/python -m scripts.analyze_parallel program.py --mark --line 12
```

This prints a JSON report containing the suggested source and kernels; it does
not overwrite the original file or execute it.

## ONNX inference

Upload an embedded ONNX protobuf as base64 `model`. Analysis uses the ONNX
checker and shape inference. The supported subset has one float32 input and
one float32 output, rank 2–4, with static sample dimensions and a static or
symbolic first batch dimension. Use IR <= 10 and standard opsets 13–21.
Operators: Identity, Relu, Sigmoid, Tanh, Add, Sub, Mul, Div, MatMul, Gemm.
Matrix operations multiply rank-2 samples by embedded constant rank-2 weights;
Gemm cannot transpose the sample axis. Broadcast weights cannot vary along the
batch axis. All intermediate tensors must preserve that axis. Nested graphs,
custom operators, local functions, training graphs, sparse weights, external
weights, cross-sample operations, and unknown shapes are rejected.

```json
{
  "kind":"onnx",
  "model":"<base64 .onnx bytes>",
  "input":{"dtype":"f32","data":"<flat row-major base64 tensor>"},
  "input_shape":[513,2],
  "batch_size":32,
  "independent":true
}
```

This creates 17 batches: 16 of 32 samples and one of 1. Static-batch models
require matching batch size and divisible sample count. Every worker downloads
and verifies the full model by hash, caches it, and runs its assigned samples
using ONNX Runtime Web's WebGPU execution provider with CPU fallback disabled.
Sessions are bounded to two cached models per worker. Results are assembled by
sample offset, regardless of completion order.

**This partitions inference inputs.** It does not shard model layers/weights,
split one dependent sample across machines, or pool GPU memory. A full model
must fit each participating device. General graph sharding would require
intermediate-tensor transfer, graph partitioning, and dependency scheduling.

## Animation

Each entry in `frames` is a Mandelbrot coordinate/iteration configuration:

```json
{
  "kind":"animation",
  "frames":[{"max_iterations":128},{"xmin":-1.5,"xmax":0.5,"max_iterations":256}],
  "fps":12,
  "distribution":"tiles"
}
```

`tiles` creates 64 independent 64×64 chunks per 512×512 frame. `frames` assigns
one complete 512×512 frame per chunk. Limit 32 frames and 1–60 fps. Frame index
is included in assignments and accepted chunk metadata. GET
`/pool/jobs/{id}/frames/{index}` returns RGBA8 bytes once that frame is complete
(409 while incomplete). The final result concatenates frames in order. The UI
previews/playbacks frames and exports individual PNGs; it does not encode video.

Generate a small dense/ReLU model and its input JSON for a first upload:

```sh
backend/.venv/bin/python -m demos.make_onnx_example /tmp/hive-onnx-example
```

Upload the generated `dense.onnx` and paste `input.json` into the UI. Use input
shape `65, 2`, batch size 32; the output shape is `65, 3`.

## Results, bounds, and compatibility

GET `/pool/jobs/{id}` reports kind, progress, chunk metadata (`tiles` retains
its historical field name), contributors, retry count, output format/shape,
frame count, fps, error, and result URL. GET `/result` returns raw little-endian
f32/u32/i32 arrays or frame-major RGBA8. Tensor output shapes describe row-major
assembly; downloads use `.f32.bin`, `.u32.bin`, or `.i32.bin`. Headers include
`X-Pixel-Format` and `X-Output-Shape`.

Assignments now include kinds `image_tile`, `compute`, and `onnx_batch`.
Result frames retain the uint32 header-length + JSON header + binary payload
layout; `output_format` must match the job. Results must match the current
worker/attempt and exact expected byte length; non-finite floats are rejected.
The application frame limit is 1,053,000 bytes, allowing full animation frames
and up to 1 MiB ONNX tensors. Existing M1 messages/default submissions remain
supported. Upgrade contributor pages together with the backend.

Compute limits: 2 million elements, chunk sizes 1–4096, at most 2048 chunks;
Python/WGSL source 32,000 characters. ONNX: embedded model <= 4 MiB, 128
nodes/initializers, input <= 500,000 elements, output <= 2 million elements,
batch size 1–256, every batch/intermediate tensor <= 1 MiB. The HTTP body limit
is still 12 MB; combinations exceeding it are rejected. Sixteen retained jobs,
a 128 MiB output/asset budget, and 30-minute terminal-job expiry remain in memory.
Leases are 15 seconds for GPU shaders and 60 seconds for ONNX (including model
preparation). Four failed attempts fail a chunk's job. GPU interruption is best
effort; unsupported/device-specific models can still fail browser compilation.

The legacy `/jobs`, `/kernels`, `/nodes`, and `/legacy-node/` API remains
unchanged. New marked Python jobs use `/pool/jobs` and the standard `/node/`.

## Validation

```sh
backend/.venv/bin/python -m pytest -q
cd HiveFrontend
npm run build
npm run lint
cd ..
# Server must already be running; requires playwright and installed Chrome:
HIVE_UI_URL=http://localhost:8000 backend/.venv/bin/python -m demos.workloads_browser_check
```

The Chrome test checks custom WGSL global offsets, marked Python output,
frame/tile animation assembly, ONNX on two isolated GPU workers, a short final
batch, cached repeat inference, React upload/analysis/submission workflows, downloads,
animation playback, and mobile layout. A test-only 100 ms delay keeps work
observable long enough for both workers to participate. These are two browser workers on one
computer; physical multi-computer throughput testing remains a deployment check.
