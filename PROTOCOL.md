# Hive protocol v1

Run from the repository root (Python 3.11+):

```sh
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r requirements.txt
backend/.venv/bin/python -m py2wgsl.demo
backend/.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --ws-max-size 12000000
```

Open **http://localhost:8000/node/** in two or more WebGPU-enabled browser tabs.
Use localhost or HTTPS; remote HTTP pages cannot use WebGPU. `/docs` has the
HTTP schemas. Set `CORS_ORIGINS` to a comma-separated origin list; defaults
allow the Vite development frontend on localhost and 127.0.0.1 port 5173.
Use one uvicorn worker: jobs and nodes share an in-memory store.

## Existing frontend

`Frontend/HiveFrontend/src/App.jsx` provides Python uploads, source analysis,
editable kernel previews, validation, configuration, and submission. It POSTs
the schemas below, retains `job_id`, and polls GET until done or failed. Vite
proxies the backend routes in development. The node page is provided separately
at `/node/`. See the Python upload section below for setup and analysis messages.

## Kernel contract

Source contains exactly one Python function, without imports, decorators,
default arguments, or top-level statements. Names like `Array`, `u32`, `f32`,
`global_id`, and `array_length` are recognized by py2wgsl directly; they do not
need Python imports. The server calls **compile_kernel_source**, never exec,
eval, import of submitted source, or compile_kernel on a submitted callable.
The library README bundled in wheel metadata and `python -m py2wgsl.demo` were
read/run before implementing the binding layout.

The MVP supports exactly one writable output array and zero or one read-only
input array, with f32/u32/i32 elements. Declare `offset: u32` and `count: u32`;
parameter-only kernels also declare `seed: u32`. Other scalar arguments are
provided through `parameters`. These reserved names cannot be user constants.
Use `i = global_id()` as a **local** buffer index; use `offset + i` for the
full-job coordinate. Guard accesses with `if i < count`. Each thread must write
its own output element and never depend on another thread or chunk. Cross-thread
algorithms are outside this contract; arbitrary WGSL dependencies cannot be
proven safe by the server.

py2wgsl supports arithmetic, conditionals, casts and built-in math, but not
arbitrary Python libraries or user function calls. This service narrows loops
to literal `range` bounds with at most 1024 combined iterations per element;
`while` and dynamic loop bounds are rejected. Integers follow WGSL 32-bit wrap
semantics. See the py2wgsl README: https://github.com/warppool/py2wgsl.

## Typed arrays

Every array is an object `{"dtype":"f32","data":"<base64>"}`. `dtype` is
`f32`, `u32`, or `i32`; data encodes tightly packed **little-endian 32-bit**
elements. No JSON number lists are accepted or emitted for arrays. Base64 is
used consistently for HTTP and WebSocket text frames; binary frames are not
part of v1. Empty input arrays, malformed base64, non-finite values, and
arrays exceeding 2,000,000 elements are rejected.

## HTTP

### POST /jobs

```json
{
  "kernel": "def kernel(offset: u32, count: u32, seed: u32, out: Array[u32]):\n    i = global_id()\n    if i < count:\n        out[i] = offset + i + seed\n",
  "mode": "parameter-only",
  "count": 10000,
  "parameters": {},
  "seed": 1,
  "reduce": "sum",
  "verify": false
}
```

`mode`: `data-slice` or `parameter-only`. For data-slice, `input` is required
and its decoded length determines the total size; `count` is ignored. For
parameter-only, `input` must be absent/null and `count` determines the total
size (default 1). `parameters` defaults to `{}`, `seed` defaults to 1 and must
fit u32. `reduce` defaults to null and accepts sum/count/min/max/mean. `verify`
defaults to false. Extra fields are rejected.

Returns **202** `{"job_id":"<uuid>"}`. **422** means invalid source,
unsupported feature, invalid bindings, dtype, parameters, or work limits;
`detail` explains the problem. **413** means request body exceeds 12 MB.
**429** means the 32 retained job capacity is full.

### GET /jobs/{job_id}

```json
{
  "job_id": "<uuid>",
  "status": "running",
  "progress": 0.5,
  "completed_chunks": 2,
  "total_chunks": 4,
  "result": null,
  "error": null
}
```

Status is queued/running/done/failed. Progress is accepted completed chunks /
total chunks; zero before chunk planning. Done result is a typed array when
reduce is null, or a scalar when reduced. Failed has an error. **404** for an
unknown ID. Jobs wait queued until a compatible node registers; verification
requires two distinct node registrations per chunk. Results persist until
server restart; there is no authentication or automatic eviction in this MVP.

## WebSocket /nodes

Use `ws://` on localhost HTTP, `wss://` on HTTPS. Each connection represents
one node. JSON text frames only, all modeled by Pydantic; extra fields rejected.
First message must be register, within 35 seconds.

### Node → server: register

```json
{
  "type": "register",
  "webgpu": true,
  "limits": {
    "max_buffer_size": 268435456,
    "max_storage_buffer_binding_size": 134217728,
    "max_compute_workgroup_size_x": 256,
    "max_compute_invocations_per_workgroup": 256,
    "max_compute_workgroups_per_dimension": 65535,
    "bandwidth_mbps": 20
  }
}
```

Limits should describe the **requested device**, which can have lower limits
than the adapter. Bandwidth is a positive hint in megabits/second, default 20;
v1 does not measure bandwidth automatically. WebGPU false is rejected. Devices
unable to run workgroups of 64 remain registered but receive no work.

### Server → node: registered

`{"type":"registered","node_id":"<uuid>"}`

### Node → server: heartbeat

`{"type":"heartbeat"}` every 5 seconds. Server evicts after 30 seconds
without heartbeat. Work is reassigned on eviction/disconnect.

### Server → node: assign_chunk

```json
{
  "type": "assign_chunk",
  "job_id": "<uuid>",
  "chunk_id": "<uuid>",
  "attempt_id": "<uuid>",
  "chunk_type": "parameters",
  "wgsl": "<complete compute module>",
  "bindings": [{"name":"out","element_type":"u32","access":"read_write","binding":1}],
  "uniforms": [{"name":"offset","type":"u32"},{"name":"count","type":"u32"},{"name":"seed","type":"u32"}],
  "parameters": {"offset":0,"count":10000,"seed":1},
  "input": null,
  "offset": 0,
  "count": 10000,
  "seed": 1,
  "workgroup_size": 64,
  "timeout_ms": 15000,
  "reduce": "sum"
}
```

Chunk type is `data_slice` (input typed array present) or `parameters` (input
null). Range is `[offset, offset + count)`. Seeds are job seed + chunk ordinal
modulo 2^32, distinct across chunks; retries retain identical seed and range.
Kernel authors must use the seed/range for independent random work. Statistical
PRNG stream independence is an algorithm property, not guaranteed by splitting.

Uniform binding 0 is padded to a multiple of 16 bytes, at least 16; scalar
fields are packed in listed order, four bytes each, according to type. Storage
bindings use the listed binding indices, one element per invocation. Entry
point is `main`. Dispatch ceil(count / 64) workgroups. A node runs one chunk at
a time, reusing the device and destroying per-chunk buffers after readback.
Timeout destroys its device to interrupt stuck work, then reconnect creates a
new one. Shader compilation/validation errors are reported.

### Node → server: chunk_result

Full output:

```json
{"type":"chunk_result","chunk_id":"<uuid>","attempt_id":"<uuid>","output":{"dtype":"f32","data":"<base64>"}}
```

Reduced output:

```json
{"type":"chunk_result","chunk_id":"<uuid>","attempt_id":"<uuid>","summary":{"value":7821,"count":10000}}
```

Exactly one of output/summary must be present. For sum, value is local sum;
count is the number of output elements. Count reduction counts **nonzero output
elements**, so kernels return 0/1 for predicate counting. Min/max send the local
extremum. Mean sends a **sum and count**, and the server divides total sum by
total count. Output lengths, types and summary counts are validated. Attempts
must match the current chunk lease and sending node; stale replies are ignored.

### Node → server: chunk_error

```json
{"type":"chunk_error","chunk_id":"<uuid>","attempt_id":"<uuid>","error":"GPU validation failed","device_lost":false}
```

Error text is capped at 2000 characters. device_lost defaults false; true
removes the node. Errors release chunks for retry; server lease expiration
also removes/closes the node so it cannot accept new work while old work runs.
Four total dispatch attempts per chunk, including verification attempts.

Invalid messages close with **1008**, oversized messages **1009**, and server
lease/heartbeat expiry **1013**. Reconnect uses exponential backoff with jitter
up to roughly 30 seconds and a fresh node ID/device.

## Scheduling and aggregation

Chunks are planned when nodes become available, bounded by buffer size,
workgroup dispatch limits, a transfer budget based on bandwidth, and 262144
elements. Initial balancing divides work among available capable nodes; 4096
or fewer elements uses one chunk if device limits allow it. This is a simple
size proxy for low compute per byte. Smaller compatible nodes can take only
chunks that fit; v1 does not dynamically subdivide planned chunks.

Reassembly sorts chunks by offset. Reduce summaries are merged without sending
full arrays back. Optional verification runs the same chunk on two distinct
node registrations before accepting it, comparing float values at relative
and absolute tolerance 1e-5. A mismatch fails the job. This is a consistency
check, not proof against malicious nodes or duplicate physical devices.

## Demos and tests

```sh
backend/.venv/bin/python -m demos.monte_carlo
backend/.venv/bin/python -m demos.elementwise
backend/.venv/bin/python -m pytest -q
```

Monte Carlo assigns one random point per output invocation, receives only hit
counts, merges with sum, and checks pi within 0.02. Elementwise sends 1,000,000
floats, uses the global offset in its formula, reassembles outputs, and checks
every value against a plain Python reference. Open 2+ tabs before submitting.

Optional browser automation (uses installed Google Chrome):

```sh
backend/.venv/bin/pip install playwright
backend/.venv/bin/python -m demos.browser_check
```

## Bounds and next improvements

Source limit 32000 characters/4000 AST nodes; arrays/count up to 2 million;
combined loops up to 1024 per element and 20 million iterations per job;
12 MB HTTP/WS payloads; 15-second chunk leases and node timeout; 32 retained
jobs. Start uvicorn with the documented WS frame limit: application validation
alone occurs after the transport has received a frame.

Before public deployment, add authentication, submission rate limits, memory
accounting and job eviction, ownership-based result access, and node admission.
Then measure bandwidth and compute cost, adapt/re-split chunks after capability
changes, add GPU reductions for heavier jobs, and consider persistent storage
and binary frames. Persistence/multiple server workers would be architectural
changes and have not been introduced. The current server never executes
submitted kernels; all GPU execution stays inside browser WebGPU.

## Python upload and conversion preview

Start the frontend alongside the backend:

```sh
cd Frontend/HiveFrontend
npm ci
npm run dev
```

Open the URL printed by Vite (normally http://localhost:5173). Vite proxies
`/kernels`, `/jobs`, `/node`, and WebSocket `/nodes` to the backend on port 8000.
Set `HIVE_API_URL` when launching Vite to use a different backend URL. The
frontend production build needs equivalent proxy routes on its host.
The UI includes file upload, source editing, findings, editable kernel preview,
explicit validation, job configuration, submission, progress polling, typed
array result downloads, and a Mandelbrot canvas preview. The input JSON array
is encoded as a base64 typed array before POST /jobs.

### POST /kernels/analyze

Request: `{"source":"<Python source, 1–32000 characters>"}`. This endpoint
parses source and optionally translates a recognized pattern; it never imports
or executes the uploaded file.

Response (HTTP 200, including unsupported code and syntax-error reports):

```json
{
  "status": "conversion_available",
  "findings": [
    {"severity":"info","message":"Converted a one-input append loop with independent arithmetic into an elementwise kernel.","line":null},
    {"severity":"warning","message":"The proposed kernel uses float32 arithmetic, which can differ from Python integers and float64. Review it and provide the input array separately.","line":null}
  ],
  "kernel": "<proposed kernel source>",
  "mode": "data-slice",
  "parameters": {}
}
```

Status is `compatible` (already a compilable service kernel),
`conversion_available` (a proposed conversion), or `manual_conversion_required`
(no generated kernel, kernel/mode null). Findings have severity info/warning/error,
message, and optional one-based line number. Invalid request fields/lengths
return 422. Source is limited to 4000 AST nodes. Reports include at most 40
individual incompatibility findings plus a summary.

Automatic conversion recognizes **only** a complete single function with one
argument and exactly three statements: an empty result list, a for-each loop
that appends an expression, and return of that list. The expression may use
its loop variable, finite numeric literals, unary +/- and binary +, -, *.
No captured variables, library calls, divisions, additional statements,
decorators, default/variadic arguments, or partial-function extraction are
converted. All inputs become f32. Compilation is not a proof of Python
semantic equivalence; the UI asks the user to review precision differences.

### POST /kernels/validate

Request: `{"source":"<edited kernel source>"}`.

Response: `{"valid":true,"error":null,"bindings":[...],"uniforms":[...],"mode":"data-slice"}`
using the same binding/uniform schemas as assign_chunk. An unsupported kernel
returns HTTP 200 with valid false, a readable error, empty metadata, and mode
null. This validates compilation and reserved argument declarations; /jobs
still validates actual constants, input buffers and the total work budget.
Validation never creates a job or dispatches work. Editing the kernel invalidates
the frontend's prior validation.

The Mandelbrot button loads a **hand-authored example**, explicitly labeled as
such rather than claiming to convert arbitrary Mandelbrot programs. It generates
256×256 pixels, up to 256 iterations each (within the 20 million job budget),
uses no input arrays, and returns u32 escape iteration counts. Width × height
must equal output count for image rendering. Larger requests must still fit
the existing work budget. Python complex numbers need manual scalar rewriting.

Optional UI integration check (backend and Vite must already be running):

```sh
backend/.venv/bin/pip install playwright
backend/.venv/bin/python -m demos.upload_ui_check
```

Set `HIVE_UI_URL` to use a different Vite URL. This opens two Chrome GPU nodes,
uploads supported/unsupported files, checks conversion and validation invalidation,
executes the generated elementwise kernel, renders Mandelbrot, checks sampled
pixels against scalar Python, and checks mobile overflow.
