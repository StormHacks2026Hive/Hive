# Milestone 1: Python-coordinated browser GPU pool

This is the historical M1 contract. The pool now also supports animation, ONNX
batches, custom WGSL, and marked Python; see [WORKLOADS.md](WORKLOADS.md) for the
current extensions, updated bounds, and active `HiveFrontend/` source. The coordinator remains FastAPI/asyncio/Pydantic.
It schedules work and copies output bytes; fractal computation runs on WebGPU.

## Run one server and share one HTTPS origin

Python 3.11+; Node 22.12+ is needed only for the existing React build tools,
not for the coordinator or volunteer computers.

```sh
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r requirements.txt
cd Frontend/HiveFrontend
npm ci
npm run build
cd ../..
backend/.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --ws-max-size 12000000
```

Build before starting/restarting uvicorn. When the build exists, FastAPI serves
it at `/`; the same server serves `/node/`, `/shared/`, and all APIs. Open
http://localhost:8000 for the submitter and http://localhost:8000/node/ for a
local contributor. Click **Start contributing** on each worker page.

For another computer, keep the server running and start an HTTPS tunnel:

```sh
cloudflared tunnel --url http://localhost:8000
```

Open `https://<printed-host>.trycloudflare.com/node/` on the other computer,
name that computer, and click Start contributing. Keep each worker visible.
The submitter can remain on localhost or use the tunnel URL. No installs are
required on contributors. HTTPS is required for remote WebGPU.

For frontend development, run `npm run dev` in Frontend/HiveFrontend; Vite
proxies `/pool`, `/shared`, `/node` and the legacy endpoints to port 8000.
Set `HIVE_API_URL` when starting Vite to change the backend destination.

## API compatibility

The M1 protocol is namespaced under `/pool/`, preserving the old `/jobs`,
`/nodes`, and `/kernels` Python-kernel prototype. Its worker page is now
`/legacy-node/`, accessed from the UI's Python kernel prototype tab. A new
`/node/` page cannot process legacy jobs, and a legacy page cannot process pool
jobs. Old Python demos use the legacy API and require legacy worker pages.
The unfinished CUDA conversion experiment was set aside, not incorporated.

## HTTP contract

All configuration/status bodies use Pydantic models with extra fields rejected.

| Route | Behavior |
| --- | --- |
| GET /pool/manifest | Built-in shader hash, image size, tile size, workgroup size |
| GET /pool/assets/{shader_id} | Immutable WGSL bytes, content-hash ID, cache headers |
| POST /pool/jobs | Create a bounded Mandelbrot image job; 202 with job_id |
| GET /pool/jobs/{job_id} | Status, tile manifest, contributions and retries |
| GET /pool/jobs/{job_id}/chunks/{chunk_id} | Accepted tile as tightly packed RGBA8 bytes |
| GET /pool/jobs/{job_id}/result | Assembled RGBA8 bytes; 409 until done |
| POST /pool/jobs/{job_id}/cancel | Cancel job and close affected worker connections |
| GET /pool/workers | Current worker capabilities, states and measured throughput |

Example job request:

```json
{
  "kind": "mandelbrot",
  "width": 512,
  "height": 512,
  "tile_size": 64,
  "parameters": {
    "xmin": -2,
    "xmax": 1,
    "ymin": -1.5,
    "ymax": 1.5,
    "max_iterations": 256
  }
}
```

All fields have the displayed defaults. M1 fixes size to 512×512 and tiles to
64×64, exactly 64 chunks. max_iterations is 1–1024; coordinate bounds must be
ordered finite numbers between -1,000,000 and 1,000,000. Larger images, custom
shaders and arbitrary Python are not accepted by this endpoint. The worst-case
budget is fixed at 262,144 pixels × 1024 iterations. The shader also has a
literal 1024-step loop bound. Broad views are preferable to precision-sensitive
f32 deep zooms.

Unknown/expired jobs or missing tiles return 404, invalid configuration 422,
and full job capacity 429. Maximum 16 retained pool jobs; terminal jobs expire
after 30 minutes when no submitter is subscribed. The old 32-job limit and
20-million loop budget apply only to the legacy Python API.

Job status fields: job_id, status (queued/running/done/failed/cancelled),
progress (accepted / 64), completed_chunks, total_chunks, width, height,
parameters, tiles, contributions, retries, error, result_url. An accepted tile
contains chunk_id, tile{x,y,width,height}, url, worker_id, worker_label and
elapsed_ms. Contributions retain worker_id, label, chunks and elapsed_ms even
when the worker disconnects. result_url is populated only on completion.

## WebSocket /pool/nodes

All text messages contain `v: 1` and `type`. One connection is one worker
registration, with at most one active attempt. Worker execution and the socket
live inside a dedicated Web Worker. The page handles controls and Wake Lock.

First message, within 30 seconds:

```json
{
  "v": 1,
  "type": "register",
  "label": "Teammate laptop",
  "capabilities": {
    "webgpu": true,
    "adapter": {"vendor":"apple","architecture":"","description":"","device":""},
    "features": [],
    "limits": {
      "maxBufferSize": 268435456,
      "maxStorageBufferBindingSize": 134217728,
      "maxUniformBufferBindingSize": 65536,
      "maxComputeWorkgroupSizeX": 256,
      "maxComputeWorkgroupSizeY": 256,
      "maxComputeInvocationsPerWorkgroup": 256,
      "maxComputeWorkgroupsPerDimension": 65535
    },
    "benchmark": {"version":"mandelbrot-v1","pixels":4096,"elapsed_ms":3.5}
  }
}
```

Adapter strings may be empty. `shader-f16` is the only reported optional feature
in M1; the shader itself uses f32. The worker requests powerPreference
high-performance, relevant maximum adapter capacities and shader-f16 when
available, then reports actual device limits. Maximum limits are capacities,
not a guarantee that buffers of those sizes can be allocated. The benchmark
warms the pipeline, then times one 64×64 tile through readback. Joining therefore
performs a small GPU workload. This is end-to-end latency, not GPU timestamp
query timing.

Server response:
`{"v":1,"type":"registered","worker_id":"<uuid>","heartbeat_ms":5000}`.

Worker control messages:

```json
{"v":1,"type":"heartbeat","visible":true,"attempt_id":null}
{"v":1,"type":"request_chunk"}
{"v":1,"type":"pause"}
{"v":1,"type":"resume"}
{"v":1,"type":"stop"}
```

Hidden pages pause new requests; an already-running tile may finish. Wake Lock
is best effort, acquired by the visible page and released on Stop or hiding.
Stop closes the socket/device and terminates the Web Worker. Reconnect uses
exponential backoff with jitter, capped around 30 seconds, a fresh device and
registration. WebGPU absence produces a clear message and prevents joining.

Assignments:

```json
{
  "v":1,
  "type":"assign_chunk",
  "job_id":"<uuid>",
  "chunk_id":"tile-00-00",
  "attempt_id":"<uuid>",
  "kind":"image_tile",
  "shader_id":"<sha256 hex>",
  "tile":{"x":0,"y":0,"width":64,"height":64},
  "image":{"width":512,"height":512},
  "parameters":{"xmin":-2,"xmax":1,"ymin":-1.5,"ymax":1.5,"max_iterations":256},
  "output_format":"rgba8",
  "timeout_ms":15000
}
```

No work response:
`{"v":1,"type":"no_work","retry_after_ms":1000,"reason":"..."}`.
Workers pull again after acceptance, or after this delay. Faster workers
naturally process more tiles. M1 fixes tile size; it does not yet adapt tile
sizes or estimate speedup. Shader bytes are fetched once and cached by hash in
the Cache API where available; pipelines are cached for the device lifetime.
Cache failure does not block execution. M1 has no uploaded model/input arrays.

Worker start/error messages:

```json
{"v":1,"type":"chunk_started","chunk_id":"tile-00-00","attempt_id":"<uuid>"}
{"v":1,"type":"chunk_error","chunk_id":"tile-00-00","attempt_id":"<uuid>","code":"execution","error":"..."}
```

Error code: validation/device_lost/timeout/execution; text maximum 2000 characters.
chunk_started does not extend the lease. The small shader preparation is included
in the lease. Any execution failure resets the worker environment; device loss
and timeout also cause disconnect/reassignment. Cancellation closes affected
connections, which destroys their devices before automatic reconnect.

### Binary results

Exactly one complete WebSocket binary message:

```text
uint32 little-endian JSON-header byte length
UTF-8 JSON header
RGBA8 bytes, row-major within the tile
```

Header:

```json
{"v":1,"type":"chunk_result","job_id":"<uuid>","chunk_id":"tile-00-00","attempt_id":"<uuid>","output_format":"rgba8","byte_length":16384,"elapsed_ms":3.5}
```

Header length ≤4096; pool application frame limit 32768 bytes; payload is exactly
64×64×4 = 16384 bytes. No JSON number lists or base64 output. Control messages
are also limited to 32768 bytes. The uvicorn transport limit remains 12 MB for
compatibility with legacy array frames; the pool enforces its smaller limit.

Acknowledgement:
`{"v":1,"type":"result_ack","chunk_id":"...","attempt_id":"...","disposition":"accepted"}`.
Disposition can be accepted/duplicate/stale. Accepted tiles are counted once.
Worker, job, chunk and attempt must match the live lease; an expired result is
not accepted even if the timeout sweeper has not run yet. Stale attempts cannot
release another worker's current assignment.

### Timeouts and retries

15-second lease from assignment, 5-second heartbeat, eviction after 20 seconds
without heartbeat. Disconnect/Stop/timeout releases unfinished work. Four total
attempts per tile; exhaustion fails the job and invalidates other assignments.
A different available compatible worker is preferred after an execution error.
Expired connections close with 1013; malformed protocol closes with 1008.

The browser also has a 15-second timeout and resets its device. This is best
effort: JS cannot guarantee interruption ahead of the GPU watchdog. Small tiles
and bounded shader loops are the primary protection. Large custom dispatches
and arbitrary uploaded WGSL are not introduced in M1.

## WebSocket /pool/events

Submitter first sends `{"v":1,"type":"subscribe","job_id":"<uuid>"}`.
Server responds with a job_snapshot and pushes job_update/job_done/job_failed:

```json
{"v":1,"type":"job_snapshot","job":{"job_id":"...","status":"running","progress":0.5,"completed_chunks":32,"total_chunks":64,"width":512,"height":512,"parameters":{},"tiles":[],"contributions":[],"retries":0,"error":null,"result_url":null},"workers":[]}
```

The above shortened snapshot illustrates field names; parameters, tiles and
workers are populated with the full models described here. Each accepted tile
also emits `{"v":1,"type":"tile_ready","job_id":"...","tile":{...}}`.
The submitter fetches its bytes and paints them at x/y. Snapshots list all accepted
tiles so reconnecting viewers can recover missed events. Bounded outgoing queues
may discard older updates; later snapshots restore the complete manifest.
Periodic snapshots keep the connection active. The UI retries socket connections.

The server assembles each tile by copying rows to the correct final RGBA offsets.
It does not compute Mandelbrot or encode PNG. Download PNG uses the assembled
submitter canvas, enabled only after all 64 tiles have been painted. An existing
job can be restored through its ID after a page reload.

## Verification and tests

```sh
backend/.venv/bin/python -m pytest -q
cd Frontend/HiveFrontend
npm run build
npm run lint
```

Optional browser integration test, with the built frontend and server running:

```sh
backend/.venv/bin/pip install playwright
HIVE_UI_URL=http://localhost:8000 backend/.venv/bin/python -m demos.pool_browser_check
```

Uses installed Chrome and two isolated WebGPU worker pages. Checks live tiles,
contributions, sampled Python reference pixels, PNG download, reload recovery,
disconnect/reassignment and cancellation. A test-only 100 ms delay before GPU
dispatch keeps active assignments observable for the disconnect test. It does
not change production code. These are browser tests on one computer; separate
physical-computer testing is the M1 handoff.

For a manual test: start contributors on two computers, render an image, inspect
each computer's accepted tile count and the submitter contribution list. Then
render again and close/Stop one contributor while working; verify retries and
completion. Increase max_iterations to 1024 if the default completes too quickly
to observe. Keep the surviving worker visible. Test restoration by reloading
and loading the job ID.

## Later milestones (now extended)

The implemented extensions are documented in [WORKLOADS.md](WORKLOADS.md).
M2 reuses tile coordinates plus a frame index and assembles frames in order.
M3 adds per-worker model caching and declared independent inference batches,
with operator/shape checks and ordered tensor assembly. M4 adds restricted
marked Python compilation. These extensions do not turn arbitrary Python/CUDA
files into portable browser execution or pool GPU memory into one device.
