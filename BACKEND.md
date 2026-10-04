# Backend

Hive retains two working APIs: `backend/pool/` coordinates browser WebGPU workers,
and `backend/main.py` also serves the legacy Python-to-WGSL kernel prototype.
The new analyzer APIs are importable Python modules. Uploaded marked Python continues through the restricted compiler. Mandelbulb
Python uploads extract literal shader/config data with AST. Uploaded programs
are never sent to Ray or executed by the server.

| Module | Decision and execution |
| --- | --- |
| `onnx_analyzer.py` | PyTorch export; inferred tensor shapes, topological DAG, Gemm/MatMul/Conv FLOPs, memory, live skip tensors and disjoint branches. Dynamic programming balances score-normalized stage work and activation cuts. Extracted stages execute in dependency order; `run_pipeline_stream` overlaps independent microbatches in one lane per stage. Proven sample-independent models can use score-weighted batch sharding. Disjoint branch subgraphs execute concurrently before their join. |
| `wgsl_analyzer.py` | Browser-compatible 1D rewriting plus conservative compute metadata/hazard inspection. Accepts canonical current-element or row-major storage indexing and literal loops. Injects an offset/extent uniform, probes caller-supplied low-resolution invocation costs, and cuts cumulative cost into score-weighted bands. |
| `node_ranker.py` | Warm CPU/GPU ALU, bandwidth and transfer microbenchmarks; an explicit remote echo callback measures network overhead. TTL-cached rankings. Integer allocation solves an affine completion-time model, excluding offline nodes. |
| `cpu_analyzer.py` | AST analysis of restricted sum/min/max/append loops, pure list comprehensions fresh current-index array maps and separate `if` branches appending pure call results. Generates iteration functions, dispatches weighted chunks in Ray and combines ordered partial results. Refuses unknown effects and dependencies. |
| `scheduler.py` | `ComputeScheduler` validates region coverage, dispatches one lane per node and stitches output. `Scheduler`/`Node` remain compatibility exports for the legacy API. |
| `common/` | Node/range types, measured execution records, browser-pool adapter and the moved legacy lease scheduler. |

Scores are workload proxies. GPU ALU/bandwidth scores should rank GPU workloads;
CPU single-thread/core scores should rank CPU workloads. They are not measured
application speedups. Browser GPU reservations now mainly use device-spec
estimates, with bounded Mandelbrot-probe adjustments and replanning when workers
disconnect or pause. Legacy registration accepts
`compute_score`; unmeasured workers fall back to a single-node plan with a warning
rather than an assumed equal split. Fixed-size browser chunks
are transport units; reservations determine weighted node shares.

To add a node, construct `ComputeNode(node_id, score, ...)` from measurements.
For local WGSL set `device` to a wgpu device. For CUDA ONNX set `device` to its
integer CUDA ID (requires a CUDA-enabled ONNX Runtime installation). For remote
Ray set `ray_node_id` to an alive cluster node ID. A custom WGSL remote executor
must implement `execute(source, domain, buffers, output_binding, output_bytes,
offset, extent) -> bytes`, returning a full-domain output buffer. The existing
browser transport adapter supports its bounded 1D ABI. The browser pool also
supports registered Mandelbulb texture tiles; generic remote 2D/3D shaders remain
unsupported. Start existing browser contributors at `/node/`.

```sh
backend/.venv/bin/pip install -r requirements.txt
backend/.venv/bin/python -m pytest
backend/.venv/bin/python -m pytest -m slow -s
```

Fast tests exclude `slow` by default. Slow tests calibrate an actual serial run
until it lasts at least 30 seconds, log the final workload, check correctness,
print baseline/parallel/share/finish/idle tables, and write `tests/results/*.json`.
`BENCH_SCALE` selects the initial workload; it never disables the 30-second floor.
`BENCH_SPEEDUP` sets the default 1.3 threshold. Single-device GPU tests still check
correctness and explicitly omit a multi-device speedup assertion. Ray workers and
Metal devices require process/GPU permissions outside a restrictive sandbox.
GPU hash calibration scales completed bounded dispatch batches, rather than one
unbounded shader dispatch. Tables include startup/dispatch/merge work as documented
by each test; persistent Ray startup is warmed before application timing.

## Mandelbulb upload

Build the frontend (`cd HiveFrontend && npm install && npm run build`), then start:

```sh
backend/.venv/bin/python -m uvicorn backend.main:app --port 8000 --ws-max-size 12000000
```

Open `http://localhost:8000`, choose **Custom WGSL**, and upload
`demos/test_files/Mandelbulb_wgsl.py`. Click **Analyze source**, open contributor
pages at `/node/` and start contributing, then click **Run on team GPUs**.
Width/height overrides are optional; defaults are 1280 x 1024. Download the
stitched result with **Download image PNG**. It renders one still frame using
RenderConfig defaults; the script's 300-frame rotation loop is not executed.

`pool/image_workloads.py` validates the registered shader, extracts literal config
with AST, packs the original uniform layout, and adds global pixel offsets with
local texture writes. `node-web/gpu.js` renders storage textures, removes GPU row
padding and sends RGBA tiles over the existing transport. The coordinator uses
spec-weighted reservations and stitches tiles. The image route accepts
formatting/comments and config changes; arbitrary altered texture shaders are
rejected. Render-time cost probing is not part of this upload route.

```sh
backend/.venv/bin/python -m pytest tests/test_image_workloads.py -q
# With uvicorn running on port 8000; requires Chrome with WebGPU:
HIVE_UI_URL=http://localhost:8000 backend/.venv/bin/python -m demos.mandelbulb_upload_check
```

The browser check uploads the actual .py through the UI, uses two contributor
contexts, compares the stitched image to a full dispatch of the original shader,
and verifies downloaded PNG pixels and dimensions. Both contexts use the same
physical GPU; this is a correctness check with deliberate tile delays, not a
speed benchmark. `HIVE_RENDER_WIDTH` / `HIVE_RENDER_HEIGHT` change check dimensions.

## Cleanup

Removed `backend/wgsl_analysis.py` after moving its implementation to
`wgsl_analyzer.py`; pool imports and tests use the canonical module. Moved the
legacy scheduler to `common/legacy_scheduler.py`, retaining its public imports.
Removed a duplicate compiler size check and unused new-code variables/imports.
Legacy ranges now use measured scores when provided; unknown scores use one node.
Active legacy routes, tested authentication code and frontend copies were retained.

## Limits and incomplete showcase

- The supplied Mandelbulb renderer now works through the upload UI as a still
  image. The 30-second cost-probed 2/3/4-device showcase remains incomplete; no
  Mandelbulb speedup or physical multi-GPU scaling measurement is claimed.
- Arbitrary WGSL cannot be proven independent here. Helpers, textures, barriers,
  atomics, pointer escapes, dynamic loops and unfamiliar indexing are refused.
  Canonical 3D metadata/offsets are supported; scheduler stitching is 1D/2D only.
- Cost probing is a callback, not an automatically instrumented arbitrary shader.
  Automatic GPU cost counters and dynamic chunk rebalancing are not implemented.
- WGSL uses full-domain buffers on each node. This preserves indexing but duplicates
  storage/transfer; it can limit makespan. Only scalar dimension uniform layouts
  are validated for row-major dispatch.
- ONNX estimates exclude unsupported operators/unknown shapes from pipeline plans.
  Batch independence is limited to the existing pool operator subset; generic
  transformer batch independence is not inferred. Pipeline stream batch shapes
  must agree with the extracted graph. CUDA multi-GPU execution was not available
  on this machine; CPU fallback and graph/branch/stream correctness were tested.
- Legacy kernel seeds still follow the historical per-chunk seed contract; legacy
  stochastic jobs are not a new single-node equivalence guarantee.
- CPU analysis is intentionally restrictive. General mutation, arbitrary helper
  effects and mutually exclusive `elif` alternatives are not parallelized.
  Supplied helpers are checked before trusted local execution. Floating reductions
  can change rounding; tests use `rtol=1e-5`, `atol=1e-6` where applicable. Integer
  reductions and hash output comparisons are exact. The Mandelbulb browser upload check compares against an unmodified full-frame
  WebGPU reference with maximum channel difference <= 1 (1/255).
- Remote echo measurement requires a transport that supports echo; the browser
  protocol currently has no echo benchmark. Its existing throughput probe is a
  distinct proxy. GPU score stability and physical multi-GPU/network scaling have
  not been measured here.

The checked-in result JSON files are measured samples from this machine, not
portable performance guarantees. See the table below for the most recent full run.

| Workload | Serial (s) | Parallel/fallback (s) | Ratio |
| --- | ---: | ---: | ---: |
| Monte Carlo (4 Ray workers) | 33.27 | 9.52 | 3.50x |
| Eight independent branches (4 Ray workers) | 33.42 | 9.22 | 3.63x |
| Ranked throttled workers | 30.41 | 10.48 | 2.90x |
| ONNX deep stack CPU fallback | 33.20 | 32.84 | 1.01x |
| WGSL hash batches, one physical GPU | 33.12 | 34.01 | 0.97x |

The equal split with the throttled worker took 23.09s. GPU hash lanes
share one device; this measures correctness and overhead, not physical GPU scaling.
All executed integer/hash comparisons matched exactly; ONNX used allclose.

## Authenticated network integration

The UserAuth UI is merged into main. Networks, membership, enrolled nodes and run
history persist in SQLite. Google ID tokens are verified server-side; sessions
use HttpOnly cookies and CSRF tokens. `.env` is loaded by `backend/config.py` and
is local-only. See `HiveFrontend/README.md` for setup and Google origin settings.

`backend/programs.py` scans uploaded Python for every literal compute shader and
supported independent numeric regions. Automatic or editable CPU/GPU markers
select targets. The existing WGSL compiler remains the GPU execution path.
`browser_cpu.py` lowers the CPU analyzer's numeric subset into an interpreter IR
for browser Web Workers; numeric CPU results use float64. Ray remains available
through the local Python API and is not required by browser contributors.

The coordinator partitions by measured CPU throughput and estimated GPU specs, excludes inactive
nodes, and adds at most 8% to the most common GPU family. Network membership
restricts scheduling, status, results and private assets. SQLite retains device
capabilities/counters; pause, resume, stop and job cancellation act on real
workers. Kill/retry invalidates attempts and reassigns unfinished work. Uploaded
Mandelbulb animations generate validated per-frame uniforms with global pixel
coordinates and independent local texture writes.

Fast integration checks: `pytest tests/test_programs.py`. Real-browser check:
`backend/.venv/bin/python -m demos.authenticated_browser_check`. The script starts
an isolated server/database and checks authenticated create/join, a CPU-only
contributor, GPU/CPU markers and results, image reference pixels, PNG downloads,
and persistent controls. This is correctness validation, not a speed benchmark.

Browser nodes also report available GPU vendor/architecture/model, fallback-adapter
status, device buffer/texture limits, logical CPU cores, platform and approximate
system memory. Expand **Device specs** in the node list to see them. Browsers may
hide the model or omit/round cores and memory; this is not an inventory of exact
CPU models or VRAM. Reports persist with the node's capabilities in SQLite.
GPU limits filter incompatible chunks, including texture dimensions. GPU weights
use device-class priors, exposed Apple model tiers and bounded host core/RAM hints;
the probe influences this estimate by at most ±10%. These are estimates, and buffer
capacity is not compute speed. The CPU path
uses one Web Worker per node, so reported core counts do not multiply its score.
Older clients that omit these optional fields remain supported. Reload contributor
pages after updating to refresh their reports.

`ProgramRequest.target` accepts `auto`, `gpu` and `cpu`; explicit choices override
CPU/GPU comment targets. GPU conversion accepts independent maps/comprehensions,
including current-index arithmetic, division and selected numeric builtins.
CPU indexed comprehensions preserve global indices while chunks carry only local
input slices. Neither path executes arbitrary Python or claims unsupported code
can be converted. The user-provided shader wrapper is bundled as a frontend example
alongside `dense.onnx`. JobResults fetches complete output once, preserves every
numeric entry, and displays cached animation frames with play/seek controls.
