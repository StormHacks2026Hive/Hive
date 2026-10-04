# Hive

Browser-only volunteer GPU compute for independent workloads. The coordinator
is **Python FastAPI**, and contributor GPUs run **WebGPU** in browser Web Workers.

**M1 is ready:** a 512×512 Mandelbrot image, 64 independent 64×64 tiles, live
assembly, capability reporting, pull scheduling, retries, Stop, and PNG download.
Animation, ONNX inference batches, custom WGSL analysis, and marked Python are
now available through the pool. See [WORKLOADS.md](WORKLOADS.md) for the current
contracts, supported subsets, and browser tests.

Build the existing React frontend (Node 22.12+ for build tools only):

```sh
cd HiveFrontend
npm ci
npm run build
cd ..
```

Start one Python server (Python 3.11+):

```sh
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r requirements.txt
backend/.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --ws-max-size 12000000
```

Open **http://localhost:8000/** for the submitter. Open
**http://localhost:8000/node/** on each contributor and click **Start contributing**.
Keep contributor pages visible. For remote computers, expose port 8000 through
an HTTPS tunnel and share its `/node/` URL. Then submit a workload, or click **Render on team GPUs** in the Mandelbrot tab.
The distributed-compute dashboard analyzes and submits the new workloads. The
Mandelbrot tab shows which computers contributed and paints each tile live.
Joining runs a short GPU benchmark. Stopping/closing a worker reassigns its
unfinished tile. The server only schedules and places bytes; it never runs the
fractal computation or submitted Python.

See [M1_PROTOCOL.md](M1_PROTOCOL.md) for all M1 messages, bounds, and a full
two-computer test. Build before starting/restarting uvicorn to enable the UI
at `/`. For frontend development, `npm run dev` proxies the same backend routes.

```sh
backend/.venv/bin/python -m pytest -q
# Optional Chrome integration test, with server already running:
backend/.venv/bin/pip install playwright
backend/.venv/bin/python -m demos.workloads_browser_check
```

The earlier Python-kernel prototype remains under the UI's **Python kernel
prototype** tab, with GPU workers at **/legacy-node/**. Its `/jobs`, `/nodes`
and `/kernels` routes remain unchanged. See [PROTOCOL.md](PROTOCOL.md). Python
files are accepted only in its restricted subset or recognized conversion
patterns; arbitrary Python, CUDA and Numba execution is outside this project.
