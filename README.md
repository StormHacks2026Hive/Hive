# 🐝 Hive

> **StormHacks 2026**

Shared computing power, straight from your browser. Bring laptops, phones, and tablets into one hive to distribute supported CPU and GPU workloads across connected devices.

![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![WebGPU](https://img.shields.io/badge/WebGPU-005A9C?style=flat)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)
![Render](https://img.shields.io/badge/Render-000000?style=flat&logo=render&logoColor=white)

**Website:** [hivehacks.tech](https://hivehacks.tech)

**Design:** [Figma Design](https://www.figma.com/design/cId1aiVpk7prtHzWrRqHMy/HIVE-Design?node-id=0-1&t=GUo3J9geh1mnxujW-1)

---

## Table of Contents

- [What is Hive?](#what-is-hive)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running Locally](#running-locally)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Testing](#testing)
- [Known Limits](#known-limits)
- [Tech Stack](#tech-stack)

---

## What is Hive?

Hive is a full-stack distributed computing application built at StormHacks 2026. It started with a problem we faced as students: getting access to powerful GPUs often meant paying for cloud compute. We wanted to explore pooling the devices we already had, especially when short demo windows made computation time matter.

Users create a password-protected computing network and invite others to join. A Python server analyzes supported computations, assigns work to connected browsers, and combines their outputs. Contributors use WebGPU for GPU execution and Web Workers for supported CPU tasks, without installing a node client.

| Feature | Description |
|---|---|
| **Browser Contributors** | Join from supported laptops, phones, and tablets over HTTPS |
| **Python and WGSL** | Upload `.py` or `.wgsl` files, or type code into the editor |
| **CPU and GPU Modes** | Edit CPU/GPU markers or select a target for supported Python regions |
| **Weighted Work Sharing** | Assign unequal shares using device estimates, benchmarks, and workload compatibility |
| **Animation** | Distribute supported image tiles and frames, preview animations, and download PNGs |
| **ONNX Inference** | Distribute supported model inference batches through the browser runtime |
| **Live Node Controls** | See device activity, pause contributors, stop nodes, and cancel jobs |
| **Google and Guest Access** | Sign in with Google or enter with a temporary guest identity |
| **QR Invites** | Share a network link without putting its password in the URL |
| **API and Timers** | Submit tasks through scoped API keys and schedule saved tasks |

---

## How It Works

**1. Create or Join a Hive**

Sign in, create a network, and share its ID or QR invite. Other users enter the network password to join. Their browsers register as contributors and report available CPU/GPU capabilities.

**2. Submit a Computation**

Upload a file or type your code. Choose Compute, Animation, or ONNX mode. Python tasks can use automatic analysis or editable `# hive:gpu` and `# hive:cpu` markers.

**3. Analyze and Review the Split**

The backend scans literal WGSL shaders and supported independent Python regions. It checks dependencies and shader hazards before proposing a split. GPU-marked Python regions are translated into WGSL; CPU-marked regions are lowered into a bounded numeric interpreter for browser workers.

**4. Assign Work to Connected Devices**

The server assigns compatible chunks with unequal shares. Browser GPU weights use device-spec estimates with a bounded rendering-probe adjustment; CPU weights use a single-worker benchmark. The coordinator handles heartbeats, node availability, retries, and unfinished work when a contributor disconnects.

**5. Combine and Download Results**

Numeric chunks are assembled in their original order and supported reductions are combined. Image tiles are stitched into frames. Users can inspect outputs, play animations, and download JSON, binary results, or rendered PNGs.

**6. Repeat Tasks Through APIs or Timers**

Create a network-scoped API key to submit work from another application, or schedule a saved task from the Timer tab. Timers run on the server, but the server and compatible contributors must be online.

---

## Prerequisites

- Python 3.12 recommended
- Node.js 22.14+ and npm for the frontend build
- A browser with WebGPU support for GPU contribution; supported CPU tasks can run without WebGPU
- HTTPS for remote contributors, or `localhost` for local development
- A Google OAuth **Web application** client for Google sign-in; guest access works without it

SQLite is included with Python. The browser contribution path does not require a GPU on the coordinating server.

---

## Installation

**Clone the repo**

```bash
git clone https://github.com/StormHacks2026Hive/Hive.git
cd Hive
```

**Backend setup**

Run from the repository root:

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r requirements.txt
```

**Frontend setup**

```bash
cd HiveFrontend
npm ci
npm run build
cd ..
```

**Create local settings**

For a fresh checkout:

```bash
cp .env.example .env
```

---

## Environment Variables

Create `.env` in the **repository root**, next to `requirements.txt`:

```env
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
COOKIE_SECURE=false
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
HIVE_DB_PATH=data/hive.sqlite3
HIVE_ALLOW_LEGACY_POOL=false
```

The root `.env` is ignored by Git. API keys and credentials should stay out of source control.

| Variable | Description |
|---|---|
| `GOOGLE_CLIENT_ID` | Google Web application client ID; optional for guest-only access |
| `COOKIE_SECURE` | `false` for local HTTP; `true` for hosted HTTPS |
| `CORS_ORIGINS` | Comma-separated allowed origins for cross-origin browser API calls |
| `HIVE_DB_PATH` | Writable SQLite file path; defaults to `data/hive.sqlite3` in the repo |
| `HIVE_ALLOW_LEGACY_POOL` | Enables the historical anonymous demo pool when `true`; normally leave `false` |

Google sign-in verifies ID tokens and does not require a client secret. In Google Cloud, authorize the exact frontend origin, such as `http://localhost:8000`, `http://localhost:5173`, or your deployed HTTPS domain.

---

## Running Locally

**Run the website and backend together**

After building the frontend, run this from the repository root:

```bash
backend/.venv/bin/python -m uvicorn backend.main:app --port 8000 --ws-max-size 12000000 --workers 1
```

Open **http://localhost:8000**. Interactive API docs are at **http://localhost:8000/docs**.

The Python server serves the built React files and the backend API on the same port. A separate frontend server is optional.

**Frontend development with live updates**

Keep the backend running, then open another terminal:

```bash
cd HiveFrontend
npm run dev -- --port 5173 --strictPort
```

Open **http://localhost:5173**. Vite forwards API and WebSocket requests to the backend on port 8000. Run `npm run build` again when you want updated frontend files served through port 8000.

**Try it with other devices**

With the backend running, a temporary HTTPS tunnel can expose it:

```bash
cloudflared tunnel --url http://localhost:8000
```

Share the generated HTTPS URL, then have contributors sign in or continue as guests and join the same network. Keep contributor tabs visible. Authorize the tunnel's exact origin in Google Cloud if using Google sign-in.

**Example files**

| File | Use |
|---|---|
| [cpu_sum.py](demos/test_files/cpu_sum.py) | CPU reduction; input `[1, 2, 3, 4]` produces `40` |
| [elementwise.wgsl](demos/test_files/elementwise.wgsl) | Custom WGSL array computation |
| [Mandelbulb_wgsl.py](demos/test_files/Mandelbulb_wgsl.py) | Supported Python-wrapped Mandelbulb renderer |
| [Packaged Mandelbulb example](HiveFrontend/public/examples/Mandelbulb.wgsl) | Renderer example available through the UI |
| [dense.onnx](HiveFrontend/public/examples/dense.onnx) | Small ONNX model; use the UI's matching example inputs |

The Compute tab's **Example** selector supplies compatible modes and settings.

---

## API Reference

Base URL: `http://localhost:8000` locally, or your deployed HTTPS address.

Create a key in the **API** tab, choose read-only or submit access, and copy it when shown. Keys are scoped to a network and stored as hashes. External requests use:

```text
Authorization: Bearer YOUR_API_KEY
```

All paths below are relative to `/api/v1/networks/{network_id}`:

| Method | Path | Purpose |
|---|---|---|
| GET | `/runs` | List saved tasks and their jobs |
| POST | `/runs` | Analyze and submit supported Python or WGSL |
| POST | `/onnx` | Submit an ONNX model and inference input |
| POST | `/runs/{run_id}/repeat` | Repeat a whole saved task |
| GET | `/jobs/{job_id}` | Get status, progress, format, and output shape |
| GET | `/jobs/{job_id}/result` | Get complete numeric output as JSON |
| GET | `/jobs/{job_id}/result?format=binary` | Get numeric bytes or rendered RGBA frames |
| POST | `/jobs/{job_id}/cancel` | Cancel an authorized job |

Submit, repeat, and cancel operations require a key with submit access. Browser session endpoints use session cookies and CSRF protection instead of bearer keys.

### Submit a CPU task

```bash
curl -X POST "http://localhost:8000/api/v1/networks/YOUR_NETWORK_ID/runs" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "sum_squares.py",
    "source": "def total(count):\n    result = 0\n    # hive:cpu begin\n    for i in range(count):\n        result += i * i\n    # hive:cpu end\n    return result\n",
    "count": 10
  }'
```

The response includes a run ID and a `jobs` list. Poll each job's status, then fetch its result when the status is `done`. The example produces `[285]`.

### Retrieve a numeric result

```bash
curl "http://localhost:8000/api/v1/networks/YOUR_NETWORK_ID/jobs/YOUR_JOB_ID/result" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Example completed response:

```json
{
  "job_id": "YOUR_JOB_ID",
  "format": "f64",
  "shape": [1],
  "values": [285.0]
}
```

Rendered frames use the binary result endpoint. The UI handles image assembly and PNG downloads. For full request schemas, see `/docs` and [the frontend guide](HiveFrontend/README.md).

---

## Deployment

Hive can run as **one Render Python Web Service** that builds and serves the React frontend.

**Build command**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt && npm --prefix HiveFrontend ci && npm --prefix HiveFrontend run build
```

**Start command**

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT --ws-max-size 12000000 --workers 1
```

Leave the Root Directory blank. Set `PYTHON_VERSION=3.12.11`, `NODE_VERSION=22.14.0`, `COOKIE_SECURE=true`, and your Google client ID in the service's environment settings.

For persistent storage, attach a disk mounted at `/var/data` and set `HIVE_DB_PATH=/var/data/hive.sqlite3`. Without a disk, use `data/hive.sqlite3`; local files are ephemeral and may disappear on restart or redeploy. An unmounted `/var/data` can cause sign-in to fail with a permission error.

Add your custom domain in Render, configure the DNS records it provides, and authorize the HTTPS origin in Google Cloud. Use one server process and instance because active connections and computed results are held in memory.

---

## Testing

**Fast backend tests**

```bash
backend/.venv/bin/python -m pytest
```

**Frontend checks**

```bash
cd HiveFrontend
npm test
npm run lint
npm run build
cd ..
```

**Real-browser integration check**

With Chrome installed:

```bash
backend/.venv/bin/pip install playwright
backend/.venv/bin/python -m demos.authenticated_browser_check
```

The check starts an isolated server and temporary database. It exercises authenticated networks, browser contributors, CPU/GPU outputs, image rendering, and node controls. Google token verification is separately tested using signed JWTs.

**Slow benchmarks**

```bash
backend/.venv/bin/python -m pytest -m slow -s
```

Benchmark tests calibrate supported workloads to at least 30 seconds of serial work and write measurements to `tests/results/`. Hardware-dependent cases may skip when suitable devices are unavailable. Browser contexts on the same physical GPU do not demonstrate a multi-GPU speedup. See [BACKEND.md](BACKEND.md) for benchmark details and limitations.

---

## Known Limits

- Python analysis supports restricted independent numeric regions. Uploaded imports, arbitrary functions, and I/O are not executed on the server.
- WGSL hazards and unsupported shader contracts are refused. Texture rendering currently supports registered renderer patterns, rather than arbitrary texture shaders.
- GPU shares are allocation estimates. Browser hardware reports can be incomplete, and a short probe may not predict a particular workload's performance.
- Browser CPU execution uses a numeric interpreter in one Web Worker per node. The separate local Python CPU component uses Ray.
- Guest access lasts for the session. Losing the guest cookie can lose access to networks created under that identity.
- Saved tasks and network history persist in SQLite, but outputs and active jobs are kept in memory. Download results you want to keep; they can expire or become unavailable after a restart.
- Parallel execution is not guaranteed to be faster. Workload size, device speed, transfer overhead, and available contributors affect the result.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React, Vite, Tailwind CSS, custom CSS and SVG hive illustrations |
| **Backend** | Python, FastAPI, Uvicorn, Pydantic |
| **Browser GPU** | WebGPU, WGSL, browser Web Workers |
| **Browser CPU** | Web Workers with a bounded numeric interpreter |
| **Model Analysis** | PyTorch, ONNX, ONNX Runtime, ONNX Runtime Web |
| **Local CPU Parallelism** | Python AST analysis and Ray |
| **Transport** | HTTP and WebSockets |
| **Authentication** | Google Identity Services, guest sessions, CSRF protection, hashed API keys |
| **Database** | SQLite with hashed sessions and salted scrypt network passwords |
| **Images and Results** | NumPy, Pillow, browser canvas |
| **Deployment** | Render; Cloudflare Tunnel for local sharing |

More implementation details: [BACKEND.md](BACKEND.md), [WORKLOADS.md](WORKLOADS.md), and [HiveFrontend/README.md](HiveFrontend/README.md).

---

*Shared power. Sweet possibilities. © 2026 Hive*
