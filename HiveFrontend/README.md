# Hive frontend

React UI from the UserAuth branch, connected to the Python compute backend.
The Network, Mapping and Compute tabs use authenticated server data. Connected
browsers contribute CPU work, and GPU work when WebGPU is available.

## Run

From the repository root, install backend dependencies and create local settings:

```sh
backend/.venv/bin/pip install -r requirements.txt
cp .env.example .env
backend/.venv/bin/python -m uvicorn backend.main:app --port 8000 --ws-max-size 12000000
```

Set `GOOGLE_CLIENT_ID` in `.env` to your Google Web application client ID.
The server verifies Google ID token signatures and issues an HttpOnly session
cookie. This flow does not use a Google client secret. A server-only placeholder
can be stored in `GOOGLE_CLIENT_SECRET`; never prefix secrets with `VITE_`.
Authorize your actual frontend origin in Google Cloud, including the HTTPS tunnel
origin when using Cloudflare. Set `COOKIE_SECURE=true` for HTTPS deployments.

On Render, set `HIVE_DB_PATH=/var/data/hive.sqlite3` only after attaching a
persistent disk mounted at `/var/data`. Without that disk (including free
instances), leave `HIVE_DB_PATH` unset or use `data/hive.sqlite3`. Local storage
is writable but ephemeral: accounts, networks and sessions disappear on
redeploy/restart. `Permission denied: '/var/data'` in sign-in logs means the disk
is absent or its mount path does not match the configured database path.

For development, in a second terminal:

```sh
cd HiveFrontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/auth`, `/api`, `/pool`, contributor
modules and WebSockets to the Python server. `HIVE_API_URL` can override the
backend URL. Authentication settings come from `/auth/config`, not a bundled
frontend environment variable.

For the single-origin server used in demos:

```sh
cd HiveFrontend
npm run build
cd ..
backend/.venv/bin/python -m uvicorn backend.main:app --port 8000 --ws-max-size 12000000
```

Open `http://localhost:8000`. Use Node 22.12+ (or another supported Vite version).
Generated `dist/` files are local build artifacts.

## Use

Guests use the same network and compute features with a server-issued session.
Refresh keeps their identity and saved workspace for the session's one-hour lifetime.
Signing out or clearing cookies ends access to that guest identity; a new guest
session cannot recover its previous networks. Google sign-in remains available.
On the map, your device is the hive and other devices are bees.

1. Sign in with Google or choose **Continue as guest**, create a password-protected network, and share its ID.
2. Other users sign in and join with that ID and password. Each browser enrolls
   a persistent device and automatically contributes while its tab is visible.
3. In Compute, upload a `.py` or `.wgsl` file or type code. Supply an input array
   for array computations, or turn Input array off for count-based work.
4. Analyze to see independent regions and planned contributor shares. Mark &
   analyze inserts `# hive:gpu begin/end` or `# hive:cpu begin/end` comments.
   Edit both comments to change the target and analyze again. Manual segmentation
   runs only explicitly marked Python regions.
5. Send the work and inspect results. Numeric outputs download as little-endian
   binary (`f32`, `f64`, `i32` or `u32`, shown by the backend); images download as PNG.
6. For Animation, upload the Mandelbulb renderer, choose dimensions/frame count
   and camera turn, or use built-in Mandelbrot. Renderer values and explicit
   per-frame JSON overrides are optional. Frames can be scrubbed and played.

Network and Mapping show actual node states, workload assignments and byte totals.
Pause prevents new work; an active chunk may finish. Kill disconnects the node,
reassigns unfinished work and persists an off state. Start/Resume restarts it.
The node owner, network owner and current job sender can control that node.
Job cancellation is available to the sender and network owner.

SQLite at `data/hive.sqlite3` persists users, hashed sessions, salted scrypt
network passwords, memberships, devices, counters and submission history.
`HIVE_DB_PATH` overrides the path. Local storage keeps only view preferences.
Results and active leases remain in memory; old results become unavailable after
expiry or a server restart, while network/device history remains saved.

## Limits

The upload analyzer executes supported independent regions, not a whole Python
program. It never runs uploaded imports, arbitrary functions or I/O. Embedded
literal WGSL shaders are all inspected. The registered Mandelbulb texture shader
is supported; arbitrary texture shaders and inter-region dependencies are refused.

Browser CPU work uses a bounded numeric interpreter lowered from the CPU analyzer.
It does not require Ray on phones or laptops. The separate local Python CPU API
uses Ray. Browser CPU results use float64 and bounded numbers; unsupported Python
integer/bit operations and call signatures are rejected. Floating reductions can
change rounding. Scheduling uses throughput benchmarks with at most an 8% bonus
for the most common GPU family, not a hardcoded laptop/phone preference.

## Checks

```sh
npm test
npm run lint
npm run build
# From repository root:
backend/.venv/bin/python -m pytest
backend/.venv/bin/python -m demos.authenticated_browser_check
```

The browser check starts an isolated temporary server/database with locally seeded
test sessions. It verifies the real UI, CPU-only fallback, GPU results, editable
markers, animations, PNG pixels and persistent node controls. Google token
verification is separately tested with signed JWTs. No test login bypass is exposed
by the production server. Two browser contexts share one physical GPU, so the check
does not claim a multi-GPU speedup.
