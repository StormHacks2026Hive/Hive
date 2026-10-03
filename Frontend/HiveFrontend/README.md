# Hive submission frontend

The default page is the M1 Mandelbrot pool: 64 independent GPU tiles, live
assembly, contributor capabilities and contribution counts, cancellation and
PNG download. Open `/node/` on each contributor and click Start contributing.
Keep contributor pages visible.

From this directory, with Node 22.12+ for build tooling:

```sh
npm ci
npm run build
```

Then start the Python backend from the repository root. FastAPI serves the built
frontend at `/`, so one HTTPS tunnel to the backend supports submitters and nodes.
See [M1_PROTOCOL.md](../../M1_PROTOCOL.md) for setup, messages and testing.

For development, start the backend on port 8000, then run `npm run dev` here.
Vite proxies `/pool`, `/shared`, `/node` and legacy API routes. Override the
backend URL with `HIVE_API_URL=http://127.0.0.1:8001 npm run dev` when needed.
A separately hosted production frontend requires equivalent HTTP/WebSocket
proxy routes. `npm run lint` checks frontend source.

The Python kernel prototype navigation tab preserves the earlier upload,
compatibility report, editable preview and submission flow. It uses worker
pages at `/legacy-node/` and the legacy `/jobs`, `/nodes`, `/kernels` API.
See [PROTOCOL.md](../../PROTOCOL.md). Arbitrary Python/CUDA files cannot run on
these browser nodes.
