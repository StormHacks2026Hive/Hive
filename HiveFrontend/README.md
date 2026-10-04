# Hive frontend

Active React frontend for distributed GPU workloads. Run `npm ci`,
`npm run dev` or `npm run build` here. Vite proxies `/pool`, contributor pages,
and the legacy APIs to FastAPI at port 8000; override with `HIVE_API_URL`.
The backend serves this directory’s `dist` at `/`.

See [WORKLOADS.md](../WORKLOADS.md) for analysis/submission APIs, accepted
Python markers, WGSL binding contracts, ONNX batch limits, and browser tests.

The original frontend source was absent in this checkout. Its previous compiled
build is preserved locally in `.previous-dist/`; the new source adds the workload
dashboard and retains the Mandelbrot and Python prototype flows.
