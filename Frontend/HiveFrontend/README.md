# Hive submission frontend

Start the FastAPI backend from the repository root:

```sh
backend/.venv/bin/python -m uvicorn backend.main:app --ws-max-size 12000000
```

Then, from this directory:

```sh
npm ci
npm run dev
```

Open the Vite URL (normally http://localhost:5173). Upload a `.py` file or edit
its source, analyze compatibility, review/edit the proposed kernel, validate,
configure inputs, and submit. Open the GPU node link in one or more tabs before
submitting. Verification requires two nodes. The Mandelbrot example renders
its completed results in a canvas.

Automatic conversion currently handles a single function with one input list,
an empty output list, a loop appending independent +, -, * arithmetic, and a
return of that output list. Other code receives findings and requires manual
rewriting. Uploaded source is never executed by the server.

Vite proxies API and node routes to http://127.0.0.1:8000. Override with
`HIVE_API_URL=http://127.0.0.1:8001 npm run dev` if needed. For production,
configure equivalent proxy routes for `/kernels`, `/jobs`, `/node`, and `/nodes`
on the frontend host. `npm run build` produces the frontend bundle;
`npm run lint` checks its source.

See ../../PROTOCOL.md for API details and limitations.
