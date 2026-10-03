# Hive

Distributed browser WebGPU compute MVP. Python kernels are parsed and translated
with py2wgsl on the server, then executed exclusively on browser nodes.

```sh
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r requirements.txt
backend/.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --ws-max-size 12000000
```

Open http://localhost:8000/node/ in **two or more browser tabs**, then run:

```sh
backend/.venv/bin/python -m demos.monte_carlo
backend/.venv/bin/python -m demos.elementwise
backend/.venv/bin/python -m pytest -q
```

See [PROTOCOL.md](PROTOCOL.md) for API messages, kernel syntax, limits,
frontend integration, optional automated Chrome checks, and next improvements.

The existing React frontend is currently a Vite starter page with no API calls;
the backend does not require changes to any existing routes or message formats.
It still needs a job submission form and status polling to become a submitter UI.

Validated locally with two Chrome WebGPU tabs: Monte Carlo π = **3.140440**;
all **1,000,000** elementwise outputs matched the plain Python reference.
