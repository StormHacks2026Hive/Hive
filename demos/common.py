import os
import time
import httpx

def submit(payload):
    base = os.getenv('HIVE_URL', 'http://localhost:8000')
    with httpx.Client(base_url=base, timeout=60) as client:
        response = client.post('/jobs', json=payload)
        response.raise_for_status()
        job_id = response.json()['job_id']
        print(f'Job {job_id}; waiting for browser nodes…')
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            response = client.get(f'/jobs/{job_id}')
            response.raise_for_status()
            job = response.json()
            if job['status'] == 'failed':
                raise RuntimeError(job['error'])
            if job['status'] == 'done':
                return job
            time.sleep(.5)
        raise TimeoutError('Demo timed out. Open WebGPU browser tabs at /node/.')
