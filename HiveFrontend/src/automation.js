export function shellQuote(value) { return `'${String(value).replaceAll("'", "'\\''")}'` }

export function jobCalls(origin, networkId, task, job, key = '') {
  const base = `${origin}/api/v1/networks/${encodeURIComponent(networkId)}`
  const header = '-H "Authorization: Bearer $HIVE_API_KEY"'
  const root = job ? `${base}/jobs/${encodeURIComponent(job.job_id)}` : ''
  return {
    auth: `export HIVE_API_KEY=${shellQuote(key || 'PASTE_YOUR_API_KEY')}`,
    status: job ? `curl --fail-with-body ${header} ${shellQuote(root)}` : '',
    result: job ? `curl --fail-with-body ${header} ${shellQuote(root + '/result' + (job.kind === 'animation' || job.kind === 'wgsl_image' || job.kind === 'mandelbrot' ? '?format=binary' : ''))}${['animation', 'wgsl_image', 'mandelbrot'].includes(job.kind) ? ' --output frames.rgba' : ''}` : '',
    repeat: task ? `curl --fail-with-body -X POST ${header} ${shellQuote(`${base}/runs/${encodeURIComponent(task.id)}/repeat`)}` : '',
    python: task ? `import os, time, requests

base = ${JSON.stringify(base)}
headers = {"Authorization": "Bearer " + os.environ["HIVE_API_KEY"]}
response = requests.post(base + ${JSON.stringify(`/runs/${task.id}/repeat`)}, headers=headers, timeout=30)
response.raise_for_status()
run = response.json()

for job in run["jobs"]:
    url = base + "/jobs/" + job["job_id"]
    deadline = time.monotonic() + 300
    while True:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        status = response.json()
        if status["status"] == "done":
            break
        if status["status"] in ("failed", "cancelled"):
            raise RuntimeError(status.get("error") or status["status"])
        if time.monotonic() > deadline:
            raise TimeoutError("Task is still waiting or running")
        time.sleep(1)
    if status["output_format"] == "rgba8":
        result = requests.get(url + "/result?format=binary", headers=headers, timeout=30)
        result.raise_for_status()
        with open(job["job_id"] + ".rgba", "wb") as output:
            output.write(result.content)
    else:
        result = requests.get(url + "/result", headers=headers, timeout=30)
        result.raise_for_status()
        print(job["name"], result.json()["values"])
` : '',
  }
}

export function localDateTime(value = Date.now() + 60000) {
  const date = new Date(value)
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16)
}
