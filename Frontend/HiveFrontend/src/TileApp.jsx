import { useEffect, useRef, useState } from 'react'

async function api(path, body) {
  const response = await fetch(path, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const value = await response.json()
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : JSON.stringify(value.detail))
  return value
}
export default function TileApp() {
  const [parameters, setParameters] = useState({ xmin: -2, xmax: 1, ymin: -1.5, ymax: 1.5, max_iterations: 256 })
  const [job, setJob] = useState(null), [workers, setWorkers] = useState([])
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [connection, setConnection] = useState('Not subscribed')
  const [painted, setPainted] = useState(0), [restoreId, setRestoreId] = useState('')
  const canvas = useRef(null), drawn = useRef(new Set()), inflight = useRef(new Set()), activeJob = useRef(null)
  useEffect(() => {
    let active = true
    const load = async () => {
      try { const current = await api('/pool/workers'); if (active) setWorkers(current) }
      catch (e) { if (active) setError(e.message) }
    }
    load(); const timer = setInterval(load, 3000)
    return () => { active = false; clearInterval(timer) }
  }, [])
  const jobId = job?.job_id
  useEffect(() => {
    if (!jobId) return
    let stopped = false, socket, retry, retries = 0
    activeJob.current = jobId; drawn.current = new Set(); inflight.current = new Set()
    const context = canvas.current.getContext('2d')
    context.fillStyle = '#101713'; context.fillRect(0, 0, 512, 512)
    async function paint(tile) {
      if (stopped || drawn.current.has(tile.chunk_id) || inflight.current.has(tile.chunk_id)) return
      inflight.current.add(tile.chunk_id)
      try {
        const response = await fetch(tile.url)
        if (!response.ok) throw new Error('Could not download a completed tile')
        const bytes = new Uint8ClampedArray(await response.arrayBuffer())
        if (stopped || activeJob.current !== jobId) return
        if (bytes.length !== tile.tile.width * tile.tile.height * 4) throw new Error('Invalid tile data')
        context.putImageData(new ImageData(bytes, tile.tile.width, tile.tile.height), tile.tile.x, tile.tile.y)
        drawn.current.add(tile.chunk_id); setPainted(drawn.current.size)
      } catch (e) { if (!stopped) setError(e.message) }
      finally { if (!stopped) inflight.current.delete(tile.chunk_id) }
    }
    function connect() {
      if (stopped) return
      const url = new URL('/pool/events', location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(url)
      socket.onopen = () => { setConnection('Live updates connected'); retries = 0; socket.send(JSON.stringify({ v: 1, type: 'subscribe', job_id: jobId })) }
      socket.onmessage = event => {
        if (stopped) return
        const message = JSON.parse(event.data)
        if (message.type === 'tile_ready') paint(message.tile)
        if (message.job) {
          setJob(message.job); setWorkers(message.workers)
          message.job.tiles.forEach(paint)
        }
      }
      socket.onclose = () => {
        if (!stopped) { setConnection('Reconnecting live updates…'); retry = setTimeout(connect, Math.min(10000, 1000 * 2 ** Math.min(retries++, 4))) }
      }
      socket.onerror = () => socket.close()
    }
    connect()
    return () => { stopped = true; clearTimeout(retry); socket?.close() }
  }, [jobId])
  async function submit() {
    setBusy(true); setError('')
    try {
      const values = Object.fromEntries(Object.entries(parameters).map(([k, v]) => [k, Number(v)]))
      if (Object.values(values).some(v => !Number.isFinite(v))) throw new Error('All parameters must be finite numbers')
      const created = await api('/pool/jobs', { kind: 'mandelbrot', parameters: values })
      setPainted(0)
      setJob(await api(`/pool/jobs/${created.job_id}`))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  async function restore() {
    setBusy(true); setError('')
    try {
      const restored = await api(`/pool/jobs/${encodeURIComponent(restoreId.trim())}`)
      if (restored.job_id !== job?.job_id) setPainted(0)
      setJob(restored)
    }
    catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  async function cancel() {
    try { setJob(await api(`/pool/jobs/${jobId}/cancel`, {})) }
    catch (e) { setError(e.message) }
  }
  function download() {
    canvas.current.toBlob(blob => {
      if (!blob) return
      const url = URL.createObjectURL(blob), link = document.createElement('a')
      link.href = url; link.download = `mandelbrot-${jobId}.png`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
    }, 'image/png')
  }
  return <main>
    <header><div><p className="eyebrow">HIVE / BROWSER GPU POOL</p><h1>One image. Your team's GPUs.</h1><p>64 independent tiles, rendered on teammates' GPUs and stitched here as they arrive.</p></div>
      <a href="/node/" target="_blank" rel="noreferrer" className="node-link">Open GPU contributor ↗</a>
    </header>
    {error && <p role="alert" className="error panel">{error}</p>}
    <div className="workspace">
      <section className="panel"><h2>M1 · Mandelbrot</h2><p>512 × 512 pixels · 64 tiles of 64 × 64 · WebGPU only</p>
        <form onSubmit={e => { e.preventDefault(); submit() }}>
          <div className="configuration">{Object.entries(parameters).map(([key, value]) => <label key={key}>
            {key === 'max_iterations' ? 'Maximum iterations (1–1024)' : key}
            <input type="number" step={key === 'max_iterations' ? '1' : 'any'} min={key === 'max_iterations' ? 1 : undefined} max={key === 'max_iterations' ? 1024 : undefined} required value={value} onChange={e => setParameters({ ...parameters, [key]: e.target.value })} />
          </label>)}</div>
          <p className="hint">Open contributor pages on your computers and click Start contributing. Keep them visible. Each device runs a short benchmark when joining.</p>
          <button disabled={busy || ['queued', 'running'].includes(job?.status)} type="submit">Render on team GPUs</button>
        </form>
        <p className="hint">Float32 deep zoom can lose detail. Begin with these broad coordinate bounds.</p>
        <details><summary>Reconnect to an existing job</summary><label>Job ID<input value={restoreId} onChange={e => setRestoreId(e.target.value)} /></label><button className="secondary" onClick={restore} disabled={busy || !restoreId.trim()}>Load job</button></details>
      </section>
      <section className="panel"><h2>Connected contributors · {workers.length}</h2>
        {!workers.length && <p className="empty">No contributors yet. Open the contributor link locally or share your HTTPS URL with a teammate.</p>}
        <ul className="findings">{workers.map(w => <li key={w.worker_id}>
          <strong>{w.label}</strong> <span className="badge">{w.state}</span>
          <p>{w.completed_chunks} accepted tiles · {w.pixels_per_second ? `${Math.round(w.pixels_per_second).toLocaleString()} pixels/s` : 'Awaiting first tile'}</p>
          <p className="hint">Benchmark: {w.capabilities.benchmark.elapsed_ms.toFixed(2)} ms · {w.capabilities.adapter.description || w.capabilities.adapter.vendor || 'Adapter info unavailable'}</p>
        </li>)}</ul>
      </section>
    </div>
    <section className="panel"><div className="panel-heading"><h2>Live assembled image</h2><span>{connection}</span></div>
      <canvas className="tile-canvas" ref={canvas} width="512" height="512" aria-label="Live Mandelbrot image" />
      {job ? <>
        <p className="mono">Job {job.job_id}</p><p role="status">{job.status} · {job.completed_chunks}/64 accepted tiles · {painted}/64 displayed · {job.retries} retries</p>
        <progress value={job.progress} max="1" />
        {job.status === 'queued' && <p>Waiting for a compatible, visible GPU contributor.</p>}
        {job.error && <p className="error">{job.error}</p>}
        <div className="actions"><button onClick={download} disabled={job.status !== 'done' || painted !== 64}>Download PNG</button>
          {['queued', 'running'].includes(job.status) && <button className="secondary" onClick={cancel}>Cancel job</button>}</div>
        <h3>Who rendered this image?</h3><ul className="findings">{job.contributions.map(c => <li key={c.worker_id}>{c.label} · {c.chunks} tiles · {c.elapsed_ms.toFixed(1)} ms execution + readback</li>)}</ul>
      </> : <p className="empty">Submit a job to see completed tiles appear here.</p>}
    </section>
  </main>
}
