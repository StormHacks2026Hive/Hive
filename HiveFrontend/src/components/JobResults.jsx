import { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'

export function saveFile(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement('a')
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
}
export default function JobResults({ jobId, onError }) {
  const [job, setJob] = useState(null), [frame, setFrame] = useState(0), [playing, setPlaying] = useState(false)
  const [preview, setPreview] = useState([]), [painted, setPainted] = useState(0)
  const canvas = useRef(null), drawn = useRef(new Set())
  useEffect(() => {
    if (!jobId) return
    let alive = true, timer
    setJob(null); setFrame(0); setPlaying(false); setPreview([])
    async function load() {
      try { const value = await api(`/pool/jobs/${jobId}`); if (alive) { setJob(value); if (['queued','running'].includes(value.status)) timer = setTimeout(load, 800) } }
      catch (e) { if (alive) onError(e.message) }
    }
    load(); return () => { alive = false; clearTimeout(timer) }
  }, [jobId, onError])
  useEffect(() => { drawn.current = new Set(); setPainted(0); const c = canvas.current; c?.getContext('2d').clearRect(0, 0, c.width, c.height) }, [jobId, frame])
  useEffect(() => {
    if (!job || job.output_format !== 'rgba8' || !canvas.current) return
    let alive = true
    const context = canvas.current.getContext('2d')
    async function draw() {
      try {
        const tiles = job.tiles.filter(t => t.frame_index === frame)
        for (const tile of tiles) {
          if (!alive || drawn.current.has(tile.chunk_id)) continue
          const response = await fetch(tile.url, { credentials: 'same-origin', cache: 'no-store' })
          if (!response.ok) throw Error('Could not load a rendered tile')
          const bytes = new Uint8ClampedArray(await response.arrayBuffer())
          if (!alive) return
          if (bytes.length !== tile.tile.width * tile.tile.height * 4) throw Error('Invalid tile size')
          context.putImageData(new ImageData(bytes, tile.tile.width, tile.tile.height), tile.tile.x, tile.tile.y)
          drawn.current.add(tile.chunk_id); setPainted(drawn.current.size)
        }
      } catch (e) { if (alive) onError(e.message) }
    }
    draw(); return () => { alive = false }
  }, [job, frame, onError])
  useEffect(() => {
    if (!job || job.status !== 'done' || job.output_format === 'rgba8') return
    let alive = true
    fetch(job.result_url).then(async r => {
      if (!r.ok) throw Error('Could not download output')
      const bytes = await r.arrayBuffer(), view = new DataView(bytes)
      const stride = job.output_format === 'f64' ? 8 : 4
      const method = { f64: 'getFloat64', f32: 'getFloat32', i32: 'getInt32', u32: 'getUint32' }[job.output_format]
      if (alive) setPreview(Array.from({ length: Math.min(32, bytes.byteLength / stride) }, (_, i) => view[method](i * stride, true)))
    }).catch(e => { if (alive) onError(e.message) })
    return () => { alive = false }
  }, [job?.status, job?.result_url, job?.output_format, onError])
  useEffect(() => {
    if (!playing || job?.status !== 'done' || job.frame_count < 2) return
    const timer = setInterval(() => setFrame(f => (f + 1) % job.frame_count), 1000 / job.fps)
    return () => clearInterval(timer)
  }, [playing, job?.status, job?.frame_count, job?.fps])
  async function cancel() { try { setJob(await api(`/pool/jobs/${jobId}/cancel`, {})) } catch (e) { onError(e.message) } }
  async function download() {
    try {
      if (job.output_format === 'rgba8') canvas.current.toBlob(blob => blob && saveFile(blob, `hive-${jobId}-${frame + 1}.png`), 'image/png')
      else { const r = await fetch(job.result_url); if (!r.ok) throw Error('Download failed'); saveFile(await r.blob(), `hive-${jobId}.${job.output_format}.bin`) }
    } catch (e) { onError(e.message) }
  }
  if (!jobId) return null
  if (!job) return <p role="status">Loading results…</p>
  const ready = job.output_format !== 'rgba8' || painted === job.tiles.filter(t => t.frame_index === frame).length && job.tiles.some(t => t.frame_index === frame)
  return <section className="job-results panel">
    <div className="panel-header"><h2>Results</h2><span className="status-pill">{job.status}</span></div>
    <p role="status">{job.completed_chunks} / {job.total_chunks} chunks · {job.retries} retries</p>
    <progress max="1" value={job.progress} />
    {job.error && <p className="form-error">{job.error}</p>}
    {job.output_format === 'rgba8' ? <>
      <canvas ref={canvas} width={job.width} height={job.height} aria-label="Rendered frame" />
      {job.frame_count > 1 && <div className="frame-controls"><label>Frame {frame + 1} / {job.frame_count}<input type="range" min="0" max={job.frame_count - 1} value={frame} onChange={e => { setPlaying(false); setFrame(Number(e.target.value)) }} /></label><button className="button button-secondary" disabled={job.status !== 'done'} onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button></div>}
    </> : preview.length > 0 && <pre aria-label="Output preview">{JSON.stringify(preview)}{job.output_shape.reduce((a,b) => a*b, 1) > 32 ? '\nFirst 32 values' : ''}</pre>}
    <div className="compute-actions"><button className="button button-secondary" disabled={job.status !== 'done' || !ready} onClick={download}>{job.output_format === 'rgba8' ? 'Download PNG' : 'Download output'}</button>{['queued','running'].includes(job.status) && <button className="button button-danger" onClick={cancel}>Cancel job</button>}</div>
    <ul className="contribution-list">{job.contributions.map(c => <li key={c.worker_id}>{c.label} · {c.chunks} chunks · {Math.round(c.elapsed_ms)} ms</li>)}</ul>
  </section>
}
