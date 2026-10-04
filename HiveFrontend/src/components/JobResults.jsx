import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api.js'
import { decodeFrameOutput, decodeNumericOutput } from '../results.js'

export function saveFile(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement('a')
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
}
export default function JobResults({ jobId, onError }) {
  const [job, setJob] = useState(null), [frame, setFrame] = useState(0), [playing, setPlaying] = useState(false)
  const [values, setValues] = useState(null), [pixels, setPixels] = useState(null)
  const outputJson = useMemo(() => values === null ? '' : JSON.stringify(values), [values])
  const canvas = useRef(null), drawn = useRef(new Set())
  useEffect(() => {
    if (!jobId) return
    let alive = true, timer
    async function load() {
      try {
        const value = await api(`/pool/jobs/${jobId}`)
        if (alive) { setJob(value); if (['queued', 'running'].includes(value.status)) timer = setTimeout(load, 800) }
      } catch (e) { if (alive) onError(e.message) }
    }
    load()
    return () => { alive = false; clearTimeout(timer) }
  }, [jobId, onError])

  // Progressive frame zero while the workers are rendering.
  useEffect(() => {
    if (!job || job.status === 'done' || job.output_format !== 'rgba8' || !canvas.current) return
    const controller = new AbortController()
    const context = canvas.current.getContext('2d')
    async function draw() {
      try {
        for (const tile of job.tiles.filter(t => t.frame_index === 0)) {
          if (drawn.current.has(tile.chunk_id)) continue
          const response = await fetch(tile.url, { credentials: 'same-origin', cache: 'no-store', signal: controller.signal })
          if (!response.ok) throw Error('Could not load a rendered tile')
          const bytes = new Uint8ClampedArray(await response.arrayBuffer())
          if (controller.signal.aborted) return
          if (bytes.length !== tile.tile.width * tile.tile.height * 4) throw Error('Invalid tile size')
          context.putImageData(new ImageData(bytes, tile.tile.width, tile.tile.height), tile.tile.x, tile.tile.y)
          drawn.current.add(tile.chunk_id)
        }
      } catch (e) { if (!controller.signal.aborted) onError(e.message) }
    }
    draw()
    return () => controller.abort()
  }, [job, onError])

  // Fetch the complete ordered output once, including every animation frame.
  useEffect(() => {
    if (!job || job.status !== 'done') return
    const controller = new AbortController()
    async function loadOutput() {
      try {
        const response = await fetch(job.result_url, { credentials: 'same-origin', cache: 'no-store', signal: controller.signal })
        if (!response.ok) throw Error('Could not load output')
        const buffer = await response.arrayBuffer()
        if (controller.signal.aborted) return
        if (job.output_format === 'rgba8') {
          setPixels(decodeFrameOutput(buffer, job.width, job.height, job.frame_count))
          setPlaying(job.frame_count > 1 && !window.matchMedia('(prefers-reduced-motion: reduce)').matches)
        } else {
          const output = decodeNumericOutput(buffer, job.output_format)
          if (output.length !== job.output_shape.reduce((a, b) => a * b, 1)) throw Error('Output shape does not match its data')
          setValues(output)
        }
      } catch (e) { if (!controller.signal.aborted) onError(e.message) }
    }
    loadOutput()
    return () => controller.abort()
  }, [job, onError])

  useEffect(() => {
    if (!pixels || !job || !canvas.current) return
    const size = job.width * job.height * 4
    canvas.current.getContext('2d').putImageData(new ImageData(pixels.subarray(frame * size, (frame + 1) * size), job.width, job.height), 0, 0)
  }, [pixels, frame, job])
  useEffect(() => {
    if (!playing || !pixels || !job || job.frame_count < 2) return
    const timer = setInterval(() => setFrame(f => (f + 1) % job.frame_count), 1000 / job.fps)
    return () => clearInterval(timer)
  }, [playing, pixels, job])

  async function cancel() { try { setJob(await api(`/pool/jobs/${jobId}/cancel`, {})) } catch (e) { onError(e.message) } }
  async function download() {
    try {
      if (job.output_format === 'rgba8') canvas.current.toBlob(blob => blob && saveFile(blob, `hive-${jobId}-${frame + 1}.png`), 'image/png')
      else { const r = await fetch(job.result_url); if (!r.ok) throw Error('Download failed'); saveFile(await r.blob(), `hive-${jobId}.${job.output_format}.bin`) }
    } catch (e) { onError(e.message) }
  }
  if (!jobId) return null
  if (!job) return <p role="status">Loading results…</p>
  const ready = job.output_format === 'rgba8' ? Boolean(pixels) : values !== null
  return <section className="job-results panel">
    <div className="panel-header"><h2>Results</h2><span className="status-pill">{job.status}</span></div>
    <p role="status">{job.completed_chunks} / {job.total_chunks} chunks · {job.retries} retries</p>
    <progress max="1" value={job.progress} />
    {job.error && <p className="form-error">{job.error}</p>}
    {job.output_format === 'rgba8' ? <>
      <canvas ref={canvas} width={job.width} height={job.height} aria-label="Rendered frame" />
      {job.frame_count > 1 && <div className="frame-controls"><label>Frame {frame + 1} / {job.frame_count}<input aria-label="Animation frame" type="range" min="0" max={job.frame_count - 1} value={frame} disabled={!pixels} onChange={e => { setPlaying(false); setFrame(Number(e.target.value)) }} /></label><button className="button button-secondary" disabled={!pixels} onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button></div>}
    </> : values !== null && <><p>{values.length.toLocaleString()} output values · shape {job.output_shape.join(' × ')}</p><pre aria-label="Output array">{outputJson}</pre></>}
    <div className="compute-actions"><button className="button button-secondary" disabled={job.status !== 'done' || !ready} onClick={download}>{job.output_format === 'rgba8' ? 'Download PNG' : 'Download output'}</button>{values !== null && <button className="button button-secondary" onClick={() => saveFile(new Blob([outputJson], { type: 'application/json' }), `hive-${jobId}.json`)}>Download JSON</button>}{['queued','running'].includes(job.status) && <button className="button button-danger" onClick={cancel}>Cancel job</button>}</div>
    <ul className="contribution-list">{job.contributions.map(c => <li key={c.worker_id}>{c.label} · {c.chunks} chunks · {Math.round(c.elapsed_ms)} ms</li>)}</ul>
  </section>
}
