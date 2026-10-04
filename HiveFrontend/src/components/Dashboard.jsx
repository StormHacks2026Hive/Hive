import { useEffect, useRef, useState } from 'react'

const WGSL = `@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    result[i] = values[i] * 2.0 + 1.0;
}`
const PYTHON = `def transform(values):
    result = [0.0] * len(values)
    for i in range(len(values)):
        scaled = values[i] * 2
        result[i] = scaled + 1
    return result
`
const BASE = { xmin: -2, xmax: 1, ymin: -1.5, ymax: 1.5, max_iterations: 256 }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const value = await response.json()
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : JSON.stringify(value.detail))
  return value
}
function base64(bytes) {
  let result = ''
  for (let i = 0; i < bytes.length; i += 8192) result += String.fromCharCode(...bytes.subarray(i, i + 8192))
  return btoa(result)
}
function typedInput(text, dtype = 'f32') {
  const values = JSON.parse(text)
  if (!Array.isArray(values) || !values.length || values.some(v => typeof v !== 'number' || !Number.isFinite(v))) throw new Error('Input must be a nonempty JSON array of finite numbers')
  if (values.length > 500000) throw new Error('The editor accepts at most 500,000 input elements')
  const bytes = new ArrayBuffer(values.length * 4), view = new DataView(bytes)
  const method = { f32: 'setFloat32', i32: 'setInt32', u32: 'setUint32' }[dtype]
  for (let i = 0; i < values.length; i++) {
    const value = values[i]
    if (dtype === 'f32' ? Math.abs(value) > 3.402823e38 : !Number.isInteger(value) || value < (dtype === 'i32' ? -2147483648 : 0) || value > (dtype === 'i32' ? 2147483647 : 4294967295)) throw new Error(`An input value does not fit ${dtype}`)
    view[method](i * 4, value, true)
  }
  return { dtype, data: base64(new Uint8Array(bytes)) }
}
function save(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement('a')
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export default function Dashboard() {
  const [kind, setKind] = useState('wgsl'), [sources, setSources] = useState({ wgsl: WGSL, python: PYTHON })
  const [input, setInput] = useState('[1, 2, 3, 4]'), [useInput, setUseInput] = useState(true)
  const [count, setCount] = useState(8192), [chunkSize, setChunkSize] = useState(1024)
  const [model, setModel] = useState(''), [modelName, setModelName] = useState(''), [shape, setShape] = useState(''), [batchSize, setBatchSize] = useState(32)
  const [frames, setFrames] = useState(12), [fps, setFps] = useState(12), [distribution, setDistribution] = useState('tiles')
  const [imageWidth, setImageWidth] = useState(''), [imageHeight, setImageHeight] = useState('')
  const [displayedFrame, setDisplayedFrame] = useState('')
  const [report, setReport] = useState(null), [candidateLine, setCandidateLine] = useState('')
  const [job, setJob] = useState(null), [workers, setWorkers] = useState([]), [busy, setBusy] = useState(false), [error, setError] = useState('')
  const [restoreId, setRestoreId] = useState(''), [frame, setFrame] = useState(0), [playing, setPlaying] = useState(false), [preview, setPreview] = useState([])
  const canvas = useRef(null), revision = useRef(0), frameCache = useRef(new Map())
  const source = sources[kind] || '', jobId = job?.job_id
  const imageSource = kind === 'wgsl' && /WGSL_SHADER|texture_storage_2d/.test(source)
  const imageOptions = { ...(imageWidth ? { width: Number(imageWidth) } : {}), ...(imageHeight ? { height: Number(imageHeight) } : {}) }
  const running = ['queued', 'running'].includes(job?.status)
  function edit(action) { revision.current++; setReport(null); setError(''); action() }
  useEffect(() => {
    let alive = true
    const load = async () => {
      try { const value = await api('/pool/workers'); if (alive) setWorkers(value) } catch (e) { if (alive) setError(e.message) }
    }
    load(); const timer = setInterval(load, 2000)
    return () => { alive = false; clearInterval(timer) }
  }, [])
  useEffect(() => {
    if (!jobId) return
    let alive = true
    const load = async () => {
      try { const value = await api(`/pool/jobs/${jobId}`); if (alive) setJob(value) } catch (e) { if (alive) setError(e.message) }
    }
    load(); const timer = setInterval(load, 1000)
    return () => { alive = false; clearInterval(timer) }
  }, [jobId])
  useEffect(() => {
    if (!jobId || job?.output_format !== 'rgba8') return
    let alive = true
    const draw = async () => {
      const tiles = job.tiles.filter(t => t.frame_index === frame)
      const expected = job.total_chunks / job.frame_count
      const key = `${jobId}:${frame}`
      const context = canvas.current?.getContext('2d')
      if (!context) return
      setDisplayedFrame('')
      context.clearRect(0, 0, job.width, job.height)
      if (tiles.length !== expected) return
      try {
        let bytes = frameCache.current.get(key)
        if (!bytes) {
          const response = await fetch(`/pool/jobs/${jobId}/frames/${frame}`)
          if (!response.ok) throw new Error('Could not load the rendered image')
          bytes = new Uint8ClampedArray(await response.arrayBuffer())
          if (frameCache.current.size >= 32) frameCache.current.delete(frameCache.current.keys().next().value)
          frameCache.current.set(key, bytes)
        }
        if (alive) { context.putImageData(new ImageData(bytes, job.width, job.height), 0, 0); setDisplayedFrame(key) }
      } catch (e) { if (alive) setError(e.message) }
    }
    draw()
    return () => { alive = false }
  }, [jobId, job, frame])
  useEffect(() => {
    if (!playing || job?.status !== 'done' || job.kind !== 'animation') return
    const timer = setInterval(() => setFrame(value => (value + 1) % job.frame_count), 1000 / job.fps)
    return () => clearInterval(timer)
  }, [playing, job?.status, job?.kind, job?.frame_count, job?.fps])
  useEffect(() => {
    if (job?.status !== 'done' || job.output_format === 'rgba8') return
    let alive = true
    fetch(job.result_url).then(async response => {
      if (!response.ok) throw new Error('Could not download completed output')
      const bytes = await response.arrayBuffer(), view = new DataView(bytes)
      const method = { f32: 'getFloat32', i32: 'getInt32', u32: 'getUint32' }[job.output_format]
      const values = Array.from({ length: Math.min(32, bytes.byteLength / 4) }, (_, i) => view[method](i * 4, true))
      if (alive) setPreview(values)
    }).catch(e => { if (alive) setError(e.message) })
    return () => { alive = false }
  }, [job?.status, job?.output_format, job?.result_url])
  async function uploadSource(file) {
    if (!file) return
    const version = ++revision.current; setReport(null); setError('')
    if (kind === 'onnx') { setModel(''); setModelName('') }
    try {
      if (file.size > (kind === 'onnx' ? 4194304 : 65536)) throw new Error('File exceeds the upload size limit')
      if (kind === 'onnx') {
        const value = base64(new Uint8Array(await file.arrayBuffer()))
        if (version === revision.current) { setModel(value); setModelName(file.name) }
      } else {
        const value = await file.text()
        if (version === revision.current) {
          const target = /WGSL_SHADER/.test(value) ? 'wgsl' : kind
          setSources(values => ({ ...values, [target]: value })); setKind(target)
          setImageWidth(''); setImageHeight('')
        }
      }
    } catch (e) { if (version === revision.current) setError(e.message) }
  }
  async function analyze(autoMark = false) {
    setBusy(true); setError(''); const version = revision.current
    try {
      let value
      if (kind === 'python') {
        value = await api('/pool/python/analyze', { source, auto_mark: autoMark, ...(candidateLine ? { candidate_line: Number(candidateLine) } : {}) })
      } else if (imageSource) {
        value = await api('/pool/wgsl/image/analyze', { source, ...imageOptions })
      } else if (kind === 'wgsl') {
        const total = useInput ? JSON.parse(input).length : Number(count)
        value = await api('/pool/wgsl/analyze', { source, count: total, chunk_size: Number(chunkSize) })
        if (value.status === 'ready' && value.input_dtype && !useInput) throw new Error('This shader reads an input array; enable Use an input array')
      } else {
        if (!model) throw new Error('Upload an ONNX model first')
        value = await api('/pool/onnx/analyze', { model, batch_size: Number(batchSize), ...(shape.trim() ? { input_shape: shape.split(',').map(Number) } : {}) })
      }
      if (version !== revision.current) return
      if (value.marked_source) setSources(values => ({ ...values, python: value.marked_source }))
      if (kind === 'onnx' && value.status === 'ready' && !shape.trim()) {
        const elements = JSON.parse(input).length, perSample = value.model_input_shape.slice(1).reduce((a, b) => a * b, 1)
        if (elements && elements % perSample === 0) {
          value = await api('/pool/onnx/analyze', { model, input_shape: [elements / perSample, ...value.model_input_shape.slice(1)], batch_size: Number(batchSize) })
          if (version !== revision.current) return
        }
        if (value.status === 'ready') setShape(value.input_shape.join(', '))
      }
      setReport(value)
    } catch (e) { if (version === revision.current) setError(e.message) } finally { setBusy(false) }
  }
  async function submit() {
    setBusy(true); setError('')
    try {
      let request
      if (kind === 'animation') {
        request = { kind, fps: Number(fps), distribution, frames: Array.from({ length: Number(frames) }, (_, i) => {
          const zoom = 1 + i * .035
          return { ...BASE, xmin: -.75 - 1.5 / zoom, xmax: -.75 + 1.5 / zoom, ymin: -1.5 / zoom, ymax: 1.5 / zoom }
        }) }
      } else if (imageSource) {
        request = { kind: 'wgsl_image', source, ...imageOptions }
      } else if (kind === 'onnx') {
        request = { kind, model, input: typedInput(input), input_shape: shape.split(',').map(Number), batch_size: Number(batchSize), independent: true }
      } else {
        request = { kind, chunk_size: Number(chunkSize), count: Number(count) }
        if (kind === 'python') { request.source = source; request.input = typedInput(input) }
        else {
          request.wgsl = source; request.output_dtype = report.output_dtype
          if (report.input_dtype) request.input = typedInput(input, report.input_dtype)
        }
      }
      const created = await api('/pool/jobs', request)
      frameCache.current.clear(); setDisplayedFrame(''); setPreview([]); setFrame(0); setPlaying(false)
      setJob(await api(`/pool/jobs/${created.job_id}`))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  async function restore() {
    try { setError(''); setFrame(0); setPlaying(false); setPreview([]); setJob(await api(`/pool/jobs/${encodeURIComponent(restoreId.trim())}`)) } catch (e) { setError(e.message) }
  }
  async function cancel() {
    try { setJob(await api(`/pool/jobs/${jobId}/cancel`, {})) } catch (e) { setError(e.message) }
  }
  async function download() {
    try {
      if (job.output_format === 'rgba8') {
        canvas.current.toBlob(blob => { if (blob) save(blob, `hive-${jobId}-frame-${frame}.png`) }, 'image/png')
      } else {
        const response = await fetch(job.result_url)
        if (!response.ok) throw new Error('Could not download output')
        save(await response.blob(), `hive-${jobId}.${job.output_format}.bin`)
      }
    } catch (e) { setError(e.message) }
  }
  return <main>
    <header><div><p className="eyebrow">HIVE / DISTRIBUTED COMPUTE</p><h1>Give your GPUs a shared task.</h1><p>Analyze independent work, send smaller chunks to your contributors, and assemble results in order.</p></div><a className="node-link" href="/node/" target="_blank" rel="noreferrer">Open GPU contributor ↗</a></header>
    {error && <p className="error panel" role="alert">{error}</p>}
    <div className="workspace">
      <section className="panel"><h2>Submit a workload</h2>
        <label>Workload<select aria-label="Workload" value={kind} onChange={e => edit(() => { setKind(e.target.value); setCandidateLine('') })}><option value="wgsl">Custom WGSL</option><option value="onnx">ONNX inference</option><option value="python">Marked Python</option><option value="animation">Animation</option></select></label>
        {kind === 'animation' ? <>
          <p>Render a Mandelbrot zoom as independent frames or 64 × 64 image tiles.</p>
          <div className="configuration"><label>Frames<input type="number" min="1" max="32" value={frames} onChange={e => setFrames(e.target.value)} /></label><label>Frames per second<input type="number" min="1" max="60" value={fps} onChange={e => setFps(e.target.value)} /></label></div>
          <label>Distribute<select aria-label="Distribute" value={distribution} onChange={e => setDistribution(e.target.value)}><option value="tiles">Image tiles across all frames</option><option value="frames">One complete frame per chunk</option></select></label>
        </> : <>
          <label className="upload">{kind === 'onnx' ? 'Upload model (.onnx, up to 4 MiB)' : `Upload ${kind === 'python' ? 'Python' : 'WGSL (.wgsl or Mandelbulb .py)'} source`}<input key={kind} type="file" accept={kind === 'onnx' ? '.onnx' : kind === 'python' ? '.py' : '.wgsl,.py'} onChange={e => uploadSource(e.target.files[0])} /></label>
          {kind === 'onnx' ? <>
            <p className="hint">{modelName || 'Choose a model to inspect its operators and sample dimensions.'} Each device caches the full model and processes a separate input batch.</p>
            <div className="configuration"><label>Input shape (comma separated)<input placeholder="Analyze to detect sample dimensions" value={shape} onChange={e => edit(() => setShape(e.target.value))} /></label><label>Samples per batch<input type="number" min="1" max="256" value={batchSize} onChange={e => edit(() => setBatchSize(e.target.value))} /></label></div>
          </> : <>
            <label htmlFor="workload-source">Source</label><textarea id="workload-source" spellCheck="false" value={source} onChange={e => edit(() => setSources(values => ({ ...values, [kind]: e.target.value })))} />
            <p className="hint">{imageSource ? 'Mandelbulb renderer: upload extracts the shader and config. Render one still image; no input array is needed.' : kind === 'python' ? 'Mark one for loop with # hive:parallel begin and # hive:parallel end, or let the analyzer insert markers. Supply its input array below.' : 'The analyzer recognizes independent array computations and rewrites buffer bindings and indexing for distributed chunks.'}</p>
            {kind === 'wgsl' && !imageSource && <label className="checkbox"><input type="checkbox" checked={useInput} onChange={e => edit(() => setUseInput(e.target.checked))} />Use an input array</label>}
            {imageSource ? <div className="configuration"><label>Image width<input aria-label="Image width" type="number" min="1" max="4096" placeholder="From RenderConfig" value={imageWidth} onChange={e => edit(() => setImageWidth(e.target.value))} /></label><label>Image height<input aria-label="Image height" type="number" min="1" max="4096" placeholder="From RenderConfig" value={imageHeight} onChange={e => edit(() => setImageHeight(e.target.value))} /></label></div> : <div className="configuration"><label>Elements per chunk<input type="number" min="1" max="4096" value={chunkSize} onChange={e => edit(() => setChunkSize(e.target.value))} /></label>{!useInput && kind === 'wgsl' && <label>Total output elements<input type="number" min="1" max="2000000" value={count} onChange={e => edit(() => setCount(e.target.value))} /></label>}</div>}
          </>}
          {!imageSource && (useInput || kind !== 'wgsl') && <><label htmlFor="workload-input">Input array (flat JSON numbers)</label><textarea id="workload-input" className="small" value={input} onChange={e => edit(() => setInput(e.target.value))} /></>}
          <div className="actions"><button className="secondary" onClick={() => analyze(false)} disabled={busy}>Analyze {kind === 'onnx' ? 'model' : 'source'}</button>{kind === 'python' && <button className="secondary" onClick={() => analyze(true)} disabled={busy}>Find and mark parallel loop</button>}</div>
          {kind === 'python' && report?.candidates?.length > 1 && <label>Loop to mark<select aria-label="Loop to mark" value={candidateLine} onChange={e => setCandidateLine(e.target.value)}><option value="">Choose a loop</option>{report.candidates.map(c => <option key={c.line} value={c.line}>Line {c.line}: {c.pattern} ({c.input_name} → {c.output_name})</option>)}</select></label>}
        </>}
        <button className="submit-job" onClick={submit} disabled={busy || running || (kind !== 'animation' && report?.status !== 'ready')}>Run on team GPUs</button>
        <details><summary>Restore a job</summary><label>Job ID<input value={restoreId} onChange={e => setRestoreId(e.target.value)} /></label><button className="secondary" onClick={restore} disabled={!restoreId.trim()}>Load job</button></details>
      </section>
      <div><section className="panel"><h2>Partition analysis</h2>
        {report ? <><span className="badge">{report.status.replaceAll('_', ' ')}</span><ul className="findings">{report.findings?.map((f, i) => <li key={i}>{f}</li>)}</ul>
          {Boolean(report.chunk_count) && <p>{report.chunk_count} independent chunks{report.batch_size && ` · ${report.batch_size} samples per batch`}</p>}
          {report.output_shape && <p className="mono">Output shape: [{report.output_shape.join(', ')}]</p>}
          {report.operators && <p className="mono">Operators: {report.operators.join(', ')}</p>}
          {report.wgsl && <details><summary>Generated WGSL</summary><pre>{report.wgsl}</pre></details>}
        </> : <p className="hint">{kind === 'animation' ? 'Frames and tiles are independent. Completed chunks are placed into their frame in sequence.' : 'Upload or edit your source, then analyze to review how the backend divides it.'}</p>}
      </section><section className="panel"><h2>Connected contributors · {workers.length}</h2>
        {workers.length ? <ul className="findings">{workers.map(w => <li key={w.worker_id}><strong>{w.label}</strong><span className="badge">{w.state}</span><p>{w.completed_chunks} accepted chunks · {w.capabilities.onnx ? 'ONNX ready' : 'WebGPU compute'}</p></li>)}</ul> : <p className="hint">Open contributor pages on your computers and keep them visible. Share an HTTPS origin with teammates.</p>}
      </section></div>
    </div>
    {job && <section className="panel"><div className="panel-heading"><h2>Results · {job.kind}</h2><span className="mono">{jobId}</span></div><p role="status">{job.status} · {job.completed_chunks}/{job.total_chunks} chunks · {job.retries} retries</p><progress max="1" value={job.progress} />
      {job.status === 'queued' && <p>Waiting for a compatible contributor.</p>}{job.error && <p className="error">{job.error}</p>}
      {job.output_format === 'rgba8' ? <><canvas ref={canvas} width={job.width} height={job.height} aria-label={job.kind === 'wgsl_image' ? 'Rendered image' : 'Animation frame'} />{job.frame_count > 1 && <><label>Frame {frame + 1} / {job.frame_count}<input type="range" min="0" max={job.frame_count - 1} value={frame} onChange={e => { setPlaying(false); setFrame(Number(e.target.value)) }} /></label><button className="secondary" disabled={job.status !== 'done'} onClick={() => setPlaying(!playing)}>{playing ? 'Pause animation' : 'Play animation'}</button></>}</> : <><p className="mono">{job.output_format} · shape [{job.output_shape.join(', ')}]</p>{preview.length > 0 && <pre aria-label="Output preview">{JSON.stringify(preview)}{job.output_shape.reduce((a,b) => a*b,1) > 32 ? '\nFirst 32 values shown' : ''}</pre>}</>}
      <div className="actions"><button disabled={job.status !== 'done' || (job.output_format === 'rgba8' && displayedFrame !== `${jobId}:${frame}`)} onClick={download}>{job.output_format === 'rgba8' ? (job.kind === 'wgsl_image' ? 'Download image PNG' : 'Download frame PNG') : 'Download output'}</button>{running && <button className="secondary" onClick={cancel}>Cancel job</button>}</div>
      <ul className="findings">{job.contributions.map(c => <li key={c.worker_id}>{c.label} · {c.chunks} chunks · {c.elapsed_ms.toFixed(1)} ms</li>)}</ul>
    </section>}
  </main>
}
