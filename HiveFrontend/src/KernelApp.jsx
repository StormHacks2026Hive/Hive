import { useEffect, useMemo, useRef, useState } from 'react'
import { pythonExample, mandelbrotKernel } from './examples'

async function api(path, body) {
  const response = await fetch(path, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
  return data
}
function typedInput(values, dtype) {
  if (!Array.isArray(values) || !values.length || values.length > 2000000 || values.some(v => typeof v !== 'number' || !Number.isFinite(v))) throw new Error('Enter 1–2,000,000 finite numbers in an input array.')
  const bytes = new Uint8Array(values.length * 4), view = new DataView(bytes.buffer)
  values.forEach((v, i) => {
    if (dtype !== 'f32' && (!Number.isInteger(v) || (dtype === 'u32' ? v < 0 || v > 4294967295 : v < -2147483648 || v > 2147483647))) throw new Error(`Input values must fit ${dtype}.`)
    view[{ f32: 'setFloat32', u32: 'setUint32', i32: 'setInt32' }[dtype]](i * 4, v, true)
  })
  let binary = ''
  for (let i = 0; i < bytes.length; i += 32768) binary += String.fromCharCode(...bytes.subarray(i, i + 32768))
  return { dtype, data: btoa(binary) }
}
function decode(array) {
  const bytes = Uint8Array.from(atob(array.data), c => c.charCodeAt(0)), view = new DataView(bytes.buffer)
  const get = { f32: 'getFloat32', u32: 'getUint32', i32: 'getInt32' }[array.dtype]
  return Array.from({ length: bytes.length / 4 }, (_, i) => view[get](i * 4, true))
}
function Result({ job, imageSize }) {
  const canvas = useRef(null)
  const values = useMemo(() => job?.status === 'done' && job.result?.data ? decode(job.result) : null, [job])
  useEffect(() => {
    if (!canvas.current || !values || !imageSize || values.length !== imageSize.width * imageSize.height) return
    const context = canvas.current.getContext('2d'), image = context.createImageData(imageSize.width, imageSize.height)
    values.forEach((v, i) => {
      const t = v / 256
      image.data.set(v >= 256 ? [12, 20, 23, 255] : [Math.round(255 * Math.sqrt(t)), Math.round(210 * t), Math.round(100 + 155 * (1 - t)), 255], i * 4)
    })
    context.putImageData(image, 0, 0)
  }, [values, imageSize])
  if (!job) return null
  return <section className="panel"><h2>Job results</h2><p className="mono">{job.job_id}</p>
    <p role="status">{job.status} · {job.completed_chunks}/{job.total_chunks} chunks</p>
    <progress value={job.progress} max="1" />
    {job.status === 'queued' && <p>Waiting for GPU nodes. Open the node page in one or more browser tabs.</p>}
    {job.error && <p className="error">{job.error}</p>}
    {job.status === 'done' && <>
      {imageSize && values?.length === imageSize.width * imageSize.height && <canvas ref={canvas} width={imageSize.width} height={imageSize.height} aria-label="Mandelbrot rendering" />}
      <p>{values ? `${values.length.toLocaleString()} outputs · first 20 shown` : 'Reduced result'}</p>
      <pre>{JSON.stringify(values ? values.slice(0, 20) : job.result, null, 2)}</pre>
      <button onClick={() => {
        const url = URL.createObjectURL(new Blob([JSON.stringify(job.result)], { type: 'application/json' }))
        const a = document.createElement('a'); a.href = url; a.download = `${job.job_id}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
      }}>Download result</button>
    </>}
  </section>
}

export default function App() {
  const [source, setSource] = useState(pythonExample), [filename, setFilename] = useState('example.py')
  const [report, setReport] = useState(null), [kernel, setKernel] = useState(''), [validation, setValidation] = useState(null)
  const [parameters, setParameters] = useState('{}'), [input, setInput] = useState('[1, 2, 3, 4, 5]')
  const [count, setCount] = useState(65536), [reduce, setReduce] = useState(''), [verify, setVerify] = useState(false)
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [job, setJob] = useState(null)
  const [mandelbrot, setMandelbrot] = useState(false), [imageSize, setImageSize] = useState(null)
  const revision = useRef(0)
  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return
    let active = true
    const timer = setInterval(async () => {
      try { const update = await api(`/jobs/${job.job_id}`); if (active) setJob(update) }
      catch (e) { if (active) setError(e.message) }
    }, 1000)
    return () => { active = false; clearInterval(timer) }
  }, [job])
  function editSource(value) {
    revision.current++; setSource(value); setReport(null); setKernel(''); setValidation(null); setError(''); setMandelbrot(false)
  }
  function editKernel(value) { revision.current++; setKernel(value); setValidation(null); setError('') }
  async function upload(event) {
    const file = event.target.files?.[0]
    if (!file) return
    try {
      if (!file.name.endsWith('.py')) throw new Error('Choose a .py file.')
      if (file.size > 128000) throw new Error('File is too large; source is limited to 32,000 characters.')
      const text = await file.text()
      if (text.length > 32000) throw new Error('Source is limited to 32,000 characters.')
      editSource(text); setFilename(file.name)
    } catch (e) { setError(e.message) }
    event.target.value = ''
  }
  async function analyze() {
    setBusy(true); setError(''); const current = revision.current
    try {
      const result = await api('/kernels/analyze', { source })
      if (revision.current !== current) return
      setReport(result); setKernel(result.kernel || ''); setValidation(null); setParameters(JSON.stringify(result.parameters, null, 2)); setMandelbrot(false)
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  async function validate() {
    setBusy(true); setError(''); const current = revision.current
    try { const result = await api('/kernels/validate', { source: kernel }); if (revision.current === current) setValidation(result) }
    catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  function loadMandelbrot() {
    revision.current++; setKernel(mandelbrotKernel); setValidation(null); setMandelbrot(true); setCount(65536); setReduce(''); setError('')
    setParameters(JSON.stringify({ width: 256, height: 256, xmin: -2.0, xmax: 1.0, ymin: -1.5, ymax: 1.5 }, null, 2))
    setReport({ status: 'example', findings: [{ severity: 'info', message: 'Hand-authored Mandelbrot example, not an automatic conversion of your file. Each pixel runs at most 256 iterations. Default size: 256 × 256.' }] })
  }
  async function submit() {
    setBusy(true); setError('')
    try {
      const constants = JSON.parse(parameters)
      if (!constants || Array.isArray(constants) || typeof constants !== 'object') throw new Error('Parameters must be a JSON object.')
      const payload = { kernel, mode: validation.mode, parameters: constants, reduce: reduce || null, verify }
      if (validation.mode === 'data-slice') {
        const binding = validation.bindings.find(b => b.access === 'read')
        payload.input = typedInput(JSON.parse(input), binding.element_type)
      } else {
        payload.count = Number(count)
        if (!Number.isInteger(payload.count) || payload.count < 1 || payload.count > 2000000) throw new Error('Output count must be 1–2,000,000.')
      }
      let size = null
      if (mandelbrot && !reduce) {
        if (!Number.isInteger(constants.width) || !Number.isInteger(constants.height) || constants.width < 1 || constants.height < 1 || constants.width * constants.height !== payload.count) throw new Error('Mandelbrot width × height must equal output count.')
        size = { width: constants.width, height: constants.height }
      }
      const created = await api('/jobs', payload)
      setImageSize(size); setJob({ ...created, status: 'queued', progress: 0, completed_chunks: 0, total_chunks: 0, result: null })
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  return <main>
    <header><div><p className="eyebrow">HIVE / GPU COMPUTE</p><h1>From Python to parallel.</h1><p>Upload a file, review its compatibility, and prepare a kernel for browser GPUs.</p></div><a href="/legacy-node/" target="_blank" rel="noreferrer" className="node-link">Open GPU node ↗</a></header>
    <div className="steps"><span>01 Upload Python</span><span>02 Review compatibility</span><span>03 Edit & run kernel</span></div>
    {error && <p role="alert" className="error panel">{error}</p>}
    <div className="workspace">
      <section className="panel"><div className="panel-heading"><h2>01 · Python source</h2><span>{filename}</span></div>
        <label className="upload">Choose a Python file<input type="file" accept=".py,text/x-python" onChange={upload} disabled={busy} /></label>
        <label htmlFor="source">Uploaded source</label><textarea id="source" spellCheck="false" value={source} onChange={e => editSource(e.target.value)} maxLength={32000} />
        <p className="hint">Your source is parsed, never executed. Start with one function that transforms an input list using +, −, or ×.</p>
        <button disabled={busy || !source.trim()} onClick={analyze}>Analyze compatibility</button>
      </section>
      <section className="panel"><h2>02 · Compatibility report</h2>
        {!report ? <p className="empty">Analyze a file to see what can run on the GPU and what needs rewriting.</p> : <><p className="badge">{report.status.replaceAll('_', ' ')}</p><ul className="findings">{report.findings.map((f, i) => <li key={i}><span className={`severity ${f.severity}`}>{f.severity}{f.line ? ` · line ${f.line}` : ''}</span><p>{f.message}</p></li>)}</ul></>}
        <div className="example"><h3>Try Mandelbrot</h3><p>Use a prepared pixel-independent kernel and render the result here.</p><button className="secondary" onClick={loadMandelbrot} disabled={busy}>Load Mandelbrot kernel</button></div>
      </section>
    </div>
    {report && <section className="panel"><h2>03 · Editable kernel preview</h2><p>Review the proposed computation. Compilation checks syntax and supported features; it does not prove equivalence with the original Python.</p>
      <label htmlFor="kernel">GPU kernel</label><textarea id="kernel" spellCheck="false" value={kernel} onChange={e => editKernel(e.target.value)} maxLength={32000} placeholder="Write a compatible kernel here, or load the Mandelbrot example." />
      <button disabled={busy || !kernel.trim()} onClick={validate}>Validate kernel</button>
      {validation && <p role="status" className={validation.valid ? 'success' : 'error'}>{validation.valid ? `Kernel compiles · ${validation.mode} · ready to configure` : validation.error}</p>}
      {validation?.valid && <div className="configuration">
        <label>Scalar parameters (JSON object)<textarea className="small" value={parameters} onChange={e => setParameters(e.target.value)} /></label>
        {validation.mode === 'data-slice' ? <label>Input values (JSON array)<textarea className="small" value={input} onChange={e => setInput(e.target.value)} /></label> : <label>Output element count<input type="number" min="1" max="2000000" value={count} onChange={e => setCount(e.target.value)} /></label>}
        <label>Reduction<select value={reduce} onChange={e => setReduce(e.target.value)}><option value="">Full output array</option>{['sum','count','min','max','mean'].map(op => <option key={op}>{op}</option>)}</select></label>
        <label className="checkbox"><input type="checkbox" checked={verify} onChange={e => setVerify(e.target.checked)} /> Verify each chunk on two nodes</label>
        <p className="hint">Open a GPU node tab before submitting; verification needs two nodes. Count reduction counts nonzero outputs. Jobs are limited to 20 million loop iterations.</p>
        <button onClick={submit} disabled={busy || ['queued','running'].includes(job?.status)}>Submit GPU job</button>
      </div>}
    </section>}
    <Result job={job} imageSize={imageSize} />
  </main>
}
