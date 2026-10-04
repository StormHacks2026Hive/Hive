import { useEffect, useRef, useState } from 'react'
import { api, networkApi } from '../api.js'
import JobResults, { saveFile } from './JobResults.jsx'

const SAMPLE = `def transform(values):\n    result = [0.0] * len(values)\n    for i in range(len(values)):\n        result[i] = values[i] * 2 + 1\n    return result\n`
const SHADER = `@group(0) @binding(0) var<storage, read> values: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    result[gid.x] = values[gid.x] * 2.0 + 1.0;\n}\n`
function encode(bytes) { let text = ''; for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192)); return btoa(text) }
function typedInput(text, dtype) {
  const values = JSON.parse(text)
  if (!Array.isArray(values) || !values.length || values.length > 500000 || values.some(v => typeof v !== 'number' || !Number.isFinite(v))) throw Error('Enter a JSON array of 1–500,000 finite numbers')
  const bytes = new ArrayBuffer(values.length * 4), view = new DataView(bytes)
  const method = { f32: 'setFloat32', i32: 'setInt32', u32: 'setUint32' }[dtype]
  values.forEach((v, i) => {
    if (dtype === 'f32' ? Math.abs(v) > 3.402823e38 : !Number.isInteger(v) || v < (dtype === 'i32' ? -2147483648 : 0) || v > (dtype === 'i32' ? 2147483647 : 4294967295)) throw Error(`Input does not fit ${dtype}`)
    view[method](i * 4, v, true)
  })
  return { dtype, data: encode(new Uint8Array(bytes)) }
}
export default function ComputePanel({ network, nodes, onKillNode, onControl }) {
  const [source, setSource] = useState(SAMPLE), [filename, setFilename] = useState('program.py')
  const [entry, setEntry] = useState('type'), [mode, setMode] = useState('compute'), [segmentation, setSegmentation] = useState('auto')
  const [input, setInput] = useState('[1, 2, 3, 4]'), [useInput, setUseInput] = useState(true), [dtype, setDtype] = useState('f32'), [count, setCount] = useState(8192)
  const [report, setReport] = useState(null), [busy, setBusy] = useState(false), [error, setError] = useState('')
  const [frames, setFrames] = useState(1), [fps, setFps] = useState(12), [width, setWidth] = useState(256), [height, setHeight] = useState(256)
  const [settings, setSettings] = useState('{}'), [frameValues, setFrameValues] = useState(''), [cameraTurn, setCameraTurn] = useState(30)
  const [model, setModel] = useState(''), [shape, setShape] = useState('4, 1'), [batchSize, setBatchSize] = useState(32)
  const [jobs, setJobs] = useState([]), [jobId, setJobId] = useState(''), [history, setHistory] = useState([])
  const revision = useRef(0)
  function edit(update) { revision.current++; setReport(null); setError(''); update() }
  useEffect(() => {
    let active = true
    api(`/api/networks/${network.id}/runs`).then(rows => { if (active) { setHistory(rows); const recent = rows.find(r => r.jobs.some(j => j.status !== 'expired')); if (recent) { setJobs(recent.jobs); setJobId(recent.jobs[0]?.job_id || '') } } }).catch(e => { if (active) setError(e.message) })
    return () => { active = false }
  }, [network.id])
  function request(mark = false) {
    return { source, filename, mode: mode === 'animation' ? 'animation' : 'compute', segmentation, mark,
      ...(mode !== 'animation' && useInput ? { input: typedInput(input, dtype) } : {}), count: Number(count),
      frames: Number(frames), fps: Number(fps), width: Number(width), height: Number(height), settings: JSON.parse(settings), camera_turn: Number(cameraTurn),
      ...(frameValues.trim() ? { frame_values: JSON.parse(frameValues) } : {}) }
  }
  async function upload(file) {
    if (!file) return
    const version = ++revision.current; setReport(null); setError('')
    try {
      const extension = file.name.split('.').pop().toLowerCase()
      if (!['py','wgsl','onnx'].includes(extension)) throw Error('Choose a .py, .wgsl, or .onnx file')
      if (file.size > (extension === 'onnx' ? 4194304 : 65536)) throw Error('File is too large')
      const content = extension === 'onnx' ? encode(new Uint8Array(await file.arrayBuffer())) : await file.text()
      if (version !== revision.current) return
      setFilename(file.name)
      if (extension === 'onnx') { setMode('onnx'); setModel(content) }
      else { setSource(content); if (/texture_storage_2d|WGSL_SHADER/.test(content)) setMode('animation') }
    } catch (e) { if (version === revision.current) setError(e.message) }
  }
  async function analyze(mark = false) {
    setBusy(true); setError(''); const version = revision.current
    try {
      const value = mode === 'onnx' ? await networkApi(network.id, '/pool/onnx/analyze', { model, input_shape: shape.split(',').map(Number), batch_size: Number(batchSize) })
        : await api(`/api/networks/${network.id}/analyze`, request(mark))
      if (version !== revision.current) return
      if (value.marked_source && mark) setSource(value.marked_source)
      setReport(value)
    } catch (e) { if (version === revision.current) setError(e.message) }
    finally { setBusy(false) }
  }
  async function submit() {
    setBusy(true); setError('')
    try {
      if (mode === 'onnx') {
        const created = await networkApi(network.id, '/pool/jobs', { kind: 'onnx', model, input: typedInput(input, 'f32'), input_shape: shape.split(',').map(Number), batch_size: Number(batchSize), independent: true })
        setJobs([{ job_id: created.job_id, name: filename, target: 'gpu' }]); setJobId(created.job_id)
      } else {
        const run = await api(`/api/networks/${network.id}/runs`, request())
        setJobs(run.jobs); setJobId(run.jobs[0].job_id); setReport(run.analysis)
        setHistory(await api(`/api/networks/${network.id}/runs`))
      }
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  return <div className="compute-panel">
    {error && <p className="form-error" role="alert">{error}</p>}
    <div className="compute-columns">
      <section className="work-editor panel">
        <div className="panel-header"><h2>Send work</h2><select aria-label="Work mode" value={mode} onChange={e => edit(() => { setMode(e.target.value); if (e.target.value === 'animation') setSource('') })}><option value="compute">Compute</option><option value="animation">Animation</option><option value="onnx">ONNX</option></select></div>
        <div className="compute-choices" aria-label="Source entry">{['type','file'].map(value => <button key={value} className={`button ${entry === value ? 'button-primary' : 'button-secondary'}`} onClick={() => setEntry(value)}>{value === 'type' ? 'Type code' : 'Upload file'}</button>)}</div>
        {entry === 'file' || mode === 'onnx' ? <label className="file-drop">{filename}<input type="file" accept={mode === 'onnx' ? '.onnx' : '.py,.wgsl'} onChange={e => upload(e.target.files[0])} /></label> : <label>Language<select value={filename.endsWith('.wgsl') ? 'wgsl' : 'py'} onChange={e => edit(() => { setFilename(`program.${e.target.value}`); setSource(e.target.value === 'wgsl' ? SHADER : SAMPLE) })}><option value="py">Python</option><option value="wgsl">WGSL</option></select></label>}
        {mode !== 'onnx' && <><label htmlFor="source-code">Code</label><textarea id="source-code" className="code-editor" spellCheck={false} value={source} onChange={e => edit(() => setSource(e.target.value))} placeholder={mode === 'animation' ? 'Paste a renderer, or leave empty for Mandelbrot' : 'Python or WGSL source'} /></>}
        {mode === 'compute' && <label>Segmentation<select value={segmentation} onChange={e => edit(() => setSegmentation(e.target.value))}><option value="auto">Automatic</option><option value="manual">Use my markers</option></select></label>}
        {mode !== 'animation' ? <>
          <div className="compute-fields"><label className="inline-check"><input type="checkbox" checked={useInput} onChange={e => edit(() => setUseInput(e.target.checked))} />Input array</label><label>Type<select value={dtype} onChange={e => edit(() => setDtype(e.target.value))}><option>f32</option><option>u32</option><option>i32</option></select></label></div>
          {useInput ? <label>Input values<textarea aria-label="Input values" rows={2} value={input} onChange={e => edit(() => setInput(e.target.value))} /></label> : <label>Element count<input type="number" min="1" max="2000000" value={count} onChange={e => edit(() => setCount(e.target.value))} /></label>}
          {mode === 'onnx' && <div className="compute-fields"><label>Input shape<input value={shape} onChange={e => edit(() => setShape(e.target.value))} /></label><label>Batch size<input type="number" min="1" max="256" value={batchSize} onChange={e => edit(() => setBatchSize(e.target.value))} /></label></div>}
        </> : <>
          <div className="compute-fields">{[['Width',width,setWidth,1,4096],['Height',height,setHeight,1,4096],['Frames',frames,setFrames,1,32],['FPS',fps,setFps,1,60]].map(([label,value,set,min,max]) => <label key={label}>{label}<input type="number" min={min} max={max} value={value} onChange={e => edit(() => set(e.target.value))} /></label>)}</div>
          <label>Camera turn (degrees)<input type="number" min="-360" max="360" value={cameraTurn} onChange={e => edit(() => setCameraTurn(e.target.value))} /></label>
          <details><summary>Renderer values</summary><label>Settings (JSON)<textarea value={settings} onChange={e => edit(() => setSettings(e.target.value))} /></label><label>Per-frame values (JSON array)<textarea value={frameValues} onChange={e => edit(() => setFrameValues(e.target.value))} placeholder='[{"camera_position":[3,2,3]},{"camera_position":[-3,2,3]}]' /></label></details>
        </>}
        <div className="compute-actions"><button className="button button-secondary" disabled={busy} onClick={() => analyze()}>Analyze</button>{mode === 'compute' && !filename.endsWith('.wgsl') && <button className="button button-secondary" disabled={busy} onClick={() => analyze(true)}>Mark & analyze</button>}<button className="button button-primary" disabled={busy || report?.status !== 'ready'} onClick={submit}>{busy ? 'Working…' : 'Send'}</button></div>
        {source.includes('# hive:') && <p className="compute-hint">Edit gpu/cpu in the comments, then analyze again.</p>}
      </section>
      <div className="compute-review">
        <section className="partition-panel panel"><h2>Segments</h2>
          {!report ? <p className="compute-hint">Analyze to review the split across connected devices.</p> : <>
            <span className="status-pill">{report.status}</span>
            <ul className="segment-list">{report.segments?.map(s => <li key={s.id}><strong>{s.name}</strong><span>{s.target.toUpperCase()} · lines {s.line}–{s.end_line} · {s.chunk_count || 0} chunks</span>{s.findings.map((f,i) => <p className="form-error" key={i}>{f}</p>)}{s.wgsl && <details><summary>WGSL</summary><pre>{s.wgsl}</pre><button className="text-button" onClick={() => saveFile(new Blob([s.wgsl], { type: 'text/plain' }), `${s.name}.wgsl`)}>Download shader</button></details>}</li>)}</ul>
            {report.output_shape && <p>Output shape: {report.output_shape.join(' × ')}</p>}
            <details><summary>Analysis notes</summary>{report.findings?.map((f,i) => <p className="compute-hint" key={i}>{f}</p>)}</details>
          </>}
        </section>
        <section className="contributors-panel panel"><h2>Nodes</h2><ul className="contribution-list">{nodes.map(n => <li key={n.id}><div><strong>{n.name}</strong><span>{n.status} · {Math.round(n.weight * 100)}% GPU share</span><small>{n.capabilities.adapter?.description || n.capabilities.adapter?.vendor || 'CPU'} · {n.completed} chunks</small>{n.work && <small>{n.work.kind} · {n.work.chunk_id} · {n.work.count} items</small>}</div>{n.can_control && <div className="node-task-actions">{n.work && <button className="text-button" onClick={() => onControl(n.id, 'cancel')}>Cancel task</button>}<button className="text-button danger-text" disabled={n.status === 'offline'} onClick={() => onKillNode(n.id)}>Kill node</button></div>}</li>)}</ul></section>
      </div>
    </div>
    {jobs.length > 0 && <label className="job-picker">Result<select value={jobId} onChange={e => setJobId(e.target.value)}>{jobs.map(j => <option key={j.job_id} value={j.job_id}>{j.name} · {j.target?.toUpperCase()}</option>)}</select></label>}
    <JobResults jobId={jobId} onError={setError} />
    {history.length > 0 && <details className="run-history"><summary>Recent submissions</summary>{history.map(run => <button key={run.id} className="text-button" disabled={run.jobs.every(j => j.status === 'expired')} onClick={() => { setJobs(run.jobs); setJobId(run.jobs.find(j => j.status !== 'expired')?.job_id || '') }}>{run.name} · {new Date(run.created * 1000).toLocaleString()}</button>)}</details>}
  </div>
}
