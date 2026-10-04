import { useEffect, useRef, useState } from 'react'
import { api, networkApi } from '../api.js'
import JobResults, { saveFile } from './JobResults.jsx'
import { nodeSpecs } from '../nodeSpecs.js'
import { HexIcon } from './HiveScene.jsx'
import { sourceType } from '../sourceType.js'
import '../compute.css'

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
  const [target, setTarget] = useState('auto'), [example, setExample] = useState('python')
  const [input, setInput] = useState('[1, 2, 3, 4]'), [useInput, setUseInput] = useState(true), [dtype, setDtype] = useState('f32'), [count, setCount] = useState(8192)
  const [report, setReport] = useState(null), [busy, setBusy] = useState(false), [error, setError] = useState('')
  const [loadingFile, setLoadingFile] = useState(false)
  const [frames, setFrames] = useState(1), [fps, setFps] = useState(12), [width, setWidth] = useState(256), [height, setHeight] = useState(256)
  const [settings, setSettings] = useState('{}'), [frameValues, setFrameValues] = useState(''), [cameraTurn, setCameraTurn] = useState(30)
  const [model, setModel] = useState(''), [shape, setShape] = useState('4, 1'), [batchSize, setBatchSize] = useState(32)
  const [jobs, setJobs] = useState([]), [jobId, setJobId] = useState(''), [history, setHistory] = useState([])
  const revision = useRef(0)
  const uploadRevision = useRef(0)
  const working = busy || loadingFile
  function edit(update) { revision.current++; setReport(null); setError(''); update() }
  function editSource(update) { uploadRevision.current++; setLoadingFile(false); edit(update) }
  function updateSource(value) {
    editSource(() => {
      setSource(value)
      const type = sourceType(value)
      if (type.rawWgsl && !filename.toLowerCase().endsWith('.wgsl')) setFilename('program.wgsl')
      if (type.image) { setMode('animation'); if (Number(frames) === 1) setFrames(8) }
    })
  }
  useEffect(() => {
    let active = true
    api(`/api/networks/${network.id}/runs`).then(rows => { if (active) { setHistory(rows); const recent = rows.find(r => r.jobs.some(j => j.status !== 'expired')); if (recent) { setJobs(recent.jobs); setJobId(recent.jobs[0]?.job_id || '') } } }).catch(e => { if (active) setError(e.message) })
    return () => { active = false }
  }, [network.id])
  function request(mark = false) {
    return { source, filename, mode: mode === 'animation' ? 'animation' : 'compute', segmentation, target, mark,
      ...(mode !== 'animation' && useInput ? { input: typedInput(input, dtype) } : {}), count: Number(count),
      frames: Number(frames), fps: Number(fps), width: Number(width), height: Number(height), settings: JSON.parse(settings), camera_turn: Number(cameraTurn),
      ...(frameValues.trim() ? { frame_values: JSON.parse(frameValues) } : {}) }
  }
  async function chooseExample(value) {
    uploadRevision.current++; setLoadingFile(false)
    const version = ++revision.current
    setReport(null); setError(''); setExample(value); setBusy(true)
    try {
      const asset = value === 'mandelbulb' ? '/examples/Mandelbulb.wgsl' : value === 'onnx' ? '/examples/dense.onnx' : null
      let content = ''
      if (asset) {
        const response = await fetch(asset)
        if (!response.ok) throw Error('Could not load the example')
        content = value === 'onnx' ? encode(new Uint8Array(await response.arrayBuffer())) : await response.text()
      }
      if (version !== revision.current) return
      setEntry('type'); setTarget('auto'); setSegmentation('auto'); setDtype('f32'); setUseInput(true)
      if (value === 'onnx') {
        setMode('onnx'); setFilename('dense.onnx'); setModel(content); setInput('[2, 3, 4, 5, 6, 7, 8, 9]'); setShape('4, 2'); setBatchSize(2)
      } else {
        setModel(''); setMode(['mandelbulb', 'mandelbrot'].includes(value) ? 'animation' : 'compute')
        setSource(value === 'mandelbulb' ? content : value === 'mandelbrot' ? '' : value === 'wgsl' ? SHADER : SAMPLE)
        setFilename(value === 'mandelbulb' ? 'Mandelbulb.wgsl' : value === 'wgsl' ? 'program.wgsl' : 'program.py')
        setInput('[1, 2, 3, 4]'); setFrames(8); setFps(8); setWidth(256); setHeight(256); setCameraTurn(90); setSettings('{}'); setFrameValues('')
      }
    } catch (e) { if (version === revision.current) setError(e.message) }
    finally { setBusy(false) }
  }
  async function upload(file) {
    if (!file) return
    const version = ++uploadRevision.current
    revision.current++; setReport(null); setError(''); setLoadingFile(true)
    try {
      const extension = file.name.split('.').pop().toLowerCase()
      if (!['py','wgsl','onnx'].includes(extension)) throw Error('Choose a .py, .wgsl, or .onnx file')
      if (file.size > (extension === 'onnx' ? 4194304 : 65536)) throw Error('File is too large')
      const content = extension === 'onnx' ? encode(new Uint8Array(await file.arrayBuffer())) : await file.text()
      if (version !== uploadRevision.current) return
      revision.current++; setReport(null)
      setFilename(file.name)
      if (extension === 'onnx') { setMode('onnx'); setModel(content) }
      else {
        const type = sourceType(content)
        setModel(''); setSource(content); setMode(type.image ? 'animation' : 'compute')
        if (type.image) setFrames(value => Number(value) === 1 ? 8 : value)
      }
    } catch (e) { if (version === uploadRevision.current) setError(e.message) }
    finally { if (version === uploadRevision.current) setLoadingFile(false) }
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
    <div className="compute-intro"><p>Give your hive a task. Review the split, then put your connected devices to work.</p><span className="compute-online"><i />{nodes.filter(n => !['offline', 'paused'].includes(n.status)).length} devices available</span></div>
    {error && <p className="form-error" role="alert">{error}</p>}
    <div className="compute-columns">
      <section className="work-editor panel">
        <div className="panel-header"><div><span className="section-kicker">01 / YOUR TASK</span><h2>Prepare your work</h2></div><select aria-label="Work mode" value={mode} disabled={working} onChange={e => { const next = e.target.value; if (next === 'animation' && !/texture_storage_2d|WGSL_SHADER/.test(source)) chooseExample('mandelbulb'); else if (next === 'onnx' && !model) chooseExample('onnx'); else if (next === 'compute' && mode === 'onnx') chooseExample('python'); else edit(() => { setMode(next); if (next === 'animation' && Number(frames) === 1) setFrames(8) }) }}><option value="compute">Compute</option><option value="animation">Animation</option><option value="onnx">ONNX</option></select></div>
        <label>Example<select value={example} disabled={working} onChange={e => chooseExample(e.target.value)}><option value="python">Python array loop</option><option value="wgsl">WGSL array loop</option><option value="mandelbulb">Mandelbulb animation</option><option value="mandelbrot">Mandelbrot animation</option><option value="onnx">ONNX · dense layer</option></select></label>
        <div className="compute-choices" aria-label="Source entry">{['type','file'].map(value => <button key={value} aria-pressed={entry === value} className={`button ${entry === value ? 'button-primary' : 'button-secondary'}`} onClick={() => setEntry(value)}>{value === 'type' ? 'Type code' : 'Upload file'}</button>)}</div>
        {entry === 'file' || mode === 'onnx' ? <label className="file-drop"><span>Choose a file for your hive</span><small>{mode === 'onnx' ? 'ONNX model' : 'Python or WGSL'} · {filename}</small><input type="file" accept={mode === 'onnx' ? '.onnx' : '.py,.wgsl'} onChange={e => { const file = e.currentTarget.files[0]; e.currentTarget.value = ''; upload(file) }} /></label> : <label>Language<select value={filename.endsWith('.wgsl') ? 'wgsl' : 'py'} onChange={e => editSource(() => { setFilename(`program.${e.target.value}`); setSource(e.target.value === 'wgsl' ? SHADER : SAMPLE) })}><option value="py">Python</option><option value="wgsl">WGSL</option></select></label>}
        {mode !== 'onnx' && <div className="source-field"><div className="source-caption"><label htmlFor="source-code">Source code</label><span>{filename}</span></div><textarea id="source-code" className="code-editor" spellCheck={false} value={source} onChange={e => updateSource(e.target.value)} placeholder={mode === 'animation' ? 'Paste a renderer, or leave empty for Mandelbrot' : 'Python or WGSL source'} /></div>}
        {mode === 'compute' && <label>Segmentation<select value={segmentation} onChange={e => edit(() => setSegmentation(e.target.value))}><option value="auto">Automatic</option><option value="manual">Use my markers</option></select></label>}
        {mode === 'compute' && !filename.endsWith('.wgsl') && <label>Python target<select value={target} onChange={e => edit(() => setTarget(e.target.value))}><option value="auto">Auto / comment targets</option><option value="gpu">GPU</option><option value="cpu">CPU</option></select></label>}
        {mode !== 'animation' ? <>
          <div className="compute-fields"><label className="inline-check"><input type="checkbox" checked={useInput} onChange={e => edit(() => setUseInput(e.target.checked))} />Input array</label><label>Type<select value={dtype} onChange={e => edit(() => setDtype(e.target.value))}><option>f32</option><option>u32</option><option>i32</option></select></label></div>
          {useInput ? <label>Input values<textarea aria-label="Input values" rows={2} value={input} onChange={e => edit(() => setInput(e.target.value))} /></label> : <label>Element count<input type="number" min="1" max="2000000" value={count} onChange={e => edit(() => setCount(e.target.value))} /></label>}
          {mode === 'onnx' && <div className="compute-fields"><label>Input shape<input value={shape} onChange={e => edit(() => setShape(e.target.value))} /></label><label>Batch size<input type="number" min="1" max="256" value={batchSize} onChange={e => edit(() => setBatchSize(e.target.value))} /></label></div>}
        </> : <>
          <div className="compute-fields">{[['Width',width,setWidth,1,4096],['Height',height,setHeight,1,4096],['Frames',frames,setFrames,1,32],['FPS',fps,setFps,1,60]].map(([label,value,set,min,max]) => <label key={label}>{label}<input type="number" min={min} max={max} value={value} onChange={e => edit(() => set(e.target.value))} /></label>)}</div>
          <label>Camera turn (degrees)<input type="number" min="-360" max="360" value={cameraTurn} onChange={e => edit(() => setCameraTurn(e.target.value))} /></label>
          <details><summary>Renderer values</summary><label>Settings (JSON)<textarea value={settings} onChange={e => edit(() => setSettings(e.target.value))} /></label><label>Per-frame values (JSON array)<textarea value={frameValues} onChange={e => edit(() => setFrameValues(e.target.value))} placeholder='[{"camera_position":[3,2,3]},{"camera_position":[-3,2,3]}]' /></label></details>
        </>}
        <div className="compute-actions"><button className="button button-secondary" disabled={working} onClick={() => analyze()}>Analyze</button>{mode === 'compute' && !filename.endsWith('.wgsl') && <button className="button button-secondary" disabled={working} onClick={() => analyze(true)}>Mark & analyze</button>}<button className="button button-primary" disabled={working || report?.status !== 'ready'} onClick={submit}>{loadingFile ? 'Reading file…' : busy ? 'Working…' : 'Send'}</button></div>
        {mode === 'animation' && <button className="text-button" onClick={() => editSource(() => { setSource(''); setFilename('animation.py') })}>Use built-in Mandelbrot</button>}
        {source.includes('# hive:') && <p className="compute-hint">Edit gpu/cpu in the comments, then analyze again.</p>}
      </section>
      <div className="compute-review">
        <section className="partition-panel panel"><div className="panel-header"><div><span className="section-kicker">02 / REVIEW THE SPLIT</span><h2>Segments</h2></div>{report && <span className="status-pill" data-state={report.status}>{report.status}</span>}</div>
          {!report ? <div className="compute-empty"><HexIcon /><strong>A plan before the work</strong><p>Analyze your code to see its CPU and GPU segments and how work will be shared.</p></div> : <>
            <ul className="segment-list">{report.segments?.map(s => <li key={s.id}><strong>{s.name}</strong><span>{s.target.toUpperCase()} · lines {s.line}–{s.end_line} · {s.chunk_count || 0} chunks</span>{s.allocations && Object.keys(s.allocations).length > 0 && <p className="compute-hint">Planned split: {Object.entries(s.allocations).map(([id,count]) => `${report.contributors?.find(n => n.worker_id === id)?.name || 'Contributor'} ${Math.round(count / Object.values(s.allocations).reduce((a,b) => a+b,0) * 100)}%`).join(' · ')}</p>}{s.findings.map((f,i) => <p className="form-error" key={i}>{f}</p>)}{s.wgsl && <details><summary>WGSL</summary><pre>{s.wgsl}</pre><button className="text-button" onClick={() => saveFile(new Blob([s.wgsl], { type: 'text/plain' }), `${s.name}.wgsl`)}>Download shader</button></details>}</li>)}</ul>
            {report.output_shape && <p>Output shape: {report.output_shape.join(' × ')}</p>}
            <details><summary>Analysis notes</summary>{report.findings?.map((f,i) => <p className="compute-hint" key={i}>{f}</p>)}</details>
          </>}
        </section>
        <section className="contributors-panel panel">
          <div className="panel-header"><div><span className="section-kicker">YOUR HIVE</span><h2>Connected devices</h2></div><span className="device-count">{nodes.length}</span></div>
          <p className="compute-hint">GPU shares favor device specs; CPU shares follow measured performance.</p>
          <ul className="contribution-list node-cards">{nodes.map(n => <li key={n.id}>
            <div className="node-card-heading"><strong>{n.name}</strong><span className="status-pill" data-state={n.status}>{n.status}</span></div>
            <div className="node-shares"><div><span>GPU share</span><strong>{Math.round(n.weight * 100)}%</strong><progress aria-label={`${n.name} GPU share`} max="1" value={n.weight} /></div><div><span>CPU share</span><strong>{Math.round((n.cpu_weight || 0) * 100)}%</strong><progress aria-label={`${n.name} CPU share`} max="1" value={n.cpu_weight || 0} /></div></div>
            <p className="node-card-meta">{n.capabilities.adapter?.description || n.capabilities.adapter?.vendor || 'CPU'}<span>{n.completed} chunks completed</span></p>
            {n.work && <p className="node-task-detail">{n.work.kind} · {n.work.count} items</p>}
            <div className="node-card-footer"><details><summary>Device specs</summary><ul>{nodeSpecs(n.capabilities).map(spec => <li key={spec}>{spec}</li>)}</ul></details>{n.can_control && <div className="node-task-actions">{n.work && <button className="text-button" onClick={() => onControl(n.id, 'cancel')}>Cancel task</button>}<button className="text-button danger-text" disabled={n.status === 'offline'} onClick={() => onKillNode(n.id)}>Kill node</button></div>}</div>
          </li>)}</ul>
          {!nodes.length && <p className="compute-hint">Devices will appear here when they join your hive.</p>}
        </section>
      </div>
    </div>
    {jobs.length > 0 && <label className="job-picker">Result<select value={jobId} onChange={e => setJobId(e.target.value)}>{jobs.map(j => <option key={j.job_id} value={j.job_id}>{j.name} · {j.target?.toUpperCase()}</option>)}</select></label>}
    <JobResults key={jobId} jobId={jobId} onError={setError} />
    {history.length > 0 && <details className="run-history"><summary>Recent submissions</summary>{history.map(run => <button key={run.id} className="text-button" disabled={run.jobs.every(j => j.status === 'expired')} onClick={() => { setJobs(run.jobs); setJobId(run.jobs.find(j => j.status !== 'expired')?.job_id || '') }}>{run.name} · {new Date(run.created * 1000).toLocaleString()}</button>)}</details>}
  </div>
}
