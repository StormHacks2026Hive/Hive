import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { jobCalls, shellQuote } from '../automation.js'
import '../automation.css'

const DEFAULT_REQUEST = JSON.stringify({ filename: 'api-task.py', source: 'def sum_squares(count):\n    result = 0\n    for i in range(count):\n        result += i * i\n    return result\n', target: 'cpu', count: 10 }, null, 2)

function Snippet({ title, text, onCopy }) {
  if (!text) return null
  return <div className="api-snippet"><div><strong>{title}</strong><button className="text-button" onClick={() => onCopy(text)} aria-label={`Copy ${title}`}>Copy</button></div><pre>{text}</pre></div>
}

export default function ApiPanel({ network }) {
  const [keys, setKeys] = useState([]), [tasks, setTasks] = useState([])
  const [name, setName] = useState('My connection'), [permission, setPermission] = useState('read'), [days, setDays] = useState(30)
  const [secret, setSecret] = useState(null), [taskId, setTaskId] = useState(''), [jobId, setJobId] = useState('')
  const [connectionKey, setConnectionKey] = useState(''), [requestBody, setRequestBody] = useState(DEFAULT_REQUEST)
  const [error, setError] = useState(''), [notice, setNotice] = useState(''), [busy, setBusy] = useState(false)
  const base = `/api/networks/${network.id}`
  useEffect(() => {
    let active = true
    async function load() {
      try {
        const [connections, rows] = await Promise.all([api(`${base}/api-keys`), api(`${base}/tasks`)])
        if (active) { setKeys(connections); setTasks(rows) }
      } catch (e) { if (active) setError(e.message) }
    }
    load(); const timer = setInterval(load, 5000)
    return () => { active = false; clearInterval(timer) }
  }, [base])
  const task = tasks.find(t => t.id === taskId) || tasks[0]
  const job = task?.jobs.find(j => j.job_id === jobId) || task?.jobs[0]
  const calls = jobCalls(window.location.origin, network.id, task, job, connectionKey)
  let submissionPath = 'runs'
  try { if (JSON.parse(requestBody).kind === 'onnx') submissionPath = 'onnx' } catch { /* editing JSON */ }
  const submitCall = `curl --fail-with-body -X POST -H "Authorization: Bearer $HIVE_API_KEY" -H "Content-Type: application/json" ${shellQuote(`${window.location.origin}/api/v1/networks/${network.id}/${submissionPath}`)} --data ${shellQuote(requestBody)}`
  async function callApi(body) {
    if (!connectionKey.trim()) throw Error('Create or paste a connection key first.')
    const response = await fetch(`/api/v1/networks/${network.id}/${body?.kind === 'onnx' ? 'onnx' : 'runs'}`, {
      headers: { Authorization: `Bearer ${connectionKey.trim()}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
      ...(body ? { method: 'POST', body: JSON.stringify(body) } : {}), cache: 'no-store',
    })
    const value = await response.json()
    if (!response.ok) throw Error(typeof value.detail === 'string' ? value.detail : 'Check the request fields.')
    return value
  }
  async function testConnection() {
    setBusy(true); setError(''); setNotice('')
    try { await callApi(); setNotice('Connected to this hive') }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  async function send() {
    setBusy(true); setError(''); setNotice('')
    try {
      const value = await callApi(JSON.parse(requestBody))
      setTasks(await api(`${base}/tasks`)); setTaskId(value.id); setJobId(''); setNotice('Task sent through API')
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); setNotice('Copied') }
    catch { setError('Select the text to copy it.') }
  }
  async function create(e) {
    e.preventDefault(); setBusy(true); setError(''); setNotice('')
    try {
      const value = await api(`${base}/api-keys`, { name, permission, expires_days: Number(days) })
      setSecret(value); setConnectionKey(value.key); setKeys(await api(`${base}/api-keys`))
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  async function revoke(id) {
    setBusy(true); setError('')
    try {
      await api(`${base}/api-keys/${id}/revoke`, {})
      if (secret?.id === id) { if (connectionKey === secret.key) setConnectionKey(''); setSecret(null) }
      setKeys(await api(`${base}/api-keys`)); setNotice('Connection revoked')
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  return <div className="compute-panel automation-panel">
    <p className="compute-hint">Connect an app to this hive. Read results or send work with a key.</p>
    {error && <p className="form-error" role="alert">{error}</p>}
    {notice && <p className="automation-notice" role="status">{notice}</p>}
    <div className="automation-columns">
      <section className="panel work-editor">
        <div className="panel-header"><h2>API connection</h2></div>
        <form className="automation-form" onSubmit={create}>
          <label>Connection name<input value={name} onChange={e => setName(e.target.value)} required maxLength={64} /></label>
          <div className="compute-fields"><label>Access<select value={permission} onChange={e => setPermission(e.target.value)}><option value="read">Read results</option><option value="run">Read + submit work</option></select></label><label>Expires in days<input type="number" min="1" max="365" value={days} onChange={e => setDays(e.target.value)} required /></label></div>
          <button className="button button-primary" disabled={busy}>Create API key</button>
        </form>
        {secret && <div className="api-secret"><label>New API key<input readOnly value={secret.key} onFocus={e => e.target.select()} /></label><p>Copy it now. It is only shown here once.</p><button className="text-button" onClick={() => copy(secret.key)}>Copy API key</button></div>}
        <ul className="automation-list">{keys.map(key => <li key={key.id}><div><strong>{key.name}</strong><p>{key.permission === 'run' ? 'Read + submit' : 'Read only'} · expires {new Date(key.expires * 1000).toLocaleDateString()}</p></div><button className="text-button danger-text" disabled={busy} onClick={() => revoke(key.id)}>Revoke</button></li>)}</ul>
        <p className="compute-hint">Keys only access this network. Revoking a key stops new API calls.</p>
        <label>Connection key<input type="password" autoComplete="off" value={connectionKey} onChange={e => setConnectionKey(e.target.value)} placeholder="Paste an existing key" /></label>
        <button className="button button-secondary" disabled={busy} onClick={testConnection}>Test connection</button>
      </section>
      <section className="panel work-editor">
        <div className="panel-header"><h2>Your tasks</h2></div>
        {task ? <>
          <label>Task<select value={task.id} onChange={e => { setTaskId(e.target.value); setJobId('') }}>{tasks.map(t => <option key={t.id} value={t.id}>{t.name} · {new Date(t.created * 1000).toLocaleString()}</option>)}</select></label>
          <p className="task-description">{task.description}</p>
          <label>Job<select value={job?.job_id || ''} onChange={e => setJobId(e.target.value)}>{task.jobs.map(j => <option key={j.job_id} value={j.job_id}>{j.name} · {j.target.toUpperCase()} · {j.status}</option>)}</select></label>
          {job && <div className="api-job-summary"><strong>{job.name}</strong><span className="status-pill">{job.status}</span><p>{job.description}</p><p>{job.target.toUpperCase()} · {job.kind || 'Retained history'}{job.output_shape.length > 0 ? ` · output ${job.output_shape.join(' × ')}` : ''}</p><code>{job.job_id}</code></div>}
          {!task.repeatable && <p className="compute-hint">Send this older task again to save its inputs for repeat calls.</p>}
        </> : <p className="compute-hint">Send a task from Compute to get its API calls here.</p>}
      </section>
    </div>
    <section className="panel work-editor"><details><summary>Send a task through API</summary><div className="automation-form">
      <label>Request JSON<textarea className="code-editor" value={requestBody} onChange={e => setRequestBody(e.target.value)} spellCheck={false} /></label>
      <p className="compute-hint">Uses your connection key. The example computes a sum of squares; count 10 returns 285.</p>
      <button className="button button-primary" disabled={busy} onClick={send}>Send through API</button>
      <Snippet title="Submit request" text={submitCall} onCopy={copy} />
    </div></details></section>
    <section className="panel api-calls"><div className="panel-header"><h2>Copy & connect</h2><span className="status-pill">Bearer key</span></div>
      <Snippet title="Set your key" text={calls.auth} onCopy={copy} />
      <Snippet title="Check job" text={calls.status} onCopy={copy} />
      <Snippet title="Get result" text={calls.result} onCopy={copy} />
      {task?.repeatable && <><p className="compute-hint">Repeating sends every segment with the saved inputs. Use a key with submit access.</p><Snippet title="Repeat whole task" text={calls.repeat} onCopy={copy} /><details><summary>Python example · repeat and wait for results</summary><Snippet title="Python example" text={calls.python} onCopy={copy} /></details></>}
      <p className="compute-hint">Numeric results are JSON. Animation results are raw RGBA bytes in frame order. Results expire; save what you need.</p>
    </section>
  </div>
}
