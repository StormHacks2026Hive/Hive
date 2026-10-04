import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { localDateTime } from '../automation.js'
import '../automation.css'

export default function TimerPanel({ network }) {
  const [tasks, setTasks] = useState([]), [timers, setTimers] = useState([]), [taskId, setTaskId] = useState('')
  const [name, setName] = useState(''), [repeat, setRepeat] = useState('every'), [minutes, setMinutes] = useState(5)
  const [first, setFirst] = useState(localDateTime), [busy, setBusy] = useState(false), [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const base = `/api/networks/${network.id}`
  useEffect(() => {
    let active = true
    async function load() {
      try {
        const [rows, schedules] = await Promise.all([api(`${base}/tasks`), api(`${base}/timers`)])
        if (active) { setTasks(rows); setTimers(schedules) }
      } catch (e) { if (active) setError(e.message) }
    }
    load(); const timer = setInterval(load, 5000)
    return () => { active = false; clearInterval(timer) }
  }, [base])
  const task = tasks.find(t => t.id === taskId) || tasks.find(t => t.repeatable)
  async function refresh() { setTimers(await api(`${base}/timers`)) }
  async function create(e) {
    e.preventDefault(); setBusy(true); setError(''); setNotice('')
    try {
      const firstRun = new Date(first).getTime() / 1000
      if (!Number.isFinite(firstRun)) throw Error('Choose a start time.')
      await api(`${base}/timers`, { run_id: task.id, name: name.trim() || `${task.name} timer`, interval_seconds: repeat === 'once' ? 0 : Math.round(Number(minutes) * 60), first_run: firstRun })
      await refresh(); setNotice('Timer saved'); setName('')
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  async function control(id, action) {
    setBusy(true); setError(''); setNotice('')
    try { await api(`${base}/timers/${id}/control`, { action }); await refresh(); setNotice(action === 'run' ? 'Whole task sent' : action === 'delete' ? 'Timer removed' : action === 'pause' ? 'Timer paused' : 'Timer resumed') }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  return <div className="compute-panel automation-panel">
    <p className="compute-hint">Run a saved whole task once or on repeat, with the same code and inputs.</p>
    {error && <p className="form-error" role="alert">{error}</p>}
    {notice && <p className="automation-notice" role="status">{notice}</p>}
    <div className="automation-columns">
      <section className="panel work-editor">
        <div className="panel-header"><h2>Set a timer</h2></div>
        {task ? <form className="automation-form" onSubmit={create}>
          <label>Saved task<select value={task.id} onChange={e => setTaskId(e.target.value)}>{tasks.map(t => <option key={t.id} value={t.id} disabled={!t.repeatable}>{t.name} · {new Date(t.created * 1000).toLocaleString()}{t.repeatable ? '' : ' · expired inputs'}</option>)}</select></label>
          <p className="task-description">{task.description}</p>
          <ul className="timer-segments">{task.jobs.map(j => <li key={j.job_id}>{j.name}<span>{j.target.toUpperCase()}</span></li>)}</ul>
          <label>Timer name<input value={name} onChange={e => setName(e.target.value)} placeholder={`${task.name} timer`} maxLength={80} /></label>
          <div className="compute-fields"><label>Run<select value={repeat} onChange={e => setRepeat(e.target.value)}><option value="every">On repeat</option><option value="once">Once</option></select></label>{repeat === 'every' && <label>Every (minutes)<input type="number" min="1" max="525600" value={minutes} onChange={e => setMinutes(e.target.value)} required /></label>}</div>
          <label>First run · your local time<input type="datetime-local" value={first} onChange={e => setFirst(e.target.value)} required /></label>
          <button className="button button-primary" disabled={busy}>Save timer</button>
        </form> : <p className="compute-hint">Send a task from Compute first. Its whole run will appear here.</p>}
        <p className="compute-hint">Timers survive refresh and server restart. The server must be running, with contributors connected. Missed runs don’t pile up.</p>
      </section>
      <section className="panel work-editor">
        <div className="panel-header"><h2>Your timers</h2><span className="device-count">{timers.length}</span></div>
        {!timers.length && <p className="compute-hint">No timers yet.</p>}
        <ul className="timer-cards">{timers.map(timer => <li key={timer.id}>
          <div className="panel-header"><strong>{timer.name}</strong><span className="status-pill">{timer.enabled ? 'Scheduled' : timer.interval_seconds === 0 && timer.last_fired && timer.next_run === 0 ? 'Finished' : 'Paused'}</span></div>
          <p>{timer.task_name} · {timer.interval_seconds ? `every ${timer.interval_seconds / 60} min` : 'once'}</p>
          {timer.enabled && <p>Next: <time dateTime={new Date(timer.next_run * 1000).toISOString()}>{new Date(timer.next_run * 1000).toLocaleString()}</time></p>}
          {timer.last_fired && <p>Last sent: {new Date(timer.last_fired * 1000).toLocaleString()}</p>}
          {timer.last_error && <p className="compute-hint">{timer.last_error}</p>}
          {timer.jobs.length > 0 && <ul className="timer-segments">{timer.jobs.map(j => <li key={j.job_id}>{j.name}<span>{j.status}</span></li>)}</ul>}
          {timer.can_control && <div className="compute-actions"><button className="button button-secondary" disabled={busy} onClick={() => control(timer.id, 'run')}>Run now</button><button className="text-button" disabled={busy} onClick={() => control(timer.id, timer.enabled ? 'pause' : 'resume')}>{timer.enabled ? 'Pause' : 'Resume'}</button><button className="text-button danger-text" disabled={busy} onClick={() => control(timer.id, 'delete')}>Remove</button></div>}
        </li>)}</ul>
      </section>
    </div>
  </div>
}
