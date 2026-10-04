export default function NodeWork({ node, local }) {
  const work = node?.work
  const phase = { working: 'Computing', receiving: 'Receiving', readback: 'Sending results', idle: 'Waiting for work', paused: 'Paused', stopped: 'Stopped', connecting: 'Connecting', disconnected: 'Reconnecting', completed: 'Completed' }[local?.state] || 'Connecting'
  return <section className="node-work" aria-labelledby="node-work-title">
    <h3 id="node-work-title">Node work</h3>
    <dl>
      <div><dt>Activity</dt><dd>{phase}</dd></div>
      <div><dt>Task</dt><dd>{work?.kind || '—'}</dd></div>
      <div><dt>Chunk</dt><dd>{work?.chunk_id || '—'}</dd></div>
      <div><dt>Device</dt><dd>{node?.gpu || (local?.capabilities?.webgpu ? 'WebGPU' : 'Checking…')}</dd></div>
      <div><dt>Completed</dt><dd>{node?.completed || 0} chunks</dd></div>
    </dl>
    {local?.error && <p className="form-error">{local.error}</p>}
  </section>
}
