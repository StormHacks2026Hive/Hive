import { useEffect, useState } from 'react'
import { formatData } from '../network.js'

const work = {
  receiving: 'Receiving preview data',
  sending: 'Sending preview data',
  idle: 'Waiting for work',
  paused: 'Paused',
  offline: 'Stopped',
}

export default function NodeWork({ node }) {
  const [gpu, setGpu] = useState('Checking…')

  useEffect(() => {
    let active = true
    async function checkGPU() {
      try {
        const adapter = await navigator.gpu?.requestAdapter()
        if (active) setGpu(adapter ? 'Ready' : 'Unavailable')
      } catch {
        if (active) setGpu('Unavailable')
      }
    }
    checkGPU()
    return () => { active = false }
  }, [])

  return (
    <section className="node-work" aria-labelledby="node-work-title">
      <h3 id="node-work-title">Node work</h3>
      <dl>
        <div><dt>Activity</dt><dd>{work[node.status]}</dd></div>
        <div><dt>Input</dt><dd>{node.status === 'receiving' ? 'Preview buffer' : '—'}</dd></div>
        <div><dt>Received</dt><dd>{formatData(node.received)}</dd></div>
        <div><dt>WebGPU</dt><dd>{gpu}</dd></div>
        <div><dt>Kernel</dt><dd>No kernel running</dd></div>
      </dl>
    </section>
  )
}
