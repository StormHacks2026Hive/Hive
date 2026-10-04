import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'

function deviceKey() {
  try {
    const key = localStorage.getItem('hive.device') || crypto.randomUUID()
    localStorage.setItem('hive.device', key)
    return key
  } catch { return crypto.randomUUID() }
}
function deviceType() {
  if (/iPad|Tablet/i.test(navigator.userAgent) || /Macintosh/i.test(navigator.userAgent) && navigator.maxTouchPoints > 1) return 'tablet'
  if (/Mobi|Android|iPhone/i.test(navigator.userAgent)) return 'phone'
  return 'laptop'
}
export default function useNetwork(user, initialNetwork) {
  const [network, setNetwork] = useState(null), [knownNetworks, setKnownNetworks] = useState([])
  const [snapshot, setSnapshot] = useState({ nodes: [], jobs: [] }), [ownId, setOwnId] = useState(null)
  const [workerState, setWorkerState] = useState({ state: 'connecting' }), [error, setError] = useState('')
  const [ready, setReady] = useState(false)
  const [device] = useState(deviceKey)
  const worker = useRef(null), generation = useRef(0)
  const refresh = useCallback(async () => {
    if (!network) return
    const current = await api(`/api/networks/${network.id}`)
    setSnapshot(current)
    return current
  }, [network])
  useEffect(() => {
    let active = true
    api('/api/networks').then(items => {
      if (!active) return
      setKnownNetworks(items)
      setNetwork(items.find(n => n.id === initialNetwork?.id) || null)
      setReady(true)
    }).catch(e => { if (active) { setError(e.message); setReady(true) } })
    return () => { active = false }
  }, [initialNetwork?.id])
  useEffect(() => {
    if (!network) return
    const version = ++generation.current
    let closed = false, timer, wake, enrolled, starting = false
    setSnapshot({ nodes: [], jobs: [] }); setOwnId(null); setWorkerState({ state: 'connecting' })
    function stopWorker() { worker.current?.postMessage({ type: 'stop' }); worker.current?.terminate(); worker.current = null; wake?.release().catch(() => {}); wake = null }
    async function load() {
      if (closed) return
      try {
        const value = await api(`/api/networks/${network.id}`)
        if (closed || generation.current !== version) return
        setSnapshot(value); setError('')
        const own = value.nodes.find(n => n.id === enrolled?.id)
        if (own?.mode === 'off') { stopWorker(); setWorkerState({ state: 'stopped' }) }
        else if (enrolled && !worker.current && !starting && own?.status === 'offline') start(own?.mode || 'running')
      } catch (e) { if (!closed) setError(e.message) }
      finally { if (!closed) timer = setTimeout(load, 1500) }
    }
    async function lock() {
      if (navigator.wakeLock && document.visibilityState === 'visible' && !wake) {
        try { wake = await navigator.wakeLock.request('screen'); wake.addEventListener('release', () => { wake = null }) } catch { /* optional */ }
      }
    }
    function start(mode) {
      if (closed || starting) return
      starting = true
      const url = new URL('/pool/nodes', location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
      const current = new Worker('/node/worker.js', { type: 'module' }); worker.current = current
      current.onmessage = event => {
        if (closed || generation.current !== version || worker.current !== current) return
        const m = event.data
        setWorkerState(previous => ({ ...previous, ...m, ...(m.state === 'idle' || m.state === 'paused' ? { job_id: null, chunk_id: null } : {}) }))
        if (m.worker_id) { starting = false; setError('') }
        if (m.state === 'stopped' || m.state === 'disconnected') {
          starting = false; current.terminate(); worker.current = null
          // Re-enrollment reads the persisted stop state before any reconnect.
          if (m.permanent) setWorkerState({ state: 'stopped' })
          else if (m.error) setError(m.error)
        }
      }
      current.onerror = e => { starting = false; setError(e.message || 'Worker failed'); current.terminate(); worker.current = null }
      current.postMessage({ type: 'start', url: url.href, label: enrolled.label, node_id: enrolled.id, network_id: network.id,
        visible: document.visibilityState === 'visible', mode, device_type: deviceType() })
      lock()
    }
    function visibility() { worker.current?.postMessage({ type: 'visibility', visible: document.visibilityState === 'visible' }); if (document.visibilityState === 'visible') lock(); else wake?.release().catch(() => {}) }
    api(`/api/networks/${network.id}/nodes`, { device_key: device, label: `${user.name.split(' ')[0]}'s ${deviceType()}` }).then(node => {
      if (closed || generation.current !== version) return
      enrolled = node; setOwnId(node.id)
      if (node.mode !== 'off') start(node.mode)
      else setWorkerState({ state: 'stopped' })
      load()
    }).catch(e => { if (!closed) setError(e.message) })
    document.addEventListener('visibilitychange', visibility)
    return () => { closed = true; generation.current++; clearTimeout(timer); stopWorker(); document.removeEventListener('visibilitychange', visibility) }
  }, [network, user.name, device])
  async function connect(details) {
    const joined = await api(details.mode === 'create' ? '/api/networks' : '/api/networks/join', details.mode === 'create'
      ? { name: details.name, password: details.password } : { network_id: details.name, password: details.password })
    setKnownNetworks(await api('/api/networks')); return joined
  }
  async function leave() { if (network) await api(`/api/networks/${network.id}/leave`, {}); setSnapshot({ nodes: [], jobs: [] }); setOwnId(null) }
  async function control(id, action) { await api(`/api/networks/${network.id}/nodes/${id}/control`, { action }); await refresh() }
  return { network, knownNetworks, snapshot, ownId, workerState, error, ready, connect, leave, control, restore: setNetwork }
}
