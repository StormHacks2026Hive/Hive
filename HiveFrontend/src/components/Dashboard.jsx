import { useEffect, useRef, useState } from 'react'
import { formatData, STATUS } from '../network.js'
import { HangingHive, HexIcon, Icon } from './HiveScene.jsx'
import NetworkSetup from './NetworkSetup.jsx'
import NodeMap from './NodeMap.jsx'
import NodeWork from './NodeWork.jsx'
import useUITransition from '../useUITransition.js'
import useNetwork from '../useNetwork.js'
import ComputePanel from './ComputePanel.jsx'

function ConfirmDialog({ type, nodeName, onClose, onConfirm }) {
  const dialog = useRef(null)
  useEffect(() => {
    dialog.current.showModal()
  }, [])
  const leaving = type === 'leave'
  const killing = type === 'kill'
  return (
    <dialog
      ref={dialog}
      className="confirm-dialog"
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
      aria-labelledby="confirm-title"
      aria-describedby="confirm-description"
    >
      <div className="dialog-icon">
        <Icon name={leaving ? 'logout' : 'power'} />
      </div>
      <h2 id="confirm-title">{leaving ? 'Leave this hive?' : killing ? `Kill ${nodeName}?` : 'Turn off your node?'}</h2>
      <p id="confirm-description">
        {leaving
          ? 'Your node will disconnect.'
          : 'Data transfer will stop.'}
      </p>
      <div className="dialog-actions">
        <button className="button button-secondary" onClick={onClose} autoFocus>
          Cancel
        </button>
        <button className="button button-danger" onClick={onConfirm}>
          {leaving ? 'Leave network' : killing ? 'Kill node' : 'Turn off node'}
        </button>
      </div>
    </dialog>
  )
}

export default function Dashboard({ user, initialState, onWorkspaceChange, onSignOut, headingRef }) {
  const [tab, setTab] = useState(initialState?.tab || 'network')
  const [positions, setPositions] = useState(initialState?.positions || {})
  const [mapView, setMapView] = useState(initialState?.mapView || { selectedId: 'you', filter: 'all', zoom: 1 })
  const [notice, setNotice] = useState(''), [confirm, setConfirm] = useState(null)
  const transition = useUITransition()
  const live = useNetwork(user, initialState?.network)
  const { network, knownNetworks } = live
  const rawNodes = network ? live.snapshot.nodes : []
  const ordered = [...rawNodes].sort((a, b) => Number(b.id === live.ownId) - Number(a.id === live.ownId) || a.id.localeCompare(b.id))
  const nodes = ordered.map((node, index) => {
    const own = node.id === live.ownId
    const angle = (index - 1) * Math.PI * 2 / Math.max(1, ordered.length - 1) - Math.PI / 2
    return { ...node, own, x: own ? 410 : 410 + Math.cos(angle) * 265, y: own ? 285 : 285 + Math.sin(angle) * 185,
      rate: 0, received: node.received / 1048576, gpu: node.capabilities.webgpu ? 'Ready' : 'CPU only' }
  })
  const own = nodes.find(node => node.own)
  const mode = own?.mode || 'running'
  const events = live.snapshot.jobs.slice(-5).reverse().map(job => ({ id: job.job_id, message: `${job.kind} · ${job.status} · ${job.completed_chunks}/${job.total_chunks}`, icon: 'network', time: '' }))
  const onlineCount = nodes.filter(node => node.status !== 'offline').length
  const receivingCount = nodes.filter(node => ['receiving', 'working'].includes(node.status)).length

  useEffect(() => {
    if (live.ready) onWorkspaceChange({ tab, network, positions, mapView })
  }, [tab, network, positions, mapView, live.ready, onWorkspaceChange])
  useEffect(() => {
    if (!live.ownId) return
    setMapView(current => current.selectedId === 'you' ? { ...current, selectedId: live.ownId } : current)
  }, [live.ownId])
  useEffect(() => {
    if (!notice) return
    const timer = setTimeout(() => setNotice(''), 4000)
    return () => clearTimeout(timer)
  }, [notice])
  async function connect(details) {
    const joined = await live.connect(details)
    transition(() => { live.restore(joined); setPositions({}); setMapView({ selectedId: 'you', filter: 'all', zoom: 1 }) }, { page: true })
  }
  function changeTab(next) { transition(() => setTab(next), { page: true }) }
  async function control(id, action) {
    try { await live.control(id, action); setNotice(action === 'kill' ? 'Node stopped' : action === 'resume' ? 'Node started' : 'Node paused') }
    catch (e) { setNotice(e.message) }
  }
  function changeMode(next) {
    if (!own) return
    if (next === 'off') setConfirm({ type: 'kill', nodeId: own.id })
    else control(own.id, next === 'running' ? 'resume' : 'pause')
  }
  async function confirmAction() {
    try {
      if (confirm === 'leave') {
        await live.leave()
        transition(() => { live.restore(null); setTab('network'); setConfirm(null) }, { page: true })
      } else {
        await live.control(confirm.nodeId, 'kill')
        setConfirm(null); setNotice('Node stopped')
      }
    } catch (e) { setConfirm(null); setNotice(e.message) }
  }
  async function copyId() {
    try { await navigator.clipboard.writeText(network.id); setNotice('Network ID copied') }
    catch { setNotice('Select the network ID to copy it.') }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Hive home">
          <HexIcon /> hive<span className="brand-dot">.</span>
        </a>
        <nav aria-label="Main navigation">
          <button
            className={tab === 'network' ? 'nav-item active' : 'nav-item'}
            aria-current={tab === 'network' ? 'page' : undefined}
            onClick={() => changeTab('network')}
          >
            <Icon name="network" />
            Network
            <Icon className="nav-arrow" name="arrow" />
          </button>
          <button
            className={tab === 'mapping' ? 'nav-item active' : 'nav-item'}
            aria-current={tab === 'mapping' ? 'page' : undefined}
            onClick={() => changeTab('mapping')}
          >
            <Icon name="map" />
            Mapping
            <Icon className="nav-arrow" name="arrow" />
          </button>
          <button className={tab === 'compute' ? 'nav-item active' : 'nav-item'} disabled={!network}
            aria-current={tab === 'compute' ? 'page' : undefined} onClick={() => changeTab('compute')}>
            <Icon name="plus" /> Compute <Icon className="nav-arrow" name="arrow" />
          </button>
        </nav>
        <div className="sidebar-account">
          <span className="account-avatar">{user.name.slice(0, 1).toUpperCase()}</span>
          <div>
            <strong>{user.name}</strong>
            <span>{user.email}</span>
          </div>
          <button
            className="icon-button"
            onClick={() => onSignOut().catch(e => setNotice(e.message))}
            aria-label="Sign out"
            title="Sign out"
          >
            <Icon name="logout" />
          </button>
        </div>
      </aside>
      <div className={`workspace ${network ? 'is-connected' : ''}`}>
        <header className="workspace-header">
          <div className="breadcrumb">
            Workspace <span>/</span>
            <strong>{{ network: 'Network', mapping: 'Mapping', compute: 'Compute' }[tab]}</strong>
          </div>
          {network && (
            <span className="workspace-status">
              <i className="connected-dot" />
              {mode === 'off' ? 'Node offline' : 'Connected'}
            </span>
          )}
        </header>
        <main className="workspace-main" key={`${tab}-${network?.id || 'setup'}`}>
          <div className="page-heading">
            <div>
              <h1 ref={headingRef} tabIndex={-1}>
                {tab === 'network' ? network?.name || `Welcome, ${user.name.split(' ')[0]}.` : tab === 'mapping' ? 'Network map' : 'Send work'}
              </h1>
            </div>
            <div className="page-heading-comb" aria-hidden="true">
              <HexIcon />
            </div>
          </div>
          <div className={`workspace-content ${tab === 'compute' ? 'compute-content' : ''} ${network ? 'connected-content' : ''}`}>
          {live.error && <p className="form-error" role="alert">{live.error}</p>}
          {tab === 'compute' && network ? <ComputePanel network={network} nodes={nodes} onControl={control} onKillNode={id => setConfirm({ type: 'kill', nodeId: id })} /> : tab === 'network' ? (
            !network ? (
              <NetworkSetup onConnect={connect} knownNetworks={knownNetworks} onRestore={n => transition(() => live.restore(n), { page: true })} />
            ) : (
              <>
                <section className="connected-network panel" aria-labelledby="network-title">
                  <div className="network-identity">
                    <div>
                      <h2 id="network-title">Network ID</h2>
                      <div className="network-id">
                        <span className="mono">{network.id}</span>
                        <button
                          className="icon-button"
                          aria-label="Copy network ID"
                          onClick={copyId}
                        >
                          <Icon name="copy" />
                        </button>
                      </div>
                    </div>
                  </div>
                  <div className="network-header-actions">
                    <button
                      className="button button-secondary"
                      aria-label="Leave network"
                      onClick={() => transition(() => setConfirm('leave'))}
                    >
                      Leave
                      <Icon name="logout" />
                    </button>
                  </div>
                </section>
                <div className="stats-row">
                  <div className="stat-card">
                    <span>Nodes online</span>
                    <strong>
                      {onlineCount}
                      <small> / {nodes.length}</small>
                    </strong>
                  </div>
                  <div className="stat-card">
                    <span>Receiving</span>
                    <strong>
                      {receivingCount}
                      <small> nodes</small>
                    </strong>
                  </div>
                  <div className="stat-card">
                    <span>Data received</span>
                    <strong>{formatData(own?.received || 0)}</strong>
                  </div>
                </div>
                <div className="network-detail-grid">
                  <section className="node-control panel" aria-labelledby="your-node-title">
                    <div className="panel-header">
                      <div>
                        <h2 id="your-node-title">Your node</h2>
                      </div>
                      <span className={`status-pill ${(own?.status || 'offline')}`}>
                        <i />
                        {STATUS[(own?.status || 'offline')].label}
                      </span>
                    </div>
                    <div className={`node-hive-preview ${(own?.status || 'offline')}`}>
                      <HangingHive />
                      <div className="node-hive-readout">
                        <strong>{String(own?.completed || 0)}</strong>
                        <span>chunks</span>
                      </div>
                    </div>
                    <NodeWork node={own} local={live.workerState} />
                    <div className="node-control-actions">
                      <button
                        className="button button-secondary"
                        onClick={() => changeMode(mode === 'running' ? 'paused' : 'running')}
                      >
                        <Icon name={mode === 'running' ? 'pause' : 'play'} />
                        {mode === 'running'
                          ? 'Pause node'
                          : mode === 'off'
                            ? 'Start node'
                            : 'Resume node'}
                      </button>
                      {mode !== 'off' && (
                        <button
                          className="text-button danger-text"
                          onClick={() => changeMode('off')}
                        >
                          <Icon name="power" />
                          Turn off node
                        </button>
                      )}
                    </div>
                  </section>
                  <section className="activity-panel panel" aria-labelledby="activity-title">
                    <div className="panel-header">
                      <div>
                        <h2 id="activity-title">Activity</h2>
                      </div>
                    </div>
                    <ol className="activity-list">
                      {events.map((event) => (
                        <li key={event.id}>
                          <span className="activity-icon">
                            <Icon name={event.icon} />
                          </span>
                          <div>
                            <p>{event.message}</p>
                            <time>{event.time}</time>
                          </div>
                        </li>
                      ))}
                    </ol>
                    <button className="activity-map-link" onClick={() => changeTab('mapping')}>
                      <div className="tiny-hex-cluster" aria-hidden="true">
                        <HexIcon />
                        <HexIcon />
                        <HexIcon />
                      </div>
                      <span>
                        Open map
                      </span>
                      <Icon name="arrow" />
                    </button>
                  </section>
                </div>
              </>
            )
          ) : (
            <NodeMap
              nodes={nodes}
              positions={positions}
              onPositionsChange={setPositions}
              network={network}
              mode={mode}
              view={mapView}
              onViewChange={setMapView}
              onModeChange={changeMode}
              onControl={control}
              onKillNode={(nodeId) => setConfirm({ type: 'kill', nodeId })}
              onNetworkTab={() => changeTab('network')}
            />
          )}
          </div>
          <footer className="workspace-footer">
            <span>{network ? `${onlineCount} nodes online` : 'Hive'}</span>
          </footer>
        </main>
      </div>
      {notice && (
        <div className="toast" role="status">
          <Icon name="check" />
          <span>{notice}</span>
          <button
            className="icon-button"
            aria-label="Dismiss notification"
            onClick={() => setNotice('')}
          >
            <Icon name="close" />
          </button>
        </div>
      )}
      {confirm && (
        <ConfirmDialog
          type={typeof confirm === 'string' ? confirm : confirm.type}
          nodeName={nodes.find((node) => node.id === confirm.nodeId)?.own ? 'your node' : nodes.find((node) => node.id === confirm.nodeId)?.name}
          onClose={() => setConfirm(null)}
          onConfirm={confirmAction}
        />
      )}
    </div>
  )
}
