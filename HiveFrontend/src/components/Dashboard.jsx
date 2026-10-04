import { useEffect, useRef, useState } from 'react'
import { advancePreview, createPreviewNodes, formatData, killPreviewNode, STATUS } from '../network.js'
import { HangingHive, HexIcon, Icon } from './HiveScene.jsx'
import NetworkSetup from './NetworkSetup.jsx'
import NodeMap from './NodeMap.jsx'
import NodeWork from './NodeWork.jsx'
import useUITransition from '../useUITransition.js'

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
  const [network, setNetwork] = useState(initialState?.network || null)
  const [knownNetworks, setKnownNetworks] = useState([])
  const [nodes, setNodes] = useState(initialState?.nodes || [])
  const [positions, setPositions] = useState(initialState?.positions || {})
  const [mode, setMode] = useState(initialState?.mode || 'running')
  const [events, setEvents] = useState(initialState?.events || [])
  const [mapView, setMapView] = useState(initialState?.mapView || { selectedId: 'you', filter: 'all', zoom: 1 })
  const [notice, setNotice] = useState('')
  const [confirm, setConfirm] = useState(null)
  const transition = useUITransition()

  useEffect(() => {
    onWorkspaceChange({ tab, network, nodes, positions, mode, events, mapView })
  }, [tab, network, nodes, positions, mode, events, mapView, onWorkspaceChange])

  useEffect(() => {
    document.title = 'Hive · Your network'
    return () => {
      document.title = 'Hive · Find your hive'
    }
  }, [])

  useEffect(() => {
    if (!network) return
    let tick = 0
    const timer = setInterval(() => {
      tick += 1
      setNodes((current) => advancePreview(current, tick, { mode }))
    }, 3000)
    return () => clearInterval(timer)
  }, [network, mode])

  useEffect(() => {
    if (!notice) return
    const timer = setTimeout(() => setNotice(''), 5000)
    return () => clearTimeout(timer)
  }, [notice])

  function log(message, icon = 'network') {
    setEvents((current) =>
      [
        {
          id: crypto.randomUUID(),
          message,
          icon,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
        ...current,
      ].slice(0, 5),
    )
  }

  function connect(nextNetwork) {
    transition(() => {
      setNetwork(nextNetwork)
      if (nextNetwork.owner && !knownNetworks.some((item) => item.id === nextNetwork.id))
        setKnownNetworks((current) => [...current, nextNetwork])
      setNodes(createPreviewNodes(user.name))
      setPositions({})
      setMode('running')
      setMapView({ selectedId: 'you', filter: 'all', zoom: 1 })
      setEvents([])
      log(
        nextNetwork.owner
          ? `Created ${nextNetwork.name}`
          : `Joined ${nextNetwork.name}`,
      )
      setNotice(
        nextNetwork.owner ? 'Network created' : 'Connected',
      )
    }, { page: true })
  }

  function changeTab(nextTab) {
    transition(() => setTab(nextTab), { page: true })
  }

  function changeMode(nextMode) {
    if (nextMode === 'off') {
      transition(() => setConfirm('off'))
      return
    }
    transition(() => applyMode(nextMode))
  }

  function applyMode(nextMode) {
    setMode(nextMode)
    const status = nextMode === 'off' ? 'offline' : nextMode === 'paused' ? 'paused' : 'idle'
    setNodes((current) => current.map((node) => (node.own ? { ...node, status, rate: 0 } : node)))
    log(
      nextMode === 'off'
        ? 'Node off'
        : nextMode === 'paused'
          ? 'Node paused'
          : 'Node started',
      nextMode === 'off' ? 'power' : nextMode === 'paused' ? 'pause' : 'play',
    )
    setNotice(
      nextMode === 'off'
        ? 'Node off'
        : nextMode === 'paused'
          ? 'Node paused'
          : 'Node started',
    )
  }

  function confirmAction() {
    const leaving = confirm === 'leave'
    transition(() => {
      if (leaving) {
        setNetwork(null)
        setNodes([])
        setPositions({})
        setEvents([])
        setNotice('Disconnected')
        setTab('network')
      } else if (confirm?.type === 'kill') {
        const target = nodes.find((node) => node.id === confirm.nodeId)
        if (target?.own) applyMode('off')
        else if (target) {
          setNodes((current) => killPreviewNode(current, target.id))
          log(`${target.name} stopped`, 'power')
          setNotice(`${target.name} stopped`)
        }
      } else applyMode('off')
      setConfirm(null)
    }, { page: leaving })
  }

  async function copyId() {
    try {
      await navigator.clipboard.writeText(network.id)
      setNotice('Network ID copied')
    } catch {
      setNotice('Copy failed. Select the ID to copy it.')
    }
  }

  const own = nodes.find((node) => node.own)
  const onlineCount = nodes.filter((node) => node.status !== 'offline').length
  const receivingCount = nodes.filter((node) => node.status === 'receiving').length

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
        </nav>
        <div className="sidebar-account">
          <span className="account-avatar">{user.name.slice(0, 1).toUpperCase()}</span>
          <div>
            <strong>{user.name}</strong>
            <span>{user.email}</span>
          </div>
          <button
            className="icon-button"
            onClick={onSignOut}
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
            <strong>{tab === 'network' ? 'Network' : 'Mapping'}</strong>
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
                {tab === 'network' ? network?.name || `Welcome, ${user.name.split(' ')[0]}.` : 'Network map'}
              </h1>
            </div>
            <div className="page-heading-comb" aria-hidden="true">
              <HexIcon />
            </div>
          </div>
          <div className={`workspace-content ${network ? 'connected-content' : ''}`}>
          {tab === 'network' ? (
            !network ? (
              <NetworkSetup onConnect={connect} knownNetworks={knownNetworks} />
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
                    <strong>{formatData(own.received)}</strong>
                  </div>
                </div>
                <div className="network-detail-grid">
                  <section className="node-control panel" aria-labelledby="your-node-title">
                    <div className="panel-header">
                      <div>
                        <h2 id="your-node-title">Your node</h2>
                      </div>
                      <span className={`status-pill ${own.status}`}>
                        <i />
                        {STATUS[own.status].label}
                      </span>
                    </div>
                    <div className={`node-hive-preview ${own.status}`}>
                      <HangingHive />
                      <div className="node-hive-readout">
                        <strong>{own.rate.toFixed(1)}</strong>
                        <span>MB/s</span>
                      </div>
                    </div>
                    <NodeWork node={own} />
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
              onKillNode={(nodeId) => setConfirm({ type: 'kill', nodeId })}
              onNetworkTab={() => changeTab('network')}
            />
          )}
          </div>
          <footer className="workspace-footer">
            <span>Simulated network</span>
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
