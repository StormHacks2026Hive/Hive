import { useEffect, useRef, useState } from 'react'
import { advancePreview, createPreviewNodes, formatData, STATUS } from '../network.js'
import { HexIcon, Icon } from './HiveScene.jsx'
import NetworkSetup from './NetworkSetup.jsx'
import NodeMap from './NodeMap.jsx'

function ConfirmDialog({ type, onClose, onConfirm }) {
  const dialog = useRef(null)
  useEffect(() => {
    dialog.current.showModal()
  }, [])
  const leaving = type === 'leave'
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
      <h2 id="confirm-title">{leaving ? 'Leave this hive?' : 'Turn off your node?'}</h2>
      <p id="confirm-description">
        {leaving
          ? 'Your node will disconnect. You can join again with the network ID and password.'
          : 'Your node will stop sending and receiving data. You can switch it back on whenever you’re ready.'}
      </p>
      <div className="dialog-actions">
        <button className="button button-secondary" onClick={onClose} autoFocus>
          Keep connected
        </button>
        <button className="button button-danger" onClick={onConfirm}>
          {leaving ? 'Leave network' : 'Turn off node'}
        </button>
      </div>
    </dialog>
  )
}

export default function Dashboard({ user, onSignOut, headingRef }) {
  const [tab, setTab] = useState('network')
  const [network, setNetwork] = useState(null)
  const [knownNetworks, setKnownNetworks] = useState([])
  const [nodes, setNodes] = useState([])
  const [positions, setPositions] = useState({})
  const [mode, setMode] = useState('running')
  const [accepting, setAccepting] = useState(true)
  const [capacity, setCapacity] = useState(60)
  const [events, setEvents] = useState([])
  const [notice, setNotice] = useState('')
  const [confirm, setConfirm] = useState(null)

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
      setNodes((current) => advancePreview(current, tick, { mode, accepting, capacity }))
    }, 3000)
    return () => clearInterval(timer)
  }, [network, mode, accepting, capacity])

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
    setNetwork(nextNetwork)
    if (nextNetwork.owner && !knownNetworks.some((item) => item.id === nextNetwork.id))
      setKnownNetworks((current) => [...current, nextNetwork])
    setNodes(createPreviewNodes(user.name))
    setPositions({})
    setMode('running')
    setAccepting(true)
    setCapacity(60)
    setEvents([])
    log(
      nextNetwork.owner
        ? `Created ${nextNetwork.name}. Your hive is ready.`
        : `Joined ${nextNetwork.name}. Welcome to the hive.`,
    )
    setNotice(
      nextNetwork.owner
        ? 'Your network is ready. Share its ID and password to invite others.'
        : 'You’re in. Your node is connected to the network.',
    )
  }

  function changeMode(nextMode) {
    if (nextMode === 'off') {
      setConfirm('off')
      return
    }
    applyMode(nextMode)
  }

  function applyMode(nextMode) {
    setMode(nextMode)
    const status = nextMode === 'off' ? 'offline' : nextMode === 'paused' ? 'paused' : 'idle'
    setNodes((current) => current.map((node) => (node.own ? { ...node, status, rate: 0 } : node)))
    log(
      nextMode === 'off'
        ? 'Your node was turned off.'
        : nextMode === 'paused'
          ? 'Your node is paused. Take a breather.'
          : 'Your node is back online.',
      nextMode === 'off' ? 'power' : nextMode === 'paused' ? 'pause' : 'play',
    )
    setNotice(
      nextMode === 'off'
        ? 'Node turned off. Your network membership is saved for this visit.'
        : nextMode === 'paused'
          ? 'Data transfer paused.'
          : 'Your node is ready to share again.',
    )
  }

  function confirmAction() {
    if (confirm === 'leave') {
      setNetwork(null)
      setNodes([])
      setPositions({})
      setEvents([])
      setNotice('You left the network.')
      setTab('network')
    } else applyMode('off')
    setConfirm(null)
  }

  function toggleReceiving() {
    setAccepting(!accepting)
    if (accepting)
      setNodes((current) =>
        current.map((node) =>
          node.own && node.status === 'receiving' ? { ...node, status: 'idle', rate: 0 } : node,
        ),
      )
    log(accepting ? 'Incoming data is now disabled.' : 'Incoming data is now enabled.', 'down')
  }

  async function copyId() {
    try {
      await navigator.clipboard.writeText(network.id)
      setNotice('Network ID copied. Share it with your people.')
    } catch {
      setNotice('Couldn’t copy. Select the network ID below and copy it manually.')
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
        <div className="workspace-label">YOUR WORKSPACE</div>
        <nav aria-label="Main navigation">
          <button
            className={tab === 'network' ? 'nav-item active' : 'nav-item'}
            aria-current={tab === 'network' ? 'page' : undefined}
            onClick={() => setTab('network')}
          >
            <Icon name="network" />
            Network
            <Icon className="nav-arrow" name="arrow" />
          </button>
          <button
            className={tab === 'mapping' ? 'nav-item active' : 'nav-item'}
            aria-current={tab === 'mapping' ? 'page' : undefined}
            onClick={() => setTab('mapping')}
          >
            <Icon name="map" />
            Mapping
            <Icon className="nav-arrow" name="arrow" />
          </button>
        </nav>
        <div className="sidebar-garden">
          <div className="garden-combs" aria-hidden="true">
            <HexIcon />
            <HexIcon />
            <HexIcon />
          </div>
          <strong>Better, together.</strong>
          <p>
            A little of your power.
            <br />A lot of possibility.
          </p>
          <span>Every connection counts.</span>
        </div>
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
      <div className="workspace">
        <header className="workspace-header">
          <div className="breadcrumb">
            Workspace <span>/</span>
            <strong>{tab === 'network' ? 'Network' : 'Mapping'}</strong>
          </div>
          <span className="workspace-status">
            <i className={network ? 'connected-dot' : ''} />
            {network
              ? mode === 'off'
                ? 'Node offline'
                : 'Connected to your hive'
              : 'Ready when you are'}
          </span>
        </header>
        <main className="workspace-main">
          <div className="page-heading">
            <div>
              <span className="section-kicker">
                {tab === 'network'
                  ? 'YOUR LITTLE PART OF SOMETHING BIGGER'
                  : 'EVERY CONNECTION COUNTS'}
              </span>
              <h1 ref={headingRef} tabIndex={-1}>
                {tab === 'network' ? `Welcome, ${user.name.split(' ')[0]}.` : 'A bird’s-eye view.'}
              </h1>
              <p>
                {tab === 'network'
                  ? network
                    ? 'You’re part of the hive. Make yourself at home.'
                    : 'Let’s find a place for your node to call home.'
                  : 'Meet your nodes. See what’s moving. Stay connected.'}
              </p>
            </div>
            <div className="page-heading-comb" aria-hidden="true">
              <HexIcon />
            </div>
          </div>
          {tab === 'network' ? (
            !network ? (
              <NetworkSetup onConnect={connect} knownNetworks={knownNetworks} />
            ) : (
              <>
                <section className="connected-network panel" aria-labelledby="network-title">
                  <div className="network-identity">
                    <span className="network-symbol">
                      <HexIcon />
                    </span>
                    <div>
                      <span className="section-kicker">
                        {network.owner ? 'YOUR NETWORK' : 'CONNECTED NETWORK'}
                      </span>
                      <h2 id="network-title">{network.name}</h2>
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
                    <span className="preview-badge">
                      <span /> Local preview
                    </span>
                    <button className="text-button" onClick={() => setConfirm('leave')}>
                      Leave network
                      <Icon name="logout" />
                    </button>
                  </div>
                </section>
                <div className="stats-row">
                  <div className="stat-card">
                    <span className="stat-icon">
                      <Icon name="network" />
                    </span>
                    <span>Nodes online</span>
                    <strong>
                      {onlineCount}
                      <small> / {nodes.length}</small>
                    </strong>
                    <p>A growing little community</p>
                  </div>
                  <div className="stat-card">
                    <span className="stat-icon amber">
                      <Icon name="down" />
                    </span>
                    <span>Receiving data</span>
                    <strong>
                      {receivingCount}
                      <small> nodes</small>
                    </strong>
                    <p>Data finding its way home</p>
                  </div>
                  <div className="stat-card">
                    <span className="stat-icon">
                      <Icon name="down" />
                    </span>
                    <span>Your data received</span>
                    <strong>{formatData(own.received)}</strong>
                    <p>Since you joined this network</p>
                  </div>
                </div>
                <div className="network-detail-grid">
                  <section className="node-control panel" aria-labelledby="your-node-title">
                    <div className="panel-header">
                      <div>
                        <span className="section-kicker">YOUR CONTRIBUTION</span>
                        <h2 id="your-node-title">Your node</h2>
                      </div>
                      <span className={`status-pill ${own.status}`}>
                        <i />
                        {STATUS[own.status].label}
                      </span>
                    </div>
                    <div className="your-node-summary">
                      <div className={`your-node-comb ${own.status}`}>
                        <HexIcon />
                      </div>
                      <div>
                        <strong>
                          {mode === 'off'
                            ? 'Resting for now.'
                            : mode === 'paused'
                              ? 'Taking a breather.'
                              : 'Small node. Real possibility.'}
                        </strong>
                        <p>
                          {mode === 'off'
                            ? 'Switch on whenever you’re ready to contribute.'
                            : mode === 'paused'
                              ? 'Your place in the network is still here.'
                              : 'Choose how much you’d like to share.'}
                        </p>
                      </div>
                    </div>
                    <div className="capacity-control">
                      <div>
                        <label htmlFor="capacity">Compute contribution</label>
                        <strong>{capacity}%</strong>
                      </div>
                      <input
                        id="capacity"
                        type="range"
                        min="10"
                        max="100"
                        step="10"
                        value={capacity}
                        disabled={mode === 'off'}
                        onChange={(event) => setCapacity(Number(event.target.value))}
                      />
                      <div className="range-labels">
                        <span>A little</span>
                        <span>All in</span>
                      </div>
                    </div>
                    <div className="receiving-control">
                      <div>
                        <strong>Receive incoming data</strong>
                        <span>Let other nodes send data to yours.</span>
                      </div>
                      <button
                        className={`switch ${accepting ? 'on' : ''}`}
                        role="switch"
                        aria-label="Receive incoming data"
                        aria-checked={accepting}
                        disabled={mode === 'off'}
                        onClick={toggleReceiving}
                      >
                        <span />
                      </button>
                    </div>
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
                        <span className="section-kicker">IN YOUR CORNER</span>
                        <h2 id="activity-title">Recent activity</h2>
                      </div>
                      <span className="little-dot" />
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
                    <button className="activity-map-link" onClick={() => setTab('mapping')}>
                      <div className="tiny-hex-cluster" aria-hidden="true">
                        <HexIcon />
                        <HexIcon />
                        <HexIcon />
                      </div>
                      <span>
                        See how your hive connects<small>Explore the network map</small>
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
              onModeChange={changeMode}
              onNetworkTab={() => setTab('network')}
            />
          )}
          <footer className="workspace-footer">
            <span>
              <HexIcon /> Small nodes. Shared possibilities.
            </span>
            <span>Local preview · Network activity is simulated</span>
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
        <ConfirmDialog type={confirm} onClose={() => setConfirm(null)} onConfirm={confirmAction} />
      )}
    </div>
  )
}
