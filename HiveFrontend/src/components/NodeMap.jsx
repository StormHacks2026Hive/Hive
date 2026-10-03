import { useEffect, useRef, useState } from 'react'
import { STATUS, formatData } from '../network.js'
import { TreeArt, HexIcon, Icon } from './HiveScene.jsx'

const hexPoints = '-36,-62 36,-62 72,0 36,62 -36,62 -72,0'

export default function NodeMap({
  nodes,
  positions,
  onPositionsChange,
  network,
  mode,
  onModeChange,
  onNetworkTab,
}) {
  const [selectedId, setSelectedId] = useState('you')
  const [filter, setFilter] = useState('all')
  const [zoom, setZoom] = useState(1)
  const [draggingId, setDraggingId] = useState(null)
  const world = useRef(null)
  const scroll = useRef(null)
  const drag = useRef(null)
  const [compact, setCompact] = useState(() => window.matchMedia('(max-width: 650px)').matches)

  useEffect(() => {
    const query = window.matchMedia('(max-width: 650px)')
    const update = () => setCompact(query.matches)
    query.addEventListener('change', update)
    return () => query.removeEventListener('change', update)
  }, [])
  useEffect(() => {
    if (compact && scroll.current) {
      scroll.current.scrollLeft = (scroll.current.scrollWidth - scroll.current.clientWidth) / 2
    }
  }, [compact, network])

  const placedNodes = nodes.map((node) => ({ ...node, ...positions[node.id] }))

  function moveNode(id, x, y) {
    onPositionsChange((current) => ({
      ...current,
      [id]: {
        x: Math.max(80, Math.min(766, x)),
        y: Math.max(95, Math.min(480, y)),
      },
    }))
  }

  function worldPoint(event) {
    const matrix = world.current?.getScreenCTM()
    return matrix
      ? new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
      : null
  }

  function startDrag(event, node) {
    if (event.button !== 0) return
    const point = worldPoint(event)
    if (!point) return
    event.currentTarget.setPointerCapture(event.pointerId)
    event.currentTarget.focus({ preventScroll: true })
    drag.current = {
      id: node.id,
      pointerId: event.pointerId,
      dx: point.x - node.x,
      dy: point.y - node.y,
    }
    setSelectedId(node.id)
    setDraggingId(node.id)
  }

  function dragNode(event) {
    if (!drag.current || drag.current.pointerId !== event.pointerId) return
    const point = worldPoint(event)
    if (point) moveNode(drag.current.id, point.x - drag.current.dx, point.y - drag.current.dy)
  }

  function endDrag(event) {
    if (!drag.current || drag.current.pointerId !== event.pointerId) return
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId)
    drag.current = null
    setDraggingId(null)
  }

  const selected = nodes.find((node) => node.id === selectedId) || nodes[0]
  const own = placedNodes.find((node) => node.own)

  if (!network)
    return (
      <div className="empty-map">
        <div className="empty-map-comb">
          <HexIcon />
        </div>
        <h2>Your hive is waiting.</h2>
        <p>
          Connect to a network to see its nodes
          <br />
          and how data moves between them.
        </p>
        <button className="button button-primary" onClick={onNetworkTab}>
          Find a network <Icon name="arrow" />
        </button>
      </div>
    )

  return (
    <div className="mapping-layout">
      <section className="map-panel" aria-labelledby="map-title">
        <div className="panel-header">
          <div>
            <span className="section-kicker">THE BIGGER PICTURE</span>
            <h2 id="map-title">Your network, connected.</h2>
          </div>
          <button
            className="button button-secondary reset-layout"
            onClick={() => {
              onPositionsChange({})
              setZoom(1)
            }}
          >
            <Icon name="network" /> Reset combs
          </button>
        </div>
        <div className="map-toolbar">
          <div className="map-filters" aria-label="Highlight nodes">
            {[
              ['all', 'All nodes'],
              ['receiving', 'Receiving'],
              ['online', 'Online'],
            ].map(([value, label]) => (
              <button
                key={value}
                className={filter === value ? 'active' : ''}
                aria-pressed={filter === value}
                onClick={() => setFilter(value)}
              >
                {label}
              </button>
            ))}
          </div>
          <span className="map-count">
            {nodes.filter((node) => node.status !== 'offline').length} / {nodes.length} online
          </span>
        </div>
        <div className="map-canvas">
          <div className="map-scroll" ref={scroll}>
            <svg
              className="node-map"
              viewBox="0 0 846 560"
              role="group"
              aria-label="Interactive honeycomb network map"
            >
              <svg
                width="846"
                height="560"
                viewBox="0 0 846 486"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                <TreeArt />
              </svg>
              <g ref={world} transform={`translate(423 280) scale(${zoom}) translate(-423 -280)`}>
                {placedNodes
                  .filter((node) => !node.own)
                  .map((node) => {
                    const connected = own.status !== 'offline' && node.status !== 'offline'
                    const moving =
                      connected &&
                      mode === 'running' &&
                      ['receiving', 'sending'].includes(node.status)
                    return (
                      <g key={`link-${node.id}`} aria-hidden="true">
                        <path
                          d={`M${own.x} ${own.y}L${node.x} ${node.y}`}
                          className={connected ? 'map-link' : 'map-link disconnected'}
                        />
                        {moving && (
                          <path
                            d={`M${own.x} ${own.y}L${node.x} ${node.y}`}
                            className={`flow-link ${node.status}`}
                          />
                        )}
                      </g>
                    )
                  })}
                {placedNodes.map((node) => {
                  const dimmed =
                    filter === 'receiving'
                      ? node.status !== 'receiving'
                      : filter === 'online' && node.status === 'offline'
                  const selectedNode = node.id === selectedId
                  return (
                    <g
                      key={node.id}
                      transform={`translate(${node.x} ${node.y})`}
                      data-node-id={node.id}
                      onPointerDown={(event) => startDrag(event, node)}
                      onPointerMove={dragNode}
                      onPointerUp={endDrag}
                      onPointerCancel={endDrag}
                      className={`map-node ${node.status} ${selectedNode ? 'selected' : ''} ${dimmed ? 'dimmed' : ''} ${draggingId === node.id ? 'dragging' : ''}`}
                      role="button"
                      tabIndex={0}
                      aria-label={`${node.name}${node.own ? ', your node' : ''}, ${STATUS[node.status].label}`}
                      aria-pressed={selectedNode}
                      onClick={() => setSelectedId(node.id)}
                      aria-keyshortcuts="ArrowUp ArrowDown ArrowLeft ArrowRight"
                      onKeyDown={(event) => {
                        const directions = {
                          ArrowLeft: [-1, 0],
                          ArrowRight: [1, 0],
                          ArrowUp: [0, -1],
                          ArrowDown: [0, 1],
                        }
                        if (directions[event.key]) {
                          event.preventDefault()
                          setSelectedId(node.id)
                          const [dx, dy] = directions[event.key]
                          const step = event.shiftKey ? 30 : 10
                          moveNode(node.id, node.x + dx * step, node.y + dy * step)
                        }
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault()
                          setSelectedId(node.id)
                        }
                      }}
                    >
                      <polygon
                        className="node-selection-ring"
                        points="-40,-69 40,-69 80,0 40,69 -40,69 -80,0"
                      />
                      <path className="node-hanger" d="M0-83V-63" />
                      <polygon className="node-hex" points={hexPoints} />
                      <path className="node-shine" d="m-31-53 59 0 17 30" />
                      <text className="node-initials" y="-10" textAnchor="middle">
                        {node.short}
                      </text>
                      <text className="node-name" y="11" textAnchor="middle">
                        {node.own ? 'Your node' : node.name}
                      </text>
                      <circle cy="31" cx="-26" r="3" fill={STATUS[node.status].color} />
                      <text className="node-status-text" y="34" x="5" textAnchor="middle">
                        {STATUS[node.status].label}
                      </text>
                    </g>
                  )
                })}
              </g>
            </svg>
          </div>
          <div className="map-zoom">
            <button
              aria-label="Zoom out"
              onClick={() => setZoom((value) => Math.max(0.75, value - 0.1))}
              disabled={zoom <= 0.75}
            >
              −
            </button>
            <button aria-label="Reset zoom" onClick={() => setZoom(1)}>
              {Math.round(zoom * 100)}%
            </button>
            <button
              aria-label="Zoom in"
              onClick={() => setZoom((value) => Math.min(1.35, value + 0.1))}
              disabled={zoom >= 1.35}
            >
              +
            </button>
          </div>
          <span className="map-hint">Drag a comb · Arrow keys work too</span>
        </div>
        <div className="map-legend">
          {['receiving', 'sending', 'idle', 'paused', 'offline'].map((status) => (
            <span key={status}>
              <i style={{ background: STATUS[status].color }} />
              {STATUS[status].label}
            </span>
          ))}
        </div>
      </section>
      <aside className="node-inspector" aria-labelledby="inspector-title">
        <span className="section-kicker">NODE DETAILS</span>
        <div className={`inspector-comb ${selected.status}`}>
          <HexIcon />
        </div>
        <h2 id="inspector-title">{selected.name}</h2>
        <p className="node-device">
          {selected.device}
          {selected.own && ' · You'}
        </p>
        <span className={`status-pill ${selected.status}`}>
          <i />
          {STATUS[selected.status].label}
        </span>
        <p className="node-description">{STATUS[selected.status].detail}</p>
        <dl className="node-details">
          <div>
            <dt>Network</dt>
            <dd>{network.name}</dd>
          </div>
          <div>
            <dt>Transfer rate</dt>
            <dd>{selected.rate.toFixed(1)} MB/s</dd>
          </div>
          <div>
            <dt>Data received</dt>
            <dd>{formatData(selected.received)}</dd>
          </div>
          <div>
            <dt>Node ID</dt>
            <dd className="mono">
              {selected.own ? 'NODE-LOCAL' : `NODE-${selected.id.toUpperCase()}`}
            </dd>
          </div>
        </dl>
        {selected.own ? (
          <div className="inspector-actions">
            <button
              className="button button-secondary"
              onClick={() => onModeChange(mode === 'running' ? 'paused' : 'running')}
            >
              <Icon name={mode === 'running' ? 'pause' : 'play'} />
              {mode === 'running' ? 'Pause node' : mode === 'off' ? 'Start node' : 'Resume node'}
            </button>
            {mode !== 'off' && (
              <button className="text-button danger-text" onClick={() => onModeChange('off')}>
                <Icon name="power" />
                Turn off node
              </button>
            )}
          </div>
        ) : (
          <p className="inspector-note">
            This node belongs to another member.
            <br />
            You can view its activity here.
          </p>
        )}
        <div className="inspector-footer">
          <span className="little-dot" /> Activity updates every 3 seconds
        </div>
      </aside>
    </div>
  )
}
