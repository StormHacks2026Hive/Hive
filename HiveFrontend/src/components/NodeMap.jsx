import { useRef, useState } from 'react'
import { STATUS, formatData } from '../network.js'
import { HiveBody, HexIcon, Icon } from './HiveScene.jsx'

export default function NodeMap({
  nodes,
  positions,
  onPositionsChange,
  network,
  mode,
  view,
  onViewChange,
  onModeChange,
  onControl,
  onKillNode,
  onNetworkTab,
}) {
  const { selectedId, filter, zoom } = view
  const setSelectedId = (value) => onViewChange((current) => ({ ...current, selectedId: value }))
  const setFilter = (value) => onViewChange((current) => ({ ...current, filter: value }))
  const setZoom = (value) => onViewChange((current) => ({ ...current, zoom: typeof value === 'function' ? value(current.zoom) : value }))
  const [draggingId, setDraggingId] = useState(null)
  const world = useRef(null)
  const drag = useRef(null)

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
  const own = placedNodes.find((node) => node.own) || placedNodes[0]

  if (!network)
    return (
      <div className="empty-map">
        <div className="empty-map-comb">
          <HexIcon />
        </div>
        <h2>No network connected</h2>
        <button className="button button-primary" onClick={onNetworkTab}>
          Connect <Icon name="arrow" />
        </button>
      </div>
    )

  if (!nodes.length) return <div className="empty-map" role="status">Connecting your node…</div>

  return (
    <div className="mapping-layout">
      <section className="map-panel" aria-labelledby="map-title">
        <div className="panel-header">
          <div>
            <h2 id="map-title">{network.name}</h2>
          </div>
          <button
            className="button button-secondary reset-layout"
            onClick={() => {
              onPositionsChange({})
              setZoom(1)
            }}
          >
            <Icon name="network" /> Reset
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
          <div className="map-scroll">
            <svg
              className="node-map"
              viewBox="0 0 846 560"
              role="group"
              aria-label="Interactive honeycomb network map"
            >
              <g className="map-world" ref={world} transform={`translate(423 280) scale(${zoom}) translate(-423 -280)`}>
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
                      <ellipse
                        className="node-selection-ring"
                        rx="82" ry="87"
                      />
                      <path className="node-hanger" d="M0-94V-75" />
                      <g className="node-hive" transform="scale(.39) translate(-200 -232)">
                        <HiveBody doorRadius={140} shaded={false} />
                      </g>
                      <text className="node-name" y="-15" textAnchor="middle">
                        {node.own ? 'Your node' : node.name}
                      </text>
                      <text className="node-rate" y="8" textAnchor="middle">
                        {node.completed || 0} chunks
                      </text>
                      <text className="node-status-text" y="29" textAnchor="middle">
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
          <span className="map-hint">Drag to move</span>
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
      <aside className="node-inspector" key={selected.id} aria-labelledby="inspector-title">
        <div className="inspector-heading">
          <h2 id="inspector-title">{selected.name}</h2>
          <p className="node-device">
            {selected.device}
            {selected.own && ' · You'}
          </p>
          <span className={`status-pill ${selected.status}`}>
            <i />
            {STATUS[selected.status].label}
          </span>
        </div>
        <dl className="node-details">
          <div>
            <dt>GPU share</dt>
            <dd>{Math.round((selected.weight || 0) * 100)}%</dd>
          </div>
          <div>
            <dt>Data received</dt>
            <dd>{formatData(selected.received)}</dd>
          </div>
          <div>
            <dt>Node ID</dt>
            <dd className="mono">
              {selected.id.slice(0, 12)}
            </dd>
          </div>
        </dl>
        {selected.work && <details className="node-task"><summary>Current work</summary><p>{selected.work.kind} · {selected.work.chunk_id}</p><p>{selected.work.count} items · attempt {selected.work.attempt}</p>{selected.can_control && <button className="text-button danger-text" onClick={() => onControl(selected.id, 'cancel')}>Cancel task</button>}</details>}
        <div className="inspector-actions">
          {selected.own && (
            <button
              className="button button-secondary"
              onClick={() => onModeChange(mode === 'running' ? 'paused' : 'running')}
            >
              <Icon name={mode === 'running' ? 'pause' : 'play'} />
              {mode === 'running' ? 'Pause node' : mode === 'off' ? 'Start node' : 'Resume node'}
            </button>
          )}
          <button
            className="button button-danger"
            onClick={() => onKillNode(selected.id)}
            disabled={selected.status === 'offline' || !selected.can_control}
          >
            <Icon name="power" />
            Kill node
          </button>
        </div>
      </aside>
    </div>
  )
}
