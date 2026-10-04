import { createPreviewNodes, STATUS } from './network.js'

export const SESSION_KEY = 'hive.session.v1'

const text = (value) => typeof value === 'string' && value.trim().length > 0 && value.length <= 256
const amount = (value) => Number.isFinite(value) && value >= 0

function workspaceState(value, user) {
  if (!value || typeof value !== 'object') return null
  const network = value.network && text(value.network.id) && text(value.network.name)
    ? { id: value.network.id, name: value.network.name, owner: value.network.owner === true }
    : null
  const mode = ['running', 'paused', 'off'].includes(value.mode) ? value.mode : 'running'
  const nodes = network ? createPreviewNodes(user.name).map((node) => {
    const saved = Array.isArray(value.nodes) && value.nodes.find((item) => item?.id === node.id)
    const previousStatus = saved && Object.hasOwn(STATUS, saved.status) ? saved.status : node.status
    const status = node.own && mode !== 'running' ? mode === 'off' ? 'offline' : 'paused' : previousStatus
    return {
      ...node,
      status,
      rate: saved && amount(saved.rate) && ['receiving', 'sending'].includes(status) ? saved.rate : 0,
      received: saved && amount(saved.received) ? saved.received : node.received,
    }
  }) : []
  const positions = {}
  for (const node of nodes) {
    const position = value.positions?.[node.id]
    if (position && Number.isFinite(position.x) && Number.isFinite(position.y)) {
      positions[node.id] = {
        x: Math.max(80, Math.min(766, position.x)),
        y: Math.max(95, Math.min(480, position.y)),
      }
    }
  }
  const events = network && Array.isArray(value.events)
    ? value.events.filter((event) => event && text(event.id) && text(event.message) && text(event.time))
      .slice(0, 5).map(({ id, message, time, icon }) => ({ id, message, time, icon: text(icon) ? icon : 'network' }))
    : []
  const view = value.mapView || {}
  return {
    network, nodes, positions, mode, events,
    tab: value.tab === 'mapping' ? 'mapping' : 'network',
    mapView: {
      selectedId: nodes.some((node) => node.id === view.selectedId) ? view.selectedId : 'you',
      filter: ['all', 'online', 'receiving'].includes(view.filter) ? view.filter : 'all',
      zoom: Number.isFinite(view.zoom) ? Math.max(.75, Math.min(1.35, view.zoom)) : 1,
    },
  }
}

export function loadSession(storage) {
  try {
    const saved = JSON.parse((storage ?? globalThis.localStorage).getItem(SESSION_KEY))
    if (saved?.version !== 1 || !text(saved.user?.id) || !text(saved.user?.name) || !text(saved.user?.email)) return null
    const { id, name, email } = saved.user
    const user = { id, name, email }
    return { user, workspace: workspaceState(saved.workspace, user) }
  } catch {
    return null
  }
}

export function saveSession(user, workspace, storage) {
  try {
    const { id, name, email } = user
    // Save display state only; Google credentials and network passwords stay out.
    const target = storage ?? globalThis.localStorage
    target.setItem(SESSION_KEY, JSON.stringify({
      version: 1, user: { id, name, email }, workspace: workspaceState(workspace, user),
    }))
    return true
  } catch {
    return false
  }
}

export function clearSession(storage) {
  try {
    const target = storage ?? globalThis.localStorage
    target.removeItem(SESSION_KEY)
  } catch {
    // The app still works when browser storage is unavailable.
  }
}
