// Only view preferences are cached here. Identity, networks and nodes come from the server.
export const SESSION_KEY = 'hive.session.v2'
const text = value => typeof value === 'string' && value.length > 0 && value.length <= 256
function workspaceState(value) {
  if (!value || typeof value !== 'object') return null
  const positions = {}
  for (const [id, position] of Object.entries(value.positions || {}).slice(0, 64)) {
    if (text(id) && Number.isFinite(position?.x) && Number.isFinite(position?.y)) positions[id] = { x: Math.max(80, Math.min(766, position.x)), y: Math.max(95, Math.min(480, position.y)) }
  }
  return {
    network: text(value.network?.id) ? { id: value.network.id } : null,
    tab: ['network', 'mapping', 'compute'].includes(value.tab) ? value.tab : 'network', positions,
    mapView: { selectedId: text(value.mapView?.selectedId) ? value.mapView.selectedId : 'you',
      filter: ['all', 'online', 'receiving'].includes(value.mapView?.filter) ? value.mapView.filter : 'all',
      zoom: Number.isFinite(value.mapView?.zoom) ? Math.max(.75, Math.min(1.35, value.mapView.zoom)) : 1 },
  }
}
export function loadSession(storage) {
  try {
    const saved = JSON.parse((storage ?? globalThis.localStorage).getItem(SESSION_KEY))
    if (saved?.version !== 2 || !text(saved.user?.id) || !text(saved.user?.name) || !text(saved.user?.email)) return null
    const { id, name, email } = saved.user
    return { user: { id, name, email }, workspace: workspaceState(saved.workspace) }
  } catch { return null }
}
export function saveSession(user, workspace, storage) {
  try {
    const { id, name, email } = user
    ;(storage ?? globalThis.localStorage).setItem(SESSION_KEY, JSON.stringify({ version: 2, user: { id, name, email }, workspace: workspaceState(workspace) }))
    return true
  } catch { return false }
}
export function clearSession(storage) {
  try { const target = storage ?? globalThis.localStorage; target.removeItem(SESSION_KEY); target.removeItem('hive.session.v1') } catch { /* storage is optional */ }
}
