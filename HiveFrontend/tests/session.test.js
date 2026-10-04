import assert from 'node:assert/strict'
import test from 'node:test'
import { createPreviewNodes, killPreviewNode } from '../src/network.js'
import { SESSION_KEY, clearSession, loadSession, saveSession } from '../src/session.js'

function memoryStorage() {
  const values = new Map()
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  }
}

const user = { id: 'user-1', name: 'Test User', email: 'test@example.com' }
const workspace = {
  network: { id: 'HIVE-TEST', name: 'Test Hive', owner: true, password: 'private-password' },
  tab: 'mapping', mode: 'paused', nodes: createPreviewNodes(user.name),
  positions: { you: { x: 440, y: 300 } },
  events: [{ id: 'event-1', message: 'Node paused', time: '10:00 AM', icon: 'pause' }],
  mapView: { selectedId: 'amber', filter: 'online', zoom: 1.2 },
}

test('refresh restores profile, network, map, activity and paused mode without saving secrets', () => {
  const storage = memoryStorage()
  assert.equal(saveSession({ ...user, credential: 'private-google-token' }, workspace, storage), true)
  const saved = loadSession(storage)
  assert.deepEqual(saved.user, user)
  assert.equal(saved.workspace.network.id, 'HIVE-TEST')
  assert.equal(saved.workspace.tab, 'mapping')
  assert.deepEqual(saved.workspace.positions, workspace.positions)
  assert.deepEqual(saved.workspace.mapView, workspace.mapView)
  assert.deepEqual(saved.workspace.events, workspace.events)
  assert.equal(saved.workspace.nodes[0].status, 'paused')
  assert.equal(saved.workspace.nodes[0].rate, 0)
  assert.equal(saved.workspace.nodes[0].received, workspace.nodes[0].received)
  const raw = storage.getItem(SESSION_KEY)
  assert.ok(!raw.includes('private-password'))
  assert.ok(!raw.includes('private-google-token'))
})

test('corrupt or unsupported storage falls back to login', () => {
  const storage = memoryStorage()
  for (const value of ['{broken', 'null', '{}', JSON.stringify({ version: 2, user }), JSON.stringify({ version: 1, user: {} })]) {
    storage.setItem(SESSION_KEY, value)
    assert.equal(loadSession(storage), null)
  }
})

test('a killed peer stays offline after saving and restoring the map', () => {
  const storage = memoryStorage()
  saveSession(user, { ...workspace, nodes: killPreviewNode(workspace.nodes, 'amber') }, storage)
  const restored = loadSession(storage).workspace
  assert.equal(restored.nodes.find((node) => node.id === 'amber').status, 'offline')
  assert.equal(restored.nodes.find((node) => node.id === 'amber').rate, 0)
  assert.equal(restored.nodes[0].status, 'paused')
})

test('invalid workspace values are repaired without losing a valid signed-in profile', () => {
  const storage = memoryStorage()
  storage.setItem(SESSION_KEY, JSON.stringify({ version: 1, user, workspace: {
    ...workspace, mode: 'invalid', tab: 'invalid', events: [null],
    nodes: [{ id: 'you', own: false, status: 'toString', received: -10, rate: 'oops' }],
    positions: { you: { x: 9000, y: -20 } },
    mapView: { selectedId: 'missing', filter: 'invalid', zoom: 900 },
  } }))
  const saved = loadSession(storage)
  assert.deepEqual(saved.user, user)
  assert.equal(saved.workspace.tab, 'network')
  assert.equal(saved.workspace.mode, 'running')
  assert.equal(saved.workspace.nodes.length, 7)
  assert.equal(saved.workspace.nodes[0].own, true)
  assert.equal(saved.workspace.nodes[0].received, 0)
  assert.equal(saved.workspace.nodes[0].rate, 0)
  assert.deepEqual(saved.workspace.positions.you, { x: 766, y: 95 })
  assert.deepEqual(saved.workspace.mapView, { selectedId: 'you', filter: 'all', zoom: 1.35 })
  assert.deepEqual(saved.workspace.events, [])
})

test('leaving saves a disconnected workspace and sign-out clears the saved session', () => {
  const storage = memoryStorage()
  saveSession(user, { ...workspace, network: null }, storage)
  assert.equal(loadSession(storage).workspace.network, null)
  assert.deepEqual(loadSession(storage).workspace.nodes, [])
  clearSession(storage)
  assert.equal(loadSession(storage), null)
})

test('blocked or full storage never prevents using the app', () => {
  const storage = {
    getItem() { throw new Error('Blocked') },
    setItem() { throw new Error('Full') },
    removeItem() { throw new Error('Blocked') },
  }
  assert.equal(loadSession(storage), null)
  assert.equal(saveSession(user, workspace, storage), false)
  assert.doesNotThrow(() => clearSession(storage))
})
