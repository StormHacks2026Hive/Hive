import assert from 'node:assert/strict'
import test from 'node:test'
import { advancePreview, createPreviewNodes, killPreviewNode } from '../src/network.js'

const settings = { mode: 'running' }

test('killing the selected peer stops only that peer across later preview ticks', () => {
  const starting = createPreviewNodes('Test User')
  let nodes = killPreviewNode(starting, 'amber')
  const stopped = nodes.find((node) => node.id === 'amber')
  assert.equal(stopped.status, 'offline')
  assert.equal(stopped.rate, 0)
  assert.equal(stopped.received, starting.find((node) => node.id === 'amber').received)
  assert.deepEqual(nodes.filter((node) => node.id !== 'amber'), starting.filter((node) => node.id !== 'amber'))
  for (let tick = 0; tick < 12; tick += 1) {
    nodes = advancePreview(nodes, tick, settings)
    assert.deepEqual(nodes.find((node) => node.id === 'amber'), stopped)
  }
  assert.ok(nodes[0].received > starting[0].received)
})

test('paused and switched-off nodes never transfer or receive more data', () => {
  for (const mode of ['paused', 'off']) {
    let nodes = createPreviewNodes('Test User')
    const startingData = nodes[0].received
    for (let tick = 0; tick < 12; tick += 1) {
      nodes = advancePreview(nodes, tick, { ...settings, mode })
      const own = nodes.find((node) => node.own)
      assert.equal(own.rate, 0)
      assert.equal(own.received, startingData)
      assert.equal(own.status, mode === 'off' ? 'offline' : 'paused')
    }
    assert.ok(
      nodes.some((node) => !node.own && node.received > 25),
      'Peers continue their own activity',
    )
  }
})

test('offline peers stay offline while running nodes receive data', () => {
  let nodes = createPreviewNodes('Test User')
  const offline = nodes.find((node) => node.status === 'offline')
  for (let tick = 0; tick < 12; tick += 1) {
    nodes = advancePreview(nodes, tick, settings)
    assert.deepEqual(
      nodes.find((node) => node.id === offline.id),
      offline,
    )
  }
  assert.ok(nodes.find((node) => node.own).received > 0)
})

test('a resumed local node receives and sends normally', () => {
  let nodes = createPreviewNodes('Test User')
  nodes = advancePreview(nodes, 0, { mode: 'paused' })
  const pausedData = nodes[0].received
  let sawSending = false
  for (let tick = 0; tick < 12; tick += 1) {
    nodes = advancePreview(nodes, tick, settings)
    if (nodes[0].status === 'sending') sawSending = true
  }
  assert.ok(nodes[0].received > pausedData)
  assert.ok(sawSending)
})
