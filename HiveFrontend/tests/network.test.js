import assert from 'node:assert/strict'
import test from 'node:test'
import { advancePreview, createPreviewNodes } from '../src/network.js'

const settings = { mode: 'running', accepting: true, capacity: 60 }

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

test('disabling incoming data blocks receipts but still permits sending', () => {
  let nodes = createPreviewNodes('Test User')
  let sawSending = false
  for (let tick = 0; tick < 12; tick += 1) {
    nodes = advancePreview(nodes, tick, { ...settings, accepting: false })
    const own = nodes.find((node) => node.own)
    assert.notEqual(own.status, 'receiving')
    assert.equal(own.received, 0)
    if (own.status === 'sending') {
      sawSending = true
      assert.ok(own.rate > 0)
    }
  }
  assert.ok(sawSending)
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

test('compute contribution scales only the local transfer rate', () => {
  const initial = createPreviewNodes('Test User')
  const low = advancePreview(initial, 0, { ...settings, capacity: 10 })
  const high = advancePreview(initial, 0, { ...settings, capacity: 100 })
  assert.ok(low[0].rate < high[0].rate)
  assert.deepEqual(
    low.filter((node) => !node.own),
    high.filter((node) => !node.own),
  )
})
