import assert from 'node:assert/strict'
import test from 'node:test'
import { readFile } from 'node:fs/promises'
import { nodeSpecs } from '../src/nodeSpecs.js'

const source = await readFile(new URL('../../node-web/device-info.js', import.meta.url), 'utf8')
const { hardwareInfo } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)

test('browser hardware reports stay optional and do not invent specs', () => {
  assert.deepEqual(hardwareInfo({}), { logical_cores: null, memory_gib: null, platform: '' })
  assert.deepEqual(hardwareInfo({ hardwareConcurrency: 12, deviceMemory: 8, userAgentData: { platform: 'macOS' } }),
    { logical_cores: 12, memory_gib: 8, platform: 'macOS' })
  const invalid = hardwareInfo({ hardwareConcurrency: -1, deviceMemory: Infinity })
  assert.equal(invalid.logical_cores, null)
  assert.equal(invalid.memory_gib, null)
})

test('spec labels distinguish reported capacity from exact hardware identity', () => {
  const specs = nodeSpecs({ webgpu: true, adapter: { vendor: 'apple', architecture: 'metal' },
    hardware: { logical_cores: 8, memory_gib: 8 }, limits: { maxStorageBufferBindingSize: 134217728, maxTextureDimension2D: 8192 } })
  assert.ok(specs.includes('GPU: apple metal'))
  assert.ok(specs.some(s => s.includes('browser estimate')))
  assert.ok(specs.some(s => s.includes('128 MiB')))
  assert.ok(specs.some(s => s.includes('8192 × 8192')))
  assert.ok(nodeSpecs({ webgpu: true }).includes('GPU: Model hidden by browser'))
  assert.ok(nodeSpecs({webgpu:true,benchmark:{pixels:4096,elapsed_ms:2.345}}).includes('GPU probe: 4096 pixels in 2.35 ms (dispatch + readback)'))
})
