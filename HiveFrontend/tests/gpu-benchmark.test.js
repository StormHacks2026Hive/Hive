import assert from 'node:assert/strict'
import test from 'node:test'
import { TileGPU } from '../../node-web/gpu.js'

test('GPU ranking uses the median of warmed comparable tiles', async () => {
  const gpu = new TileGPU()
  const timings = [100, 50, 8, 3, 4, 5, 6]
  const chunks = []
  gpu.render = async chunk => {
    chunks.push(chunk)
    return { elapsed_ms: timings[chunks.length - 1] }
  }
  const result = await gpu.benchmark('test-shader')
  assert.deepEqual(result, { version: 'mandelbrot-v1', pixels: 4096, elapsed_ms: 5 })
  assert.equal(chunks.length, 7)
  for (const chunk of chunks) {
    assert.equal(chunk.shader_id, 'test-shader')
    assert.deepEqual(chunk.tile, { x: 192, y: 192, width: 64, height: 64 })
    assert.equal(chunk.parameters.max_iterations, 256)
  }
})
