import assert from 'node:assert/strict'
import test from 'node:test'
import { decodeFrameOutput, decodeNumericOutput } from '../src/results.js'

test('every output element is decoded in order beyond the old 32-value limit', () => {
  for (const [format, method, stride] of [['f32', 'setFloat32', 4], ['f64', 'setFloat64', 8], ['i32', 'setInt32', 4], ['u32', 'setUint32', 4]]) {
    const buffer = new ArrayBuffer(8205 * stride), view = new DataView(buffer)
    for (let i = 0; i < 8205; i++) view[method](i * stride, i * 2 + 1, true)
    const values = decodeNumericOutput(buffer, format)
    assert.equal(values.length, 8205)
    assert.equal(values[0], 1)
    assert.equal(values[32], 65)
    assert.equal(values[4096], 8193)
    assert.equal(values[8204], 16409)
  }
})

test('cached animation output preserves distinct complete frame slices', () => {
  const buffer = new ArrayBuffer(3 * 2 * 2 * 4)
  const bytes = new Uint8Array(buffer)
  bytes.fill(42, 0, 16); bytes.fill(84, 16, 32); bytes.fill(126, 32)
  const pixels = decodeFrameOutput(buffer, 2, 2, 3)
  assert.equal(pixels.subarray(0, 16)[0], 42)
  assert.equal(pixels.subarray(16, 32)[0], 84)
  assert.equal(pixels.subarray(32, 48)[0], 126)
  assert.throws(() => decodeFrameOutput(buffer, 2, 2, 4), /size/)
  assert.throws(() => decodeNumericOutput(new ArrayBuffer(7), 'f64'), /Invalid/)
})
