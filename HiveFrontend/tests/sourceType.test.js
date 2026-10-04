import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { sourceType } from '../src/sourceType.js'

const wrapped = await readFile(new URL('../public/examples/Mandelbulb.wgsl', import.meta.url), 'utf8')
const raw = wrapped.match(/WGSL_SHADER\s*=\s*r"""([\s\S]*?)"""/)[1]

test('pasted Mandelbulb wrappers and raw shaders select image mode', () => {
  assert.deepEqual(sourceType(wrapped), { image: true, rawWgsl: false })
  assert.deepEqual(sourceType(raw), { image: true, rawWgsl: true })
  assert.equal(sourceType('/* renderer */\n' + raw).image, true)
})

test('numeric WGSL and Python keep their correct source type', () => {
  assert.deepEqual(sourceType('@group(0) @binding(1) var<storage, read_write> out: array<f32>; @compute @workgroup_size(64) fn main() {}'), { image: false, rawWgsl: true })
  assert.deepEqual(sourceType('def f(values):\n    return [x * 2 for x in values]'), { image: false, rawWgsl: false })
  assert.deepEqual(sourceType('WGSL_SHADER = "@compute @workgroup_size(64) fn main() {}"'), { image: false, rawWgsl: false })
  assert.deepEqual(sourceType('// texture_storage_2d<rgba8unorm, write> @compute'), { image: false, rawWgsl: false })
})
