export function decodeNumericOutput(buffer, format) {
  const stride = format === 'f64' ? 8 : 4
  const method = { f64: 'getFloat64', f32: 'getFloat32', i32: 'getInt32', u32: 'getUint32' }[format]
  if (!method || buffer.byteLength % stride) throw Error('Invalid numeric output')
  const view = new DataView(buffer)
  return Array.from({ length: buffer.byteLength / stride }, (_, i) => view[method](i * stride, true))
}

export function decodeFrameOutput(buffer, width, height, frames) {
  if (buffer.byteLength !== width * height * frames * 4) throw Error('Invalid animation output size')
  return new Uint8ClampedArray(buffer)
}
