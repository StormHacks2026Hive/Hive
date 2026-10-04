// Identify the editor mode only; the server validates the complete shader.
export function sourceType(source) {
  const code = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '').trim()
  const wrapped = /\bWGSL_SHADER\s*=/.test(code)
  const compute = /@compute\b/.test(code)
  return {
    image: compute && /\btexture_storage_2d\s*</.test(code),
    rawWgsl: !wrapped && compute && /^(?:@|struct\b|fn\b|var(?:\s|<)|const\b|alias\b|override\b|enable\b|requires\b|diagnostic\b)/.test(code),
  }
}
