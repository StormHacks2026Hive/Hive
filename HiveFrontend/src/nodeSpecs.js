export function nodeSpecs(caps = {}) {
  const adapter = caps.adapter || {}, hardware = caps.hardware || {}, limits = caps.limits || {};
  const gpu = adapter.description || [adapter.vendor, adapter.architecture, adapter.device].filter(Boolean).join(' ') || 'Model hidden by browser';
  return [
    caps.webgpu ? `GPU: ${gpu}${adapter.is_fallback ? ' (software fallback)' : ''}` : 'GPU: unavailable; CPU only',
    caps.webgpu && caps.benchmark ? `GPU probe: ${caps.benchmark.pixels} pixels in ${caps.benchmark.elapsed_ms.toFixed(2)} ms (dispatch + readback)` : null,
    hardware.logical_cores ? `CPU: ${hardware.logical_cores} browser-reported logical cores` : 'CPU core count unavailable',
    hardware.memory_gib ? `Memory: ~${hardware.memory_gib} GiB (browser estimate)` : 'Memory unavailable',
    limits.maxStorageBufferBindingSize ? `GPU storage buffer: ${(limits.maxStorageBufferBindingSize / 1048576).toFixed(0)} MiB maximum` : null,
    limits.maxTextureDimension2D ? `GPU texture: ${limits.maxTextureDimension2D} × ${limits.maxTextureDimension2D} maximum` : null,
    hardware.platform ? `Platform: ${hardware.platform}` : null,
  ].filter(Boolean);
}
