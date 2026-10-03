// All device operations run inside a dedicated Web Worker.
export class TileGPU {
  async init(onLost) {
    if (!navigator.gpu) throw new Error('WebGPU unavailable in this browser. Use a supported browser on HTTPS or localhost.');
    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!this.adapter) throw new Error('No WebGPU adapter found.');
    const requested = {};
    // Request relevant maximum capacities, not minimum-alignment limits.
    for (const name of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxUniformBufferBindingSize',
      'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupsPerDimension']) {
      requested[name] = this.adapter.limits[name];
    }
    const features = this.adapter.features.has('shader-f16') ? ['shader-f16'] : [];
    this.device = await this.adapter.requestDevice({ requiredLimits: requested, requiredFeatures: features });
    this.device.lost.then(info => onLost(info.message || 'GPU device lost'));
    this.pipelines = new Map();
    const info = this.adapter.info || {};
    this.capabilities = {
      webgpu: true,
      adapter: Object.fromEntries(['vendor', 'architecture', 'description', 'device'].map(key => [key, String(info[key] || '').slice(0, 256)])),
      limits: Object.fromEntries(Object.keys(requested).map(key => [key, this.device.limits[key]])),
      features,
    };
  }
  async pipeline(shaderId) {
    if (this.pipelines.has(shaderId)) return this.pipelines.get(shaderId);
    const url = new URL(`/pool/assets/${shaderId}`, self.location.origin).href;
    let response, cache;
    // Cache failures are optional; private browsing/storage pressure must not block work.
    try { cache = await caches.open('hive-shaders-v1'); response = await cache.match(url); } catch { /* use fetch */ }
    if (!response) {
      response = await fetch(url);
      if (!response.ok) throw new Error('Shader download failed');
      try { await cache?.put(url, response.clone()); } catch { /* cache is best effort */ }
    }
    const source = await response.text();
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(source));
    const actual = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
    if (actual !== shaderId) throw new Error('Shader content hash mismatch');
    const d = this.device;
    d.pushErrorScope('validation');
    try {
      const module = d.createShaderModule({ code: source });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join('\n'));
      const pipeline = await d.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      this.pipelines.set(shaderId, pipeline);
      return pipeline;
    } finally {
      const error = await d.popErrorScope();
      if (error) throw new Error(error.message);
    }
  }
  async render(chunk) {
    const start = performance.now();
    const pipeline = await this.pipeline(chunk.shader_id);
    const d = this.device, buffers = [];
    d.pushErrorScope('validation');
    let popped = false;
    try {
      const make = (size, usage) => {
        const buffer = d.createBuffer({ size, usage }); buffers.push(buffer); return buffer;
      };
      const t = chunk.tile, image = chunk.image, p = chunk.parameters;
      const byteLength = t.width * t.height * 4;
      const bytes = new ArrayBuffer(48), view = new DataView(bytes);
      [t.x, t.y, t.width, t.height, image.width, image.height, p.max_iterations, 0].forEach((v, i) => view.setUint32(i * 4, v, true));
      [p.xmin, p.xmax, p.ymin, p.ymax].forEach((v, i) => view.setFloat32(32 + i * 4, v, true));
      const uniform = make(48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const output = make(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const readback = make(byteLength, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      d.queue.writeBuffer(uniform, 0, bytes);
      const bind = d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: output } },
      ] });
      const encoder = d.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8)); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, byteLength);
      d.queue.submit([encoder.finish()]);
      const validation = await d.popErrorScope(); popped = true;
      if (validation) throw new Error(validation.message);
      await readback.mapAsync(GPUMapMode.READ);
      const pixels = new Uint8Array(readback.getMappedRange().slice(0)); readback.unmap();
      return { pixels, elapsed_ms: Math.max(.01, performance.now() - start) };
    } finally {
      if (!popped) await d.popErrorScope().catch(() => {});
      buffers.forEach(b => b.destroy());
    }
  }
  async benchmark(shaderId) {
    const chunk = { shader_id: shaderId, tile: { x: 192, y: 192, width: 64, height: 64 },
      image: { width: 512, height: 512 }, parameters: { xmin: -2, xmax: 1, ymin: -1.5, ymax: 1.5, max_iterations: 256 } };
    await this.render(chunk); // warm shader/pipeline before measuring
    const result = await this.render(chunk);
    return { version: 'mandelbrot-v1', pixels: 4096, elapsed_ms: result.elapsed_ms };
  }
  destroy() { this.device?.destroy(); }
}
