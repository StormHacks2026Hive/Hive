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
    this.sessions = new Map();
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
      if (this.pipelines.size >= 8) this.pipelines.delete(this.pipelines.keys().next().value);
      this.pipelines.set(shaderId, pipeline);
      return pipeline;
    } finally {
      const error = await d.popErrorScope();
      if (error) throw new Error(error.message);
    }
  }
  async render(chunk) {
    if (chunk.kind === 'texture_tile') return this.textureRender(chunk);
    if (chunk.kind === 'compute') return this.compute(chunk);
    if (chunk.kind === 'onnx_batch') return this.infer(chunk);
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
  async textureRender(chunk) {
    const start = performance.now(), pipeline = await this.pipeline(chunk.shader_id);
    const d = this.device, buffers = [];
    let texture, popped = false;
    d.pushErrorScope('validation');
    try {
      const make = (size, usage) => {
        const buffer = d.createBuffer({ size, usage }); buffers.push(buffer); return buffer;
      };
      const t = chunk.tile;
      const bytes = Uint8Array.from(atob(chunk.uniform_data), c => c.charCodeAt(0));
      if (bytes.length !== 176) throw new Error('Renderer uniform size mismatch');
      const uniform = make(176, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const tile = make(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      d.queue.writeBuffer(uniform, 0, bytes);
      d.queue.writeBuffer(tile, 0, new Uint32Array([t.x, t.y, 0, 0, t.width, t.height, 1, 0]));
      texture = d.createTexture({ size: [t.width, t.height], format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
      const stride = Math.ceil(t.width * 4 / 256) * 256;
      const readback = make(stride * t.height, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const bind = d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: { buffer: uniform } },
        { binding: 2, resource: { buffer: tile } },
      ] });
      const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8)); pass.end();
      encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: stride }, [t.width, t.height]);
      d.queue.submit([encoder.finish()]);
      const validation = await d.popErrorScope(); popped = true;
      if (validation) throw new Error(validation.message);
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = new Uint8Array(readback.getMappedRange()), pixels = new Uint8Array(t.width * t.height * 4);
      for (let y = 0; y < t.height; y++) pixels.set(mapped.subarray(y * stride, y * stride + t.width * 4), y * t.width * 4);
      readback.unmap();
      return { pixels, elapsed_ms: Math.max(.01, performance.now() - start) };
    } finally {
      if (!popped) await d.popErrorScope().catch(() => {});
      buffers.forEach(buffer => buffer.destroy()); texture?.destroy();
    }
  }
  async compute(chunk) {
    const start = performance.now(), pipeline = await this.pipeline(chunk.shader_id);
    const d = this.device, buffers = [];
    d.pushErrorScope('validation');
    let popped = false;
    try {
      const make = (size, usage) => {
        const buffer = d.createBuffer({ size, usage }); buffers.push(buffer); return buffer;
      };
      const uniformBytes = new ArrayBuffer(Math.max(16, Math.ceil(chunk.uniforms.length * 4 / 16) * 16));
      const view = new DataView(uniformBytes);
      chunk.uniforms.forEach((u, index) => {
        const method = { u32: 'setUint32', i32: 'setInt32', f32: 'setFloat32' }[u.type];
        view[method](index * 4, chunk.compute_parameters[u.name], true);
      });
      const uniform = make(uniformBytes.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      d.queue.writeBuffer(uniform, 0, uniformBytes);
      const entries = [{ binding: 0, resource: { buffer: uniform } }];
      const byteLength = chunk.count * 4;
      let output;
      for (const binding of chunk.bindings) {
        const isOutput = binding.access === 'read_write';
        const buffer = make(byteLength, GPUBufferUsage.STORAGE | (isOutput ? GPUBufferUsage.COPY_SRC : GPUBufferUsage.COPY_DST));
        if (isOutput) output = buffer;
        else {
          const bytes = Uint8Array.from(atob(chunk.input.data), c => c.charCodeAt(0));
          if (bytes.byteLength !== byteLength) throw new Error('Input buffer length mismatch');
          d.queue.writeBuffer(buffer, 0, bytes);
        }
        entries.push({ binding: binding.binding, resource: { buffer } });
      }
      const readback = make(byteLength, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const bind = d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(chunk.count / 64)); pass.end();
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
  async infer(chunk) {
    const start = performance.now();
    if (!this.ort) {
      this.ort = await import('/pool/runtime/ort.webgpu.bundle.min.mjs');
      this.ort.env.wasm.numThreads = 1;
      this.ort.env.wasm.proxy = false;
      this.ort.env.wasm.wasmPaths = '/pool/runtime/';
      this.ort.env.webgpu.device = this.device;
    }
    let session = this.sessions.get(chunk.model_id);
    if (!session) {
      const url = `/pool/assets/${chunk.model_id}`;
      let response, cache;
      try { cache = await caches.open('hive-models-v1'); response = await cache.match(url); } catch { /* optional cache */ }
      if (!response) {
        response = await fetch(url);
        if (!response.ok) throw new Error('Model download failed');
        try { await cache?.put(url, response.clone()); } catch { /* optional cache */ }
      }
      const bytes = await response.arrayBuffer();
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      if (Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('') !== chunk.model_id) throw new Error('Model content hash mismatch');
      if (this.sessions.size >= 2) {
        const key = this.sessions.keys().next().value;
        await this.sessions.get(key).release(); this.sessions.delete(key);
      }
      session = await this.ort.InferenceSession.create(bytes, {
        executionProviders: ['webgpu'], extra: { session: { disable_cpu_ep_fallback: '1' } },
      });
      this.sessions.set(chunk.model_id, session);
    }
    const bytes = Uint8Array.from(atob(chunk.input.data), c => c.charCodeAt(0));
    const input = new this.ort.Tensor('float32', new Float32Array(bytes.buffer), chunk.input_shape);
    let outputs;
    try {
      outputs = await session.run({ [chunk.input_name]: input });
      const result = outputs[chunk.output_name];
      if (result.type !== 'float32' || JSON.stringify(result.dims) !== JSON.stringify(chunk.output_shape)) throw new Error('ONNX output shape/type mismatch');
      const data = await result.getData();
      const pixels = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
      return { pixels, elapsed_ms: Math.max(.01, performance.now() - start) };
    } finally {
      input.dispose();
      Object.values(outputs || {}).forEach(tensor => tensor.dispose());
    }
  }
  async benchmark(shaderId) {
    const chunk = { shader_id: shaderId, tile: { x: 192, y: 192, width: 64, height: 64 },
      image: { width: 512, height: 512 }, parameters: { xmin: -2, xmax: 1, ymin: -1.5, ymax: 1.5, max_iterations: 256 } };
    await this.render(chunk); // warm shader/pipeline before measuring
    const result = await this.render(chunk);
    return { version: 'mandelbrot-v1', pixels: 4096, elapsed_ms: result.elapsed_ms };
  }
  destroy() {
    for (const session of this.sessions?.values() || []) session.release().catch(() => {});
    this.sessions?.clear(); this.device?.destroy();
  }
}
