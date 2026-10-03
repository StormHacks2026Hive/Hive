const types = { f32: Float32Array, u32: Uint32Array, i32: Int32Array };
export function encode(array) {
  const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
  let text = '';
  for (let i = 0; i < bytes.length; i += 32768) text += String.fromCharCode(...bytes.subarray(i, i + 32768));
  return btoa(text);
}
function decode(data, dtype) {
  const text = atob(data);
  const bytes = Uint8Array.from(text, c => c.charCodeAt(0));
  return new types[dtype](bytes.buffer);
}
export class GPUWorker {
  async init(onLost) {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Use a WebGPU-enabled browser on localhost or HTTPS.');
    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) throw new Error('No WebGPU adapter found.');
    this.device = await this.adapter.requestDevice();
    this.lost = false;
    this.device.lost.then(info => { this.lost = true; onLost(info.message || 'GPU device lost'); });
    const l = this.device.limits;
    this.limits = {
      max_buffer_size: l.maxBufferSize,
      max_storage_buffer_binding_size: l.maxStorageBufferBindingSize,
      max_compute_workgroup_size_x: l.maxComputeWorkgroupSizeX,
      max_compute_invocations_per_workgroup: l.maxComputeInvocationsPerWorkgroup,
      max_compute_workgroups_per_dimension: l.maxComputeWorkgroupsPerDimension,
      bandwidth_mbps: 20,
    };
  }
  async run(chunk) {
    if (this.lost) throw new Error('GPU device lost');
    let timer;
    try {
      return await Promise.race([
        this.compute(chunk),
        new Promise((_, reject) => { timer = setTimeout(() => {
          this.device.destroy();
          reject(new Error('Per-chunk GPU timeout; device destroyed'));
        }, chunk.timeout_ms); }),
      ]);
    } finally { clearTimeout(timer); }
  }
  async compute(chunk) {
    const d = this.device, buffers = [];
    d.pushErrorScope('validation');
    const make = (size, usage) => {
      const b = d.createBuffer({ size, usage }); buffers.push(b); return b;
    };
    let scopePopped = false;
    try {
      if (chunk.count < 1 || chunk.count > 2000000) throw new Error('Invalid chunk count');
      const entries = [];
      if (chunk.uniforms.length) {
        const bytes = new ArrayBuffer(Math.max(16, Math.ceil(chunk.uniforms.length * 4 / 16) * 16));
        const view = new DataView(bytes);
        chunk.uniforms.forEach((u, i) => {
          const value = chunk.parameters[u.name];
          if (!Number.isFinite(value)) throw new Error('Invalid uniform');
          view[{ f32: 'setFloat32', u32: 'setUint32', i32: 'setInt32' }[u.type]](i * 4, value, true);
        });
        const uniform = make(bytes.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        d.queue.writeBuffer(uniform, 0, bytes);
        entries.push({ binding: 0, resource: { buffer: uniform } });
      }
      let output, outputType;
      for (const b of chunk.bindings) {
        const buffer = make(chunk.count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        if (b.access === 'read') {
          if (chunk.chunk_type !== 'data_slice' || !chunk.input || chunk.input.dtype !== b.element_type) throw new Error('Invalid input binding');
          const input = decode(chunk.input.data, b.element_type);
          if (input.length !== chunk.count) throw new Error('Invalid input length');
          d.queue.writeBuffer(buffer, 0, input);
        } else { output = buffer; outputType = b.element_type; }
        entries.push({ binding: b.binding, resource: { buffer } });
      }
      const shader = d.createShaderModule({ code: chunk.wgsl });
      const info = await shader.getCompilationInfo();
      const errors = info.messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join('\n'));
      const pipeline = await d.createComputePipelineAsync({ layout: 'auto', compute: { module: shader, entryPoint: 'main' } });
      const group = d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      const readback = make(chunk.count * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const encoder = d.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(chunk.count / chunk.workgroup_size)); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, chunk.count * 4);
      d.queue.submit([encoder.finish()]);
      const validation = await d.popErrorScope(); scopePopped = true;
      if (validation) throw new Error(validation.message);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new types[outputType](readback.getMappedRange().slice(0)); readback.unmap();
      if (values.some(v => !Number.isFinite(v))) throw new Error('Non-finite GPU output');
      if (!chunk.reduce) return { output: { dtype: outputType, data: encode(values) } };
      let value = chunk.reduce === 'min' ? Infinity : chunk.reduce === 'max' ? -Infinity : 0;
      for (const v of values) {
        if (chunk.reduce === 'min') value = Math.min(value, v);
        else if (chunk.reduce === 'max') value = Math.max(value, v);
        else if (chunk.reduce === 'count') value += v !== 0 ? 1 : 0;
        else value += v; // mean sends sum and count, never an unweighted mean
      }
      return { summary: { value, count: values.length } };
    } finally {
      if (!scopePopped) await d.popErrorScope().catch(() => {});
      buffers.forEach(b => b.destroy());
    }
  }
}
