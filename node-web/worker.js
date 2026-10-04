import { TileGPU } from './gpu.js';
import { executeCPU, benchmarkCPU } from './cpu.js';
import { hardwareInfo } from './device-info.js';
import { resultFrame } from '/shared/protocol.js';
let gpu, socket, heartbeat, poll, stopped = false, visible = true, current = null, awaitingAck = false, enabled = true, phase = 'idle', gpuReady = false;
const send = message => { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ v: 1, ...message })); };
const report = (state, extra = {}) => { phase = state === 'readback' ? 'sending' : ['receiving','working'].includes(state) ? state : 'idle'; self.postMessage({ state, ...extra }); }; 
function request(delay = 0) {
  clearTimeout(poll);
  if (stopped || !visible || !enabled || current || awaitingAck) return;
  poll = setTimeout(() => send({ type: 'request_chunk' }), delay);
}
function reset(error, permanent = false) {
  if (stopped) return;
  stopped = true; clearInterval(heartbeat); clearTimeout(poll);
  socket?.close(); gpu?.destroy(); report(permanent ? 'stopped' : 'disconnected', { error, permanent });
}
async function connect(config) {
  visible = config.visible;
  try {
    const cpu_score = benchmarkCPU().score;
    let capabilities = { webgpu: false, adapter: {}, limits: null, features: [], benchmark: null, cpu: true, cpu_score, hardware: hardwareInfo(navigator), device_type: config.device_type || 'browser' };
    try {
      gpu = new TileGPU();
      await gpu.init(message => {
        if (!gpuReady || stopped) return;
        if (current) send({ type: 'chunk_error', chunk_id: current.chunk_id, attempt_id: current.attempt_id, code: 'device_lost', error: message.slice(0, 2000) });
        reset(message);
      });
      const manifest = await fetch('/pool/manifest').then(r => { if (!r.ok) throw Error('Manifest unavailable'); return r.json(); });
      let timer;
      const benchmark = await Promise.race([gpu.benchmark(manifest.shader_id), new Promise((_, reject) => {
        timer = setTimeout(() => reject(Error('GPU benchmark timed out')), 15000);
      })]).finally(() => clearTimeout(timer));
      gpuReady = true;
      capabilities = { ...capabilities, ...gpu.capabilities, onnx: manifest.onnx_runtime_ready, benchmark };
    } catch (error) {
      gpu?.destroy(); gpu = null;
      report('connecting', { detail: `CPU available. ${error.message}` });
    }
    report('connecting', { capabilities });
    socket = new WebSocket(config.url);
    socket.onopen = () => send({ type: 'register', label: config.label, capabilities, network_id: config.network_id || null, node_id: config.node_id || null });
    socket.onclose = event => reset(event.code === 4001 ? 'Node stopped' : 'Connection closed; unfinished work will be reassigned', event.code === 4001);
    socket.onerror = () => reset('Could not connect to coordinator');
    socket.onmessage = async event => {
      let chunk;
      try {
        chunk = JSON.parse(event.data);
        if (chunk.type === 'registered') {
          enabled = chunk.mode !== 'paused';
          heartbeat = setInterval(() => send({ type: 'heartbeat', visible, phase, attempt_id: current?.attempt_id || null }), chunk.heartbeat_ms);
          send({ type: visible && enabled ? 'resume' : 'pause' });
          send({ type: 'heartbeat', visible, phase, attempt_id: null });
          report(visible && enabled ? 'idle' : 'paused', { worker_id: chunk.worker_id }); request(); return;
        }
        if (chunk.type === 'node_control') { enabled = chunk.mode === 'running'; if (!current && !awaitingAck) report(enabled ? 'idle' : 'paused'); if (enabled) request(); else clearTimeout(poll); return; }
        if (chunk.type === 'no_work') { report(visible && enabled ? 'idle' : 'paused', { detail: chunk.reason }); request(chunk.retry_after_ms); return; }
        if (chunk.type === 'result_ack') {
          awaitingAck = false;
          if (chunk.disposition === 'accepted') report('completed');
          if (chunk.disposition === 'stale') { reset('Lease expired; resetting GPU'); return; }
          report(visible && enabled ? 'idle' : 'paused'); request(); return;
        }
        if (chunk.type === 'cancel_attempt') { reset(chunk.reason); return; }
        if (chunk.type !== 'assign_chunk') return;
        if (current || awaitingAck) throw new Error('Worker received concurrent assignments');
        current = chunk; report('receiving', { tile: chunk.tile, kind: chunk.kind, offset: chunk.offset, count: chunk.count, frame_index: chunk.frame_index, job_id: chunk.job_id });
        send({ type: 'chunk_started', chunk_id: chunk.chunk_id, attempt_id: chunk.attempt_id });
        report('working', { job_id: chunk.job_id, chunk_id: chunk.chunk_id, kind: chunk.kind });
        let timer;
        const execution = chunk.kind === 'cpu' ? Promise.resolve().then(() => executeCPU(chunk)) : gpu.render(chunk);
        const result = await Promise.race([execution, new Promise((_, reject) => {
          timer = setTimeout(() => { reject(new Error('Chunk timeout')); }, chunk.timeout_ms);
        })]).finally(() => clearTimeout(timer));
        if (stopped || socket.readyState !== WebSocket.OPEN) return;
        awaitingAck = true;
        socket.send(resultFrame({ v: 1, type: 'chunk_result', job_id: chunk.job_id,
          chunk_id: chunk.chunk_id, attempt_id: chunk.attempt_id, output_format: chunk.output_format,
          byte_length: result.pixels.byteLength, elapsed_ms: result.elapsed_ms }, result.pixels));
        report('readback', { elapsed_ms: result.elapsed_ms });
      } catch (e) {
        if (chunk?.chunk_id) send({ type: 'chunk_error', chunk_id: chunk.chunk_id,
          attempt_id: chunk.attempt_id, code: e.message === 'Chunk timeout' ? 'timeout' : 'execution', error: e.message.slice(0, 2000) });
        reset(e.message);
      } finally { if (chunk?.type === 'assign_chunk') current = null; }
    };
  } catch (e) { reset(e.message); }
}
self.onmessage = event => {
  const message = event.data;
  if (message.type === 'start') connect(message);
  if (message.type === 'visibility') {
    visible = message.visible; send({ type: visible && enabled ? 'resume' : 'pause' });
    send({ type: 'heartbeat', visible, phase, attempt_id: current?.attempt_id || null });
    if (!current) report(visible && enabled ? 'idle' : 'paused');
    if (visible) request(); else clearTimeout(poll);
  }
  if (message.type === 'stop') { send({ type: 'stop' }); stopped = true; clearInterval(heartbeat); clearTimeout(poll); socket?.close(); gpu?.destroy(); }
};
