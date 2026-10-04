import { TileGPU } from './gpu.js';
import { resultFrame } from '/shared/protocol.js';
let gpu, socket, heartbeat, poll, stopped = false, visible = true, current = null, awaitingAck = false;
const send = message => { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ v: 1, ...message })); };
const report = (state, extra = {}) => self.postMessage({ state, ...extra });
function request(delay = 0) {
  clearTimeout(poll);
  if (stopped || !visible || current || awaitingAck) return;
  poll = setTimeout(() => send({ type: 'request_chunk' }), delay);
}
function reset(error) {
  if (stopped) return;
  stopped = true; clearInterval(heartbeat); clearTimeout(poll);
  socket?.close(); gpu?.destroy(); report('disconnected', { error });
}
async function connect(config) {
  visible = config.visible;
  try {
    gpu = new TileGPU();
    await gpu.init(message => {
      if (!stopped && current) send({ type: 'chunk_error', chunk_id: current.chunk_id,
        attempt_id: current.attempt_id, code: 'device_lost', error: message.slice(0, 2000) });
      reset(message);
    });
    // The coordinator exposes the immutable built-in shader ID in this manifest.
    const manifest = await fetch('/pool/manifest').then(r => { if (!r.ok) throw new Error('Manifest unavailable'); return r.json(); });
    gpu.capabilities.onnx = manifest.onnx_runtime_ready;
    let timer;
    const benchmark = await Promise.race([gpu.benchmark(manifest.shader_id), new Promise((_, reject) => {
      timer = setTimeout(() => { gpu.destroy(); reject(new Error('GPU benchmark timed out')); }, 15000);
    })]).finally(() => clearTimeout(timer));
    report('connecting', { capabilities: { ...gpu.capabilities, benchmark } });
    socket = new WebSocket(config.url);
    socket.onopen = () => send({ type: 'register', label: config.label, capabilities: { ...gpu.capabilities, benchmark } });
    socket.onclose = () => reset('Connection closed; unfinished work will be reassigned');
    socket.onerror = () => reset('Could not connect to coordinator');
    socket.onmessage = async event => {
      let chunk;
      try {
        chunk = JSON.parse(event.data);
        if (chunk.type === 'registered') {
          heartbeat = setInterval(() => send({ type: 'heartbeat', visible, attempt_id: current?.attempt_id || null }), chunk.heartbeat_ms);
          send({ type: visible ? 'resume' : 'pause' });
          send({ type: 'heartbeat', visible, attempt_id: null });
          report(visible ? 'idle' : 'paused', { worker_id: chunk.worker_id }); request(); return;
        }
        if (chunk.type === 'no_work') { report(visible ? 'idle' : 'paused', { detail: chunk.reason }); request(chunk.retry_after_ms); return; }
        if (chunk.type === 'result_ack') {
          awaitingAck = false;
          if (chunk.disposition === 'accepted') report('completed');
          if (chunk.disposition === 'stale') { reset('Lease expired; resetting GPU'); return; }
          report(visible ? 'idle' : 'paused'); request(); return;
        }
        if (chunk.type === 'cancel_attempt') { reset(chunk.reason); return; }
        if (chunk.type !== 'assign_chunk') return;
        if (current || awaitingAck) throw new Error('Worker received concurrent assignments');
        current = chunk; report('working', { tile: chunk.tile, kind: chunk.kind, offset: chunk.offset, count: chunk.count, frame_index: chunk.frame_index, job_id: chunk.job_id });
        send({ type: 'chunk_started', chunk_id: chunk.chunk_id, attempt_id: chunk.attempt_id });
        let timer;
        const result = await Promise.race([gpu.render(chunk), new Promise((_, reject) => {
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
    visible = message.visible; send({ type: visible ? 'resume' : 'pause' });
    send({ type: 'heartbeat', visible, attempt_id: current?.attempt_id || null });
    if (!current) report(visible ? 'idle' : 'paused');
    if (visible) request(); else clearTimeout(poll);
  }
  if (message.type === 'stop') { send({ type: 'stop' }); stopped = true; clearInterval(heartbeat); clearTimeout(poll); socket?.close(); gpu?.destroy(); }
};
