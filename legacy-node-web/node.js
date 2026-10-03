import { GPUWorker } from './gpu.js';
const status = document.querySelector('#status'), error = document.querySelector('#error');
let completed = 0, retries = 0;
async function connect() {
  let socket, heartbeat, busy = false, stopped = false;
  const worker = new GPUWorker();
  try {
    await worker.init(message => {
      error.textContent = message;
      socket?.close();
    });
  } catch (e) { status.textContent = 'WebGPU unavailable'; error.textContent = e.message; return; }
  document.querySelector('#limits').textContent = JSON.stringify(worker.limits, null, 2);
  const url = new URL('/nodes', location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(url);
  const send = message => { if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message)); };
  socket.onopen = () => {
    send({ type: 'register', webgpu: true, limits: worker.limits });
    heartbeat = setInterval(() => send({ type: 'heartbeat' }), 5000);
  };
  socket.onmessage = async event => {
    let chunk;
    try {
      chunk = JSON.parse(event.data);
      if (chunk.type === 'registered') { retries = 0; status.textContent = 'Connected · idle'; return; }
      if (chunk.type !== 'assign_chunk') return;
      if (busy) throw new Error('Node is already working');
      busy = true; status.textContent = `Working · ${chunk.count.toLocaleString()} elements`;
      const result = await worker.run(chunk);
      if (!stopped) {
        send({ type: 'chunk_result', chunk_id: chunk.chunk_id, attempt_id: chunk.attempt_id, ...result });
        document.querySelector('#completed').textContent = String(++completed);
        status.textContent = 'Connected · idle';
      }
    } catch (e) {
      error.textContent = e.message;
      if (chunk?.chunk_id) send({ type: 'chunk_error', chunk_id: chunk.chunk_id, attempt_id: chunk.attempt_id, error: e.message.slice(0, 2000), device_lost: worker.lost });
      status.textContent = 'Connected · error';
    } finally { busy = false; }
  };
  socket.onclose = () => {
    stopped = true; clearInterval(heartbeat); worker.device.destroy();
    status.textContent = 'Disconnected · reconnecting';
    const delay = Math.min(30000, 1000 * 2 ** Math.min(retries++, 5)) + Math.random() * 500;
    setTimeout(connect, delay);
  };
  socket.onerror = () => { error.textContent = 'Connection failed; check that the Hive server is running.'; };
}
connect();
