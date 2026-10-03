const $ = id => document.getElementById(id);
let worker, running = false, wake, reconnect, attempts = 0, completed = 0;
$('label').value = localStorage.getItem('hive-worker-label') || `Laptop · ${navigator.platform || 'browser'}`;
async function wakeLock() {
  if (!running || document.visibilityState !== 'visible' || !navigator.wakeLock) return;
  try {
    wake = await navigator.wakeLock.request('screen'); $('wake').textContent = 'Screen wake lock active';
    wake.addEventListener('release', () => { $('wake').textContent = 'Wake lock released — keep the tab visible'; wake = null; });
  } catch { $('wake').textContent = 'Wake lock unavailable — keep this tab visible and your computer awake'; }
}
function spawn() {
  if (!running) return;
  worker?.terminate();
  const label = $('label').value.trim().slice(0, 80) || 'GPU teammate';
  localStorage.setItem('hive-worker-label', label);
  const url = new URL('/pool/nodes', location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  worker = new Worker('./worker.js', { type: 'module' });
  $('status').textContent = 'Initializing GPU and benchmarking…';
  worker.postMessage({ type: 'start', label, url: url.href, visible: document.visibilityState === 'visible' });
  worker.onmessage = event => {
    const m = event.data;
    if (m.capabilities) {
      $('capabilities').textContent = JSON.stringify(m.capabilities, null, 2);
      $('benchmark').textContent = `${m.capabilities.benchmark.elapsed_ms.toFixed(2)} ms / 4,096 pixels (dispatch + readback)`;
    }
    if (m.worker_id) { $('worker-id').textContent = m.worker_id; attempts = 0; $('error').textContent = ''; }
    if (m.state === 'completed') { $('completed').textContent = String(++completed); return; }
    const states = { idle: 'Connected · idle', paused: 'Connected · paused (tab hidden)', connecting: 'Connecting…',
      working: 'Working on GPU', readback: 'Awaiting server acceptance', disconnected: 'Disconnected · reconnecting' };
    $('status').textContent = states[m.state] || m.state;
    if (m.tile) $('last').textContent = `Tile (${m.tile.x}, ${m.tile.y}) · ${m.tile.width} × ${m.tile.height}`;
    if (m.elapsed_ms) $('timing').textContent = `${m.elapsed_ms.toFixed(2)} ms`;
    if (m.error) $('error').textContent = m.error;
    if (m.state === 'disconnected' && running) {
      reconnect = setTimeout(spawn, Math.min(30000, 1000 * 2 ** Math.min(attempts++, 5)) + Math.random() * 500);
    }
  };
  worker.onerror = event => {
    $('error').textContent = event.message || 'Worker failed';
    if (running) reconnect = setTimeout(spawn, 3000);
  };
}
$('start').onclick = () => {
  if (!navigator.gpu) { $('error').textContent = 'WebGPU is unavailable. Use a supported browser on HTTPS or localhost.'; return; }
  running = true; $('start').disabled = true; $('stop').disabled = false; $('label').disabled = true;
  wakeLock(); spawn();
};
$('stop').onclick = () => {
  running = false; clearTimeout(reconnect);
  worker?.postMessage({ type: 'stop' });
  // Give the socket a moment to send Stop, then terminate any stalled task.
  const old = worker; worker = null; setTimeout(() => old?.terminate(), 100);
  wake?.release(); $('status').textContent = 'Stopped'; $('start').disabled = false; $('stop').disabled = true; $('label').disabled = false;
};
document.addEventListener('visibilitychange', () => {
  worker?.postMessage({ type: 'visibility', visible: document.visibilityState === 'visible' });
  if (document.visibilityState === 'visible') wakeLock(); else wake?.release();
});
window.addEventListener('pagehide', () => { worker?.postMessage({ type: 'stop' }); worker?.terminate(); });
if (!navigator.gpu) { $('start').disabled = true; $('error').textContent = 'WebGPU is unavailable. Use a supported browser on HTTPS or localhost.'; }
