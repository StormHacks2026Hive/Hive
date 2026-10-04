let configPromise;
export function authConfig(refresh = false) {
  if (refresh) configPromise = null;
  if (!configPromise) configPromise = fetch('/auth/config', { credentials: 'same-origin', cache: 'no-store' })
    .then(async response => { if (!response.ok) throw Error('Could not connect to Hive'); return response.json(); })
    .catch(error => { configPromise = null; throw error; });
  return configPromise;
}
export async function api(path, body) {
  const headers = {};
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    headers['X-CSRF-Token'] = (await authConfig()).csrf_token;
  }
  const response = await fetch(path, { credentials: 'same-origin', cache: 'no-store', headers,
    ...(body === undefined ? {} : { method: 'POST', body: JSON.stringify(body) }) });
  const value = await response.json();
  if (!response.ok) {
    if (response.status === 401 && !path.startsWith('/auth/')) window.dispatchEvent(new Event('hive:unauthorized'));
    throw Error(typeof value.detail === 'string' ? value.detail : JSON.stringify(value.detail || 'Request failed'));
  }
  return value;
}
export function networkApi(networkId, path, body) {
  return api(`${path}${path.includes('?') ? '&' : '?'}network_id=${encodeURIComponent(networkId)}`, body);
}
