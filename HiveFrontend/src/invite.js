const NETWORK_ID = /^HIVE-[A-Z0-9]{1,59}$/

export function readInvite(search) {
  const params = new URLSearchParams(search)
  const networkId = (params.get('join') || '').trim().toUpperCase()
  return NETWORK_ID.test(networkId) ? { networkId, guest: params.get('guest') === '1' } : null
}

export function inviteLink(networkId) {
  if (!NETWORK_ID.test(networkId)) throw Error('Invalid network ID')
  const url = new URL('https://hivehacks.tech/')
  url.searchParams.set('join', networkId)
  url.searchParams.set('guest', '1')
  return url.href
}

export function clearInvite() {
  const url = new URL(window.location.href)
  url.searchParams.delete('join')
  url.searchParams.delete('guest')
  window.history.replaceState(window.history.state, '', url)
}
