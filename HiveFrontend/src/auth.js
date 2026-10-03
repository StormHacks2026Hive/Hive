// Decode profile claims for this frontend-only UI. This does not verify the
// signature or establish a server session; never use it to authorize API access.
export function readGoogleProfile(credential, clientId) {
  try {
    const parts = credential.split('.')
    if (parts.length !== 3) throw new Error('Invalid credential')
    const base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/')
    const bytes = Uint8Array.from(atob(base64.padEnd(Math.ceil(base64.length / 4) * 4, '=')), (char) => char.charCodeAt(0))
    const profile = JSON.parse(new TextDecoder().decode(bytes))
    if (
      profile.aud !== clientId ||
      !['https://accounts.google.com', 'accounts.google.com'].includes(profile.iss) ||
      typeof profile.exp !== 'number' || profile.exp * 1000 <= Date.now() ||
      typeof profile.sub !== 'string' || !profile.sub ||
      typeof profile.email !== 'string' || !profile.email
    ) throw new Error('Invalid profile')
    return {
      id: profile.sub,
      name: typeof profile.name === 'string' && profile.name.trim() ? profile.name : profile.email,
      email: profile.email,
    }
  } catch {
    throw new Error('Could not read your Google account. Please try signing in again.')
  }
}

let googleScript
export function loadGoogle() {
  if (window.google?.accounts?.id) return Promise.resolve(window.google.accounts.id)
  if (!googleScript) {
    googleScript = new Promise((resolve, reject) => {
      const script = document.createElement('script')
      script.src = 'https://accounts.google.com/gsi/client'
      script.async = true
      const fail = () => {
        clearTimeout(timeout)
        script.remove()
        googleScript = undefined
        reject(new Error('Google sign-in could not load. Check your connection and try again.'))
      }
      const timeout = setTimeout(fail, 10000)
      script.onload = () => {
        clearTimeout(timeout)
        if (window.google?.accounts?.id) resolve(window.google.accounts.id)
        else fail()
      }
      script.onerror = fail
      document.head.appendChild(script)
    })
  }
  return googleScript
}
