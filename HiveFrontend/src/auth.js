let googleScript
let initializedGoogle, initializedClient, credentialHandler

export function subscribeGoogleSignIn(google, clientId, callback) {
  // GIS has one global callback. Remounts replace the subscriber, not the SDK.
  credentialHandler = callback
  if (initializedGoogle !== google || initializedClient !== clientId) {
    google.initialize({ client_id: clientId, auto_select: false,
      callback: credential => credentialHandler?.(credential) })
    initializedGoogle = google
    initializedClient = clientId
  }
  return () => { if (credentialHandler === callback) credentialHandler = undefined }
}

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
