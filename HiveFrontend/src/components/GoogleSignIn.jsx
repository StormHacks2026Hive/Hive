import { useEffect, useRef, useState } from 'react'
import { readGoogleProfile, loadGoogle } from '../auth.js'

export default function GoogleSignIn({ clientId, onSignIn, onError }) {
  const button = useRef(null)
  const [busy, setBusy] = useState(true)

  useEffect(() => {
    let active = true
    const container = button.current
    loadGoogle()
      .then((google) => {
        if (!active) return
        google.initialize({
          client_id: clientId,
          auto_select: false,
          callback: ({ credential }) => {
            if (!active) return
            onError('')
            try {
              onSignIn(readGoogleProfile(credential, clientId))
            } catch (err) {
              onError(err.message)
            }
          },
        })
        google.renderButton(container, {
          type: 'standard',
          theme: 'outline',
          size: 'large',
          shape: 'pill',
          text: 'continue_with',
          width: Math.min(240, container.parentElement.clientWidth),
        })
        setBusy(false)
      })
      .catch((err) => {
        if (active) {
          setBusy(false)
          onError(err.message)
        }
      })
    return () => {
      active = false
      container.replaceChildren()
    }
  }, [clientId, onSignIn, onError])

  return (
    <div className="signin-area" aria-busy={busy}>
      <div ref={button} className={busy ? 'google-button busy' : 'google-button'} inert={busy} />
      <p className="note" role="status">
        {busy ? 'Loading Google sign-in…' : 'Sign in to get buzzing.'}
      </p>
    </div>
  )
}
