import { useEffect, useRef, useState } from 'react'
import { loadGoogle, subscribeGoogleSignIn } from '../auth.js'
import { api } from '../api.js'

export default function GoogleSignIn({ clientId, onSignIn, onError }) {
  const button = useRef(null)
  const [busy, setBusy] = useState(true)

  useEffect(() => {
    let active = true
    let resizeObserver
    let unsubscribe
    const container = button.current
    loadGoogle()
      .then((google) => {
        if (!active) return
        unsubscribe = subscribeGoogleSignIn(google, clientId, async ({ credential }) => {
          if (!active) return
          onError('')
          try {
            setBusy(true)
            const { user } = await api('/auth/google', { credential })
            if (active) onSignIn(user)
          } catch (err) {
            if (active) onError(err.message)
          } finally {
            if (active) setBusy(false)
          }
        })
        let lastWidth = 0
        const renderButton = () => {
          const width = Math.min(240, Math.floor(container.parentElement.clientWidth))
          if (width === lastWidth) return
          lastWidth = width
          container.replaceChildren()
          google.renderButton(container, {
            type: 'standard',
            theme: 'outline',
            size: 'large',
            shape: 'rectangular',
            text: 'continue_with',
            width,
          })
        }
        renderButton()
        resizeObserver = new ResizeObserver(renderButton)
        resizeObserver.observe(container.parentElement)
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
      unsubscribe?.()
      resizeObserver?.disconnect()
      container.replaceChildren()
    }
  }, [clientId, onSignIn, onError])

  return (
    <div className="signin-area" aria-busy={busy}>
      <div ref={button} className={busy ? 'google-button busy' : 'google-button'} inert={busy} />
      <p className="note" role="status">
        {busy ? 'Loading Google sign-in…' : 'Your Google account is all you need.'}
      </p>
    </div>
  )
}
