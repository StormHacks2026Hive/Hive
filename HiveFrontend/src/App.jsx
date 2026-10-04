import { useCallback, useEffect, useRef, useState } from 'react'
import GoogleSignIn from './components/GoogleSignIn.jsx'
import HiveScene, { HangingHive, HexIcon, Icon } from './components/HiveScene.jsx'
import Dashboard from './components/Dashboard.jsx'

const clientId = (import.meta.env.VITE_GOOGLE_CLIENT_ID || '').trim()

function App() {
  const [user, setUser] = useState(null)
  const [stage, setStage] = useState('login')
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const heading = useRef(null)

  const enterHive = useCallback((profile) => {
    setUser(profile)
    setError('')
    setStage('opening')
  }, [])

  useEffect(() => {
    if (stage !== 'opening') return
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const timer = setTimeout(() => setStage('app'), reducedMotion ? 50 : 1200)
    return () => clearTimeout(timer)
  }, [stage])

  useEffect(() => {
    if (stage === 'app') heading.current?.focus()
  }, [stage])

  function signOut() {
    window.google?.accounts?.id?.disableAutoSelect()
    setUser(null)
    setError('')
    setStage('login')
  }

  if (stage === 'app') return <Dashboard user={user} onSignOut={signOut} headingRef={heading} />

  return (
    <div className={`login-page ${stage === 'opening' ? 'is-opening' : ''}`}>
      <HiveScene />
      <header className="login-header">
        <a className="brand" href="/" aria-label="Hive home">
          <HexIcon /> hive<span className="brand-dot">.</span>
        </a>
        <span>Shared power. Sweet possibilities.</span>
      </header>
      <main className="login-main">
        <section className="login-comb" aria-labelledby="login-title">
          <HangingHive />
          <div className="login-content">
            <h1 id="login-title">Find your hive.</h1>
            <p className="login-intro">A home for your shared computing power.</p>
            {stage === 'opening' ? (
              <div className="entering-message" role="status">
                <Icon name="check" /> Welcome in, {user.name.split(' ')[0]}.
              </div>
            ) : clientId ? (
              <GoogleSignIn
                key={retry}
                clientId={clientId}
                onSignIn={enterHive}
                onError={setError}
              />
            ) : (
              <p className="setup-state">
                Google sign-in is being set up.
                <br />
                Come back soon.
              </p>
            )}
            {error && (
              <div className="error" role="alert">
                <p>{error}</p>
                <button
                  onClick={() => {
                    setError('')
                    setRetry((value) => value + 1)
                  }}
                >
                  Try again
                </button>
              </div>
            )}
          </div>
        </section>
        <div className="honey-swatches" aria-hidden="true"><span /><span /><span /></div>
      </main>
      <footer className="login-footer">
        <span>A home for shared computing.</span>
        <span>
          Made to work together <HexIcon />
        </span>
      </footer>
    </div>
  )
}

export default App
