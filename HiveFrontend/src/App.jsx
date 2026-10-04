import { useCallback, useEffect, useRef, useState } from 'react'
import GoogleSignIn from './components/GoogleSignIn.jsx'
import HiveScene, { HangingHive, HexIcon, Icon } from './components/HiveScene.jsx'
import Dashboard from './components/Dashboard.jsx'
import { clearSession, loadSession, saveSession } from './session.js'
import useUITransition from './useUITransition.js'
import { api, authConfig } from './api.js'
import { clearInvite, readInvite } from './invite.js'

function App() {
  const [saved, setSaved] = useState(loadSession)
  const [user, setUser] = useState(null)
  const [stage, setStage] = useState('checking')
  const [clientId, setClientId] = useState('')
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const [guestBusy, setGuestBusy] = useState(false)
  const [invitation, setInvitation] = useState(() => readInvite(window.location.search))
  const startupInvite = useRef(invitation)
  const initialSession = useRef(null)
  const heading = useRef(null)
  const savingEnabled = useRef(Boolean(saved?.user))
  const transition = useUITransition()

  useEffect(() => {
    let active = true
    if (!initialSession.current) initialSession.current = Promise.all([authConfig(), api('/auth/me')])
      .then(async ([config, session]) => {
        if (!session.user && startupInvite.current?.guest) session = await api('/auth/guest', {})
        return [config, session]
      })
    initialSession.current.then(([config, session]) => {
      if (!active) return
      setClientId(config.client_id)
      setUser(session.user)
      savingEnabled.current = Boolean(session.user)
      setStage(session.user ? 'app' : 'login')
      if (!session.user) clearSession()
    }).catch(e => { if (active) { setError(e.message); setStage('login') } })
    const expired = () => { savingEnabled.current = false; clearSession(); setUser(null); setStage('login'); setError('Your session ended. Sign in again.') }
    window.addEventListener('hive:unauthorized', expired)
    return () => { active = false; window.removeEventListener('hive:unauthorized', expired) }
  }, [])

  const consumeInvite = useCallback(() => {
    clearInvite()
    setInvitation(null)
  }, [])

  const saveWorkspace = useCallback((workspace) => {
    if (savingEnabled.current && user) saveSession(user, workspace)
  }, [user])

  const enterHive = useCallback((profile) => {
    savingEnabled.current = true
    setUser(profile)
    setSaved(null)
    saveSession(profile, null)
    setError('')
    setStage('opening')
  }, [])

  useEffect(() => {
    if (stage !== 'opening') return
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const timer = setTimeout(() => transition(() => setStage('app')), reducedMotion ? 50 : 800)
    return () => clearTimeout(timer)
  }, [stage, transition])

  useEffect(() => {
    if (stage === 'app') heading.current?.focus()
  }, [stage])

  async function signOut() {
    await api('/auth/logout', {})
    savingEnabled.current = false
    window.google?.accounts?.id?.disableAutoSelect()
    clearSession()
    transition(() => {
      setUser(null)
      setSaved(null)
      setError('')
      setStage('login')
    })
  }

  async function enterGuest() {
    setGuestBusy(true)
    setError('')
    try {
      const session = await api('/auth/guest', {})
      enterHive(session.user)
    } catch (e) { setError(e.message) }
    finally { setGuestBusy(false) }
  }

  if (stage === 'app') return (
    <Dashboard user={user} initialState={saved?.user.id === user.id ? saved.workspace : null}
      invitedNetwork={invitation?.networkId} onInviteJoined={consumeInvite}
      onWorkspaceChange={saveWorkspace} onSignOut={signOut} headingRef={heading} />
  )

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
            {stage === 'checking' ? <p className="setup-state" role="status">Connecting…</p> : stage === 'opening' ? (
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
            {stage === 'login' && (
              <button className="guest-signin" onClick={enterGuest} disabled={guestBusy}>
                {guestBusy ? 'Opening…' : 'Continue as guest'}
              </button>
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
