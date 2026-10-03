import { useState } from 'react'
import GoogleSignIn from './components/GoogleSignIn.jsx'

const clientId = (import.meta.env.VITE_GOOGLE_CLIENT_ID || '').trim()

function App() {
  const [user, setUser] = useState(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)

  function signOut() {
    window.google?.accounts?.id?.disableAutoSelect()
    setUser(null)
    setError('')
  }

  function tryAgain() {
    setError('')
    setRetry((value) => value + 1)
  }

  return (
    <div className="page">
      <header className="header">
        <a className="brand" href="/" aria-label="Hive home">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <path d="M16 3 27.3 9.5v13L16 29 4.7 22.5v-13Z" />
            <path d="m10 12 6-3.5 6 3.5v8l-6 3.5-6-3.5Z" />
          </svg>
          Hive<span className="brand-dot">.</span>
        </a>
        <span className="header-caption">Shared compute. Simple connections.</span>
      </header>
      <main>
        <section className="card" aria-labelledby="title">
          <div className="eyebrow"><span /> {user ? 'YOUR ACCOUNT' : 'WELCOME TO HIVE'}</div>
          <h1 id="title">{user ? `Hello, ${user.name.split(' ')[0]}.` : 'A little power.\nA bigger possibility.'}</h1>
          <p className="intro">{user ? 'You’re signed in and ready to go.' : 'A home for shared computing. Sign in to get started.'}</p>
          {user ? (
            <>
              <div className="account">
                <span className="avatar" aria-hidden="true">{user.name.slice(0, 1).toUpperCase()}</span>
                <div><strong>{user.name}</strong><span>{user.email}</span></div>
                <span className="account-check" aria-label="Signed in">✓</span>
              </div>
              <button className="logout" onClick={signOut}>
                Sign out
              </button>
            </>
          ) : clientId ? (
            <GoogleSignIn key={retry} clientId={clientId} onSignIn={setUser} onError={setError} />
          ) : !error ? (
            <div className="setup-state">
              <p>Google sign-in is coming soon.</p>
              <span>We’re getting things ready. Check back shortly.</span>
            </div>
          ) : null}
          {error && <div className="error" role="alert"><p>{error}</p><button onClick={tryAgain}>Try again</button></div>}
          <div className="card-footer"><span /> Better together.</div>
        </section>
      </main>
      <footer className="footer"><span>Hive</span><span>Built to bring computing together.</span></footer>
    </div>
  )
}

export default App
