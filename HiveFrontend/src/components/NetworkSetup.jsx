import { useState } from 'react'
import { Icon } from './HiveScene.jsx'

export default function NetworkSetup({ onConnect, knownNetworks, onRestore }) {
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState('join')
  const [networkName, setNetworkName] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')

  function chooseMode(value) {
    setMode(value)
    setNetworkName('')
    setPassword('')
    setError('')
  }

  async function submit(event) {
    event.preventDefault()
    const name = networkName.trim()
    if (!name) {
      setError(mode === 'join' ? 'Enter a network ID to join.' : 'Give your network a name.')
      return
    }
    if (password.length < 8) {
      setError('Use a password with at least 8 characters.')
      return
    }
    setBusy(true)
    try { await onConnect({ mode, name, password }) }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }

  return (
    <div className="network-setup">
      <div className="setup-copy">
        <h2>Hive Connection</h2>
        <p>
          Join your team or let them join you
          <br />
          parallelize your compute
        </p>
        <div className="mini-comb-art" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
          <span />
          <span />
          <span />
        </div>
      </div>
      <div className="setup-form-panel">
        {knownNetworks.length > 0 && <div className="saved-networks"><label htmlFor="saved-network">Your networks</label><select id="saved-network" value="" onChange={e => { const n = knownNetworks.find(n => n.id === e.target.value); if(n) onRestore(n) }}><option value="">Choose a saved network</option>{knownNetworks.map(n => <option key={n.id} value={n.id}>{n.name}</option>)}</select></div>}
        <div className="network-choices" aria-label="Choose how to connect">
          <button
            className={mode === "join" ? "choice selected" : "choice"}
            onClick={() => chooseMode("join")}
            aria-pressed={mode === "join"}
          >
            <Icon name="network" />
            <strong>Join a network</strong>
          </button>
          <button
            className={mode === "create" ? "choice selected" : "choice"}
            onClick={() => chooseMode("create")}
            aria-pressed={mode === "create"}
          >
            <Icon name="plus" />
            <strong>Create a network</strong>
          </button>
        </div>
        <form onSubmit={submit}>
          <label htmlFor="network-name">
            {mode === "join" ? "Network ID" : "Network name"}
          </label>
          <input
            id="network-name"
            value={networkName}
            onChange={(event) => {
              setNetworkName(event.target.value);
              setError("");
            }}
            placeholder={
              mode === "join"
                ? "e.g. HIVE-8F2A1B3C"
                : "e.g. Team Hive"
            }
            maxLength={64}
            required
            autoComplete="off"
          />
          <label htmlFor="network-password">Network password</label>
          <div className="password-input">
            <input
              id="network-password"
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(event) => {
                setPassword(event.target.value);
                setError("");
              }}
              placeholder="At least 8 characters"
              minLength={8}
              maxLength={128}
              required
              autoComplete={
                mode === "create" ? "new-password" : "current-password"
              }
            />
            <button
              type="button"
              aria-label={showPassword ? "Hide password" : "Show password"}
              aria-pressed={showPassword}
              onClick={() => setShowPassword(!showPassword)}
            >
              <Icon name="eye" />
            </button>
          </div>
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          <button className="button button-primary form-submit" type="submit" disabled={busy}>
            {mode === "join" ? "Connect to network" : "Create network"}
            <Icon name="arrow" />
          </button>
        </form>
      </div>
    </div>
  );
}
