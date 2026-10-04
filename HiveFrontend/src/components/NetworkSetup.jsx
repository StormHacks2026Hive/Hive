import { useState } from 'react'
import { Icon } from './HiveScene.jsx'

export default function NetworkSetup({ onConnect, knownNetworks }) {
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

  function submit(event) {
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
    const known = knownNetworks.find((network) => network.id.toLowerCase() === name.toLowerCase())
    if (mode === 'join' && known && known.password !== password) {
      setError('That password doesn’t match this network.')
      return
    }
    const network =
      mode === 'create'
        ? {
            id: `HIVE-${crypto.randomUUID().slice(0, 8).toUpperCase()}`,
            name,
            password,
            owner: true,
          }
        : known || { id: name, name: name, password, owner: false }
    onConnect(network)
  }

  return (
    <div className="network-setup">
      <div className="setup-copy">
        <span className="section-kicker">A PLACE TO BEGIN</span>
        <h2>
          Good things{' '}
          <br />
          happen in a hive.
        </h2>
        <p>
          Join your people, or bring them together.
          <br />
          Your network starts with a single connection.
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
        <span className="setup-caption">Every node makes the network stronger.</span>
      </div>
      <div className="setup-form-panel">
        <div className="network-choices" aria-label="Choose how to connect">
          <button
            className={mode === 'join' ? 'choice selected' : 'choice'}
            onClick={() => chooseMode('join')}
            aria-pressed={mode === 'join'}
          >
            <Icon name="network" />
            <strong>Join a network</strong>
            <span>Find your existing hive</span>
          </button>
          <button
            className={mode === 'create' ? 'choice selected' : 'choice'}
            onClick={() => chooseMode('create')}
            aria-pressed={mode === 'create'}
          >
            <Icon name="plus" />
            <strong>Create a network</strong>
            <span>Make room for your people</span>
          </button>
        </div>
        <form onSubmit={submit}>
          <h3>{mode === 'join' ? 'Let’s get you connected.' : 'Make it your own.'}</h3>
          <p className="form-description">
            {mode === 'join'
              ? 'Enter the network ID and password shared with you.'
              : 'Choose a name and a password to share with your group.'}
          </p>
          <label htmlFor="network-name">{mode === 'join' ? 'Network ID' : 'Network name'}</label>
          <input
            id="network-name"
            value={networkName}
            onChange={(event) => {
              setNetworkName(event.target.value)
              setError('')
            }}
            placeholder={mode === 'join' ? 'e.g. HIVE-8F2A1B3C' : 'e.g. The garden collective'}
            maxLength={64}
            required
            autoComplete="off"
          />
          <label htmlFor="network-password">Network password</label>
          <div className="password-input">
            <input
              id="network-password"
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(event) => {
                setPassword(event.target.value)
                setError('')
              }}
              placeholder="At least 8 characters"
              minLength={8}
              maxLength={128}
              required
              autoComplete={mode === 'create' ? 'new-password' : 'current-password'}
            />
            <button
              type="button"
              aria-label={showPassword ? 'Hide password' : 'Show password'}
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
          <button className="button button-primary form-submit" type="submit">
            {mode === 'join' ? 'Connect to network' : 'Create network'}
            <Icon name="arrow" />
          </button>
          <p className="form-footnote">
            <Icon name="lock" /> Share your network ID and password with your group.
          </p>
        </form>
      </div>
    </div>
  )
}
