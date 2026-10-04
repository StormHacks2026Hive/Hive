import assert from 'node:assert/strict'
import test from 'node:test'
import { subscribeGoogleSignIn } from '../src/auth.js'

test('Google button remounts reuse initialization and route credentials to the current subscriber', () => {
  let config, calls = 0
  const google = { initialize(value) { config = value; calls++ } }
  const received = []
  const first = value => received.push(['first', value])
  const second = value => received.push(['second', value])
  const removeFirst = subscribeGoogleSignIn(google, 'client', first)
  const removeSecond = subscribeGoogleSignIn(google, 'client', second)
  removeFirst()
  config.callback('token')
  assert.equal(calls, 1)
  assert.deepEqual(received, [['second', 'token']])
  removeSecond()
  config.callback('after unmount')
  assert.equal(received.length, 1)
  subscribeGoogleSignIn(google, 'another-client', first)
  assert.equal(calls, 2)
})
