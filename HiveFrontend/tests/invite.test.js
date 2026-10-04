import assert from 'node:assert/strict'
import test from 'node:test'
import { inviteLink, readInvite } from '../src/invite.js'

test('network invites open the requested site and preserve guest join intent', () => {
  const link = new URL(inviteLink('HIVE-123456ABCDEF'))
  assert.equal(link.origin, 'https://hivehacks.tech')
  assert.deepEqual(readInvite(link.search), { networkId: 'HIVE-123456ABCDEF', guest: true })
  assert.deepEqual([...link.searchParams.keys()], ['join', 'guest'])
  assert.deepEqual(readInvite('?join=hive-123456abcdef'), { networkId: 'HIVE-123456ABCDEF', guest: false })
})

test('malformed and foreign invite IDs are ignored', () => {
  for (const search of ['', '?guest=1', '?join=https://other.example', '?join=HIVE-a%26password%3Db', `?join=HIVE-${'A'.repeat(60)}`]) {
    assert.equal(readInvite(search), null)
  }
  assert.throws(() => inviteLink('https://other.example'), /Invalid network ID/)
})
