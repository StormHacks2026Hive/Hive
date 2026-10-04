import test from 'node:test'
import assert from 'node:assert/strict'
import { jobCalls, localDateTime, shellQuote } from '../src/automation.js'

test('API calls target the selected network, job, and complete run', () => {
  const task = { id: 'whole-task' }, job = { job_id: 'segment-2', kind: 'cpu' }
  const calls = jobCalls('http://localhost:8000', 'HIVE-TEST', task, job, 'hive_testkey')
  assert.match(calls.auth, /hive_testkey/)
  assert.match(calls.status, /\/api\/v1\/networks\/HIVE-TEST\/jobs\/segment-2/)
  assert.match(calls.result, /segment-2\/result/)
  assert.match(calls.repeat, /whole-task\/repeat/)
  assert.match(calls.python, /for job in run\["jobs"\]/)
  assert.match(calls.python, /raise_for_status/)
  assert.match(calls.python, /TimeoutError/)
})

test('image calls save raw frames and missing jobs do not invent IDs', () => {
  assert.match(jobCalls('http://localhost:8000', 'HIVE', { id: 'run' }, { job_id: 'image', kind: 'wgsl_image' }).result, /format=binary.*--output frames.rgba/)
  const empty = jobCalls('http://localhost:8000', 'HIVE')
  assert.equal(empty.status, '')
  assert.equal(empty.repeat, '')
  assert.match(empty.auth, /PASTE_YOUR_API_KEY/)
  assert.equal(shellQuote("it's"), "'it'\\''s'")
})

test('timer datetime preserves local wall clock time', () => {
  const date = new Date(2026, 9, 4, 12, 34)
  assert.equal(localDateTime(date.getTime()), '2026-10-04T12:34')
})
