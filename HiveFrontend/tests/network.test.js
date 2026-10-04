import assert from 'node:assert/strict'
import test from 'node:test'
import { STATUS, formatData } from '../src/network.js'
test('all actual contributor phases have descriptions',()=>{
 for(const phase of ['receiving','working','sending','idle','paused','offline'])assert.ok(STATUS[phase].label&&STATUS[phase].detail)
})
test('format persisted byte totals converted to MiB for the map',()=>{
 assert.equal(typeof formatData(0),'string');assert.ok(formatData(4).includes('4'))
})
