import assert from 'node:assert/strict'
import test from 'node:test'
import { SESSION_KEY, clearSession, loadSession, saveSession } from '../src/session.js'
function memoryStorage() {
  const values = new Map()
  return {getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)}
}
const user = {id:'user-1',name:'Test User',email:'test@example.com'}
test('guest view preferences survive refresh without a made-up email',()=>{
 const storage=memoryStorage()
 const guest={id:'guest:123',name:'Guest',email:''}
 saveSession(guest,{network:{id:'HIVE-GUEST'},tab:'mapping'},storage)
 assert.deepEqual(loadSession(storage).user,guest)
 assert.equal(loadSession(storage).workspace.network.id,'HIVE-GUEST')
 saveSession({...user,email:''},null,storage)
 assert.equal(loadSession(storage),null)
})
test('restore preferences without persisting secrets or simulated network data',()=>{
  const storage=memoryStorage()
  saveSession({...user,credential:'secret-token'}, {network:{id:'HIVE-TEST',password:'secret-password'},tab:'compute',nodes:[{id:'fake'}],events:[{message:'fake'}],positions:{node:{x:400,y:300}},mapView:{selectedId:'node',filter:'online',zoom:1.2}},storage)
  const saved=loadSession(storage)
  assert.deepEqual(saved.user,user)
  assert.deepEqual(saved.workspace.network,{id:'HIVE-TEST'})
  assert.equal(saved.workspace.tab,'compute')
  assert.deepEqual(saved.workspace.positions,{node:{x:400,y:300}})
  assert.equal(saved.workspace.nodes,undefined)
  assert.equal(saved.workspace.events,undefined)
  assert.ok(!storage.getItem(SESSION_KEY).includes('secret'))
})
test('invalid data and old versions are rejected',()=>{
 const storage=memoryStorage()
 for(const data of ['{broken','null','{}',JSON.stringify({version:1,user}),JSON.stringify({version:2,user:{id:'bad'}})]) {
  storage.setItem(SESSION_KEY,data);assert.equal(loadSession(storage),null)
 }
})
test('repair view bounds and clear both cache versions',()=>{
 const storage=memoryStorage()
 saveSession(user,{network:null,tab:'invalid',positions:{node:{x:9999,y:-1}},mapView:{filter:'invalid',zoom:999}},storage)
 const saved=loadSession(storage)
 assert.equal(saved.workspace.tab,'network')
 assert.equal(saved.workspace.network,null)
 assert.deepEqual(saved.workspace.positions.node,{x:766,y:95})
 assert.equal(saved.workspace.mapView.zoom,1.35)
 storage.setItem('hive.session.v1','old');clearSession(storage)
 assert.equal(storage.getItem('hive.session.v1'),null)
 assert.equal(loadSession(storage),null)
})
test('blocked storage does not prevent using the app',()=>{
 const storage={getItem(){throw Error('blocked')},setItem(){throw Error('blocked')},removeItem(){throw Error('blocked')}}
 assert.equal(loadSession(storage),null);assert.equal(saveSession(user,null,storage),false)
 assert.doesNotThrow(()=>clearSession(storage))
})
