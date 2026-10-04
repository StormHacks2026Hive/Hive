// Numeric IR only. No eval, Function constructor, Python imports, or ambient names.
export function executeCPU(chunk) {
  const start = performance.now(), p = chunk.cpu_program;
  const bytes = chunk.input ? Uint8Array.from(atob(chunk.input.data), c => c.charCodeAt(0)) : null;
  const input = bytes ? new DataView(bytes.buffer) : null;
  const output = new ArrayBuffer(chunk.count * 8), view = new DataView(output);
  let budget = 2_000_000;
  function divmod(a,b) {
    if(!b)throw Error('Division by zero');
    let remainder=a%b, quotient=(a-remainder)/b;
    if(remainder && (remainder<0)!==(b<0)){remainder+=b;quotient-=1;}
    let floor=Math.floor(quotient);
    if(quotient-floor>.5)floor+=1;
    return [floor,remainder===0?0:remainder];
  }
  const binary = {
    add: (a,b)=>a+b, sub:(a,b)=>a-b, mul:(a,b)=>a*b,
    div:(a,b)=>{if(!b)throw Error('Division by zero');return a/b},
    floor_div:(a,b)=>divmod(a,b)[0],
    mod:(a,b)=>divmod(a,b)[1],
    pow:(a,b)=>{if(Math.abs(b)>256)throw Error('Exponent too large');if(Number.isInteger(a)&&Number.isInteger(b)&&b>=0){const result=BigInt(a)**BigInt(b);if(result>BigInt(Number.MAX_SAFE_INTEGER)||result < -BigInt(Number.MAX_SAFE_INTEGER))throw Error('CPU arithmetic exceeds the supported numeric range');return Number(result);}return a**b},
    eq:(a,b)=>a===b, ne:(a,b)=>a!==b, lt:(a,b)=>a<b, le:(a,b)=>a<=b, gt:(a,b)=>a>b, ge:(a,b)=>a>=b,
  };
  function run(e, env) {
    if (--budget < 0) throw Error('CPU operation budget exceeded');
    const [op, ...args] = e;
    if (op === 'constant') return args[0];
    if (op === 'name') { if(!Object.hasOwn(env,args[0]))throw Error('Unknown CPU variable');return env[args[0]]; }
    if (op === 'if') return run(args[run(args[0],env)?1:2],env);
    if (op === 'and' || op === 'or') {let v;for(const a of args){v=run(a,env);if(op==='and'?!v:v)return v;}return v;}
    if (op === 'compare') {let a=run(args[1],env);for(let j=0;j<args[0].length;j++){const b=run(args[j+2],env);if(!binary[args[0][j]](a,b))return false;a=b;}return true;}
    if (op === 'call') {
      const [name,...exprs]=args, v=exprs.map(x=>run(x,env));
      if(name==='abs')return Math.abs(v[0]);if(name==='min')return Math.min(...v);if(name==='max')return Math.max(...v);
      if(name==='int')return Math.trunc(v[0]);if(name==='float')return Number(v[0]);if(name==='bool')return Boolean(v[0]);
      if(name==='pow'){const result=binary.pow(...v);if(!Number.isFinite(result)||Math.abs(result)>Number.MAX_SAFE_INTEGER)throw Error('CPU arithmetic exceeds the supported numeric range');return result;}
      throw Error('Unsupported CPU call');
    }
    const a=run(args[0],env);
    if(op==='positive')return +a;if(op==='negative')return -a;if(op==='not')return !a;
    if(binary[op]) { const value=binary[op](a,run(args[1],env)); if(typeof value === 'number' && (!Number.isFinite(value) || Math.abs(value)>Number.MAX_SAFE_INTEGER))throw Error('CPU arithmetic exceeds the supported numeric range'); return value; }
    throw Error('Unsupported CPU operation');
  }
  for(let i=0;i<chunk.count;i++) {
    const value=input?input.getFloat32(i*4,true):chunk.offset+i;
    const env=Object.create(null);env[p.arguments[0]]=p.indexed?chunk.offset+i:value;
    if(p.arguments[1])env[p.arguments[1]]=value;
    for(const step of p.steps) {
      if(step[0]==='set')env[step[1]]=run(step[2],env);
      else {const value=Number(run(step[1],env));if(!Number.isFinite(value)||Math.abs(value)>Number.MAX_SAFE_INTEGER)throw Error('CPU output exceeds the supported numeric range');view.setFloat64(i*8,value,true);}
    }
  }
  return {pixels:new Uint8Array(output),elapsed_ms:Math.max(.01,performance.now()-start)};
}
export function benchmarkCPU() {
  const start=performance.now();let n=1;for(let i=0;i<250000;i++)n=(Math.imul(n,1664525)+1013904223)|0;
  return {score:250000/(Math.max(.01,performance.now()-start)/1000),checksum:n};
}
