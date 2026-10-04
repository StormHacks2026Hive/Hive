// Numeric IR only. No eval, Function constructor, Python imports, or ambient names.
export function executeCPU(chunk) {
  const start = performance.now(), p = chunk.cpu_program;
  const bytes = chunk.input ? Uint8Array.from(atob(chunk.input.data), c => c.charCodeAt(0)) : null;
  const input = bytes ? new DataView(bytes.buffer) : null;
  const output = new ArrayBuffer(chunk.count * 4), view = new DataView(output);
  let budget = 2_000_000;
  const binary = {
    add: (a,b)=>a+b, sub:(a,b)=>a-b, mul:(a,b)=>a*b,
    div:(a,b)=>{if(!b)throw Error('Division by zero');return a/b},
    floor_div:(a,b)=>{if(!b)throw Error('Division by zero');return Math.floor(a/b)},
    mod:(a,b)=>{if(!b)throw Error('Division by zero');return ((a%b)+b)%b},
    pow:(a,b)=>{if(Math.abs(b)>256)throw Error('Exponent too large');return a**b},
    eq:(a,b)=>a===b, ne:(a,b)=>a!==b, lt:(a,b)=>a<b, le:(a,b)=>a<=b, gt:(a,b)=>a>b, ge:(a,b)=>a>=b,
    bit_and:(a,b)=>a&b, bit_or:(a,b)=>a|b, bit_xor:(a,b)=>a^b, shift_left:(a,b)=>a<<b, shift_right:(a,b)=>a>>b,
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
      if(name==='pow')return binary.pow(...v);
      if(name==='round'){const scale=10**(v[1]||0), n=v[0]*scale, f=Math.floor(n);return (n-f===.5?(f%2===0?f:f+1):Math.round(n))/scale;}
      throw Error('Unsupported CPU call');
    }
    const a=run(args[0],env);
    if(op==='positive')return +a;if(op==='negative')return -a;if(op==='not')return !a;
    if(binary[op])return binary[op](a,run(args[1],env));
    throw Error('Unsupported CPU operation');
  }
  for(let i=0;i<chunk.count;i++) {
    const value=input?input.getFloat32(i*4,true):chunk.offset+i;
    const env=Object.create(null);env[p.arguments[0]]=p.indexed?chunk.offset+i:value;
    if(p.arguments[1])env[p.arguments[1]]=value;
    for(const step of p.steps) {
      if(step[0]==='set')env[step[1]]=run(step[2],env);
      else {const value=Number(run(step[1],env));if(!Number.isFinite(value)||Math.abs(value)>3.402823e38)throw Error('CPU output must fit float32');view.setFloat32(i*4,value,true);}
    }
  }
  return {pixels:new Uint8Array(output),elapsed_ms:Math.max(.01,performance.now()-start)};
}
export function benchmarkCPU() {
  const start=performance.now();let n=1;for(let i=0;i<250000;i++)n=(Math.imul(n,1664525)+1013904223)|0;
  return {score:250000/(Math.max(.01,performance.now()-start)/1000),checksum:n};
}
