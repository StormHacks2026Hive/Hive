"""Conservative WGSL range analysis for a small elementwise shader grammar.

General WGSL is compiled by WebGPU. This analyzer never claims that arbitrary
shaders with synchronization or dependent writes can be partitioned safely.
"""
import re
import math
from typing import Literal
from pydantic import Field
from .models import Model

class WGSLAnalysisRequest(Model):
    source: str = Field(min_length=1, max_length=32000)
    count: int = Field(default=1, ge=1, le=2_000_000)
    chunk_size: int = Field(default=4096, ge=1, le=4096)

class WGSLAnalysis(Model):
    status: Literal['ready', 'manual_contract_required', 'unsupported']
    findings: list[str]
    wgsl: str | None = None
    input_dtype: str | None = None
    output_dtype: str | None = None
    chunk_count: int = 0
    ranges: list[dict[str, int]] = Field(default_factory=list)

HEADER = '''struct HiveParams { offset: u32, count: u32, seed: u32, padding: u32, }
@group(0) @binding(0) var<uniform> hive_params: HiveParams;
'''


def analyze_wgsl(request: WGSLAnalysisRequest):
    if math.ceil(request.count / request.chunk_size) > 2048:
        return WGSLAnalysis(status='unsupported', findings=['Maximum 2048 chunks; increase chunk_size'])
    code = re.sub(r'/\*.*?\*/|//[^\n]*', '', request.source, flags=re.S).strip()
    findings = []
    if re.search(r'\b(atomic\w*|workgroupBarrier|storageBarrier|texture\w*|workgroup)\b', code):
        return WGSLAnalysis(status='unsupported', findings=['Shared memory, atomics, textures, and barriers require a manual algorithm rewrite before range splitting'])
    # Only recognize complete modules consisting of 0/1 inputs, one output,
    # and a straight-line compute function. Every token must be accounted for.
    declaration = r'@group\(\s*0\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var<\s*storage\s*,\s*(read|read_write)\s*>\s*(\w+)\s*:\s*array<\s*(f32|u32|i32)\s*>\s*;'
    buffers = re.findall(declaration, code)
    remainder = re.sub(declaration, '', code).strip()
    main = re.fullmatch(r'@compute\s*@workgroup_size\(\s*(\d+)(?:\s*,\s*1\s*,\s*1)?\s*\)\s*fn\s+main\s*\(\s*@builtin\(global_invocation_id\)\s*(\w+)\s*:\s*vec3<u32>\s*\)\s*\{(.*)\}', remainder, flags=re.S)
    reads = [b for b in buffers if b[1] == 'read']
    writes = [b for b in buffers if b[1] == 'read_write']
    if not main or len(reads) > 1 or len(writes) != 1 or len({b[0] for b in buffers}) != len(buffers) or len({b[2] for b in buffers}) != len(buffers):
        return WGSLAnalysis(status='manual_contract_required', findings=['Automatic analysis accepts a complete shader with one writable array, at most one read-only array, and a straight-line 1D main function. For other independent algorithms use the declared Hive binding contract.'])
    gid, body = main[2], main[3].strip()
    reserved = {gid, *(b[2] for b in buffers), 'hive_params', 'HiveParams', 'hive_local_index', 'hive_gid', 'hive_input_value', 'hive_global_index'}
    if len(reserved) != len(buffers)+7:
        return WGSLAnalysis(status='unsupported', findings=['Rename identifiers that collide with HiveParams or hive_params'])
    index = re.match(r'(?:let|var)\s+(\w+)(?:\s*:\s*u32)?\s*=\s*' + re.escape(gid) + r'\.x\s*;', body)
    local_index = index[1] if index else None
    if index:
        body = body[index.end():].strip()
        if local_index in reserved:
            return WGSLAnalysis(status='unsupported', findings=['The local index must have a distinct name'])
    # Strip one conventional bounds guard; add the actual per-chunk count guard.
    guard = re.fullmatch(r'if\s*\(?\s*(.*?)\s*\)?\s*\{(.*)\}', body, flags=re.S)
    if guard:
        expected = (local_index or gid+'.x') + r'\s*<\s*arrayLength\(\s*&\s*' + re.escape(writes[0][2]) + r'\s*\)'
        if not re.fullmatch(expected, guard[1].strip()):
            return WGSLAnalysis(status='unsupported', findings=['Only an output arrayLength guard can be rewritten automatically'])
        body = guard[2].strip()
    statements = [s.strip() for s in body.split(';') if s.strip()]
    allowed_names = set()
    builtins = {'abs','min','max','clamp','sqrt','sin','cos','exp','log','f32','u32','i32'}
    out_name, out_dtype = writes[0][2:]
    normalized = []
    input_used = False
    for number, statement in enumerate(statements):
        expr = None
        local = re.fullmatch(r'let\s+(\w+)(?:\s*:\s*(f32|u32|i32))?\s*=\s*(.*)', statement, flags=re.S)
        target = re.fullmatch(re.escape(out_name) + r'\[\s*(.*?)\s*\]\s*=\s*(.*)', statement, flags=re.S)
        if local and number < len(statements)-1 and local[1] not in reserved | allowed_names | builtins | {local_index}:
            expr = local[3]
        elif target and number == len(statements)-1 and target[1] in (local_index, gid+'.x'):
            expr = target[2]
        else:
            findings.append('Use iteration-local let assignments followed by exactly one output[index] write')
            break
        if re.search(r'\bhive_(?:input_value|global_index|local_index|gid|params)\b', expr):
            findings.append('Rename identifiers reserved for Hive shader rewriting')
            break
        # Replace only exact current-element reads. No neighboring reads,
        # output reads, pointer operations, assignment expressions, or captures.
        if reads:
            input_name = reads[0][2]
            pattern = re.escape(input_name) + r'\[\s*(' + re.escape(local_index or gid+'.x') + r')\s*\]'
            expr, replacements = re.subn(pattern, 'hive_input_value', expr)
            input_used = input_used or replacements > 0
        expr = re.sub(r'\b' + re.escape(gid) + r'\.x\b', 'hive_global_index', expr)
        if local_index:
            expr = re.sub(r'\b' + re.escape(local_index) + r'\b', 'hive_global_index', expr)
        tokens = re.findall(r'(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[fui]?)|(?:[A-Za-z_]\w*)|[()+*/%,\-]', expr)
        if ''.join(tokens) != re.sub(r'\s+', '', expr) or any(re.match(r'[A-Za-z_]', t) and t not in allowed_names | builtins | {'hive_input_value','hive_global_index'} for t in tokens):
            findings.append('Expressions may read only the current input element, global index, numeric constants, scalar math, and previously assigned local values')
            break
        expr = expr.replace('hive_global_index', '(hive_params.offset + hive_local_index)')
        if reads:
            expr = expr.replace('hive_input_value', f'{reads[0][2]}[hive_local_index]')
        if local:
            normalized.append(f'let {local[1]}' + (f': {local[2]}' if local[2] else '') + f' = {expr};')
            allowed_names.add(local[1])
        else:
            normalized.append(f'{out_name}[hive_local_index] = {expr};')
    if reads and not input_used:
        findings.append('Remove the unused input array declaration; it would not appear in the WebGPU binding layout')
    if findings or not statements:
        return WGSLAnalysis(status='unsupported', findings=findings or ['The shader must write one output per invocation'])
    declaration_code = ''
    if reads:
        declaration_code += f'@group(0) @binding(1) var<storage, read> {reads[0][2]}: array<{reads[0][3]}>;\n'
    declaration_code += f'@group(0) @binding(2) var<storage, read_write> {out_name}: array<{out_dtype}>;\n'
    rewritten = HEADER + declaration_code + '@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) hive_gid: vec3<u32>) {\n    let hive_local_index = hive_gid.x;\n    if (hive_local_index < hive_params.count) {\n' + ''.join(f'        {s}\n' for s in normalized) + '    }\n}\n'
    ranges = [{'offset':o, 'count':min(request.chunk_size,request.count-o)} for o in range(0,min(request.count,64*request.chunk_size),request.chunk_size)]
    return WGSLAnalysis(status='ready', findings=['Each invocation writes one distinct output element. Input reads are local to the same element; no inter-chunk dependencies were found.', 'The shader was rewritten with local buffer indexing, global offsets, a count guard, and workgroups of 64. Browser WebGPU compilation is still required.'], wgsl=rewritten, input_dtype=reads[0][3] if reads else None, output_dtype=out_dtype, chunk_count=math.ceil(request.count/request.chunk_size), ranges=ranges)
