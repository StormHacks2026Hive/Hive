"""Run with two or more /node/ tabs already registered."""
import math
from backend.models import TypedArray
from demos.common import submit

KERNEL = '''def elementwise(offset: u32, count: u32, scale: f32, bias: f32, values: Array[f32], result: Array[f32]):
    i = global_id()
    if i < count:
        result[i] = values[i] * scale + bias + f32(offset + i)
'''

if __name__ == '__main__':
    values = [float(i % 1000) / 4 for i in range(1_000_000)]
    response = submit(dict(kernel=KERNEL, mode='data-slice', input=TypedArray.encode(values).model_dump(), parameters={'scale':2.0,'bias':1.0}))
    output = TypedArray(**response['result']).decode()
    assert response['total_chunks'] >= 2
    assert len(output) == len(values)
    assert all(math.isclose(v, x * 2 + 1 + i, rel_tol=1e-6, abs_tol=1e-5) for i, (x, v) in enumerate(zip(values, output)))
    print(f"Checked {len(output):,} results against Python; {response['total_chunks']} chunks")
