from demos.common import submit

KERNEL = '''def monte_carlo(offset: u32, count: u32, seed: u32, hits: Array[u32]):
    i = global_id()
    if i < count:
        state = (offset + i) * 747796405 + seed * 2891336453 + 1
        word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737
        rx = (word >> 22) ^ word
        state = state * 747796405 + 2891336453
        word2 = ((state >> ((state >> 28) + 4)) ^ state) * 277803737
        ry = (word2 >> 22) ^ word2
        x = f32(rx) / 4294967295.0
        y = f32(ry) / 4294967295.0
        hits[i] = u32(1) if x * x + y * y <= 1.0 else u32(0)
'''

if __name__ == '__main__':
    n = 1_000_000
    result = submit(dict(kernel=KERNEL, mode='parameter-only', count=n, reduce='sum', seed=12345))
    pi = 4 * result['result'] / n
    assert abs(pi - 3.141592653589793) < .02, pi
    print(f"pi ≈ {pi:.6f}; {result['total_chunks']} chunks; nodes received no input arrays")
