import math
from .models import TypedArray


def aggregate(chunks, results, operation, dtype):
    ordered = [results[c.message.chunk_id] for c in sorted(chunks, key=lambda c: c.message.offset)]
    if operation is None:
        return TypedArray.encode([v for r in ordered for v in r.output.decode()], dtype)
    summaries = [r.summary for r in ordered]
    if operation in ('sum', 'count'):
        return math.fsum(s.value for s in summaries)
    if operation == 'min':
        return min(s.value for s in summaries)
    if operation == 'max':
        return max(s.value for s in summaries)
    return math.fsum(s.value for s in summaries) / sum(s.count for s in summaries)


def equivalent(a, b):
    if a.summary is not None and b.summary is not None:
        return a.summary.count == b.summary.count and math.isclose(a.summary.value, b.summary.value, rel_tol=1e-5, abs_tol=1e-5)
    if a.output is None or b.output is None or a.output.dtype != b.output.dtype:
        return False
    x, y = a.output.decode(), b.output.decode()
    return len(x) == len(y) and all(math.isclose(u, v, rel_tol=1e-5, abs_tol=1e-5) for u, v in zip(x, y))
