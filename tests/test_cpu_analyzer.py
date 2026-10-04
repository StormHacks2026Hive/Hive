"""Conservative CPU AST findings and reduction correctness."""

import pytest
from backend.cpu_analyzer import analyze, run
from backend.common.nodes import ComputeNode

SUM = """def compute(values):
    total = 7
    for i in range(len(values)):
        x = values[i] * values[i]
        total += x
    return total
"""


def test_reduction_and_serial_fallback():
    report = analyze(SUM)
    assert report.splittable and report.reduction == "sum"
    assert run(SUM, list(range(20)), [ComputeNode("one", 1)]) == 7 + sum(
        i * i for i in range(20)
    )


@pytest.mark.parametrize(
    "source,reason",
    [
        (SUM.replace("values[i] * values[i]", "values[i-1]"), "Neighbor"),
        (SUM.replace("    total = 7", "    global total\n    total = 7"), "global"),
        (
            SUM.replace("        total += x", "        print(x)\n        total += x"),
            "I/O",
        ),
        (SUM.replace("        total += x", "        total += total*x"), "Captured"),
        (SUM.replace("        x =", "        values[i] ="), "Mutation"),
    ],
)
def test_negative_reports(source, reason):
    report = analyze(source)
    assert not report.splittable and reason.lower() in " ".join(report.reasons).lower()


def test_comprehension_append_min_max_empty():
    assert run(
        "def f(values):\n    return [abs(x) for x in values]",
        [-2, 1],
        [ComputeNode("one", 1)],
    ) == [2, 1]
    for op, initial, expected in [("min", 100, 1), ("max", -100, 9)]:
        source = f"def f(values):\n    total = {initial}\n    for x in values:\n        total = {op}(total, x)\n    return total"
        assert run(source, [1, 9], [ComputeNode("one", 1)]) == expected
        assert run(source, [], [ComputeNode("one", 1)]) == initial


def test_ray_unequal_chunks():
    ray = pytest.importorskip("ray")
    values = list(range(1000))
    try:
        assert run(SUM, values, [ComputeNode("a", 3), ComputeNode("b", 1)]) == 7 + sum(
            i * i for i in values
        )
    finally:
        ray.shutdown()


def sample_hit(index):
    """Deterministic per-index RNG; no global random generator state."""
    value = (index * 747796405 + 2891336453) & 0xFFFFFFFF
    for _ in range(16):
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    x = value / 4294967296
    value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    y = value / 4294967296
    return int(x * x + y * y <= 1)


PI = """def pi(samples):
    hits = 0
    for i in range(samples):
        hits += sample_hit(i)
    return hits
"""


@pytest.mark.slow
def test_monte_carlo_benchmark():
    import os
    import ray
    from tests.benchmark_support import calibrate, report, check_speedup
    from backend.node_ranker import benchmark_cpu

    workers = min(4, os.cpu_count() or 1)
    measured = benchmark_cpu(cores=1)
    nodes = [ComputeNode(f"cpu-{i}", measured.node.score) for i in range(workers)]
    serial = lambda n: sum(sample_hit(i) for i in range(n))
    try:
        # Exclude one-time Ray startup from the parallel timing, include dispatch/merge.
        ray.init(num_cpus=workers, include_dashboard=False)
        run(PI, 10, nodes, {"sample_hit": sample_hit})
        size, baseline_output, baseline = calibrate(serial, 100_000)
        actual, plan, elapsed, finishes, idle = run(
            PI, size, nodes, {"sample_hit": sample_hit}, return_metrics=True
        )
        assert actual == baseline_output
        result = report(
            "cpu_monte_carlo",
            baseline,
            elapsed,
            {p.node.node_id: p.stop - p.start for p in plan},
            finishes,
            idle,
            size=size,
            physical_cpu_cores=os.cpu_count(),
        )
        check_speedup(result, workers)
    finally:
        ray.shutdown()


def branch_work(size):
    value = 0
    for i in range(size):
        value = (value + i * 1664525) & 0xFFFFFFFF
    return value


@pytest.mark.slow
def test_eight_independent_calls_benchmark():
    """Eight explicitly selected pure calls, with real Ray concurrency."""
    import os, ray
    from tests.benchmark_support import calibrate, report, check_speedup

    workers = min(4, os.cpu_count() or 1)
    from backend.node_ranker import benchmark_cpu

    measured = benchmark_cpu(cores=1)
    nodes = [ComputeNode(f"cpu-{i}", measured.node.score) for i in range(workers)]
    try:
        ray.init(num_cpus=workers, include_dashboard=False)
        source = (
            "def branches(size):\n    result = []\n"
            + "".join(
                f"    if size > {i}:\n        result.append(branch_work(size))\n"
                for i in range(8)
            )
            + "    return result\n"
        )
        assert analyze(source, {"branch_work"}).splittable
        run(source, 10, nodes, {"branch_work": branch_work})
        size, reference, baseline = calibrate(
            lambda n: [branch_work(n) for _ in range(8)], 100_000
        )
        actual, elapsed, shares, finishes, idle = run(
            source, size, nodes, {"branch_work": branch_work}, return_metrics=True
        )
        assert actual == reference
        result = report(
            "cpu_eight_calls",
            baseline,
            elapsed,
            shares,
            finishes,
            idle,
            iterations_per_call=size,
            calls=8,
            workers=workers,
        )
        check_speedup(result, workers)
    finally:
        ray.shutdown()


def test_current_index_write_set_map():
    source = """def f(values):
    result = [0] * len(values)
    for i in range(len(values)):
        result[i] = values[i] * 3 + i
    return result
"""
    assert analyze(source).splittable
    assert run(source, [2, 4, 6], [ComputeNode("one", 1)]) == [6, 13, 20]
    report = analyze(source.replace("values[i] * 3", "result[i-1] * 3"))
    assert not report.splittable and "Neighbor" in " ".join(report.reasons)


def test_if_branches_and_elif_refusal():
    source = """def f(values):
    result = []
    if values > 0:
        result.append(abs(values))
    if values < 0:
        result.append(abs(values))
    return result
"""
    assert analyze(source).splittable
    assert not analyze(
        source.replace("    if values < 0:", "    elif values < 0:")
    ).splittable
