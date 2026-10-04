"""Allocation, link penalties, offline filtering and cache invalidation."""

import time
import numpy as np
import pytest
from backend.common.nodes import ComputeNode
from backend.node_ranker import allocate, ranges, NodeRanker, Measurement, benchmark_cpu
from backend.wgsl_analyzer import split_cost_map


def test_proportional_and_ranges():
    a, b = ComputeNode("a", 3), ComputeNode("b", 1)
    assert allocate(100, [a, b]) == {a: 75, b: 25}
    assert [(r.start, r.stop) for r in ranges(100, [a, b])] == [(0, 75), (75, 100)]
    assert 75 / a.score == 25 / b.score


def test_network_offline():
    a = ComputeNode("a", 100)
    slow = ComputeNode("b", 100, bandwidth=1, bytes_per_work=1000)
    offline = ComputeNode("c", 10000, online=False)
    result = allocate(1000, [a, slow, offline])
    assert result[a] > result[slow] and offline not in result
    assert allocate(1, [a, ComputeNode("late", 100, latency=5)])[a] == 1


def test_rank_cache():
    ranker = NodeRanker(ttl=100)
    calls = []

    def probe(node):
        calls.append(node.node_id)
        return Measurement(node, {"alu": node.score}, time.monotonic())

    a, b = ComputeNode("a", 3), ComputeNode("b", 1)
    assert [m.node for m in ranker.rank([b, a], probe)] == [a, b]
    ranker.rank([a, b], probe)
    assert calls == ["b", "a"]
    ranker.rank([ComputeNode("b", 1, online=False), a], probe)
    assert "b" not in ranker.cache
    ranker.rank([a], probe, refresh=True)
    assert calls[-1] == "a"


def test_cost_aware_makespan_model():
    nodes = [ComputeNode("a", 3), ComputeNode("b", 1)]
    costs = np.concatenate([np.full(60, 100.0), np.ones(40)])
    plans = split_cost_map(costs, nodes)
    weighted = max(p.estimated_cost / p.node.score for p in plans)
    equal = max(costs[:50].sum() / 3, costs[50:].sum())
    assert weighted < equal
    assert plans[0].stop != 75  # Cost balance differs from pixel balance.
    assert plans[0].start == 0 and plans[-1].stop == 100


def test_cpu_probe_raw_measurements():
    measurement = benchmark_cpu(cores=1, repeats=2)
    assert measurement.node.score > 0 and measurement.breakdown["cores"] == 1


@pytest.mark.parametrize("total", [-1, 1.5, True])
def test_invalid_total(total):
    with pytest.raises(ValueError):
        allocate(total, [ComputeNode("a", 1)])


def test_local_probe_stability_five_runs():
    import statistics

    rates = [benchmark_cpu(cores=1).node.score for _ in range(5)]
    assert statistics.pstdev(rates) / statistics.mean(rates) < 0.10


def throttled_samples(start, stop, slowdown):
    from tests.test_cpu_analyzer import sample_hit

    hits = 0
    for index in range(start, stop):
        for _ in range(slowdown):
            value = sample_hit(index)
        hits += value
    return hits


@pytest.mark.slow
def test_ranked_allocation_beats_equal_on_throttled_workers():
    """Throttle one actual Ray task by recomputing each sample three times."""
    import os, time, ray
    from tests.test_cpu_analyzer import sample_hit
    from tests.benchmark_support import calibrate, report, check_speedup

    workers = min(4, os.cpu_count() or 1)
    slowdowns = [1] * (workers - 1) + [3]
    try:
        ray.init(num_cpus=workers, include_dashboard=False)
        remote = ray.remote(throttled_samples)
        measurements = []
        for index, slowdown in enumerate(slowdowns):
            ray.get(remote.remote(0, 10, slowdown))
            begin = time.perf_counter()
            ray.get(remote.remote(0, 100_000, slowdown))
            elapsed = time.perf_counter() - begin
            measurements.append(ComputeNode(f"cpu-{index}", 100_000 / elapsed))
        size, reference, baseline = calibrate(
            lambda n: sum(sample_hit(i) for i in range(n)), 100_000
        )

        def execute(shares):
            begin = time.perf_counter()
            start = 0
            refs = []
            owners = []
            for node, count in shares.items():
                refs.append(
                    remote.remote(
                        start, start + count, slowdowns[int(node.node_id.split("-")[1])]
                    )
                )
                owners.append(node.node_id)
                start += count
            pending = list(refs)
            finishes = {}
            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                finishes[owners[refs.index(ready[0])]] = time.perf_counter() - begin
            result = sum(ray.get(refs))
            elapsed = time.perf_counter() - begin
            return (
                result,
                elapsed,
                finishes,
                {node: max(0.0, elapsed - finish) for node, finish in finishes.items()},
            )

        equal_shares = allocate(size, [ComputeNode(n.node_id, 1) for n in measurements])
        equal, equal_time, _, _ = execute(equal_shares)
        shares = allocate(size, measurements)
        ranked, elapsed, finishes, idle = execute(shares)
        assert equal == ranked == reference
        result = report(
            "ranked_throttled_workers",
            baseline,
            elapsed,
            {n.node_id: c for n, c in shares.items()},
            finishes,
            idle,
            size=size,
            equal_split_seconds=equal_time,
            scores={n.node_id: n.score for n in measurements},
            slowdowns=slowdowns,
        )
        if workers > 1:
            assert elapsed < equal_time
        check_speedup(result, workers)
    finally:
        ray.shutdown()


def test_browser_pool_reservations_use_probe_scores():
    from backend.pool.coordinator import Coordinator
    from backend.pool.models import JobRequest
    from tests.test_pool import add_worker, CAPABILITIES
    import copy

    pool = Coordinator()
    job = pool.create(JobRequest())
    fast = add_worker(pool, "fast")
    caps = copy.deepcopy(CAPABILITIES)
    caps["benchmark"]["elapsed_ms"] *= 3
    slow = add_worker(pool, "slow", caps)
    pool.rank_pending(job)
    assert sum(c.preferred_worker == fast.worker_id for c in job.chunks) == 48
    assert sum(c.preferred_worker == slow.worker_id for c in job.chunks) == 16
    pool.disconnect(fast.worker_id)
    pool.rank_pending(job)
    assert all(c.preferred_worker == slow.worker_id for c in job.chunks)
