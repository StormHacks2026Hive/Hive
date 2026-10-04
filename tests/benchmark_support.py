"""Calibration and JSON reporting; times are measured, never inferred speedups."""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable

log = logging.getLogger(__name__)
RESULTS = Path(__file__).parent / "results"


def calibrate(run: Callable[[int], Any], initial: int = 1) -> tuple[int, Any, float]:
    """Scale until an actual single serial run is at least 30 seconds."""
    size = max(initial, int(os.getenv("BENCH_SCALE", initial)))
    for _ in range(20):
        begin = time.perf_counter()
        result = run(size)
        elapsed = time.perf_counter() - begin
        log.info("Calibration size=%d elapsed=%.6fs", size, elapsed)
        if elapsed >= 30:
            log.info("Final calibrated size=%d baseline=%.6fs", size, elapsed)
            return size, result, elapsed
        size = max(size + 1, int(size * min(16, 33 / max(elapsed, 0.001))))
    raise RuntimeError("Could not calibrate to a 30-second serial run")


def report(
    name: str,
    baseline: float,
    parallel: float,
    shares: dict[str, int],
    finishes: dict[str, float],
    idle: dict[str, float],
    **metadata: Any,
) -> dict[str, Any]:
    """Print requested benchmark table and persist measured data."""
    if baseline < 30:
        raise ValueError("Benchmark baseline must be >=30s")
    result = {
        "baseline_seconds": baseline,
        "parallel_seconds": parallel,
        "speedup": baseline / parallel,
        "shares": shares,
        "finish_seconds": finishes,
        "idle_seconds": idle,
        **metadata,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(
        f"\n{name}\nBaseline (s) | Parallel (s) | Speedup | Node | Share | Finish (s) | Idle (s)"
    )
    for node, share in shares.items():
        print(
            f"{baseline:.3f} | {parallel:.3f} | {baseline / parallel:.3f}x | {node} | {share} | {finishes[node]:.3f} | {idle[node]:.3f}"
        )
    return result


def check_speedup(result: dict[str, Any], workers: int) -> None:
    """Assert configured threshold only with two physically independent workers."""
    if workers < 2:
        print(
            "Speedup assertion skipped: fewer than two independent hardware workers; correctness was checked."
        )
    else:
        assert result["speedup"] > float(os.getenv("BENCH_SPEEDUP", "1.3"))
