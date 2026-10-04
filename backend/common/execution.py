"""Concurrent region execution with measured timing and ordered validation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Callable
from .nodes import ComputeNode


@dataclass(frozen=True)
class ExecutionResult:
    """Wall-clock timings measured from shared dispatch origin."""

    outputs: list[Any]
    elapsed: float
    finish_times: dict[str, float]
    idle_times: dict[str, float]


def dispatch(plans: list[Any], execute: Callable[[Any], Any]) -> ExecutionResult:
    """Group work per node; one executor lane per node, ordered gathered results."""
    lanes: dict[ComputeNode, list[tuple[int, Any]]] = {}
    for index, plan in enumerate(plans):
        if not plan.node.online:
            raise ValueError(f"Node offline: {plan.node.node_id}; re-rank and re-plan")
        lanes.setdefault(plan.node, []).append((index, plan))
    if not lanes:
        raise ValueError("Empty split plan")
    started = time.perf_counter()

    def run_lane(items: list[tuple[int, Any]]) -> tuple[list[tuple[int, Any]], float]:
        return [
            (index, execute(plan)) for index, plan in items
        ], time.perf_counter() - started

    outputs = [None] * len(plans)
    finishes = {}
    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = {
            node: executor.submit(run_lane, items) for node, items in lanes.items()
        }
        for node, future in futures.items():
            completed, finished = future.result()
            finishes[node.node_id] = finished
            for index, value in completed:
                outputs[index] = value
    elapsed = time.perf_counter() - started
    return ExecutionResult(
        outputs,
        elapsed,
        finishes,
        {node: max(0.0, elapsed - finish) for node, finish in finishes.items()},
    )
