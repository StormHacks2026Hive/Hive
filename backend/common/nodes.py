"""Node measurements and transport-independent work descriptions."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ComputeNode:
    """Score is work units/second; bandwidth is bytes/second, latency seconds."""

    node_id: str
    score: float
    online: bool = True
    latency: float = 0.0
    bandwidth: float = float("inf")
    bytes_per_work: float = 0.0
    device: Any = field(default=None, compare=False, hash=False, repr=False)
    execute: Callable[..., Any] | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    ray_node_id: str | None = None


@dataclass(frozen=True)
class WorkRange:
    """Half-open ordered range assigned to a measured node."""

    node: ComputeNode
    start: int
    stop: int
