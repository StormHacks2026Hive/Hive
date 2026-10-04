"""Dispatch ranked nodes and split plans, validate coverage and merge outputs.

The legacy push scheduler remains exported for existing API clients. New compute
execution uses one lane per measured node and ordered range/row-band assembly.
Remote full-domain executors are explicit callbacks; the browser pool adapter
supports the existing bounded 1D ABI rather than assuming a new wire contract.
"""

from typing import Any, Callable
from .common.legacy_scheduler import Scheduler as Scheduler, Node as Node


class ComputeScheduler:
    """Dispatch ranked range/tile plans through local or explicit node transports."""

    def dispatch(self, plans: list[Any], execute: Callable[[Any], Any]) -> Any:
        """Run one lane per node and retain ordered outputs plus measured timings."""
        from .common.execution import dispatch

        return dispatch(plans, execute)

    def render_wgsl(
        self,
        source: str,
        domain: tuple[int, int, int],
        buffers: dict[int, bytes],
        output_binding: int,
        bytes_per_element: int,
        plans: list[Any],
    ) -> tuple[bytes, Any]:
        """Dispatch full-domain shaders and copy only each owned row/cell region."""
        from .wgsl_analyzer import dispatch_local

        width, height, depth = domain
        if depth != 1:
            raise ValueError("Stitching currently supports 1D/2D domains")
        total = width * height * bytes_per_element
        is_1d = height == 1
        expected = width if is_1d else height
        ordered = sorted(plans, key=lambda p: p.start)
        end = 0
        for plan in ordered:
            if plan.start != end or plan.stop <= plan.start:
                raise ValueError("Tiles must cover the domain without overlap or gaps")
            end = plan.stop
        if end != expected:
            raise ValueError("Incomplete tile coverage")

        def execute(plan: Any) -> bytes:
            offset = (plan.start, 0, 0) if is_1d else (0, plan.start, 0)
            extent = (
                (plan.stop - plan.start, 1, 1)
                if is_1d
                else (width, plan.stop - plan.start, 1)
            )
            if plan.node.execute:
                raw = plan.node.execute(
                    source, domain, buffers, output_binding, total, offset, extent
                )
            elif plan.node.device is not None:
                raw = dispatch_local(
                    source,
                    domain,
                    buffers,
                    output_binding,
                    total,
                    plan.node.device,
                    offset,
                    extent,
                )
            else:
                raise ValueError(
                    "Node needs a local wgpu device or remote transport executor"
                )
            if len(raw) != total:
                raise ValueError("Transport returned wrong full-domain buffer size")
            stride = bytes_per_element if is_1d else width * bytes_per_element
            return raw[plan.start * stride : plan.stop * stride]

        result = self.dispatch(ordered, execute)
        return b"".join(result.outputs), result
