"""Analyze inferred ONNX DAGs exported from PyTorch inference models.

Parse tensors, initializers and operators in topological order; estimate Gemm,
MatMul and Conv multiply/add FLOPs and tensor bytes. Unknown costs/shapes are
reported, never silently treated as measured costs. Pipeline cuts carry every
live tensor (including skip edges), minimizing maximum score-normalized stage
work plus activation transfer cost. Branches are disjoint ancestor sets before
joins. Batch splits require a proven sample-preserving operator subset and use
node_ranker allocation. Execution uses ONNX Runtime sessions per device, ordered
batch gathering, or extracted stage graphs; stage dependencies remain sequential
for one microbatch. General control flow, training and unknown batch semantics
are refused. Estimates do not promise runtime speedups.
"""

from __future__ import annotations
import io
import logging
import math
from dataclasses import dataclass
from typing import Any
import onnx
from onnx import helper, numpy_helper
from .common.nodes import ComputeNode

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OperatorCost:
    """Static operation and memory estimate; None indicates unknown FLOPs."""

    index: int
    op: str
    flops: int | None
    memory_bytes: int


@dataclass
class GraphAnalysis:
    """Inferred graph, ordered operator estimates and cut frontiers."""

    model: Any
    costs: list[OperatorCost]
    tensor_bytes: dict[str, int]
    frontiers: list[list[str]]
    branches: list[list[int]]
    findings: list[str]


def estimate_flops(
    op: str,
    inputs: list[tuple[int, ...]],
    output: tuple[int, ...],
    attributes: dict[str, Any] | None = None,
) -> int | None:
    """Count multiply/add as two FLOPs; include Gemm/Conv bias addition."""
    a = attributes or {}
    if op in {"Gemm", "MatMul"} and len(inputs) >= 2:
        k = inputs[0][-2] if a.get("transA", 0) else inputs[0][-1]
        return 2 * math.prod(output) * k + (
            math.prod(output) if op == "Gemm" and len(inputs) > 2 else 0
        )
    if op == "Conv" and len(inputs) >= 2:
        return math.prod(output) * (2 * math.prod(inputs[1][1:]) + (len(inputs) > 2))
    if op in {"Add", "Sub", "Mul", "Div", "Relu"}:
        return math.prod(output)
    if op in {"Identity", "Reshape", "Flatten", "Concat", "Transpose"}:
        return 0
    return None


def analyze(
    model: Any, dynamic_dimensions: dict[str, int] | None = None
) -> GraphAnalysis:
    """Infer shapes and construct dependency-safe live-tensor cut frontiers."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    if any(
        t.external_data or t.data_location == onnx.TensorProto.EXTERNAL
        for t in model.graph.initializer
    ):
        raise ValueError("External weights must be embedded before analysis")
    for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        for d in v.type.tensor_type.shape.dim:
            if d.dim_param in (dynamic_dimensions or {}):
                d.dim_value = dynamic_dimensions[d.dim_param]
    model = onnx.shape_inference.infer_shapes(model)
    shapes: dict[str, tuple[int, ...]] = {}
    sizes: dict[str, int] = {}
    findings: list[str] = []
    for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        dims = v.type.tensor_type.shape.dim
        if all(d.HasField("dim_value") and d.dim_value > 0 for d in dims):
            shapes[v.name] = tuple(d.dim_value for d in dims)
            sizes[v.name] = (
                math.prod(shapes[v.name])
                * numpy_helper.to_array(
                    helper.make_tensor("", v.type.tensor_type.elem_type, [1], [0])
                ).dtype.itemsize
            )
    weights = {t.name for t in model.graph.initializer}
    for t in model.graph.initializer:
        shapes[t.name] = tuple(t.dims)
        sizes[t.name] = numpy_helper.to_array(t).nbytes
    available = {v.name for v in model.graph.input} | weights
    pending = list(model.graph.node)
    ordered = []
    while pending:
        ready = next(
            (n for n in pending if all(not i or i in available for i in n.input)), None
        )
        if ready is None:
            raise ValueError("Graph has unresolved inputs or a cycle")
        ordered.append(ready)
        pending.remove(ready)
        available.update(ready.output)
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    costs = []
    ancestors: dict[str, set[int]] = {
        n: set()
        for n in available
        if n in weights or n in {v.name for v in model.graph.input}
    }
    branches = []
    for index, node in enumerate(ordered):
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        flops = (
            estimate_flops(
                node.op_type,
                [shapes[i] for i in node.input if i],
                shapes.get(node.output[0], ()),
                attrs,
            )
            if all(i in shapes for i in node.input if i) and node.output[0] in shapes
            else None
        )
        if flops is None:
            findings.append(
                f"{index}: unknown cost for {node.op_type} or unresolved shape"
            )
        costs.append(
            OperatorCost(
                index,
                node.op_type,
                flops,
                sum(sizes.get(i, 0) for i in [*node.input, *node.output]),
            )
        )
        parents = [ancestors.get(i, set()) for i in node.input if i not in weights]
        if (
            len(parents) > 1
            and all(parents)
            and all(not p & q for j, p in enumerate(parents) for q in parents[j + 1 :])
        ):
            branches.extend(sorted(p) for p in parents)
        lineage = {index}.union(*parents)
        for out in node.output:
            ancestors[out] = lineage
    frontiers = []
    produced = {v.name for v in model.graph.input}
    for cut in range(len(ordered) + 1):
        needed = {i for n in ordered[cut:] for i in n.input} | {
            o.name for o in model.graph.output
        }
        frontiers.append(sorted((produced & needed) - weights))
        if cut < len(ordered):
            produced.update(ordered[cut].output)
    return GraphAnalysis(model, costs, sizes, frontiers, branches, findings)


def pipeline_cuts(
    graph: GraphAnalysis, nodes: list[ComputeNode], transfer_weight: float = 1e-9
) -> list[int]:
    """Dynamic program minimizes maximum weighted stage time and cut bytes."""
    nodes = [n for n in nodes if n.online and n.score > 0]
    if not nodes or len(nodes) > len(graph.costs):
        raise ValueError("Need 1..operator_count online nodes")
    if graph.findings or any(
        name not in graph.tensor_bytes for f in graph.frontiers for name in f
    ):
        raise ValueError(
            "Resolve all tensor shapes and operator costs before pipeline planning"
        )
    prefix = [0]
    for c in graph.costs:
        prefix.append(prefix[-1] + c.flops)
    dp = {(0, 0): (0.0, [])}
    for stage, node in enumerate(nodes, 1):
        for end in range(stage, len(graph.costs) + 1):
            choices = []
            for start in range(stage - 1, end):
                previous = dp.get((stage - 1, start))
                if previous is None:
                    continue
                transfer = (
                    sum(graph.tensor_bytes[n] for n in graph.frontiers[end])
                    * transfer_weight
                    if end < len(graph.costs)
                    else 0
                )
                duration = (prefix[end] - prefix[start]) / node.score + transfer
                choices.append((max(previous[0], duration), previous[1] + [end]))
            if choices:
                dp[stage, end] = min(choices, key=lambda x: x[0])
    return dp[len(nodes), len(graph.costs)][1][:-1]


def export_model(model: Any, example: Any) -> Any:
    """Export an eval model in memory without altering its training flag."""
    import torch

    buffer = io.BytesIO()
    training = model.training
    try:
        model.eval()
        torch.onnx.export(
            model,
            example,
            buffer,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=18,
            dynamo=False,
        )
    finally:
        model.train(training)
    return onnx.load_model_from_string(buffer.getvalue())


def session(model: Any, node: ComputeNode | None = None) -> Any:
    """Create a session on an explicit CUDA device, otherwise CPU fallback."""
    import onnxruntime as ort

    providers: list[Any] = ["CPUExecutionProvider"]
    if node is not None and isinstance(node.device, int):
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA ONNX Runtime provider unavailable")
        providers = [("CUDAExecutionProvider", {"device_id": node.device})]
    else:
        log.warning("No CUDA GPU selected; executing ONNX on CPU")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=providers
    )


def run_pipeline(
    graph: GraphAnalysis, feeds: dict[str, Any], nodes: list[ComputeNode]
) -> list[Any]:
    """Execute extracted stages, carrying live skip activations across cuts."""
    active = [n for n in nodes if n.online and n.score > 0]
    if len(active) <= 1:
        log.warning("Pipeline has fewer than two GPUs; running one full graph")
        return session(graph.model, active[0] if active else None).run(None, feeds)
    cuts = [0, *pipeline_cuts(graph, active), len(graph.costs)]
    tensors = dict(feeds)
    for node, start, end in zip(active, cuts, cuts[1:]):
        stage = onnx.utils.Extractor(graph.model).extract_model(
            graph.frontiers[start], graph.frontiers[end]
        )
        outputs = session(stage, node).run(
            None, {i.name: tensors[i.name] for i in stage.graph.input}
        )
        tensors.update(zip(graph.frontiers[end], outputs))
    return [tensors[o.name] for o in graph.model.graph.output]


def run_batches(model: Any, values: Any, nodes: list[ComputeNode]) -> list[Any]:
    """Prove batch independence using the existing validated pool contract."""
    import base64
    import numpy as np
    from .pool.workloads import OnnxAnalysisRequest, onnx_plan
    from .node_ranker import allocate
    from concurrent.futures import ThreadPoolExecutor

    onnx_plan(
        OnnxAnalysisRequest(
            model=base64.b64encode(model.SerializeToString()).decode(),
            input_shape=list(values.shape),
            batch_size=1,
        ),
        True,
    )
    active = [n for n in nodes if n.online and n.score > 0]
    if len(active) <= 1:
        log.warning("Batch execution has fewer than two GPUs; using one session")
        runtime = session(model, active[0] if active else None)
        return runtime.run(None, {runtime.get_inputs()[0].name: values})
    shares = allocate(len(values), active)
    start = 0
    tasks = []

    def run(node: ComputeNode, batch: Any) -> list[Any]:
        runtime = session(model, node)
        return runtime.run(None, {runtime.get_inputs()[0].name: batch})

    with ThreadPoolExecutor(max_workers=len(active)) as executor:
        for node, count in shares.items():
            if count:
                tasks.append(executor.submit(run, node, values[start : start + count]))
                start += count
        results = [t.result() for t in tasks]
    return [np.concatenate(parts, axis=0) for parts in zip(*results)]


def run_branches(
    graph: GraphAnalysis, feeds: dict[str, Any], nodes: list[ComputeNode]
) -> list[Any]:
    """Execute disjoint branch subgraphs concurrently, then run their join graph."""
    from .common.nodes import WorkRange
    from .common.execution import dispatch

    active = [n for n in nodes if n.online and n.score > 0]
    if len(active) < 2 or len(graph.branches) < 2:
        log.warning("No independent multi-node branch plan; running full graph")
        return session(graph.model, active[0] if active else None).run(None, feeds)
    if any(
        set(a) & set(b)
        for i, a in enumerate(graph.branches)
        for b in graph.branches[i + 1 :]
    ):
        raise ValueError("Nested branch groups need an explicit execution DAG")
    inputs = [v.name for v in graph.model.graph.input]
    output_names = {v.name for v in graph.model.graph.output}
    stages = []
    for branch in graph.branches:
        members = set(branch)
        outside_inputs = {
            i
            for j, n in enumerate(graph.model.graph.node)
            if j not in members
            for i in n.input
        }
        boundary = [
            o
            for j in branch
            for o in graph.model.graph.node[j].output
            if o in outside_inputs | output_names
        ]
        extracted = onnx.utils.Extractor(graph.model).extract_model(inputs, boundary)
        cost = sum(graph.costs[j].flops or 1 for j in branch)
        stages.append((extracted, boundary, cost))
    loads = {n: 0.0 for n in active}
    plans = []
    for index, (_, _, cost) in sorted(
        enumerate(stages), key=lambda item: item[1][2], reverse=True
    ):
        node = min(active, key=lambda n: loads[n] + cost / n.score)
        loads[node] += cost / node.score
        plans.append(WorkRange(node, index, index + 1))

    def execute(plan: Any) -> dict[str, Any]:
        model, names, _ = stages[plan.start]
        values = session(model, plan.node).run(
            None, {v.name: feeds[v.name] for v in model.graph.input}
        )
        return dict(zip(names, values))

    result = dispatch(plans, execute)
    merged = dict(feeds)
    boundary = []
    for output in result.outputs:
        merged.update(output)
        boundary.extend(output)
    downstream = onnx.utils.Extractor(graph.model).extract_model(
        list(dict.fromkeys(inputs + boundary)),
        [v.name for v in graph.model.graph.output],
    )
    runtime = session(downstream, max(active, key=lambda n: n.score))
    return runtime.run(None, {v.name: merged[v.name] for v in downstream.graph.input})


def run_pipeline_stream(
    graph: GraphAnalysis,
    microbatches: list[dict[str, Any]],
    nodes: list[ComputeNode],
) -> list[list[Any]]:
    """Overlap independent microbatches across dependency-ordered GPU stages.

    Each stage owns one session and one execution lane. Microbatches must match
    the graph's declared shapes; dynamic sample-axis proof is a separate batch
    sharding decision. Gather preserves original microbatch and output order.
    """
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack

    active = [node for node in nodes if node.online and node.score > 0]
    if len(active) < 2:
        log.warning("Fewer than two GPUs; pipeline stream uses one full session")
        runtime = session(graph.model, active[0] if active else None)
        return [runtime.run(None, feeds) for feeds in microbatches]
    cuts = [0, *pipeline_cuts(graph, active), len(graph.costs)]
    stages = []
    for node, start, end in zip(active, cuts, cuts[1:]):
        model = onnx.utils.Extractor(graph.model).extract_model(
            graph.frontiers[start], graph.frontiers[end]
        )
        stages.append((session(model, node), graph.frontiers[end]))

    def step(previous: Any, original: dict[str, Any], stage: int) -> dict[str, Any]:
        tensors = dict(previous.result()) if previous is not None else dict(original)
        runtime, outputs = stages[stage]
        values = runtime.run(
            None, {v.name: tensors[v.name] for v in runtime.get_inputs()}
        )
        tensors.update(zip(outputs, values))
        return tensors

    with ExitStack() as stack:
        lanes = [stack.enter_context(ThreadPoolExecutor(max_workers=1)) for _ in stages]
        completed = []
        for feeds in microbatches:
            previous = None
            for index, lane in enumerate(lanes):
                previous = lane.submit(step, previous, feeds, index)
            completed.append(previous)
        gathered = [future.result() for future in completed]
    return [[values[o.name] for o in graph.model.graph.output] for values in gathered]
