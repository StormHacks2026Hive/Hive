"""Static graph estimates and dependency-safe ONNX pipeline correctness."""

import numpy as np
from onnx import helper, TensorProto
from backend.onnx_analyzer import analyze, estimate_flops, pipeline_cuts, run_pipeline
from backend.common.nodes import ComputeNode


def chain():
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["X", "W"], ["A"]),
                helper.make_node("Relu", ["A"], ["B"]),
                helper.make_node("Add", ["B", "X"], ["Y"]),
            ],
            "skip",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 2])],
            [helper.make_tensor("W", TensorProto.FLOAT, [2, 2], [1, 2, 3, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )


def test_flops():
    assert estimate_flops("Gemm", [(3, 4), (4, 5), (5,)], (3, 5)) == 135
    assert (
        estimate_flops("Conv", [(1, 4, 8, 8), (6, 2, 3, 3)], (1, 6, 6, 6), {"group": 2})
        == 7776
    )
    assert estimate_flops("Mystery", [], ()) is None


def test_live_skip_and_weighted_cuts():
    graph = analyze(chain())
    assert graph.frontiers[1] == ["A", "X"]
    cuts = pipeline_cuts(graph, [ComputeNode("a", 3), ComputeNode("b", 1)], 0)
    assert cuts == [2]  # 40/3 vs 8 beats 32/3 vs 16
    assert all(c.flops is not None for c in graph.costs)


def test_pipeline_correctness():
    import pytest

    pytest.importorskip("onnxruntime")
    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    graph = analyze(chain())
    reference = run_pipeline(graph, {"X": x}, [])
    actual = run_pipeline(graph, {"X": x}, [ComputeNode("a", 3), ComputeNode("b", 1)])
    assert np.allclose(actual[0], reference[0], rtol=1e-5, atol=1e-6)


def test_eight_branches():
    branches = [helper.make_node("Relu", ["X"], [f"b{i}"]) for i in range(8)]
    branches.append(helper.make_node("Sum", [f"b{i}" for i in range(8)], ["Y"]))
    model = helper.make_model(
        helper.make_graph(
            branches,
            "branches",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    assert len(analyze(model).branches) == 8


def test_branch_execution():
    from backend.onnx_analyzer import run_branches, session

    branches = [helper.make_node("Relu", ["X"], [f"b{i}"]) for i in range(8)]
    branches.append(helper.make_node("Sum", [f"b{i}" for i in range(8)], ["Y"]))
    model = helper.make_model(
        helper.make_graph(
            branches,
            "branches",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 3) - 2}
    reference = session(model).run(None, feeds)
    result = run_branches(
        analyze(model), feeds, [ComputeNode("a", 3), ComputeNode("b", 1)]
    )
    assert np.allclose(reference[0], result[0])


import pytest


@pytest.mark.slow
def test_deep_stack_benchmark():
    """Calibrate actual single-session deep MLP work; verify CPU fallback honestly."""
    import time
    from backend.onnx_analyzer import session
    from tests.benchmark_support import calibrate, report, check_speedup

    width = 512
    weights = (
        np.random.default_rng(7).normal(0, 0.01, (width, width)).astype(np.float32)
    )
    from onnx import numpy_helper

    nodes = []
    previous = "X"
    for i in range(32):
        nodes.extend(
            [
                helper.make_node("MatMul", [previous, "W"], [f"a{i}"]),
                helper.make_node("Relu", [f"a{i}"], [f"b{i}"]),
            ]
        )
        previous = f"b{i}"
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "deep-mlp",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", width])],
            [
                helper.make_tensor_value_info(
                    previous, TensorProto.FLOAT, ["batch", width]
                )
            ],
            [numpy_helper.from_array(weights, "W")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    runtime = session(model)

    def serial(batch):
        values = (
            np.random.default_rng(42).normal(0, 1, (batch, width)).astype(np.float32)
        )
        return values, runtime.run(None, {"X": values})[0]

    size, (values, reference), baseline = calibrate(serial, 512)
    graph = analyze(model, {"batch": size})
    begin = time.perf_counter()
    actual = run_pipeline(graph, {"X": values}, [])[0]
    elapsed = time.perf_counter() - begin
    assert np.allclose(actual, reference, rtol=1e-5, atol=1e-6)
    result = report(
        "onnx_deep_stack_cpu_fallback",
        baseline,
        elapsed,
        {"cpu-fallback": size},
        {"cpu-fallback": elapsed},
        {"cpu-fallback": 0.0},
        layers=32,
        batch=size,
        physical_gpus=0,
    )
    check_speedup(result, 0)


def test_weighted_batch_gather_and_torch_export():
    import torch
    from backend.onnx_analyzer import export_model, run_batches

    module = torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
    )
    module.eval()
    values = torch.arange(200, dtype=torch.float32).reshape(100, 2) / 100
    model = export_model(module, values[:2])
    actual = run_batches(
        model, values.numpy(), [ComputeNode("a", 3), ComputeNode("b", 1)]
    )[0]
    with torch.no_grad():
        reference = module(values).numpy()
    assert np.allclose(actual, reference, rtol=1e-5, atol=1e-6)


def test_pipeline_stream_preserves_microbatch_order():
    from backend.onnx_analyzer import run_pipeline_stream, session

    graph = analyze(chain())
    batches = [
        {"X": np.full((4, 2), value, dtype=np.float32)} for value in [4, 1, 3, 2]
    ]
    reference = session(graph.model)
    actual = run_pipeline_stream(
        graph, batches, [ComputeNode("a", 3), ComputeNode("b", 1)]
    )
    for feeds, outputs in zip(batches, actual):
        assert np.allclose(
            outputs[0], reference.run(None, feeds)[0], rtol=1e-5, atol=1e-6
        )
