"""Create a small independent-batch model and input JSON for the Hive UI.

Usage: python -m demos.make_onnx_example /tmp/hive-onnx-example
"""
import argparse
import json
from pathlib import Path
from onnx import helper, TensorProto


def main():
    parser = argparse.ArgumentParser(description='Create a 2-input-feature dense/ReLU ONNX example')
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph([
        helper.make_node('MatMul', ['values', 'weights'], ['hidden']),
        helper.make_node('Relu', ['hidden'], ['result']),
    ], 'hive-independent-dense',
        [helper.make_tensor_value_info('values', TensorProto.FLOAT, ['batch', 2])],
        [helper.make_tensor_value_info('result', TensorProto.FLOAT, ['batch', 3])],
        [helper.make_tensor('weights', TensorProto.FLOAT, [2, 3], [1, 2, 3, 4, 5, 6])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    (args.directory / 'dense.onnx').write_bytes(model.SerializeToString())
    values = [float(i % 9 - 4) for i in range(130)]
    (args.directory / 'input.json').write_text(json.dumps(values))
    print(f'Model: {args.directory / "dense.onnx"}')
    print(f'Input: {args.directory / "input.json"}')
    print('Upload dense.onnx; paste input.json; shape 65, 2; batch size 32. Output shape is 65, 3.')


if __name__ == '__main__':
    main()
