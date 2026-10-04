# Browser upload examples

Sign in, create/join a hive and keep contributor tabs visible. Open Compute.

| Mode | Upload | Input values | Settings |
| --- | --- | --- | --- |
| Compute GPU | elementwise.wgsl | array_input.json | f32 input enabled |
| Compute Python GPU | elementwise.py | array_input.json | Analyze or Mark & analyze |
| Compute CPU | cpu_sum.py | [1,2,3,4] | CPU markers produce the scalar 40 |
| ONNX | dense.onnx | input.json | shape 65,2; batch size 16 |
| Animation | Mandelbulb_wgsl.py or Mandelbulb.wgsl | none | choose width, height, frames and camera turn |

Click Analyze and Send. Edit both CPU/GPU begin/end comments and analyze again
to change the target. Independent regions each use the supplied input; arbitrary
surrounding Python is not executed. The animation mode handles validated shader
config and generated frame values instead of executing the renderer's Python
rotation loop.

WGSL/Python array output begins [1,3,5,7,9] and has 16384 elements. ONNX output
has shape [65,3]; its first row is [0,0,0], and input [2,3] produces [14,19,24].
Check Nodes and Contributions to see actual accepted work. Small jobs may finish
on one contributor. These examples check correctness, not speedup. Browser CPU
work uses the bounded numeric interpreter; the separate local CPU API uses Ray.
