# Browser upload examples

Connect contributors at /node/, then open the distributed-compute dashboard.

| Workload | Upload | Paste into Input array | Settings |
| --- | --- | --- | --- |
| WGSL | elementwise.wgsl | array_input.json | chunk size 1024, input enabled |
| Marked Python | elementwise.py | array_input.json | chunk size 1024 |
| ONNX | dense.onnx | input.json | shape 65, 2; batch size 16 |

Click Analyze source/model, then Run on team GPUs. WGSL/Python output begins
[1,3,5,7,9] and has 16384 elements. ONNX output has shape [65,3]; its first row
is [0,0,0], and the row for input [2,3] is [14,19,24]. Check Contributions for
accepted work from each computer. Small jobs may complete using one contributor;
these examples establish correctness, not GPU speedup. The Marked Python option
runs converted WGSL in browsers, not the new Ray CPU analyzer.
