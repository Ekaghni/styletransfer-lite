# Benchmarks

`latency.py` times stylization at several sizes. Run it yourself:

```
python benchmarks/latency.py --device cpu
python benchmarks/latency.py --device cuda
```

`results_cpu.md` and `results_cuda.md` are the numbers quoted in the main README, from a Windows 11 laptop
(AMD CPU, RTX 4060 Laptop GPU, Python 3.10, onnxruntime 1.23.2). The CUDA run used `onnxruntime-gpu` with
the CUDA 12 and cuDNN 9 libraries from the `nvidia-*-cu12` pip packages on PATH.

There is no quality benchmark: style transfer has no ground truth. Conversion fidelity against the original
TensorFlow graph is reported by `tools/convert_pb_to_onnx.py`.
