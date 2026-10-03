"""One-off script: turn the quantized Magenta multi-style .pb into a float ONNX model.

The original graph (TensorFlow Android demo, 26 styles, conditional instance
norm) stores its conv weights as uint8 with a min/max range. This script
dequantizes those weights, rebuilds the network in PyTorch, exports it to ONNX
and compares the result against the original TensorFlow graph.

Needs: tensorflow, torch, onnx (not needed by the styletransfer-lite package).

    python tools/convert_pb_to_onnx.py legacy/stylize_quantized.pb src/styletransfer_lite/models/multistyle26.onnx
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.python.framework import tensor_util
import tensorflow as tf

EPS = 1e-5
NUM_STYLES = 26


class CondIN(nn.Module):
    """Instance norm whose gamma/beta are a weighted mix over the 26 styles."""

    def __init__(self, gamma, beta):
        super().__init__()
        self.register_buffer("gamma", torch.from_numpy(gamma))  # (26, C)
        self.register_buffer("beta", torch.from_numpy(beta))

    def forward(self, x, w):
        g = (w[:, None] * self.gamma).sum(0).view(1, -1, 1, 1)
        b = (w[:, None] * self.beta).sum(0).view(1, -1, 1, 1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)  # two-pass: stays >= 0
        return (x - mean) * torch.rsqrt(var + EPS) * g + b


class ConvIN(nn.Module):
    def __init__(self, weights, gamma, beta, stride=1, up=False, act="relu"):
        super().__init__()
        # TF stores (kh, kw, in, out); torch wants (out, in, kh, kw)
        self.register_buffer("w", torch.from_numpy(weights.transpose(3, 2, 0, 1).copy()))
        self.norm = CondIN(gamma, beta)
        self.stride, self.up, self.act = stride, up, act
        self.pad = weights.shape[0] // 2

    def forward(self, x, w):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.pad(x, (self.pad,) * 4, mode="reflect")
        x = F.conv2d(x, self.w, stride=self.stride)
        x = self.norm(x, w)
        if self.act == "relu":
            x = F.relu(x)
        elif self.act == "sigmoid":
            x = torch.sigmoid(x)
        return x


class MultiStyleNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.contract = nn.ModuleList([layers[f"contract/conv{i}"] for i in (1, 2, 3)])
        self.res = nn.ModuleList(
            [nn.ModuleList([layers[f"residual/residual{i}/conv1"], layers[f"residual/residual{i}/conv2"]])
             for i in range(1, 7)]
        )
        self.expand = nn.ModuleList([layers[f"expand/conv{i}/conv"] for i in (1, 2, 3)])

    def forward(self, image, style_weights):
        # image: (1, 3, H, W) in [0, 1]; style_weights: (26,)
        x = image
        for m in self.contract:
            x = m(x, style_weights)
        for a, b in self.res:
            x = x + b(a(x, style_weights), style_weights)
        for m in self.expand:
            x = m(x, style_weights)
        return x


def load_layers(pb_path):
    gd = tf.compat.v1.GraphDef()
    with open(pb_path, "rb") as f:
        gd.ParseFromString(f.read())
    const = {n.name: tensor_util.MakeNdarray(n.attr["value"].tensor) for n in gd.node if n.op == "Const"}

    def conv(prefix, stride=1, up=False, act="relu"):
        q = const[f"transformer/{prefix}/weights_quint8_const"].astype(np.float32)
        lo = float(const[f"transformer/{prefix}/weights_min"])
        hi = float(const[f"transformer/{prefix}/weights_max"])
        w = lo + q * (hi - lo) / 255.0
        gamma = const[f"transformer/{prefix}/InstanceNorm/gamma"].astype(np.float32)
        beta = const[f"transformer/{prefix}/InstanceNorm/beta"].astype(np.float32)
        return ConvIN(w, gamma, beta, stride=stride, up=up, act=act)

    layers = {
        "contract/conv1": conv("contract/conv1"),
        "contract/conv2": conv("contract/conv2", stride=2),
        "contract/conv3": conv("contract/conv3", stride=2),
        "expand/conv1/conv": conv("expand/conv1/conv", up=True),
        "expand/conv2/conv": conv("expand/conv2/conv", up=True),
        "expand/conv3/conv": conv("expand/conv3/conv", act="sigmoid"),
    }
    for i in range(1, 7):
        layers[f"residual/residual{i}/conv1"] = conv(f"residual/residual{i}/conv1")
        layers[f"residual/residual{i}/conv2"] = conv(f"residual/residual{i}/conv2", act=None)
    return MultiStyleNet(layers), gd


def run_tf(gd, img_nhwc, style):
    g = tf.Graph()
    with g.as_default():
        tf.import_graph_def(gd, name="")
    with tf.compat.v1.Session(graph=g) as s:
        return s.run("transformer/expand/conv3/conv/Sigmoid:0", {"input:0": img_nhwc, "style_num:0": style})


def main(pb, out):
    model, gd = load_layers(pb)
    model.eval()
    rng = np.random.RandomState(0)
    # smooth-ish random image so statistics resemble a photo
    img = np.clip(rng.rand(1, 128, 160, 3).astype(np.float32) * 0.4 + 0.3, 0, 1)
    for idx in (0, 7, 19):
        style = np.zeros(NUM_STYLES, np.float32)
        style[idx] = 1.0
        ref = run_tf(gd, img, style)
        with torch.no_grad():
            mine = model(torch.from_numpy(img.transpose(0, 3, 1, 2)), torch.from_numpy(style))
        mine = mine.numpy().transpose(0, 2, 3, 1)
        print(f"style {idx}: mean abs diff vs quantized TF graph = {np.abs(ref - mine).mean():.4f}")

    dummy = (torch.zeros(1, 3, 64, 64), torch.zeros(NUM_STYLES))
    torch.onnx.export(
        model, dummy, out,
        input_names=["image", "style_weights"], output_names=["stylized"],
        dynamic_axes={"image": {2: "h", 3: "w"}, "stylized": {2: "h", 3: "w"}},
        opset_version=17, dynamo=False,
    )
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
