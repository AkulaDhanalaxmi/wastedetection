"""Torch-only model helpers (imported lazily so inference with ONNX needs no torch)."""
from __future__ import annotations


def build_model(arch: str, num_classes: int, pretrained: bool):
    import torch.nn as nn
    import torchvision.models as tvm
    if arch == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    elif arch == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def load_checkpoint(path, device="cpu"):
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck["meta"]
    model = build_model(meta["arch"], len(meta["class_names"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), meta
