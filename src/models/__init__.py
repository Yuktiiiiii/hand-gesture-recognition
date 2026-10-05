"""Model registry. Every model takes a (B, 1, S, S) normalised grayscale tensor."""
from __future__ import annotations

import torch.nn as nn

from .custom_cnn import GestureCNN
from .mlp import HOGMLP, PixelMLP
from .transfer import TransferNet

MODEL_NAMES = ["mlp", "hog_mlp", "cnn", "mobilenet_v2", "resnet18"]

# Which team member owns which model family (used in reports / README)
MODEL_FAMILY = {
    "mlp": "Model 1 · MLP baseline (pixels)",
    "hog_mlp": "Model 1 · MLP on HOG features (ablation)",
    "cnn": "Model 2 · Custom CNN (scratch)",
    "mobilenet_v2": "Model 3 · Transfer learning (MobileNetV2)",
    "resnet18": "Model 3 · Transfer learning (ResNet18)",
}


def build_model(name: str, num_classes: int, img_size: int, pretrained: bool = True) -> nn.Module:
    if name == "mlp":
        return PixelMLP(num_classes, img_size)
    if name == "hog_mlp":
        return HOGMLP(num_classes, img_size)
    if name == "cnn":
        return GestureCNN(num_classes)
    if name in ("mobilenet_v2", "resnet18"):
        return TransferNet(name, num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")


def is_transfer(name: str) -> bool:
    return name in ("mobilenet_v2", "resnet18")
