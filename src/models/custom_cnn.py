"""Model 2 — Custom CNN trained from scratch (Prisha Chadha).

Four VGG-style stages of (Conv3x3-BN-ReLU) x2 + MaxPool, channel widths
32-64-128-256, followed by global average pooling and a small dense head
with dropout. ~1.2 M parameters — compact enough for edge devices while
exploiting the locality / translation-equivariance inductive bias that the
MLP lacks.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int, drop: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(drop),
    )


class GestureCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, widths=(32, 64, 128, 256),
                 conv_dropout: float = 0.1, head_dropout: float = 0.4, **_):
        super().__init__()
        stages, c = [], in_channels
        for w in widths:
            stages.append(conv_block(c, w, conv_dropout))
            c = w
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(c, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout / 2),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
