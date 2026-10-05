"""Model 3 — Transfer-learning CNN (Yukti Bhatia).

ImageNet-pretrained MobileNetV2 or ResNet18 with a new classifier head.
The 1-channel normalised grayscale input used by every model is converted
inside ``forward`` to the 3-channel ImageNet-normalised input the backbone
expects, so the data pipeline is shared with the other models.

Training uses two phases (see ``train.py``):
  1. ``freeze_backbone()`` – train only the new head for a few epochs.
  2. ``unfreeze()``        – fine-tune the whole network with a lower LR.
Setting ``--freeze-epochs`` >= ``--epochs`` gives the pure
feature-extraction ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from ..data import MEAN, STD

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TransferNet(nn.Module):
    def __init__(self, backbone: str = "mobilenet_v2", num_classes: int = 10, pretrained: bool = True,
                 dropout: float = 0.2, **_):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.mobilenet_v2(weights=weights)
            in_f = net.classifier[-1].in_features
            net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
            self.head = net.classifier
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
            in_f = net.fc.in_features
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
            self.head = net.fc
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'")
        self.net = net
        self._frozen = False
        self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def freeze_backbone(self) -> None:
        self._frozen = True
        for p in self.net.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze(self) -> None:
        self._frozen = False
        for p in self.net.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        # While the backbone is frozen keep its BatchNorm statistics fixed
        # (true feature extraction); only the head trains.
        super().train(mode)
        if mode and self._frozen:
            for m in self.net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def head_parameters(self):
        return self.head.parameters()

    def backbone_parameters(self):
        head_ids = {id(p) for p in self.head.parameters()}
        return (p for p in self.net.parameters() if id(p) not in head_ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * STD + MEAN  # back to [0,1]
        x = x.expand(-1, 3, -1, -1)
        x = (x - self.in_mean) / self.in_std
        return self.net(x)
