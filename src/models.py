"""The three compared models: MLP baseline, custom CNN, MobileNetV2 transfer learning."""
import torch
import torch.nn as nn
from torchvision import models

from . import config as C


class MLP(nn.Module):
    """Fully connected baseline on flattened 128x128 images (or HOG vectors)."""

    def __init__(self, in_features: int = C.IMG_SIZE * C.IMG_SIZE,
                 hidden: tuple[int, ...] = (512, 256), num_classes: int = C.NUM_CLASSES,
                 dropout: float = 0.4):
        super().__init__()
        layers, d = [nn.Flatten()], in_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """CNN trained from scratch: conv + batch norm + pooling + dropout."""

    def __init__(self, num_classes: int = C.NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),    # 128 -> 64
            conv_block(16, 32),   # 64 -> 32
            conv_block(32, 64),   # 32 -> 16
            conv_block(64, 128),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MobileNetV2Transfer(nn.Module):
    """ImageNet-pretrained MobileNetV2 with a new 10-class head.

    Accepts (B, 1, H, W) grayscale in [0, 1]; replicates to 3 channels and
    applies ImageNet normalisation internally, as the backbone expects.
    """

    def __init__(self, num_classes: int = C.NUM_CLASSES, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v2(weights=weights)
        self.features = net.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(net.last_channel, num_classes),
        )
        self.register_buffer("mean", torch.tensor(C.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(C.IMAGENET_STD).view(1, 3, 1, 1))

    def freeze_backbone(self, freeze: bool = True, trainable_last_blocks: int = 0):
        for p in self.features.parameters():
            p.requires_grad = not freeze
        if freeze and trainable_last_blocks > 0:
            for block in self.features[-trainable_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        return self.classifier(self.features(x))


def build_model(name: str, **kw) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLP(**kw)
    if name == "cnn":
        return CustomCNN(**kw)
    if name == "mobilenetv2":
        return MobileNetV2Transfer(**kw)
    raise ValueError(f"Unknown model {name}")
