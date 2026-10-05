"""Model 1 — Classical / MLP baseline (Upasana Sahukara).

* ``PixelMLP``  : fully-connected network on the flattened grayscale image.
                  No spatial inductive bias at all -> the performance floor.
* ``HOGMLP``    : same MLP head on Histogram-of-Oriented-Gradients features
                  (hand-crafted feature ablation). HOG is implemented in pure
                  PyTorch so it runs on the GPU inside ``forward`` and the
                  data pipeline stays identical for every model. HOG has no
                  learnable parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp_head(in_dim: int, num_classes: int, hidden=(512, 256), dropout: float = 0.4) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, num_classes))
    return nn.Sequential(*layers)


class PixelMLP(nn.Module):
    def __init__(self, num_classes: int = 10, img_size: int = 128, hidden=(512, 256), dropout: float = 0.4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.head = mlp_head(img_size * img_size, num_classes, hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.flatten(x))


class TorchHOG(nn.Module):
    """Dalal-Triggs HOG (unsigned gradients, L2-Hys block normalisation),
    equivalent in spirit to ``skimage.feature.hog`` but batched on GPU.

    Input  : (B, 1, H, W) float tensor.
    Output : (B, n_blocks_y * n_blocks_x * block^2 * orientations)
    """

    def __init__(self, orientations: int = 9, pixels_per_cell: int = 8, cells_per_block: int = 2, eps: float = 1e-6):
        super().__init__()
        self.o, self.c, self.b, self.eps = orientations, pixels_per_cell, cells_per_block, eps
        self.register_buffer("kx", torch.tensor([[[[-1.0, 0.0, 1.0]]]]), persistent=False)
        self.register_buffer("ky", torch.tensor([[[[-1.0], [0.0], [1.0]]]]), persistent=False)

    def out_dim(self, img_size: int) -> int:
        n_cells = img_size // self.c
        n_blocks = n_cells - self.b + 1
        return n_blocks * n_blocks * self.b * self.b * self.o

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        gx = F.conv2d(F.pad(x, (1, 1, 0, 0), mode="replicate"), self.kx)
        gy = F.conv2d(F.pad(x, (0, 0, 1, 1), mode="replicate"), self.ky)
        mag = torch.sqrt(gx * gx + gy * gy)
        ang = torch.remainder(torch.atan2(gy, gx), math.pi)  # unsigned [0, pi)
        pos = ang / (math.pi / self.o) - 0.5  # bin centres at (k+0.5)*pi/o
        lo = torch.floor(pos)
        w_hi = pos - lo
        lo_idx = torch.remainder(lo, self.o).long()
        hi_idx = torch.remainder(lo + 1, self.o).long()
        B, _, H, W = x.shape
        hist = x.new_zeros(B, self.o, H, W)
        hist.scatter_add_(1, lo_idx, mag * (1 - w_hi))
        hist.scatter_add_(1, hi_idx, mag * w_hi)
        cells = F.avg_pool2d(hist, self.c)  # (B, o, Hc, Wc)
        blocks = cells.unfold(2, self.b, 1).unfold(3, self.b, 1)  # (B, o, Hb, Wb, b, b)
        blocks = blocks.permute(0, 2, 3, 4, 5, 1).reshape(B, blocks.shape[2], blocks.shape[3], -1)
        # L2-Hys
        blocks = blocks / torch.sqrt((blocks ** 2).sum(-1, keepdim=True) + self.eps ** 2)
        blocks = blocks.clamp(max=0.2)
        blocks = blocks / torch.sqrt((blocks ** 2).sum(-1, keepdim=True) + self.eps ** 2)
        return blocks.reshape(B, -1)


class HOGMLP(nn.Module):
    def __init__(self, num_classes: int = 10, img_size: int = 128, hidden=(512, 256), dropout: float = 0.4,
                 orientations: int = 9, pixels_per_cell: int = 8, cells_per_block: int = 2):
        super().__init__()
        self.hog = TorchHOG(orientations, pixels_per_cell, cells_per_block)
        self.head = mlp_head(self.hog.out_dim(img_size), num_classes, hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hog(x))
