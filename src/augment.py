"""Batched, on-device data augmentation (Section 2.6 of the report).

Applied to training batches only: small rotation, translation, mild zoom and
brightness/contrast jitter. No horizontal flip, since mirrored gestures
(e.g. 'l', 'thumb') are not guaranteed to keep their meaning.
"""
import math

import torch
import torch.nn.functional as F


def augment_batch(x: torch.Tensor, max_rot: float = 10.0, max_shift: float = 0.08,
                  zoom: tuple[float, float] = (0.9, 1.1), brightness: float = 0.15,
                  contrast: float = 0.15) -> torch.Tensor:
    """x: (B, C, H, W) float in [0, 1]. Each sample gets its own random transform."""
    b = x.shape[0]
    dev = x.device
    angle = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot)
    scale = torch.empty(b, device=dev).uniform_(*zoom)
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2  # grid coords span [-1, 1]
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2

    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos, -sin, tx], dim=1),
        torch.stack([sin, cos, ty], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    c = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * contrast
    br = (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean + br
    return x.clamp_(0.0, 1.0)
