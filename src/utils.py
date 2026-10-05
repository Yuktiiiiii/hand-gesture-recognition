"""Shared helpers: seeding, device, complexity (params / FLOPs) and latency."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_size_mb(model: nn.Module) -> float:
    n = sum(p.numel() * p.element_size() for p in model.parameters())
    n += sum(b.numel() * b.element_size() for b in model.buffers())
    return n / 1024 ** 2


@torch.no_grad()
def count_macs(model: nn.Module, img_size: int, in_ch: int = 1) -> int:
    """Multiply-accumulate count for Conv2d and Linear layers on one image
    (FLOPs ~= 2 x MACs). Implemented with forward hooks to avoid an extra
    dependency; BN/activations/pooling are ignored (standard convention)."""
    macs = 0
    hooks = []

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal macs
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs += out.numel() * k

    def linear_hook(m: nn.Linear, inp, out):
        nonlocal macs
        macs += m.in_features * m.out_features * (out.numel() // m.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    model(torch.zeros(1, in_ch, img_size, img_size, device=dev))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return int(macs)


@torch.no_grad()
def measure_latency(model: nn.Module, img_size: int, device: str = "cpu", runs: int = 100,
                    warmup: int = 15, batch_size: int = 1, threads: int | None = 1) -> dict[str, float]:
    """Per-image inference latency. Defaults to CPU, batch 1, single thread
    — the most comparable setting for edge deployment."""
    dev = torch.device(device)
    prev_threads = torch.get_num_threads()
    if dev.type == "cpu" and threads:
        torch.set_num_threads(threads)
    model = model.to(dev).eval()
    x = torch.randn(batch_size, 1, img_size, img_size, device=dev)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000 / batch_size)
    torch.set_num_threads(prev_threads)
    t = np.array(times)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "fps": float(1000 / t.mean()), "device": str(dev), "threads": threads if dev.type == "cpu" else None}


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)
