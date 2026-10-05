"""Create a tiny fake dataset with the exact LeapGestRecog folder layout
(<root>/<subject>/<NN_gesture>/frame_*.png, 640x240 grayscale) so the whole
pipeline can be smoke-tested without downloading 2 GB. Each class is a
different bright "hand-like" blob shape on a dark background.

    python tests/make_synthetic_dataset.py --out data/synthetic --per-class 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

GESTURES = ["01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
            "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]


def draw(cls: int, rng: np.random.Generator) -> Image.Image:
    im = Image.new("L", (640, 240), 0)
    d = ImageDraw.Draw(im)
    cx, cy = 320 + rng.integers(-60, 60), 120 + rng.integers(-25, 25)
    s = rng.uniform(0.8, 1.2)
    b = int(rng.integers(150, 255))
    d.ellipse([cx - 40 * s, cy - 35 * s, cx + 40 * s, cy + 35 * s], fill=b)  # "palm"
    n_fingers = cls % 5 + 1
    for k in range(n_fingers):
        ang = np.deg2rad(-90 + (k - n_fingers / 2) * (20 + 4 * (cls // 5)) + 30 * (cls >= 5))
        x2, y2 = cx + np.cos(ang) * 90 * s, cy + np.sin(ang) * 90 * s
        d.line([cx, cy, x2, y2], fill=b, width=int(14 * s))
    arr = np.asarray(im, dtype=np.float32) + rng.normal(0, 8, (240, 640))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out) / "leapGestRecog"
    for s in range(a.subjects):
        for c, g in enumerate(GESTURES):
            d = root / f"{s:02d}" / g
            d.mkdir(parents=True, exist_ok=True)
            for i in range(a.per_class):
                draw(c, rng).save(d / f"frame_{s:02d}_{c+1:02d}_{i+1:04d}.png")
    print(f"wrote synthetic dataset to {root}")


if __name__ == "__main__":
    main()
