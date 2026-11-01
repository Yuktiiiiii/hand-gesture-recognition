"""Dataset acquisition and preprocessing (Section 2 of the report).

Pipeline:
 1. Locate / download LeapGestRecog
 2. Image integrity and duplicate checking (MD5 hash)
 3. Metadata table: path, subject ID, gesture label
 4. Subject-wise train / val / test split
 5. Resize to 128x128
 6. Encode labels 0..9
 7. Normalise to [0, 1] (done on the fly in to_tensor / model-specific prep)
 8. Augmentation on training images only (see augment.py)
"""
import argparse
import hashlib
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from . import config as C

FRAME_RE = re.compile(r"frame_(\d{2})_(\d{2})_(\d{4})\.png$")


def find_dataset_root(path: str | None = None) -> Path:
    """Return the folder holding the subject sub-folders 00..09."""
    if path is None:
        path = os.environ.get("LEAPGEST_DIR")
    if path is None:
        import kagglehub
        path = kagglehub.dataset_download("gti-upm/leapgestrecog")
    path = Path(path)
    for cand in [path, path / "leapGestRecog", path / "leapGestRecog" / "leapGestRecog"]:
        if (cand / "00").is_dir():
            return cand
    raise FileNotFoundError(f"Could not find LeapGestRecog subject folders under {path}")


def build_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan the dataset, check integrity and duplicates, and return (metadata, removed)."""
    rows, removed = [], []
    seen_hashes: dict[str, str] = {}
    files = sorted(root.rglob("*.png"))
    for f in tqdm(files, desc="Scanning images"):
        m = FRAME_RE.search(f.name)
        if m is None:
            removed.append({"path": str(f), "reason": "unrecognised filename"})
            continue
        subject, gesture_idx = int(m.group(1)), int(m.group(2))
        gesture_folder = f.parent.name
        if gesture_folder not in C.GESTURES or C.GESTURES.index(gesture_folder) != gesture_idx - 1:
            removed.append({"path": str(f), "reason": "label/folder mismatch"})
            continue
        data = f.read_bytes()
        digest = hashlib.md5(data).hexdigest()
        if digest in seen_hashes:
            removed.append({"path": str(f), "reason": f"duplicate of {seen_hashes[digest]}"})
            continue
        try:
            with Image.open(f) as im:
                im.verify()
        except Exception as e:  # corrupted / unreadable
            removed.append({"path": str(f), "reason": f"unreadable: {e}"})
            continue
        seen_hashes[digest] = str(f)
        rows.append({
            "path": str(f),
            "subject": subject,
            "gesture": gesture_folder,
            "label": gesture_idx - 1,
            "md5": digest,
        })
    meta = pd.DataFrame(rows).sort_values(["subject", "label", "path"]).reset_index(drop=True)
    meta["split"] = np.select(
        [meta.subject.isin(C.TRAIN_SUBJECTS), meta.subject.isin(C.VAL_SUBJECTS),
         meta.subject.isin(C.TEST_SUBJECTS)],
        ["train", "val", "test"], default="unused")
    return meta, pd.DataFrame(removed, columns=["path", "reason"])


def load_and_resize(path: str, size: int = C.IMG_SIZE) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def prepare(dataset_dir: str | None = None, force: bool = False) -> None:
    """Run the full preprocessing pipeline once and cache the result as uint8 arrays."""
    if C.CACHE_FILE.exists() and not force:
        print(f"Cache already exists: {C.CACHE_FILE}")
        return
    root = find_dataset_root(dataset_dir)
    print(f"Dataset root: {root}")
    meta, removed = build_metadata(root)
    meta.to_csv(C.METADATA_FILE, index=False)
    removed.to_csv(C.REMOVED_FILE, index=False)
    print(f"Kept {len(meta)} images, removed {len(removed)} "
          f"({removed.reason.str.split(' ').str[0].value_counts().to_dict()})")

    missing = set(range(C.NUM_CLASSES)) - set(meta.label)
    assert not missing, f"Missing gesture classes: {missing}"

    counts = meta.pivot_table(index="subject", columns="gesture", values="path", aggfunc="count")
    counts.to_csv(C.RESULTS_DIR / "class_subject_counts.csv")
    print("Images per subject x gesture:\n", counts)

    images = np.stack([load_and_resize(p) for p in tqdm(meta.path, desc="Resizing to 128x128")])
    np.savez_compressed(
        C.CACHE_FILE,
        images=images,
        labels=meta.label.to_numpy(np.int64),
        subjects=meta.subject.to_numpy(np.int64),
        split=meta.split.to_numpy(),
    )
    print(f"Saved cache {C.CACHE_FILE} with images {images.shape}")


def load_splits() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {'train': (X, y), 'val': ..., 'test': ...} with X as uint8 (N, 128, 128)."""
    if not C.CACHE_FILE.exists():
        prepare()
    d = np.load(C.CACHE_FILE, allow_pickle=True)
    out = {}
    for s in ("train", "val", "test"):
        m = d["split"] == s
        out[s] = (d["images"][m], d["labels"][m])
    return out


class TensorBatches:
    """Minimal in-memory batch iterator; data lives on the target device as uint8."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
                 device: torch.device, seed: int = C.SEED):
        self.X = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.bs, self.shuffle = batch_size, shuffle
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return math.ceil(len(self.y) / self.bs)

    def __iter__(self):
        n = len(self.y)
        idx = torch.randperm(n, generator=self.gen) if self.shuffle else torch.arange(n)
        idx = idx.to(self.X.device)
        for i in range(0, n, self.bs):
            b = idx[i:i + self.bs]
            xb = self.X[b]
            if xb.dtype == torch.uint8:
                # uint8 image -> float in [0, 1], add channel dim: (B, 1, H, W)
                xb = xb.unsqueeze(1).float().div_(255.0)
            yield xb, self.y[b]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run LeapGestRecog preprocessing")
    ap.add_argument("--dataset-dir", default=None, help="Path to the extracted dataset")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    prepare(args.dataset_dir, args.force)
