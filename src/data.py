"""Data pipeline for LeapGestRecog.

Responsibilities
----------------
* Locate the dataset root (the Kaggle archive contains a duplicated nested
  copy, so we pick exactly one root and ignore the other).
* Build a sample list of (path, label, subject).
* Split by **subject** (default) so that no person appears in more than one
  split. LeapGestRecog frames are consecutive video frames, so a random
  split leaks near-identical images between train and test and inflates
  accuracy to ~100%. The random split is still available for comparison.
* Cache all images as a resized uint8 array (``data/cache/*.npz``) so that
  every model / run reads the exact same pixels quickly.
* Provide identical augmentation + normalisation for every model.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

SUBJECT_DIR_RE = re.compile(r"^\d{2}$")
GESTURE_DIR_RE = re.compile(r"^(\d{2})_(.+)$")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# Normalisation applied to every model: grayscale in [0,1] -> [-1,1]
MEAN, STD = 0.5, 0.5


@dataclass
class Sample:
    path: str
    label: int
    subject: str


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def _is_leap_root(p: Path) -> bool:
    try:
        subjects = [c for c in p.iterdir() if c.is_dir() and SUBJECT_DIR_RE.match(c.name)]
    except (PermissionError, FileNotFoundError):
        return False
    if len(subjects) < 2:
        return False
    first = sorted(subjects)[0]
    return any(g.is_dir() and GESTURE_DIR_RE.match(g.name) for g in first.iterdir())


def find_dataset_root(data_dir: str | os.PathLike, max_depth: int = 4) -> Path:
    """Return the shallowest directory under ``data_dir`` that looks like
    ``<root>/<subject 00..09>/<NN_gesture>/*.png``."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. Run `python scripts/download_data.py` "
            "or pass --data-dir pointing at the extracted LeapGestRecog folder."
        )
    base_depth = len(data_dir.resolve().parts)
    for dirpath, dirnames, _ in sorted(os.walk(data_dir), key=lambda t: (len(Path(t[0]).parts), t[0])):
        p = Path(dirpath)
        if len(p.resolve().parts) - base_depth > max_depth:
            dirnames[:] = []
            continue
        if _is_leap_root(p):
            return p
    raise FileNotFoundError(f"Could not find LeapGestRecog structure under '{data_dir}'.")


def gesture_display_name(folder: str) -> str:
    m = GESTURE_DIR_RE.match(folder)
    return m.group(2) if m else folder


def build_samples(root: Path) -> tuple[list[Sample], list[str]]:
    """Scan the LeapGestRecog tree. Returns samples and ordered class names."""
    subjects = sorted(c for c in root.iterdir() if c.is_dir() and SUBJECT_DIR_RE.match(c.name))
    gesture_folders = sorted(
        {g.name for s in subjects for g in s.iterdir() if g.is_dir() and GESTURE_DIR_RE.match(g.name)}
    )
    class_names = [gesture_display_name(g) for g in gesture_folders]
    folder_to_idx = {g: i for i, g in enumerate(gesture_folders)}
    samples: list[Sample] = []
    for s in subjects:
        for g in sorted(s.iterdir()):
            if not (g.is_dir() and g.name in folder_to_idx):
                continue
            for f in sorted(g.iterdir()):
                if f.suffix.lower() in IMG_EXTS:
                    samples.append(Sample(str(f), folder_to_idx[g.name], s.name))
    if not samples:
        raise RuntimeError(f"No images found under {root}")
    return samples, class_names


def build_flat_samples(root: Path, class_names: list[str]) -> list[Sample]:
    """Samples from a flat ``<root>/<class_name>/*.png`` folder (e.g. the
    self-recorded webcam set). Folder names must match ``class_names``."""
    samples = []
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        name = gesture_display_name(d.name)
        if name not in name_to_idx:
            print(f"[warn] skipping folder '{d.name}': not one of {class_names}")
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                samples.append(Sample(str(f), name_to_idx[name], "webcam"))
    return samples


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------
def split_samples(
    samples: list[Sample],
    mode: str = "subject",
    val_subjects: tuple[str, ...] = ("06", "07"),
    test_subjects: tuple[str, ...] = ("08", "09"),
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Return indices for train / val / test."""
    idx = np.arange(len(samples))
    if mode == "subject":
        subj = np.array([s.subject for s in samples])
        test = idx[np.isin(subj, test_subjects)]
        val = idx[np.isin(subj, val_subjects)]
        train = idx[~np.isin(subj, list(val_subjects) + list(test_subjects))]
        if len(test) == 0 or len(val) == 0:
            raise ValueError(
                f"Subject split produced an empty split. Subjects present: {sorted(set(subj))}"
            )
    elif mode == "random":
        from sklearn.model_selection import train_test_split

        labels = np.array([s.label for s in samples])
        trval, test = train_test_split(idx, test_size=test_frac, stratify=labels, random_state=seed)
        train, val = train_test_split(
            trval, test_size=val_frac / (1 - test_frac), stratify=labels[trval], random_state=seed
        )
    else:
        raise ValueError(f"Unknown split mode '{mode}'")
    return {"train": sorted(train.tolist()), "val": sorted(val.tolist()), "test": sorted(test.tolist())}


# ----------------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------------
def load_image(path: str, img_size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L").resize((img_size, img_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def load_or_build_cache(samples: list[Sample], img_size: int, cache_dir: str | os.PathLike) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(("|".join(s.path for s in samples) + f"@{img_size}").encode()).hexdigest()[:12]
    f = cache_dir / f"images_{img_size}_{len(samples)}_{key}.npy"
    if f.exists():
        arr = np.load(f)
        if arr.shape == (len(samples), img_size, img_size):
            return arr
    print(f"[data] building image cache ({len(samples)} images @ {img_size}px) -> {f}")
    from concurrent.futures import ThreadPoolExecutor

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731
    arr = np.empty((len(samples), img_size, img_size), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 2) * 2)) as ex:
        for i, im in enumerate(tqdm(ex.map(lambda s: load_image(s.path, img_size), samples), total=len(samples))):
            arr[i] = im
    np.save(f, arr)
    return arr


# ----------------------------------------------------------------------------
# Transforms / Dataset
# ----------------------------------------------------------------------------
def build_transforms(train: bool, augment: bool = True):
    """Identical transforms for every model. Input: uint8 tensor (1,H,W)."""
    ops = []
    if train and augment:
        ops += [
            T.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),  # left vs right hand
            T.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    ops += [T.ToDtype(torch.float32, scale=True), T.Normalize([MEAN], [STD])]
    return T.Compose(ops)


class GestureArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.images[i]))[None]  # (1,H,W) uint8
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[i])


def preprocess_gray(img_gray: np.ndarray, img_size: int) -> torch.Tensor:
    """Single grayscale image (H,W uint8) -> normalised tensor (1,1,S,S).
    Used by the webcam demo so inference matches training preprocessing."""
    im = Image.fromarray(img_gray).resize((img_size, img_size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None, None]
    return (x - MEAN) / STD


@dataclass
class DataBundle:
    loaders: dict[str, DataLoader]
    class_names: list[str]
    split_sizes: dict[str, int]
    root: str


def make_dataloaders(
    data_dir: str,
    img_size: int = 128,
    batch_size: int = 64,
    split: str = "subject",
    augment: bool = True,
    num_workers: int = 2,
    seed: int = 42,
    cache_dir: str = "data/cache",
    val_subjects=("06", "07"),
    test_subjects=("08", "09"),
) -> DataBundle:
    root = find_dataset_root(data_dir)
    samples, class_names = build_samples(root)
    images = load_or_build_cache(samples, img_size, cache_dir)
    labels = np.array([s.label for s in samples], dtype=np.int64)
    idx = split_samples(samples, split, tuple(val_subjects), tuple(test_subjects), seed=seed)

    g = torch.Generator().manual_seed(seed)
    loaders = {}
    for name, ids in idx.items():
        ds = GestureArrayDataset(images[ids], labels[ids], build_transforms(name == "train", augment))
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            generator=g if name == "train" else None,
            persistent_workers=num_workers > 0,
        )
    sizes = {k: len(v) for k, v in idx.items()}
    print(f"[data] root={root}  classes={class_names}  split={split} sizes={sizes}")
    return DataBundle(loaders, class_names, sizes, str(root))


def make_flat_loader(folder: str, class_names: list[str], img_size: int, batch_size: int = 64) -> DataLoader:
    samples = build_flat_samples(Path(folder), class_names)
    if not samples:
        raise RuntimeError(f"No images found in {folder}")
    images = np.stack([load_image(s.path, img_size) for s in samples])
    labels = np.array([s.label for s in samples], dtype=np.int64)
    ds = GestureArrayDataset(images, labels, build_transforms(train=False))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
