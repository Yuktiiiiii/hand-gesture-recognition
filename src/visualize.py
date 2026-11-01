"""Dataset figures: samples per class, augmentation examples, split distribution."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config as C
from .augment import augment_batch
from .data import load_splits


def main():
    splits = load_splits()
    X, y = splits["train"]
    rng = np.random.default_rng(C.SEED)

    fig, axes = plt.subplots(3, C.NUM_CLASSES, figsize=(16, 5.4))
    for c in range(C.NUM_CLASSES):
        idx = rng.choice(np.flatnonzero(y == c), 3, replace=False)
        for r, i in enumerate(idx):
            axes[r, c].imshow(X[i], cmap="gray")
            axes[r, c].axis("off")
        axes[0, c].set_title(f"{c}: {C.CLASS_NAMES[c]}", fontsize=9)
    fig.suptitle("LeapGestRecog samples after resizing to 128x128 (training subjects)")
    plt.tight_layout()
    fig.savefig(C.FIG_DIR / "dataset_samples.png", dpi=130)
    plt.close(fig)

    torch.manual_seed(C.SEED)
    picks = [np.flatnonzero(y == c)[50] for c in (0, 1, 4, 6, 8)]
    base = torch.from_numpy(X[picks]).unsqueeze(1).float() / 255.0
    fig, axes = plt.subplots(len(picks), 6, figsize=(11, 9))
    for r in range(len(picks)):
        axes[r, 0].imshow(base[r, 0], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title("original" if r == 0 else "", fontsize=9)
        aug = augment_batch(base[r:r + 1].repeat(5, 1, 1, 1))
        for j in range(5):
            axes[r, j + 1].imshow(aug[j, 0], cmap="gray", vmin=0, vmax=1)
            if r == 0:
                axes[r, j + 1].set_title(f"augmented {j + 1}", fontsize=9)
        for a in axes[r]:
            a.axis("off")
    fig.suptitle("Training-time augmentation: rotation, translation, zoom, brightness/contrast")
    plt.tight_layout()
    fig.savefig(C.FIG_DIR / "augmentation_examples.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.27
    x = np.arange(C.NUM_CLASSES)
    for i, s in enumerate(("train", "val", "test")):
        ax.bar(x + (i - 1) * w, np.bincount(splits[s][1], minlength=C.NUM_CLASSES), w,
               label=f"{s} ({len(splits[s][1])})")
    ax.set_xticks(x, C.CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Images")
    ax.set_title("Class distribution per subject-wise split (train: S0-6, val: S7, test: S8-9)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(C.FIG_DIR / "split_distribution.png", dpi=130)
    plt.close(fig)
    print("Saved dataset figures to", C.FIG_DIR)


if __name__ == "__main__":
    main()
