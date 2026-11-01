"""Evaluation: classification metrics, confusion matrices and computational cost."""
import copy
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from . import config as C


@torch.no_grad()
def predict(model: torch.nn.Module, batches) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for xb, yb in batches:
        preds.append(model(xb).argmax(1).cpu())
        ys.append(yb.cpu())
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
    }


def per_class_report(y_true, y_pred) -> dict:
    return classification_report(y_true, y_pred, labels=list(range(C.NUM_CLASSES)),
                                 target_names=C.CLASS_NAMES, output_dict=True, zero_division=0)


def plot_confusion(y_true, y_pred, title: str, path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C.NUM_CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, ax=ax,
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES,
                cbar_kws={"label": "Row-normalised recall"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: list[dict], title: str, path) -> None:
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["val_loss"] for h in history], label="val")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.legend(); a1.set_title("Loss")
    a2.plot(ep, [h["train_acc"] for h in history], label="train")
    a2.plot(ep, [h["val_acc"] for h in history], label="val")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy"); a2.legend(); a2.set_title("Accuracy")
    stage_changes = [h["epoch"] for i, h in enumerate(history)
                     if i > 0 and h.get("stage") != history[i - 1].get("stage")]
    for e in stage_changes:
        for a in (a1, a2):
            a.axvline(e - 0.5, color="gray", ls="--", lw=1)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_flops(model: torch.nn.Module, input_shape: tuple) -> float:
    """Multiply-accumulate operations (MACs) for one sample, via thop."""
    from thop import profile
    m = copy.deepcopy(model).cpu().eval()
    macs, _ = profile(m, inputs=(torch.zeros(1, *input_shape),), verbose=False)
    return float(macs)


@torch.no_grad()
def measure_latency(model: torch.nn.Module, input_shape: tuple, device: torch.device,
                    batch_size: int = 1, warmup: int = 10, runs: int = 100) -> float:
    """Mean milliseconds per forward pass of `batch_size` samples."""
    m = copy.deepcopy(model).to(device).eval()
    x = torch.rand(batch_size, *input_shape, device=device)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        m(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(runs):
        m(x)
    sync()
    return (time.perf_counter() - t0) / runs * 1000
