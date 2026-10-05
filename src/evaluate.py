"""Evaluation: accuracy, macro precision/recall/F1, per-class report,
confusion matrix, parameters, FLOPs and latency.

Usage (stand-alone)
-------------------
    # re-evaluate a trained checkpoint on the LeapGestRecog test split
    python -m src.evaluate --checkpoint checkpoints/cnn.pt --data-dir data/leapgestrecog

    # domain-shift test on your own webcam images (data/webcam/<class>/*.png)
    python -m src.evaluate --checkpoint checkpoints/cnn.pt --webcam-dir data/webcam
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from .models import MODEL_FAMILY, build_model
from .utils import count_macs, count_params, get_device, measure_latency, model_size_mb, save_json


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True)).float()
        pr = out.softmax(1)
        ys.append(y.numpy())
        ps.append(pr.argmax(1).cpu().numpy())
        probs.append(pr.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)


def classification_metrics(y_true, y_pred, class_names) -> dict:
    labels = list(range(len(class_names)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "per_class": {
            n: {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(fc[i]), "support": int(sc[i])}
            for i, n in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion(cm, class_names, path, title=""):
    cm = np.asarray(cm)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (row-normalised %)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def complexity_report(model, img_size: int, latency_runs: int = 100) -> dict:
    model_cpu = model.to("cpu").eval()
    macs = count_macs(model_cpu, img_size)
    return {
        "params": count_params(model_cpu),
        "size_mb": model_size_mb(model_cpu),
        "macs": macs,
        "gflops": 2 * macs / 1e9,
        "latency_cpu_bs1": measure_latency(model_cpu, img_size, "cpu", runs=latency_runs),
    }


def full_evaluation(model, loader, class_names, out_dir: Path, device, img_size: int, run_name: str,
                    model_name: str, split_name: str = "test", latency_runs: int = 100,
                    extra: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y, p, _ = predict(model.to(device), loader, device)
    m = classification_metrics(y, p, class_names)
    report = classification_report(y, p, labels=list(range(len(class_names))), target_names=class_names,
                                   digits=4, zero_division=0)
    (out_dir / f"classification_report_{split_name}.txt").write_text(report)
    plot_confusion(m["confusion_matrix"], class_names, out_dir / f"confusion_matrix_{split_name}.png",
                   f"{run_name} — {split_name} (acc {m['accuracy']*100:.2f}%)")
    result = {"run_name": run_name, "model": model_name, "family": MODEL_FAMILY.get(model_name, model_name),
              "split": split_name, "class_names": class_names, **m}
    result["complexity"] = complexity_report(model, img_size, latency_runs)
    model.to(device)
    if extra:
        result.update(extra)
    save_json(result, out_dir / f"metrics_{split_name}.json")
    print(report)
    c = result["complexity"]
    print(f"[eval] {run_name}/{split_name}: acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
          f"params={c['params']['total']:,} GFLOPs={c['gflops']:.3f} "
          f"latency={c['latency_cpu_bs1']['mean_ms']:.2f}ms")
    return result


def load_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["class_names"]), ckpt["img_size"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/leapgestrecog")
    ap.add_argument("--webcam-dir", default=None, help="flat folder <class>/*.png for domain-shift testing")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    run_name = ckpt.get("run_name", ckpt["model_name"])
    out_dir = Path(args.out_dir or Path("results") / run_name)
    if args.webcam_dir:
        from .data import make_flat_loader

        loader = make_flat_loader(args.webcam_dir, ckpt["class_names"], ckpt["img_size"], args.batch_size)
        split = "webcam"
    else:
        from .data import make_dataloaders

        cfg = ckpt["config"]
        bundle = make_dataloaders(args.data_dir, ckpt["img_size"], args.batch_size, cfg["split"], num_workers=0,
                                  seed=cfg["seed"], val_subjects=cfg["val_subjects"],
                                  test_subjects=cfg["test_subjects"])
        loader, split = bundle.loaders["test"], "test"
    full_evaluation(model, loader, ckpt["class_names"], out_dir, device, ckpt["img_size"], run_name,
                    ckpt["model_name"], split)


if __name__ == "__main__":
    main()
