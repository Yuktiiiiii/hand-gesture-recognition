"""Train and evaluate one experiment under an identical protocol.

Experiments
  mlp              MLP on flattened 128x128 pixels (augmented training)
  mlp_hog          MLP on hand-crafted HOG features (feature ablation)
  cnn              Custom CNN from scratch (augmented training)
  cnn_noaug        Custom CNN without augmentation (augmentation ablation)
  mobilenetv2      MobileNetV2: stage 1 frozen backbone, stage 2 full fine-tuning

Usage: python -m src.train --exp cnn [--epochs N]
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .augment import augment_batch
from .data import TensorBatches, load_splits
from .evaluate import (classification_metrics, compute_flops, count_params, measure_latency,
                       per_class_report, plot_confusion, plot_history, predict)
from .models import MLP, CustomCNN, MobileNetV2Transfer

PRETTY = {
    "mlp": "MLP (pixels)",
    "mlp_hog": "MLP (HOG features)",
    "cnn": "Custom CNN",
    "cnn_noaug": "Custom CNN (no augmentation)",
    "mobilenetv2": "MobileNetV2 (fine-tuned)",
    "mobilenetv2_frozen": "MobileNetV2 (frozen backbone)",
}

# Per-experiment hyperparameters (identical protocol: AdamW, cosine LR, best-val checkpoint)
DEFAULTS = {
    "mlp":       dict(epochs=40, lr=1e-3, wd=1e-4, augment=True,  batch=128, patience=8),
    "mlp_hog":   dict(epochs=40, lr=1e-3, wd=1e-4, augment=False, batch=128, patience=8),
    "cnn":       dict(epochs=30, lr=1e-3, wd=1e-4, augment=True,  batch=64,  patience=8),
    "cnn_noaug": dict(epochs=25, lr=1e-3, wd=1e-4, augment=False, batch=64,  patience=8),
}
MOBILENET_STAGES = [
    dict(stage="frozen",   epochs=6,  lr=1e-3, wd=1e-4, augment=True, batch=64, patience=6),
    dict(stage="finetune", epochs=10, lr=1e-4, wd=1e-4, augment=True, batch=64, patience=5),
]


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hog_features(images: np.ndarray) -> np.ndarray:
    from skimage.feature import hog
    return np.stack([
        hog(im, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
            block_norm="L2-Hys", feature_vector=True)
        for im in images
    ]).astype(np.float32)


def load_hog_splits(splits):
    cache = C.DATA_DIR / "hog_features.npz"
    if cache.exists():
        d = np.load(cache)
        feats = {s: d[s] for s in ("train", "val", "test")}
    else:
        print("Extracting HOG features ...")
        feats = {s: hog_features(splits[s][0]) for s in ("train", "val", "test")}
        np.savez_compressed(cache, **feats)
    mu, sd = feats["train"].mean(0), feats["train"].std(0) + 1e-6
    return {s: ((feats[s] - mu) / sd, splits[s][1]) for s in feats}, (mu, sd)


def run_epoch(model, batches, criterion, optimizer=None, augment=False):
    train = optimizer is not None
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in batches:
            if train and augment:
                xb = augment_batch(xb)
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
    return tot_loss / n, correct / n


def fit(model, data, device, stage_cfg, history, ckpt_path, stage="main"):
    """Train with AdamW + cosine LR, keep the checkpoint with best val accuracy."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=stage_cfg["lr"], weight_decay=stage_cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stage_cfg["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    train_b = TensorBatches(*data["train"], stage_cfg["batch"], True, device)
    val_b = TensorBatches(*data["val"], 256, False, device)

    best = (-1.0, float("inf"))
    best_state, bad = None, 0
    start_epoch = len(history)
    for e in range(stage_cfg["epochs"]):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_b, criterion, opt, stage_cfg["augment"])
        va_loss, va_acc = run_epoch(model, val_b, criterion)
        sched.step()
        history.append(dict(epoch=start_epoch + e + 1, stage=stage, train_loss=tr_loss,
                            train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc,
                            lr=opt.param_groups[0]["lr"], seconds=time.time() - t0))
        print(f"[{stage}] epoch {e + 1:02d}/{stage_cfg['epochs']}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}"
              f"  ({time.time() - t0:.1f}s)", flush=True)
        if (va_acc, -va_loss) > (best[0], -best[1]):
            best, bad = (va_acc, va_loss), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
        else:
            bad += 1
            if bad >= stage_cfg["patience"]:
                print(f"Early stopping after {e + 1} epochs")
                break
    model.load_state_dict(best_state)
    return best


def evaluate_and_save(name, model, data, device, input_shape, history, train_seconds, extra=None):
    out_dir = C.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"experiment": name, "model": PRETTY[name]}
    for split in ("train", "val", "test"):
        y, p = predict(model, TensorBatches(*data[split], 256, False, device))
        result[split] = classification_metrics(y, p)
        if split == "test":
            np.savez(out_dir / "test_predictions.npz", y_true=y, y_pred=p)
            with open(out_dir / "per_class_report.json", "w") as f:
                json.dump(per_class_report(y, p), f, indent=2)
        if split in ("train", "test"):
            plot_confusion(y, p, f"{PRETTY[name]} - {split} confusion matrix",
                           C.FIG_DIR / f"cm_{name}_{split}.png")

    cpu = torch.device("cpu")
    result["params"] = count_params(model)
    result["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result["macs"] = compute_flops(model, input_shape)
    result["latency_ms_cpu_b1"] = measure_latency(model, input_shape, cpu, 1)
    result["latency_ms_device_b1"] = measure_latency(model, input_shape, device, 1)
    result["throughput_img_s_device_b64"] = 64 / measure_latency(model, input_shape, device, 64,
                                                                 runs=30) * 1000
    result["device"] = device.type
    result["train_seconds"] = train_seconds
    result["epochs_run"] = len(history)
    if extra:
        result.update(extra)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    if history:
        plot_history(history, PRETTY[name], C.FIG_DIR / f"curves_{name}.png")
    t = result["test"]
    print(f"==> {PRETTY[name]}: test acc {t['accuracy']:.4f}  P {t['precision_macro']:.4f}  "
          f"R {t['recall_macro']:.4f}  F1 {t['f1_macro']:.4f}  params {result['params']:,}")
    return result


def main(exp: str, epochs: int | None = None):
    set_seed()
    device = C.get_device()
    print(f"Experiment {exp} on {device}")
    splits = load_splits()
    for s, (X, y) in splits.items():
        print(f"  {s:5s}: {len(y):5d} images, class counts {np.bincount(y, minlength=C.NUM_CLASSES)}")

    if exp in ("mlp", "mlp_hog", "cnn", "cnn_noaug"):
        cfg = dict(DEFAULTS[exp])
        if epochs:
            cfg["epochs"] = epochs
        extra = {"config": cfg}
        if exp == "mlp_hog":
            data, _ = load_hog_splits(splits)
            in_dim = data["train"][0].shape[1]
            model = MLP(in_features=in_dim)
            input_shape = (in_dim,)
            t0 = time.perf_counter()
            hog_features(splits["test"][0][:200])
            extra["hog_ms_per_image"] = (time.perf_counter() - t0) / 200 * 1000
            extra["hog_dim"] = in_dim
        else:
            data = splits
            model = MLP() if exp == "mlp" else CustomCNN()
            input_shape = (1, C.IMG_SIZE, C.IMG_SIZE)
        model.to(device)
        history = []
        t0 = time.time()
        fit(model, data, device, cfg, history, C.CKPT_DIR / f"{exp}.pt", stage=exp)
        return evaluate_and_save(exp, model, data, device, input_shape, history,
                                 time.time() - t0, extra)

    if exp == "mobilenetv2":
        model = MobileNetV2Transfer(pretrained=True).to(device)
        input_shape = (1, C.IMG_SIZE, C.IMG_SIZE)
        stages = [dict(s) for s in MOBILENET_STAGES]
        if epochs:
            for s in stages:
                s["epochs"] = epochs
        history = []
        t0 = time.time()

        model.freeze_backbone(True)
        # BatchNorm statistics in the frozen backbone stay fixed: keep it in eval mode
        fit_frozen(model, splits, device, stages[0], history)
        frozen_time = time.time() - t0
        res_frozen = evaluate_and_save("mobilenetv2_frozen", model, splits, device, input_shape,
                                       list(history), frozen_time, {"config": stages[0]})

        model.freeze_backbone(False)
        fit(model, splits, device, stages[1], history, C.CKPT_DIR / "mobilenetv2.pt",
            stage="finetune")
        res = evaluate_and_save("mobilenetv2", model, splits, device, input_shape, history,
                                time.time() - t0, {"config": stages})
        return res_frozen, res

    raise ValueError(f"Unknown experiment {exp}")


def fit_frozen(model, data, device, cfg, history):
    """Stage 1: train only the head; frozen backbone BatchNorm layers kept in eval mode."""
    orig_train = model.train

    def train_mode(mode=True):
        orig_train(mode)
        if mode:
            model.features.eval()
        return model

    model.train = train_mode
    try:
        fit(model, data, device, cfg, history, C.CKPT_DIR / "mobilenetv2_frozen.pt", stage="frozen")
    finally:
        model.train = orig_train


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["mlp", "mlp_hog", "cnn", "cnn_noaug", "mobilenetv2"])
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs (quick runs)")
    args = ap.parse_args()
    main(args.exp, args.epochs)
