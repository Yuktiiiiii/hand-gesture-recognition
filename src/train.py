"""Train one model under the shared experimental protocol, then evaluate it
on the held-out test subjects.

Examples
--------
    python -m src.train --model mlp
    python -m src.train --model hog_mlp
    python -m src.train --model cnn
    python -m src.train --model cnn --no-augment --run-name cnn_noaug
    python -m src.train --model mobilenet_v2
    python -m src.train --model mobilenet_v2 --freeze-epochs 999 --run-name mobilenet_v2_frozen
    python -m src.train --model resnet18

Outputs
-------
    checkpoints/<run_name>.pt                 best-on-validation weights + metadata
    results/<run_name>/history.csv            per-epoch loss / accuracy / lr
    results/<run_name>/training_curves.png
    results/<run_name>/metrics_test.json      all metrics + complexity
    results/<run_name>/confusion_matrix_test.png
    results/<run_name>/classification_report_test.txt
"""
from __future__ import annotations

import argparse
import csv
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .data import make_dataloaders
from .evaluate import full_evaluation
from .models import MODEL_NAMES, build_model, is_transfer
from .utils import count_params, get_device, save_json, seed_everything

# Sensible per-model defaults (can be overridden on the CLI)
DEFAULT_LR = {"mlp": 1e-3, "hog_mlp": 1e-3, "cnn": 2e-3, "mobilenet_v2": 1e-3, "resnet18": 1e-3}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--run-name", default=None, help="output name (default: model name)")
    ap.add_argument("--data-dir", default="data/leapgestrecog")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--split", default="subject", choices=["subject", "random"])
    ap.add_argument("--val-subjects", nargs="+", default=["06", "07"])
    ap.add_argument("--test-subjects", nargs="+", default=["08", "09"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=8, help="early-stopping patience (epochs)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="transfer models: random init")
    ap.add_argument("--freeze-epochs", type=int, default=3,
                    help="transfer models: epochs training only the head before full fine-tuning")
    ap.add_argument("--finetune-lr-mult", type=float, default=0.1,
                    help="transfer models: backbone LR = lr * mult during fine-tuning")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latency-runs", type=int, default=100)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    return ap.parse_args(argv)


def make_optimizer(model, args, phase: str):
    lr = args.lr
    if is_transfer(args.model) and phase == "finetune":
        groups = [
            {"params": list(model.backbone_parameters()), "lr": lr * args.finetune_lr_mult},
            {"params": list(model.head_parameters()), "lr": lr},
        ]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            autocast = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
            with autocast:
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def plot_history(hist, path, title):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="val")
    ax[0].set_title("Loss")
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val")
    ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    args.lr = args.lr or DEFAULT_LR[args.model]
    run_name = args.run_name or args.model
    seed_everything(args.seed)
    device = get_device(args.device)
    amp = device.type == "cuda"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt_dir) / f"{run_name}.pt"
    print(f"[train] run={run_name} model={args.model} device={device} amp={amp}")

    bundle = make_dataloaders(args.data_dir, args.img_size, args.batch_size, args.split, not args.no_augment,
                              args.num_workers, args.seed, args.cache_dir, args.val_subjects, args.test_subjects)
    class_names = bundle.class_names
    model = build_model(args.model, len(class_names), args.img_size, pretrained=not args.no_pretrained).to(device)
    print(f"[train] params: {count_params(model)['total']:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    # Phase setup
    transfer = is_transfer(args.model)
    # Freezing a randomly initialised backbone is meaningless, so the head-only
    # phase is only used with pretrained weights.
    phase = "head" if transfer and args.freeze_epochs > 0 and not args.no_pretrained else "full"
    if phase == "head":
        model.freeze_backbone()
    optimizer = make_optimizer(model, args, phase)
    head_epochs = min(args.freeze_epochs, args.epochs) if phase == "head" else 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, (head_epochs or args.epochs)))

    best_acc, best_epoch, bad, hist = -1.0, 0, 0, []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        if transfer and phase == "head" and epoch == head_epochs + 1:
            phase = "finetune"
            model.unfreeze()
            optimizer = make_optimizer(model, args, phase)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - head_epochs))
            bad = 0
            print(f"[train] epoch {epoch}: unfreezing backbone for fine-tuning")
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, bundle.loaders["train"], criterion, device, optimizer, scaler, amp)
        va_loss, va_acc = run_epoch(model, bundle.loaders["val"], criterion, device, amp=amp)
        lr = optimizer.param_groups[-1]["lr"]
        scheduler.step()
        hist.append({"epoch": epoch, "phase": phase, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": va_loss, "val_acc": va_acc, "lr": lr, "time_s": time.time() - t0})
        flag = ""
        if va_acc > best_acc:
            best_acc, best_epoch, bad, flag = va_acc, epoch, 0, " *"
            torch.save({"model_name": args.model, "run_name": run_name, "class_names": class_names,
                        "img_size": args.img_size, "state_dict": model.state_dict(), "epoch": epoch,
                        "val_acc": va_acc, "config": vars(args)}, ckpt_path)
        else:
            bad += 1
        print(f"[{run_name}] ep {epoch:02d}/{args.epochs} {phase:8s} train {tr_loss:.4f}/{tr_acc:.4f} "
              f"val {va_loss:.4f}/{va_acc:.4f} lr {lr:.2e} ({time.time()-t0:.1f}s){flag}")
        # Never early-stop during the frozen-head phase of transfer learning
        if bad >= args.patience and phase != "head":
            print(f"[train] early stopping at epoch {epoch} (best epoch {best_epoch})")
            break
    train_time = time.time() - t_start

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    plot_history(hist, out_dir / "training_curves.png", run_name)

    # Evaluate best checkpoint on the untouched test subjects
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    extra = {"best_epoch": best_epoch, "best_val_acc": best_acc, "epochs_run": len(hist),
             "train_time_s": train_time, "split_sizes": bundle.split_sizes, "config": vars(args)}
    res = full_evaluation(model, bundle.loaders["test"], class_names, out_dir, device, args.img_size, run_name,
                          args.model, "test", args.latency_runs, extra)
    save_json({"run_name": run_name, "test_accuracy": res["accuracy"], "test_f1_macro": res["f1_macro"]},
              out_dir / "summary.json")
    return res


if __name__ == "__main__":
    main()
