"""Run the full experiment suite (all models + ablations) and build the
comparison table. Cross-platform alternative to run_all.sh.

    python scripts/run_all.py                    # everything, default epochs
    python scripts/run_all.py --epochs 15        # faster
    python scripts/run_all.py --only cnn mlp     # subset
    python scripts/run_all.py --skip-existing    # resume after an interruption
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (run_name, extra args)
RUNS = [
    ("mlp", ["--model", "mlp"]),                                   # Model 1
    ("hog_mlp", ["--model", "hog_mlp"]),                           # Model 1 ablation (HOG features)
    ("cnn", ["--model", "cnn"]),                                   # Model 2
    ("cnn_noaug", ["--model", "cnn", "--no-augment"]),             # Model 2 ablation (no augmentation)
    ("mobilenet_v2", ["--model", "mobilenet_v2"]),                 # Model 3 (head -> full fine-tune)
    ("mobilenet_v2_frozen", ["--model", "mobilenet_v2", "--freeze-epochs", "9999"]),  # Model 3 ablation
    ("resnet18", ["--model", "resnet18"]),                         # Model 3 alt. backbone
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="run names to include")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args, passthrough = ap.parse_known_args()
    os.chdir(ROOT)
    for name, extra in RUNS:
        if args.only and name not in args.only:
            continue
        if args.skip_existing and (ROOT / "results" / name / "metrics_test.json").exists():
            print(f"=== skipping {name} (already done)")
            continue
        cmd = [sys.executable, "-m", "src.train", "--run-name", name, *extra, *passthrough]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        print("===", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    results_dir = "results"
    if "--out-dir" in passthrough:
        results_dir = passthrough[passthrough.index("--out-dir") + 1]
    subprocess.run([sys.executable, "-m", "src.compare", "--results-dir", results_dir,
                    *(["--no-readme"] if results_dir != "results" else [])], check=True)


if __name__ == "__main__":
    main()
