"""Run the full pipeline: preprocessing, all experiments, comparison tables."""
import argparse

from src import compare, data, train

EXPERIMENTS = ["mlp", "mlp_hog", "cnn", "cnn_noaug", "mobilenetv2"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=None, help="Extracted LeapGestRecog (else kagglehub)")
    ap.add_argument("--only", nargs="*", choices=EXPERIMENTS, default=EXPERIMENTS)
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs for a quick run")
    args = ap.parse_args()

    data.prepare(args.dataset_dir)
    for exp in args.only:
        train.main(exp, args.epochs)
    compare.main()
