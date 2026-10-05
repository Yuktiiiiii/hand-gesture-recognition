"""Aggregate every ``results/<run>/metrics_test.json`` into a comparison
table and plots, and write the table into README.md.

    python -m src.compare

Outputs (in results/):
    comparison.csv, comparison.md
    comparison_accuracy_f1.png       accuracy and macro-F1 per run
    comparison_accuracy_vs_latency.png   accuracy vs CPU latency (bubble = params)
    comparison_per_class_f1.png      per-class F1 heat-map across runs
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import load_json

ABLATION_LABEL = {
    "cnn_noaug": "Model 2 · Custom CNN, no augmentation (ablation)",
    "mobilenet_v2_frozen": "Model 3 · MobileNetV2, frozen backbone (ablation)",
}
ORDER = ["mlp", "hog_mlp", "cnn_noaug", "cnn", "mobilenet_v2_frozen", "mobilenet_v2", "resnet18"]


def collect(results_dir: Path, split: str = "test") -> tuple[pd.DataFrame, dict]:
    rows, per_class = [], {}
    for f in sorted(results_dir.glob(f"*/metrics_{split}.json")):
        m = load_json(f)
        c = m["complexity"]
        rows.append({
            "run": m["run_name"],
            "family": ABLATION_LABEL.get(m["run_name"], m["family"]),
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            "params_M": c["params"]["total"] / 1e6,
            "size_MB": c["size_mb"],
            "GFLOPs": c["gflops"],
            "latency_ms": c["latency_cpu_bs1"]["mean_ms"],
            "fps": c["latency_cpu_bs1"]["fps"],
            "best_epoch": m.get("best_epoch"),
            "train_time_min": (m.get("train_time_s") or 0) / 60,
        })
        per_class[m["run_name"]] = {k: v["f1"] for k, v in m["per_class"].items()}
    if not rows:
        raise SystemExit(f"No metrics_{split}.json found under {results_dir}. Train some models first.")
    df = pd.DataFrame(rows)
    df["_o"] = df["run"].apply(lambda r: ORDER.index(r) if r in ORDER else len(ORDER))
    df = df.sort_values(["_o", "run"]).drop(columns="_o").reset_index(drop=True)
    return df, per_class


def to_markdown(df: pd.DataFrame) -> str:
    hdr = ("| Run | Model family | Accuracy | Precision | Recall | Macro-F1 | Params (M) | GFLOPs "
           "| CPU latency (ms) |\n|---|---|---|---|---|---|---|---|---|\n")
    best = df["accuracy"].max()
    lines = []
    for _, r in df.iterrows():
        acc = f"{r.accuracy*100:.2f}%"
        if r.accuracy == best:
            acc = f"**{acc}**"
        lines.append(f"| `{r.run}` | {r.family} | {acc} | {r.precision_macro*100:.2f}% | {r.recall_macro*100:.2f}% "
                     f"| {r.f1_macro*100:.2f}% | {r.params_M:.2f} | {r.GFLOPs:.3f} | {r.latency_ms:.2f} |")
    return hdr + "\n".join(lines) + "\n"


def plots(df: pd.DataFrame, per_class: dict, out: Path):
    # 1. accuracy / F1 bars
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(df)), 4.2))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.accuracy * 100, 0.4, label="Accuracy", color="#3b6fb6")
    ax.bar(x + 0.2, df.f1_macro * 100, 0.4, label="Macro-F1", color="#e08a3c")
    for i, (a, f) in enumerate(zip(df.accuracy, df.f1_macro)):
        ax.text(i - 0.2, a * 100 + 0.5, f"{a*100:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, f * 100 + 0.5, f"{f*100:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, df.run, rotation=20, ha="right")
    ax.set_ylabel("%")
    ax.set_ylim(max(0, (df[["accuracy", "f1_macro"]].min().min() * 100) - 10), 102)
    ax.set_title("Test performance (held-out subjects)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "comparison_accuracy_f1.png", dpi=150)
    plt.close(fig)

    # 2. accuracy vs latency
    fig, ax = plt.subplots(figsize=(7, 4.8))
    sizes = 60 + 900 * df.params_M / max(df.params_M.max(), 1e-9)
    ax.scatter(df.latency_ms, df.accuracy * 100, s=sizes, alpha=0.6, color="#3b6fb6", edgecolor="k")
    for _, r in df.iterrows():
        ax.annotate(f"{r.run}\n{r.params_M:.2f}M", (r.latency_ms, r.accuracy * 100), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("CPU latency per image, batch 1, 1 thread (ms, log)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs. computational cost (bubble area ∝ parameters)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "comparison_accuracy_vs_latency.png", dpi=150)
    plt.close(fig)

    # 3. per-class F1 heat-map
    runs = list(df.run)
    classes = list(next(iter(per_class.values())).keys())
    mat = np.array([[per_class[r][c] for c in classes] for r in runs])
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(classes), 0.9 + 0.5 * len(runs)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=max(0, mat.min() - 0.05), vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(runs)), runs)
    for i in range(len(runs)):
        for j in range(len(classes)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Per-class F1 (test)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out / "comparison_per_class_f1.png", dpi=150)
    plt.close(fig)


def update_readme(readme: Path, table_md: str):
    if not readme.exists():
        return
    text = readme.read_text()
    block = ("<!-- RESULTS_START -->\n" + table_md +
             "\n![Accuracy and F1](results/comparison_accuracy_f1.png)\n"
             "![Accuracy vs latency](results/comparison_accuracy_vs_latency.png)\n"
             "![Per-class F1](results/comparison_per_class_f1.png)\n<!-- RESULTS_END -->")
    new = re.sub(r"<!-- RESULTS_START -->.*?<!-- RESULTS_END -->", lambda _: block, text, flags=re.S)
    readme.write_text(new)
    print(f"[compare] updated results table in {readme}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()
    out = Path(args.results_dir)
    df, per_class = collect(out)
    df.to_csv(out / "comparison.csv", index=False)
    md = to_markdown(df)
    (out / "comparison.md").write_text(md)
    plots(df, per_class, out)
    print(md)
    if not args.no_readme:
        update_readme(Path(args.readme), md)


if __name__ == "__main__":
    main()
