"""Aggregate per-experiment metrics into comparison tables and figures."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

ORDER = ["mlp", "mlp_hog", "cnn", "cnn_noaug", "mobilenetv2_frozen", "mobilenetv2"]


def load_results() -> list[dict]:
    out = []
    for name in ORDER:
        f = C.RESULTS_DIR / name / "metrics.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def build_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        t = r["test"]
        rows.append({
            "Model": r["model"],
            "Train acc": r["train"]["accuracy"],
            "Val acc": r["val"]["accuracy"],
            "Test acc": t["accuracy"],
            "Test precision": t["precision_macro"],
            "Test recall": t["recall_macro"],
            "Test F1": t["f1_macro"],
            "Params (M)": r["params"] / 1e6,
            "MMACs": r["macs"] / 1e6,
            "CPU latency (ms, b=1)": r["latency_ms_cpu_b1"] + r.get("hog_ms_per_image", 0.0),
            f"{r['device'].upper()} throughput (img/s)": r["throughput_img_s_device_b64"],
            "Train time (min)": r["train_seconds"] / 60,
            "Epochs": r["epochs_run"],
        })
    return pd.DataFrame(rows)


def per_class_f1(results: list[dict]) -> pd.DataFrame:
    data = {}
    for r in results:
        rep = json.loads((C.RESULTS_DIR / r["experiment"] / "per_class_report.json").read_text())
        data[r["model"]] = [rep[c]["f1-score"] for c in C.CLASS_NAMES]
    return pd.DataFrame(data, index=C.CLASS_NAMES)


def to_markdown(df: pd.DataFrame, floatfmt: dict) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.get(c, "{:.4f}").format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_comparison(df: pd.DataFrame) -> None:
    metrics = ["Test acc", "Test precision", "Test recall", "Test F1"]
    x = np.arange(len(df))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, df[m], w, label=m.replace("Test ", ""))
    ax.set_xticks(x, df["Model"], rotation=20, ha="right")
    ax.set_ylim(max(0, df[metrics].min().min() - 0.1), 1.06)
    ax.set_ylabel("Score (macro avg)")
    ax.set_title("Test-set performance (held-out subjects 8 and 9)")
    ax.legend(ncol=4, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(C.FIG_DIR / "comparison_metrics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df["MMACs"], df["Test acc"], s=df["Params (M)"] * 40 + 30, alpha=0.7)
    for _, r in df.iterrows():
        ax.annotate(r["Model"], (r["MMACs"], r["Test acc"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("MMACs per image (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs compute (marker size ~ parameters)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(C.FIG_DIR / "accuracy_vs_compute.png", dpi=150)
    plt.close(fig)


def main():
    results = load_results()
    if not results:
        print("No results found; run training first.")
        return
    df = build_table(results)
    df.to_csv(C.RESULTS_DIR / "comparison.csv", index=False)
    f1 = per_class_f1(results)
    f1.to_csv(C.RESULTS_DIR / "per_class_f1.csv")
    plot_comparison(df)

    fmt = {"Params (M)": "{:.3f}", "MMACs": "{:.1f}", "CPU latency (ms, b=1)": "{:.2f}",
           "Train time (min)": "{:.1f}"}
    fmt.update({c: "{:.0f}" for c in df.columns if "throughput" in c})
    md = ["# Results", "", "## Overall comparison", "", to_markdown(df, fmt), "",
          "## Per-class test F1", "", to_markdown(f1.reset_index(names="Gesture"), {}), ""]
    (C.RESULTS_DIR / "RESULTS.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
