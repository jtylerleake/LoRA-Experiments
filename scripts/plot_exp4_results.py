"""Generate experiment 4's condition-comparison plot from a synced metrics.jsonl
(see src/config/exp4_lpn_pattern2d.yaml).

Produces one plot, built with seaborn:

  condition_comparison.png — mean accuracy and pixel-correctness (bars, with
  95% CI whiskers across tasks) for each condition run: `mean` (no test-time
  adaptation), `gradient_ascent` (the paper's own latent-vector search), and
  `lora_ascent` (ours -- a per-task LoRA adapter on the decoder, see
  src/lpn_exp/). This is the actual point of the experiment: does adapting
  the decoder's weights per task beat searching its latent, and does either
  beat doing nothing.

Colors follow the same validated palette as scripts/plot_results.py: `mean`
(the no-adaptation baseline) gets the neutral/muted slot, the two adaptation
conditions get the two lead categorical hues, consistent with how this repo
already assigns categorical color by identity, not by value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CONDITION_ORDER = ["mean", "gradient_ascent", "lora_ascent"]
CONDITION_LABELS = {
    "mean": "mean\n(no adaptation)",
    "gradient_ascent": "gradient ascent\n(latent search)",
    "lora_ascent": "LoRA ascent\n(decoder search)",
}
CONDITION_COLORS = {"mean": "#898781", "gradient_ascent": "#eb6834", "lora_ascent": "#2a78d6"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def apply_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=1.05,
        rc={
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.family": "sans-serif",
        },
    )


def load_metrics(path: Path) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} has no records to plot.")
    return pd.DataFrame(records)


def plot_condition_comparison(df: pd.DataFrame, out_path: Path) -> None:
    present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    metrics_and_titles = [
        ("accuracy", "Exact-match accuracy"),
        ("pixel_correctness", "Pixel correctness"),
    ]
    for ax, (metric, title) in zip(axes, metrics_and_titles):
        sns.barplot(
            data=df,
            x="condition",
            y=metric,
            order=present,
            hue="condition",
            hue_order=present,
            palette=CONDITION_COLORS,
            legend=False,
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(title, color=INK, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in present])
        ax.set_ylim(0, 1)
        sns.despine(ax=ax)

    fig.suptitle("Experiment 4: test-time adaptation on Pattern-2D", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="outputs/exp4_lpn_pattern2d/metrics.jsonl")
    parser.add_argument("--output-dir", default="outputs/exp4_lpn_pattern2d/plots")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run experiment 4 on Colab "
            "(notebooks/exp4_test_time_tuning.ipynb) first."
        )

    apply_theme()
    df = load_metrics(metrics_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_condition_comparison(df, out_dir / "condition_comparison.png")

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
