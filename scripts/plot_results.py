"""Generate the experiment-1 rank-ablation plots from a synced metrics.jsonl.

Run after scripts/sync_outputs.py has pulled a run's metrics.jsonl back from
Drive (or point --metrics at it directly). Produces three plots per
build_instructions_2.md's "Experiment 1 Outputs" spec:

  1. accuracy_vs_trainable_params.png — eval accuracy (y) vs. trainable
     parameters on a log x-axis, one curve per task; full-FT is plotted as a
     separate marker per task since its parameter count dwarfs every LoRA rank.
  2. accuracy_vs_rank.png — eval accuracy (y) vs. LoRA rank (x), one curve
     per task (full-FT has no rank, so it isn't part of this plot).
  3. training_curves_<task>.png (one per task) — training loss vs. step,
     overlaying every rank/method run for that task.

Colors follow a validated categorical/ordinal palette (see the dataviz
skill): task identity uses three fixed categorical hues; LoRA rank uses a
single-hue ordinal ramp (light = low rank, dark = high rank) since rank is
an ordered magnitude, not an arbitrary category; full-FT gets its own fixed
accent since it isn't part of the rank order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TASK_ORDER = ["sst2", "rte", "gsm8k"]
TASK_COLORS = {"sst2": "#2a78d6", "rte": "#eb6834", "gsm8k": "#1baf7a"}  # categorical slots 1-3
RANK_RAMP = {1: "#86b6ef", 2: "#6da7ec", 4: "#3987e5", 8: "#2a78d6", 16: "#1c5cab", 64: "#104281"}
FULL_FT_COLOR = "#eb6834"  # distinct accent, off the rank ramp

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def load_metrics(path: Path) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} has no records to plot.")
    return pd.json_normalize(records, sep="_")


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_title(title, color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)


def plot_accuracy_vs_trainable_params(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    for task in TASK_ORDER:
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        color = TASK_COLORS[task]
        lora = sub[sub["method_type"] == "lora"].sort_values("trainable_params")
        if not lora.empty:
            ax.plot(
                lora["trainable_params"], lora["eval_accuracy"],
                marker="o", markersize=8, linewidth=2, color=color, label=task,
            )
        full_ft = sub[sub["method_type"] == "full_ft"]
        if not full_ft.empty:
            ax.scatter(
                full_ft["trainable_params"], full_ft["eval_accuracy"],
                marker="*", s=180, color=color, edgecolor=SURFACE, linewidth=1, zorder=5,
            )
    ax.set_xscale("log")
    _style_axes(ax, "Performance vs. trainable parameters", "Trainable parameters (log scale)", "Eval accuracy")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy_vs_rank(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    for task in TASK_ORDER:
        sub = df[(df["task"] == task) & (df["method_type"] == "lora")].sort_values("method_rank")
        if sub.empty:
            continue
        ax.plot(
            sub["method_rank"], sub["eval_accuracy"],
            marker="o", markersize=8, linewidth=2, color=TASK_COLORS[task], label=task,
        )
    ax.set_xscale("log", base=2)
    ranks = sorted(RANK_RAMP)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    _style_axes(ax, "Performance vs. LoRA rank", "LoRA rank", "Eval accuracy")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_training_curves(df: pd.DataFrame, out_dir: Path) -> None:
    for task in TASK_ORDER:
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
        plotted = False
        for _, row in sub.sort_values("trainable_params").iterrows():
            curve = row.get("training_curve") or []
            if not curve:
                continue
            steps = [point["step"] for point in curve]
            losses = [point["loss"] for point in curve]
            if row["method_type"] == "full_ft":
                color, label = FULL_FT_COLOR, "full_ft"
            else:
                rank = int(row["method_rank"])
                color, label = RANK_RAMP.get(rank, INK), f"rank {rank}"
            ax.plot(steps, losses, linewidth=2, color=color, label=label)
            plotted = True
        _style_axes(ax, f"Training curves — {task}", "Training step", "Training loss")
        if plotted:
            ax.legend(frameon=False, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out_dir / f"training_curves_{task}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="outputs/exp1_rank_ablation/metrics.jsonl")
    parser.add_argument("--output-dir", default="outputs/exp1_rank_ablation/plots")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run experiment 1 on Colab "
            "(notebooks/exp1_rank_ablation.ipynb), then pull its metrics.jsonl back "
            "locally (scripts/sync_outputs.py) before plotting."
        )

    df = load_metrics(metrics_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_vs_trainable_params(df, out_dir / "accuracy_vs_trainable_params.png")
    plot_accuracy_vs_rank(df, out_dir / "accuracy_vs_rank.png")
    plot_training_curves(df, out_dir)

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
