"""Generate the experiment-1 rank-ablation plots from a synced metrics.jsonl.

Run after scripts/sync_outputs.py has pulled a run's metrics.jsonl back from
Drive (or point --metrics at it directly). Produces three plots per
build_instructions_2.md's "Experiment 1 Outputs" spec, built with seaborn
for a clean, consistent look:

  1. accuracy_vs_trainable_params.png — eval accuracy (y) vs. trainable
     parameters on a log x-axis, one curve per task; full-FT is plotted as a
     separate marker per task since its parameter count dwarfs every LoRA rank.
  2. accuracy_vs_rank.png — eval accuracy (y) vs. LoRA rank (x), one curve
     per task (full-FT has no rank, so it isn't part of this plot).
  3. training_curves.png — a single combined plot overlaying every
     task/rank/method run's training loss. The x-axis is normalized to %
     of training complete rather than raw step count, since different
     tasks in the same experiment can have very different dataset sizes
     (e.g. sst2's ~67k examples vs. rte's ~2.5k) and therefore very
     different total step counts — plotting raw steps would squash a
     short task's whole curve into a sliver at the left of a long task's
     x-range. The y-axis (loss) is log-scaled since magnitudes can differ
     substantially across task types (e.g. classification vs. open-ended
     generation). Color encodes method/rank; line style (solid/dashed/
     dotted) encodes task.

Colors follow a validated categorical/ordinal palette (see the dataviz
skill): task identity uses three fixed categorical hues; LoRA rank uses a
single-hue ordinal ramp (light = low rank, dark = high rank) since rank is
an ordered magnitude, not an arbitrary category; full-FT gets its own fixed
accent since it isn't part of the rank order. seaborn draws the marks
(lineplot/scatterplot) and supplies the theme; these palettes are passed in
via `palette=` rather than replaced with seaborn's defaults.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

TASK_ORDER = ["sst2", "rte", "gsm8k"]
TASK_COLORS = {"sst2": "#2a78d6", "rte": "#eb6834", "gsm8k": "#1baf7a"}  # categorical slots 1-3
RANK_RAMP = {1: "#86b6ef", 2: "#6da7ec", 4: "#3987e5", 8: "#2a78d6", 16: "#1c5cab", 64: "#104281"}
FULL_FT_COLOR = "#eb6834"  # distinct accent, off the rank ramp
CURVE_LABEL_ORDER = ["full_ft"] + [f"rank {r}" for r in sorted(RANK_RAMP)]
CURVE_PALETTE = {"full_ft": FULL_FT_COLOR, **{f"rank {r}": c for r, c in RANK_RAMP.items()}}
# (on, off) dash tuples per task for the combined training-curve plot, since
# color there is already spent on method/rank.
TASK_DASHES = {"sst2": (1, 0), "rte": (4, 1.5), "gsm8k": (1, 1)}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

FIGSIZE = (7.5, 5)


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
    return pd.json_normalize(records, sep="_")


def _present(order: list[str], values) -> list[str]:
    values = set(values)
    return [v for v in order if v in values]


def _finish(ax, title: str, xlabel: str, ylabel: str, legend_outside: bool = False) -> None:
    ax.set_title(title, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    if ax.get_legend() is not None:
        if legend_outside:
            # A two-section (hue + style) legend has enough rows that "best"
            # placement ends up overlapping the data instead of avoiding it.
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False, title=None)
        else:
            sns.move_legend(ax, "best", frameon=False, title=None)


def plot_accuracy_vs_trainable_params(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    lora = df[df["method_type"] == "lora"].sort_values("trainable_params")
    if not lora.empty:
        sns.lineplot(
            data=lora,
            x="trainable_params",
            y="eval_accuracy",
            hue="task",
            hue_order=_present(TASK_ORDER, lora["task"]),
            palette=TASK_COLORS,
            marker="o",
            markersize=8,
            linewidth=2,
            errorbar=None,
            ax=ax,
        )

    full_ft = df[df["method_type"] == "full_ft"]
    if not full_ft.empty:
        sns.scatterplot(
            data=full_ft,
            x="trainable_params",
            y="eval_accuracy",
            hue="task",
            hue_order=_present(TASK_ORDER, full_ft["task"]),
            palette=TASK_COLORS,
            marker="*",
            s=280,
            edgecolor=SURFACE,
            linewidth=1,
            legend=False,
            zorder=5,
            ax=ax,
        )

    ax.set_xscale("log")
    _finish(ax, "Performance vs. trainable parameters", "Trainable parameters (log scale)", "Eval accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_vs_rank(df: pd.DataFrame, out_path: Path) -> None:
    lora = df[df["method_type"] == "lora"]
    if lora.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.lineplot(
        data=lora,
        x="method_rank",
        y="eval_accuracy",
        hue="task",
        hue_order=_present(TASK_ORDER, lora["task"]),
        palette=TASK_COLORS,
        marker="o",
        markersize=8,
        linewidth=2,
        errorbar=None,
        ax=ax,
    )
    ax.set_xscale("log", base=2)
    ranks = sorted(lora["method_rank"].unique())
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(int(r)) for r in ranks])
    _finish(ax, "Performance vs. LoRA rank", "LoRA rank", "Eval accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _explode_training_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format {task, run_name, label, step, progress, loss} rows.

    `progress` is `step` divided by that run's own final logged step (its
    last log_history entry, effectively "end of training") — used as the
    x-axis for the combined plot so runs with very different total step
    counts (different dataset sizes, e.g. sst2 vs. rte) overlay on the same
    0-100% scale instead of raw step counts distorting the comparison.
    """
    rows = []
    for _, row in df.iterrows():
        curve = row.get("training_curve") or []
        if not curve:
            continue
        label = "full_ft" if row["method_type"] == "full_ft" else f"rank {int(row['method_rank'])}"
        final_step = curve[-1]["step"] or 1
        for point in curve:
            rows.append(
                {
                    "task": row["task"],
                    "run_name": row["run_name"],
                    "label": label,
                    "step": point["step"],
                    "progress": point["step"] / final_step,
                    "loss": point["loss"],
                }
            )
    return pd.DataFrame(rows)


def plot_training_curves(df: pd.DataFrame, out_path: Path) -> None:
    curves = _explode_training_curves(df)
    if curves.empty:
        return

    present_tasks = _present(TASK_ORDER, curves["task"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=curves,
        x="progress",
        y="loss",
        hue="label",
        hue_order=_present(CURVE_LABEL_ORDER, curves["label"]),
        palette=CURVE_PALETTE,
        style="task",
        style_order=present_tasks,
        dashes={t: TASK_DASHES[t] for t in present_tasks},
        linewidth=2,
        errorbar=None,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    _finish(
        ax,
        "Training curves (all tasks)",
        "Training progress",
        "Training loss (log scale)",
        legend_outside=True,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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

    apply_theme()
    df = load_metrics(metrics_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_vs_trainable_params(df, out_dir / "accuracy_vs_trainable_params.png")
    plot_accuracy_vs_rank(df, out_dir / "accuracy_vs_rank.png")
    plot_training_curves(df, out_dir / "training_curves.png")

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
