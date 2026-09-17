"""Generate the experiment-2 matrix-application-study plots from a synced
metrics.jsonl (see experiment_2_matrix_study.yaml).

Produces two plots, built with seaborn:

  1. heatmap.png — task x LoRA-target-module heatmap. Color is each LoRA
     config's eval accuracy normalized to % of that task's own full
     fine-tuning accuracy (from the config's `full_ft` method), so tasks
     with very different absolute accuracy scales (e.g. SST-2 vs. GSM8K)
     are directly comparable. Diverging color centered at exactly 100%
     (matches full-FT): blue below, red at/above, since the value's job is
     "how does this compare to a baseline," not raw magnitude.
  2. ablation_delta.png — one panel per task, bars showing the accuracy
     change from removing each of W_q/W_k/W_v/W_o one at a time out of the
     full-attention-LoRA (W_q+W_k+W_v+W_o) condition, controlling for
     interactions between matrices the way a marginal (in-isolation)
     contribution plot doesn't. Shared y-axis across panels so the drop
     *magnitude* is honestly comparable across tasks, not just its shape.

Colors follow the same validated palette as plot_results.py (see the
dataviz skill): the heatmap uses the documented diverging pair (blue <->
red, neutral gray at the 100% center); the ablation-delta bars reuse the
categorical slots for W_q/W_k/W_v/W_o so matrix identity reads consistently
across both plots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

TASK_ORDER = ["sst2", "rte", "gsm8k"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# Diverging pair (blue <-> red, neutral gray midpoint) centered at 100% —
# this heatmap's job is "how does this compare to the full-FT baseline,"
# a baseline/delta job, not a plain low->high magnitude job.
DIVERGING_LOW = "#1c5cab"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#e34948"

# Categorical slots 1-4, reused for W_q/W_k/W_v/W_o so matrix identity
# reads consistently between the heatmap's columns and the ablation bars.
REMOVED_LABELS = {"q_proj": "W_q", "k_proj": "W_k", "v_proj": "W_v", "o_proj": "W_o"}
REMOVED_COLORS = {"q_proj": "#2a78d6", "k_proj": "#eb6834", "v_proj": "#1baf7a", "o_proj": "#eda100"}
_ALL_ATTN = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})

_MODULE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_LABELS = {
    ("q_proj",): "W_q",
    ("k_proj",): "W_k",
    ("v_proj",): "W_v",
    ("o_proj",): "W_o",
    ("q_proj", "k_proj"): "W_q+W_k",
    ("q_proj", "v_proj"): "W_q+W_v",
    ("q_proj", "k_proj", "v_proj", "o_proj"): "All attention",
    ("gate_proj", "up_proj", "down_proj"): "MLP only",
    ("k_proj", "v_proj", "o_proj"): "All attn − W_q",
    ("q_proj", "v_proj", "o_proj"): "All attn − W_k",
    ("q_proj", "k_proj", "o_proj"): "All attn − W_v",
    ("q_proj", "k_proj", "v_proj"): "All attn − W_o",
}
COLUMN_ORDER = [
    "W_q", "W_k", "W_v", "W_o",
    "W_q+W_k", "W_q+W_v",
    "All attention", "MLP only",
    "All attn − W_q", "All attn − W_k", "All attn − W_v", "All attn − W_o",
]


def canon_modules(target_modules) -> tuple[str, ...]:
    modules = set(target_modules)
    return tuple(m for m in _MODULE_ORDER if m in modules)


def matrix_label(target_modules) -> str:
    key = canon_modules(target_modules)
    return _LABELS.get(key, "+".join(key))


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


def build_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    full_ft_acc = df[df["method_type"] == "full_ft"].set_index("task")["eval_accuracy"]

    lora = df[df["method_type"] == "lora"].copy()
    lora["matrix"] = lora["method_target_modules"].apply(matrix_label)

    def normalize(row):
        baseline = full_ft_acc.get(row["task"])
        if not baseline:
            return float("nan")
        return 100.0 * row["eval_accuracy"] / baseline

    lora["normalized"] = lora.apply(normalize, axis=1)
    pivot = lora.pivot_table(index="task", columns="matrix", values="normalized", aggfunc="mean")
    row_order = [t for t in TASK_ORDER if t in pivot.index]
    col_order = [c for c in COLUMN_ORDER if c in pivot.columns]
    return pivot.reindex(index=row_order, columns=col_order)


def _annotation_text_colors(pivot: pd.DataFrame, norm, cmap) -> list[str]:
    colors = []
    for value in pivot.to_numpy().flatten():
        if pd.isna(value):
            continue
        r, g, b, _ = cmap(norm(value))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        colors.append(INK if luminance > 0.55 else "#ffffff")
    return colors


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    pivot = build_heatmap_data(df)
    if pivot.empty:
        return

    values = pivot.to_numpy()
    finite = values[~pd.isna(values)]
    vmin = min(60.0, float(finite.min()) - 5) if finite.size else 60.0
    vmax = max(140.0, float(finite.max()) + 5) if finite.size else 140.0
    norm = TwoSlopeNorm(vcenter=100.0, vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("perf_vs_full_ft", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH])

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(pivot.columns) + 2), 3.8))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        norm=norm,
        mask=pivot.isna(),
        annot=True,
        fmt=".0f",
        linewidths=1.5,
        linecolor=SURFACE,
        cbar_kws={"label": "% of full fine-tuning accuracy"},
    )
    for text, color in zip(ax.texts, _annotation_text_colors(pivot, norm, cmap)):
        text.set_color(color)

    ax.set_title("Matrix-application study: performance vs. full fine-tuning", color=INK, fontweight="bold")
    ax.set_xlabel("LoRA target modules")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_ablation_delta_data(df: pd.DataFrame) -> pd.DataFrame:
    lora = df[df["method_type"] == "lora"].copy()
    lora["module_set"] = lora["method_target_modules"].apply(lambda m: frozenset(m))

    rows = []
    for task, group in lora.groupby("task"):
        full_attn = group[group["module_set"] == _ALL_ATTN]
        if full_attn.empty:
            continue
        baseline_acc = float(full_attn["eval_accuracy"].iloc[0])
        for removed, label in REMOVED_LABELS.items():
            remaining = _ALL_ATTN - {removed}
            match = group[group["module_set"] == remaining]
            if match.empty:
                continue
            delta = (float(match["eval_accuracy"].iloc[0]) - baseline_acc) * 100
            rows.append({"task": task, "removed": label, "delta": delta})
    return pd.DataFrame(rows)


def plot_ablation_delta(df: pd.DataFrame, out_path: Path) -> None:
    data = build_ablation_delta_data(df)
    if data.empty:
        return

    present_tasks = [t for t in TASK_ORDER if t in data["task"].unique()]
    removed_order = ["W_q", "W_k", "W_v", "W_o"]
    palette = {REMOVED_LABELS[k]: v for k, v in REMOVED_COLORS.items()}

    fig, axes = plt.subplots(1, len(present_tasks), figsize=(4.3 * len(present_tasks), 4.5), sharey=True)
    axes = [axes] if len(present_tasks) == 1 else list(axes)

    for i, (ax, task) in enumerate(zip(axes, present_tasks)):
        sub = data[data["task"] == task]
        sns.barplot(
            data=sub,
            x="removed",
            y="delta",
            order=removed_order,
            hue="removed",
            hue_order=removed_order,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title(task, color=INK, fontweight="bold")
        ax.set_xlabel("Matrix removed")
        ax.set_ylabel("Δ accuracy vs. full attention (pts)" if i == 0 else "")
        sns.despine(ax=ax)

    fig.suptitle("Ablation delta from full-attention LoRA", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="outputs/exp2_matrix_study/metrics.jsonl")
    parser.add_argument("--output-dir", default="outputs/exp2_matrix_study/plots")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run experiment 2 on Colab "
            "(notebooks/exp2_matrix_study.ipynb) first."
        )

    apply_theme()
    df = load_metrics(metrics_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap(df, out_dir / "heatmap.png")
    plot_ablation_delta(df, out_dir / "ablation_delta.png")

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
