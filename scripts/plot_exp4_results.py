"""Generate experiment 4's results table and condition-comparison plot from
a synced metrics.jsonl (see src/config/exp4_lpn_pattern2d.yaml).

  results_table.csv / results_table.md — per condition, exact-match
  accuracy and pixel correctness pooled over two sets of rounds: "All
  rounds" (the paper's protocol) and "Clean-only rounds" (dropping rounds
  whose held-out pair duplicates a context pair -- see
  pattern2d_tasks.clean_round_mask), each with a 95% bootstrap CI that
  resamples whole tasks (rounds within a task aren't independent). Also
  printed.

  condition_comparison.png — mean accuracy and pixel-correctness (bars, with
  95% CI whiskers across tasks) for each condition run: `mean` (no test-time
  adaptation), `gradient_ascent` (the paper's own latent-vector search), and
  the three LoRA-ascent variants (ours -- a per-task LoRA adapter on the
  MLP weights of the decoder, the encoder, or both, see src/lpn_exp/). This
  is the actual point of the experiment: does adapting the model's weights
  per task beat searching its latent, which part of the model is worth
  adapting, and does any of it beat doing nothing.

Colors follow the same validated palette as scripts/plot_results.py: `mean`
(the no-adaptation baseline) gets the neutral/muted slot, the four
adaptation conditions get categorical hues 1-4, consistent with how this
repo already assigns categorical color by identity, not by value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CONDITION_ORDER = [
    "mean",
    "gradient_ascent",
    "lora_ascent_decoder",
    "lora_ascent_encoder",
    "lora_ascent_encoder_decoder",
]
CONDITION_LABELS = {
    "mean": "mean\n(no adaptation)",
    "gradient_ascent": "gradient ascent\n(latent search)",
    "lora_ascent_decoder": "LoRA ascent\n(decoder)",
    "lora_ascent_encoder": "LoRA ascent\n(encoder)",
    "lora_ascent_encoder_decoder": "LoRA ascent\n(encoder + decoder)",
}
CONDITION_COLORS = {
    "mean": "#898781",
    "gradient_ascent": "#eb6834",
    "lora_ascent_decoder": "#2a78d6",
    "lora_ascent_encoder": "#1baf7a",
    "lora_ascent_encoder_decoder": "#eda100",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

ROUND_COLUMNS = ["round_accuracy", "round_pixel_correctness", "round_is_clean"]
SUBSETS = [("All rounds", False), ("Clean-only rounds", True)]


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
    """The rows to report: for each condition, only rows from its latest
    settings (fingerprint), one per task (the latest if a run was repeated).
    A partial last line from a disconnect mid-write is skipped.
    """
    records = []
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print(f"Note: skipped a partial last line in {path} (interrupted write).")
    if not records:
        raise ValueError(f"{path} has no records to plot.")
    df = pd.DataFrame(records)
    missing = [c for c in [*ROUND_COLUMNS, "fingerprint"] if c not in df.columns or df[c].isna().any()]
    if missing:
        raise ValueError(
            f"{path} has rows without {missing} -- written by an older version of run_exp4.py "
            "(before per-round results, the GA-matched LoRA objective, and resume). Move that file "
            "aside and re-run rather than mixing the two."
        )
    # A condition re-run with new settings (e.g. a new tuned lr) may be only
    # partly redone; report its latest settings only, never a mix.
    latest = df.groupby("condition")["fingerprint"].last()
    current = df[df["fingerprint"] == df["condition"].map(latest)]
    if len(current) < len(df):
        print(f"Note: ignored {len(df) - len(current)} rows from conditions' earlier settings.")
    # Keep one row per (task, condition) -- repeats would count as extra tasks.
    deduped = current.drop_duplicates(subset=["task_id", "condition"], keep="last")
    if len(deduped) < len(current):
        print(f"Note: dropped {len(current) - len(deduped)} duplicate (task_id, condition) rows, kept the latest.")
    return deduped.reset_index(drop=True)


def _explode_rounds(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, condition, round)."""
    rounds = df[["task_id", "condition", *ROUND_COLUMNS]].explode(ROUND_COLUMNS)
    return rounds.astype({"round_accuracy": float, "round_pixel_correctness": float, "round_is_clean": bool})


def _pooled_with_task_bootstrap_ci(rounds: pd.DataFrame, metric: str, num_resamples: int, rng) -> tuple:
    """Mean of `metric` pooled over rounds, with a 95% CI from resampling
    whole tasks with replacement.
    """
    per_task = rounds.groupby("task_id")[metric].agg(["sum", "count"])
    sums, counts = per_task["sum"].to_numpy(), per_task["count"].to_numpy()
    idx = rng.integers(0, len(sums), size=(num_resamples, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return sums.sum() / counts.sum(), *np.percentile(boot, [2.5, 97.5])


def build_results_table(df: pd.DataFrame, num_resamples: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per condition x {All rounds, Clean-only rounds}: pooled accuracy and
    pixel correctness with task-bootstrap 95% CIs, and round/task counts.
    """
    rng = np.random.default_rng(seed)
    rounds = _explode_rounds(df)
    present = [c for c in CONDITION_ORDER if c in set(rounds["condition"])]
    table = []
    for condition in present:
        for subset, clean_only in SUBSETS:
            sub = rounds[rounds["condition"] == condition]
            if clean_only:
                sub = sub[sub["round_is_clean"]]
            row = {"condition": condition, "rounds": subset, "num_rounds": len(sub), "num_tasks": sub["task_id"].nunique()}
            for metric, name in [("round_accuracy", "accuracy"), ("round_pixel_correctness", "pixel_correctness")]:
                if len(sub):
                    mean, low, high = _pooled_with_task_bootstrap_ci(sub, metric, num_resamples, rng)
                else:
                    mean = low = high = float("nan")
                row.update({name: mean, f"{name}_ci_low": low, f"{name}_ci_high": high})
            table.append(row)
    return pd.DataFrame(table)


def results_table_markdown(table: pd.DataFrame) -> str:
    def fmt(row, name):
        return f"{row[name]:.3f} [{row[name + '_ci_low']:.3f}, {row[name + '_ci_high']:.3f}]"

    lines = [
        "| Condition | Rounds | Accuracy [95% CI] | Pixel correctness [95% CI] | # rounds | # tasks |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['condition']} | {row['rounds']} | {fmt(row, 'accuracy')} | "
            f"{fmt(row, 'pixel_correctness')} | {row['num_rounds']} | {row['num_tasks']} |"
        )
    return "\n".join(lines) + "\n"


def plot_condition_comparison(df: pd.DataFrame, out_path: Path) -> None:
    present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

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

    table = build_results_table(df)
    table.to_csv(out_dir / "results_table.csv", index=False)
    markdown = results_table_markdown(table)
    (out_dir / "results_table.md").write_text(markdown, encoding="utf-8")
    print(markdown)

    plot_condition_comparison(df, out_dir / "condition_comparison.png")

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
