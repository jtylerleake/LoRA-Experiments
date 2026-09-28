"""Generate experiment 4's results table and condition-comparison plot from
a synced metrics.jsonl (see src/config/exp4_lpn_pattern2d.yaml), for one
config's output folder. (scripts/plot_exp4_ladder.py reports across the
benchmark ladder's levels.)

  results_table.{csv,md,png} — per condition, exact-match accuracy and pixel
  correctness pooled over two sets of rounds: "All rounds" (the paper's
  protocol) and "Clean-only rounds" (dropping rounds whose held-out pair
  duplicates a context pair -- see pattern2d_tasks.clean_round_mask), each
  with a 95% bootstrap CI that resamples whole tasks (rounds within a task
  aren't independent). Also printed.

  condition_comparison.{png,pdf} — clean-round accuracy and pixel
  correctness per condition, with the same task-bootstrap CIs: `mean` (no
  test-time adaptation), `gradient_ascent` (the paper's own latent-vector
  search), and the three LoRA-ascent variants (ours -- a per-task LoRA
  adapter on the MLP weights of the decoder, the encoder, or both, see
  src/lpn_exp/).

Fonts, colors and labels come from exp4_plot_style.py (Aptos Display; pass
the font folder as --font-dir).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import exp4_plot_style as style
from exp4_plot_style import CONDITION_ORDER

ROUND_COLUMNS = ["round_accuracy", "round_pixel_correctness", "round_is_clean"]
SUBSETS = [("All rounds", False), ("Clean-only rounds", True)]


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


def _table_rows(table: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    def fmt(row, name):
        return f"{100 * row[name]:.1f} [{100 * row[name + '_ci_low']:.1f}, {100 * row[name + '_ci_high']:.1f}]"

    header = ["Condition", "Rounds", "Accuracy (%)", "Pixel correctness (%)", "Rounds (n)", "Tasks (n)"]
    rows = [
        [style.CONDITION_LABELS.get(r["condition"], r["condition"]), r["rounds"].replace("-only", ""),
         fmt(r, "accuracy"), fmt(r, "pixel_correctness"), str(r["num_rounds"]), str(r["num_tasks"])]
        for _, r in table.iterrows()
    ]
    for i in range(len(rows) - 1, 0, -1):  # condition name only on its first row
        if rows[i][0] == rows[i - 1][0]:
            rows[i][0] = ""
    return header, rows


def results_table_markdown(table: pd.DataFrame) -> str:
    header, rows = _table_rows(table)
    lines = [" | ".join(header).join(["| ", " |"]), "|:---|:---|---:|---:|---:|---:|"]
    lines += [" | ".join(r).join(["| ", " |"]) for r in rows]
    return "\n".join(lines) + "\n"


def plot_condition_comparison(table: pd.DataFrame, out_dir: Path) -> None:
    """Clean-round accuracy and pixel correctness per condition, as columns
    with task-bootstrap 95% CI whiskers.
    """
    clean = table[table["rounds"] == "Clean-only rounds"].set_index("condition")
    present = [c for c in CONDITION_ORDER if c in clean.index]
    x = np.arange(len(present))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for ax, (metric, panel_title) in zip(axes, [("accuracy", "Exact-match accuracy"),
                                                ("pixel_correctness", "Pixel correctness")]):
        values = clean.loc[present, metric].to_numpy(dtype=float)
        err = np.vstack([values - clean.loc[present, f"{metric}_ci_low"],
                         clean.loc[present, f"{metric}_ci_high"] - values])
        ax.bar(x, values, width=0.36, color=[style.CONDITION_COLORS[c] for c in present], zorder=2)
        ax.errorbar(x, values, yerr=err, fmt="none", ecolor=style.INK_SECONDARY, elinewidth=1.2, capsize=0, zorder=3)
        ax.set_title(panel_title)
        ax.set_xticks(x, [style.CONDITION_LABELS[c].replace(" (", "\n(") for c in present])
        ax.set_ylim(0, 1.04)
        style.percent_axis(ax)
    height = fig.get_figheight()
    fig.text(0.0, 1 - 0.05 / height, "Test-time adaptation results", fontsize=14, fontweight="bold",
             color=style.INK, va="top")
    fig.text(0.0, 1 - 0.39 / height, "Clean held-out pairs, pooled over rounds. Whiskers: 95% bootstrap CI over tasks.",
             fontsize=10, color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(top=1 - 1.15 / height, wspace=0.15)
    style.save_figure(fig, out_dir, "condition_comparison")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="outputs/exp4_lpn_pattern2d/metrics.jsonl")
    parser.add_argument("--output-dir", default="outputs/exp4_lpn_pattern2d/plots")
    parser.add_argument("--font-dir", required=True, help="Folder with the Aptos Display .ttf files.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run experiment 4 on Colab "
            "(notebooks/exp4_test_time_tuning.ipynb) first."
        )

    style.setup(args.font_dir)
    df = load_metrics(metrics_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_results_table(df)
    table.to_csv(out_dir / "results_table.csv", index=False)
    markdown = results_table_markdown(table)
    (out_dir / "results_table.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    header, rows = _table_rows(table)
    fig = style.render_table_figure(rows, header, "Test-time adaptation results",
                                    "Brackets give 95% bootstrap CIs over tasks. Clean rounds exclude held-out pairs "
                                    "that duplicate a context pair.", [False, False, True, True, True, True])
    fig.savefig(out_dir / "results_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot_condition_comparison(table, out_dir)

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
