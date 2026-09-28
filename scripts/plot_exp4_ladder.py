"""Tables and figures for experiment 4's benchmark ladder (L0-L4).

Reads each level's metrics.jsonl under --output-root (whichever levels and
conditions have results so far -- the pilot has only `mean` and
`gradient_ascent`; full runs add the LoRA conditions) and writes, under
<output-root>/exp4_ladder_report/:

  tables/ladder_accuracy.{csv,md,tex,png}
      Per level and condition: exact-match accuracy and pixel correctness,
      over clean rounds and over all rounds, each with a 95% bootstrap CI
      over tasks, and round counts.
  tables/ladder_adaptation_speed.{csv,md,tex,png}
      Per level and adaptation condition (from the recorded trajectories):
      best-so-far accuracy on clean rounds after 0, 1, 2 and 5 steps and at
      the end, the mean step finally kept, and how many parameters it adapts.
  figures/ladder_accuracy.{png,pdf}, figures/ladder_pixel_correctness.{png,pdf}
      Each metric across the ladder, one line per condition, for clean
      rounds and for all rounds side by side.
  figures/ladder_adaptation_curves.{png,pdf}
      One panel per level: best-so-far accuracy after each adaptation step,
      one line per adaptation condition, with the no-adaptation baseline.
  figures/ladder_objective_curves.{png,pdf}
      One panel per level: the context log-likelihood gained after each
      step (the objective every adaptation condition climbs).

CSV files keep raw fractions; the Markdown, LaTeX and PNG tables are
formatted as percentages. The .csv/.md/.tex tables are the table view that
goes with every figure. All text is set in Aptos Display (see
exp4_plot_style.py); pass the font folder as --font-dir.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import exp4_plot_style as style
from plot_exp4_results import build_results_table, load_metrics

SPEED_STEPS = [0, 1, 2, 5]
NUM_RESAMPLES = 1000


def load_levels(output_root: Path) -> dict[str, pd.DataFrame]:
    """{level code: that level's reportable rows}, for levels with results."""
    levels = {}
    for code, subdir, _ in style.LEVELS:
        path = output_root / subdir / "metrics.jsonl"
        if path.exists():
            levels[code] = load_metrics(path)
        else:
            print(f"{code}: no results yet ({path}) -- skipped.")
    if not levels:
        raise FileNotFoundError(f"No ladder results under {output_root}.")
    return levels


def accuracy_table(levels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (level, condition): clean-round and all-round accuracy
    and pixel correctness with CIs (from plot_exp4_results.build_results_table).
    """
    rows = []
    for code, df in levels.items():
        table = build_results_table(df)
        for condition in [c for c in style.CONDITION_ORDER if c in set(table["condition"])]:
            by_rounds = table[table["condition"] == condition].set_index("rounds")
            row = {"level": code, "task_family": style.LEVEL_NAMES[code], "condition": condition}
            for key, rounds in [("clean", "Clean-only rounds"), ("all", "All rounds")]:
                for metric in ["accuracy", "pixel_correctness"]:
                    for suffix in ["", "_ci_low", "_ci_high"]:
                        row[f"{key}_{metric}{suffix}"] = by_rounds.loc[rounds, f"{metric}{suffix}"]
                row[f"{key}_rounds"] = int(by_rounds.loc[rounds, "num_rounds"])
            rows.append(row)
    return pd.DataFrame(rows)


def _trajectory_rounds(df: pd.DataFrame, condition: str, field: str, clean_only: bool = True) -> pd.DataFrame:
    """Long form of one condition's trajectories: one row per (task, round,
    step) with the value of `field` (e.g. traj_accuracy).
    """
    sub = df[df["condition"] == condition]
    if field not in sub.columns or sub[field].isna().any():
        return pd.DataFrame(columns=["task_id", "round", "step", "value"])
    records = []
    for _, row in sub.iterrows():
        for r, (curve, is_clean) in enumerate(zip(row[field], row["round_is_clean"])):
            if is_clean or not clean_only:
                records.extend((row["task_id"], r, step, float(v)) for step, v in enumerate(curve))
    return pd.DataFrame(records, columns=["task_id", "round", "step", "value"])


def pooled_curve(long: pd.DataFrame, rng, num_resamples: int = NUM_RESAMPLES) -> pd.DataFrame:
    """Per step: the value pooled over rounds, with a 95% CI from resampling
    whole tasks (rounds within a task aren't independent).
    """
    per_task = long.groupby(["step", "task_id"])["value"].agg(["sum", "count"]).reset_index()
    steps = sorted(per_task["step"].unique())
    tasks = sorted(per_task["task_id"].unique())
    # A task missing a step contributes nothing to it (0 rounds), not NaN.
    sums = per_task.pivot(index="task_id", columns="step", values="sum").reindex(index=tasks, columns=steps)
    counts = per_task.pivot(index="task_id", columns="step", values="count").reindex(index=tasks, columns=steps)
    sums, counts = sums.fillna(0).to_numpy(), counts.fillna(0).to_numpy()
    idx = rng.integers(0, len(tasks), size=(num_resamples, len(tasks)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)  # NaN if a resample has no rounds at a step
    low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame({"step": steps, "mean": sums.sum(0) / counts.sum(0), "low": low, "high": high})


def _log_prob_gain(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    long = _trajectory_rounds(df, condition, "traj_context_log_prob")
    if long.empty:
        return long
    start = long[long["step"] == 0].set_index(["task_id", "round"])["value"]
    long["value"] = long["value"] - start.reindex(pd.MultiIndex.from_frame(long[["task_id", "round"]])).to_numpy()
    return long


def adaptation_speed_table(levels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for code, df in levels.items():
        for condition in style.ADAPTATION_CONDITIONS:
            long = _trajectory_rounds(df, condition, "traj_accuracy")
            if long.empty:
                continue
            by_step = long.groupby("step")["value"].mean()
            final_step = int(by_step.index.max())
            sub = df[df["condition"] == condition]
            best_steps = [s for row_steps, clean in zip(sub["best_step"], sub["round_is_clean"])
                          for s, c in zip(row_steps, clean) if c]
            row = {"level": code, "task_family": style.LEVEL_NAMES[code], "condition": condition,
                   "num_adapted_params": int(sub["num_adapted_params"].iloc[0]),
                   "num_steps": final_step}
            row |= {f"accuracy_step_{s}": by_step.get(s, np.nan) for s in SPEED_STEPS}
            row["accuracy_final"] = by_step[final_step]
            row["mean_best_step"] = float(np.mean(best_steps)) if best_steps else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tables


def _pct(value: float) -> str:
    return "–" if pd.isna(value) else f"{100 * value:.1f}"


def _pct_ci(row, prefix: str) -> str:
    return f"{_pct(row[prefix])} [{_pct(row[prefix + '_ci_low'])}, {_pct(row[prefix + '_ci_high'])}]"


def _latex_escape(text: str) -> str:
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                     ("×", r"$\times$"), ("–", "--")]:
        text = text.replace(old, new)
    return text


def write_table(display: pd.DataFrame, raw: pd.DataFrame, out_dir: Path, stem: str, title: str, subtitle: str,
                numeric: list[bool]) -> None:
    """`raw` -> stem.csv (unformatted numbers); `display` -> stem.md,
    stem.tex (booktabs) and stem.png, with a title and a note.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / f"{stem}.csv", index=False)
    header = list(display.columns)
    cells = display.astype(str).values.tolist()
    # Level and task family only on each level's first row: easier to scan.
    for i in range(len(cells) - 1, 0, -1):
        if cells[i][0] == cells[i - 1][0]:
            cells[i][0] = cells[i][1] = ""

    md = [f"**{title}**", "", " | ".join(header).join(["| ", " |"]),
          "|" + "|".join("---:" if n else ":---" for n in numeric) + "|"]
    md += [" | ".join(r).join(["| ", " |"]) for r in cells]
    (out_dir / f"{stem}.md").write_text("\n".join(md + ["", f"*{subtitle}*", ""]), encoding="utf-8")

    # Needs \usepackage{booktabs} and \usepackage{graphicx} (for \resizebox).
    spec = "".join("r" if n else "l" for n in numeric)
    tex = [r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{_latex_escape(title)}. {_latex_escape(subtitle)}}}",
           r"\resizebox{\linewidth}{!}{%", f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
           " & ".join(_latex_escape(h) for h in header) + r" \\", r"\midrule"]
    tex += [" & ".join(_latex_escape(c) for c in r) + r" \\" for r in cells]
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    (out_dir / f"{stem}.tex").write_text("\n".join(tex), encoding="utf-8")

    fig = style.render_table_figure(cells, header, title, subtitle, numeric)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_accuracy_table(table: pd.DataFrame, out_dir: Path) -> None:
    display = pd.DataFrame(
        {
            "Level": table["level"],
            "Task family": table["task_family"],
            "Condition": table["condition"].map(style.CONDITION_LABELS),
            "Accuracy, clean (%)": [_pct_ci(r, "clean_accuracy") for _, r in table.iterrows()],
            "Accuracy, all (%)": [_pct_ci(r, "all_accuracy") for _, r in table.iterrows()],
            "Pixel correctness, clean (%)": [_pct_ci(r, "clean_pixel_correctness") for _, r in table.iterrows()],
            "Pixel correctness, all (%)": [_pct_ci(r, "all_pixel_correctness") for _, r in table.iterrows()],
            "Clean / all rounds": [f"{r['clean_rounds']} / {r['all_rounds']}" for _, r in table.iterrows()],
        }
    )
    write_table(
        display, table, out_dir, "ladder_accuracy",
        "Accuracy across the benchmark ladder",
        "Held-out pairs, pooled over leave-one-out rounds; brackets give 95% bootstrap CIs over tasks. "
        "Clean rounds exclude held-out pairs that duplicate a context pair.",
        [False, False, False, True, True, True, True, True],
    )


def write_speed_table(table: pd.DataFrame, out_dir: Path) -> None:
    if table.empty:
        print("No trajectories recorded yet -- skipped the adaptation-speed table.")
        return
    display = pd.DataFrame(
        {
            "Level": table["level"],
            "Task family": table["task_family"],
            "Condition": table["condition"].map(style.CONDITION_LABELS),
            "Adapted parameters": table["num_adapted_params"].map("{:,}".format),
            **{f"Step {s} (%)": table[f"accuracy_step_{s}"].map(_pct) for s in SPEED_STEPS},
            "Final (%)": table["accuracy_final"].map(_pct),
            "Mean best step": table["mean_best_step"].map(lambda v: "–" if pd.isna(v) else f"{v:.1f}"),
        }
    )
    write_table(
        display, table, out_dir, "ladder_adaptation_speed",
        "How quickly each method adapts",
        "Best-so-far accuracy on clean held-out pairs after each number of adaptation steps "
        "(step 0 is the unadapted model). Final is after the last step.",
        [False, False, False, True, True, True, True, True, True, True],
    )


# ---------------------------------------------------------------- figures


def _level_ticks(codes: list[str]) -> list[str]:
    return [f"{code}\n" + "\n".join(textwrap.wrap(style.LEVEL_NAMES[code], 14)) for code in codes]


def _header_and_legend(fig, title: str, subtitle: str, conditions: list[str], baseline: bool = False) -> float:
    """Title, subtitle and legend stacked at the top of `fig`, spaced in
    inches so they never collide whatever the figure's height. Returns the
    figure-fraction `top` to pass to subplots_adjust (leaving room for panel
    titles).
    """
    height = fig.get_figheight()
    y = 1 - 0.05 / height
    fig.text(0.0, y, title, fontsize=14, fontweight="bold", color=style.INK, ha="left", va="top")
    y -= 0.34 / height
    fig.text(0.0, y, subtitle, fontsize=10, color=style.INK_SECONDARY, ha="left", va="top")
    y -= 0.32 / height
    handles = []
    for condition in conditions:
        line_style = ({"linewidth": 1.2} if baseline and condition == "mean"
                      else {"linewidth": 2.0, "marker": "o", "markersize": 6})
        handles.append(plt.Line2D([], [], color=style.CONDITION_COLORS[condition], **line_style,
                                  label=style.CONDITION_LABELS[condition]))
    ncol = min(len(handles), 3)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, y), ncol=ncol, frameon=False,
               handlelength=2.2, columnspacing=1.8, borderaxespad=0.0)
    y -= (0.26 * int(np.ceil(len(handles) / ncol)) + 0.42) / height
    return y


def plot_ladder_metric(table: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """`metric` ('accuracy' or 'pixel_correctness') across the ladder, one
    line per condition, clean rounds and all rounds side by side.
    """
    codes = [code for code, _, _ in style.LEVELS if code in set(table["level"])]
    conditions = [c for c in style.CONDITION_ORDER if c in set(table["condition"])]
    x = np.arange(len(codes))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), sharey=True)
    for ax, (key, panel_title) in zip(axes, [("clean", "Clean rounds"), ("all", "All rounds")]):
        for i, condition in enumerate(conditions):
            rows = table[table["condition"] == condition].set_index("level").reindex(codes)
            offset = (i - (len(conditions) - 1) / 2) * 0.07
            mean = rows[f"{key}_{metric}"].to_numpy(dtype=float)
            err = np.vstack([mean - rows[f"{key}_{metric}_ci_low"], rows[f"{key}_{metric}_ci_high"] - mean])
            ax.errorbar(x + offset, mean, yerr=err, color=style.CONDITION_COLORS[condition], linewidth=2.0,
                        elinewidth=1.2, capsize=0, marker="o", markersize=6, markeredgecolor=style.SURFACE,
                        markeredgewidth=1.5, zorder=3)
        ax.set_title(panel_title)
        ax.set_xticks(x, _level_ticks(codes))
        ax.set_xlim(-0.5, len(codes) - 0.5)
        style.percent_axis(ax)
    if metric == "accuracy":
        axes[0].set_ylim(0, 1.04)
    else:
        low = np.nanmin(table[[f"{k}_{metric}_ci_low" for k in ("clean", "all")]].to_numpy())
        axes[0].set_ylim(max(0.0, np.floor(low * 20) / 20 - 0.05), 1.01)
    axes[0].set_ylabel("Accuracy" if metric == "accuracy" else "Pixel correctness")
    name = "Exact-match accuracy" if metric == "accuracy" else "Pixel correctness"
    top = _header_and_legend(fig, f"{name} across the benchmark ladder",
                             "Held-out pairs, pooled over rounds. Whiskers: 95% bootstrap CI over tasks.", conditions)
    fig.subplots_adjust(top=top, wspace=0.08)
    style.save_figure(fig, out_dir, f"ladder_{metric}")


def plot_curves(levels: dict[str, pd.DataFrame], kind: str, out_dir: Path, seed: int = 0) -> None:
    """Per-level panels of best-so-far accuracy ('accuracy') or context
    log-likelihood gain ('objective') against adaptation step.
    """
    rng = np.random.default_rng(seed)
    panels = []
    for code, df in levels.items():
        curves = {}
        for condition in style.ADAPTATION_CONDITIONS:
            long = (_trajectory_rounds(df, condition, "traj_accuracy") if kind == "accuracy"
                    else _log_prob_gain(df, condition))
            if not long.empty:
                curves[condition] = pooled_curve(long, rng)
        if curves:
            panels.append((code, df, curves))
    if not panels:
        print(f"No trajectories recorded yet -- skipped the {kind} curves.")
        return
    ncols = min(len(panels), 3)
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.3 * nrows + 1.6), squeeze=False,
                             sharey=(kind == "accuracy"))
    shown = set()
    for ax, (code, df, curves) in zip(axes.flat, panels):
        if kind == "accuracy" and (df["condition"] == "mean").any():
            baseline = build_results_table(df[df["condition"] == "mean"])
            value = baseline.set_index("rounds").loc["Clean-only rounds", "accuracy"]
            ax.axhline(value, color=style.CONDITION_COLORS["mean"], linewidth=1.2, zorder=1)
            shown.add("mean")
        for condition, curve in curves.items():
            color = style.CONDITION_COLORS[condition]
            ax.fill_between(curve["step"], curve["low"], curve["high"], color=color, alpha=0.14, linewidth=0,
                            zorder=2)
            ax.plot(curve["step"], curve["mean"], color=color, linewidth=2.0, zorder=3)
            ax.plot(curve["step"].iloc[-1], curve["mean"].iloc[-1], "o", color=color, markersize=6,
                    markeredgecolor=style.SURFACE, markeredgewidth=1.5, zorder=4)
            shown.add(condition)
        ax.set_title(f"{code} · {style.LEVEL_NAMES[code]}", fontsize=11)
        steps = curves[next(iter(curves))]["step"]
        ax.set_xlim(steps.min(), steps.max())
        ax.set_xticks(range(int(steps.min()), int(steps.max()) + 1, max(1, int(steps.max()) // 5)))
        ax.set_xlabel("Adaptation step")
        if kind == "accuracy":
            ax.set_ylim(0, 1.04)
            style.percent_axis(ax)
        else:
            ax.axhline(0, color=style.GRID, linewidth=1.0, zorder=1)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Best-so-far accuracy" if kind == "accuracy" else "Log-likelihood gain (nats)")
    conditions = [c for c in style.CONDITION_ORDER if c in shown]
    if kind == "accuracy":
        title, subtitle = ("How quickly each method adapts",
                           "Best-so-far accuracy on clean held-out pairs after each step; step 0 is the unadapted "
                           "model. Bands: 95% bootstrap CI over tasks.")
    else:
        title, subtitle = ("Context log-likelihood gained per adaptation step",
                           "The objective every adaptation method climbs, relative to step 0, on clean rounds. "
                           "Bands: 95% bootstrap CI over tasks. Y-axis scales differ by panel.")
    top = _header_and_legend(fig, title, subtitle, conditions, baseline=True)
    fig.subplots_adjust(top=top, hspace=0.55, wspace=0.18)
    style.save_figure(fig, out_dir, f"ladder_{'adaptation' if kind == 'accuracy' else 'objective'}_curves")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True, help="Root holding each level's output folder.")
    parser.add_argument("--font-dir", required=True, help="Folder with the Aptos Display .ttf files.")
    parser.add_argument("--report-dir", default=None, help="Defaults to <output-root>/exp4_ladder_report.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    style.setup(args.font_dir)
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir) if args.report_dir else output_root / "exp4_ladder_report"
    levels = load_levels(output_root)

    accuracy = accuracy_table(levels)
    write_accuracy_table(accuracy, report_dir / "tables")
    write_speed_table(adaptation_speed_table(levels), report_dir / "tables")
    for metric in ["accuracy", "pixel_correctness"]:
        plot_ladder_metric(accuracy, metric, report_dir / "figures")
    for kind in ["accuracy", "objective"]:
        plot_curves(levels, kind, report_dir / "figures")

    print(f"Levels: {', '.join(levels)}. Wrote tables and figures to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
