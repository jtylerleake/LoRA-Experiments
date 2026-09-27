"""Picks each adaptation condition's learning rate from a sweep's results
(scripts/tune_exp4_lr.py writes the rows, run_exp4.py --learning-rates
consumes the picks).

The selection metric is exact-match accuracy pooled over *clean* rounds
(pattern2d_tasks.clean_round_mask) -- the leakage-free measure, so a
learning rate can't win by memorizing a context pair that duplicates the
query. Ties go to pixel correctness on the same rounds, then to the smaller
learning rate. If a sweep somehow has no clean rounds at all, all rounds are
used instead.

Plain Python only (no jax), so it's unit-testable in the CPU test image.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _pooled_scores(rows: list[dict[str, Any]], clean_only: bool) -> tuple[float, float, int]:
    accuracies, pixel_corrects = [], []
    for row in rows:
        for accuracy, pixel_correctness, is_clean in zip(
            row["round_accuracy"], row["round_pixel_correctness"], row["round_is_clean"]
        ):
            if is_clean or not clean_only:
                accuracies.append(accuracy)
                pixel_corrects.append(pixel_correctness)
    if not accuracies:
        return float("nan"), float("nan"), 0
    return sum(accuracies) / len(accuracies), sum(pixel_corrects) / len(pixel_corrects), len(accuracies)


def summarize_sweep(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary per (condition, learning_rate): pooled accuracy and pixel
    correctness over clean rounds and over all rounds, with round counts.
    """
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["learning_rate"])].append(row)
    summaries = []
    for (condition, learning_rate), group in sorted(groups.items()):
        clean_acc, clean_pix, clean_n = _pooled_scores(group, clean_only=True)
        all_acc, all_pix, all_n = _pooled_scores(group, clean_only=False)
        summaries.append(
            {
                "condition": condition,
                "learning_rate": learning_rate,
                "clean_accuracy": clean_acc,
                "clean_pixel_correctness": clean_pix,
                "clean_rounds": clean_n,
                "all_accuracy": all_acc,
                "all_pixel_correctness": all_pix,
                "all_rounds": all_n,
            }
        )
    return summaries


def select_best_learning_rates(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    """{condition: best learning rate} -- see the module docstring for the rule."""
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summarize_sweep(rows):
        by_condition[summary["condition"]].append(summary)
    best = {}
    for condition, summaries in by_condition.items():
        prefix = "clean" if any(s["clean_rounds"] for s in summaries) else "all"
        best[condition] = max(
            summaries,
            key=lambda s: (s[f"{prefix}_accuracy"], s[f"{prefix}_pixel_correctness"], -s["learning_rate"]),
        )["learning_rate"]
    return best
