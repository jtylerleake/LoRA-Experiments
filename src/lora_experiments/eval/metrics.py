"""Evaluation metrics shared across experiments.

Metrics are written as JSON-lines to the run's output directory so
scripts/sync_outputs.py can pull just this small file back from Drive for
local plotting, without touching checkpoints.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def write_metric(output_path: str | Path, record: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def extract_final_answer(text: str) -> str | None:
    """Pull the number after GSM8K's standard "#### <number>" marker.

    Returns None if the marker isn't present (e.g. a model that never
    finishes its reasoning within max_new_tokens) — treated as incorrect by
    exact_match_accuracy rather than crashing the eval loop.
    """
    match = _GSM8K_ANSWER_RE.search(text)
    if not match:
        return None
    return match.group(1).replace(",", "").strip()


def exact_match_accuracy(predictions: list[str], references: list[str]) -> float:
    """Fraction of predictions whose final "#### <number>" matches the reference."""
    if not predictions:
        return 0.0
    correct = sum(
        1
        for pred, ref in zip(predictions, references)
        if (p := extract_final_answer(pred)) is not None and p == extract_final_answer(ref)
    )
    return correct / len(predictions)


def extract_choice_label(text: str, choices: list[str]) -> str | None:
    """Return whichever `choices` label appears earliest (case-insensitive) in `text`.

    Used for classification-style tasks (SST-2, RTE) where the model
    generates free text but the label word is what's scored.
    """
    lowered = text.lower()
    best_label, best_pos = None, None
    for choice in choices:
        pos = lowered.find(choice.lower())
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_label, best_pos = choice, pos
    return best_label


def classification_accuracy(predictions: list[str], references: list[str], choices: list[str]) -> float:
    """Fraction of predictions whose extracted label matches the reference's label."""
    if not predictions:
        return 0.0
    correct = sum(
        1
        for pred, ref in zip(predictions, references)
        if extract_choice_label(pred, choices) == extract_choice_label(ref, choices)
    )
    return correct / len(predictions)


def task_accuracy(predictions: list[str], references: list[str], choices: list[str] | None) -> float:
    """Dispatch to the right scorer: numeric exact-match (choices=None, e.g. GSM8K) or classification."""
    if choices is None:
        return exact_match_accuracy(predictions, references)
    return classification_accuracy(predictions, references, choices)
