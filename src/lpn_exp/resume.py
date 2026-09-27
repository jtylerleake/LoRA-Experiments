"""Resuming experiment 4's runs after a Colab disconnect.

run_exp4.py and tune_exp4_lr.py append one JSON row per finished run to a
metrics file on Google Drive. On restart they read that file back and skip
every run that already has a row with the same fingerprint -- so re-running
the same notebook cell after a reconnect picks up where it left off.

A run's fingerprint hashes everything that determines its result: the
checkpoint, which tasks (generator settings + seed), the condition, and that
condition's own hyperparameters (steps, learning rate, LoRA rank/scale), plus
ALGORITHM_VERSION. Changing any of them changes the fingerprint, so those
runs are redone rather than silently reused. Batch size is deliberately left
out -- it changes how runs are grouped, not what each one computes.

Each task's random key is derived from (seed, task_id) alone, so a resumed
run gets exactly the key it would have had in one uninterrupted pass.

Plain Python only (no jax), so it's unit-testable in the CPU test image.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Bump when a change to the adaptation/scoring code changes what a run
# computes, so rows from the old code are never reused.
# 2: GA-matched LoRA objective, clipping, best-step selection, per-round results.
ALGORITHM_VERSION = 2


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Every complete row of a JSON-lines file ([] if it doesn't exist).

    A disconnect mid-write can leave a partial last line. That line is
    dropped *and cut from the file*, so the next append starts on a fresh
    line instead of gluing a new row onto the fragment. A malformed line
    anywhere else is a real problem, not a disconnect, and raises.
    """
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"{path}: malformed line {i + 1} (not a truncated last line)") from None
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines[:i]) + ("\n" if i else ""))
    return rows


def run_fingerprint(config, condition: str, seed: int, learning_rate: float | None) -> str:
    """Short hash of everything that determines one (condition, task) run's
    result -- see the module docstring. `config` is an Exp4Config; `seed` is
    the task seed (the eval `seed`, or `lr_sweep.seed` for sweep rows).
    """
    settings: dict[str, Any] = {
        "algorithm_version": ALGORITHM_VERSION,
        "checkpoint": [config.checkpoint_repo, config.checkpoint_name],
        "task_generator": config.task_generator.model_dump(),
        "seed": seed,
        "condition": condition,
    }
    if condition != "mean":
        settings["learning_rate"] = learning_rate
    if condition == "gradient_ascent":
        settings["gradient_ascent"] = config.gradient_ascent.model_dump()
    elif condition.startswith("lora_ascent"):
        settings["lora"] = config.lora.model_dump()
    blob = json.dumps(settings, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def completed_runs(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    """{(fingerprint, task_id)} for every row that has a fingerprint (rows
    written before resume existed have none, so they never count as done).
    """
    return {(row["fingerprint"], row["task_id"]) for row in rows if "fingerprint" in row}
