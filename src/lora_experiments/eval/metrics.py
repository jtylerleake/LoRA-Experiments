"""Evaluation metrics shared across experiments.

Metrics are written as JSON-lines to the run's output directory so
scripts/sync_outputs.py can pull just this small file back from Drive for
local plotting, without touching checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_metric(output_path: str | Path, record: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
