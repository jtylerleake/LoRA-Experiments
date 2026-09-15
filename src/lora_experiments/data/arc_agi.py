"""ARC-AGI-style few-shot task loader for Experiment 4 (test-time tuning).

Not implemented yet. Each task is expected to yield a small set of
input/output demonstration pairs plus one held-out query, which
training/test_time_tuning.py uses to adapt and then score the model.
"""
from __future__ import annotations


def load_arc_agi_tasks(name: str):
    raise NotImplementedError(f"ARC-AGI task loader for {name!r} not implemented yet.")
