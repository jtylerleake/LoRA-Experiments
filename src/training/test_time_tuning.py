"""Experiment 4: test-time LoRA tuning for ARC-AGI-style few-shot tasks.

For each task, a fresh LoRA adapter is trained on the task's few-shot
demonstration pairs and then used to answer the held-out query; accuracy is
compared against the unadapted base model on the same query.
"""
from __future__ import annotations

from config.schema import MethodSpec


def adapt_and_evaluate_task(base_model, task, method: MethodSpec):
    raise NotImplementedError("Test-time tuning not implemented yet.")
