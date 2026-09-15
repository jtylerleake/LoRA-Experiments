"""Full fine-tuning baseline, used by experiment 1's rank-ablation baseline
and experiment 3's method comparison.
"""
from __future__ import annotations

from lora_experiments.config.schema import MethodSpec


def prepare_full_finetune(base_model, method: MethodSpec):
    raise NotImplementedError("Full fine-tuning not implemented yet.")
