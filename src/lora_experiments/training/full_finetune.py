"""Full fine-tuning baseline, used by experiment 1's rank-ablation baseline
and experiment 3's method comparison.
"""
from __future__ import annotations

from lora_experiments.config.schema import MethodSpec


def prepare_full_finetune(base_model, method: MethodSpec):
    """Make every parameter of `base_model` trainable.

    No architectural change is needed for full fine-tuning; this just
    guards against a base model loaded with frozen weights (e.g. leftover
    `requires_grad=False` from a prior PEFT-wrapped load).
    """
    for param in base_model.parameters():
        param.requires_grad_(True)
    return base_model
