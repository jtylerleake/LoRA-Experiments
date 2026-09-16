"""LoRA training via Hugging Face PEFT.

PEFT (not the microsoft/LoRA research repo linked in the project brief) is
the runtime library here: it's the actively-maintained, transformers-
integrated implementation, and its `target_modules` argument is exactly
what experiment 2 (matrix-application study) sweeps over. See
docs/colab_workflow.md for why microsoft/LoRA is referenced only as the
algorithm source, not a dependency.
"""
from __future__ import annotations

from lora_experiments.config.schema import MethodSpec


def build_lora_model(base_model, method: MethodSpec):
    """Wrap `base_model` with a peft.LoraConfig-based adapter."""
    from peft import LoraConfig, TaskType, get_peft_model

    if method.rank is None or not method.target_modules:
        raise ValueError("LoRA method requires 'rank' and 'target_modules' in the config.")

    lora_config = LoraConfig(
        r=method.rank,
        lora_alpha=method.alpha or method.rank * 2,
        target_modules=method.target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(base_model, lora_config)
