"""Prefix tuning (via PEFT), for the experiment 3 comparison."""
from __future__ import annotations

from config.schema import MethodSpec

_DEFAULT_NUM_VIRTUAL_TOKENS = 20


def build_prefix_tuning_model(base_model, method: MethodSpec):
    """Wrap `base_model` with a peft.PrefixTuningConfig-based adapter."""
    from peft import PrefixTuningConfig, TaskType, get_peft_model

    prefix_config = PrefixTuningConfig(
        num_virtual_tokens=method.num_virtual_tokens or _DEFAULT_NUM_VIRTUAL_TOKENS,
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(base_model, prefix_config)
