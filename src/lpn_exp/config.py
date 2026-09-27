"""Pydantic schema for experiment 4's YAML config.

Deliberately separate from src/config/schema.py's ExperimentConfig/MethodSpec/
TrainingConfig: those are shaped around a HuggingFace `model.name` and PEFT-style
`target_modules`, neither of which mean anything for LPN (a JAX/Flax model with no
`transformers`/`peft` integration -- see CLAUDE-CODING-SKILL.md). Reusing that schema
here would be actively misleading rather than convenient.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from lpn_exp.lora import LORA_CONDITION_TARGETS

KNOWN_CONDITIONS = ("mean", "gradient_ascent", *LORA_CONDITION_TARGETS)


class TaskGeneratorConfig(BaseModel):
    """Mirrors the pretrained checkpoint's own task_generator block (PATTERN,
    pattern_size=2, num_rows=num_cols=4) so the generated eval tasks match the
    distribution the model was actually trained on.
    """

    num_pairs: int = 4
    num_rows: int = 4
    num_cols: int = 4
    pattern_size: int = 2


class GradientAscentConfig(BaseModel):
    """The paper's own latent-vector search (LPN._get_gradient_ascent_context),
    run as one of the baseline conditions alongside our LoRA condition.
    """

    num_steps: int = 10
    lr: float = 0.1


class LoRAConfig(BaseModel):
    """Our test-time adaptation conditions: freeze the pretrained LPN and fit
    a per-task low-rank update to the MLP Dense kernels of its decoder, its
    encoder, or both (one condition each -- see src/lpn_exp/lora.py's
    LORA_CONDITION_TARGETS) via gradient ascent on the same leave-one-out
    objective the paper's own gradient-ascent condition uses. These
    hyperparameters are shared by all three LoRA-ascent variants, so the
    only thing that differs between them is which weights get adapted.
    """

    rank: int = 4
    scale: float = 1.0
    num_steps: int = 10
    lr: float = 0.1


class Exp4Config(BaseModel):
    experiment: str = "exp4_lpn_pattern2d"
    description: str = ""
    checkpoint_repo: str = "clement-bonnet/lpn-2d"
    checkpoint_name: str = "quiet-thunder-789--checkpoint:v0"
    output_subdir: str = "exp4_lpn_pattern2d"
    seed: int = 42
    num_eval_tasks: int = 96  # matches the checkpoint's own eval block (length: 96)
    # Which conditions to run and compare: "mean" (no adaptation), "gradient_ascent"
    # (the paper's latent search), and ours -- "lora_ascent_decoder",
    # "lora_ascent_encoder", "lora_ascent_encoder_decoder" (LoRA on the MLP weights
    # of that part of the model). Every task is scored under every listed condition
    # so the comparison is apples-to-apples.
    conditions: list[str] = list(KNOWN_CONDITIONS)
    task_generator: TaskGeneratorConfig = TaskGeneratorConfig()
    gradient_ascent: GradientAscentConfig = GradientAscentConfig()
    lora: LoRAConfig = LoRAConfig()

    @field_validator("conditions")
    @classmethod
    def _check_conditions(cls, conditions: list[str]) -> list[str]:
        unknown = [c for c in conditions if c not in KNOWN_CONDITIONS]
        if unknown:
            raise ValueError(f"Unknown condition(s) {unknown}; expected any of {KNOWN_CONDITIONS}")
        return conditions

    @classmethod
    def from_yaml(cls, path: str | Path) -> Exp4Config:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
