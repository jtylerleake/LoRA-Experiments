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
from pydantic import BaseModel


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
    """Our test-time adaptation condition: freeze the pretrained decoder and
    fit a per-task low-rank update to its MLP Dense kernels (see
    src/lpn_exp/lora.py) via gradient ascent on the same leave-one-out
    objective the paper's own gradient-ascent condition uses.
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
    # (the paper's latent search), "lora_ascent" (ours). Every task is scored under
    # every listed condition so the comparison is apples-to-apples.
    conditions: list[str] = ["mean", "gradient_ascent", "lora_ascent"]
    task_generator: TaskGeneratorConfig = TaskGeneratorConfig()
    gradient_ascent: GradientAscentConfig = GradientAscentConfig()
    lora: LoRAConfig = LoRAConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> Exp4Config:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
