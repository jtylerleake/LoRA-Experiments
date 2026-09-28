"""Pydantic schema for experiment 4's YAML config.

Deliberately separate from src/config/schema.py's ExperimentConfig/MethodSpec/
TrainingConfig: those are shaped around a HuggingFace `model.name` and PEFT-style
`target_modules`, neither of which mean anything for LPN (a JAX/Flax model with no
`transformers`/`peft` integration -- see CLAUDE-CODING-SKILL.md). Reusing that schema
here would be actively misleading rather than convenient.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, PositiveInt, field_validator, model_validator

from lpn_exp.lora import LORA_CONDITION_TARGETS

# Every condition that runs a test-time optimizer, and so has a learning rate.
ADAPTATION_CONDITIONS = ("gradient_ascent", *LORA_CONDITION_TARGETS)
KNOWN_CONDITIONS = ("mean", *ADAPTATION_CONDITIONS)


# The pretrained checkpoint's grid size (its max_rows/max_cols). Every task
# family must produce exactly this, since the model can't read or emit
# anything bigger and only ever saw full-size grids in training.
CHECKPOINT_GRID_SIZE = 4


class TaskGeneratorConfig(BaseModel):
    """Which tasks to generate (see src/lpn_exp/task_families.py).

    `lpn_pattern` (the default) is lpn's own Pattern-2D generator -- with the
    defaults below it mirrors the checkpoint's own task_generator block
    (PATTERN, pattern_size=2, num_rows=num_cols=4), the distribution the model
    was trained on. The other families are our harder benchmark ladder:
    `pattern` (our pattern generator: any size that fits, sparse patterns via
    `pattern_density`, a per-task `anchor` corner, and repeat-free marker
    positions so every round is clean) and `color_permutation` (random grids
    recolored by a per-task color mapping over `num_colors` colors).
    """

    family: Literal["lpn_pattern", "pattern", "color_permutation"] = "lpn_pattern"
    num_pairs: int = 4
    num_rows: int = 4
    num_cols: int = 4
    pattern_size: int = 2
    pattern_density: float = 1.0
    anchor: Literal["top_left", "random_corner"] = "top_left"
    num_colors: int = 4

    @model_validator(mode="after")
    def _check_family(self) -> TaskGeneratorConfig:
        if (self.num_rows, self.num_cols) != (CHECKPOINT_GRID_SIZE, CHECKPOINT_GRID_SIZE):
            raise ValueError(
                f"Grids must be {CHECKPOINT_GRID_SIZE}x{CHECKPOINT_GRID_SIZE}, the pretrained checkpoint's size; "
                f"got {self.num_rows}x{self.num_cols}"
            )
        if self.num_pairs < 2:
            raise ValueError("num_pairs must be at least 2 (one held out, the rest as context)")
        if self.family == "lpn_pattern" and (self.pattern_density != 1.0 or self.anchor != "top_left"):
            raise ValueError("lpn_pattern supports only pattern_density=1.0 and anchor=top_left; use family=pattern")
        if self.family in ("lpn_pattern", "pattern") and not 1 <= self.pattern_size < min(self.num_rows, self.num_cols):
            raise ValueError(f"pattern_size must be in [1, {min(self.num_rows, self.num_cols) - 1}]")
        if self.family == "pattern":
            if not 0.0 < self.pattern_density <= 1.0:
                raise ValueError("pattern_density must be in (0, 1]")
            positions = (self.num_rows - self.pattern_size + 1) * (self.num_cols - self.pattern_size + 1)
            if self.num_pairs > positions:
                raise ValueError(
                    f"{self.num_pairs} pairs need {self.num_pairs} distinct marker positions, but a "
                    f"{self.pattern_size}x{self.pattern_size} pattern has only {positions} in a "
                    f"{self.num_rows}x{self.num_cols} grid"
                )
        if self.family == "color_permutation" and not 1 <= self.num_colors <= 9:
            raise ValueError("num_colors must be in [1, 9]")
        return self

    def fingerprint_fields(self) -> dict:
        """The fields that determine this family's tasks, for resume
        fingerprints. `lpn_pattern` keeps exactly its original four fields,
        so rows written before the other families existed still match.
        """
        common = {"num_pairs": self.num_pairs, "num_rows": self.num_rows, "num_cols": self.num_cols}
        if self.family == "lpn_pattern":
            return {**common, "pattern_size": self.pattern_size}
        if self.family == "pattern":
            return {
                "family": self.family,
                **common,
                "pattern_size": self.pattern_size,
                "pattern_density": self.pattern_density,
                "anchor": self.anchor,
            }
        return {"family": self.family, **common, "num_colors": self.num_colors}


class GradientAscentConfig(BaseModel):
    """The paper's own latent-vector search (LPN._get_gradient_ascent_context),
    run as one of the baseline conditions alongside our LoRA conditions. Its
    learning rate lives in Exp4Config.learning_rates, like every other
    adaptation condition's.
    """

    num_steps: int = 10


class LoRAConfig(BaseModel):
    """Our test-time adaptation conditions: freeze the pretrained LPN and fit
    a per-task low-rank update to the MLP Dense kernels of its decoder, its
    encoder, or both (one condition each -- see src/lpn_exp/lora.py's
    LORA_CONDITION_TARGETS). They optimize exactly what the paper's
    gradient-ascent condition does -- the decoder log-likelihood of every
    context pair given one shared (mean) latent -- with the same optimizer
    (SGD, gradients clipped to global norm 1.0) and the same best-step
    selection (see src/lpn_exp/test_time_adapt.py), so what gets adapted
    (the latent vs. these weights) is the only difference. These
    hyperparameters are shared by all three variants; learning rates are
    tuned per condition (Exp4Config.learning_rates).
    """

    rank: int = 4
    scale: float = 1.0
    num_steps: int = 10


class LRSweepConfig(BaseModel):
    """Per-condition learning-rate tuning (scripts/tune_exp4_lr.py). Runs on
    its own seed's tasks -- never the evaluation tasks -- so choosing a
    learning rate can't leak the test set into the result.
    """

    seed: int = 1
    num_tasks: int = 64
    learning_rates: list[float] = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


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
    # One learning rate per adaptation condition. The defaults (0.1) are the
    # paper's own gradient-ascent lr; the real runs override them with the
    # sweep's tuned values (run_exp4.py --learning-rates).
    learning_rates: dict[str, float] = {c: 0.1 for c in ADAPTATION_CONDITIONS}
    lr_sweep: LRSweepConfig = LRSweepConfig()
    # Tasks evaluated per jitted call (vmapped over tasks). Only changes speed,
    # not results; a short final batch is padded so every call has one shape.
    batch_size: PositiveInt = 16

    @field_validator("conditions")
    @classmethod
    def _check_conditions(cls, conditions: list[str]) -> list[str]:
        unknown = [c for c in conditions if c not in KNOWN_CONDITIONS]
        if unknown:
            raise ValueError(f"Unknown condition(s) {unknown}; expected any of {KNOWN_CONDITIONS}")
        return conditions

    @field_validator("learning_rates")
    @classmethod
    def _check_learning_rates(cls, learning_rates: dict[str, float]) -> dict[str, float]:
        unknown = [c for c in learning_rates if c not in ADAPTATION_CONDITIONS]
        if unknown:
            raise ValueError(
                f"learning_rates has non-adaptation condition(s) {unknown}; expected any of {ADAPTATION_CONDITIONS}"
            )
        return learning_rates

    @model_validator(mode="after")
    def _check_consistency(self) -> Exp4Config:
        missing = [c for c in self.conditions if c in ADAPTATION_CONDITIONS and c not in self.learning_rates]
        if missing:
            raise ValueError(f"No learning rate for condition(s) {missing}")
        if self.lr_sweep.seed == self.seed:
            raise ValueError("lr_sweep.seed must differ from seed, or tuning would run on the evaluation tasks")
        return self

    def with_learning_rates(self, overrides: dict[str, float]) -> Exp4Config:
        """A validated copy with `overrides` merged into `learning_rates`."""
        data = self.model_dump()
        data["learning_rates"] = {**self.learning_rates, **overrides}
        return type(self).model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Exp4Config:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
