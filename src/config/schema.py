"""Pydantic schema for experiment YAML configs.

Every experiment config (see the *.yaml files in this directory) is loaded
through ExperimentConfig so run_experiment.py has one validated shape to work
with regardless of which experiment it's running.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class ModelSpec(BaseModel):
    name: str  # key into models.yaml


class MethodSpec(BaseModel):
    type: Literal["lora", "full_ft", "adapters", "prefix_tuning"]
    # Method-specific knobs. Only the fields relevant to `type` are used;
    # left loose (dict-like via extra="allow") since each experiment sweeps
    # a different subset (rank, target_modules, etc.).
    rank: int | None = None
    alpha: int | None = None
    target_modules: list[str] | None = None
    num_virtual_tokens: int | None = None
    # Overrides TrainingConfig.learning_rate for this method only — full
    # fine-tuning typically needs a much smaller LR than LoRA to stay stable.
    learning_rate: float | None = None


class TrainingConfig(BaseModel):
    epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2.0e-4
    max_seq_length: int = 512
    max_new_tokens: int = 256
    # Generation-based eval (exact-match on GSM8K's final answer) is much
    # slower than training loss, so it runs on a subset of the test split.
    eval_samples: int = 200
    # Caps the training split too — used by "_mini" experiment configs to
    # smoke-test the full pipeline (training, metrics, plotting) in minutes
    # instead of hours. None means use the full training split.
    max_train_samples: int | None = None


class ExperimentConfig(BaseModel):
    experiment: str
    description: str = ""
    model: ModelSpec
    methods: list[MethodSpec]
    # Every method in `methods` is run against every task here (a full
    # cross-product) — a single-task experiment just lists one task.
    tasks: list[str]
    output_subdir: str
    seed: int = 42
    training: TrainingConfig = TrainingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
