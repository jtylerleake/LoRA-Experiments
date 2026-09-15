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


class ExperimentConfig(BaseModel):
    experiment: str
    description: str = ""
    model: ModelSpec
    methods: list[MethodSpec]
    dataset: str
    output_subdir: str
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
