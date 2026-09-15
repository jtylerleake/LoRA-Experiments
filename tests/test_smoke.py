"""Smoke tests runnable in the Docker CPU dev image — confirm the package
imports and the experiment configs are well-formed, without touching any
real model weights or datasets.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import lora_experiments
from lora_experiments.config.schema import ExperimentConfig
from lora_experiments.models.registry import get_model_spec

CONFIG_DIR = Path(__file__).resolve().parent.parent / "src" / "lora_experiments" / "config"
EXPERIMENT_CONFIGS = sorted(CONFIG_DIR.glob("experiment_*.yaml"))


def test_package_imports():
    assert lora_experiments.__version__


@pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
def test_experiment_config_loads(config_path: Path):
    config = ExperimentConfig.from_yaml(config_path)
    assert config.methods
    # every referenced model must exist in the registry
    get_model_spec(config.model.name)
