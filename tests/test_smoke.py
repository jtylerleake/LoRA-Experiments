"""Smoke tests runnable in the Docker CPU dev image — confirm the package
imports and the experiment configs are well-formed, without touching any
real model weights or datasets.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import lora_experiments
from lora_experiments.config.schema import ExperimentConfig
from lora_experiments.data.loaders import _format_gsm8k, _format_rte, _format_sst2, get_choices
from lora_experiments.eval.metrics import (
    classification_accuracy,
    exact_match_accuracy,
    extract_choice_label,
    extract_final_answer,
    task_accuracy,
)
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


def test_exp1_methods_span_full_ft_and_all_ranks():
    config = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_1_rank_ablation.yaml")
    assert config.tasks == ["sst2", "rte", "gsm8k"]
    method_types = {m.type for m in config.methods}
    assert method_types == {"full_ft", "lora"}
    ranks = sorted(m.rank for m in config.methods if m.type == "lora")
    assert ranks == [1, 2, 4, 8, 16, 64]


def test_extract_final_answer_parses_gsm8k_format():
    text = "First we add 2 and 2 to get 4.\n#### 4"
    assert extract_final_answer(text) == "4"


def test_extract_final_answer_handles_commas_and_missing_marker():
    assert extract_final_answer("The total is #### 1,024") == "1024"
    assert extract_final_answer("no marker here") is None


def test_exact_match_accuracy():
    predictions = ["reasoning...\n#### 4", "reasoning...\n#### 5", "no answer given"]
    references = ["#### 4", "#### 6", "#### 7"]
    assert exact_match_accuracy(predictions, references) == pytest.approx(1 / 3)


def test_format_gsm8k_keeps_final_answer_in_completion():
    example = {"question": "What is 2+2?", "answer": "2+2=4\n#### 4"}
    formatted = _format_gsm8k(example)
    assert "2+2?" in formatted["prompt"]
    assert extract_final_answer(formatted["completion"]) == "4"


def test_extract_choice_label_picks_earliest_match():
    choices = ["entailment", "not_entailment"]
    assert extract_choice_label("The answer is not_entailment.", choices) == "not_entailment"
    assert extract_choice_label("entailment seems right, not not_entailment", choices) == "entailment"
    assert extract_choice_label("unrelated text", choices) is None


def test_classification_accuracy():
    predictions = [" positive", " negative", " I think positive"]
    references = [" positive", " positive", " positive"]
    assert classification_accuracy(predictions, references, ["positive", "negative"]) == pytest.approx(2 / 3)


def test_task_accuracy_dispatches_on_choices():
    assert task_accuracy([" positive"], [" positive"], ["positive", "negative"]) == 1.0
    assert task_accuracy(["#### 4"], ["#### 4"], None) == 1.0


def test_format_sst2_and_rte_produce_valid_labels():
    sst2 = _format_sst2({"sentence": "A great film.", "label": 1})
    assert extract_choice_label(sst2["completion"], get_choices("sst2")) == "positive"

    rte = _format_rte({"sentence1": "A cat sat.", "sentence2": "An animal sat.", "label": 0})
    assert extract_choice_label(rte["completion"], get_choices("rte")) == "entailment"
