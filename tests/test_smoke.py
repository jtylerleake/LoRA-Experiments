"""Smoke tests runnable in the Docker CPU dev image — confirm the package
imports and the experiment configs are well-formed, without touching any
real model weights or datasets.
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from transformers import TrainerCallback

from config.schema import ExperimentConfig, MethodSpec
from data.loaders import _format_gsm8k, _format_rte, _format_sst2, get_choices
from eval.metrics import (
    classification_accuracy,
    exact_match_accuracy,
    extract_choice_label,
    extract_final_answer,
    task_accuracy,
)
from models.registry import get_model_spec
from training.adapters import build_adapter_model
from training.common import (
    CompactProgressCallback,
    cap_dataset,
    compute_logging_steps,
    target_modules_slug,
)
from utils.logging_utils import file_logging

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "src" / "config"
EXPERIMENT_CONFIGS = sorted(CONFIG_DIR.glob("experiment_*.yaml"))

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import plot_matrix_study_results
import plot_results


def test_package_imports():
    import data.loaders  # noqa: F401
    import eval.metrics  # noqa: F401
    import models.registry  # noqa: F401
    import training.lora  # noqa: F401
    import utils.logging_utils  # noqa: F401


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


def test_exp1_mini_mirrors_full_structure_but_capped():
    full = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_1_rank_ablation.yaml")
    mini = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_1_rank_ablation_mini.yaml")
    # Same tasks/methods shape (so the plots exercise real multi-task/rank data)...
    assert mini.tasks == full.tasks
    assert [m.type for m in mini.methods] == [m.type for m in full.methods]
    assert [m.rank for m in mini.methods] == [m.rank for m in full.methods]
    # ...but drastically smaller so it finishes fast.
    assert mini.training.max_train_samples is not None
    assert mini.training.max_train_samples < 100
    assert mini.training.epochs == 1
    assert mini.output_subdir != full.output_subdir


def test_exp1a_has_no_full_ft_and_mirrors_exp1_lora_ranks():
    exp1 = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_1_rank_ablation.yaml")
    exp1a = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_1a_rank_ablation.yaml")
    assert exp1a.tasks == exp1.tasks
    assert {m.type for m in exp1a.methods} == {"lora"}
    exp1_ranks = sorted(m.rank for m in exp1.methods if m.type == "lora")
    exp1a_ranks = sorted(m.rank for m in exp1a.methods)
    assert exp1a_ranks == exp1_ranks == [1, 2, 4, 8, 16, 64]
    assert exp1a.output_subdir != exp1.output_subdir


def test_exp1a_mini_mirrors_full_structure_but_capped():
    full = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_1a_rank_ablation.yaml")
    mini = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_1a_rank_ablation_mini.yaml")
    assert mini.tasks == full.tasks
    assert [m.rank for m in mini.methods] == [m.rank for m in full.methods]
    assert {m.type for m in mini.methods} == {"lora"}
    assert mini.training.max_train_samples is not None
    assert mini.training.max_train_samples < 100
    assert mini.training.epochs == 1
    assert mini.output_subdir != full.output_subdir


def test_exp1a_mini_1_5b_mirrors_0_5b_mini_shape_with_bigger_model():
    """Same tiny scale as the 0.5b mini -- only the model changes -- so its
    elapsed_seconds in metrics.jsonl is an apples-to-apples runtime
    comparison for calibrating whether to run the full sweep at 1.5b.
    """
    mini_0_5b = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_1a_rank_ablation_mini.yaml")
    mini_1_5b = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_1a_rank_ablation_mini_1.5b.yaml")
    assert mini_1_5b.model.name == "qwen2.5-1.5b-instruct"
    assert mini_1_5b.tasks == mini_0_5b.tasks
    assert [m.rank for m in mini_1_5b.methods] == [m.rank for m in mini_0_5b.methods]
    assert mini_1_5b.training == mini_0_5b.training
    assert mini_1_5b.output_subdir != mini_0_5b.output_subdir


def test_exp2_full_and_mini_share_target_module_sweep():
    full = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_2_matrix_study.yaml")
    mini = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_2_matrix_study_mini.yaml")
    assert full.tasks == mini.tasks == ["sst2", "rte", "gsm8k"]
    for task in full.tasks:
        get_choices(task)  # raises KeyError if not a registered task

    lora_methods = [m for m in full.methods if m.type == "lora"]
    assert all(m.rank == 8 for m in lora_methods)
    # full-FT baseline (for the heatmap's normalization) plus every
    # individual/paired/grouped/leave-one-out target-module variant the
    # heatmap and ablation-delta plots need.
    assert {m.type for m in full.methods} == {"full_ft", "lora"}
    target_module_sets = {tuple(sorted(m.target_modules)) for m in lora_methods}
    expected = {
        ("q_proj",),
        ("k_proj",),
        ("v_proj",),
        ("o_proj",),
        ("k_proj", "q_proj"),
        ("q_proj", "v_proj"),
        ("k_proj", "o_proj", "q_proj", "v_proj"),  # all attention
        ("down_proj", "gate_proj", "up_proj"),  # MLP only
        ("k_proj", "o_proj", "v_proj"),  # all attn - q
        ("o_proj", "q_proj", "v_proj"),  # all attn - k
        ("k_proj", "o_proj", "q_proj"),  # all attn - v
        ("k_proj", "q_proj", "v_proj"),  # all attn - o
    }
    assert target_module_sets == expected

    assert [m.target_modules for m in mini.methods] == [m.target_modules for m in full.methods]
    assert mini.training.max_train_samples is not None
    assert mini.training.max_train_samples < 100
    assert mini.training.epochs == 1
    assert mini.output_subdir != full.output_subdir


def test_target_modules_slug_is_order_independent_and_abbreviated():
    # order in the YAML shouldn't matter -- same set, same slug
    assert target_modules_slug(["q_proj", "k_proj"]) == target_modules_slug(["k_proj", "q_proj"]) == "qk"
    assert target_modules_slug(["gate_proj", "up_proj", "down_proj"]) == "gateupdown"
    assert target_modules_slug(["o_proj"]) == "o"


def test_exp2_lora_methods_have_distinct_run_names():
    """Regression test: method_run_name() used to key off rank alone
    ("lora_r8"), so experiment 2's 12 same-rank/different-target_modules
    LoRA methods would all collide on one run_dir/train.log and overwrite
    each other. Every method in a config must produce a unique run name.
    """
    from training.common import method_run_name

    full = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_2_matrix_study.yaml")
    names = [method_run_name(m) for m in full.methods]
    assert len(names) == len(set(names)), names


def test_exp3_full_and_mini_share_method_comparison():
    full = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_3_method_comparison.yaml")
    mini = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_3_method_comparison_mini.yaml")
    assert full.tasks == mini.tasks == ["sst2"]
    get_choices("sst2")
    assert [m.type for m in full.methods] == ["full_ft", "adapters", "prefix_tuning", "lora"]
    assert [m.type for m in mini.methods] == [m.type for m in full.methods]
    assert mini.training.max_train_samples is not None
    assert mini.training.max_train_samples < 100
    assert mini.training.epochs == 1
    assert mini.output_subdir != full.output_subdir


def test_build_adapter_model_freezes_base_and_starts_as_identity():
    import torch
    from torch import nn

    class _FakeDecoderLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, hidden_states):
            return (self.linear(hidden_states),)

    class _FakeInnerModel(nn.Module):
        def __init__(self, hidden_size, num_layers):
            super().__init__()
            self.layers = nn.ModuleList(_FakeDecoderLayer(hidden_size) for _ in range(num_layers))

    class _FakeCausalLM(nn.Module):
        def __init__(self, hidden_size, num_layers):
            super().__init__()
            self.model = _FakeInnerModel(hidden_size, num_layers)
            self.config = SimpleNamespace(hidden_size=hidden_size)

        def forward(self, hidden_states):
            for layer in self.model.layers:
                hidden_states = layer(hidden_states)[0]
            return hidden_states

    torch.manual_seed(0)
    fake = _FakeCausalLM(hidden_size=8, num_layers=2)
    x = torch.randn(2, 3, 8)
    before = fake(x)

    model = build_adapter_model(fake, MethodSpec(type="adapters", bottleneck_size=4))

    base_params = [p for n, p in model.named_parameters() if "bottleneck_adapters" not in n]
    adapter_params = list(model.bottleneck_adapters.parameters())
    assert base_params and all(not p.requires_grad for p in base_params)
    assert adapter_params and all(p.requires_grad for p in adapter_params)

    # up_proj is zero-initialized, so the adapter should be a no-op at the
    # start of training.
    after = model(x)
    assert torch.allclose(before, after, atol=1e-6)


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


def test_file_logging_captures_info_messages_to_file(tmp_path):
    log_path = tmp_path / "run.log"
    probe = logging.getLogger("test.file_logging_probe")
    with file_logging(log_path):
        probe.info("hello from inside the context")
    content = log_path.read_text(encoding="utf-8")
    assert "hello from inside the context" in content

    # After the context exits, the file handler is detached again, so
    # further logging shouldn't keep appending to the same run's log.
    probe.info("this should not be appended")
    assert "this should not be appended" not in log_path.read_text(encoding="utf-8")


def test_compact_progress_callback_logs_steps_to_file(tmp_path):
    log_path = tmp_path / "cb.log"
    run_logger = logging.getLogger("test.callback_run")
    callback = CompactProgressCallback("[test] train", run_logger)
    state = SimpleNamespace(max_steps=10, global_step=0)

    with file_logging(log_path):
        callback.on_train_begin(args=None, state=state, control=None)
        state.global_step = 5
        callback.on_log(args=None, state=state, control=None, logs={"loss": 1.2345, "epoch": 0.5})
        callback.on_step_end(args=None, state=state, control=None)
        callback.on_train_end(args=None, state=state, control=None)

    content = log_path.read_text(encoding="utf-8")
    assert "step=5" in content
    assert "1.2345" in content


_TRAINER_CALLBACK_EVENTS = [
    "on_init_end",
    "on_train_begin",
    "on_epoch_begin",
    "on_step_begin",
    "on_substep_end",
    "on_step_end",
    "on_evaluate",
    "on_predict",
    "on_save",
    "on_log",
    "on_prediction_step",
    "on_epoch_end",
    "on_train_end",
]


def test_compact_progress_callback_handles_every_trainer_event(tmp_path):
    """Regression test: a Colab run crashed with
    `AttributeError: 'CompactProgressCallback' object has no attribute
    'on_epoch_begin'` because the class didn't subclass TrainerCallback, so
    it had no no-op default for events besides the four it overrides. The
    isinstance check plus calling every known event is what would have
    caught this before it reached a real Trainer.train() call.
    """
    assert issubclass(CompactProgressCallback, TrainerCallback)

    log_path = tmp_path / "events.log"
    run_logger = logging.getLogger("test.callback_events")
    callback = CompactProgressCallback("[test] train", run_logger)
    state = SimpleNamespace(max_steps=10, global_step=0)

    with file_logging(log_path):
        for event in _TRAINER_CALLBACK_EVENTS:
            # metrics={} is only required by on_predict, but harmless
            # everywhere else since every event accepts **kwargs.
            getattr(callback, event)(args=None, state=state, control=None, metrics={})


def test_cap_dataset_limits_size():
    from datasets import Dataset

    ds = Dataset.from_dict({"x": list(range(100))})
    capped = cap_dataset(ds, 10)
    assert len(capped) == 10
    assert list(capped["x"]) == list(range(10))


def test_cap_dataset_is_noop_when_none_or_already_smaller():
    from datasets import Dataset

    ds = Dataset.from_dict({"x": list(range(5))})
    assert len(cap_dataset(ds, None)) == 5
    assert len(cap_dataset(ds, 100)) == 5


def test_compute_logging_steps_never_exceeds_total_steps():
    """Regression test: a mini run (32 examples, batch 4, 1 epoch -> 8 total
    steps) with a hardcoded logging_steps=10 never logged a single loss
    value, since global_step never reaches a multiple of 10 -- the
    training-curve plot came back completely blank. logging_steps must
    always be <= the run's own total step count.
    """
    mini_steps = compute_logging_steps(num_examples=32, batch_size=4, grad_accum_steps=1, epochs=1)
    assert 1 <= mini_steps <= 8

    full_steps = compute_logging_steps(num_examples=7473, batch_size=8, grad_accum_steps=1, epochs=3)
    total = math.ceil(7473 / 8) * 3
    assert 1 <= full_steps <= total

    # fewer examples than one batch should still yield a valid (>=1) interval
    tiny_steps = compute_logging_steps(num_examples=3, batch_size=4, grad_accum_steps=1, epochs=1)
    assert tiny_steps == 1


def _run_dry_run(config_path: Path, extra_args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--dry-run",
            "--output-root",
            str(tmp_path),
            *extra_args,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_run_experiment_task_flag_restricts_to_one_task(tmp_path):
    result = _run_dry_run(
        CONFIG_DIR / "experiment_1a_rank_ablation.yaml", ["--task", "rte"], tmp_path
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "task=rte" in output
    assert "task=sst2" not in output
    assert "task=gsm8k" not in output


def test_run_experiment_rejects_unknown_task(tmp_path):
    result = _run_dry_run(
        CONFIG_DIR / "experiment_1a_rank_ablation.yaml", ["--task", "not_a_real_task"], tmp_path
    )
    assert result.returncode != 0
    assert "not_a_real_task" in (result.stdout + result.stderr)


def test_run_experiment_without_task_flag_runs_every_task(tmp_path):
    result = _run_dry_run(CONFIG_DIR / "experiment_1a_rank_ablation.yaml", [], tmp_path)
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    for task in ("sst2", "rte", "gsm8k"):
        assert f"task={task}" in output


def _fake_metrics_df():
    records = [
        {
            "task": "sst2",
            "run_name": "sst2__lora_r8",
            "method": {"type": "lora", "rank": 8},
            "eval_accuracy": 0.8,
            "trainable_params": 1_000,
            "total_params": 1_000_000,
            "training_curve": [{"step": s, "loss": 1.0 / s} for s in range(1, 5)],
        },
        {
            "task": "rte",
            "run_name": "rte__lora_r8",
            "method": {"type": "lora", "rank": 8},
            "eval_accuracy": 0.6,
            "trainable_params": 1_000,
            "total_params": 1_000_000,
            # a much shorter run, e.g. a smaller dataset
            "training_curve": [{"step": s, "loss": 1.5 / s} for s in range(1, 3)],
        },
    ]
    return pd.json_normalize(records, sep="_")


def test_explode_training_curves_keeps_raw_step_per_task():
    df = _fake_metrics_df()
    curves = plot_results._explode_training_curves(df)

    sst2_curve = curves[curves["run_name"] == "sst2__lora_r8"].sort_values("step")
    assert sst2_curve["step"].tolist() == [1, 2, 3, 4]
    assert sst2_curve["loss"].tolist() == pytest.approx([1.0, 0.5, 1 / 3, 0.25])

    rte_curve = curves[curves["run_name"] == "rte__lora_r8"].sort_values("step")
    assert rte_curve["step"].tolist() == [1, 2]


def test_plot_results_produces_one_training_curves_file_per_task(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    records = [
        {
            "experiment": "fake",
            "task": task,
            "run_name": f"{task}__lora_r{rank}",
            "method": {"type": "lora", "rank": rank},
            "eval_accuracy": 0.5,
            "trainable_params": rank * 1000,
            "total_params": 1_000_000,
            "training_curve": [{"step": s, "loss": 1.0 / s} for s in range(1, 4)],
        }
        for task in ("sst2", "rte")
        for rank in (1, 8)
    ]
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)

    out_dir = tmp_path / "plots"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_results.py",
            "--metrics",
            str(metrics_path),
            "--output-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "accuracy_vs_trainable_params.png").exists()
    assert (out_dir / "accuracy_vs_rank.png").exists()
    assert (out_dir / "training_curves_sst2.png").exists()
    assert (out_dir / "training_curves_rte.png").exists()
    # no combined file should be produced
    assert not (out_dir / "training_curves.png").exists()


def test_matrix_label_maps_known_configs_and_is_order_independent():
    assert plot_matrix_study_results.matrix_label(["q_proj"]) == "W_q"
    assert plot_matrix_study_results.matrix_label(["k_proj", "q_proj"]) == "W_q+W_k"
    assert plot_matrix_study_results.matrix_label(["q_proj", "k_proj"]) == "W_q+W_k"
    assert plot_matrix_study_results.matrix_label(
        ["o_proj", "q_proj", "k_proj", "v_proj"]
    ) == "All attention"
    assert plot_matrix_study_results.matrix_label(["down_proj", "gate_proj", "up_proj"]) == "MLP only"
    assert plot_matrix_study_results.matrix_label(["k_proj", "v_proj", "o_proj"]) == "All attn − W_q"
    # an unmapped combo falls back to a joined label instead of crashing
    assert plot_matrix_study_results.matrix_label(["down_proj"]) == "down_proj"


def _fake_matrix_study_records():
    return [
        {"task": "sst2", "run_name": "sst2__full_ft", "method": {"type": "full_ft"}, "eval_accuracy": 0.80},
        {
            "task": "sst2",
            "run_name": "sst2__lora_r8_q",
            "method": {"type": "lora", "rank": 8, "target_modules": ["q_proj"]},
            "eval_accuracy": 0.60,
        },
        {
            "task": "sst2",
            "run_name": "sst2__lora_r8_qkvo",
            "method": {"type": "lora", "rank": 8, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
            "eval_accuracy": 0.72,
        },
        {
            "task": "sst2",
            "run_name": "sst2__lora_r8_kvo",
            "method": {"type": "lora", "rank": 8, "target_modules": ["k_proj", "v_proj", "o_proj"]},
            "eval_accuracy": 0.64,
        },
        {
            "task": "sst2",
            "run_name": "sst2__lora_r8_qvo",
            "method": {"type": "lora", "rank": 8, "target_modules": ["q_proj", "v_proj", "o_proj"]},
            "eval_accuracy": 0.70,
        },
        {
            "task": "sst2",
            "run_name": "sst2__lora_r8_qko",
            "method": {"type": "lora", "rank": 8, "target_modules": ["q_proj", "k_proj", "o_proj"]},
            "eval_accuracy": 0.71,
        },
        {
            "task": "sst2",
            "run_name": "sst2__lora_r8_qkv",
            "method": {"type": "lora", "rank": 8, "target_modules": ["q_proj", "k_proj", "v_proj"]},
            "eval_accuracy": 0.68,
        },
    ]


def _fake_matrix_study_df():
    return pd.json_normalize(_fake_matrix_study_records(), sep="_")


def test_build_heatmap_data_normalizes_against_full_ft():
    df = _fake_matrix_study_df()
    pivot = plot_matrix_study_results.build_heatmap_data(df)

    assert pivot.loc["sst2", "W_q"] == pytest.approx(100 * 0.60 / 0.80)
    assert pivot.loc["sst2", "All attention"] == pytest.approx(100 * 0.72 / 0.80)


def test_build_ablation_delta_data_measures_drop_from_full_attention():
    df = _fake_matrix_study_df()
    deltas = plot_matrix_study_results.build_ablation_delta_data(df).set_index("removed")["delta"]

    # full attention (0.72) minus each leave-one-out condition, in points
    assert deltas["W_q"] == pytest.approx((0.64 - 0.72) * 100)
    assert deltas["W_k"] == pytest.approx((0.70 - 0.72) * 100)
    assert deltas["W_v"] == pytest.approx((0.71 - 0.72) * 100)
    assert deltas["W_o"] == pytest.approx((0.68 - 0.72) * 100)


def test_plot_matrix_study_results_produces_both_plots(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record) + "\n" for record in _fake_matrix_study_records())

    out_dir = tmp_path / "plots"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_matrix_study_results.py",
            "--metrics",
            str(metrics_path),
            "--output-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "heatmap.png").exists()
    assert (out_dir / "ablation_delta.png").exists()
