"""Smoke tests runnable in the Docker CPU dev image — confirm the package
imports and the experiment configs are well-formed, without touching any
real model weights or datasets.
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    write_metric,
)
from lpn_exp.config import ADAPTATION_CONDITIONS, Exp4Config, TaskGeneratorConfig
from lpn_exp.lora import LORA_CONDITION_TARGETS, find_mlp_kernel_paths, merge_lora
from lpn_exp.lr_sweep import select_best_learning_rates
from lpn_exp.pattern2d_tasks import Pattern2DTask, clean_round_mask
from lpn_exp.resume import SWEEP_ALGORITHM_VERSION, read_jsonl_rows, run_fingerprint
from lpn_exp.task_families import all_rounds_solvable, make_tasks, solve_round
from models.registry import get_model_spec
from training.adapters import build_adapter_model
from training.common import (
    CompactProgressCallback,
    cap_dataset,
    compute_logging_steps,
    target_modules_slug,
)
from utils.logging_utils import file_logging, silence_library_noise

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


def test_exp1_methods_are_lora_only_across_all_ranks():
    """Full fine-tuning was dropped from experiment 1 for compute cost --
    experiment 3 is where it gets a direct comparison instead.
    """
    config = ExperimentConfig.from_yaml(CONFIG_DIR / "experiment_1_rank_ablation.yaml")
    assert config.tasks == ["sst2", "rte", "gsm8k"]
    method_types = {m.type for m in config.methods}
    assert method_types == {"lora"}
    ranks = sorted(m.rank for m in config.methods)
    assert ranks == [1, 8, 64]


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
    assert exp1a_ranks == exp1_ranks == [1, 8, 64]
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
    # Full fine-tuning was dropped for compute cost (experiment 3 is where
    # it gets a direct comparison instead), so this is every
    # individual/paired/grouped/leave-one-out target-module variant the
    # heatmap and ablation-delta plots need, and nothing else.
    assert {m.type for m in full.methods} == {"lora"}
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


def _fake_lpn_params():
    import numpy as np

    return {
        "decoder": {
            "TransformerLayer_0": {
                "MlpBlock_0": {
                    "Dense_0": {"kernel": np.arange(6.0).reshape(2, 3)},
                    "Dense_1": {"kernel": np.arange(6.0).reshape(3, 2)},
                },
            },
            "context_embed": {"kernel": np.ones((2, 3))},
        },
        "encoder": {
            "TransformerLayer_0": {
                "MlpBlock_0": {
                    "Dense_0": {"kernel": np.arange(6.0).reshape(2, 3)},
                    "Dense_1": {"kernel": np.arange(6.0).reshape(3, 2)},
                },
            },
            "Dense_0": {"kernel": np.ones((3, 2))},
        },
    }


_DECODER_MLP_PATHS = [
    ("decoder", "TransformerLayer_0", "MlpBlock_0", "Dense_0", "kernel"),
    ("decoder", "TransformerLayer_0", "MlpBlock_0", "Dense_1", "kernel"),
]
_ENCODER_MLP_PATHS = [
    ("encoder", "TransformerLayer_0", "MlpBlock_0", "Dense_0", "kernel"),
    ("encoder", "TransformerLayer_0", "MlpBlock_0", "Dense_1", "kernel"),
]


def test_find_mlp_kernel_paths_respects_each_lora_condition_target():
    """Regression test: each of experiment 4's LoRA-ascent conditions targets
    only the MlpBlock Dense kernels of its own submodule(s) (see
    src/lpn_exp/lora.py's LORA_CONDITION_TARGETS) -- never the decoder's
    context_embed, the encoder's latent heads, or the other submodule's
    MLP kernels.
    """
    params = _fake_lpn_params()
    expected = {
        "lora_ascent_decoder": _DECODER_MLP_PATHS,
        "lora_ascent_encoder": _ENCODER_MLP_PATHS,
        "lora_ascent_encoder_decoder": _ENCODER_MLP_PATHS + _DECODER_MLP_PATHS,
    }
    assert set(LORA_CONDITION_TARGETS) == set(expected)
    for condition, modules in LORA_CONDITION_TARGETS.items():
        assert find_mlp_kernel_paths(params, modules) == expected[condition], condition


def test_find_mlp_kernel_paths_rejects_unknown_module():
    params = _fake_lpn_params()
    with pytest.raises(KeyError):
        find_mlp_kernel_paths(params, ("decodr",))


def test_merge_lora_with_zero_b_is_a_no_op():
    """Regression test: a fresh LoRA adapter (zero-initialized `b`, see
    src/lpn_exp/lora.py's init_lora_params) must leave the pretrained LPN
    encoder and decoder completely unchanged before any test-time gradient-ascent steps
    have run -- same identity-at-init principle as the PyTorch bottleneck
    adapter above. This exercises merge_lora's pytree-patching logic with
    plain numpy, without needing jax/flax installed (see
    CLAUDE-CODING-SKILL.md).
    """
    import numpy as np

    params = _fake_lpn_params()
    paths = find_mlp_kernel_paths(params, ("encoder", "decoder"))
    rank = 4
    lora_params = {}
    for path in paths:
        kernel = params[path[0]]["TransformerLayer_0"]["MlpBlock_0"][path[-2]]["kernel"]
        in_dim, out_dim = kernel.shape
        lora_params[path] = {"a": np.ones((in_dim, rank)), "b": np.zeros((rank, out_dim))}

    merged = merge_lora(params, lora_params, scale=1.0)

    for path in paths:
        original = params
        patched = merged
        for key in path:
            original, patched = original[key], patched[key]
        np.testing.assert_array_equal(patched, original)


def test_merge_lora_adds_low_rank_update_when_b_is_nonzero():
    import numpy as np

    params = _fake_lpn_params()
    path = ("decoder", "TransformerLayer_0", "MlpBlock_0", "Dense_0", "kernel")
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    lora_params = {path: {"a": a, "b": b}}

    merged = merge_lora(params, lora_params, scale=2.0)

    original_kernel = params["decoder"]["TransformerLayer_0"]["MlpBlock_0"]["Dense_0"]["kernel"]
    expected = original_kernel + 2.0 * (a @ b)
    np.testing.assert_array_equal(
        merged["decoder"]["TransformerLayer_0"]["MlpBlock_0"]["Dense_0"]["kernel"], expected
    )
    # Untouched leaves elsewhere in the pytree are unchanged.
    np.testing.assert_array_equal(
        merged["decoder"]["context_embed"]["kernel"], params["decoder"]["context_embed"]["kernel"]
    )


def test_exp4_config_lists_all_three_lora_ascent_variants():
    """exp4's YAML must validate against Exp4Config and run every LoRA-ascent
    variant (decoder, encoder, encoder+decoder) alongside both baselines.
    """
    config = Exp4Config.from_yaml(CONFIG_DIR / "exp4_lpn_pattern2d.yaml")
    assert config.conditions == ["mean", "gradient_ascent", *LORA_CONDITION_TARGETS]


def test_exp4_mini_mirrors_full_structure_but_capped():
    full = Exp4Config.from_yaml(CONFIG_DIR / "exp4_lpn_pattern2d.yaml")
    mini = Exp4Config.from_yaml(CONFIG_DIR / "exp4_lpn_pattern2d_mini.yaml")
    assert mini.conditions == full.conditions
    assert mini.checkpoint_repo == full.checkpoint_repo
    assert mini.checkpoint_name == full.checkpoint_name
    assert mini.task_generator == full.task_generator
    assert mini.lora.rank == full.lora.rank
    assert mini.num_eval_tasks < full.num_eval_tasks
    assert mini.gradient_ascent.num_steps <= full.gradient_ascent.num_steps
    assert mini.lora.num_steps <= full.lora.num_steps
    assert mini.output_subdir != full.output_subdir
    assert mini.learning_rates.keys() == full.learning_rates.keys()
    assert mini.lr_sweep.num_tasks < full.lr_sweep.num_tasks


def test_exp4_config_rejects_unknown_condition():
    with pytest.raises(ValueError):
        Exp4Config(conditions=["mean", "lora_ascent"])


def test_exp4_every_adaptation_condition_has_a_learning_rate():
    config = Exp4Config.from_yaml(CONFIG_DIR / "exp4_lpn_pattern2d.yaml")
    assert set(config.learning_rates) == set(ADAPTATION_CONDITIONS)
    with pytest.raises(ValueError):
        Exp4Config(learning_rates={"gradient_ascent": 0.1})  # LoRA conditions missing
    with pytest.raises(ValueError):
        Exp4Config(learning_rates={**config.learning_rates, "mean": 0.1})  # mean has no lr


def test_exp4_lr_sweep_never_uses_the_evaluation_seed():
    with pytest.raises(ValueError):
        Exp4Config(seed=5, lr_sweep={"seed": 5})


def test_exp4_with_learning_rates_merges_and_validates():
    config = Exp4Config()
    tuned = config.with_learning_rates({"lora_ascent_encoder": 0.3})
    assert tuned.learning_rates["lora_ascent_encoder"] == 0.3
    assert tuned.learning_rates["gradient_ascent"] == config.learning_rates["gradient_ascent"]
    with pytest.raises(ValueError):
        config.with_learning_rates({"not_a_condition": 0.3})


def _pattern_task(positions):
    """A Pattern2DTask whose pairs place one fixed 2x2 pattern at `positions`."""
    import numpy as np

    pattern = np.array([[1, 2], [3, 4]])
    pairs = []
    for row, col in positions:
        grid_in, grid_out = np.zeros((4, 4), dtype=int), np.zeros((4, 4), dtype=int)
        grid_in[row, col] = 1
        grid_out[row : row + 2, col : col + 2] = pattern
        pairs.append({"input": grid_in, "output": grid_out})
    return Pattern2DTask(task_id=0, pairs=pairs)


def test_clean_round_mask_flags_rounds_whose_query_is_in_its_context():
    # Pairs 0 and 2 share a position, so each is the other's duplicate.
    task = _pattern_task([(0, 0), (1, 1), (0, 0), (2, 2)])
    assert clean_round_mask(task) == [False, True, False, True]
    assert clean_round_mask(_pattern_task([(0, 0), (0, 1), (1, 0), (1, 1)])) == [True] * 4


def _sweep_row(condition, lr, accuracies, clean):
    return {
        "condition": condition,
        "learning_rate": lr,
        "round_accuracy": accuracies,
        "round_pixel_correctness": accuracies,
        "round_is_clean": clean,
    }


def test_select_best_learning_rates_scores_clean_rounds_only():
    # lr 1.0 wins only on the leaked (unclean) round; on clean rounds 0.1 is better.
    rows = [
        _sweep_row("gradient_ascent", 0.1, [1.0, 1.0, 0.0], [True, True, False]),
        _sweep_row("gradient_ascent", 1.0, [1.0, 0.0, 1.0], [True, True, False]),
    ]
    assert select_best_learning_rates(rows) == {"gradient_ascent": 0.1}


def test_select_best_learning_rates_breaks_ties_toward_the_smaller_lr():
    rows = [
        _sweep_row("lora_ascent_decoder", 0.3, [1.0, 0.0], [True, True]),
        _sweep_row("lora_ascent_decoder", 0.03, [0.0, 1.0], [True, True]),
    ]
    assert select_best_learning_rates(rows) == {"lora_ascent_decoder": 0.03}


def test_select_best_learning_rates_falls_back_to_all_rounds_without_clean_ones():
    rows = [
        _sweep_row("lora_ascent_encoder", 0.1, [0.0, 0.0], [False, False]),
        _sweep_row("lora_ascent_encoder", 1.0, [1.0, 0.0], [False, False]),
    ]
    assert select_best_learning_rates(rows) == {"lora_ascent_encoder": 1.0}


def test_exp4_results_table_reports_all_and_clean_rounds():
    from plot_exp4_results import build_results_table

    df = pd.DataFrame(
        [
            {"task_id": 0, "condition": "mean", "round_accuracy": [1.0, 0.0, 1.0, 1.0],
             "round_pixel_correctness": [1.0, 0.5, 1.0, 1.0], "round_is_clean": [True, True, False, False]},
            {"task_id": 1, "condition": "mean", "round_accuracy": [0.0, 1.0, 1.0, 1.0],
             "round_pixel_correctness": [0.5, 1.0, 1.0, 1.0], "round_is_clean": [True, True, True, True]},
        ]
    )
    table = build_results_table(df, num_resamples=200).set_index("rounds")
    assert table.loc["All rounds", "accuracy"] == pytest.approx(6 / 8)
    assert table.loc["All rounds", "num_rounds"] == 8
    # Clean rounds: task 0's first two (1, 0) + all four of task 1's (0, 1, 1, 1) -> 4/6.
    assert table.loc["Clean-only rounds", "accuracy"] == pytest.approx(4 / 6)
    assert table.loc["Clean-only rounds", "num_rounds"] == 6
    assert table.loc["Clean-only rounds", "accuracy_ci_low"] <= 4 / 6 <= table.loc["Clean-only rounds", "accuracy_ci_high"]


def test_read_jsonl_rows_repairs_a_partial_last_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    assert read_jsonl_rows(path) == []  # missing file
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b"', encoding="utf-8")  # cut off mid-write
    assert read_jsonl_rows(path) == [{"a": 1}, {"a": 2}]
    # The fragment is gone from the file, so the next append lands on its own line.
    write_metric(path, {"a": 4})
    assert read_jsonl_rows(path) == [{"a": 1}, {"a": 2}, {"a": 4}]


def test_read_jsonl_rows_rejects_a_malformed_middle_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"a": 1}\nnot json\n{"a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl_rows(path)


LADDER_CONFIGS = sorted(p for p in CONFIG_DIR.glob("exp4_ladder_L*.yaml") if not p.stem.endswith("_mini"))


def test_exp4_ladder_has_every_planned_level():
    families = {p.stem: Exp4Config.from_yaml(p).task_generator for p in LADDER_CONFIGS}
    assert [p.stem.split("_")[2] for p in LADDER_CONFIGS] == ["L1", "L2", "L3", "L4"]
    assert families["exp4_ladder_L1_sparse2x2"].pattern_density < 1.0
    assert families["exp4_ladder_L2_pattern3x3"].pattern_size == 3
    assert families["exp4_ladder_L3_pattern3x3_corner"].anchor == "random_corner"
    assert families["exp4_ladder_L4_color_permutation"].family == "color_permutation"


@pytest.mark.parametrize("config_path", LADDER_CONFIGS, ids=lambda p: p.stem)
def test_exp4_ladder_level_matches_l0_except_its_tasks(config_path):
    """Only the tasks may differ from L0, so the levels stay comparable."""
    ignored = {"experiment", "description", "output_subdir", "task_generator"}
    level = Exp4Config.from_yaml(config_path).model_dump(exclude=ignored)
    l0 = Exp4Config.from_yaml(CONFIG_DIR / "exp4_lpn_pattern2d.yaml").model_dump(exclude=ignored)
    assert level == l0


@pytest.mark.parametrize("config_path", LADDER_CONFIGS, ids=lambda p: p.stem)
def test_exp4_ladder_tasks_are_well_posed_and_clean(config_path):
    task_generator = Exp4Config.from_yaml(config_path).task_generator
    tasks = make_tasks(task_generator, 64, seed=42)
    assert [t.task_id for t in tasks] == list(range(64))
    for task in tasks:
        assert len(task.pairs) == task_generator.num_pairs
        for pair in task.pairs:
            assert pair["input"].shape == pair["output"].shape == (4, 4)
            assert pair["output"].min() >= 0 and pair["output"].max() <= 9
        # The exact solver recovers every held-out output from its context alone.
        assert all_rounds_solvable(task_generator, task.pairs)
        if task_generator.family == "pattern":
            assert clean_round_mask(task) == [True] * task_generator.num_pairs


def test_exp4_ladder_task_k_depends_only_on_seed_and_k():
    task_generator = TaskGeneratorConfig(family="pattern", pattern_size=3)
    few, many = make_tasks(task_generator, 3, seed=7), make_tasks(task_generator, 40, seed=7)
    for a, b in zip(few, many):
        assert all(np.array_equal(p["output"], q["output"]) for p, q in zip(a.pairs, b.pairs))
    other_seed = make_tasks(task_generator, 3, seed=8)
    assert any(
        not np.array_equal(p["output"], q["output"]) for p, q in zip(few[0].pairs, other_seed[0].pairs)
    )


def test_exp4_pattern_family_respects_its_settings():
    full = make_tasks(TaskGeneratorConfig(family="pattern", pattern_size=3), 32, seed=0)
    for task in full:
        for pair in task.pairs:
            assert (pair["input"] != 0).sum() == 1 and pair["input"].max() == 1  # one marker pixel
            assert (pair["output"] != 0).sum() == 9  # density 1.0: a full 3x3 pattern
    sparse = make_tasks(TaskGeneratorConfig(family="pattern", pattern_size=2, pattern_density=0.5), 64, seed=0)
    filled = [(t.pairs[0]["output"] != 0).sum() for t in sparse]
    assert min(filled) >= 1 and max(filled) <= 4 and any(n < 4 for n in filled)
    # random_corner: the marker sits at different corners of the pattern across tasks.
    corner = TaskGeneratorConfig(family="pattern", pattern_size=3, anchor="random_corner")
    offsets = set()
    for task in make_tasks(corner, 64, seed=0):
        marker = tuple(np.argwhere(task.pairs[0]["input"])[0])
        top_left = tuple(np.argwhere(task.pairs[0]["output"]).min(axis=0))
        offsets.add((marker[0] - top_left[0], marker[1] - top_left[1]))
    assert offsets == {(0, 0), (0, 2), (2, 0), (2, 2)}


def test_exp4_color_permutation_family_respects_its_settings():
    task_generator = TaskGeneratorConfig(family="color_permutation", num_colors=4)
    for task in make_tasks(task_generator, 32, seed=0):
        colors = set(np.unique(np.concatenate([p["input"].ravel() for p in task.pairs])))
        assert len(colors) <= 4 and 0 not in colors
        # One consistent recoloring across all pairs.
        mapping = {}
        for pair in task.pairs:
            for c_in, c_out in zip(pair["input"].ravel(), pair["output"].ravel()):
                assert mapping.setdefault(c_in, c_out) == c_out


def test_exp4_solver_refuses_rounds_the_context_does_not_determine():
    task_generator = TaskGeneratorConfig(family="color_permutation", num_colors=4)
    context = [{"input": np.full((4, 4), 1), "output": np.full((4, 4), 5)}]
    assert solve_round(task_generator, context, np.full((4, 4), 2)) is None  # color 2 never shown
    assert (solve_round(task_generator, context, np.full((4, 4), 1)) == 5).all()


@pytest.mark.parametrize(
    "bad",
    [
        {"num_rows": 5, "num_cols": 5},  # bigger than the checkpoint's 4x4
        {"family": "pattern", "pattern_size": 3, "num_pairs": 5},  # only 4 positions for 3x3
        {"family": "pattern", "pattern_size": 4},  # doesn't fit with a marker
        {"family": "lpn_pattern", "pattern_density": 0.5},  # lpn's generator has no sparse mode here
        {"family": "color_permutation", "num_colors": 10},
        {"num_pairs": 1},
    ],
)
def test_exp4_task_generator_rejects_unsupported_settings(bad):
    with pytest.raises(ValueError):
        TaskGeneratorConfig(**bad)


def test_exp4_l0_fingerprints_unchanged_by_the_ladder():
    """Pinned to the values already on Drive: the ladder's new generator
    settings must not change L0's fingerprints, or resume would redo (and
    the results table would ignore) completed L0 work.
    """
    config = Exp4Config.from_yaml(CONFIG_DIR / "exp4_lpn_pattern2d.yaml")
    assert run_fingerprint(config, "mean", config.seed, None) == "fb620b635eb239ba"
    sweep = {
        "gradient_ascent": "62f79ec3ae4109cf",
        "lora_ascent_decoder": "4151c737c3803283",
        "lora_ascent_encoder": "2084e4044ece8c7a",
        "lora_ascent_encoder_decoder": "a4781bc0d5b7bee1",
    }
    for condition, expected in sweep.items():
        assert run_fingerprint(config, condition, config.lr_sweep.seed, 0.1, version=SWEEP_ALGORITHM_VERSION) == expected


def test_exp4_ladder_levels_have_distinct_fingerprints():
    fingerprints = {
        run_fingerprint(Exp4Config.from_yaml(p), "gradient_ascent", 42, 0.1)
        for p in [CONFIG_DIR / "exp4_lpn_pattern2d.yaml", *LADDER_CONFIGS]
    }
    assert len(fingerprints) == 1 + len(LADDER_CONFIGS)


def test_run_fingerprint_tracks_what_changes_a_result():
    config = Exp4Config.from_yaml(CONFIG_DIR / "exp4_lpn_pattern2d.yaml")
    base = run_fingerprint(config, "lora_ascent_decoder", config.seed, 0.1)
    assert run_fingerprint(config, "lora_ascent_decoder", config.seed, 0.1) == base
    assert run_fingerprint(config, "lora_ascent_decoder", config.seed, 0.3) != base
    assert run_fingerprint(config, "lora_ascent_encoder", config.seed, 0.1) != base
    assert run_fingerprint(config, "lora_ascent_decoder", config.seed + 1, 0.1) != base
    more_steps = config.model_copy(update={"lora": config.lora.model_copy(update={"num_steps": 20})})
    assert run_fingerprint(more_steps, "lora_ascent_decoder", config.seed, 0.1) != base
    # Batch size only changes grouping, not results; mean has no learning rate.
    rebatched = config.model_copy(update={"batch_size": 7})
    assert run_fingerprint(rebatched, "lora_ascent_decoder", config.seed, 0.1) == base
    assert run_fingerprint(config, "mean", config.seed, None) == run_fingerprint(config, "mean", config.seed, 0.5)


def _write_exp4_rows(path, config, condition, seed, lr, task_ids, version=None, **extra):
    for task_id in task_ids:
        write_metric(
            path,
            {
                "task_id": task_id,
                "condition": condition,
                "learning_rate": lr,
                "round_accuracy": [1.0, 1.0, 0.0, 1.0],
                "round_pixel_correctness": [1.0, 1.0, 0.5, 1.0],
                "round_is_clean": [True, True, True, False],
                "fingerprint": run_fingerprint(config, condition, seed, lr, **({"version": version} if version else {})),
                **extra,
            },
        )


def test_run_exp4_resume_skips_every_completed_run(tmp_path, caplog):
    """With every (task, condition) already recorded, run_exp4 must finish
    without redoing anything -- it returns before importing jax at all,
    which this CPU test image doesn't have.
    """
    from run_exp4 import main as run_exp4_main

    config_path = CONFIG_DIR / "exp4_lpn_pattern2d_mini.yaml"
    config = Exp4Config.from_yaml(config_path)
    metrics = tmp_path / config.output_subdir / "metrics.jsonl"
    for condition in config.conditions:
        _write_exp4_rows(metrics, config, condition, config.seed, config.learning_rates.get(condition),
                         range(config.num_eval_tasks))
    before = metrics.read_text(encoding="utf-8")
    with caplog.at_level(logging.INFO):
        assert run_exp4_main(["--config", str(config_path), "--output-root", str(tmp_path)]) == 0
    assert "Nothing to do" in caplog.text
    assert metrics.read_text(encoding="utf-8") == before


def test_run_exp4_num_steps_for_each_condition():
    from run_exp4 import num_steps_for

    config = Exp4Config.from_yaml(CONFIG_DIR / "exp4_lpn_pattern2d.yaml")
    assert num_steps_for("mean", config) == 0
    assert num_steps_for("gradient_ascent", config) == config.gradient_ascent.num_steps
    for condition in LORA_CONDITION_TARGETS:
        assert num_steps_for(condition, config) == config.lora.num_steps


def test_tune_exp4_lr_resume_resummarizes_a_finished_sweep(tmp_path):
    from tune_exp4_lr import main as tune_main

    config_path = CONFIG_DIR / "exp4_lpn_pattern2d_mini.yaml"
    config = Exp4Config.from_yaml(config_path)
    rows = tmp_path / f"{config.output_subdir}_lr_sweep" / "lr_sweep.jsonl"
    for condition in ADAPTATION_CONDITIONS:
        for lr in config.lr_sweep.learning_rates:
            _write_exp4_rows(rows, config, condition, config.lr_sweep.seed, lr, range(config.lr_sweep.num_tasks),
                             version=SWEEP_ALGORITHM_VERSION)
    # A stale row from other settings must be ignored, not mixed in.
    _write_exp4_rows(rows, config, "gradient_ascent", config.lr_sweep.seed, 99.0, [0], version=SWEEP_ALGORITHM_VERSION)
    assert tune_main(["--config", str(config_path), "--output-root", str(tmp_path)]) == 0
    best = json.loads((rows.parent / "best_learning_rates.json").read_text(encoding="utf-8"))
    # Every lr scored identically, so ties go to the smallest.
    assert best == {c: min(config.lr_sweep.learning_rates) for c in ADAPTATION_CONDITIONS}


def test_plot_exp4_load_metrics_reports_only_each_conditions_latest_settings(tmp_path):
    from plot_exp4_results import load_metrics

    config = Exp4Config()
    path = tmp_path / "metrics.jsonl"
    _write_exp4_rows(path, config, "lora_ascent_decoder", config.seed, 0.1, [0, 1, 2])  # old lr
    _write_exp4_rows(path, config, "lora_ascent_decoder", config.seed, 0.03, [0, 1])  # redo, cut off
    _write_exp4_rows(path, config, "mean", config.seed, None, [0, 1, 1])  # task 1 repeated
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"task_id": 2, "cond')  # interrupted write
    df = load_metrics(path)
    lora = df[df["condition"] == "lora_ascent_decoder"]
    assert sorted(lora["task_id"]) == [0, 1] and set(lora["learning_rate"]) == {0.03}
    assert sorted(df[df["condition"] == "mean"]["task_id"]) == [0, 1]


def test_notebooks_have_no_literal_backslash_n():
    """A literal backslash-n (instead of a real line break) in a cell breaks
    `!` shell line continuations: the shell passes a stray `n` argument.
    """
    for path in (REPO_ROOT / "notebooks").glob("*.ipynb"):
        for i, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
            source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            assert "\\n" not in source, f"{path.name} cell {i} has a literal backslash-n"


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


def test_silence_library_noise_sets_env_vars_and_quiets_libraries(monkeypatch):
    """Regression test: a full sweep was crashing the Colab browser tab
    because model config dumps, tokenizer/weight-loading messages, and
    huggingface_hub's tqdm-based download progress bars were never
    suppressed -- file_logging() only ever redirected Python `logging`
    calls made *during* training, which is too late (model loading happens
    before that) and doesn't touch hub's tqdm bars at all (not logging-based).
    """
    for var in (
        "TRANSFORMERS_VERBOSITY",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "DATASETS_VERBOSITY",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(var, raising=False)

    silence_library_noise()
    silence_library_noise()  # must be safe to call more than once

    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["DATASETS_VERBOSITY"] == "error"

    import transformers.utils.logging as hf_logging

    assert hf_logging.get_verbosity() == hf_logging.ERROR


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


def test_build_heatmap_data_uses_raw_accuracy():
    """Regression test: experiment 2 dropped its full-FT baseline for
    compute cost, so the heatmap can no longer normalize against it --
    it shows each LoRA config's raw eval accuracy instead.
    """
    df = _fake_matrix_study_df()
    pivot = plot_matrix_study_results.build_heatmap_data(df)

    assert pivot.loc["sst2", "W_q"] == pytest.approx(100 * 0.60)
    assert pivot.loc["sst2", "All attention"] == pytest.approx(100 * 0.72)


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
