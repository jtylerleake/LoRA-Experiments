"""Experiment 4 entrypoint: LPN on Pattern-2D, comparing `mean` (no
adaptation), `gradient_ascent` (the paper's own latent-vector search), and
`lora_ascent` (ours -- a per-task LoRA adapter on the decoder's MLP
kernels, see src/lpn_exp/) on the same fixed, seeded set of tasks.

Unlike scripts/run_experiment.py, this is JAX/Flax, not PyTorch/HF -- it
does not go through config/schema.py's ExperimentConfig or
training/common.py. See CLAUDE-CODING-SKILL.md and this experiment's plan
for why experiment 4 is a separate pipeline.

Colab-only (see notebooks/exp4_test_time_tuning.ipynb for the JAX/lpn
bootstrap cell) -- called in-process from the notebook, not `!python ...`,
so tqdm's progress bar renders correctly (see CLAUDE-CODING-SKILL.md's
tqdm/Colab/subprocess note).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.metrics import write_metric
from lpn_exp.config import Exp4Config
from utils.logging_utils import get_logger, quiet_console, silence_library_noise

silence_library_noise()

log = get_logger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an exp4 YAML config.")
    parser.add_argument("--output-root", default="outputs", help="Root dir for run artifacts.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = Exp4Config.from_yaml(args.config)

    output_dir = Path(args.output_root) / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    with quiet_console(log):
        log.info("Loaded experiment %r: %s", config.experiment, config.description.strip())
        log.info(
            "Checkpoint: %s (%s) | Conditions: %s",
            config.checkpoint_repo,
            config.checkpoint_name,
            config.conditions,
        )

        import jax
        from tqdm.auto import tqdm

        from lpn_exp.checkpoint import load_pretrained
        from lpn_exp.pattern2d_tasks import generate_tasks
        from lpn_exp.test_time_adapt import (
            evaluate_condition_lora_ascent,
            evaluate_condition_mean_or_gradient_ascent,
        )

        model, frozen_params = load_pretrained(config.checkpoint_repo, config.checkpoint_name)
        max_rows, max_cols = model.decoder.config.max_rows, model.decoder.config.max_cols

        tasks = generate_tasks(
            num_tasks=config.num_eval_tasks,
            num_pairs=config.task_generator.num_pairs,
            num_rows=config.task_generator.num_rows,
            num_cols=config.task_generator.num_cols,
            pattern_size=config.task_generator.pattern_size,
            seed=config.seed,
        )

        key = jax.random.PRNGKey(config.seed)
        overall = tqdm(
            total=len(tasks) * len(config.conditions),
            desc="Experiment",
            unit="run",
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}]{postfix}",
        )
        for task in tasks:
            for condition in config.conditions:
                start = time.time()
                key, sub_key = jax.random.split(key)
                if condition == "mean":
                    accuracy, pixel_correctness = evaluate_condition_mean_or_gradient_ascent(
                        model,
                        frozen_params,
                        task,
                        max_rows,
                        max_cols,
                        mode="mean",
                        mode_kwargs={},
                        key=sub_key,
                    )
                elif condition == "gradient_ascent":
                    ga_kwargs = {
                        "num_steps": config.gradient_ascent.num_steps,
                        "lr": config.gradient_ascent.lr,
                    }
                    accuracy, pixel_correctness = evaluate_condition_mean_or_gradient_ascent(
                        model,
                        frozen_params,
                        task,
                        max_rows,
                        max_cols,
                        mode="gradient_ascent",
                        mode_kwargs=ga_kwargs,
                        key=sub_key,
                    )
                elif condition == "lora_ascent":
                    accuracy, pixel_correctness = evaluate_condition_lora_ascent(
                        model,
                        frozen_params,
                        task,
                        max_rows,
                        max_cols,
                        rank=config.lora.rank,
                        scale=config.lora.scale,
                        num_steps=config.lora.num_steps,
                        lr=config.lora.lr,
                        prior_kl_coeff=0.001,  # matches the checkpoint's own training.kl_coeff
                        key=sub_key,
                    )
                else:
                    raise ValueError(f"Unknown condition: {condition!r}")

                elapsed = time.time() - start
                write_metric(
                    output_dir / "metrics.jsonl",
                    {
                        "experiment": config.experiment,
                        "task_id": task.task_id,
                        "condition": condition,
                        "accuracy": accuracy,
                        "pixel_correctness": pixel_correctness,
                        "elapsed_seconds": elapsed,
                    },
                )
                overall.set_postfix_str(f"last={condition} task={task.task_id} acc={accuracy:.3f}")
                overall.update(1)
        overall.close()

        log.info("Done. Output dir: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
