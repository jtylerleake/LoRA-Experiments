"""Experiment 4 entrypoint: LPN on Pattern-2D, comparing `mean` (no
adaptation), `gradient_ascent` (the paper's own latent-vector search), and
three LoRA-ascent variants (ours -- a per-task LoRA adapter on the MLP
kernels of the decoder, the encoder, or both: `lora_ascent_decoder`,
`lora_ascent_encoder`, `lora_ascent_encoder_decoder`, see src/lpn_exp/) on
the same fixed, seeded set of tasks.

Each metrics.jsonl row is one (task, condition): its per-round results
(`round_accuracy`, `round_pixel_correctness`), which rounds are clean
(`round_is_clean` -- see pattern2d_tasks.clean_round_mask), their all-round
means (`accuracy`, `pixel_correctness`), the learning rate, step count and
number of adapted parameters (`num_adapted_params`). Adaptation conditions
also store their optimization trajectories, per round over steps
0..num_steps: `traj_context_log_prob` (the shared objective),
`traj_accuracy` / `traj_pixel_correctness` (the query score of the best
candidate so far), and `best_step` -- see src/lpn_exp/test_time_adapt.py.
Step 0 of every trajectory is the unadapted model, i.e. `mean`.
Learning rates come from the config, or from scripts/tune_exp4_lr.py's
`best_learning_rates.json` via --learning-rates.

Resumable: every row carries a fingerprint of the settings that produced it
(see src/lpn_exp/resume.py), and on start this skips every (task, condition)
already in metrics.jsonl with a matching fingerprint. After a Colab
disconnect, re-run the setup cells and then the same run cell -- it picks up
where it left off. Changing a condition's settings (e.g. a new learning rate
from a re-run sweep) changes its fingerprint, so that condition is redone.
Each task's random key is fold_in(seed, task_id), shared by every
condition, so a resumed run gets the same key as an uninterrupted one and
conditions are compared on identical latent-sampling noise.

Tasks run in batches of `batch_size` per jitted call; a row is written for
each task as soon as its batch finishes.

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
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.metrics import write_metric
from lpn_exp.config import Exp4Config
from lpn_exp.resume import completed_runs, read_jsonl_rows, run_fingerprint
from utils.logging_utils import get_logger, quiet_console, silence_library_noise

silence_library_noise()

log = get_logger(__name__)


def num_steps_for(condition: str, config: Exp4Config) -> int:
    """Adaptation steps a condition runs (0 for `mean`)."""
    if condition == "mean":
        return 0
    if condition == "gradient_ascent":
        return config.gradient_ascent.num_steps
    return config.lora.num_steps


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an exp4 YAML config.")
    parser.add_argument("--output-root", default="outputs", help="Root dir for run artifacts.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Run only these conditions (a subset of the config's list), e.g. so each "
        "condition can run in its own notebook cell. Defaults to every configured condition.",
    )
    parser.add_argument(
        "--learning-rates",
        default=None,
        help="JSON file of {condition: lr} overriding the config's learning_rates -- "
        "scripts/tune_exp4_lr.py's best_learning_rates.json.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = Exp4Config.from_yaml(args.config)
    if args.conditions is not None:
        unknown = [c for c in args.conditions if c not in config.conditions]
        if unknown:
            raise ValueError(f"--conditions {unknown} not in config's conditions {config.conditions}")
        config = config.model_copy(update={"conditions": args.conditions})
    if args.learning_rates is not None:
        with open(args.learning_rates, "r", encoding="utf-8") as f:
            config = config.with_learning_rates(json.load(f))

    output_dir = Path(args.output_root) / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    fingerprints = {
        c: run_fingerprint(config, c, config.seed, config.learning_rates.get(c)) for c in config.conditions
    }
    done = completed_runs(read_jsonl_rows(metrics_path))
    todo = {
        c: [t for t in range(config.num_eval_tasks) if (fingerprints[c], t) not in done]
        for c in config.conditions
    }
    num_done = sum(config.num_eval_tasks - len(ids) for ids in todo.values())
    if num_done:
        log.info("Resuming: %d of %d runs already in %s", num_done, config.num_eval_tasks * len(config.conditions), metrics_path)
    if not any(todo.values()):
        log.info("Nothing to do -- every run is already in %s", metrics_path)
        return 0

    with quiet_console(log):
        log.info("Loaded experiment %r: %s", config.experiment, config.description.strip())
        log.info(
            "Checkpoint: %s (%s) | Conditions: %s | Learning rates: %s",
            config.checkpoint_repo,
            config.checkpoint_name,
            config.conditions,
            {c: config.learning_rates[c] for c in config.conditions if c in config.learning_rates},
        )

        import jax
        from tqdm.auto import tqdm

        from lpn_exp.checkpoint import load_pretrained
        from lpn_exp.pattern2d_tasks import clean_round_mask
        from lpn_exp.task_families import make_tasks
        from lpn_exp.test_time_adapt import build_condition_evaluator, num_adapted_params

        model, frozen_params = load_pretrained(config.checkpoint_repo, config.checkpoint_name)
        max_rows, max_cols = model.decoder.config.max_rows, model.decoder.config.max_cols

        tasks = make_tasks(config.task_generator, config.num_eval_tasks, seed=config.seed)
        base_key = jax.random.PRNGKey(config.seed)

        overall = tqdm(
            total=config.num_eval_tasks * len(config.conditions),
            initial=num_done,
            desc="Experiment",
            unit="run",
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}]{postfix}",
        )
        for condition in config.conditions:
            remaining = [tasks[t] for t in todo[condition]]
            if not remaining:
                continue
            # Built once per condition: it wraps a jitted function that compiles
            # on its first batch and is reused for the rest (see
            # src/lpn_exp/test_time_adapt.py), so the first batch's
            # elapsed_seconds includes that one-time compile.
            evaluate = build_condition_evaluator(
                model, frozen_params, max_rows, max_cols, condition, config, record_trajectory=condition != "mean"
            )
            adapted_params = num_adapted_params(model, frozen_params, condition, config)
            for i in range(0, len(remaining), config.batch_size):
                batch = remaining[i : i + config.batch_size]
                start = time.time()
                results = evaluate(batch, [jax.random.fold_in(base_key, task.task_id) for task in batch])
                elapsed = (time.time() - start) / len(batch)  # per task, amortized over the batch
                for task, result in zip(batch, results):
                    accuracy = sum(result["round_accuracy"]) / len(result["round_accuracy"])
                    write_metric(
                        metrics_path,
                        {
                            "experiment": config.experiment,
                            "task_id": task.task_id,
                            "condition": condition,
                            "accuracy": accuracy,
                            "pixel_correctness": sum(result["round_pixel_correctness"])
                            / len(result["round_pixel_correctness"]),
                            # round_accuracy, round_pixel_correctness, and for
                            # adaptation conditions traj_* / best_step
                            **result,
                            "round_is_clean": clean_round_mask(task),
                            "learning_rate": config.learning_rates.get(condition),
                            "num_steps": num_steps_for(condition, config),
                            "num_adapted_params": adapted_params,
                            "fingerprint": fingerprints[condition],
                            "elapsed_seconds": elapsed,
                        },
                    )
                overall.set_postfix_str(f"last={condition} task={batch[-1].task_id} acc={accuracy:.3f}")
                overall.update(len(batch))
        overall.close()

        log.info("Done. Output dir: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
