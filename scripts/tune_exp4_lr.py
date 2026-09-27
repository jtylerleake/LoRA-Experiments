"""Experiment 4 learning-rate sweep: for every adaptation condition
(`gradient_ascent` and the three `lora_ascent_*` variants), score each
learning rate in the config's `lr_sweep.learning_rates` on the sweep's own
tasks, and pick the best per condition (see src/lpn_exp/lr_sweep.py for
the rule).

The sweep's tasks come from `lr_sweep.seed`, which the config requires to
differ from the evaluation `seed` -- tuning never sees the evaluation tasks.
Every condition and learning rate gets the same random key per task, so
learning rates are compared on identical latent-sampling noise.

Writes, under <output-root>/<output_subdir>_lr_sweep/:
  lr_sweep.jsonl            one row per (condition, learning_rate, task)
  lr_sweep_summary.csv      pooled clean-round and all-round scores per
                            (condition, learning_rate)
  best_learning_rates.json  {condition: lr} -- pass to
                            run_exp4.py --learning-rates

Resumable, like run_exp4.py: every row carries a fingerprint (see
src/lpn_exp/resume.py), and a re-run skips every (condition, learning rate,
task) already in lr_sweep.jsonl -- so after a Colab disconnect, re-running
the same cell picks up where it left off. The summary and picks only ever
use rows matching the current config's settings, so rows from an older
sweep (e.g. before a config change) are ignored rather than mixed in.

Colab-only, like run_exp4.py -- called in-process from the notebook.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.metrics import write_metric
from lpn_exp.config import ADAPTATION_CONDITIONS, Exp4Config
from lpn_exp.lr_sweep import select_best_learning_rates, summarize_sweep
from lpn_exp.resume import completed_runs, read_jsonl_rows, run_fingerprint
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
    conditions = [c for c in config.conditions if c in ADAPTATION_CONDITIONS]
    sweep = config.lr_sweep

    output_dir = Path(args.output_root) / f"{config.output_subdir}_lr_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "lr_sweep.jsonl"

    fingerprints = {
        (c, lr): run_fingerprint(config, c, sweep.seed, lr) for c in conditions for lr in sweep.learning_rates
    }
    wanted = set(fingerprints.values())
    rows = [row for row in read_jsonl_rows(rows_path) if row.get("fingerprint") in wanted]
    done = completed_runs(rows)
    todo = {
        run: [t for t in range(sweep.num_tasks) if (fp, t) not in done] for run, fp in fingerprints.items()
    }
    total = len(fingerprints) * sweep.num_tasks
    num_done = total - sum(len(ids) for ids in todo.values())
    if num_done:
        log.info("Resuming: %d of %d sweep runs already in %s", num_done, total, rows_path)

    with quiet_console(log):
        log.info(
            "LR sweep for %r | Conditions: %s | Learning rates: %s | %d tasks (seed %d)",
            config.experiment,
            conditions,
            sweep.learning_rates,
            sweep.num_tasks,
            sweep.seed,
        )

        if num_done < total:  # a finished sweep just re-summarizes its rows
            import jax
            from tqdm.auto import tqdm

            from lpn_exp.checkpoint import load_pretrained
            from lpn_exp.pattern2d_tasks import clean_round_mask, generate_tasks
            from lpn_exp.test_time_adapt import build_condition_evaluator

            model, frozen_params = load_pretrained(config.checkpoint_repo, config.checkpoint_name)
            max_rows, max_cols = model.decoder.config.max_rows, model.decoder.config.max_cols

            tasks = generate_tasks(
                num_tasks=sweep.num_tasks,
                num_pairs=config.task_generator.num_pairs,
                num_rows=config.task_generator.num_rows,
                num_cols=config.task_generator.num_cols,
                pattern_size=config.task_generator.pattern_size,
                seed=sweep.seed,
            )
            base_key = jax.random.PRNGKey(sweep.seed)

            overall = tqdm(
                total=total,
                initial=num_done,
                desc="LR sweep",
                unit="run",
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}]{postfix}",
            )
            for (condition, lr), task_ids in todo.items():
                if not task_ids:
                    continue
                # One compile per (condition, lr): lr is a constant of the jitted function.
                evaluate = build_condition_evaluator(
                    model, frozen_params, max_rows, max_cols, condition, config, lr=lr
                )
                remaining = [tasks[t] for t in task_ids]
                for i in range(0, len(remaining), config.batch_size):
                    batch = remaining[i : i + config.batch_size]
                    results = evaluate(batch, [jax.random.fold_in(base_key, task.task_id) for task in batch])
                    for task, (round_accuracy, round_pixel_correctness) in zip(batch, results):
                        row = {
                            "experiment": config.experiment,
                            "condition": condition,
                            "learning_rate": lr,
                            "task_id": task.task_id,
                            "round_accuracy": round_accuracy,
                            "round_pixel_correctness": round_pixel_correctness,
                            "round_is_clean": clean_round_mask(task),
                            "fingerprint": fingerprints[(condition, lr)],
                        }
                        write_metric(rows_path, row)
                        rows.append(row)
                    overall.set_postfix_str(f"{condition} lr={lr:g}")
                    overall.update(len(batch))
            overall.close()

        summaries = summarize_sweep(rows)
        with open(output_dir / "lr_sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)

        best = select_best_learning_rates(rows)
        with open(output_dir / "best_learning_rates.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

    # Outside quiet_console, so the picks show up in the notebook cell.
    log.info("Best learning rates: %s", best)
    log.info("Done. Output dir: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
