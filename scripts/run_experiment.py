"""Single CLI entrypoint for running an experiment config.

Used in two places:
  - Docker (local, CPU): `--device cpu --dry-run` to validate that a config
    loads, resolves its model/method(s), and can write to the output dir,
    without downloading real weights or datasets. Logs to the console as
    normal, since there's no tqdm bar running alongside it.
  - Colab (GPU): real run, `--device cuda`, called in-process from a
    notebook cell (not `!python ...` -- see the notebooks/ cells themselves
    for why: subprocess stdout piping breaks tqdm's in-place \\r updates,
    turning every bar refresh into its own printed line). This module's own
    logging is routed to a `session.log` file instead of the console for
    the duration of the run, so tqdm progress bars are the only thing that
    render in the notebook output cell.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.logging_utils import file_logging, get_logger, quiet_console, silence_library_noise

# Must run before transformers/datasets/huggingface_hub are imported
# anywhere else (including transitively, e.g. via config.schema below) —
# some of what it sets only takes effect if set before those libraries'
# own logging is first initialized. See silence_library_noise's docstring
# for why this exists: without it, a full sweep prints enough raw model-
# loading/dataset-download output to crash the browser tab rendering
# Colab's output cell.
silence_library_noise()

from config.schema import ExperimentConfig, MethodSpec
from utils.seeding import set_seed

log = get_logger(__name__)

# Each builder wraps a freshly loaded base model for its method and returns
# the trainable model; training/common.py's train_and_evaluate() runs the
# rest (train, generate, score, write metrics) the same way regardless of
# which builder produced the model.
METHOD_BUILDERS = {
    "full_ft": "training.full_finetune.prepare_full_finetune",
    "lora": "training.lora.build_lora_model",
    "adapters": "training.adapters.build_adapter_model",
    "prefix_tuning": "training.prefix_tuning.build_prefix_tuning_model",
}


def _import_builder(dotted_path: str):
    module_name, func_name = dotted_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/model/method resolution without loading real weights or training.",
    )
    parser.add_argument("--output-root", default="outputs", help="Root dir for run artifacts.")
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Restrict this invocation to a single task from the config's `tasks` list "
            "(runs every task if omitted). Every invocation writes to the same "
            "metrics.jsonl, so splitting a large sweep into one Colab cell per task "
            "(different sessions, resumable if one disconnects) still aggregates "
            "correctly when scripts/plot_results.py reads it afterward."
        ),
    )
    return parser.parse_args(argv)


def run_method(
    method: MethodSpec,
    task: str,
    config: ExperimentConfig,
    train_dataset,
    eval_dataset,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    from models.registry import load_model_and_tokenizer
    from training.common import train_and_evaluate

    builder_path = METHOD_BUILDERS.get(method.type)
    if builder_path is None:
        raise ValueError(f"Unknown method type: {method.type!r}")
    builder = _import_builder(builder_path)

    base_model, tokenizer = load_model_and_tokenizer(config.model.name, device=device)
    model = builder(base_model, method)

    metrics = train_and_evaluate(
        model, tokenizer, task, train_dataset, eval_dataset, method, config, output_dir, device
    )

    del model, base_model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return metrics


def main(argv=None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    set_seed(config.seed)

    tasks = config.tasks
    if args.task is not None:
        if args.task not in config.tasks:
            raise ValueError(f"--task {args.task!r} is not one of this config's tasks: {config.tasks}")
        tasks = [args.task]

    output_dir = Path(args.output_root) / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        log.info("Loaded experiment %r: %s", config.experiment, config.description.strip())
        log.info("Model: %s | Tasks: %s | Device: %s", config.model.name, tasks, args.device)
        for task in tasks:
            for method in config.methods:
                builder_path = METHOD_BUILDERS.get(method.type)
                if builder_path is None:
                    raise ValueError(f"Unknown method type: {method.type!r}")
                log.info(
                    "[dry-run] Skipping model load and training for task=%s method=%s (builder=%s)",
                    task,
                    method.model_dump(exclude_none=True),
                    builder_path,
                )
        log.info("Done. Output dir: %s", output_dir)
        return 0

    # Real (GPU) run: tqdm progress bars must be the only thing that renders
    # in the notebook output cell -- see silence_library_noise's and
    # CompactProgressCallback's docstrings for the rest of this effort.
    # Everything this module would otherwise print to the console instead
    # goes to session.log; quiet_console only detaches `log`'s own console
    # handler, so those messages still reach that file via root-logger
    # propagation inside the file_logging block.
    with file_logging(output_dir / "session.log"), quiet_console(log):
        log.info("Loaded experiment %r: %s", config.experiment, config.description.strip())
        log.info("Model: %s | Tasks: %s | Device: %s", config.model.name, tasks, args.device)

        from tqdm.auto import tqdm

        from data.loaders import load_dataset

        total_runs = len(tasks) * len(config.methods)
        overall = tqdm(
            total=total_runs,
            desc="Experiment",
            unit="run",
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}]{postfix}",
        )
        for task in tasks:
            dataset = load_dataset(task)
            train_dataset, eval_dataset = dataset["train"], dataset["eval"]
            for method in config.methods:
                metrics = run_method(method, task, config, train_dataset, eval_dataset, output_dir, args.device)
                overall.set_postfix_str(f"last={metrics['run_name']} acc={metrics['eval_accuracy']:.3f}")
                overall.update(1)
        overall.close()

        log.info("Done. Output dir: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
