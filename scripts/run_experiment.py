"""Single CLI entrypoint for running an experiment config.

Used identically in two places:
  - Docker (local, CPU): `--device cpu --dry-run` to validate that a config
    loads, resolves its model/method(s), and can write to the output dir,
    without downloading real weights or datasets.
  - Colab (GPU): real run, `--device cuda`.

There is only one code path — the flags change behavior, not the script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lora_experiments.config.schema import ExperimentConfig
from lora_experiments.utils.logging_utils import get_logger
from lora_experiments.utils.seeding import set_seed

log = get_logger(__name__)

METHOD_DISPATCH = {
    "lora": "lora_experiments.training.lora",
    "full_ft": "lora_experiments.training.full_finetune",
    "adapters": "lora_experiments.training.adapters",
    "prefix_tuning": "lora_experiments.training.prefix_tuning",
}


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
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    set_seed(config.seed)

    log.info("Loaded experiment %r: %s", config.experiment, config.description.strip())
    log.info("Model: %s | Dataset: %s | Device: %s", config.model.name, config.dataset, args.device)

    output_dir = Path(args.output_root) / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in config.methods:
        module_name = METHOD_DISPATCH.get(method.type)
        if module_name is None:
            raise ValueError(f"Unknown method type: {method.type!r}")
        log.info("Method: %s (module=%s)", method.model_dump(exclude_none=True), module_name)

        if args.dry_run:
            log.info("[dry-run] Skipping model load and training for method=%s", method.type)
            continue

        raise NotImplementedError(
            "Real training is not implemented yet — run with --dry-run to validate "
            "the config/model/method wiring, or see the training/ modules for the "
            "NotImplementedError stubs to fill in first."
        )

    log.info("Done. Output dir: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
