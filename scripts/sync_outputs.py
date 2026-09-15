"""Pulls the small metrics/logs files (not checkpoints) for a run from
Google Drive down into the local outputs/ tree, so they can be plotted
locally (e.g. with seaborn inside the Docker dev container) without ever
downloading model weights.

Not implemented yet — the intended approach is `rclone` configured against
Google Drive (simplest for a Colab Pro / personal Drive setup), invoked as a
subprocess scoped to the run's metrics/ and logs/ subdirectories. Filled in
once the first real Colab run produces something to sync.
"""
from __future__ import annotations

import argparse


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    raise NotImplementedError(
        f"sync_outputs not implemented yet (experiment={args.experiment!r}, "
        f"run_id={args.run_id!r}). See module docstring for the planned rclone approach."
    )


if __name__ == "__main__":
    raise SystemExit(main())
