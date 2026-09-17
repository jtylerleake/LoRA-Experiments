from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def silence_library_noise() -> None:
    """Call once, as early as possible in a real run (before transformers/
    datasets/huggingface_hub are imported anywhere else, so env-var-gated
    settings take effect) — stops them printing to the console at all.

    A 18-39-run sweep otherwise prints, per run: a full model config as
    formatted JSON, tokenizer/weight-loading messages, "some weights were
    not initialized" boilerplate, and tqdm-based download/loading progress
    bars from huggingface_hub — none of which go through Python's
    `logging` module, so redirecting *that* (see `file_logging` below)
    never touched them. Across a whole sweep this is enough raw output to
    crash the browser tab rendering Colab's output cell, which is exactly
    what happened before this fix — suppress at the source instead of
    trying to redirect it after the fact.
    """
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import transformers.utils.logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:
        pass

    try:
        import datasets.utils.logging as ds_logging

        ds_logging.set_verbosity_error()
        ds_logging.disable_progress_bar()
    except ImportError:
        pass

    try:
        import huggingface_hub.utils as hf_hub_utils

        hf_hub_utils.disable_progress_bars()
    except (ImportError, AttributeError):
        pass


@contextmanager
def quiet_console(logger: logging.Logger):
    """Temporarily detach `logger`'s own handlers (its direct-to-stdout
    StreamHandler from get_logger, plus any others) for the duration of the
    `with` block, so nothing it logs reaches the console -- used around
    Colab's real (GPU) runs, where tqdm progress bars must be the only
    thing that renders in the notebook output cell. `logger` still
    propagates to the root logger as usual, so anything routed through
    `file_logging` at the same time is unaffected -- only the direct
    console handler is removed.
    """
    removed = list(logger.handlers)
    for handler in removed:
        logger.removeHandler(handler)
    try:
        yield
    finally:
        for handler in removed:
            logger.addHandler(handler)


@contextmanager
def file_logging(log_path: str | Path):
    """Route anything logged through a plain `logging.getLogger(...)` (with
    no handler of its own — e.g. training/common.py's per-run run_logger)
    to `log_path` instead of nowhere, for the duration of the `with` block.

    This only captures *our own* run-scoped logging (per-step loss, run
    start/end); call `silence_library_noise()` once beforehand to stop
    transformers/datasets/hub noise at the source instead — trying to
    redirect that here too, after the fact, previously missed most of it
    (model loading happens before this context is even entered) and didn't
    compose safely with nesting.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)

    root = logging.getLogger()
    previous_root_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_root_level)
        handler.close()
