from __future__ import annotations

import logging
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


@contextmanager
def file_logging(log_path: str | Path):
    """Route verbose logging (transformers/datasets internals, plus anything
    logged through a plain `logging.getLogger(...)` with no handler of its
    own) to `log_path` instead of the console, for the duration of the
    `with` block.

    Used to keep a long Colab training run's console output down to just the
    compact progress bars in training/common.py, while every step's detail
    is still captured to a file under the run's output directory for later
    inspection.
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

    # transformers manages its own StreamHandler + verbosity separately from
    # the root logger, so it needs pointing at our file handler explicitly.
    hf_logging = None
    try:
        import transformers.utils.logging as hf_logging

        hf_logging.disable_default_handler()
        hf_logging.add_handler(handler)
        hf_logging.set_verbosity_info()
    except ImportError:
        pass

    # datasets just uses plain `logging.getLogger(...)` under the hood (no
    # separate default handler to redirect), so it already propagates to the
    # root handler above — only its verbosity and progress bars need setting.
    try:
        import datasets.utils.logging as ds_logging

        ds_logging.set_verbosity_info()
        ds_logging.disable_progress_bar()
    except ImportError:
        pass

    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_root_level)
        if hf_logging is not None:
            hf_logging.remove_handler(handler)
            hf_logging.enable_default_handler()
        handler.close()
