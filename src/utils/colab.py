"""Helpers used only when running inside a Colab notebook.

Every function here is a no-op / raises outside of Colab so the same
`lora_experiments` package stays importable in the Docker CPU dev image
without pulling in Colab-only modules at import time.
"""
from __future__ import annotations

DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/lora_experiments_outputs"


def mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
    except ImportError as e:
        raise RuntimeError("mount_drive() is only available inside a Colab runtime.") from e
    drive.mount("/content/drive")


def run_output_dir(experiment: str, run_id: str) -> str:
    return f"{DRIVE_OUTPUT_ROOT}/{experiment}/{run_id}"
