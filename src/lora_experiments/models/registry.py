"""Resolves a friendly model name (from config/models.yaml) to a loaded
HF model + tokenizer, applying the right quantization for the model's size.

Real weight loading only happens where there's a GPU to put them on (Colab)
or in a CPU dry run against a tiny model; it is never invoked as part of
local Docker dev/lint/test beyond that dry run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_MODELS_YAML = Path(__file__).parent.parent / "config" / "models.yaml"


def _load_registry() -> dict[str, Any]:
    with open(_MODELS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_spec(name: str) -> dict[str, Any]:
    registry = _load_registry()
    if name not in registry:
        raise KeyError(f"Unknown model {name!r}. Known models: {sorted(registry)}")
    return registry[name]


def load_model_and_tokenizer(name: str, device: str = "cuda"):
    """Load a registered model + tokenizer.

    Not implemented yet — real loading (transformers AutoModelForCausalLM /
    AutoTokenizer, plus a bitsandbytes quantization config when the spec
    calls for it) lands alongside the first real experiment run, once
    Docker/Colab plumbing is verified end to end with a dry run.
    """
    spec = get_model_spec(name)
    raise NotImplementedError(
        f"Model loading for {name!r} ({spec['repo_id']}) not implemented yet."
    )
