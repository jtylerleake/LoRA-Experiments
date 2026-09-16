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

    Applies 4-bit/8-bit bitsandbytes quantization when the registry entry
    calls for it (see config/models.yaml), so larger models fit on a single
    Colab GPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    spec = get_model_spec(name)
    repo_id = spec["repo_id"]
    quantization = spec.get("quantization")

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if quantization in ("4bit", "8bit"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if quantization_config is not None else None,
    )
    if device == "cuda" and quantization_config is None:
        model = model.to(device)
    return model, tokenizer
