"""Load the pretrained LPN checkpoint from Hugging Face Hub.

`clement-bonnet/lpn-2d` is a raw Flax `TrainState` dump (a `config.yaml` +
`state.msgpack` pair, the same shape the original repo's own
src/evaluate_checkpoint.py loads from a WandB artifact directory) -- not a
`transformers`-compatible repo, so there's no `from_pretrained`. This module
mirrors evaluate_checkpoint.py's `instantiate_model` / `instantiate_train_state`
/ `load_model_weights`, swapping the WandB download for
`huggingface_hub.snapshot_download`.

Requires the `lpn` repo (github.com/clement-bonnet/lpn) cloned with its
directory added to `sys.path` (its own code imports itself as
`src.models...`, absolute from its repo root -- not `pip install`-able, see
notebooks/exp4_test_time_tuning.ipynb's bootstrap cell) plus its pinned
jax/flax/optax installed. Everything here is imported lazily so this module
(and the rest of src/lpn_exp/) stays importable without those dependencies
present, matching this repo's existing convention of lazy-importing heavy
deps inside functions (see scripts/run_experiment.py, src/training/common.py).

Note (unverified without a live JAX environment -- see CLAUDE-CODING-SKILL.md's
"Milestone 0" note): the checkpoint's own config.yaml uses stale
`_target_: src_v2.models.utils...` paths from an older internal module
layout. We deliberately never call `hydra.utils.instantiate` on it -- we
read the plain nested fields ourselves and construct the config dataclasses
directly, which is robust to that mismatch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _strip_target(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "_target_"}


def _download_checkpoint_dir(repo_id: str, checkpoint_name: str) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=repo_id, allow_patterns=f"{checkpoint_name}/*")
    return Path(local_dir) / checkpoint_name


def load_pretrained(repo_id: str, checkpoint_name: str):
    """Returns (lpn_model, frozen_params) -- an `LPN` flax Module plus the
    pretrained parameter pytree extracted from the checkpoint's TrainState.

    `frozen_params` is a plain nested dict of jax arrays, rooted at
    {"encoder": {...}, "decoder": {...}} (LPN's two submodule fields) -- the
    shape src/lpn_exp/lora.py's path-finding/merging functions expect.
    """
    import jax
    import optax
    import yaml
    from flax.serialization import from_bytes
    from flax.training.train_state import TrainState

    from src.models.lpn import LPN
    from src.models.transformer import DecoderTransformer, EncoderTransformer
    from src.models.utils import (
        DecoderTransformerConfig,
        EncoderTransformerConfig,
        TransformerLayerConfig,
    )

    ckpt_dir = _download_checkpoint_dir(repo_id, checkpoint_name)
    with open(ckpt_dir / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def _layer_config(d: dict[str, Any]) -> TransformerLayerConfig:
        return TransformerLayerConfig(**_strip_target(d))

    encoder_d = _strip_target(cfg["encoder_transformer"])
    decoder_d = _strip_target(cfg["decoder_transformer"])
    encoder_config = EncoderTransformerConfig(
        **{**encoder_d, "transformer_layer": _layer_config(encoder_d["transformer_layer"])}
    )
    decoder_config = DecoderTransformerConfig(
        **{**decoder_d, "transformer_layer": _layer_config(decoder_d["transformer_layer"])}
    )

    encoder = EncoderTransformer(encoder_config)
    decoder = DecoderTransformer(decoder_config)
    lpn = LPN(encoder=encoder, decoder=decoder)

    # Build an arbitrary-weights TrainState shell to init the params pytree's
    # structure/shapes -- copied from evaluate_checkpoint.py's
    # instantiate_train_state. The optimizer itself is never stepped (we only
    # ever read `.params` after loading); it has to match the *structure* the
    # checkpoint was serialized with, though, for `from_bytes` to deserialize
    # correctly, hence copying their exact optimizer chain rather than using a
    # simpler placeholder.
    key = jax.random.PRNGKey(0)
    grids = jax.random.randint(
        key,
        (1, 3, decoder_config.max_rows, decoder_config.max_cols, 2),
        minval=0,
        maxval=decoder_config.vocab_size,
    )
    shapes = jax.random.randint(
        key,
        (1, 3, 2, 2),
        minval=1,
        maxval=min(decoder_config.max_rows, decoder_config.max_cols) + 1,
    )
    variables = lpn.init(
        key,
        grids,
        shapes,
        dropout_eval=False,
        prior_kl_coeff=0.0,
        pairwise_kl_coeff=0.0,
        mode="mean",
    )
    warmup_steps = 0
    scheduler = optax.warmup_exponential_decay_schedule(
        init_value=0.0,
        peak_value=0.0,
        warmup_steps=warmup_steps,
        transition_steps=1,
        end_value=0.0,
        decay_rate=1.0,
    )
    optimizer = optax.MultiSteps(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(scheduler)), every_k_schedule=1
    )
    train_state = TrainState.create(apply_fn=lpn.apply, tx=optimizer, params=variables["params"])

    with open(ckpt_dir / "state.msgpack", "rb") as f:
        train_state = from_bytes(train_state, f.read())

    return lpn, train_state.params
