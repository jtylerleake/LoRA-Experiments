"""Functional LoRA for LPN's Flax decoder.

This is the JAX/Flax analogue of `peft.LoraConfig` (used for the PyTorch
experiments elsewhere in this repo, see src/training/lora.py): same
zero-init-at-start, low-rank-additive-update idea, but expressed as pure
pytree math instead of wrapping an `nn.Module` -- PEFT only wraps PyTorch
modules, and Flax modules are immutable (they take their params as a plain
pytree argument to `.apply()`, so "adapting" a Flax model means patching
that pytree, not subclassing anything). See CLAUDE-CODING-SKILL.md for the
fuller writeup of this pattern.

v1 targets only the decoder's MlpBlock Dense kernels (two explicit,
unambiguous `nn.Dense` calls per TransformerLayer -- see
src.models.utils.MlpBlock). Flax's built-in `nn.MultiHeadAttention` bundles
its own query/key/value/out Dense sublayers internally, and their exact
parameter-path names need to be confirmed by inspecting a live params
pytree before an attention target list can be added (documented fast-follow
-- see the plan's "Milestone 0").

`find_decoder_mlp_kernel_paths` and `merge_lora` deliberately avoid any
jax/flax import: they walk a plain nested dict and only ever use `@` and
`+` on the leaves, which works identically for numpy or jax arrays. That's
what makes them unit-testable with plain numpy in this repo's Docker CPU
test path, which has no jax/flax installed (see tests/test_smoke.py) --
real usage passes them the jax-array params pytree loaded by
src/lpn_exp/checkpoint.py, and the same code works unchanged.
"""
from __future__ import annotations

from typing import Any

PathTuple = tuple[str, ...]


def find_decoder_mlp_kernel_paths(params: dict[str, Any]) -> list[PathTuple]:
    """Every `kernel` leaf under a decoder `MlpBlock` submodule, found by
    walking the params pytree rather than hardcoding Flax's auto-generated
    submodule names (e.g. `TransformerLayer_0`) -- those names come from
    instantiation order and shouldn't be guessed.
    """
    paths: list[PathTuple] = []

    def _walk(node: Any, path: PathTuple) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, path + (key,))
        elif path and path[-1] == "kernel" and "decoder" in path and "MlpBlock_0" in path:
            paths.append(path)

    _walk(params, ())
    return paths


def _get(node: dict[str, Any], path: PathTuple) -> Any:
    for key in path:
        node = node[key]
    return node


def _set_copy(node: dict[str, Any], path: PathTuple, value: Any) -> dict[str, Any]:
    """Returns a copy of `node` with `path` replaced by `value`, sharing
    structure with the original everywhere else (only the dicts along
    `path` are shallow-copied).
    """
    if not path:
        return value
    key, rest = path[0], path[1:]
    copy = dict(node)
    copy[key] = _set_copy(node[key], rest, value)
    return copy


def init_lora_params(
    params: dict[str, Any], target_paths: list[PathTuple], rank: int, key
) -> dict[PathTuple, dict[str, Any]]:
    """A fresh, zero-initialized LoRA `(a, b)` pair per target kernel path.

    `b` is zero-initialized -- this repo's usual identity-at-init convention
    (see src/training/adapters.py's bottleneck adapter) -- so the adapted
    decoder is exactly the frozen pretrained one before any test-time
    gradient-ascent steps have run.
    """
    import jax
    import jax.numpy as jnp

    lora_params = {}
    for path in target_paths:
        kernel = _get(params, path)
        in_dim, out_dim = kernel.shape
        key, a_key = jax.random.split(key)
        a = jax.random.normal(a_key, (in_dim, rank)) * (1.0 / in_dim) ** 0.5
        b = jnp.zeros((rank, out_dim))
        lora_params[path] = {"a": a, "b": b}
    return lora_params


def merge_lora(
    frozen_params: dict[str, Any], lora_params: dict[PathTuple, dict[str, Any]], scale: float
) -> dict[str, Any]:
    """Returns a copy of `frozen_params` with `scale * a @ b` added onto
    each targeted kernel leaf. With every `b` zero-initialized (see
    `init_lora_params`), this is a no-op -- the returned params equal
    `frozen_params` exactly.
    """
    patched = frozen_params
    for path, pair in lora_params.items():
        delta = scale * (pair["a"] @ pair["b"])
        patched = _set_copy(patched, path, _get(patched, path) + delta)
    return patched
