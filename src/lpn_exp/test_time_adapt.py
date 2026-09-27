"""Per-task test-time adaptation and scoring, for every condition: `mean`,
`gradient_ascent`, and the three LoRA-ascent variants
(`lora_ascent_decoder`, `lora_ascent_encoder`, `lora_ascent_encoder_decoder`
-- see lpn_exp.lora.LORA_CONDITION_TARGETS).

Reuses LPN's own two public entry points rather than reimplementing its
internals:
  - `LPN.__call__` (its training-time forward pass): given a task's demo
    pairs, returns the leave-one-out reconstruction loss/metrics for a given
    `mode`. This is exactly "the leave-one-out demo-pair objective" every
    LoRA-ascent condition gradient-ascends -- called with only the LoRA
    `(a, b)` pytree as the differentiated argument, everything else
    (including the pretrained weights) closed over as a constant. That
    forward pass runs the encoder (pairs -> latents) *and* the decoder
    (latent + input -> output), so gradients reach LoRA factors in either
    submodule with no special-casing.
  - `LPN.generate_output` (its inference entrypoint, `method=model.
    generate_output`): given context pairs + one query input, decodes the
    predicted output grid under a given `mode`. Used to score every
    condition. For the LoRA conditions it's called with `mode="mean"`
    against the LoRA-patched params: an encoder adapter changes the latent
    the context pairs are encoded to, a decoder adapter changes how that
    latent is decoded, and the encoder+decoder variant does both.

Every pair in a task takes a turn as the held-out target (the rest as
context), matching pattern2d_tasks.py's evaluation convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lpn_exp.lora import find_mlp_kernel_paths, init_lora_params, merge_lora
from lpn_exp.pattern2d_tasks import Pattern2DTask, task_to_arrays


@dataclass
class LeaveOneOutRound:
    context_grids: Any
    context_shapes: Any
    query_input: Any
    query_shape: Any
    label_grid: Any
    label_shape: Any


def _score_predictions(pred_grids, pred_shapes, label_grids, label_shapes) -> tuple[float, float]:
    import jax.numpy as jnp

    correct_shapes = jnp.all(pred_shapes == label_shapes, axis=-1)
    row_mask = jnp.arange(pred_grids.shape[-2]) < label_shapes[..., :1]
    col_mask = jnp.arange(pred_grids.shape[-1]) < label_shapes[..., 1:]
    mask = row_mask[..., None] & col_mask[..., None, :]
    grids_equal = pred_grids == label_grids
    pixels_equal = jnp.where(mask & correct_shapes[..., None, None], grids_equal, False)
    num_pixels = label_shapes.prod(axis=-1)
    pixel_correctness = (pixels_equal.sum(axis=(-1, -2)) / num_pixels).mean()
    accuracy = (pixels_equal.sum(axis=(-1, -2)) == num_pixels).mean()
    return float(accuracy), float(pixel_correctness)


def _leave_one_out_rounds(grids, shapes) -> list[LeaveOneOutRound]:
    """One round per pair in a task, taking its turn as the held-out target
    -- see src/data_utils.py's `make_leave_one_out`.
    """
    from src.data_utils import make_leave_one_out

    leave_one_out_grids = make_leave_one_out(grids, axis=-4)  # (N, N-1, R, C, 2)
    leave_one_out_shapes = make_leave_one_out(shapes, axis=-3)  # (N, N-1, 2, 2)
    return [
        LeaveOneOutRound(
            context_grids=leave_one_out_grids[i],
            context_shapes=leave_one_out_shapes[i],
            query_input=grids[i, ..., 0],
            query_shape=shapes[i, :, 0],
            label_grid=grids[i, ..., 1],
            label_shape=shapes[i, :, 1],
        )
        for i in range(grids.shape[0])
    ]


def _generate_and_score(
    model, params, round_: LeaveOneOutRound, key, mode: str, mode_kwargs: dict
) -> tuple[float, float]:
    """Decodes one round's held-out query via `LPN.generate_output` and scores it.

    `generate_output` expects a leading batch axis on every argument (the
    decoder asserts it -- see lpn's evaluate_checkpoint.py, which always calls
    it batched), so each single round is passed as a batch of one -- except
    `key`, which it `jax.random.split`s as a single key. Keyword
    arguments, not positional: its signature is (pairs, grid_shapes, input,
    input_grid_shape, key, ...), i.e. the context pairs come *first*.
    """
    output_grid, output_shape, _info = model.apply(
        {"params": params},
        pairs=round_.context_grids[None],
        grid_shapes=round_.context_shapes[None],
        input=round_.query_input[None],
        input_grid_shape=round_.query_shape[None],
        key=key,  # a single key, not batched -- generate_output splits it directly
        dropout_eval=True,
        mode=mode,
        **mode_kwargs,
        method=model.generate_output,
    )
    return _score_predictions(
        output_grid, output_shape, round_.label_grid[None], round_.label_shape[None]
    )


def evaluate_condition_mean_or_gradient_ascent(
    model,
    params,
    task: Pattern2DTask,
    max_rows: int,
    max_cols: int,
    mode: str,
    mode_kwargs: dict,
    key,
) -> tuple[float, float]:
    """Scores one of LPN's own inference modes ("mean" or "gradient_ascent")
    against every pair of `task`, unmodified from the paper's own mechanism.
    """
    import jax

    grids, shapes = task_to_arrays(task, max_rows, max_cols)
    accuracies, pixel_corrects = [], []
    for round_ in _leave_one_out_rounds(grids, shapes):
        key, sub_key = jax.random.split(key)
        accuracy, pixel_correctness = _generate_and_score(
            model, params, round_, sub_key, mode, mode_kwargs
        )
        accuracies.append(accuracy)
        pixel_corrects.append(pixel_correctness)
    return sum(accuracies) / len(accuracies), sum(pixel_corrects) / len(pixel_corrects)


def evaluate_condition_lora_ascent(
    model,
    frozen_params: dict[str, Any],
    task: Pattern2DTask,
    max_rows: int,
    max_cols: int,
    target_modules: tuple[str, ...],
    rank: int,
    scale: float,
    num_steps: int,
    lr: float,
    prior_kl_coeff: float,
    key,
) -> tuple[float, float]:
    """Our test-time adaptation conditions: for each held-out pair, fit a
    fresh per-round LoRA adapter on the MLP kernels of `target_modules`
    (`("decoder",)`, `("encoder",)`, or `("encoder", "decoder")` -- see
    lpn_exp.lora.LORA_CONDITION_TARGETS) against the other pairs'
    reconstruction objective, then encode the context and decode the
    held-out query through the LoRA-patched model. Whichever submodule
    isn't targeted runs with its pretrained weights unchanged.
    """
    import jax
    import optax

    target_paths = find_mlp_kernel_paths(frozen_params, target_modules)
    grids, shapes = task_to_arrays(task, max_rows, max_cols)
    accuracies, pixel_corrects = [], []

    def loss_fn(lora_params, context_grids, context_shapes, rng):
        patched = merge_lora(frozen_params, lora_params, scale)
        loss, _metrics = model.apply(
            {"params": patched},
            context_grids[None],  # batch of one -- see _generate_and_score
            context_shapes[None],
            dropout_eval=True,
            mode="mean",
            prior_kl_coeff=prior_kl_coeff,
            pairwise_kl_coeff=None,
            rngs={"latents": rng},
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn)

    for round_ in _leave_one_out_rounds(grids, shapes):
        key, init_key = jax.random.split(key)
        lora_params = init_lora_params(frozen_params, target_paths, rank, init_key)
        optimizer = optax.sgd(lr)
        opt_state = optimizer.init(lora_params)

        for _ in range(num_steps):
            key, step_key = jax.random.split(key)
            # "Gradient ascent" on the paper's own likelihood objective is
            # gradient *descent* on this `loss` (cross-entropy + KL) -- same
            # numerical direction, just their terminology for maximizing
            # log-likelihood.
            _loss, grads = grad_fn(
                lora_params, round_.context_grids, round_.context_shapes, step_key
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            lora_params = optax.apply_updates(lora_params, updates)

        patched_params = merge_lora(frozen_params, lora_params, scale)
        key, sub_key = jax.random.split(key)
        accuracy, pixel_correctness = _generate_and_score(
            model, patched_params, round_, sub_key, "mean", {}
        )
        accuracies.append(accuracy)
        pixel_corrects.append(pixel_correctness)
    return sum(accuracies) / len(accuracies), sum(pixel_corrects) / len(pixel_corrects)
