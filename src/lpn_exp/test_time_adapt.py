"""Per-task test-time adaptation and scoring, for all three conditions
(`mean`, `gradient_ascent`, `lora_ascent`).

Reuses LPN's own two public entry points rather than reimplementing its
internals:
  - `LPN.__call__` (its training-time forward pass): given a task's demo
    pairs, returns the leave-one-out reconstruction loss/metrics for a given
    `mode`. This is exactly "the leave-one-out demo-pair objective" our
    `lora_ascent` condition gradient-ascends -- called with only the LoRA
    `(a, b)` pytree as the differentiated argument, everything else
    (including the pretrained weights) closed over as a constant.
  - `LPN.generate_output` (its inference entrypoint, `method=model.
    generate_output`): given context pairs + one query input, decodes the
    predicted output grid under a given `mode`. Used to score every
    condition, including `lora_ascent` -- since LoRA only patches decoder
    MLP kernels, calling this with `mode="mean"` against the LoRA-patched
    params reuses the exact same encoder (untouched) and just decodes
    through the adapted decoder.

Every pair in a task takes a turn as the held-out target (the rest as
context), matching pattern2d_tasks.py's evaluation convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lpn_exp.lora import find_decoder_mlp_kernel_paths, init_lora_params, merge_lora
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
        output_grid, output_shape, _info = model.apply(
            {"params": params},
            round_.query_input,
            round_.query_shape,
            round_.context_grids,
            round_.context_shapes,
            sub_key,
            dropout_eval=True,
            mode=mode,
            **mode_kwargs,
            method=model.generate_output,
        )
        accuracy, pixel_correctness = _score_predictions(
            output_grid[None], output_shape[None], round_.label_grid[None], round_.label_shape[None]
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
    rank: int,
    scale: float,
    num_steps: int,
    lr: float,
    prior_kl_coeff: float,
    key,
) -> tuple[float, float]:
    """Our test-time adaptation condition: for each held-out pair, fit a
    fresh per-round LoRA adapter on the decoder's MLP kernels against the
    other pairs' reconstruction objective, then decode the held-out query
    through the LoRA-patched decoder (encoder and z-computation untouched).
    """
    import jax
    import optax

    target_paths = find_decoder_mlp_kernel_paths(frozen_params)
    grids, shapes = task_to_arrays(task, max_rows, max_cols)
    accuracies, pixel_corrects = [], []

    def loss_fn(lora_params, context_grids, context_shapes, rng):
        patched = merge_lora(frozen_params, lora_params, scale)
        loss, _metrics = model.apply(
            {"params": patched},
            context_grids,
            context_shapes,
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
        output_grid, output_shape, _info = model.apply(
            {"params": patched_params},
            round_.query_input,
            round_.query_shape,
            round_.context_grids,
            round_.context_shapes,
            sub_key,
            dropout_eval=True,
            mode="mean",
            method=model.generate_output,
        )
        accuracy, pixel_correctness = _score_predictions(
            output_grid[None], output_shape[None], round_.label_grid[None], round_.label_shape[None]
        )
        accuracies.append(accuracy)
        pixel_corrects.append(pixel_correctness)
    return sum(accuracies) / len(accuracies), sum(pixel_corrects) / len(pixel_corrects)
