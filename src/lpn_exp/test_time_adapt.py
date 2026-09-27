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
    (including the pretrained weights) held fixed. That
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

Performance: each condition's whole per-task computation -- every
leave-one-out round (`jax.vmap`ed, not a Python loop) and, for the LoRA
conditions, every adaptation step (`jax.lax.scan`) -- is one `jax.jit`ed
call. The `build_*_evaluator` factories compile it once per condition and
reuse it for every task (every task has the same array shapes), so only the
first task per condition pays the compile. Running this eagerly, op by op,
left the GPU mostly idle waiting on Python dispatch for a model this small.
Params are passed to the jitted function as arguments rather than closed
over, so they aren't baked into the compiled program as constants.

Random keys are derived exactly as the original sequential per-round loop
did (see `_split_chain`), so these compiled evaluators reproduce that
loop's results for the same input key.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from lpn_exp.lora import find_mlp_kernel_paths, init_lora_params, merge_lora
from lpn_exp.pattern2d_tasks import Pattern2DTask, task_to_arrays

# (task, key) -> (accuracy, pixel_correctness), each averaged over the task's
# leave-one-out rounds.
TaskEvaluator = Callable[[Pattern2DTask, Any], tuple[float, float]]


class LeaveOneOutRounds(NamedTuple):
    """Every leave-one-out round of one task, stacked along a leading round
    axis (one round per pair) so they can be `jax.vmap`ed over. A NamedTuple
    so it's a jax pytree with no registration needed.
    """

    context_grids: Any  # (N, N-1, R, C, 2)
    context_shapes: Any  # (N, N-1, 2, 2)
    query_input: Any  # (N, R, C)
    query_shape: Any  # (N, 2)
    label_grid: Any  # (N, R, C)
    label_shape: Any  # (N, 2)


def _score_predictions(pred_grids, pred_shapes, label_grids, label_shapes):
    """Returns (accuracy, pixel_correctness) as jax scalars, so it can run
    inside a jitted function.
    """
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
    return accuracy, pixel_correctness


def _leave_one_out_rounds(grids, shapes) -> LeaveOneOutRounds:
    """One round per pair in a task, taking its turn as the held-out target
    -- see src/data_utils.py's `make_leave_one_out`.
    """
    from src.data_utils import make_leave_one_out

    return LeaveOneOutRounds(
        context_grids=make_leave_one_out(grids, axis=-4),
        context_shapes=make_leave_one_out(shapes, axis=-3),
        query_input=grids[..., 0],
        query_shape=shapes[:, :, 0],
        label_grid=grids[..., 1],
        label_shape=shapes[:, :, 1],
    )


def _split_chain(key, n: int):
    """The `n` sub-keys a Python loop of `key, sub_key = jax.random.split(key)`
    would produce, in order, as one stacked array -- via `lax.scan`, so it
    traces to a small loop instead of `n` unrolled splits.
    """
    import jax

    def step(key, _):
        key, sub_key = jax.random.split(key)
        return key, sub_key

    _, sub_keys = jax.lax.scan(step, key, None, length=n)
    return sub_keys


def _generate_and_score(model, params, round_: LeaveOneOutRounds, key, mode: str, mode_kwargs: dict):
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


def _as_task_evaluator(jitted, params, max_rows: int, max_cols: int) -> TaskEvaluator:
    def evaluate(task: Pattern2DTask, key) -> tuple[float, float]:
        grids, shapes = task_to_arrays(task, max_rows, max_cols)
        accuracy, pixel_correctness = jitted(params, grids, shapes, key)
        return float(accuracy), float(pixel_correctness)

    return evaluate


def build_mean_or_gradient_ascent_evaluator(
    model, params, max_rows: int, max_cols: int, mode: str, mode_kwargs: dict
) -> TaskEvaluator:
    """Scores one of LPN's own inference modes ("mean" or "gradient_ascent")
    against every pair of a task, unmodified from the paper's own mechanism.
    """
    import jax

    def evaluate(params, grids, shapes, key):
        rounds = _leave_one_out_rounds(grids, shapes)
        round_keys = _split_chain(key, grids.shape[0])
        accuracies, pixel_corrects = jax.vmap(
            lambda round_, round_key: _generate_and_score(
                model, params, round_, round_key, mode, mode_kwargs
            )
        )(rounds, round_keys)
        return accuracies.mean(), pixel_corrects.mean()

    return _as_task_evaluator(jax.jit(evaluate), params, max_rows, max_cols)


def build_lora_ascent_evaluator(
    model,
    frozen_params: dict[str, Any],
    max_rows: int,
    max_cols: int,
    target_modules: tuple[str, ...],
    rank: int,
    scale: float,
    num_steps: int,
    lr: float,
    prior_kl_coeff: float,
) -> TaskEvaluator:
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
    optimizer = optax.sgd(lr)

    def loss_fn(lora_params, frozen_params, context_grids, context_shapes, rng):
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

    grad_fn = jax.value_and_grad(loss_fn)  # w.r.t. lora_params only

    def fit_and_score(frozen_params, round_: LeaveOneOutRounds, keys):
        # keys: (num_steps + 2, ...) -- init, one per step, decode; the same
        # order the original sequential loop split them in.
        lora_params = init_lora_params(frozen_params, target_paths, rank, keys[0])

        def step(carry, step_key):
            lora_params, opt_state = carry
            # "Gradient ascent" on the paper's own likelihood objective is
            # gradient *descent* on this `loss` (cross-entropy + KL) -- same
            # numerical direction, just their terminology for maximizing
            # log-likelihood.
            _loss, grads = grad_fn(
                lora_params, frozen_params, round_.context_grids, round_.context_shapes, step_key
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            return (optax.apply_updates(lora_params, updates), opt_state), None

        (lora_params, _opt_state), _ = jax.lax.scan(
            step, (lora_params, optimizer.init(lora_params)), keys[1:-1]
        )
        patched_params = merge_lora(frozen_params, lora_params, scale)
        return _generate_and_score(model, patched_params, round_, keys[-1], "mean", {})

    def evaluate(frozen_params, grids, shapes, key):
        num_rounds = grids.shape[0]
        rounds = _leave_one_out_rounds(grids, shapes)
        keys = _split_chain(key, num_rounds * (num_steps + 2))
        round_keys = keys.reshape(num_rounds, num_steps + 2, *keys.shape[1:])
        accuracies, pixel_corrects = jax.vmap(fit_and_score, in_axes=(None, 0, 0))(
            frozen_params, rounds, round_keys
        )
        return accuracies.mean(), pixel_corrects.mean()

    return _as_task_evaluator(jax.jit(evaluate), frozen_params, max_rows, max_cols)
