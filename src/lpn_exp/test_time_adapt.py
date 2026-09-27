"""Per-task test-time adaptation and scoring, for every condition: `mean`,
`gradient_ascent`, and the three LoRA-ascent variants
(`lora_ascent_decoder`, `lora_ascent_encoder`, `lora_ascent_encoder_decoder`
-- see lpn_exp.lora.LORA_CONDITION_TARGETS).

Every pair in a task takes a turn as the held-out target (the rest as
context), matching pattern2d_tasks.py's evaluation convention. Evaluators
return one result per round, so rounds whose held-out pair duplicates a
context pair can be reported separately (pattern2d_tasks.clean_round_mask).

`mean` and `gradient_ascent` are LPN's own inference modes, run through its
public `LPN.generate_output` unmodified.

The LoRA-ascent conditions are built to differ from `gradient_ascent` in
exactly one way -- *what* gets optimized (a low-rank update to MLP weights
instead of the 2D latent). Everything else mirrors lpn's
`_get_gradient_ascent_context`:
  - Objective: the decoder log-likelihood of every context pair given one
    shared context latent -- the mean of the context pairs' latents, sampled
    from the encoder's posterior -- summed over the pairs
    (`_context_log_prob`). With an encoder adapter, the latent itself moves
    too, since it's re-encoded through the patched encoder.
  - Optimizer: SGD with gradients clipped to global norm 1.0, ascending the
    log-likelihood.
  - Step selection: every candidate is scored on that objective -- the
    starting point (the unadapted model, since LoRA's `b` starts at zero)
    and each step's result -- and the best is kept, not the last. So, like
    gradient ascent, it can never end worse on its own objective than
    where it started.
The latent-sampling noise is fixed per round, with the same key split
`LPN.generate_output` uses, so the objective is deterministic across steps
(as gradient ascent's is), and the final decode uses exactly the latent the
objective scored. Every condition gets the same per-round keys for a given
task key, so with no adaptation (b=0) a LoRA condition reproduces `mean`
exactly, and gradient ascent starts from the same latent.

Performance: each condition's whole computation for a batch of tasks --
every task (`jax.vmap`), every round (`jax.vmap`), and every adaptation step
(`jax.lax.scan`) -- is one `jax.jit`ed call, compiled once per condition and
reused for every batch. A short final batch is padded with copies of its
last task so every call has the same shape (and so the same compiled
program); the padding's results are dropped. Params are passed as arguments
rather than closed over, so they aren't baked into the compiled program as
constants.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from lpn_exp.lora import LORA_CONDITION_TARGETS, find_mlp_kernel_paths, init_lora_params, merge_lora
from lpn_exp.pattern2d_tasks import Pattern2DTask, task_to_arrays

# (tasks, one key per task) -> for each task, (per-round accuracy, per-round
# pixel_correctness), one entry per leave-one-out round in the order
# pattern2d_tasks.clean_round_mask uses. At most `batch_size` tasks per call.
BatchEvaluator = Callable[[list[Pattern2DTask], list[Any]], list[tuple[list[float], list[float]]]]


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


def _context_log_prob(lpn, pairs, grid_shapes, key):
    """Gradient ascent's objective, as a Flax `method=` function: the decoder
    log-likelihood of every context pair given one shared context latent.

    Mirrors `LPN.generate_output` up to its search step (encode the pairs,
    sample latents from the posterior with the same `key` split, take their
    mean -- `_prepare_latents_before_search`'s default starting point) and
    then `_get_gradient_ascent_context`'s `log_probs_fn` (repeat that latent
    per pair, decode with teacher forcing, `_compute_log_probs` summed over
    the pairs). Batch of one: returns shape (1,).
    """
    import jax

    latents_mu, latents_logvar = lpn.encoder(pairs, grid_shapes, True)
    if latents_logvar is not None:
        _, key_latents = jax.random.split(key)  # the split generate_output does
        latents, *_ = lpn._sample_latents(latents_mu, latents_logvar, key_latents)
    else:
        latents = latents_mu
    input_seq, output_seq = lpn._flatten_input_output_for_decoding(pairs, grid_shapes)
    context = latents.mean(axis=-2)[..., None, :].repeat(output_seq.shape[-2], axis=-2)
    row_logits, col_logits, grid_logits = lpn.decoder(input_seq, output_seq, context, dropout_eval=True)
    return lpn._compute_log_probs(row_logits, col_logits, grid_logits, output_seq)


def _as_batch_evaluator(evaluate_one, params, max_rows: int, max_cols: int, batch_size: int) -> BatchEvaluator:
    """Wraps a single-task `evaluate_one(params, grids, shapes, key)` into a
    jitted, task-vmapped evaluator over batches of up to `batch_size` tasks.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    jitted = jax.jit(jax.vmap(evaluate_one, in_axes=(None, 0, 0, 0)))

    def evaluate(tasks: list[Pattern2DTask], keys: list[Any]) -> list[tuple[list[float], list[float]]]:
        if not 0 < len(tasks) <= batch_size or len(keys) != len(tasks):
            raise ValueError(f"Need 1-{batch_size} tasks with one key each; got {len(tasks)} tasks, {len(keys)} keys")
        arrays = [task_to_arrays(task, max_rows, max_cols) for task in tasks]
        padding = batch_size - len(tasks)  # repeat the last task: one shape, one compile
        grids = jnp.stack([g for g, _ in arrays] + [arrays[-1][0]] * padding)
        shapes = jnp.stack([s for _, s in arrays] + [arrays[-1][1]] * padding)
        batch_keys = jnp.stack(list(keys) + [keys[-1]] * padding)
        accuracies, pixel_corrects = jitted(params, grids, shapes, batch_keys)
        accuracies, pixel_corrects = np.asarray(accuracies), np.asarray(pixel_corrects)
        return [
            ([float(a) for a in accuracies[i]], [float(p) for p in pixel_corrects[i]])
            for i in range(len(tasks))
        ]

    return evaluate


def build_mean_or_gradient_ascent_evaluator(
    model, params, max_rows: int, max_cols: int, mode: str, mode_kwargs: dict, batch_size: int
) -> BatchEvaluator:
    """Scores one of LPN's own inference modes ("mean" or "gradient_ascent")
    against every pair of a task, unmodified from the paper's own mechanism.
    """
    import jax

    def evaluate(params, grids, shapes, key):
        rounds = _leave_one_out_rounds(grids, shapes)
        round_keys = _split_chain(key, grids.shape[0])
        return jax.vmap(
            lambda round_, round_key: _generate_and_score(
                model, params, round_, round_key, mode, mode_kwargs
            )
        )(rounds, round_keys)

    return _as_batch_evaluator(evaluate, params, max_rows, max_cols, batch_size)


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
    batch_size: int,
) -> BatchEvaluator:
    """Our test-time adaptation conditions: for each held-out pair, fit a
    fresh per-round LoRA adapter on the MLP kernels of `target_modules`
    (`("decoder",)`, `("encoder",)`, or `("encoder", "decoder")` -- see
    lpn_exp.lora.LORA_CONDITION_TARGETS) by gradient ascent on the context
    pairs' log-likelihood, keep the best step, then decode the held-out
    query through the LoRA-patched model. See the module docstring for how
    each piece mirrors the paper's gradient-ascent condition. Whichever
    submodule isn't targeted runs with its pretrained weights unchanged.
    """
    import jax
    import jax.numpy as jnp
    import optax

    target_paths = find_mlp_kernel_paths(frozen_params, target_modules)
    # Same optimizer chain as lpn's _get_gradient_ascent_context (optimizer="sgd").
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=lr))

    def log_prob_fn(lora_params, frozen_params, context_grids, context_shapes, key):
        patched = merge_lora(frozen_params, lora_params, scale)
        log_probs = model.apply(
            {"params": patched},
            context_grids[None],  # batch of one -- see _generate_and_score
            context_shapes[None],
            key,
            method=_context_log_prob,
        )
        return log_probs[0]

    value_and_grad_fn = jax.value_and_grad(log_prob_fn)  # w.r.t. lora_params only

    def fit_and_score(frozen_params, round_: LeaveOneOutRounds, key):
        # `key` is this round's latent-sampling/decode key -- the same one the
        # mean/gradient_ascent evaluators give this round -- so at step 0 (b=0)
        # this is exactly the `mean` condition. The init key is derived from it.
        init_key = jax.random.fold_in(key, 1)
        args = (frozen_params, round_.context_grids, round_.context_shapes, key)
        lora_params = init_lora_params(frozen_params, target_paths, rank, init_key)

        def keep_better(best, candidate):
            (best_params, best_log_prob), (params, log_prob) = best, candidate
            better = log_prob > best_log_prob  # ties keep the earlier step
            best_params = jax.tree_util.tree_map(
                lambda b, p: jnp.where(better, p, b), best_params, params
            )
            return best_params, jnp.where(better, log_prob, best_log_prob)

        def step(carry, _):
            lora_params, opt_state, best = carry
            # Scores the current params (step 0: the unadapted model) and
            # gets the gradient for the next step in one pass.
            log_prob, grads = value_and_grad_fn(lora_params, *args)
            best = keep_better(best, (lora_params, log_prob))
            # Ascent: step along +grad, i.e. feed the optimizer -grad.
            updates, opt_state = optimizer.update(jax.tree_util.tree_map(jnp.negative, grads), opt_state)
            return (optax.apply_updates(lora_params, updates), opt_state, best), None

        init_best = (lora_params, jnp.array(-jnp.inf))
        (lora_params, _opt_state, best), _ = jax.lax.scan(
            step, (lora_params, optimizer.init(lora_params), init_best), None, length=num_steps
        )
        best_params, _ = keep_better(best, (lora_params, log_prob_fn(lora_params, *args)))
        patched_params = merge_lora(frozen_params, best_params, scale)
        return _generate_and_score(model, patched_params, round_, key, "mean", {})

    def evaluate(frozen_params, grids, shapes, key):
        rounds = _leave_one_out_rounds(grids, shapes)
        round_keys = _split_chain(key, grids.shape[0])  # same per-round keys as mean/GA
        return jax.vmap(fit_and_score, in_axes=(None, 0, 0))(frozen_params, rounds, round_keys)

    return _as_batch_evaluator(evaluate, frozen_params, max_rows, max_cols, batch_size)


def build_condition_evaluator(model, params, max_rows: int, max_cols: int, condition: str, config, lr=None):
    """The evaluator for one of the config's conditions, using `lr` if given
    (the learning-rate sweep) or else the config's own
    `learning_rates[condition]`. `config` is an Exp4Config.
    """
    if condition == "mean":
        return build_mean_or_gradient_ascent_evaluator(
            model, params, max_rows, max_cols, "mean", {}, config.batch_size
        )
    lr = config.learning_rates[condition] if lr is None else lr
    if condition == "gradient_ascent":
        mode_kwargs = {"num_steps": config.gradient_ascent.num_steps, "lr": lr}
        return build_mean_or_gradient_ascent_evaluator(
            model, params, max_rows, max_cols, "gradient_ascent", mode_kwargs, config.batch_size
        )
    if condition in LORA_CONDITION_TARGETS:
        return build_lora_ascent_evaluator(
            model,
            params,
            max_rows,
            max_cols,
            target_modules=LORA_CONDITION_TARGETS[condition],
            rank=config.lora.rank,
            scale=config.lora.scale,
            num_steps=config.lora.num_steps,
            lr=lr,
            batch_size=config.batch_size,
        )
    raise ValueError(f"Unknown condition: {condition!r}")
