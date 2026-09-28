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
Optimization trajectories (`record_trajectory`, used by run_exp4.py): for
every adaptation condition, per round and per step 0..num_steps, the
objective (`traj_context_log_prob` -- the same quantity for GA and LoRA, so
directly comparable), the query score of the best candidate so far
(`traj_accuracy`, `traj_pixel_correctness` -- what the method would answer if
stopped at that step), and which step was finally kept (`best_step`). Step 0
is the unadapted model, i.e. the `mean` condition. GA's trajectory is
recorded by re-running lpn's search step by step (see
`build_mean_or_gradient_ascent_evaluator`); its headline numbers still come
from lpn's own `generate_output`.

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

# (tasks, one key per task) -> one dict per task of per-round lists (one entry
# per leave-one-out round, in the order pattern2d_tasks.clean_round_mask uses):
# always `round_accuracy` and `round_pixel_correctness`; with trajectory
# recording, also `traj_context_log_prob`, `traj_accuracy`,
# `traj_pixel_correctness` (each round a list over steps 0..num_steps) and
# `best_step`. At most `batch_size` tasks per call.
BatchEvaluator = Callable[[list[Pattern2DTask], list[Any]], list[dict[str, list]]]


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


def _start_latent(lpn, pairs, grid_shapes, key):
    """`LPN.generate_output`'s steps before its search, as a Flax `method=`
    function: encode the context pairs, sample their latents from the
    posterior with the same `key` split, and take their mean (the default
    starting point of `_get_gradient_ascent_context`). Batch of one: (1, H).
    """
    import jax

    latents_mu, latents_logvar = lpn.encoder(pairs, grid_shapes, True)
    if latents_logvar is not None:
        _, key_latents = jax.random.split(key)  # the split generate_output does
        latents, *_ = lpn._sample_latents(latents_mu, latents_logvar, key_latents)
    else:
        latents = latents_mu
    return latents.mean(axis=-2)


def _log_prob_given_latent(lpn, pairs, grid_shapes, latent):
    """`_get_gradient_ascent_context`'s `log_probs_fn`, as a Flax `method=`
    function: repeat one latent per context pair, decode with teacher forcing,
    and sum `_compute_log_probs` over the pairs. Batch of one: shape (1,).
    """
    input_seq, output_seq = lpn._flatten_input_output_for_decoding(pairs, grid_shapes)
    context = latent[..., None, :].repeat(output_seq.shape[-2], axis=-2)
    row_logits, col_logits, grid_logits = lpn.decoder(input_seq, output_seq, context, dropout_eval=True)
    return lpn._compute_log_probs(row_logits, col_logits, grid_logits, output_seq)


def _context_log_prob(lpn, pairs, grid_shapes, key):
    """Gradient ascent's objective at its starting latent: the decoder
    log-likelihood of every context pair given the mean context latent.
    The LoRA conditions ascend this through the patched weights.
    """
    return _log_prob_given_latent(lpn, pairs, grid_shapes, _start_latent(lpn, pairs, grid_shapes, key))


def _decode_from_latent(lpn, latent, input, input_grid_shape):
    """`LPN.generate_output`'s final step: decode the query from a context latent."""
    return lpn._generate_output_from_context(latent, input, input_grid_shape, True)


def _ascend(value_and_grad_fn, init_params, optimizer, num_steps: int, decode_best=None):
    """Gradient ascent on `value_and_grad_fn` (-> (log_prob, grads)), scoring
    every candidate -- the starting point and the result of each of
    `num_steps` steps -- and keeping the best (ties go to the earlier step),
    exactly like lpn's `_get_gradient_ascent_context` selects its latent.

    Returns (best_params, trajectory). The trajectory always has `best_step`
    and `context_log_prob` (one per candidate, shape (num_steps + 1,)); with
    `decode_best(best_params) -> (accuracy, pixel_correctness)` it also has
    `accuracy`/`pixel_correctness` per candidate -- the query score of the
    answer the method would give if stopped after that step (the best
    candidate so far). Candidate 0 is the unadapted starting point.
    """
    import jax
    import jax.numpy as jnp
    import optax

    def step(carry, index):
        params, opt_state, best_params, best_log_prob, best_step = carry
        log_prob, grads = value_and_grad_fn(params)
        better = log_prob > best_log_prob
        best_params = jax.tree_util.tree_map(lambda b, p: jnp.where(better, p, b), best_params, params)
        best_log_prob = jnp.where(better, log_prob, best_log_prob)
        best_step = jnp.where(better, index, best_step)
        out = {"context_log_prob": log_prob}
        if decode_best is not None:
            out["accuracy"], out["pixel_correctness"] = decode_best(best_params)
        # Ascent: step along +grad, i.e. feed the optimizer -grad. (The update
        # after the last candidate is computed but never scored.)
        updates, opt_state = optimizer.update(jax.tree_util.tree_map(jnp.negative, grads), opt_state)
        return (optax.apply_updates(params, updates), opt_state, best_params, best_log_prob, best_step), out

    init = (init_params, optimizer.init(init_params), init_params, jnp.array(-jnp.inf), jnp.array(0))
    (_, _, best_params, _, best_step), trajectory = jax.lax.scan(step, init, jnp.arange(num_steps + 1))
    return best_params, {"best_step": best_step, **trajectory}


def _gradient_ascent_optimizer(lr: float):
    """lpn's _get_gradient_ascent_context optimizer chain (optimizer="sgd")."""
    import optax

    return optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=lr))


def _prefix_trajectory(trajectory: dict) -> dict:
    return {f"traj_{name}": value for name, value in trajectory.items() if name != "best_step"} | {
        "best_step": trajectory["best_step"]
    }


def _as_batch_evaluator(evaluate_one, params, max_rows: int, max_cols: int, batch_size: int) -> BatchEvaluator:
    """Wraps a single-task `evaluate_one(params, grids, shapes, key) -> {name:
    per-round array}` into a jitted, task-vmapped evaluator over batches of
    up to `batch_size` tasks, returning one {name: list} dict per task.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    jitted = jax.jit(jax.vmap(evaluate_one, in_axes=(None, 0, 0, 0)))

    def evaluate(tasks: list[Pattern2DTask], keys: list[Any]) -> list[dict[str, list]]:
        if not 0 < len(tasks) <= batch_size or len(keys) != len(tasks):
            raise ValueError(f"Need 1-{batch_size} tasks with one key each; got {len(tasks)} tasks, {len(keys)} keys")
        arrays = [task_to_arrays(task, max_rows, max_cols) for task in tasks]
        padding = batch_size - len(tasks)  # repeat the last task: one shape, one compile
        grids = jnp.stack([g for g, _ in arrays] + [arrays[-1][0]] * padding)
        shapes = jnp.stack([s for _, s in arrays] + [arrays[-1][1]] * padding)
        batch_keys = jnp.stack(list(keys) + [keys[-1]] * padding)
        outputs = {name: np.asarray(value) for name, value in jitted(params, grids, shapes, batch_keys).items()}
        return [{name: value[i].tolist() for name, value in outputs.items()} for i in range(len(tasks))]

    return evaluate


def build_mean_or_gradient_ascent_evaluator(
    model,
    params,
    max_rows: int,
    max_cols: int,
    mode: str,
    mode_kwargs: dict,
    batch_size: int,
    record_trajectory: bool = False,
) -> BatchEvaluator:
    """Scores one of LPN's own inference modes ("mean" or "gradient_ascent")
    against every pair of a task, unmodified from the paper's own mechanism
    -- `round_accuracy`/`round_pixel_correctness` always come from
    `LPN.generate_output` itself.

    With `record_trajectory` (gradient_ascent only), also records its
    per-step trajectory (see `_ascend`) by re-running the same search step
    by step outside lpn -- same start latent, objective, optimizer and
    best-candidate rule -- since `generate_output` only returns the final
    latent. The trajectory's last step reproduces the headline result.
    """
    import jax

    if record_trajectory and mode != "gradient_ascent":
        raise ValueError("Only gradient_ascent has a trajectory to record")

    def trajectory_one(params, round_: LeaveOneOutRounds, key):
        pairs, shapes = round_.context_grids[None], round_.context_shapes[None]
        start = model.apply({"params": params}, pairs, shapes, key, method=_start_latent)

        def value_and_grad_fn(latent):
            return jax.value_and_grad(
                lambda z: model.apply({"params": params}, pairs, shapes, z, method=_log_prob_given_latent)[0]
            )(latent)

        def decode_best(latent):
            grid, shape = model.apply(
                {"params": params},
                latent,
                round_.query_input[None],
                round_.query_shape[None],
                method=_decode_from_latent,
            )
            return _score_predictions(grid, shape, round_.label_grid[None], round_.label_shape[None])

        _, trajectory = _ascend(
            value_and_grad_fn,
            start,
            _gradient_ascent_optimizer(mode_kwargs["lr"]),
            mode_kwargs["num_steps"],
            decode_best,
        )
        return _prefix_trajectory(trajectory)

    def evaluate_round(params, round_: LeaveOneOutRounds, key):
        accuracy, pixel_correctness = _generate_and_score(model, params, round_, key, mode, mode_kwargs)
        out = {"round_accuracy": accuracy, "round_pixel_correctness": pixel_correctness}
        if record_trajectory:
            out |= trajectory_one(params, round_, key)
        return out

    def evaluate(params, grids, shapes, key):
        rounds = _leave_one_out_rounds(grids, shapes)
        round_keys = _split_chain(key, grids.shape[0])
        return jax.vmap(evaluate_round, in_axes=(None, 0, 0))(params, rounds, round_keys)

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
    record_trajectory: bool = False,
) -> BatchEvaluator:
    """Our test-time adaptation conditions: for each held-out pair, fit a
    fresh per-round LoRA adapter on the MLP kernels of `target_modules`
    (`("decoder",)`, `("encoder",)`, or `("encoder", "decoder")` -- see
    lpn_exp.lora.LORA_CONDITION_TARGETS) by gradient ascent on the context
    pairs' log-likelihood, keep the best step, then decode the held-out
    query through the LoRA-patched model. See the module docstring for how
    each piece mirrors the paper's gradient-ascent condition. Whichever
    submodule isn't targeted runs with its pretrained weights unchanged.
    With `record_trajectory`, also records the per-step trajectory (see
    `_ascend`); its last step is the headline result.
    """
    import jax

    target_paths = find_mlp_kernel_paths(frozen_params, target_modules)
    optimizer = _gradient_ascent_optimizer(lr)

    def fit_and_score(frozen_params, round_: LeaveOneOutRounds, key):
        # `key` is this round's latent-sampling/decode key -- the same one the
        # mean/gradient_ascent evaluators give this round -- so at step 0 (b=0)
        # this is exactly the `mean` condition. The init key is derived from it.
        pairs, shapes = round_.context_grids[None], round_.context_shapes[None]

        def value_and_grad_fn(lora_params):
            return jax.value_and_grad(
                lambda lp: model.apply(
                    {"params": merge_lora(frozen_params, lp, scale)}, pairs, shapes, key, method=_context_log_prob
                )[0]
            )(lora_params)

        def decode_best(lora_params):
            patched = merge_lora(frozen_params, lora_params, scale)
            return _generate_and_score(model, patched, round_, key, "mean", {})

        init = init_lora_params(frozen_params, target_paths, rank, jax.random.fold_in(key, 1))
        best, trajectory = _ascend(
            value_and_grad_fn, init, optimizer, num_steps, decode_best if record_trajectory else None
        )
        if record_trajectory:
            accuracy, pixel_correctness = trajectory["accuracy"][-1], trajectory["pixel_correctness"][-1]
            out = _prefix_trajectory(trajectory)
        else:
            accuracy, pixel_correctness = decode_best(best)
            out = {}
        return {"round_accuracy": accuracy, "round_pixel_correctness": pixel_correctness} | out

    def evaluate(frozen_params, grids, shapes, key):
        rounds = _leave_one_out_rounds(grids, shapes)
        round_keys = _split_chain(key, grids.shape[0])  # same per-round keys as mean/GA
        return jax.vmap(fit_and_score, in_axes=(None, 0, 0))(frozen_params, rounds, round_keys)

    return _as_batch_evaluator(evaluate, frozen_params, max_rows, max_cols, batch_size)


def num_adapted_params(model, params, condition: str, config) -> int:
    """How many numbers test-time adaptation optimizes per round: 0 for
    `mean`, the latent's size for `gradient_ascent`, and the LoRA factors'
    total size (rank * (in + out) per targeted kernel) for `lora_ascent_*`.
    """
    if condition == "mean":
        return 0
    if condition == "gradient_ascent":
        return int(model.encoder.config.latent_dim)
    total = 0
    for path in find_mlp_kernel_paths(params, LORA_CONDITION_TARGETS[condition]):
        kernel = params
        for key in path:
            kernel = kernel[key]
        in_dim, out_dim = kernel.shape
        total += config.lora.rank * (in_dim + out_dim)
    return total


def build_condition_evaluator(
    model, params, max_rows: int, max_cols: int, condition: str, config, lr=None, record_trajectory=False
):
    """The evaluator for one of the config's conditions, using `lr` if given
    (the learning-rate sweep) or else the config's own
    `learning_rates[condition]`. `config` is an Exp4Config. With
    `record_trajectory`, adaptation conditions also return their per-step
    trajectories (`mean` has none: it is every method's step 0).
    """
    if condition == "mean":
        return build_mean_or_gradient_ascent_evaluator(
            model, params, max_rows, max_cols, "mean", {}, config.batch_size
        )
    lr = config.learning_rates[condition] if lr is None else lr
    if condition == "gradient_ascent":
        mode_kwargs = {"num_steps": config.gradient_ascent.num_steps, "lr": lr}
        return build_mean_or_gradient_ascent_evaluator(
            model, params, max_rows, max_cols, "gradient_ascent", mode_kwargs, config.batch_size, record_trajectory
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
            record_trajectory=record_trajectory,
        )
    raise ValueError(f"Unknown condition: {condition!r}")
