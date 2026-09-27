"""Builds a fixed, seeded set of Pattern-2D evaluation tasks.

Pattern-2D isn't a downloadable dataset (see CLAUDE-CODING-SKILL.md) --
`clement-bonnet/lpn`'s own `PatternTaskGenerator` (a `torch.utils.data.
IterableDataset`) generates it procedurally. We seed it once and draw a
fixed number of tasks up front so every run (and every condition within a
run) scores the exact same tasks -- comparing `mean` vs. `gradient_ascent`
vs. the `lora_ascent_*` variants only means something if they all see
identical inputs.

Each task is `num_pairs` (input, output) grid pairs sharing one random
pattern. We don't reserve a separate held-out query pair on top of that --
we follow the paper's own evaluation convention (see
evaluate_checkpoint.py's `build_generate_output_batch_to_be_pmapped` and
`LPN.__call__`'s "mode" docstring): every one of the `num_pairs` pairs gets
a turn as the target, decoded from the other `num_pairs - 1` via
`lpn`'s own `make_leave_one_out` (src/data_utils.py), and accuracy is
averaged over all of them. This matches `pattern_2d.yaml`'s own eval block
exactly (`num_pairs: 4`), so results are comparable to the paper's reported
numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Pattern2DTask:
    task_id: int
    pairs: list[dict[str, Any]]  # each {"input": np.ndarray, "output": np.ndarray}


def generate_tasks(
    num_tasks: int, num_pairs: int, num_rows: int, num_cols: int, pattern_size: int, seed: int
) -> list[Pattern2DTask]:
    from src.datasets.task_gen.task_generator import PatternTaskGenerator

    generator = PatternTaskGenerator(
        num_pairs=num_pairs,
        seed=seed,
        num_rows=num_rows,
        num_cols=num_cols,
        pattern_size=pattern_size,
    )
    iterator = iter(generator)
    tasks = []
    for task_id in range(num_tasks):
        pairs, _info = next(iterator)
        tasks.append(Pattern2DTask(task_id=task_id, pairs=pairs))
    return tasks


def task_to_arrays(task: Pattern2DTask, max_rows: int, max_cols: int):
    """Converts one task's raw {"input", "output"} numpy-grid pairs into the
    (grids, shapes) jax-array shape `LPN.__call__`/`generate_output` expect:
    grids (N, R, C, 2), shapes (N, 2, 2) -- see lpn.py's `__call__` docstring
    for the exact convention (last axis of `grids` is [input, output]; last
    two axes of `shapes` are [[rows_in, rows_out], [cols_in, cols_out]]).

    PATTERN's grids are already exactly (num_rows, num_cols) with no padding
    needed, since the pretrained checkpoint's max_rows/max_cols (4) match
    the task generator's num_rows/num_cols (4) -- unlike ARC's variable-size
    grids, there's no crop/pad step here.
    """
    import jax.numpy as jnp
    import numpy as np

    assert all(pair["input"].shape == (max_rows, max_cols) for pair in task.pairs), (
        "Pattern2D grids must already match the model's max_rows/max_cols; "
        "got a task/checkpoint size mismatch."
    )
    grids = jnp.array(
        np.stack(
            [np.stack([pair["input"], pair["output"]], axis=-1) for pair in task.pairs], axis=0
        )
    )
    # [[rows_input, rows_output], [cols_input, cols_output]] -- see LPN.__call__'s
    # `grid_shapes` docstring. PATTERN always fills the whole grid, so input and
    # output shapes are identical and equal to (max_rows, max_cols) for every pair.
    shape = jnp.array([[max_rows, max_rows], [max_cols, max_cols]])
    shapes = jnp.broadcast_to(shape, (len(task.pairs), 2, 2))
    return grids, shapes


def clean_round_mask(task: Pattern2DTask) -> list[bool]:
    """One flag per leave-one-out round (round i holds out pair i, the same
    order as test_time_adapt's rounds): True if the held-out pair appears
    nowhere else in the task, False if an identical (input, output) pair is
    among its own context.

    Pattern-2D places its 2x2 pattern at one of only 9 positions on a 4x4
    grid, so a held-out pair often duplicates a context pair exactly --
    roughly 30% of rounds with 4 pairs. In such a round the answer is sitting
    in the context, and test-time adaptation that fits the context pairs
    (especially a LoRA adapter, which has the capacity to memorize them) is
    effectively trained on the test example. We keep the paper's protocol
    (every round is scored) and report clean rounds separately alongside all
    rounds (see scripts/plot_exp4_results.py's results table).

    Plain numpy only, so it's unit-testable without jax.
    """
    keys = [(pair["input"].tobytes(), pair["output"].tobytes()) for pair in task.pairs]
    return [keys.count(key) == 1 for key in keys]
