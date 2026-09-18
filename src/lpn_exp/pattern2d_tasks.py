"""Builds a fixed, seeded set of Pattern-2D evaluation tasks.

Pattern-2D isn't a downloadable dataset (see CLAUDE-CODING-SKILL.md) --
`clement-bonnet/lpn`'s own `PatternTaskGenerator` (a `torch.utils.data.
IterableDataset`) generates it procedurally. We seed it once and draw a
fixed number of tasks up front so every run (and every condition within a
run) scores the exact same tasks -- comparing `mean` vs. `gradient_ascent`
vs. `lora_ascent` only means something if they all see identical inputs.

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
