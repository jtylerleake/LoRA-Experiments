"""Task families for experiment 4's harder benchmark ladder.

Pattern-2D as the checkpoint was trained on it (2x2 patterns, 4x4 grids) is
solved by every adaptation method, so it can't show whether LoRA adapts
better than the paper's latent search. These families make harder tasks that
still fit the pretrained checkpoint (4x4 grids, colors 0-9, full-size
outputs -- see config.CHECKPOINT_GRID_SIZE):

  lpn_pattern        lpn's own Pattern-2D generator (level L0, the tasks the
                     completed runs used) -- see pattern2d_tasks.generate_tasks.
  pattern            our pattern generator: a k x k pattern stamped at a
                     marker pixel, any k that fits, optionally sparse
                     (`pattern_density` < 1: empty cells are color 0), and
                     optionally a per-task `anchor` corner (which corner of
                     the pattern the marker marks). Marker positions within a
                     task never repeat, so no held-out pair ever duplicates a
                     context pair -- every round is clean by construction.
  color_permutation  random input grids over `num_colors` colors, recolored
                     by a per-task mapping of colors 1-9. Unlike the pattern
                     families, no single pair reveals the whole rule: each
                     shows only the colors it uses, so the model has to
                     combine information across pairs.

Every generated task is well-posed: for each leave-one-out round, the exact
solver (`solve_round`) must recover the held-out output from the context
pairs alone, uniquely. Tasks that fail (e.g. a sparse pattern whose anchor
is ambiguous, or a query color no context pair shows) are redrawn.

Task k is drawn from np.random.default_rng([seed, k]), so it depends only on
(seed, k): growing num_tasks never changes existing tasks, which keeps resume
fingerprints stable.

numpy only (no jax, and not lpn's global `random`), so it's unit-testable in
the CPU test image. `make_tasks` dispatches `lpn_pattern` to lpn's generator,
which needs the lpn repo on sys.path.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from lpn_exp.pattern2d_tasks import Pattern2DTask

MAX_DRAWS_PER_TASK = 1000


def make_tasks(task_generator, num_tasks: int, seed: int) -> list[Pattern2DTask]:
    """`num_tasks` tasks of `task_generator.family` (a TaskGeneratorConfig)."""
    if task_generator.family == "lpn_pattern":
        from lpn_exp.pattern2d_tasks import generate_tasks

        return generate_tasks(
            num_tasks=num_tasks,
            num_pairs=task_generator.num_pairs,
            num_rows=task_generator.num_rows,
            num_cols=task_generator.num_cols,
            pattern_size=task_generator.pattern_size,
            seed=seed,
        )
    return [generate_family_task(task_generator, seed, task_id) for task_id in range(num_tasks)]


def generate_family_task(task_generator, seed: int, task_id: int) -> Pattern2DTask:
    """One well-posed task of a `pattern` or `color_permutation` family."""
    rng = np.random.default_rng([seed, task_id])
    draw = {"pattern": _draw_pattern_pairs, "color_permutation": _draw_color_permutation_pairs}[task_generator.family]
    for _ in range(MAX_DRAWS_PER_TASK):
        pairs = draw(task_generator, rng)
        if all_rounds_solvable(task_generator, pairs):
            return Pattern2DTask(task_id=task_id, pairs=pairs)
    raise RuntimeError(f"No well-posed task in {MAX_DRAWS_PER_TASK} draws for {task_generator}")


def _anchors(task_generator) -> list[tuple[int, int]]:
    """The marker's possible offsets inside the pattern window."""
    k = task_generator.pattern_size
    if task_generator.anchor == "top_left":
        return [(0, 0)]
    return [(0, 0), (0, k - 1), (k - 1, 0), (k - 1, k - 1)]


def _draw_pattern_pairs(task_generator, rng) -> list[dict[str, Any]]:
    k, rows, cols = task_generator.pattern_size, task_generator.num_rows, task_generator.num_cols
    colors = rng.integers(1, 10, size=(k, k))
    pattern = np.where(rng.random((k, k)) < task_generator.pattern_density, colors, 0)
    if not pattern.any():  # never an all-empty pattern (a blank output)
        pattern[rng.integers(k), rng.integers(k)] = rng.integers(1, 10)
    anchors = _anchors(task_generator)
    anchor_row, anchor_col = anchors[rng.integers(len(anchors))]
    windows = [(r, c) for r in range(rows - k + 1) for c in range(cols - k + 1)]
    pairs = []
    for index in rng.choice(len(windows), size=task_generator.num_pairs, replace=False):
        top, left = windows[index]
        grid_in, grid_out = np.zeros((rows, cols), dtype=int), np.zeros((rows, cols), dtype=int)
        grid_in[top + anchor_row, left + anchor_col] = 1
        grid_out[top : top + k, left : left + k] = pattern
        pairs.append({"input": grid_in, "output": grid_out})
    return pairs


def _draw_color_permutation_pairs(task_generator, rng) -> list[dict[str, Any]]:
    rows, cols = task_generator.num_rows, task_generator.num_cols
    palette = rng.choice(np.arange(1, 10), size=task_generator.num_colors, replace=False)
    mapping = np.arange(10)
    mapping[1:] = rng.permutation(np.arange(1, 10))  # 0 is never used, so it maps to itself
    pairs = []
    for _ in range(task_generator.num_pairs):
        grid_in = rng.choice(palette, size=(rows, cols))
        pairs.append({"input": grid_in, "output": mapping[grid_in]})
    return pairs


def solve_round(task_generator, context_pairs: list[dict[str, Any]], query_input: np.ndarray):
    """The held-out output implied by `context_pairs` for `query_input`, or
    None if the context doesn't determine it uniquely. An exact solver that
    knows the family's rules -- used to guarantee every round is answerable,
    and in the tests to check the generator.
    """
    if task_generator.family == "color_permutation":
        return _solve_color_permutation(context_pairs, query_input)
    return _solve_pattern(task_generator, context_pairs, query_input)


def _solve_pattern(task_generator, context_pairs, query_input):
    k = task_generator.pattern_size
    rows, cols = query_input.shape

    def marker(grid):
        cells = np.argwhere(grid != 0)
        return tuple(cells[0]) if len(cells) == 1 else None

    def window(position, anchor):
        top, left = position[0] - anchor[0], position[1] - anchor[1]
        return (top, left) if 0 <= top <= rows - k and 0 <= left <= cols - k else None

    predictions = []
    for anchor in _anchors(task_generator):
        pattern = None
        for pair in context_pairs:
            position = marker(pair["input"])
            corner = window(position, anchor) if position is not None else None
            if corner is None:
                break
            top, left = corner
            candidate = pair["output"][top : top + k, left : left + k]
            outside = pair["output"].copy()
            outside[top : top + k, left : left + k] = 0
            if outside.any() or (pattern is not None and not np.array_equal(candidate, pattern)):
                break
            pattern = candidate
        else:
            position = marker(query_input)
            corner = window(position, anchor) if position is not None else None
            if corner is not None and pattern is not None:
                top, left = corner
                prediction = np.zeros_like(query_input)
                prediction[top : top + k, left : left + k] = pattern
                predictions.append(prediction)
    distinct = {prediction.tobytes(): prediction for prediction in predictions}
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def _solve_color_permutation(context_pairs, query_input):
    mapping: dict[int, int] = {}
    for pair in context_pairs:
        for color_in, color_out in zip(pair["input"].ravel(), pair["output"].ravel()):
            if mapping.setdefault(int(color_in), int(color_out)) != color_out:
                return None
    if any(int(color) not in mapping for color in np.unique(query_input)):
        return None
    return np.vectorize(mapping.__getitem__)(query_input)


def all_rounds_solvable(task_generator, pairs: list[dict[str, Any]]) -> bool:
    """True if every leave-one-out round's held-out output is uniquely
    determined by its context pairs (see `solve_round`).
    """
    for i, query in enumerate(pairs):
        context = pairs[:i] + pairs[i + 1 :]
        prediction = solve_round(task_generator, context, query["input"])
        if prediction is None or not np.array_equal(prediction, query["output"]):
            return False
    return True
