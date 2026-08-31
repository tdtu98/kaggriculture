"""Dependency-free classification and paired game-level evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bc_core.constants import OPERATIONS


_CLASS_COUNT = len(OPERATIONS)
_SLICE_DIMENSIONS = (
    "actor_type",
    "seat",
    "day_band",
    "source_date",
    "route_family",
)


def _validated_classification_inputs(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(logits)
    targets = np.asarray(labels)
    if scores.ndim != 2 or scores.shape[1:] != (_CLASS_COUNT,):
        raise ValueError(
            f"logits must have shape (rows, {_CLASS_COUNT}), got {scores.shape}"
        )
    if scores.shape[0] == 0:
        raise ValueError("classification metrics require at least one row")
    if not np.issubdtype(scores.dtype, np.number) or np.issubdtype(
        scores.dtype, np.complexfloating
    ):
        raise ValueError("logits must have a real numeric dtype")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits must contain only finite values")
    if targets.ndim != 1 or targets.shape[0] != scores.shape[0]:
        raise ValueError(
            f"labels must have shape ({scores.shape[0]},), got {targets.shape}"
        )
    if np.issubdtype(targets.dtype, np.bool_) or not np.issubdtype(
        targets.dtype, np.integer
    ):
        raise ValueError("labels must have an integer dtype")
    if np.any(targets < 0) or np.any(targets >= _CLASS_COUNT):
        raise ValueError(f"labels must be in the range [0, {_CLASS_COUNT - 1}]")
    return scores, targets.astype(np.int64, copy=False)


def _top_three_members(logits: np.ndarray) -> np.ndarray:
    """Return top-three class IDs, choosing lower IDs at a tied boundary."""
    partition = np.argpartition(logits, -3, axis=1)[:, -3:]
    thresholds = np.min(np.take_along_axis(logits, partition, axis=1), axis=1)
    members = np.empty((logits.shape[0], 3), dtype=np.int64)
    for row_index, threshold in enumerate(thresholds):
        above = np.flatnonzero(logits[row_index] > threshold)
        tied = np.flatnonzero(logits[row_index] == threshold)
        needed = 3 - above.size
        members[row_index] = np.concatenate((above, tied[:needed]))
    return members


def classification_report(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Compute fixed-17-class accuracy and confusion-derived metrics."""
    scores, targets = _validated_classification_inputs(logits, labels)
    predictions = np.argmax(scores, axis=1)
    confusion = np.zeros((_CLASS_COUNT, _CLASS_COUNT), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)

    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = np.zeros(_CLASS_COUNT, dtype=np.float64)
    recall = np.zeros(_CLASS_COUNT, dtype=np.float64)
    np.divide(true_positive, predicted, out=precision, where=predicted != 0)
    np.divide(true_positive, support, out=recall, where=support != 0)
    f1 = np.zeros(_CLASS_COUNT, dtype=np.float64)
    denominator = precision + recall
    np.divide(2.0 * precision * recall, denominator, out=f1, where=denominator != 0)

    top_three = _top_three_members(scores)
    top_three_correct = np.any(top_three == targets[:, None], axis=1)
    per_class = [
        {
            "class_id": class_id,
            "operation": OPERATIONS[class_id],
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id in range(_CLASS_COUNT)
    ]
    return {
        "top1": float(np.mean(predictions == targets)),
        "top3": float(np.mean(top_three_correct)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def slice_reports(
    logits: np.ndarray,
    labels: np.ndarray,
    slices: Sequence[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Compute reports for every observed value of each required slice field."""
    scores, targets = _validated_classification_inputs(logits, labels)
    try:
        rows = tuple(slices)
    except TypeError as error:
        raise ValueError("slices must be a sequence of row dictionaries") from error
    if len(rows) != scores.shape[0]:
        raise ValueError(
            f"slices must contain {scores.shape[0]} rows, got {len(rows)}"
        )
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"slice row {row_index} must be a dictionary")
        for dimension in _SLICE_DIMENSIONS:
            if dimension not in row or not isinstance(row[dimension], str):
                raise ValueError(
                    f"slice row {row_index} field {dimension!r} must be a string"
                )

    reports: dict[str, dict[str, Any]] = {}
    for dimension in _SLICE_DIMENSIONS:
        values = sorted({row[dimension] for row in rows})
        reports[dimension] = {}
        for value in values:
            mask = np.fromiter(
                (row[dimension] == value for row in rows),
                dtype=np.bool_,
                count=len(rows),
            )
            reports[dimension][value] = classification_report(scores[mask], targets[mask])
    return reports


def _correctness_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    is_bool = np.issubdtype(array.dtype, np.bool_)
    is_integer = np.issubdtype(array.dtype, np.integer) and not is_bool
    if not is_bool and not is_integer:
        raise ValueError(f"{name} must contain booleans or zero/one integers")
    if is_integer and np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain only zero or one")
    return array.astype(np.bool_, copy=False)


def paired_game_bootstrap(
    state_correct: np.ndarray,
    clock_correct: np.ndarray,
    game_ids: Sequence[str],
    resamples: int = 10000,
    seed: int = 20260824,
) -> dict[str, float]:
    """Bootstrap the unweighted paired state-minus-clock accuracy by game."""
    state = _correctness_vector(state_correct, "state_correct")
    clock = _correctness_vector(clock_correct, "clock_correct")
    if state.shape != clock.shape:
        raise ValueError("state_correct and clock_correct must have the same length")
    if isinstance(game_ids, (str, bytes)):
        raise ValueError("game_ids must be a sequence of strings")
    try:
        games = tuple(game_ids)
    except TypeError as error:
        raise ValueError("game_ids must be a sequence of strings") from error
    if len(games) != state.shape[0]:
        raise ValueError(f"game_ids must contain {state.shape[0]} values, got {len(games)}")
    if any(not isinstance(game_id, str) or not game_id for game_id in games):
        raise ValueError("game_ids must contain only non-empty strings")
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)):
        raise ValueError("resamples must be a positive integer")
    if int(resamples) <= 0:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    if not 0 <= int(seed) <= np.iinfo(np.uint64).max:
        raise ValueError("seed must fit in an unsigned 64-bit integer")

    _, inverse = np.unique(np.asarray(games, dtype=np.str_), return_inverse=True)
    game_count = int(inverse.max()) + 1
    rows_per_game = np.bincount(inverse, minlength=game_count)
    state_accuracy = np.bincount(
        inverse, weights=state.astype(np.float64), minlength=game_count
    ) / rows_per_game
    clock_accuracy = np.bincount(
        inverse, weights=clock.astype(np.float64), minlength=game_count
    ) / rows_per_game
    game_deltas = state_accuracy - clock_accuracy

    rng = np.random.default_rng(int(seed))
    sampled_games = rng.integers(
        0, game_count, size=(int(resamples), game_count), endpoint=False
    )
    bootstrap_deltas = game_deltas[sampled_games].mean(axis=1)
    ci95_low, ci95_high = np.percentile(bootstrap_deltas, [2.5, 97.5])
    return {
        "point_delta": float(game_deltas.mean()),
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
        "seed": int(seed),
        "resamples": int(resamples),
        "games": game_count,
    }
