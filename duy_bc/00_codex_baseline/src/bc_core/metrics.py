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


def _integer_sequence_metadata(
    values: np.ndarray, name: str, rows: int
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != rows:
        raise ValueError(f"{name} must have shape ({rows},)")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype, np.integer
    ):
        raise ValueError(f"{name} must have an integer dtype")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return array.astype(np.int64, copy=False)


def _validate_contiguous_trajectories(
    trajectories: dict[Any, dict[int, bool]], name: str
) -> None:
    for identity, by_step in trajectories.items():
        ordered_steps = sorted(by_step)
        expected = list(range(ordered_steps[0], ordered_steps[-1] + 1))
        if ordered_steps != expected:
            raise ValueError(
                f"step_indices must be contiguous within {name} {identity!r}"
            )


def _prefix_survival(
    trajectories: dict[Any, dict[int, bool]], requested_horizon: int
) -> tuple[float, list[dict[str, int | float]], int]:
    evaluated_horizon = min(
        requested_horizon, max(len(by_step) for by_step in trajectories.values())
    )
    survival: list[dict[str, int | float]] = []
    ordered = [
        np.asarray([by_step[step] for step in sorted(by_step)], dtype=np.bool_)
        for by_step in trajectories.values()
    ]
    for horizon in range(1, evaluated_horizon + 1):
        windows = 0
        perfect_windows = 0
        for correctness in ordered:
            trajectory_windows = correctness.shape[0] - horizon + 1
            if trajectory_windows <= 0:
                continue
            windows += trajectory_windows
            errors = np.logical_not(correctness).astype(np.int64, copy=False)
            cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(errors)))
            window_errors = cumulative[horizon:] - cumulative[:-horizon]
            perfect_windows += int(np.count_nonzero(window_errors == 0))
        survival.append(
            {
                "horizon": horizon,
                "perfect_windows": perfect_windows,
                "windows": windows,
                "rate": float(perfect_windows / windows),
            }
        )
    return (
        float(np.mean([item["rate"] for item in survival])),
        survival,
        evaluated_horizon,
    )


def _daily_prefix_scores(
    trajectories: dict[Any, dict[int, bool]], turns_per_day: int
) -> list[float]:
    scores: list[float] = []
    for by_step in trajectories.values():
        by_day: dict[int, list[bool]] = {}
        for step in sorted(by_step):
            by_day.setdefault(step // turns_per_day, []).append(by_step[step])
        for correctness in by_day.values():
            correct_prefix = 0
            for correct in correctness:
                if not correct:
                    break
                correct_prefix += 1
            scores.append(correct_prefix / len(correctness))
    return scores


def core_cloning_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    game_ids: Sequence[str],
    step_indices: np.ndarray,
    actor_ids: np.ndarray,
    *,
    step_horizon: int = 24,
    turns_per_day: int = 24,
) -> dict[str, Any]:
    """Measure action balance and prefix survival on held-out actor trajectories.

    The primary prefix metrics isolate each actor slot and follow its observed
    decision opportunities, so one hand's disagreement does not erase correct
    behavior by the farmer or other hands. Strict joint-farm variants are
    retained as diagnostics and require every actor at an environment step to
    match.
    """
    scores, targets = _validated_classification_inputs(logits, labels)
    if isinstance(step_horizon, bool) or not isinstance(step_horizon, int):
        raise ValueError("step_horizon must be a positive integer")
    if isinstance(turns_per_day, bool) or not isinstance(turns_per_day, int):
        raise ValueError("turns_per_day must be a positive integer")
    if step_horizon <= 0:
        raise ValueError("step_horizon must be a positive integer")
    if turns_per_day <= 0:
        raise ValueError("turns_per_day must be a positive integer")

    if isinstance(game_ids, (str, bytes)):
        raise ValueError("game_ids must be a sequence of strings")
    try:
        games = tuple(game_ids)
    except TypeError as error:
        raise ValueError("game_ids must be a sequence of strings") from error
    if len(games) != scores.shape[0] or any(
        not isinstance(game, str) or not game for game in games
    ):
        raise ValueError(
            f"game_ids must contain {scores.shape[0]} non-empty strings"
        )
    steps = _integer_sequence_metadata(step_indices, "step_indices", scores.shape[0])
    actors = _integer_sequence_metadata(actor_ids, "actor_ids", scores.shape[0])

    predictions = np.argmax(scores, axis=1)
    row_correct = predictions == targets
    actor_trajectories: dict[tuple[str, int], dict[int, bool]] = {}
    joint_farm_trajectories: dict[str, dict[int, bool]] = {}
    for game, step, actor, correct in zip(
        games, steps.tolist(), actors.tolist(), row_correct.tolist()
    ):
        actor_steps = actor_trajectories.setdefault((game, actor), {})
        if step in actor_steps:
            raise ValueError(
                f"duplicate actor row game={game!r} actor={actor} step={step}"
            )
        actor_steps[step] = bool(correct)
        joint_steps = joint_farm_trajectories.setdefault(game, {})
        joint_steps[step] = joint_steps.get(step, True) and bool(correct)

    _validate_contiguous_trajectories(joint_farm_trajectories, "game")
    actor_auc, actor_survival, actor_horizon = _prefix_survival(
        actor_trajectories, step_horizon
    )
    joint_auc, joint_survival, joint_horizon = _prefix_survival(
        joint_farm_trajectories, step_horizon
    )
    actor_daily = _daily_prefix_scores(actor_trajectories, turns_per_day)
    joint_daily = _daily_prefix_scores(joint_farm_trajectories, turns_per_day)

    classification = classification_report(scores, targets)
    observed_f1 = [
        float(item["f1"])
        for item in classification["per_class"]
        if int(item["support"]) > 0
    ]
    return {
        f"step_prefix_auc_at_{step_horizon}": actor_auc,
        "daily_gated_prefix_auc": float(np.mean(actor_daily)),
        "action_macro_f1": float(np.mean(observed_f1)),
        "raw_accuracy": float(classification["top1"]),
        "step_prefix_survival": actor_survival,
        "requested_step_horizon": step_horizon,
        "evaluated_step_horizon": actor_horizon,
        "actor_steps": int(scores.shape[0]),
        "actor_trajectories": len(actor_trajectories),
        "actor_days": len(actor_daily),
        f"joint_farm_step_prefix_auc_at_{step_horizon}": joint_auc,
        "joint_farm_daily_gated_prefix_auc": float(np.mean(joint_daily)),
        "joint_farm_step_prefix_survival": joint_survival,
        "joint_farm_evaluated_step_horizon": joint_horizon,
        "environment_steps": int(
            sum(len(by_step) for by_step in joint_farm_trajectories.values())
        ),
        "days": len(joint_daily),
        "supported_actions": len(observed_f1),
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
