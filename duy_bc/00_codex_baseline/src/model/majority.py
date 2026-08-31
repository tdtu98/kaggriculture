"""Train-fitted majority behavior-cloning baselines."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bc_core.constants import OPERATIONS


_CLASS_COUNT = len(OPERATIONS)


@dataclass(frozen=True)
class MajorityRules:
    """Complete global and actor-stratified class rankings."""

    global_label: int
    farmer_label: int
    hand_label: int
    global_ranking: tuple[int, ...]
    farmer_ranking: tuple[int, ...]
    hand_ranking: tuple[int, ...]

    def predict(self, actor_is_farmer: np.ndarray) -> np.ndarray:
        """Predict the train mode for farmer rows and the train mode for hand rows."""
        actors = np.asarray(actor_is_farmer)
        if not np.all((actors == 0) | (actors == 1)):
            raise ValueError("actor_is_farmer must contain only zero or one")
        return np.where(actors == 1, self.farmer_label, self.hand_label).astype(
            np.int64, copy=False
        )


def _class_ranking(counts: np.ndarray) -> tuple[int, ...]:
    return tuple(
        sorted(range(_CLASS_COUNT), key=lambda label: (-int(counts[label]), label))
    )


def _majority_from_counts(
    counts: np.ndarray, farmer_counts: np.ndarray, hand_counts: np.ndarray
) -> MajorityRules:
    global_ranking = _class_ranking(counts)
    farmer_ranking = _class_ranking(farmer_counts)
    hand_ranking = _class_ranking(hand_counts)
    return MajorityRules(
        global_label=global_ranking[0],
        farmer_label=farmer_ranking[0],
        hand_label=hand_ranking[0],
        global_ranking=global_ranking,
        farmer_ranking=farmer_ranking,
        hand_ranking=hand_ranking,
    )


def fit_majority_rules(
    labels: np.ndarray, actor_is_farmer: np.ndarray
) -> MajorityRules:
    """Fit complete global/farmer/hand rankings with stable lower-ID tie breaks."""
    label_values = np.asarray(labels)
    actor_values = np.asarray(actor_is_farmer)
    if (
        label_values.ndim != 1
        or actor_values.ndim != 1
        or label_values.shape != actor_values.shape
    ):
        raise ValueError("labels and actor_is_farmer must be equal-length vectors")
    if label_values.size == 0:
        raise ValueError("cannot fit majority rules without labels")
    if not np.issubdtype(label_values.dtype, np.integer) or np.any(
        (label_values < 0) | (label_values >= _CLASS_COUNT)
    ):
        raise ValueError("labels contain an invalid operation ID")
    if not np.all((actor_values == 0) | (actor_values == 1)):
        raise ValueError("actor_is_farmer must contain only zero or one")

    counts = np.bincount(label_values.astype(np.int64), minlength=_CLASS_COUNT)
    farmer_counts = np.bincount(
        label_values[actor_values == 1].astype(np.int64), minlength=_CLASS_COUNT
    )
    hand_counts = np.bincount(
        label_values[actor_values == 0].astype(np.int64), minlength=_CLASS_COUNT
    )
    return _majority_from_counts(counts, farmer_counts, hand_counts)


def ranking_to_logits(ranking: Sequence[int]) -> np.ndarray:
    """Convert a complete class ranking to descending reusable class logits."""
    order = tuple(int(label) for label in ranking)
    if len(order) != _CLASS_COUNT or set(order) != set(range(_CLASS_COUNT)):
        raise ValueError("ranking must be a complete operation permutation")
    logits = np.empty(_CLASS_COUNT, dtype=np.float32)
    for rank_index, label in enumerate(order):
        logits[label] = float(_CLASS_COUNT - rank_index)
    return logits
