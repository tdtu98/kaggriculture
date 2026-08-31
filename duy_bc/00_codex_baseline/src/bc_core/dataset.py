"""Train-only artifacts and lazy PyTorch materialization for replay shards."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from bc_core.constants import ACTOR_DIM, CLOCK_DIM, GLOBAL_DIM, OPERATIONS
from bc_core.features import EncodedGame, read_shard
from model.majority import MajorityRules, _majority_from_counts


_CLASS_COUNT = len(OPERATIONS)
_ARTIFACT_FIELDS = (
    "global_mean",
    "global_std",
    "actor_mean",
    "actor_std",
    "class_counts",
    "class_weights",
    "majority_labels",
    "global_ranking",
    "farmer_ranking",
    "hand_ranking",
)


@dataclass(frozen=True)
class NormalizationStats:
    global_mean: np.ndarray
    global_std: np.ndarray
    actor_mean: np.ndarray
    actor_std: np.ndarray


@dataclass(frozen=True)
class Batch:
    grid: torch.Tensor
    global_features: torch.Tensor
    actor_features: torch.Tensor
    clock_features: torch.Tensor
    label: torch.Tensor
    game_id: tuple[str, ...]
    slices: tuple[dict[str, str], ...]


class _RunningStatistics:
    def __init__(self) -> None:
        self.count = 0
        self.width: int | None = None
        self.total: np.ndarray | None = None
        self.total_sq: np.ndarray | None = None

    def add(self, raw_array: np.ndarray) -> None:
        array = np.asarray(raw_array)
        if array.ndim != 2:
            raise ValueError(f"statistics arrays must be two-dimensional, got {array.shape}")
        if self.width is None:
            self.width = int(array.shape[1])
            self.total = np.zeros(self.width, dtype=np.float64)
            self.total_sq = np.zeros(self.width, dtype=np.float64)
        elif array.shape[1] != self.width:
            raise ValueError(
                f"statistics column count changed from {self.width} to {array.shape[1]}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("statistics arrays must contain only finite values")
        values = array.astype(np.float64, copy=False)
        self.count += int(values.shape[0])
        assert self.total is not None and self.total_sq is not None
        self.total += values.sum(axis=0, dtype=np.float64)
        self.total_sq += np.square(values).sum(axis=0, dtype=np.float64)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if self.width is None or self.count == 0:
            raise ValueError("cannot fit statistics without rows")
        assert self.total is not None and self.total_sq is not None
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std[np.abs(std) < 1e-8] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)


def fit_array_statistics(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Fit population mean and standard deviation with float64 running sums."""
    running = _RunningStatistics()
    for raw_array in arrays:
        running.add(raw_array)
    return running.finish()


def inverse_sqrt_class_weights(counts: np.ndarray, cap: float = 4.0) -> np.ndarray:
    """Return inverse-square-root weights with water-filled cap and mean one."""
    values = np.asarray(counts)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("class counts must be a non-empty one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values <= 0):
        raise ValueError("class counts must be positive integers")
    if not math.isfinite(cap) or cap < 1.0:
        raise ValueError("class weight cap must be finite and at least one")

    raw = 1.0 / np.sqrt(values.astype(np.float64))
    weights = np.zeros_like(raw)
    active = np.ones(raw.shape, dtype=bool)
    remaining_total = float(raw.size)
    while active.any():
        scale = remaining_total / raw[active].sum()
        over = active & (raw * scale > cap)
        if not over.any():
            weights[active] = raw[active] * scale
            break
        weights[over] = cap
        remaining_total -= cap * int(over.sum())
        active[over] = False
    result = weights.astype(np.float32)
    cap_float32 = np.float32(cap)
    if float(cap_float32) > cap:
        cap_float32 = np.nextafter(cap_float32, np.float32(-np.inf))
    result = np.minimum(result, cap_float32)
    if (
        not np.isclose(result.mean(dtype=np.float64), 1.0, rtol=1e-6, atol=1e-7)
        or float(result.max()) > cap
    ):
        raise ValueError("class-weight normalization invariant failed")
    return result


def _derive_train_artifacts(
    train_shards: Sequence[Path], weight_cap: float = 4.0
) -> tuple[NormalizationStats, np.ndarray, np.ndarray, MajorityRules]:
    if not train_shards:
        raise ValueError("at least one train shard is required")
    global_statistics = _RunningStatistics()
    actor_statistics = _RunningStatistics()
    class_counts = np.zeros(_CLASS_COUNT, dtype=np.int64)
    farmer_counts = np.zeros(_CLASS_COUNT, dtype=np.int64)
    hand_counts = np.zeros(_CLASS_COUNT, dtype=np.int64)
    for path in train_shards:
        game = read_shard(Path(path))
        if game.metadata.get("split") != "train":
            raise ValueError(
                "fit_train_artifacts requires split='train', "
                f"got split={game.metadata.get('split')!r} "
                f"path={path}"
            )
        global_statistics.add(game.global_features[game.step_index])
        actor_statistics.add(game.actor_features)
        actors = game.actor_features[:, 0]
        if not np.all((actors == 0) | (actors == 1)):
            raise ValueError(f"actor type must be zero or one in train shard path={path}")
        shard_counts = np.bincount(game.label, minlength=_CLASS_COUNT).astype(np.int64)
        shard_farmer_counts = np.bincount(
            game.label[actors == 1], minlength=_CLASS_COUNT
        ).astype(np.int64)
        class_counts += shard_counts
        farmer_counts += shard_farmer_counts
        hand_counts += shard_counts - shard_farmer_counts

    global_mean, global_std = global_statistics.finish()
    actor_mean, actor_std = actor_statistics.finish()
    if np.any(class_counts <= 0):
        missing = np.flatnonzero(class_counts <= 0).tolist()
        raise ValueError(f"all 17 operations require a positive train count; missing={missing}")
    class_weights = inverse_sqrt_class_weights(class_counts, cap=weight_cap)
    majority = _majority_from_counts(class_counts, farmer_counts, hand_counts)
    stats = NormalizationStats(global_mean, global_std, actor_mean, actor_std)
    return stats, class_counts, class_weights, majority


def fit_train_artifacts(
    train_shards: Sequence[Path], weight_cap: float = 4.0
) -> tuple[NormalizationStats, np.ndarray, np.ndarray, MajorityRules]:
    """Fit normalization, class weights, and majority rules from train shards only."""
    return _derive_train_artifacts(train_shards, weight_cap)


def _validate_json_compatible(
    value: Any, path: str = "$", active_containers: set[int] | None = None
) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(
            f"train artifact metadata must contain only JSON-compatible values; "
            f"non-finite number at {path}"
        )
    if not isinstance(value, (dict, list)):
        raise ValueError(
            f"train artifact metadata must contain only JSON-compatible values; "
            f"got {type(value).__name__} at {path}"
        )

    active = active_containers if active_containers is not None else set()
    container_id = id(value)
    if container_id in active:
        raise ValueError(
            f"train artifact metadata must contain only JSON-compatible values; "
            f"cycle at {path}"
        )
    active.add(container_id)
    try:
        if isinstance(value, dict):
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "train artifact metadata must contain only JSON-compatible "
                        f"object keys; got {type(key).__name__} at {path}"
                    )
                _validate_json_compatible(nested, f"{path}.{key}", active)
        else:
            for index, nested in enumerate(value):
                _validate_json_compatible(nested, f"{path}[{index}]", active)
    finally:
        active.remove(container_id)


def _canonical_json_bytes(metadata: dict[str, Any]) -> bytes:
    _validate_json_compatible(metadata)
    try:
        return json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "train artifact metadata must contain only JSON-compatible values"
        ) from error


def _digest_part(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _artifact_identity(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in _ARTIFACT_FIELDS:
        array = arrays[name]
        _digest_part(digest, name.encode("utf-8"))
        _digest_part(digest, array.dtype.str.encode("ascii"))
        _digest_part(
            digest, json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        _digest_part(digest, np.ascontiguousarray(array).tobytes(order="C"))
    _digest_part(digest, b"metadata")
    _digest_part(digest, _canonical_json_bytes(metadata))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_metadata(metadata: dict[str, Any]) -> float:
    if not isinstance(metadata, dict):
        raise ValueError("train artifact metadata must be an object")
    if metadata.get("schema_version") != "ryo-bc-v0":
        raise ValueError("train artifact schema_version does not match ryo-bc-v0")
    if metadata.get("feature_schema_version") != "ryo-features-v0":
        raise ValueError("train artifact feature_schema_version does not match ryo-features-v0")
    if metadata.get("operations") != list(OPERATIONS):
        raise ValueError("train artifact operations do not match the fixed vocabulary")
    identities = metadata.get("train_shard_identities")
    if not isinstance(identities, list) or not identities or not all(
        _is_sha256(identity) for identity in identities
    ):
        raise ValueError("train artifact metadata requires train_shard_identities")
    if not _is_sha256(metadata.get("preparation_manifest_sha256")):
        raise ValueError("train artifact metadata requires preparation_manifest_sha256")
    weight_cap = metadata.get("weight_cap")
    if (
        isinstance(weight_cap, bool)
        or not isinstance(weight_cap, (int, float))
        or not math.isfinite(float(weight_cap))
        or float(weight_cap) < 1.0
    ):
        raise ValueError("train artifact metadata requires a valid weight_cap")
    _canonical_json_bytes(metadata)
    return float(weight_cap)


def _artifact_arrays(
    stats: NormalizationStats,
    class_counts: np.ndarray,
    class_weights: np.ndarray,
    majority: MajorityRules,
) -> dict[str, np.ndarray]:
    return {
        "global_mean": np.asarray(stats.global_mean),
        "global_std": np.asarray(stats.global_std),
        "actor_mean": np.asarray(stats.actor_mean),
        "actor_std": np.asarray(stats.actor_std),
        "class_counts": np.asarray(class_counts),
        "class_weights": np.asarray(class_weights),
        "majority_labels": np.asarray(
            [majority.global_label, majority.farmer_label, majority.hand_label],
            dtype=np.int64,
        ),
        "global_ranking": np.asarray(majority.global_ranking, dtype=np.int64),
        "farmer_ranking": np.asarray(majority.farmer_ranking, dtype=np.int64),
        "hand_ranking": np.asarray(majority.hand_ranking, dtype=np.int64),
    }


def _validate_artifact(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> tuple[NormalizationStats, np.ndarray, np.ndarray, MajorityRules]:
    expected = {
        "global_mean": (np.dtype(np.float32), (GLOBAL_DIM,)),
        "global_std": (np.dtype(np.float32), (GLOBAL_DIM,)),
        "actor_mean": (np.dtype(np.float32), (ACTOR_DIM,)),
        "actor_std": (np.dtype(np.float32), (ACTOR_DIM,)),
        "class_counts": (np.dtype(np.int64), (_CLASS_COUNT,)),
        "class_weights": (np.dtype(np.float32), (_CLASS_COUNT,)),
        "majority_labels": (np.dtype(np.int64), (3,)),
        "global_ranking": (np.dtype(np.int64), (_CLASS_COUNT,)),
        "farmer_ranking": (np.dtype(np.int64), (_CLASS_COUNT,)),
        "hand_ranking": (np.dtype(np.int64), (_CLASS_COUNT,)),
    }
    for name, (dtype, shape) in expected.items():
        array = arrays.get(name)
        if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape:
            actual = None if not isinstance(array, np.ndarray) else (array.dtype, array.shape)
            raise ValueError(f"train artifact {name} has invalid dtype/shape: {actual}")

    stats = NormalizationStats(
        arrays["global_mean"], arrays["global_std"], arrays["actor_mean"], arrays["actor_std"]
    )
    _validate_stats(stats)
    class_counts = arrays["class_counts"]
    if np.any(class_counts <= 0):
        raise ValueError("train artifact class_counts must contain 17 positive counts")
    weight_cap = _validate_metadata(metadata)
    class_weights = arrays["class_weights"]
    if (
        not np.all(np.isfinite(class_weights))
        or np.any(class_weights <= 0)
        or not np.isclose(class_weights.mean(dtype=np.float64), 1.0, rtol=1e-6, atol=1e-7)
        or float(class_weights.max()) > weight_cap
    ):
        raise ValueError("train artifact class_weights violate finite mean-one cap invariants")

    labels = arrays["majority_labels"]
    rankings = (
        arrays["global_ranking"], arrays["farmer_ranking"], arrays["hand_ranking"]
    )
    names = ("global", "farmer", "hand")
    for index, (name, ranking) in enumerate(zip(names, rankings)):
        order = tuple(int(value) for value in ranking)
        if set(order) != set(range(_CLASS_COUNT)):
            raise ValueError(f"train artifact {name}_ranking is not a permutation")
        if int(labels[index]) != order[0]:
            raise ValueError(f"train artifact {name}_label does not lead its ranking")
    majority = MajorityRules(
        int(labels[0]),
        int(labels[1]),
        int(labels[2]),
        tuple(int(value) for value in rankings[0]),
        tuple(int(value) for value in rankings[1]),
        tuple(int(value) for value in rankings[2]),
    )
    return stats, class_counts, class_weights, majority


def save_train_artifacts(
    path: Path,
    stats: NormalizationStats,
    class_counts: np.ndarray,
    class_weights: np.ndarray,
    majority: MajorityRules,
    metadata: dict[str, Any],
) -> str:
    """Atomically save logically identified train-derived artifacts."""
    path = Path(path)
    arrays = _artifact_arrays(stats, class_counts, class_weights, majority)
    _validate_artifact(arrays, metadata)
    identity = _artifact_identity(arrays, metadata)
    if not path.parent.is_dir():
        raise ValueError(f"train artifact parent directory does not exist path={path.parent}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                **arrays,
                metadata=np.asarray(_canonical_json_bytes(metadata).decode("utf-8")),
            )
        loaded = load_train_artifacts(temporary_path)
        loaded_arrays = _artifact_arrays(*loaded[:4])
        if _artifact_identity(loaded_arrays, loaded[4]) != identity:
            raise ValueError(f"temporary train artifact verification failed path={temporary_path}")
        if path.exists():
            existing = load_train_artifacts(path)
            existing_arrays = _artifact_arrays(*existing[:4])
            if _artifact_identity(existing_arrays, existing[4]) != identity:
                raise ValueError(f"refusing to replace non-identical train artifact path={path}")
            temporary_path.unlink()
            temporary_path = None
            return identity
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return identity


def load_train_artifacts(
    path: Path,
) -> tuple[NormalizationStats, np.ndarray, np.ndarray, MajorityRules, dict[str, Any]]:
    """Load and validate one frozen train-artifact archive."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected_fields = set(_ARTIFACT_FIELDS) | {"metadata"}
            if set(archive.files) != expected_fields:
                raise ValueError(
                    f"train artifact fields={sorted(archive.files)} "
                    f"expected={sorted(expected_fields)}"
                )
            metadata_array = archive["metadata"]
            if metadata_array.shape != () or metadata_array.dtype.kind != "U":
                raise ValueError("train artifact metadata must be a scalar Unicode array")
            metadata_text = str(metadata_array.item())
            try:
                metadata = json.loads(metadata_text)
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError("train artifact metadata is not valid JSON") from error
            if metadata_text != _canonical_json_bytes(metadata).decode("utf-8"):
                raise ValueError("train artifact metadata text is not canonical JSON")
            arrays = {name: np.array(archive[name], copy=True) for name in _ARTIFACT_FIELDS}
    except ValueError:
        raise
    except (OSError, EOFError) as error:
        raise ValueError(f"cannot read train artifact path={path}") from error
    stats, counts, weights, majority = _validate_artifact(arrays, metadata)
    return stats, counts, weights, majority, metadata


def train_artifact_identity(path: Path) -> str:
    """Return the verified logical identity of a frozen train artifact."""
    stats, counts, weights, majority, metadata = load_train_artifacts(path)
    return _artifact_identity(
        _artifact_arrays(stats, counts, weights, majority), metadata
    )


def validate_train_artifacts(
    path: Path,
    train_shards: Sequence[Path],
    expected_metadata: dict[str, Any],
    *,
    weight_cap: float,
) -> tuple[
    NormalizationStats,
    np.ndarray,
    np.ndarray,
    MajorityRules,
    dict[str, Any],
    str,
]:
    """Verify a frozen archive against independently re-derived train-only values."""
    stats, counts, weights, majority, metadata = load_train_artifacts(path)
    if _canonical_json_bytes(metadata) != _canonical_json_bytes(expected_metadata):
        raise ValueError(
            "train artifact metadata is not bound to the verified training inputs"
        )
    expected = _derive_train_artifacts(train_shards, weight_cap)
    actual_arrays = _artifact_arrays(stats, counts, weights, majority)
    expected_arrays = _artifact_arrays(*expected)
    mismatches = [
        name
        for name in _ARTIFACT_FIELDS
        if not np.array_equal(actual_arrays[name], expected_arrays[name])
    ]
    if mismatches:
        raise ValueError(
            f"train artifact values do not match verified train shards fields={mismatches}"
        )
    identity = _artifact_identity(actual_arrays, metadata)
    return stats, counts, weights, majority, metadata, identity


def _validate_stats(stats: NormalizationStats) -> None:
    expected = (
        ("global_mean", stats.global_mean, (GLOBAL_DIM,)),
        ("global_std", stats.global_std, (GLOBAL_DIM,)),
        ("actor_mean", stats.actor_mean, (ACTOR_DIM,)),
        ("actor_std", stats.actor_std, (ACTOR_DIM,)),
    )
    for name, array, shape in expected:
        if not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.shape != shape:
            actual = None if not isinstance(array, np.ndarray) else (array.dtype, array.shape)
            raise ValueError(f"normalization {name} has invalid dtype/shape: {actual}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"normalization {name} contains non-finite values")
    if np.any(stats.global_std <= 0) or np.any(stats.actor_std <= 0):
        raise ValueError("normalization standard deviations must be positive")


def _day_band(day: int) -> str:
    if not 0 <= day <= 29:
        raise ValueError(f"raw observation day is out of range: {day}")
    if day <= 6:
        return "days-1-7"
    if day <= 13:
        return "days-8-14"
    if day <= 20:
        return "days-15-21"
    return "days-22-plus"


class ShardDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Map shard sample rows lazily through a bounded LRU game cache."""

    def __init__(
        self, shard_paths: Sequence[Path], stats: NormalizationStats, cache_size: int = 2
    ) -> None:
        _validate_stats(stats)
        if (
            isinstance(cache_size, bool)
            or not isinstance(cache_size, int)
            or not 1 <= cache_size <= 2
        ):
            raise ValueError("cache_size must be a positive integer at most two")
        self.shard_paths = tuple(Path(path) for path in shard_paths)
        self.stats = stats
        self.cache_size = cache_size
        prefix = [0]
        for path in self.shard_paths:
            game = read_shard(path)
            prefix.append(prefix[-1] + int(game.label.shape[0]))
        self._prefix = tuple(prefix)
        self._cache: OrderedDict[int, EncodedGame] = OrderedDict()

    def __len__(self) -> int:
        return self._prefix[-1]

    def _game(self, shard_index: int) -> EncodedGame:
        cached = self._cache.pop(shard_index, None)
        if cached is not None:
            self._cache[shard_index] = cached
            return cached
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        game = read_shard(self.shard_paths[shard_index])
        self._cache[shard_index] = game
        return game

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        resolved = int(index)
        if resolved < 0:
            resolved += len(self)
        if not 0 <= resolved < len(self):
            raise IndexError(f"dataset index out of range: {index}")
        shard_index = bisect_right(self._prefix, resolved) - 1
        local_index = resolved - self._prefix[shard_index]
        game = self._game(shard_index)
        return _materialize_example(game, local_index, self.stats)


class EncodedGameDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Materialize rows only from already authenticated immutable game snapshots."""

    def __init__(
        self, games: Sequence[EncodedGame], stats: NormalizationStats
    ) -> None:
        _validate_stats(stats)
        snapshots = tuple(games)
        prefix = [0]
        for game in snapshots:
            if not isinstance(game, EncodedGame):
                raise TypeError("encoded game snapshots must be EncodedGame instances")
            for name in (
                "grid",
                "global_features",
                "actor_features",
                "step_index",
                "label",
                "argument_item",
                "argument_quantity",
            ):
                if getattr(game, name).flags.writeable:
                    raise ValueError("encoded game snapshots must be immutable")
            prefix.append(prefix[-1] + int(game.label.shape[0]))
        self.games = snapshots
        self.stats = stats
        self._prefix = tuple(prefix)

    def __len__(self) -> int:
        return self._prefix[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        resolved = int(index)
        if resolved < 0:
            resolved += len(self)
        if not 0 <= resolved < len(self):
            raise IndexError(f"dataset index out of range: {index}")
        game_index = bisect_right(self._prefix, resolved) - 1
        local_index = resolved - self._prefix[game_index]
        return _materialize_example(self.games[game_index], local_index, self.stats)


def _materialize_example(
    game: EncodedGame, local_index: int, stats: NormalizationStats
) -> dict[str, Any]:
    step = int(game.step_index[local_index])

    raw_global = game.global_features[step].copy()
    raw_actor = game.actor_features[local_index].copy()
    actor_flag = int(np.rint(float(raw_actor[0])))
    seat = int(np.rint(float(raw_global[5])))
    day = int(np.rint(float(raw_global[1]) * 29.0))
    if actor_flag not in (0, 1):
        raise ValueError(f"raw actor type is invalid: {raw_actor[0]}")
    if seat not in (0, 1):
        raise ValueError(f"raw seat is invalid: {raw_global[5]}")

    x, y = np.rint(raw_actor[2:4] * 9.0).astype(np.int64)
    if not (0 <= x < 10 and 0 <= y < 10):
        raise ValueError(f"raw actor coordinates are out of range: x={x} y={y}")
    grid = game.grid[step].copy()
    if np.any(grid[43] != 0):
        raise ValueError("opponent actor-position channel 43 must stay zero")
    grid[21, y, x] = 1.0
    if np.any(grid[43] != 0):
        raise ValueError("opponent actor-position channel 43 changed during injection")

    normalized_global = ((raw_global - stats.global_mean) / stats.global_std).astype(
        np.float32, copy=False
    )
    normalized_actor = ((raw_actor - stats.actor_mean) / stats.actor_std).astype(
        np.float32, copy=False
    )
    clock = np.concatenate(
        (normalized_global[:6], normalized_actor[:2]), dtype=np.float32
    )
    if clock.shape != (CLOCK_DIM,):
        raise ValueError(f"clock feature shape={clock.shape} expected={(CLOCK_DIM,)}")
    slices = {
        "actor_type": "farmer" if actor_flag == 1 else "hand",
        "seat": str(seat),
        "day_band": _day_band(day),
        "source_date": str(game.metadata["source_date"]),
        "route_family": str(game.metadata["route_family"]),
    }
    return {
        "grid": grid,
        "global_features": normalized_global,
        "actor_features": normalized_actor,
        "clock_features": clock,
        "label": np.int64(game.label[local_index]),
        "game_id": str(game.metadata["episode_id"]),
        "slices": slices,
    }


def collate_examples(rows: Sequence[dict[str, Any]]) -> Batch:
    """Stack materialized examples into the fixed training/evaluation batch contract."""
    if not rows:
        raise ValueError("cannot collate an empty example sequence")
    return Batch(
        grid=torch.from_numpy(np.stack([row["grid"] for row in rows])).float(),
        global_features=torch.from_numpy(
            np.stack([row["global_features"] for row in rows])
        ).float(),
        actor_features=torch.from_numpy(
            np.stack([row["actor_features"] for row in rows])
        ).float(),
        clock_features=torch.from_numpy(
            np.stack([row["clock_features"] for row in rows])
        ).float(),
        label=torch.from_numpy(np.asarray([row["label"] for row in rows], dtype=np.int64)),
        game_id=tuple(str(row["game_id"]) for row in rows),
        slices=tuple(dict(row["slices"]) for row in rows),
    )
