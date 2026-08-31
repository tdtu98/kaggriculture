"""Current-observation feature encoding and deterministic replay shards."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bc_core.constants import (
    ACTOR_DIM,
    ANIMALS,
    ARGUMENT_ITEMS,
    CROPS,
    GLOBAL_DIM,
    GRID_CHANNELS,
    KNOWN_SHOPS,
    OPERATIONS,
    PRODUCTS,
    SHOPS,
    TILE_KINDS,
)
from bc_core.replay import ReplayError, SourceReplay, iter_decisions, operation_and_arguments


class FeatureError(ValueError):
    """Raised when an observation or shard violates the feature contract."""


@dataclass(frozen=True)
class EncodedGame:
    grid: np.ndarray
    global_features: np.ndarray
    actor_features: np.ndarray
    step_index: np.ndarray
    label: np.ndarray
    argument_item: np.ndarray
    argument_quantity: np.ndarray
    metadata: dict[str, Any]


_QUADRANTS = ("NW", "NE", "SW", "SE")
_FARM_CHANNELS = GRID_CHANNELS // 2
_TILE_FEATURES = _FARM_CHANNELS - 1
_ARRAY_FIELDS = (
    "grid",
    "global_features",
    "actor_features",
    "step_index",
    "label",
    "argument_item",
    "argument_quantity",
)


def _number(value: Any, field: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise FeatureError(f"invalid number {field}={value!r} {context}")
    number = float(value)
    if not math.isfinite(number):
        raise FeatureError(f"non-finite number {field}={value!r} {context}")
    return number


def _count(value: Any, field: str, context: str) -> float:
    number = _number(value, field, context)
    if number < 0:
        raise FeatureError(f"negative count {field}={value!r} {context}")
    return number


def _log_count(value: Any, field: str, context: str) -> float:
    return math.log1p(_count(value, field, context))


def _log_clipped(value: Any, field: str, context: str) -> float:
    return math.log1p(max(0.0, _number(value, field, context)))


def _flag(value: Any, field: str, context: str) -> float:
    if value is True or value == 1:
        return 1.0
    if value is False or value == 0:
        return 0.0
    raise FeatureError(f"invalid flag {field}={value!r} {context}")


def _coordinate(value: Any, field: str, context: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in value)
    ):
        raise FeatureError(f"invalid coordinate {field}={value!r} {context}")
    x, y = int(value[0]), int(value[1])
    if not (0 <= x < 10 and 0 <= y < 10):
        raise FeatureError(f"out-of-range coordinate {field}={value!r} {context}")
    return x, y


def _mapping(value: Any, field: str, allowed: Sequence[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeatureError(f"invalid mapping {field} {context}")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise FeatureError(f"unexpected {field} category {unknown[0]} {context}")
    return value


def _tile_features(raw_tile: Any, day: float, context: str) -> np.ndarray:
    encoded = np.zeros(_TILE_FEATURES, dtype=np.float32)
    cursor = 0
    if raw_tile is None:
        kind = "EMPTY"
        tile: Mapping[str, Any] = {}
    elif isinstance(raw_tile, str):
        kind = raw_tile
        tile = {}
    elif isinstance(raw_tile, Mapping):
        kind = raw_tile.get("kind")
        tile = raw_tile
    else:
        raise FeatureError(f"invalid tile {raw_tile!r} {context}")
    if kind not in TILE_KINDS:
        raise FeatureError(f"unexpected tile kind {kind!r} {context}")
    encoded[cursor + TILE_KINDS.index(kind)] = 1.0
    cursor += len(TILE_KINDS)

    crop = tile.get("crop")
    if crop is not None:
        if crop not in CROPS:
            raise FeatureError(f"unexpected crop {crop!r} {context}")
        encoded[cursor + CROPS.index(crop)] = 1.0
    elif kind == "PLANT":
        raise FeatureError(f"missing crop for PLANT {context}")
    cursor += len(CROPS)

    animal = tile.get("animal")
    if animal is not None:
        if animal not in ANIMALS:
            raise FeatureError(f"unexpected animal {animal!r} {context}")
        encoded[cursor + ANIMALS.index(animal)] = 1.0
    cursor += len(ANIMALS)

    yield_units = _count(tile.get("yield_units", 0), "yield_units", context)
    encoded[cursor] = np.clip(yield_units / 6.0, 0.0, 1.0)
    cursor += 1
    encoded[cursor] = _flag(tile.get("watered_today", False), "watered_today", context)
    cursor += 1
    encoded[cursor] = _flag(tile.get("fed_today", False), "fed_today", context)
    cursor += 1
    encoded[cursor] = _flag(tile.get("cared_today", False), "cared_today", context)
    cursor += 1
    if "fertilizer_available" in tile:
        fertilizer = _flag(tile["fertilizer_available"], "fertilizer_available", context)
    elif kind == "PLANT":
        until = _number(tile.get("fertilized_until_day", -1), "fertilized_until_day", context)
        fertilizer = float(until >= day)
    else:
        fertilizer = 0.0
    encoded[cursor] = fertilizer
    cursor += 1
    consecutive_unwatered = _count(
        tile.get("consecutive_unwatered", 0), "consecutive_unwatered", context
    )
    encoded[cursor] = np.clip(consecutive_unwatered, 0.0, 1.0)
    cursor += 1
    consecutive_unfed = _count(tile.get("consecutive_unfed", 0), "consecutive_unfed", context)
    encoded[cursor] = np.clip(consecutive_unfed, 0.0, 1.0)
    cursor += 1
    if cursor != _TILE_FEATURES:
        raise FeatureError(f"tile feature cursor={cursor} expected={_TILE_FEATURES} {context}")
    return encoded


def _encode_farm_grid(farm: Mapping[str, Any], day: float, context: str) -> np.ndarray:
    encoded = np.zeros((_FARM_CHANNELS, 10, 10), dtype=np.float32)
    cursor = 0
    try:
        tiles = farm["tiles"]
    except (KeyError, TypeError) as error:
        raise FeatureError(f"missing farm tiles {context}") from error
    if not isinstance(tiles, list) or len(tiles) != 10 or any(
        not isinstance(row, list) or len(row) != 10 for row in tiles
    ):
        raise FeatureError(f"farm tiles must be 10x10 {context}")
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            encoded[:_TILE_FEATURES, y, x] = _tile_features(
                tile, day, f"{context} tile=({x},{y})"
            )
    cursor += _TILE_FEATURES
    # Reserved for per-sample actor injection by the dataset loader.
    cursor += 1
    if cursor != _FARM_CHANNELS:
        raise FeatureError(f"farm grid cursor={cursor} expected={_FARM_CHANNELS} {context}")
    return encoded


def _encode_actor(
    farm: Mapping[str, Any], inventory: Any, actor_index: int, day: float, context: str
) -> np.ndarray:
    encoded = np.zeros(ACTOR_DIM, dtype=np.float32)
    cursor = 0
    encoded[cursor] = float(actor_index == 0)
    cursor += 1
    encoded[cursor] = actor_index / 8.0
    cursor += 1
    try:
        position = farm["farmer"] if actor_index == 0 else farm["hands"][actor_index - 1]
    except (KeyError, IndexError, TypeError) as error:
        raise FeatureError(f"missing actor position actor={actor_index} {context}") from error
    x, y = _coordinate(position, f"actor[{actor_index}]", context)
    encoded[cursor : cursor + 2] = (x / 9.0, y / 9.0)
    cursor += 2
    actor_context = f"{context} actor={actor_index}"
    carried = _mapping(inventory, f"inventory[{actor_index}]", ARGUMENT_ITEMS, actor_context)
    for item in ARGUMENT_ITEMS:
        encoded[cursor] = _log_count(
            carried.get(item, 0), f"inventory.{item}", actor_context
        )
        cursor += 1
    encoded[cursor] = float((x, y) in {(4, 4), (5, 4), (4, 5), (5, 5)})
    cursor += 1
    try:
        tile = farm["tiles"][y][x]
    except (KeyError, IndexError, TypeError) as error:
        raise FeatureError(f"missing current tile actor={actor_index} {context}") from error
    tile_features = _tile_features(tile, day, f"{context} actor={actor_index} tile=({x},{y})")
    encoded[cursor : cursor + _TILE_FEATURES] = tile_features
    cursor += _TILE_FEATURES
    if cursor != ACTOR_DIM:
        raise FeatureError(f"actor feature cursor={cursor} expected={ACTOR_DIM} {context}")
    return encoded


def _encode_global(
    observation: Mapping[str, Any], seat: int, step: int, context: str
) -> np.ndarray:
    encoded = np.zeros(GLOBAL_DIM, dtype=np.float32)
    cursor = 0
    try:
        day = _number(observation["day"], "day", context)
        hour = _number(observation["hour"], "hour", context)
        farms = observation["farms"]
        private = observation["private"]
        market = observation["market"]
        town = observation["town"]
    except (KeyError, TypeError) as error:
        raise FeatureError(f"missing global state {context}") from error
    if not 0 <= step <= 719 or not 0 <= day <= 29 or not 0 <= hour <= 23:
        raise FeatureError(f"clock out of range step={step} day={day} hour={hour} {context}")
    if not isinstance(farms, list) or len(farms) != 2 or not all(isinstance(farm, Mapping) for farm in farms):
        raise FeatureError(f"expected two farms {context}")
    self_farm, opponent_farm = farms[seat], farms[1 - seat]
    encoded[cursor : cursor + 6] = (
        step / 719.0,
        day / 29.0,
        hour / 23.0,
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
        seat,
    )
    cursor += 6
    for name, farm in (("self", self_farm), ("opponent", opponent_farm)):
        encoded[cursor] = _log_clipped(farm.get("money"), f"{name}.money", context)
        cursor += 1
    for name, farm in (("self", self_farm), ("opponent", opponent_farm)):
        hands = farm.get("hands")
        if not isinstance(hands, list):
            raise FeatureError(f"invalid {name}.hands {context}")
        encoded[cursor] = len(hands)
        cursor += 1
        encoded[cursor] = _count(farm.get("hires_today"), f"{name}.hires_today", context)
        cursor += 1
    for name, farm in (("self", self_farm), ("opponent", opponent_farm)):
        quadrants = farm.get("unlocked_quadrants")
        if not isinstance(quadrants, list):
            raise FeatureError(f"invalid {name}.unlocked_quadrants {context}")
        unknown = sorted(set(quadrants) - set(_QUADRANTS))
        if unknown:
            raise FeatureError(f"unexpected quadrant {unknown[0]} {context}")
        for quadrant in _QUADRANTS:
            encoded[cursor] = float(quadrant in quadrants)
            cursor += 1
    if not isinstance(private, Mapping):
        raise FeatureError(f"invalid private state {context}")
    shed = _mapping(private.get("shed"), "shed", ARGUMENT_ITEMS, context)
    for item in ARGUMENT_ITEMS:
        encoded[cursor] = _log_count(shed.get(item, 0), f"shed.{item}", context)
        cursor += 1
    seeds = _mapping(private.get("seeds"), "seeds", CROPS, context)
    for crop in CROPS:
        encoded[cursor] = _log_count(seeds.get(crop, 0), f"seeds.{crop}", context)
        cursor += 1
    if not isinstance(market, Mapping):
        raise FeatureError(f"invalid market state {context}")
    inventory = _mapping(market.get("inventory"), "market.inventory", PRODUCTS, context)
    prices = _mapping(market.get("prices"), "market.prices", PRODUCTS, context)
    missing_market = [product for product in PRODUCTS if product not in inventory or product not in prices]
    if missing_market:
        raise FeatureError(f"missing market product {missing_market[0]} {context}")
    for product in PRODUCTS:
        encoded[cursor] = _log_count(
            inventory[product], f"market.inventory.{product}", context
        )
        cursor += 1
        encoded[cursor] = _log_clipped(prices[product], f"market.prices.{product}", context)
        cursor += 1
    if not isinstance(town, Mapping) or not isinstance(town.get("unlocked_shops"), list):
        raise FeatureError(f"invalid town unlocked_shops {context}")
    unlocked_shops = town["unlocked_shops"]
    for shop in unlocked_shops:
        if shop not in KNOWN_SHOPS:
            raise FeatureError(f"unexpected shop {shop!r} {context}")
    for shop in SHOPS:
        encoded[cursor] = unlocked_shops.count(shop)
        cursor += 1
    if cursor != GLOBAL_DIM:
        raise FeatureError(f"global feature cursor={cursor} expected={GLOBAL_DIM} {context}")
    return encoded


def encode_game(source: SourceReplay, replay: dict[str, Any]) -> EncodedGame:
    """Encode every Ryo current observation and shifted unit action in a replay."""
    grid = np.zeros((719, GRID_CHANNELS, 10, 10), dtype=np.float32)
    global_features = np.zeros((719, GLOBAL_DIM), dtype=np.float32)
    actor_rows: list[np.ndarray] = []
    step_rows: list[int] = []
    labels: list[int] = []
    argument_items: list[int] = []
    argument_quantities: list[int] = []
    ryo_seat: int | None = None

    for decision in iter_decisions(source, replay):
        ryo_seat = decision.seat
        context = (
            f"split={source.split} episode={source.episode_id} "
            f"step={decision.step} seat={decision.seat}"
        )
        observation = decision.observation
        try:
            farms = observation["farms"]
            day = _number(observation["day"], "day", context)
            private = observation["private"]
            inventories = private["inventories"]
        except (KeyError, IndexError, TypeError) as error:
            raise FeatureError(f"missing feature state {context}") from error
        if not isinstance(farms, list) or len(farms) != 2:
            raise FeatureError(f"expected two farms {context}")
        self_farm = farms[decision.seat]
        opponent_farm = farms[1 - decision.seat]
        if not isinstance(self_farm, Mapping) or not isinstance(opponent_farm, Mapping):
            raise FeatureError(f"invalid farm state {context}")
        hands = self_farm.get("hands")
        if not isinstance(hands, list):
            raise FeatureError(f"invalid self hands {context}")
        if not isinstance(inventories, list) or len(inventories) != 1 + len(hands):
            actual = len(inventories) if isinstance(inventories, list) else "invalid"
            raise FeatureError(
                f"private inventories length={actual} expected={1 + len(hands)} {context}"
            )

        grid[decision.step, :_FARM_CHANNELS] = _encode_farm_grid(self_farm, day, context)
        grid[decision.step, _FARM_CHANNELS:] = _encode_farm_grid(opponent_farm, day, context)
        global_features[decision.step] = _encode_global(
            observation, decision.seat, decision.step, context
        )

        unit_actions = [decision.action["farmer"], *decision.action["hands"]]
        for actor_index, (unit_action, inventory) in enumerate(zip(unit_actions, inventories)):
            actor_rows.append(_encode_actor(self_farm, inventory, actor_index, day, context))
            try:
                operation, item, quantity = operation_and_arguments(unit_action)
            except ReplayError as error:
                raise FeatureError(f"{error} {context} actor={actor_index}") from error
            step_rows.append(decision.step)
            labels.append(operation)
            argument_items.append(item)
            argument_quantities.append(quantity)

    if ryo_seat is None:
        raise FeatureError(f"replay yielded no decisions split={source.split} episode={source.episode_id}")
    actor_features = np.asarray(actor_rows, dtype=np.float32).reshape(-1, ACTOR_DIM)
    step_index = np.asarray(step_rows, dtype=np.int32)
    label = np.asarray(labels, dtype=np.int64)
    argument_item = np.asarray(argument_items, dtype=np.int32)
    argument_quantity = np.asarray(argument_quantities, dtype=np.int32)
    metadata = {
        "schema_version": "ryo-features-v0",
        "split": source.split,
        "episode_id": source.episode_id,
        "ryo_seat": ryo_seat,
        "source_path": str(source.path),
        "source_sha256": source.sha256,
        "source_date": source.source_date,
        "route_family": source.route_family,
        "sample_count": int(label.shape[0]),
        "shapes": {
            "grid": list(grid.shape),
            "global_features": list(global_features.shape),
            "actor_features": list(actor_features.shape),
            "step_index": list(step_index.shape),
            "label": list(label.shape),
        },
    }
    return EncodedGame(
        grid,
        global_features,
        actor_features,
        step_index,
        label,
        argument_item,
        argument_quantity,
        metadata,
    )


def _validate_game(game: EncodedGame, context: str) -> None:
    expected_dtypes = {
        "grid": np.dtype(np.float32),
        "global_features": np.dtype(np.float32),
        "actor_features": np.dtype(np.float32),
        "step_index": np.dtype(np.int32),
        "label": np.dtype(np.int64),
        "argument_item": np.dtype(np.int32),
        "argument_quantity": np.dtype(np.int32),
    }
    for field, expected_dtype in expected_dtypes.items():
        array = getattr(game, field)
        if not isinstance(array, np.ndarray):
            raise FeatureError(f"{field} must be an array {context}")
        if array.dtype != expected_dtype:
            raise FeatureError(f"{field} dtype={array.dtype} expected={expected_dtype} {context}")
    if game.grid.shape != (719, GRID_CHANNELS, 10, 10):
        raise FeatureError(
            f"grid shape={game.grid.shape} expected={(719, GRID_CHANNELS, 10, 10)} {context}"
        )
    if game.global_features.shape != (719, GLOBAL_DIM):
        raise FeatureError(
            f"global_features shape={game.global_features.shape} expected={(719, GLOBAL_DIM)} {context}"
        )
    if game.actor_features.ndim != 2 or game.actor_features.shape[1] != ACTOR_DIM:
        raise FeatureError(f"actor_features shape={game.actor_features.shape} expected=(*,{ACTOR_DIM}) {context}")
    sample_count = game.actor_features.shape[0]
    for field in ("step_index", "label", "argument_item", "argument_quantity"):
        array = getattr(game, field)
        if array.shape != (sample_count,):
            raise FeatureError(f"{field} shape={array.shape} expected={(sample_count,)} {context}")
    for field in ("grid", "global_features", "actor_features"):
        if not np.all(np.isfinite(getattr(game, field))):
            raise FeatureError(f"{field} contains non-finite values {context}")
    if sample_count and (np.min(game.step_index) < 0 or np.max(game.step_index) >= 719):
        raise FeatureError(f"step_index contains out-of-range values {context}")
    if sample_count and (np.min(game.label) < 0 or np.max(game.label) >= len(OPERATIONS)):
        raise FeatureError(f"label contains out-of-range values {context}")
    if sample_count and (
        np.min(game.argument_item) < -1 or np.max(game.argument_item) >= len(ARGUMENT_ITEMS)
    ):
        raise FeatureError(f"argument_item contains out-of-range values {context}")
    if sample_count and np.min(game.argument_quantity) < -1:
        raise FeatureError(f"argument_quantity contains out-of-range values {context}")

    metadata = game.metadata
    required_metadata = {
        "schema_version",
        "split",
        "episode_id",
        "ryo_seat",
        "source_path",
        "source_sha256",
        "source_date",
        "route_family",
        "sample_count",
        "shapes",
    }
    if not isinstance(metadata, dict) or set(metadata) != required_metadata:
        raise FeatureError(f"metadata fields do not match feature schema {context}")
    if metadata.get("schema_version") != "ryo-features-v0":
        raise FeatureError(f"metadata schema_version is invalid {context}")
    if metadata.get("ryo_seat") not in (0, 1):
        raise FeatureError(f"metadata ryo_seat is invalid {context}")
    if metadata.get("sample_count") != sample_count:
        raise FeatureError(f"metadata sample_count does not match arrays {context}")
    expected_shapes = {
        "grid": list(game.grid.shape),
        "global_features": list(game.global_features.shape),
        "actor_features": list(game.actor_features.shape),
        "step_index": list(game.step_index.shape),
        "label": list(game.label.shape),
    }
    if metadata.get("shapes") != expected_shapes:
        raise FeatureError(f"metadata shapes do not match arrays {context}")


def _canonical_metadata_bytes(metadata: dict[str, Any], context: str) -> bytes:
    try:
        return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FeatureError(f"metadata is not canonical JSON {context}") from error


def _digest_part(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def logical_shard_identity(game: EncodedGame) -> str:
    """Return a stable SHA-256 over logical arrays and canonical metadata."""
    _validate_game(game, "logical shard")
    digest = hashlib.sha256()
    for field in _ARRAY_FIELDS:
        array = getattr(game, field)
        _digest_part(digest, field.encode("utf-8"))
        _digest_part(digest, array.dtype.str.encode("ascii"))
        _digest_part(digest, json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        _digest_part(digest, np.ascontiguousarray(array).tobytes(order="C"))
    _digest_part(digest, b"metadata")
    _digest_part(digest, _canonical_metadata_bytes(game.metadata, "logical shard"))
    return digest.hexdigest()


def read_shard(path: Path) -> EncodedGame:
    """Load and strictly validate one encoded replay shard."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected_fields = set(_ARRAY_FIELDS) | {"metadata"}
            if set(archive.files) != expected_fields:
                raise FeatureError(
                    f"shard fields={sorted(archive.files)} expected={sorted(expected_fields)} path={path}"
                )
            metadata_array = archive["metadata"]
            if metadata_array.shape != () or metadata_array.dtype.kind != "U":
                raise FeatureError(f"metadata must be a scalar Unicode array path={path}")
            try:
                metadata = json.loads(str(metadata_array.item()))
            except (json.JSONDecodeError, TypeError) as error:
                raise FeatureError(f"metadata is not valid JSON path={path}") from error
            arrays = {field: np.array(archive[field], copy=True) for field in _ARRAY_FIELDS}
    except FeatureError:
        raise
    except (OSError, ValueError, EOFError) as error:
        raise FeatureError(f"cannot read shard path={path}") from error
    game = EncodedGame(
        arrays["grid"],
        arrays["global_features"],
        arrays["actor_features"],
        arrays["step_index"],
        arrays["label"],
        arrays["argument_item"],
        arrays["argument_quantity"],
        metadata,
    )
    _validate_game(game, f"path={path}")
    return game


def write_shard(game: EncodedGame, path: Path, *, allow_identical: bool = True) -> str:
    """Atomically write a shard, refusing to replace different logical contents."""
    identity = logical_shard_identity(game)
    if not path.parent.is_dir():
        raise FeatureError(f"shard parent directory does not exist path={path.parent}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                **{field: getattr(game, field) for field in _ARRAY_FIELDS},
                metadata=np.asarray(
                    _canonical_metadata_bytes(game.metadata, f"path={path}").decode("utf-8")
                ),
            )
        written_identity = logical_shard_identity(read_shard(temporary_path))
        if written_identity != identity:
            raise FeatureError(f"temporary shard verification failed path={temporary_path}")
        if path.exists():
            if not allow_identical:
                raise FeatureError(f"shard already exists path={path}")
            existing_identity = logical_shard_identity(read_shard(path))
            if existing_identity != identity:
                raise FeatureError(f"refusing to replace non-identical shard path={path}")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return identity
