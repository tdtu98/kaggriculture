"""Validate Kaggriculture replays and extract compact strategy evidence."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import zlib
from collections import Counter
from pathlib import Path


REQUIRED_MODULE_VERSION = "1.32.7"
EXPECTED_CONFIGURATION = {"episodeSteps": 720, "turnsPerDay": 24}
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


class ReplayError(RuntimeError):
    """Raised when replay evidence is malformed or incompatible."""


def stable_json(payload) -> str:
    """Render stable, human-readable JSON with one trailing newline."""
    return json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"


def _validate_action(action, source: Path, step: int, seat: int) -> None:
    if not isinstance(action, dict):
        raise ReplayError(
            f"{source}: step {step} seat {seat} action is not an object"
        )
    farmer = action.get("farmer")
    hands = action.get("hands")
    market = action.get("market")
    if not isinstance(farmer, list) or not farmer:
        raise ReplayError(
            f"{source}: step {step} seat {seat} farmer action is invalid"
        )
    if not isinstance(hands, list):
        raise ReplayError(
            f"{source}: step {step} seat {seat} hands action is invalid"
        )
    if not isinstance(market, list):
        raise ReplayError(
            f"{source}: step {step} seat {seat} market action is invalid"
        )
    if any(not isinstance(order, list) or not order for order in hands):
        raise ReplayError(
            f"{source}: step {step} seat {seat} contains an invalid hand order"
        )
    if any(not isinstance(order, list) or not order for order in market):
        raise ReplayError(
            f"{source}: step {step} seat {seat} contains an invalid market order"
        )


def validate_replay(replay: dict, source: Path | str = "<memory>") -> None:
    """Require the exact supported replay shape and environment version."""
    source = Path(source)
    if not isinstance(replay, dict):
        raise ReplayError(f"{source}: replay root is not an object")
    for key in ("module_version", "configuration", "info", "rewards", "steps"):
        if key not in replay:
            raise ReplayError(f"{source}: missing required field {key}")
    if replay["module_version"] != REQUIRED_MODULE_VERSION:
        raise ReplayError(
            f"{source}: module_version={replay['module_version']!r}; "
            f"expected {REQUIRED_MODULE_VERSION!r}"
        )

    configuration = replay["configuration"]
    if not isinstance(configuration, dict):
        raise ReplayError(f"{source}: configuration is not an object")
    for key, expected in EXPECTED_CONFIGURATION.items():
        if configuration.get(key) != expected:
            raise ReplayError(
                f"{source}: configuration {key}={configuration.get(key)!r}; "
                f"expected {expected!r}"
            )

    info = replay["info"]
    names = info.get("TeamNames") if isinstance(info, dict) else None
    if not isinstance(names, list) or len(names) != 2:
        raise ReplayError(f"{source}: expected exactly two TeamNames")
    rewards = replay["rewards"]
    if not isinstance(rewards, list) or len(rewards) != 2:
        raise ReplayError(f"{source}: expected exactly two rewards")
    steps = replay["steps"]
    if not isinstance(steps, list) or len(steps) != 720:
        count = len(steps) if isinstance(steps, list) else "non-list"
        raise ReplayError(f"{source}: expected 720 states, found {count}")

    for step, states in enumerate(steps):
        if not isinstance(states, list) or len(states) != 2:
            raise ReplayError(
                f"{source}: step {step} does not contain two player states"
            )
        for seat, state in enumerate(states):
            if not isinstance(state, dict):
                raise ReplayError(
                    f"{source}: step {step} seat {seat} state is invalid"
                )
            for key in ("action", "observation", "reward", "status"):
                if key not in state:
                    raise ReplayError(
                        f"{source}: step {step} seat {seat} missing {key}"
                    )
            if not isinstance(state["observation"], dict):
                raise ReplayError(
                    f"{source}: step {step} seat {seat} observation is invalid"
                )
            _validate_action(state["action"], source, step, seat)


def load_replay(path: Path) -> dict:
    """Load one JSON replay and fail rather than silently skipping it."""
    path = Path(path)
    try:
        replay = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"{path}: unable to read replay: {exc}") from exc
    validate_replay(replay, path)
    return replay


def shifted_actions(replay: dict, seat: int) -> list[dict]:
    """Align stored actions with the preceding observation that produced them."""
    if seat not in (0, 1):
        raise ReplayError(f"seat must be 0 or 1, got {seat!r}")
    return [
        copy.deepcopy(states[seat]["action"])
        for states in replay["steps"][1:]
    ] + [copy.deepcopy(PASS_ACTION)]


def _quantity(order: list) -> int:
    if len(order) < 3:
        return 1
    try:
        return max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0


def _canonical_tile(tile):
    if tile is None or tile == "LOCKED":
        return tile
    if not isinstance(tile, dict):
        return str(tile)
    return {
        key: tile.get(key)
        for key in (
            "kind",
            "crop",
            "animal",
            "yield_units",
            "watered_today",
            "fed_today",
        )
        if tile.get(key) is not None
    }


def _sparse_tiles(farm: dict) -> dict:
    tiles = farm.get("tiles", [])
    sparse = {}
    for y, row in enumerate(tiles):
        if not isinstance(row, list):
            continue
        for x, tile in enumerate(row):
            if tile not in (None, "LOCKED"):
                sparse[f"{x},{y}"] = _canonical_tile(tile)
    return sparse


def _sorted_counts(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): value[key]
        for key in sorted(value)
        if isinstance(value[key], (int, float)) and value[key] != 0
    }


def _canonical_state(observation: dict, seat: int, purchases: Counter) -> dict:
    farms = observation.get("farms", [])
    farm = farms[seat] if isinstance(farms, list) and seat < len(farms) else {}
    private = observation.get("private", {})
    inventories = private.get("inventories", []) if isinstance(private, dict) else []
    return {
        "farmer": list(farm.get("farmer", [])),
        "hands": [list(position) for position in farm.get("hands", [])],
        "inventories": [
            _sorted_counts(inventory) for inventory in inventories
        ],
        "unlocked_quadrants": sorted(farm.get("unlocked_quadrants", [])),
        "tiles": _sparse_tiles(farm),
        "seeds": _sorted_counts(private.get("seeds", {})),
        "shed": _sorted_counts(private.get("shed", {})),
        "purchases": dict(sorted(purchases.items())),
        "money": farm.get("money"),
        "hires_today": farm.get("hires_today", 0),
        "controller": {},
    }


def _tile_at(farm: dict, position):
    try:
        x, y = int(position[0]), int(position[1])
        return farm["tiles"][y][x]
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _comparison_field(action: dict, observation: dict, seat: int) -> list[list]:
    orders = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    farms = observation.get("farms", [])
    farm = farms[seat] if isinstance(farms, list) and seat < len(farms) else {}
    positions = [farm.get("farmer", []), *farm.get("hands", [])]
    compared = []
    for index, order in enumerate(orders):
        normalized = list(order)
        if normalized and normalized[0] == "DIG" and index < len(positions):
            tile = _tile_at(farm, positions[index])
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                normalized = ["WEED_ONLY"]
        compared.append(normalized)
    return compared


def _normalize_market(order: list) -> list:
    if order and order[0] == "SELL":
        return list(order[:2])
    return list(order)


def extract_seat_record(
    replay: dict, source: Path | str, seat: int
) -> dict:
    """Extract one compact, deterministic full-season seat record."""
    validate_replay(replay, source)
    if seat not in (0, 1):
        raise ReplayError(f"seat must be 0 or 1, got {seat!r}")
    actions = shifted_actions(replay, seat)
    names = replay["info"]["TeamNames"]
    rewards = replay["rewards"]
    winner_seat = None
    if rewards[0] != rewards[1]:
        winner_seat = 0 if rewards[0] > rewards[1] else 1

    operation_counts = Counter()
    purchase_totals = Counter()
    sale_totals = Counter()
    market_operation_counts = Counter()
    first_purchase_steps = {}
    cumulative_purchases = Counter()
    shop_sequence = []
    previous_shops = []
    canonical_states = {}
    comparison_timeline = []

    for step, action_value in enumerate(actions):
        observation = replay["steps"][step][seat]["observation"]
        canonical_states[str(step)] = _canonical_state(
            observation, seat, cumulative_purchases
        )
        actor_orders = [
            action_value.get("farmer", ["PASS"]),
            *action_value.get("hands", []),
        ]
        for order in actor_orders:
            if order:
                operation_counts[str(order[0])] += 1
        for order in action_value.get("market", []):
            operation = str(order[0])
            market_operation_counts[operation] += 1
            item = str(order[1]) if len(order) > 1 else operation
            quantity = _quantity(order)
            if operation == "HIRE":
                cumulative_purchases["HIRE"] += 1
            elif operation == "BUY_LAND":
                cumulative_purchases["LAND"] += 1
            elif operation in ("BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT"):
                purchase_key = item
                if operation == "BUY_SEED":
                    purchase_key = f"{item}_SEED"
                elif operation == "BUY_PRODUCT":
                    purchase_key = f"{item}_PRODUCT"
                cumulative_purchases[purchase_key] += quantity
                purchase_totals[purchase_key] += quantity
                first_purchase_steps.setdefault(purchase_key, step)
            if operation == "SELL" and len(order) > 1:
                sale_totals[item] += quantity

        shops = observation.get("town", {}).get("unlocked_shops", [])
        shops = list(shops) if isinstance(shops, list) else []
        if shops != previous_shops:
            shop_sequence.append({"step": step, "shops": shops})
            previous_shops = shops

        comparison_timeline.append(
            {
                "field": _comparison_field(action_value, observation, seat),
                "market": [
                    _normalize_market(order)
                    for order in action_value.get("market", [])
                ],
            }
        )

    route_bytes = json.dumps(
        comparison_timeline,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    replay_bytes = json.dumps(
        replay,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "source": Path(source).name,
        "source_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "seat": seat,
        "team": names[seat],
        "opponent": names[1 - seat],
        "team_signature": " vs ".join(sorted(str(name) for name in names)),
        "winner_team": names[winner_seat] if winner_seat is not None else None,
        "won": winner_seat == seat,
        "final_margin": rewards[seat] - rewards[1 - seat],
        "shop_sequence": shop_sequence,
        "actions": actions,
        "comparison_timeline": comparison_timeline,
        "canonical_states": canonical_states,
        "features": {
            "operation_counts": dict(sorted(operation_counts.items())),
            "market_operation_counts": dict(
                sorted(market_operation_counts.items())
            ),
            "purchase_totals": dict(sorted(purchase_totals.items())),
            "sale_totals": dict(sorted(sale_totals.items())),
            "first_purchase_steps": dict(sorted(first_purchase_steps.items())),
            "hire_count": cumulative_purchases["HIRE"],
            "land_count": cumulative_purchases["LAND"],
        },
        "route_signature": hashlib.sha256(route_bytes).hexdigest(),
    }


_SPLIT_FIELDS = (
    "winner_team",
    "winner_seat",
    "opponent_pair",
    "shop_signature",
    "core_family",
)


def stratified_split(replay_summaries: list[dict]) -> dict[str, str]:
    """Assign exactly one third of 90 replays to deterministic holdout."""
    if len(replay_summaries) != 90:
        raise ReplayError(
            f"deterministic split requires exactly 90 replays, "
            f"found {len(replay_summaries)}"
        )
    sources = [str(row.get("source")) for row in replay_summaries]
    if len(set(sources)) != len(sources):
        raise ReplayError("deterministic split requires unique sources")

    rows = sorted(replay_summaries, key=lambda row: str(row["source"]))
    labels_by_source = {}
    totals = Counter()
    for row in rows:
        labels = tuple(
            (field, json.dumps(row.get(field), ensure_ascii=False, sort_keys=True))
            for field in _SPLIT_FIELDS
        )
        labels_by_source[str(row["source"])] = labels
        totals.update(labels)

    selected = set()
    selected_counts = Counter()
    for _ in range(30):
        best = None
        for source in sources:
            if source in selected:
                continue
            candidate_labels = labels_by_source[source]
            error = 0.0
            for label, total in totals.items():
                projected = selected_counts[label]
                if label in candidate_labels:
                    projected += 1
                target = total / 3.0
                error += ((projected - target) ** 2) / total
            tie_break = hashlib.sha256(source.encode("utf-8")).hexdigest()
            score = (error, tie_break, source)
            if best is None or score < best[0]:
                best = (score, source)
        source = best[1]
        selected.add(source)
        selected_counts.update(labels_by_source[source])

    return {
        source: "holdout" if source in selected else "discovery"
        for source in sorted(sources)
    }


def _core_family(record: dict) -> str:
    purchases = record["features"]["purchase_totals"]
    fields = (
        ("c", "COW"),
        ("s", "SHEEP"),
        ("g", "GOOSE"),
        ("w", "WHEAT_SEED"),
        ("t", "TOMATO_SEED"),
        ("st", "STRAWBERRY_SEED"),
        ("m", "MELON_SEED"),
    )
    parts = [f"{short}{int(purchases.get(item, 0))}" for short, item in fields]
    parts.extend(
        (
            f"h{int(record['features']['hire_count'])}",
            f"l{int(record['features']['land_count'])}",
        )
    )
    return "-".join(parts)


def _payload(value) -> tuple[str, str]:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return (
        base64.b85encode(compressed).decode("ascii"),
        hashlib.sha256(raw).hexdigest(),
    )


def decode_trace_payload(payload: str) -> dict:
    """Decode one deterministic catalog trace payload."""
    try:
        raw = zlib.decompress(base64.b85decode(payload.encode("ascii")))
        decoded = json.loads(raw)
    except (ValueError, zlib.error, json.JSONDecodeError) as exc:
        raise ReplayError(f"invalid trace payload: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ReplayError("invalid trace payload root")
    return decoded


def _compact_seat_record(record: dict, source_sha256: str) -> dict:
    trace_payload, trace_sha256 = _payload(
        {
            "actions": record["actions"],
            "comparison_timeline": record["comparison_timeline"],
        }
    )
    state_fingerprints = []
    for step in range(720):
        state = record["canonical_states"][str(step)]
        raw = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        state_fingerprints.append(hashlib.sha256(raw).hexdigest())
    return {
        "source": record["source"],
        "source_sha256": source_sha256,
        "seat": record["seat"],
        "team": record["team"],
        "opponent": record["opponent"],
        "team_signature": record["team_signature"],
        "winner_team": record["winner_team"],
        "won": record["won"],
        "final_margin": record["final_margin"],
        "shop_sequence": record["shop_sequence"],
        "features": record["features"],
        "route_signature": record["route_signature"],
        "core_family": _core_family(record),
        "trace_payload": trace_payload,
        "trace_sha256": trace_sha256,
        "state_fingerprints": state_fingerprints,
    }


def build_catalog(replay_dir: Path) -> dict:
    """Build stable compact evidence for all 90 accepted replay files."""
    replay_dir = Path(replay_dir)
    paths = sorted(replay_dir.glob("*.json"))
    if len(paths) != 90:
        raise ReplayError(
            f"catalog requires exactly 90 JSON files, found {len(paths)}"
        )

    replays = []
    seat_records = []
    split_inputs = []
    for path in paths:
        try:
            raw_bytes = path.read_bytes()
            replay = json.loads(raw_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayError(f"{path}: unable to read replay: {exc}") from exc
        validate_replay(replay, path)
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        records = [
            extract_seat_record(replay, path.name, seat) for seat in (0, 1)
        ]
        compact_records = [
            _compact_seat_record(record, source_sha256) for record in records
        ]
        seat_records.extend(compact_records)

        rewards = replay["rewards"]
        winner_seat = None
        if rewards[0] != rewards[1]:
            winner_seat = 0 if rewards[0] > rewards[1] else 1
        names = replay["info"]["TeamNames"]
        shop_raw = json.dumps(
            records[0]["shop_sequence"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        winner_record = records[winner_seat] if winner_seat is not None else records[0]
        summary = {
            "source": path.name,
            "source_sha256": source_sha256,
            "module_version": replay["module_version"],
            "teams": list(names),
            "rewards": list(rewards),
            "winner_team": names[winner_seat] if winner_seat is not None else None,
            "winner_seat": winner_seat,
            "opponent_pair": " vs ".join(sorted(str(name) for name in names)),
            "shop_signature": hashlib.sha256(shop_raw).hexdigest()[:16],
            "core_family": winner_record["core_family"]
            if "core_family" in winner_record
            else _core_family(winner_record),
        }
        replays.append(summary)
        split_inputs.append(summary)

    assignment = stratified_split(split_inputs)
    for replay in replays:
        replay["split"] = assignment[replay["source"]]
    for record in seat_records:
        record["split"] = assignment[record["source"]]

    return {
        "schema_version": 1,
        "required_module_version": REQUIRED_MODULE_VERSION,
        "replay_count": len(replays),
        "seat_record_count": len(seat_records),
        "split_counts": {"discovery": 60, "holdout": 30},
        "replays": sorted(replays, key=lambda row: row["source"]),
        "seat_records": sorted(
            seat_records, key=lambda row: (row["source"], row["seat"])
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    catalog = build_catalog(args.replay_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stable_json(catalog))
    print(
        f"cataloged {catalog['replay_count']} replays, "
        f"{catalog['seat_record_count']} seat records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
