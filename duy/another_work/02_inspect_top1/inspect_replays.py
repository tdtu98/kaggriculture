"""Validate and inspect public Kaggriculture replay JSON files."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_CONFIGURATION = {
    "startingMoney": 3000,
    "episodeSteps": 720,
    "turnsPerDay": 24,
    "townCenterSellInterval": 24,
}
REQUIRED_MODULE_VERSION = "1.32.7"
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
SUPPORTED_BRANCHES = (
    "c10-s4-straw42-melon12",
    "c8-s6-straw42-melon12",
    "c6-s8-straw42-melon12",
    "c6-s12-straw42-melon12",
)
DEFAULT_BRANCH = "c10-s4-straw42-melon12"
HANDOFF_STEPS = {
    "c6-s12-straw42-melon12": 144,
    "c6-s8-straw42-melon12": 216,
    "c8-s6-straw42-melon12": 216,
}


class ReplayError(RuntimeError):
    """Raised when replay evidence is malformed or ambiguous."""


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
            f"{source}: step {step} seat {seat} has invalid farmer action"
        )
    if not isinstance(hands, list):
        raise ReplayError(
            f"{source}: step {step} seat {seat} hands is not a list"
        )
    if not isinstance(market, list):
        raise ReplayError(
            f"{source}: step {step} seat {seat} market is not a list"
        )


def validate_replay(replay: dict, source: Path | str = "<memory>") -> None:
    """Validate the replay shape and competition configuration."""
    source = Path(source)
    if not isinstance(replay, dict):
        raise ReplayError(f"{source}: replay root is not an object")
    for key in ("configuration", "info", "rewards", "steps"):
        if key not in replay:
            raise ReplayError(f"{source}: missing required field {key}")

    configuration = replay["configuration"]
    if not isinstance(configuration, dict):
        raise ReplayError(f"{source}: configuration is not an object")
    for key, expected in EXPECTED_CONFIGURATION.items():
        actual = configuration.get(key)
        if actual != expected:
            raise ReplayError(
                f"{source}: configuration {key}={actual!r}; "
                f"expected {expected!r}"
            )

    names = replay["info"].get("TeamNames")
    if not isinstance(names, list) or len(names) != 2:
        raise ReplayError(f"{source}: expected exactly two TeamNames")
    if not isinstance(replay["rewards"], list) or len(replay["rewards"]) != 2:
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
            _validate_action(state["action"], source, step, seat)


def load_replay(path: Path) -> dict:
    """Load and validate one replay file."""
    path = Path(path)
    try:
        replay = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"{path}: unable to read replay: {exc}") from exc
    validate_replay(replay, path)
    return replay


def replay_module_version(
    replay: dict, source: Path | str = "<memory>"
) -> str:
    """Return the replay's required root module version."""
    version = replay.get("module_version") if isinstance(replay, dict) else None
    if not isinstance(version, str) or not version:
        raise ReplayError(f"{Path(source)}: missing root module_version")
    return version


def load_compatible_replays(
    paths,
    required_module_version: str = REQUIRED_MODULE_VERSION,
) -> tuple[list[tuple[Path, dict]], list[dict]]:
    """Load compatible replays and report valid older-version rejections."""
    accepted = []
    rejected = []
    for path in sorted(Path(path) for path in paths):
        replay = load_replay(path)
        version = replay_module_version(replay, path)
        if version != required_module_version:
            rejected.append(
                {
                    "source": path.name,
                    "module_version": version,
                    "reason": "module_version_mismatch",
                }
            )
            continue
        accepted.append((path, replay))
    return accepted, rejected


def find_seat(
    replay: dict,
    team_name: str,
    self_seat: int | None = None,
) -> int:
    """Return the requested team seat, requiring a seat for self-play."""
    names = replay.get("info", {}).get("TeamNames", [])
    matches = [seat for seat, name in enumerate(names) if name == team_name]
    if not matches:
        raise ReplayError(f"team {team_name!r} not found")
    if len(matches) == 1:
        return matches[0]
    if self_seat not in matches:
        raise ReplayError(
            f"self-play replay for {team_name!r} requires explicit self-play "
            "seat 0 or 1"
        )
    return int(self_seat)


def shifted_actions(replay: dict, seat: int) -> list[dict]:
    """Align stored replay actions with the observations that produced them."""
    if seat not in (0, 1):
        raise ReplayError(f"seat must be 0 or 1, got {seat!r}")
    actions = [
        copy.deepcopy(states[seat]["action"])
        for states in replay["steps"][1:]
    ]
    actions.append(copy.deepcopy(PASS_ACTION))
    return actions


def _quantity(order: list) -> int:
    if len(order) < 3:
        return 1
    try:
        return max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0


def _actor_orders(action: dict) -> list[list]:
    return [
        list(action.get("farmer") or ["PASS"]),
        *[list(order or ["PASS"]) for order in action.get("hands", [])],
    ]


def _actor_positions(farm: dict) -> list[list]:
    return [farm.get("farmer", [4, 4]), *list(farm.get("hands", []))]


def _tile_at(farm: dict, position) -> object:
    try:
        x, y = int(position[0]), int(position[1])
        return farm["tiles"][y][x]
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _same_turn_deposits(action: dict) -> Counter:
    deposits = Counter()
    for order in _actor_orders(action):
        if len(order) >= 2 and order[0] == "PLACE":
            deposits[str(order[1])] += _quantity(order)
    return deposits


def _pickup_reserves(action: dict) -> Counter:
    reserves = Counter()
    for order in _actor_orders(action):
        if len(order) >= 2 and order[0] == "PICKUP":
            reserves[str(order[1])] += _quantity(order)
    return reserves


def _weed_only_actors(replay: dict, seat: int, actions: list[dict]) -> dict:
    annotations = {}
    for step, action in enumerate(actions):
        observation = replay["steps"][step][seat]["observation"]
        farms = observation.get("farms", [])
        if seat >= len(farms):
            continue
        farm = farms[seat]
        positions = _actor_positions(farm)
        actors = []
        for index, order in enumerate(_actor_orders(action)):
            if not order or order[0] != "DIG" or index >= len(positions):
                continue
            tile = _tile_at(farm, positions[index])
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                actors.append("farmer" if index == 0 else f"hand:{index - 1}")
        if actors:
            annotations[str(step)] = actors
    return annotations


def _comparison_actions(
    actions: list[dict], weed_only: dict[str, list[str]]
) -> list[list[list]]:
    comparison = []
    for step, action in enumerate(actions):
        annotated = set(weed_only.get(str(step), []))
        row = []
        for index, order in enumerate(_actor_orders(action)):
            actor = "farmer" if index == 0 else f"hand:{index - 1}"
            row.append(["WEED_ONLY"] if actor in annotated else list(order))
        comparison.append(row)
    return comparison


def normalize_market_order(order: list) -> list:
    """Keep market intent exact, except sale quantities are immaterial."""
    normalized = list(order)
    if normalized and normalized[0] == "SELL":
        return normalized[:2]
    return normalized


def comparison_timeline(record: dict) -> list[dict]:
    """Pair compared field work with normalized market orders by step."""
    rows = []
    for field, action in zip(record["comparison_actions"], record["actions"]):
        rows.append({
            "field": field,
            "market": [
                normalize_market_order(order)
                for order in action.get("market", [])
            ],
        })
    return rows


def cumulative_purchases(record: dict, stop: int) -> dict[str, int]:
    """Summarize route-defining purchase attempts before ``stop``."""
    purchases = Counter()
    for action in record["actions"][:stop]:
        for order in action.get("market", []):
            operation = str(order[0])
            item = str(order[1]) if len(order) > 1 else ""
            quantity = _quantity(order)
            if operation == "HIRE":
                purchases["HIRE"] += 1
            elif operation == "BUY_LAND":
                purchases["LAND"] += 1
            elif operation == "BUY_ANIMAL":
                purchases[item] += quantity
            elif operation == "BUY_SEED":
                purchases[f"{item}_SEED"] += quantity
            elif operation == "BUY_PRODUCT":
                purchases[f"{item}_PRODUCT"] += quantity
    return dict(sorted(purchases.items()))


def opening_fingerprint(record: dict, stop: int = 72) -> str:
    """Hash a bounded canonical form of a route's opening timeline."""
    payload = json.dumps(
        record["comparison_timeline"][:stop],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_tile(tile):
    if tile is None or tile == "LOCKED":
        return tile
    if not isinstance(tile, dict):
        return str(tile)
    return {
        key: tile.get(key)
        for key in ("kind", "crop", "animal")
        if tile.get(key) is not None
    }


def canonical_farm_state(replay: dict, seat: int, step: int) -> dict:
    """Return only route-defining farm state at a decision observation."""
    observation = replay["steps"][step][seat]["observation"]
    farm = observation["farms"][seat]
    return {
        "farmer": list(farm.get("farmer", [])),
        "hands": [list(position) for position in farm.get("hands", [])],
        "unlocked_quadrants": list(farm.get("unlocked_quadrants", [])),
        "tiles": [
            [_canonical_tile(tile) for tile in row]
            for row in farm.get("tiles", [])
        ],
    }


def _first_difference(left: list, right: list, stop: int) -> int | None:
    for step in range(stop):
        left_value = left[step] if step < len(left) else None
        right_value = right[step] if step < len(right) else None
        if left_value != right_value:
            return step
    return None


def build_handoff_report(
    base: dict, branch: dict, decision_step: int
) -> dict:
    """Explain whether a full-route switch is safe at ``decision_step``."""
    base_field = [_actor_orders(action) for action in base["actions"]]
    branch_field = [_actor_orders(action) for action in branch["actions"]]
    first_field_difference = _first_difference(
        base_field, branch_field, decision_step
    )
    base_market = [
        [normalize_market_order(order) for order in action.get("market", [])]
        for action in base["actions"]
    ]
    branch_market = [
        [normalize_market_order(order) for order in action.get("market", [])]
        for action in branch["actions"]
    ]
    first_market_difference = _first_difference(
        base_market, branch_market, decision_step
    )
    field_prefix_equal = first_field_difference is None
    purchase_prefix_equal = (
        cumulative_purchases(base, decision_step)
        == cumulative_purchases(branch, decision_step)
    )
    state_key = str(decision_step)
    farm_state_equal = (
        state_key in base.get("canonical_states", {})
        and state_key in branch.get("canonical_states", {})
        and base["canonical_states"][state_key]
        == branch["canonical_states"][state_key]
    )
    return {
        "decision_step": decision_step,
        "safe": all((
            field_prefix_equal,
            purchase_prefix_equal,
            farm_state_equal,
        )),
        "field_prefix_equal": field_prefix_equal,
        "purchase_prefix_equal": purchase_prefix_equal,
        "farm_state_equal": farm_state_equal,
        "first_field_difference": first_field_difference,
        "first_market_difference": first_market_difference,
    }


def _valid_field_counts(
    replay: dict, seat: int, actions: list[dict]
) -> dict[str, int]:
    valid = Counter()
    for step, action in enumerate(actions):
        observation = replay["steps"][step][seat]["observation"]
        farms = observation.get("farms", [])
        if seat >= len(farms):
            continue
        farm = farms[seat]
        positions = _actor_positions(farm)
        inventories = observation.get("private", {}).get("inventories", [])
        for index, order in enumerate(_actor_orders(action)):
            if not order or index >= len(positions):
                continue
            operation = order[0]
            tile = _tile_at(farm, positions[index])
            inventory = inventories[index] if index < len(inventories) else {}
            animal = isinstance(tile, dict) and tile.get("animal")
            if (
                operation == "FEED"
                and animal
                and not tile.get("fed_today")
                and int(inventory.get("WHEAT", 0) or 0) > 0
            ):
                valid[operation] += 1
            elif operation == "CARE" and animal and not tile.get("cared_today"):
                valid[operation] += 1
            elif (
                operation == "COLLECT_FERTILIZER"
                and animal
                and int(tile.get("fertilizer_available", 0) or 0) > 0
            ):
                valid[operation] += 1
            elif (
                operation == "HARVEST"
                and isinstance(tile, dict)
                and int(tile.get("yield_units", 0) or 0) > 0
            ):
                valid[operation] += 1
    return dict(sorted(valid.items()))


def _farm_checkpoint(replay: dict, seat: int, step: int) -> dict:
    observation = replay["steps"][step][seat]["observation"]
    farm = observation["farms"][seat]
    tiles = [tile for row in farm.get("tiles", []) for tile in row]
    crops = Counter(
        tile.get("crop")
        for tile in tiles
        if isinstance(tile, dict) and tile.get("kind") == "PLANT"
    )
    animals = Counter(
        tile.get("animal")
        for tile in tiles
        if isinstance(tile, dict) and tile.get("animal")
    )
    return {
        "step": step,
        "day": int(observation.get("day", step // 24) or 0),
        "money": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands", [])),
        "unlocked_quadrants": len(farm.get("unlocked_quadrants", [])),
        "crops": dict(sorted(crops.items())),
        "animals": dict(sorted(animals.items())),
        "shed_total": sum(
            int(value or 0)
            for value in observation.get("private", {}).get("shed", {}).values()
        ),
    }


def inspect_replay(
    replay: dict,
    seat: int,
    source_name: str,
) -> dict:
    """Extract route and market evidence from one validated replay."""
    actions = shifted_actions(replay, seat)
    purchases = Counter()
    sales = Counter()
    market_orders = Counter()
    field_actions = Counter()
    daily_hires = [0 for _ in range(30)]

    for step, action in enumerate(actions):
        for order in _actor_orders(action):
            field_actions[str(order[0])] += 1
        for order in action.get("market", []):
            operation = str(order[0])
            item = str(order[1]) if len(order) > 1 else ""
            quantity = _quantity(order)
            market_orders[f"{operation}:{item}"] += 1
            if operation == "HIRE":
                purchases["HIRE"] += 1
                daily_hires[min(step // 24, 29)] += 1
            elif operation == "BUY_LAND":
                purchases["LAND"] += 1
            elif operation == "BUY_ANIMAL":
                purchases[item] += quantity
            elif operation == "BUY_SEED":
                purchases[f"{item}_SEED"] += quantity
            elif operation == "BUY_PRODUCT":
                purchases[f"{item}_PRODUCT"] += quantity
            elif operation == "SELL":
                sales[item] += quantity

    weed_only = _weed_only_actors(replay, seat, actions)
    names = replay["info"]["TeamNames"]
    opponent_seat = 1 - seat
    rewards = replay["rewards"]
    checkpoints = [
        _farm_checkpoint(replay, seat, step)
        for step in (0, 24, 72, 144, 216, 288, 360, 480, 600, 719)
    ]
    record = {
        "source": source_name,
        "seed": int(replay["info"].get("seed", 0) or 0),
        "seat": seat,
        "opponent": names[opponent_seat],
        "self_play": names[0] == names[1],
        "reward": float(rewards[seat]),
        "opponent_reward": float(rewards[opponent_seat]),
        "margin": float(rewards[seat]) - float(rewards[opponent_seat]),
        "purchases": dict(sorted(purchases.items())),
        "sales_requested": dict(sorted(sales.items())),
        "market_order_counts": dict(sorted(market_orders.items())),
        "field_actions": dict(sorted(field_actions.items())),
        "valid_field_actions": _valid_field_counts(replay, seat, actions),
        "daily_hires": daily_hires,
        "checkpoints": checkpoints,
        "shop_checkpoints": {
            str(step): replay["steps"][step][seat]["observation"]
            .get("town", {})
            .get("unlocked_shops", [])
            for step in (72, 144, 216, 288)
        },
        "weed_only_count": sum(len(actors) for actors in weed_only.values()),
        "actions": actions,
        "weed_only": weed_only,
        "comparison_actions": _comparison_actions(actions, weed_only),
        "canonical_states": {
            str(step): canonical_farm_state(replay, seat, step)
            for step in (144, 216)
        },
    }
    record["comparison_timeline"] = comparison_timeline(record)
    record["branch"] = branch_key(record)
    return record


def collect_team_records(
    replays, team_name: str, self_seat: int | None = None
) -> list[dict]:
    """Inspect compatible replays that contain the requested team."""
    records = []
    for path, replay in replays:
        names = replay["info"]["TeamNames"]
        if team_name not in names:
            continue
        seat = find_seat(replay, team_name, self_seat)
        records.append(inspect_replay(replay, seat, path.name))
    return records


def branch_key(evidence: dict) -> str:
    """Name a route branch by its defining animal and premium seed buys."""
    purchases = evidence.get("purchases", {})
    return (
        f"c{int(purchases.get('COW', 0))}"
        f"-s{int(purchases.get('SHEEP', 0))}"
        f"-straw{int(purchases.get('STRAWBERRY_SEED', 0))}"
        f"-melon{int(purchases.get('MELON_SEED', 0))}"
    )


def actor_disagreement(left: dict, right: dict) -> int:
    """Count changed or missing actor orders across two complete routes."""
    left_steps = left["comparison_actions"]
    right_steps = right["comparison_actions"]
    disagreement = 0
    for step in range(max(len(left_steps), len(right_steps))):
        left_row = left_steps[step] if step < len(left_steps) else []
        right_row = right_steps[step] if step < len(right_steps) else []
        for actor in range(max(len(left_row), len(right_row))):
            left_order = left_row[actor] if actor < len(left_row) else None
            right_order = right_row[actor] if actor < len(right_row) else None
            disagreement += left_order != right_order
    return disagreement


def route_distance(left: dict, right: dict) -> int:
    """Count field and per-step normalized-market route disagreement."""
    distance = actor_disagreement(left, right)
    left_timeline = left["comparison_timeline"]
    right_timeline = right["comparison_timeline"]
    for step in range(max(len(left_timeline), len(right_timeline))):
        left_market = (
            left_timeline[step]["market"]
            if step < len(left_timeline)
            else None
        )
        right_market = (
            right_timeline[step]["market"]
            if step < len(right_timeline)
            else None
        )
        distance += left_market != right_market
    return distance


def select_opening_family(records: list[dict]) -> tuple[str, list[dict]]:
    """Choose the largest 42/12 opening represented by every branch."""
    groups = defaultdict(list)
    for record in records:
        if record.get("branch") in SUPPORTED_BRANCHES:
            groups[opening_fingerprint(record)].append(record)
    candidates = [
        (fingerprint, family)
        for fingerprint, family in groups.items()
        if {record["branch"] for record in family} >= set(SUPPORTED_BRANCHES)
    ]
    if not candidates:
        raise ReplayError("no opening family contains all supported branches")
    return min(candidates, key=lambda item: (-len(item[1]), item[0]))


def select_branch_medoids(records: list[dict]) -> dict[str, dict]:
    """Select a deterministic route-distance medoid for each branch."""
    groups = defaultdict(list)
    for record in records:
        groups[record["branch"]].append(record)
    medoids = {}
    for branch, branch_records in sorted(groups.items()):
        ranked = []
        for record in branch_records:
            distance = sum(
                route_distance(record, other)
                for other in branch_records
                if other is not record
            )
            ranked.append((distance, record["source"], record))
        medoids[branch] = min(ranked, key=lambda item: (item[0], item[1]))[2]
    return medoids


def select_medoid(records: list[dict]) -> dict:
    """Select the coherent route minimizing disagreement with its peers."""
    if not records:
        raise ReplayError("cannot select a medoid from no records")
    ranked = []
    for record in records:
        distance = sum(
            actor_disagreement(record, other)
            for other in records
            if other is not record
        )
        ranked.append((distance, record["source"], record))
    return min(ranked, key=lambda item: (item[0], item[1]))[2]


def _public_record(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in {
            "actions",
            "canonical_states",
            "comparison_actions",
            "comparison_timeline",
            "weed_only",
        }
    }


def build_required_handoff_reports(medoids: dict[str, dict]) -> dict:
    """Build required branch reports, rejecting any unsafe route splice."""
    missing = sorted(set(SUPPORTED_BRANCHES) - set(medoids))
    if missing:
        raise ReplayError(
            "missing required branch medoids: " + ", ".join(missing)
        )
    base = medoids[DEFAULT_BRANCH]
    reports = {
        branch: build_handoff_report(base, medoids[branch], decision_step)
        for branch, decision_step in sorted(HANDOFF_STEPS.items())
    }
    unsafe = [
        branch for branch, report in reports.items() if not report["safe"]
    ]
    if unsafe:
        raise ReplayError(
            "unsafe required handoff: " + ", ".join(sorted(unsafe))
        )
    return reports


def build_outputs(
    paths: list[Path],
    team_name: str,
    self_seat: int | None = None,
    required_module_version: str = REQUIRED_MODULE_VERSION,
) -> tuple[dict, dict]:
    """Build deterministic analysis and canonical-route payloads."""
    accepted, rejected = load_compatible_replays(
        paths, required_module_version=required_module_version
    )
    records = collect_team_records(accepted, team_name, self_seat)
    if not records:
        raise ReplayError("no compatible target-team replays supplied")

    fingerprint, family = select_opening_family(records)
    medoids = select_branch_medoids(family)
    handoff_reports = build_required_handoff_reports(medoids)
    branch_counts = dict(sorted(
        Counter(record["branch"] for record in family).items()
    ))
    analysis = {
        "schema_version": 2,
        "required_module_version": required_module_version,
        "team_name": team_name,
        "input_replay_count": len(accepted) + len(rejected),
        "accepted_sources": sorted(path.name for path, _ in accepted),
        "rejected_records": rejected,
        "target_record_count": len(records),
        "opening_fingerprint": fingerprint,
        "family_member_count": len(family),
        "family_sources": sorted(record["source"] for record in family),
        "branch_counts": branch_counts,
        "medoid_sources": {
            branch: medoid["source"]
            for branch, medoid in sorted(medoids.items())
        },
        "records": [_public_record(record) for record in records],
    }
    route = {
        "schema_version": 2,
        "required_module_version": required_module_version,
        "team_name": team_name,
        "opening_fingerprint": fingerprint,
        "selector": {
            "early_step": 144,
            "freeze_step": 216,
            "default_branch": DEFAULT_BRANCH,
        },
        "handoffs": handoff_reports,
        "branches": {
            branch: {
                "source": medoid["source"],
                "actions": medoid["actions"],
                "weed_only": medoid["weed_only"],
            }
            for branch, medoid in sorted(medoids.items())
        },
    }
    return analysis, route


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Inspect replay paths and write deterministic analysis and route JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--self-seat", type=int, choices=(0, 1))
    parser.add_argument("--analysis-output", required=True, type=Path)
    parser.add_argument("--route-output", required=True, type=Path)
    args = parser.parse_args(argv)

    analysis, route = build_outputs(
        args.paths,
        team_name=args.team_name,
        self_seat=args.self_seat,
    )
    _write_json(args.analysis_output, analysis)
    _write_json(args.route_output, route)
    print(
        f"Accepted {len(analysis['accepted_sources'])} compatible replays; "
        f"selected {analysis['family_member_count']} target routes in "
        f"opening family {analysis['opening_fingerprint'][:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
