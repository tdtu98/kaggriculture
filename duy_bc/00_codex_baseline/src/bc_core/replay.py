"""Strict corpus boundary for Ryo replay files and shifted decisions."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from bc_core.constants import (
    ARGUMENT_ITEMS,
    EXPECTED_MODULE_VERSION,
    EXPECTED_REPLAY_CONFIGURATION,
    OPERATION_TO_ID,
)


class ReplayError(ValueError):
    """Raised when corpus metadata or a replay violates the v0 contract."""


@dataclass(frozen=True)
class SourceReplay:
    split: str
    episode_id: str
    path: Path
    sha256: str
    source_date: str
    route_family: str
    audit_source_path: str = ""


@dataclass(frozen=True)
class ReplaySnapshot:
    source: SourceReplay
    content: bytes


@dataclass(frozen=True)
class Decision:
    source: SourceReplay
    step: int
    seat: int
    observation: dict[str, Any]
    action: dict[str, Any]


_MANIFEST_COLUMNS = {
    "episode_id", "split", "source_date", "source_path", "source_sha256", "route_family",
}
_SPLIT_COUNTS = {"train": 70, "val": 15, "test": 15}
_STRATIFY_FIELDS = [
    "source_date", "opponent", "ryo_seat", "margin_quartile", "shop_profile", "route_family",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of file bytes without loading a replay into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_manifest(corpus_root: Path) -> tuple[SourceReplay, ...]:
    """Read the fixed 70/15/15 split manifest and verify its local replay bytes."""
    sources, _ = _load_split_manifest(corpus_root, capture_snapshots=False)
    return sources


def load_split_manifest_snapshots(corpus_root: Path) -> tuple[ReplaySnapshot, ...]:
    """Verify the fixed manifest while retaining each exact hashed replay payload."""
    _, snapshots = _load_split_manifest(corpus_root, capture_snapshots=True)
    return snapshots


def _load_split_manifest(
    corpus_root: Path, *, capture_snapshots: bool
) -> tuple[tuple[SourceReplay, ...], tuple[ReplaySnapshot, ...]]:
    _validate_split_summary(corpus_root / "split_summary.json")
    manifest_path = corpus_root / "manifest.csv"
    try:
        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            reader = csv.DictReader(manifest_file)
            if reader.fieldnames is None or not _MANIFEST_COLUMNS.issubset(reader.fieldnames):
                raise ReplayError(f"manifest missing required columns path={manifest_path}")
            rows = list(reader)
    except OSError as error:
        raise ReplayError(f"cannot read manifest path={manifest_path}") from error

    seen_episode: dict[str, str] = {}
    seen_hash: dict[str, str] = {}
    sources: list[SourceReplay] = []
    snapshots: list[ReplaySnapshot] = []
    split_counts = {split: 0 for split in _SPLIT_COUNTS}
    for row_number, row in enumerate(rows, start=1):
        missing = [column for column in _MANIFEST_COLUMNS if not row.get(column)]
        if missing:
            raise ReplayError(f"manifest row {row_number} missing {', '.join(sorted(missing))}")
        split = row["split"]
        if split not in _SPLIT_COUNTS:
            raise ReplayError(f"invalid split row {row_number}: {split!r}")
        episode_id = row["episode_id"]
        source_hash = row["source_sha256"]
        row_ref = f"row {row_number} split={split} episode={episode_id}"
        if episode_id in seen_episode:
            raise ReplayError(f"duplicate episode {episode_id}: {seen_episode[episode_id]} conflicts with {row_ref}")
        if source_hash in seen_hash:
            raise ReplayError(f"duplicate source hash {source_hash}: {seen_hash[source_hash]} conflicts with {row_ref}")
        seen_episode[episode_id] = row_ref
        seen_hash[source_hash] = row_ref
        split_counts[split] += 1
        linked_path = corpus_root / split / f"{episode_id}.json"
        try:
            replay_path = linked_path.resolve(strict=True)
        except OSError as error:
            raise ReplayError(f"missing replay for {row_ref}: {linked_path}") from error
        try:
            content = replay_path.read_bytes()
        except OSError as error:
            raise ReplayError(f"cannot load replay for {row_ref}: {replay_path}") from error
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != source_hash:
            raise ReplayError(f"hash mismatch {row_ref}: manifest={source_hash} actual={actual_hash}")
        source = SourceReplay(
            split, episode_id, replay_path, source_hash, row["source_date"], row["route_family"], row["source_path"],
        )
        sources.append(source)
        if capture_snapshots:
            snapshots.append(ReplaySnapshot(source, content))
    if split_counts != _SPLIT_COUNTS:
        raise ReplayError(f"manifest split totals must be {_SPLIT_COUNTS}, got {split_counts}")
    return tuple(sources), tuple(snapshots)


def _validate_split_summary(path: Path) -> None:
    try:
        with path.open(encoding="utf-8") as summary_file:
            summary: Any = json.load(summary_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read split summary path={path}") from error
    if not isinstance(summary, dict):
        raise ReplayError("split summary must be a JSON object")
    expected = {
        "schema_version": 1,
        "selected_win_count": 100,
        "unique_episode_ids": 100,
        "unique_source_hashes": 100,
        "split_counts": _SPLIT_COUNTS,
        "stratify_fields": _STRATIFY_FIELDS,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ReplayError(f"invalid split summary {field}: expected {value!r}, got {summary.get(field)!r}")


def load_validated_replay(source: SourceReplay, expected_module_version: str) -> dict[str, Any]:
    """Load a replay file after validating the immutable replay-level contract."""
    try:
        content = source.path.read_bytes()
    except OSError as error:
        raise ReplayError(f"cannot load replay split={source.split} episode={source.episode_id}") from error
    return load_validated_replay_bytes(source, content, expected_module_version)


def read_replay_snapshot(source: SourceReplay) -> ReplaySnapshot:
    """Read and hash one replay exactly once for later immutable decoding."""
    try:
        content = source.path.read_bytes()
    except OSError as error:
        raise ReplayError(
            f"cannot load replay split={source.split} episode={source.episode_id}"
        ) from error
    actual = hashlib.sha256(content).hexdigest()
    if actual != source.sha256:
        raise ReplayError(
            "hash mismatch "
            f"split={source.split} episode={source.episode_id}: "
            f"manifest={source.sha256} actual={actual}"
        )
    return ReplaySnapshot(source, content)


def load_validated_replay_bytes(
    source: SourceReplay, content: bytes, expected_module_version: str
) -> dict[str, Any]:
    """Decode and validate replay JSON from already authenticated immutable bytes."""
    if not isinstance(content, bytes):
        raise ReplayError(
            f"replay bytes must be bytes split={source.split} episode={source.episode_id}"
        )
    try:
        replay: Any = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayError(
            f"cannot load replay split={source.split} episode={source.episode_id}"
        ) from error
    if not isinstance(replay, dict):
        raise ReplayError(f"replay must be an object split={source.split} episode={source.episode_id}")
    _validate_replay(source, replay, expected_module_version)
    return replay


def iter_decisions(source: SourceReplay, replay: dict[str, Any]) -> Iterator[Decision]:
    """Yield observation[t] paired with Ryo's action[t + 1], excluding terminal state."""
    ryo_seat = _validate_replay(source, replay, EXPECTED_MODULE_VERSION)
    steps = replay["steps"]
    for step in range(719):
        record = steps[step][ryo_seat]
        next_record = steps[step + 1][ryo_seat]
        try:
            observation = record["observation"]
            action = next_record["action"]
        except KeyError as error:
            raise ReplayError(
                f"missing {error.args[0]} split={source.split} episode={source.episode_id} "
                f"step={step} seat={ryo_seat}"
            ) from error
        if not isinstance(observation, dict) or not isinstance(action, dict):
            raise ReplayError(f"invalid observation/action split={source.split} episode={source.episode_id} step={step} seat={ryo_seat}")
        if observation.get("player") != ryo_seat:
            raise ReplayError(
                f"wrong delivered seat split={source.split} episode={source.episode_id} "
                f"step={step} seat={ryo_seat} observation_player={observation.get('player')}"
            )
        try:
            hand_count = len(observation["farms"][ryo_seat]["hands"])
        except (KeyError, IndexError, TypeError) as error:
            raise ReplayError(
                f"missing farm hands split={source.split} episode={source.episode_id} step={step} seat={ryo_seat}"
            ) from error
        if set(action) != {"farmer", "hands", "market"} or not isinstance(action.get("hands"), list) or len(action["hands"]) != hand_count:
            actions = 1 + len(action.get("hands", [])) if isinstance(action.get("hands", []), list) else 1
            raise ReplayError(
                f"hand/action mismatch split={source.split} episode={source.episode_id} "
                f"step={step} seat={ryo_seat} actors={hand_count + 1} actions={actions}"
            )
        yield Decision(source, step, ryo_seat, observation, action)


def _validate_replay(source: SourceReplay, replay: dict[str, Any], expected_module_version: str | None) -> int:
    if expected_module_version is not None and replay.get("module_version") != expected_module_version:
        raise ReplayError(f"wrong module_version split={source.split} episode={source.episode_id}")
    if replay.get("configuration") != EXPECTED_REPLAY_CONFIGURATION:
        raise ReplayError(f"wrong configuration split={source.split} episode={source.episode_id}")
    steps = replay.get("steps")
    if not isinstance(steps, list) or len(steps) != 720:
        raise ReplayError(f"replay must contain 720 steps split={source.split} episode={source.episode_id}")
    for step, agents in enumerate(steps):
        if not isinstance(agents, list) or len(agents) != 2 or not all(isinstance(agent, dict) for agent in agents):
            raise ReplayError(f"expected two agent records split={source.split} episode={source.episode_id} step={step}")
    try:
        info = replay["info"]
        team_names = info["TeamNames"]
        agent_names = [agent["Name"] for agent in info["Agents"]]
    except (KeyError, TypeError) as error:
        raise ReplayError(f"missing replay identity split={source.split} episode={source.episode_id}") from error
    if not isinstance(team_names, list) or len(team_names) != 2 or team_names != agent_names:
        raise ReplayError(f"TeamNames and Agents disagree split={source.split} episode={source.episode_id}")
    if team_names.count("Ryo Hasegawa") != 1:
        raise ReplayError(f"expected exactly one Ryo Hasegawa split={source.split} episode={source.episode_id}")
    if str(info.get("EpisodeId")) != source.episode_id:
        raise ReplayError(f"episode id mismatch split={source.split} episode={source.episode_id}")
    ryo_seat = team_names.index("Ryo Hasegawa")
    _validate_terminal(source, replay, ryo_seat)
    return ryo_seat


def _validate_terminal(source: SourceReplay, replay: dict[str, Any], ryo_seat: int) -> None:
    rewards = replay.get("rewards")
    statuses = replay.get("statuses")
    if not isinstance(rewards, list) or len(rewards) != 2 or rewards[ryo_seat] <= rewards[1 - ryo_seat]:
        raise ReplayError(f"Ryo must have winning rewards split={source.split} episode={source.episode_id}")
    if not isinstance(statuses, list) or statuses != ["DONE", "DONE"]:
        raise ReplayError(f"terminal statuses must be DONE split={source.split} episode={source.episode_id}")
    terminal = replay["steps"][-1]
    for seat, agent in enumerate(terminal):
        if agent.get("status") != statuses[seat] or agent.get("reward") != rewards[seat]:
            raise ReplayError(f"terminal outcome mismatch split={source.split} episode={source.episode_id} seat={seat}")


def operation_and_arguments(unit_action: Any) -> tuple[int, int, int]:
    """Map one farmer/hand action to fixed operation, item, and quantity IDs."""
    if isinstance(unit_action, str):
        tokens = [unit_action]
    elif isinstance(unit_action, list):
        tokens = unit_action
    else:
        raise ReplayError(f"invalid unit action {unit_action!r}")
    if not 1 <= len(tokens) <= 3 or not isinstance(tokens[0], str):
        raise ReplayError(f"invalid unit action {unit_action!r}")
    operation = tokens[0]
    if operation not in OPERATION_TO_ID:
        raise ReplayError(f"unknown operation {operation}")
    item_id = -1
    quantity = -1
    if len(tokens) >= 2:
        item = tokens[1]
        if not isinstance(item, str) or item not in ARGUMENT_ITEMS:
            raise ReplayError(f"unknown action item {item!r} for {operation}")
        item_id = ARGUMENT_ITEMS.index(item)
    if len(tokens) == 3:
        if isinstance(tokens[2], bool) or not isinstance(tokens[2], int):
            raise ReplayError(f"invalid action quantity {tokens[2]!r} for {operation}")
        quantity = tokens[2]
    return OPERATION_TO_ID[operation], item_id, quantity
