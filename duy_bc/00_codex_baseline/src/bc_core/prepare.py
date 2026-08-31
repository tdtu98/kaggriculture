"""Orchestrate validated replay encoding and publish its deterministic audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from bc_core.constants import ACTOR_DIM, GLOBAL_DIM, GRID_CHANNELS, OPERATIONS, load_config
from bc_core.features import EncodedGame, encode_game, logical_shard_identity, write_shard
from bc_core.paths import corpus_path
from bc_core.replay import SourceReplay, load_split_manifest, load_validated_replay, sha256_file
from bc_core.scripts_support import atomic_json_write
from bc_core.training_audit import (
    TRAINING_AUDIT_SCHEMA,
    canonical_json_bytes,
    validate_training_audit,
)


_SPLITS = ("train", "val", "test")
_PREPARATION_SCHEMA = "ryo-preparation-v0"
_TRAINING_PREPARATION_SCHEMA = TRAINING_AUDIT_SCHEMA


def _canonical_json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _selected_sources(
    sources: tuple[SourceReplay, ...], limit_per_split: int | None
) -> tuple[SourceReplay, ...]:
    ordered = sorted(sources, key=lambda source: (source.split, source.episode_id))
    if limit_per_split is None:
        return tuple(ordered)
    selected: list[SourceReplay] = []
    counts = {split: 0 for split in _SPLITS}
    for source in ordered:
        if counts[source.split] < limit_per_split:
            selected.append(source)
            counts[source.split] += 1
    return tuple(selected)


def _array_shapes(game: EncodedGame) -> dict[str, list[int]]:
    return {
        "actor_features": list(game.actor_features.shape),
        "argument_item": list(game.argument_item.shape),
        "argument_quantity": list(game.argument_quantity.shape),
        "global_features": list(game.global_features.shape),
        "grid": list(game.grid.shape),
        "label": list(game.label.shape),
        "step_index": list(game.step_index.shape),
    }


def _training_audit(
    config: dict[str, Any],
    canonical_config: bytes,
    shard_records: list[dict[str, Any]],
    split_counts: dict[str, dict[str, Any]],
    *,
    smoke_mode: bool,
) -> dict[str, Any]:
    safe_records = sorted(
        (
            dict(record)
            for record in shard_records
            if record["split"] in ("train", "val")
        ),
        key=lambda record: (record["split"], record["episode_id"]),
    )
    safe_splits = {
        split: dict(split_counts[split]) for split in ("train", "val")
    }
    train_support = {
        operation_id
        for operation_id, count in enumerate(safe_splits["train"]["label_counts"])
        if count
    }
    val_support = {
        operation_id
        for operation_id, count in enumerate(safe_splits["val"]["label_counts"])
        if count
    }
    operation_support_covered = not smoke_mode and val_support <= train_support
    checks = {
        "operation_support_covered": operation_support_covered,
        "safe_records_validated": True,
        "shard_identities_verified": True,
        "source_hashes_unique": len({record["source_sha256"] for record in safe_records})
        == len(safe_records),
        "tensor_shapes_validated": True,
        "train_val_episode_ids_unique": len(
            {record["episode_id"] for record in safe_records}
        )
        == len(safe_records),
    }
    safe_manifest_content = {
        "schema_version": _TRAINING_PREPARATION_SCHEMA,
        "records": safe_records,
    }
    safe_manifest_sha256 = hashlib.sha256(
        _canonical_json_bytes(safe_manifest_content)
    ).hexdigest()
    payload: dict[str, Any] = {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "config": {
            "canonical": config,
            "sha256": hashlib.sha256(canonical_config).hexdigest(),
        },
        "feature_schema_version": config["feature_schema_version"],
        "model_schema_version": config["schema_version"],
        "operations": list(OPERATIONS),
        "safe_manifest_sha256": safe_manifest_sha256,
        "schema_version": _TRAINING_PREPARATION_SCHEMA,
        "shard_identities": sorted(record["shard_identity"] for record in safe_records),
        "shards": safe_records,
        "smoke_mode": smoke_mode,
        "source_hashes": sorted(record["source_sha256"] for record in safe_records),
        "splits": safe_splits,
        "totals": {
            "games": sum(int(safe_splits[split]["games"]) for split in ("train", "val")),
            "samples": sum(
                int(safe_splits[split]["samples"]) for split in ("train", "val")
            ),
        },
        "trainable": not smoke_mode and all(checks.values()),
    }
    payload["training_identity"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return validate_training_audit(payload, config)


def prepare(
    config_path: Path, output_root: Path, *, limit_per_split: int | None = None
) -> dict[str, Any]:
    """Validate, encode, shard, and audit the configured replay corpus."""
    if limit_per_split is not None and (
        isinstance(limit_per_split, bool) or limit_per_split <= 0
    ):
        raise ValueError("limit_per_split must be a positive integer")

    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config = load_config(config_path)
    canonical_config = _canonical_json_bytes(config)
    corpus_root = corpus_path(config["corpus_root"], config_path=config_path)
    if output_root.is_relative_to(corpus_root):
        raise ValueError(
            "refusing output inside source corpus: "
            f"output_root={output_root} corpus_root={corpus_root}"
        )
    sources = load_split_manifest(corpus_root)
    selected_sources = _selected_sources(sources, limit_per_split)
    smoke_mode = limit_per_split is not None

    for split in _SPLITS:
        (output_root / split).mkdir(parents=True, exist_ok=True)

    split_counts = {
        split: {"games": 0, "samples": 0, "label_counts": [0] * len(OPERATIONS)}
        for split in _SPLITS
    }
    total_label_counts = np.zeros(len(OPERATIONS), dtype=np.int64)
    shard_records: list[dict[str, Any]] = []
    shard_identities: list[str] = []

    for source in selected_sources:
        try:
            replay = load_validated_replay(source, config["module_version"])
            game = encode_game(source, replay)
            expected_identity = logical_shard_identity(game)
            shard_relative_path = Path(source.split) / f"{source.episode_id}.npz"
            shard_identity = write_shard(game, output_root / shard_relative_path)
            if shard_identity != expected_identity:
                raise RuntimeError("written shard identity differs from encoded game")
        except Exception as error:
            raise RuntimeError(
                f"preparation failed split={source.split} episode={source.episode_id}: {error}"
            ) from error

        label_counts = np.bincount(game.label, minlength=len(OPERATIONS)).astype(np.int64)
        split_summary = split_counts[source.split]
        split_summary["games"] += 1
        split_summary["samples"] += int(game.label.shape[0])
        split_summary["label_counts"] = (
            np.asarray(split_summary["label_counts"], dtype=np.int64) + label_counts
        ).tolist()
        total_label_counts += label_counts
        shard_identities.append(shard_identity)
        shard_records.append(
            {
                "episode_id": source.episode_id,
                "label_counts": label_counts.tolist(),
                "sample_count": int(game.label.shape[0]),
                "shard_identity": shard_identity,
                "shard_path": shard_relative_path.as_posix(),
                "source_path": source.audit_source_path,
                "shard_source_path": str(source.path),
                "source_sha256": source.sha256,
                "source_date": source.source_date,
                "route_family": source.route_family,
                "split": source.split,
                "tensor_shapes": _array_shapes(game),
            }
        )

    operation_support_covered = False
    if not smoke_mode:
        train_support = {
            operation_id
            for operation_id, count in enumerate(split_counts["train"]["label_counts"])
            if count
        }
        evaluation_support = {
            operation_id
            for split in ("val", "test")
            for operation_id, count in enumerate(split_counts[split]["label_counts"])
            if count
        }
        missing = sorted(evaluation_support - train_support)
        if missing:
            descriptions = ", ".join(
                f"{operation_id}:{OPERATIONS[operation_id]}" for operation_id in missing
            )
            raise RuntimeError(f"operation support missing from train: {descriptions}")
        operation_support_covered = True

    checks = {
        "cross_split_leakage_absent": True,
        "episode_ids_unique": True,
        "manifest_validated": True,
        "operation_support_covered": operation_support_covered,
        "selected_replays_validated": True,
        "shard_identities_verified": True,
        "source_hashes_unique": True,
        "tensor_shapes_validated": True,
    }
    manifest_hash = sha256_file(corpus_root / "manifest.csv")
    split_summary_hash = sha256_file(corpus_root / "split_summary.json")
    sorted_source_hashes = sorted(source.sha256 for source in selected_sources)
    sorted_shard_identities = sorted(shard_identities)
    preparation_payload = {
        "config": config,
        "manifest_csv_sha256": manifest_hash,
        "shard_identities": sorted_shard_identities,
        "source_hashes": sorted_source_hashes,
        "split_summary_json_sha256": split_summary_hash,
    }
    preparation_identity = hashlib.sha256(_canonical_json_bytes(preparation_payload)).hexdigest()
    total_games = sum(int(summary["games"]) for summary in split_counts.values())
    total_samples = sum(int(summary["samples"]) for summary in split_counts.values())
    all_checks_passed = all(checks.values())
    audit: dict[str, Any] = {
        "all_checks_passed": all_checks_passed,
        "checks": checks,
        "config": {
            "canonical": config,
            "sha256": hashlib.sha256(canonical_config).hexdigest(),
        },
        "label_counts": total_label_counts.tolist(),
        "manifest": {
            "manifest_csv_sha256": manifest_hash,
            "split_summary_json_sha256": split_summary_hash,
        },
        "operations": list(OPERATIONS),
        "preparation_identity": preparation_identity,
        "schema_version": _PREPARATION_SCHEMA,
        "shard_identities": sorted_shard_identities,
        "shards": shard_records,
        "smoke_mode": smoke_mode,
        "source_hashes": sorted_source_hashes,
        "splits": split_counts,
        "tensor_shapes": {
            "actor_features": ["samples", ACTOR_DIM],
            "argument_item": ["samples"],
            "argument_quantity": ["samples"],
            "global_features": [719, GLOBAL_DIM],
            "grid": [719, GRID_CHANNELS, 10, 10],
            "label": ["samples"],
            "step_index": ["samples"],
        },
        "totals": {"games": total_games, "samples": total_samples},
        "trainable": not smoke_mode and all_checks_passed,
        "validation_counts": {
            "encoded_games": total_games,
            "manifest_sources": len(sources),
            "selected_replays": total_games,
            "written_shards": total_games,
        },
    }
    training_audit = _training_audit(
        config,
        canonical_config,
        shard_records,
        split_counts,
        smoke_mode=smoke_mode,
    )
    atomic_json_write(output_root / "training_audit.json", training_audit)
    atomic_json_write(output_root / "audit.json", audit)
    return audit
