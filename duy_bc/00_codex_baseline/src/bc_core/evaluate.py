"""Frozen-input evaluation and go/no-go reporting for Ryo v0."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from bc_core.constants import ACTOR_DIM, GLOBAL_DIM, GRID_CHANNELS, OPERATIONS
from bc_core.dataset import (
    EncodedGameDataset,
    collate_examples,
    load_train_artifacts,
    validate_train_artifacts,
)
from bc_core.features import EncodedGame, encode_game, logical_shard_identity, read_shard
from bc_core.metrics import classification_report, paired_game_bootstrap, slice_reports
from bc_core.paths import corpus_path
from bc_core.checkpoints import architecture_metadata, load_checkpoint
from bc_core.replay import (
    ReplaySnapshot,
    SourceReplay,
    load_split_manifest_snapshots,
    load_validated_replay_bytes,
    read_replay_snapshot,
)
from bc_core.train import (
    preflight_model_identity,
    preflight_resumed_model,
    preflight_selection,
)
from bc_core.training_audit import validate_training_audit
from model.clock import ClockOnlyModel
from model.majority import ranking_to_logits
from model.state import StateAwareModel


_SELECTION_FIELDS = {
    "config",
    "created_at",
    "feature_schema_version",
    "model_schema_version",
    "models",
    "operations",
    "run_id",
    "safe_manifest_sha256",
    "schema_version",
    "selection_identity",
    "train_artifacts",
    "training_audit",
    "training_identity",
}
_AUDIT_FIELDS = {
    "all_checks_passed",
    "checks",
    "config",
    "label_counts",
    "manifest",
    "operations",
    "preparation_identity",
    "schema_version",
    "shard_identities",
    "shards",
    "smoke_mode",
    "source_hashes",
    "splits",
    "tensor_shapes",
    "totals",
    "trainable",
    "validation_counts",
}
_AUDIT_CHECK_FIELDS = {
    "cross_split_leakage_absent",
    "episode_ids_unique",
    "manifest_validated",
    "operation_support_covered",
    "selected_replays_validated",
    "shard_identities_verified",
    "source_hashes_unique",
    "tensor_shapes_validated",
}
_AUDIT_RECORD_FIELDS = {
    "episode_id",
    "label_counts",
    "route_family",
    "sample_count",
    "shard_identity",
    "shard_path",
    "shard_source_path",
    "source_date",
    "source_path",
    "source_sha256",
    "split",
    "tensor_shapes",
}
_SHAPE_FIELDS = {
    "actor_features",
    "argument_item",
    "argument_quantity",
    "global_features",
    "grid",
    "label",
    "step_index",
}
_SLICE_DIMENSIONS = {
    "actor_type",
    "seat",
    "day_band",
    "source_date",
    "route_family",
}
_SLICE_FAMILY_ORDER = (
    "actor_type",
    "seat",
    "day_band",
    "source_date",
    "route_family",
)
_PRIMARY_SYSTEMS = ("majority_actor", "clock", "state")


class EvaluationError(ValueError):
    """Raised when frozen evaluation inputs or outputs violate their contract."""


@dataclass(frozen=True)
class _OwnedOutput:
    device: int
    inode: int


def _canonical_json_bytes(value: Any, *, document: bool = False) -> bytes:
    separators = None if document else (",", ":")
    encoded = json.dumps(
        value,
        indent=2 if document else None,
        sort_keys=True,
        separators=separators,
        allow_nan=False,
    )
    if document:
        encoded += "\n"
    return encoded.encode("utf-8")


def _selection_fail(message: str) -> None:
    raise EvaluationError(f"selection.json {message}")


def _exact_fields(value: Any, fields: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _selection_fail(f"{location} fields are invalid")
    return value


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _selection_fail(f"{location} must be a lowercase SHA-256")
    return value


def _absolute_path(value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value:
        _selection_fail(f"{location} must be an absolute canonical path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        _selection_fail(f"{location} must be an absolute canonical path")
    return path


def _validate_selection_document(
    run_dir: Path, raw: bytes, selection: Any
) -> dict[str, Any]:
    result = _exact_fields(selection, _SELECTION_FIELDS, "top-level")
    try:
        canonical = _canonical_json_bytes(result, document=True)
    except (TypeError, ValueError) as error:
        raise EvaluationError("selection.json must contain finite canonical JSON") from error
    if raw != canonical:
        _selection_fail("is not canonical JSON")
    if result["schema_version"] != "ryo-selection-v0":
        _selection_fail("schema_version is incompatible")
    if result["model_schema_version"] != "ryo-bc-v0":
        _selection_fail("model_schema_version is incompatible")
    if result["feature_schema_version"] != "ryo-features-v0":
        _selection_fail("feature_schema_version is incompatible")
    if result["run_id"] != run_dir.name:
        _selection_fail("run_id does not match the selected run directory")
    if result["operations"] != list(OPERATIONS):
        _selection_fail("operations do not match the fixed vocabulary")
    _sha256(result["safe_manifest_sha256"], "safe_manifest_sha256")
    _sha256(result["training_identity"], "training_identity")

    models = _exact_fields(result["models"], {"clock", "state"}, "models")
    expected_architectures = {
        "clock": architecture_metadata(ClockOnlyModel()),
        "state": architecture_metadata(StateAwareModel()),
    }
    resolved_run = run_dir.resolve()
    for name in ("clock", "state"):
        model = _exact_fields(
            models[name],
            {"architecture", "checkpoint_path", "epoch", "sha256", "sidecar"},
            f"models.{name}",
        )
        epoch = model["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            _selection_fail(f"models.{name}.epoch must be a positive integer")
        if model["architecture"] != expected_architectures[name]:
            _selection_fail(f"models.{name}.architecture is incompatible")
        checkpoint_path = _absolute_path(
            model["checkpoint_path"], f"models.{name}.checkpoint_path"
        )
        expected_checkpoint = resolved_run / name / f"epoch-{epoch:03d}.pt"
        if checkpoint_path != expected_checkpoint:
            _selection_fail(
                f"models.{name}.checkpoint_path is outside its selected history"
            )
        checkpoint_sha = _sha256(model["sha256"], f"models.{name}.sha256")
        sidecar = _exact_fields(
            model["sidecar"], {"canonical", "path", "sha256"}, f"models.{name}.sidecar"
        )
        sidecar_path = _absolute_path(
            sidecar["path"], f"models.{name}.sidecar.path"
        )
        if sidecar_path != checkpoint_path.with_name(checkpoint_path.name + ".sha256"):
            _selection_fail(f"models.{name}.sidecar.path is incompatible")
        _sha256(sidecar["sha256"], f"models.{name}.sidecar.sha256")
        if sidecar["canonical"] != checkpoint_sha:
            _selection_fail(f"models.{name}.sidecar.canonical is incompatible")

    config = _exact_fields(result["config"], {"canonical", "path", "sha256"}, "config")
    _absolute_path(config["path"], "config.path")
    _sha256(config["sha256"], "config.sha256")
    canonical_config = config["canonical"]
    if not isinstance(canonical_config, dict):
        _selection_fail("config.canonical must be an object")
    if (
        canonical_config.get("schema_version") != result["model_schema_version"]
        or canonical_config.get("feature_schema_version")
        != result["feature_schema_version"]
        or canonical_config.get("bootstrap_resamples") != 10000
    ):
        _selection_fail("config.canonical is incompatible")

    artifacts = _exact_fields(
        result["train_artifacts"],
        {"logical_identity", "path", "sha256"},
        "train_artifacts",
    )
    artifact_path = _absolute_path(artifacts["path"], "train_artifacts.path")
    if artifact_path != resolved_run / "train_artifacts.npz":
        _selection_fail("train_artifacts.path does not belong to the selected run")
    _sha256(artifacts["sha256"], "train_artifacts.sha256")
    _sha256(artifacts["logical_identity"], "train_artifacts.logical_identity")

    training_audit = _exact_fields(
        result["training_audit"], {"path", "sha256"}, "training_audit"
    )
    audit_path = _absolute_path(training_audit["path"], "training_audit.path")
    if audit_path.name != "training_audit.json":
        _selection_fail("training_audit.path must name training_audit.json")
    _sha256(training_audit["sha256"], "training_audit.sha256")

    created_at = result["created_at"]
    try:
        timestamp = datetime.fromisoformat(created_at)
    except (TypeError, ValueError) as error:
        raise EvaluationError("selection.json created_at is invalid") from error
    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
        or timestamp.isoformat() != created_at
    ):
        _selection_fail("created_at is not canonical UTC")
    recorded_identity = _sha256(result["selection_identity"], "selection_identity")
    payload = dict(result)
    payload.pop("selection_identity")
    payload.pop("created_at")
    expected_identity = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if recorded_identity != expected_identity:
        _selection_fail("selection_identity does not match its payload")
    return result


def _audit_fail(message: str) -> None:
    raise EvaluationError(f"full audit {message}")


def _audit_object(value: Any, fields: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _audit_fail(f"{location} fields are invalid")
    return value


def _audit_bool(value: Any, location: str) -> bool:
    if type(value) is not bool:
        _audit_fail(f"{location} must be a boolean")
    return value


def _audit_int(value: Any, location: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        description = "positive" if positive else "non-negative"
        _audit_fail(f"{location} must be a {description} integer")
    return value


def _audit_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        _audit_fail(f"{location} must be a non-empty string")
    return value


def _audit_sha(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _audit_fail(f"{location} must be a lowercase SHA-256")
    return value


def _audit_counts(value: Any, location: str) -> list[int]:
    if not isinstance(value, list) or len(value) != len(OPERATIONS):
        _audit_fail(f"{location} must contain exactly {len(OPERATIONS)} counts")
    return [
        _audit_int(count, f"{location}[{index}]")
        for index, count in enumerate(value)
    ]


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


def _validate_audit_record(value: Any, index: int) -> dict[str, Any]:
    location = f"shards[{index}]"
    record = _audit_object(value, _AUDIT_RECORD_FIELDS, location)
    split = record["split"]
    if split not in ("train", "val", "test"):
        _audit_fail(f"{location}.split is invalid")
    episode_id = _audit_string(record["episode_id"], f"{location}.episode_id")
    if (
        episode_id in (".", "..")
        or PurePosixPath(episode_id).name != episode_id
        or "/" in episode_id
        or "\\" in episode_id
    ):
        _audit_fail(f"{location}.episode_id must be one path component")
    expected_path = f"{split}/{episode_id}.npz"
    if record["shard_path"] != expected_path:
        _audit_fail(f"{location}.shard_path must equal {expected_path}")
    for name in ("source_path", "shard_source_path", "source_date", "route_family"):
        _audit_string(record[name], f"{location}.{name}")
    _audit_sha(record["source_sha256"], f"{location}.source_sha256")
    _audit_sha(record["shard_identity"], f"{location}.shard_identity")
    sample_count = _audit_int(
        record["sample_count"], f"{location}.sample_count", positive=True
    )
    counts = _audit_counts(record["label_counts"], f"{location}.label_counts")
    if sum(counts) != sample_count:
        _audit_fail(f"{location}.label_counts do not sum to sample_count")
    shapes = _audit_object(
        record["tensor_shapes"], _SHAPE_FIELDS, f"{location}.tensor_shapes"
    )
    expected_shapes = {
        "actor_features": [sample_count, ACTOR_DIM],
        "argument_item": [sample_count],
        "argument_quantity": [sample_count],
        "global_features": [719, GLOBAL_DIM],
        "grid": [719, GRID_CHANNELS, 10, 10],
        "label": [sample_count],
        "step_index": [sample_count],
    }
    if shapes != expected_shapes:
        _audit_fail(f"{location}.tensor_shapes are incompatible")
    return record


def _validate_full_audit(
    path: Path, selection: dict[str, Any]
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError("full audit audit.json must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
        audit = json.loads(raw.decode("utf-8"))
        canonical = _canonical_json_bytes(audit, document=True)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise EvaluationError("full audit audit.json is not finite canonical JSON") from error
    if raw != canonical:
        _audit_fail("audit.json is not canonical JSON")
    result = _audit_object(audit, _AUDIT_FIELDS, "top-level")
    if result["schema_version"] != "ryo-preparation-v0":
        _audit_fail("schema_version is incompatible")
    smoke_mode = _audit_bool(result["smoke_mode"], "smoke_mode")
    all_checks = _audit_bool(result["all_checks_passed"], "all_checks_passed")
    trainable = _audit_bool(result["trainable"], "trainable")
    if smoke_mode:
        _audit_fail("smoke-mode data cannot authorize frozen evaluation")
    config = _audit_object(result["config"], {"canonical", "sha256"}, "config")
    if config["canonical"] != selection["config"]["canonical"]:
        _audit_fail("configuration does not match the frozen selection")
    expected_config_sha = hashlib.sha256(
        _canonical_json_bytes(config["canonical"])
    ).hexdigest()
    if config["sha256"] != expected_config_sha:
        _audit_fail("config.sha256 is incompatible")
    if result["operations"] != list(OPERATIONS):
        _audit_fail("operations do not match the fixed vocabulary")

    records_value = result["shards"]
    if not isinstance(records_value, list) or not records_value:
        _audit_fail("shards must be a non-empty list")
    records = [
        _validate_audit_record(record, index)
        for index, record in enumerate(records_value)
    ]
    if records != sorted(
        records, key=lambda record: (record["split"], record["episode_id"])
    ):
        _audit_fail("shards are not in canonical order")
    episode_ids = [record["episode_id"] for record in records]
    shard_identities = [record["shard_identity"] for record in records]
    source_hashes = [record["source_sha256"] for record in records]
    if len(set(episode_ids)) != len(episode_ids):
        _audit_fail("episode IDs are duplicated")
    if len(set(shard_identities)) != len(shard_identities):
        _audit_fail("shard identities are duplicated")
    if len(set(source_hashes)) != len(source_hashes):
        _audit_fail("source identities are duplicated")

    expected_splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record["split"] == split]
        if not split_records:
            _audit_fail(f"split {split} must contain at least one shard")
        counts = np.sum(
            np.asarray([record["label_counts"] for record in split_records], dtype=np.int64),
            axis=0,
        ).tolist()
        expected_splits[split] = {
            "games": len(split_records),
            "samples": sum(record["sample_count"] for record in split_records),
            "label_counts": counts,
        }
    splits = _audit_object(result["splits"], {"train", "val", "test"}, "splits")
    for split in ("train", "val", "test"):
        summary = _audit_object(
            splits[split], {"games", "label_counts", "samples"}, f"splits.{split}"
        )
        _audit_int(summary["games"], f"splits.{split}.games", positive=True)
        _audit_int(summary["samples"], f"splits.{split}.samples", positive=True)
        _audit_counts(summary["label_counts"], f"splits.{split}.label_counts")
        if summary != expected_splits[split]:
            _audit_fail(f"splits.{split} does not equal its shard projection")
    expected_total_counts = np.sum(
        np.asarray(
            [expected_splits[split]["label_counts"] for split in ("train", "val", "test")],
            dtype=np.int64,
        ),
        axis=0,
    ).tolist()
    if _audit_counts(result["label_counts"], "label_counts") != expected_total_counts:
        _audit_fail("label_counts do not equal the shard projection")
    expected_totals = {
        "games": len(records),
        "samples": sum(record["sample_count"] for record in records),
    }
    totals = _audit_object(result["totals"], {"games", "samples"}, "totals")
    if totals != expected_totals:
        _audit_fail("totals do not equal the shard projection")
    if result["shard_identities"] != sorted(shard_identities):
        _audit_fail("shard_identities do not equal the shard projection")
    if result["source_hashes"] != sorted(source_hashes):
        _audit_fail("source_hashes do not equal the shard projection")

    expected_tensor_shapes = {
        "actor_features": ["samples", ACTOR_DIM],
        "argument_item": ["samples"],
        "argument_quantity": ["samples"],
        "global_features": [719, GLOBAL_DIM],
        "grid": [719, GRID_CHANNELS, 10, 10],
        "label": ["samples"],
        "step_index": ["samples"],
    }
    if result["tensor_shapes"] != expected_tensor_shapes:
        _audit_fail("tensor_shapes are incompatible")
    manifest = _audit_object(
        result["manifest"],
        {"manifest_csv_sha256", "split_summary_json_sha256"},
        "manifest",
    )
    for name in ("manifest_csv_sha256", "split_summary_json_sha256"):
        _audit_sha(manifest[name], f"manifest.{name}")
    preparation_payload = {
        "config": config["canonical"],
        "manifest_csv_sha256": manifest["manifest_csv_sha256"],
        "shard_identities": sorted(shard_identities),
        "source_hashes": sorted(source_hashes),
        "split_summary_json_sha256": manifest["split_summary_json_sha256"],
    }
    expected_preparation_identity = hashlib.sha256(
        _canonical_json_bytes(preparation_payload)
    ).hexdigest()
    if result["preparation_identity"] != expected_preparation_identity:
        _audit_fail("preparation_identity is incompatible")

    train_support = {
        index
        for index, count in enumerate(expected_splits["train"]["label_counts"])
        if count
    }
    evaluation_support = {
        index
        for split in ("val", "test")
        for index, count in enumerate(expected_splits[split]["label_counts"])
        if count
    }
    expected_checks = {
        "cross_split_leakage_absent": True,
        "episode_ids_unique": True,
        "manifest_validated": True,
        "operation_support_covered": evaluation_support <= train_support,
        "selected_replays_validated": True,
        "shard_identities_verified": True,
        "source_hashes_unique": True,
        "tensor_shapes_validated": True,
    }
    checks = _audit_object(result["checks"], _AUDIT_CHECK_FIELDS, "checks")
    for name, value in checks.items():
        _audit_bool(value, f"checks.{name}")
    if checks != expected_checks:
        _audit_fail("checks do not equal the recomputed corpus checks")
    expected_all_checks = all(expected_checks.values())
    if all_checks is not expected_all_checks or trainable is not expected_all_checks:
        _audit_fail("all_checks_passed or trainable is inconsistent")

    validation = _audit_object(
        result["validation_counts"],
        {"encoded_games", "manifest_sources", "selected_replays", "written_shards"},
        "validation_counts",
    )
    for name, value in validation.items():
        _audit_int(value, f"validation_counts.{name}", positive=True)
    if validation != {
        "encoded_games": len(records),
        "manifest_sources": len(records),
        "selected_replays": len(records),
        "written_shards": len(records),
    }:
        _audit_fail("validation_counts do not match the complete corpus")
    return result


def _training_projection(full_audit: dict[str, Any]) -> dict[str, Any]:
    records = [
        dict(record)
        for record in full_audit["shards"]
        if record["split"] in ("train", "val")
    ]
    splits = {
        split: dict(full_audit["splits"][split]) for split in ("train", "val")
    }
    train_support = {
        index for index, count in enumerate(splits["train"]["label_counts"]) if count
    }
    val_support = {
        index for index, count in enumerate(splits["val"]["label_counts"]) if count
    }
    checks = {
        "operation_support_covered": val_support <= train_support,
        "safe_records_validated": True,
        "shard_identities_verified": True,
        "source_hashes_unique": True,
        "tensor_shapes_validated": True,
        "train_val_episode_ids_unique": True,
    }
    config = full_audit["config"]
    safe_manifest = {
        "schema_version": "ryo-training-preparation-v0",
        "records": records,
    }
    payload: dict[str, Any] = {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "config": config,
        "feature_schema_version": config["canonical"]["feature_schema_version"],
        "model_schema_version": config["canonical"]["schema_version"],
        "operations": list(OPERATIONS),
        "safe_manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(safe_manifest)
        ).hexdigest(),
        "schema_version": "ryo-training-preparation-v0",
        "shard_identities": sorted(record["shard_identity"] for record in records),
        "shards": records,
        "smoke_mode": False,
        "source_hashes": sorted(record["source_sha256"] for record in records),
        "splits": splits,
        "totals": {
            "games": sum(splits[split]["games"] for split in ("train", "val")),
            "samples": sum(splits[split]["samples"] for split in ("train", "val")),
        },
        "trainable": all(checks.values()),
    }
    payload["training_identity"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return validate_training_audit(payload, config["canonical"], require_trainable=True)


def _source_corpus_root(selection: dict[str, Any]) -> Path:
    config_path = Path(selection["config"]["path"])
    configured = selection["config"]["canonical"].get("corpus_root")
    if not isinstance(configured, str) or not configured:
        raise EvaluationError("source corpus_root is missing from frozen configuration")
    configured_path = Path(configured).expanduser()
    value: Path
    if configured_path.is_absolute():
        value = configured_path
    else:
        relative = PurePosixPath(configured.replace("\\", "/"))
        if any(part in ("", ".", "..") for part in relative.parts):
            raise EvaluationError("source corpus_root is not a safe repository-relative path")
        value = Path(*relative.parts)
    try:
        resolved = corpus_path(value, config_path=config_path).resolve(strict=True)
    except (OSError, ValueError) as error:
        raise EvaluationError(
            f"source corpus path must resolve to a directory path={configured_path}"
        ) from error
    if not resolved.is_dir():
        raise EvaluationError(
            f"source corpus path must resolve to a directory path={configured_path}"
        )
    return resolved


def _authenticate_source_corpus(
    audit: dict[str, Any], selection: dict[str, Any]
) -> tuple[dict[str, Any], dict[tuple[str, str], ReplaySnapshot]]:
    corpus_root = _source_corpus_root(selection)
    manifest_path = corpus_root / "manifest.csv"
    summary_path = corpus_root / "split_summary.json"
    for description, path in (
        ("manifest.csv", manifest_path),
        ("split_summary.json", summary_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise EvaluationError(
                f"source corpus {description} must be a regular non-symlink file"
            )
    expected_manifest = audit["manifest"]
    if _sha256_file(manifest_path) != expected_manifest["manifest_csv_sha256"]:
        raise EvaluationError("source corpus manifest.csv hash does not match full audit")
    if _sha256_file(summary_path) != expected_manifest["split_summary_json_sha256"]:
        raise EvaluationError(
            "source corpus split_summary.json hash does not match full audit"
        )
    for split in ("train", "val", "test"):
        split_root = corpus_root / split
        if split_root.is_symlink() or not split_root.is_dir():
            raise EvaluationError(
                f"source corpus split must be a regular non-symlink directory path={split_root}"
            )
    try:
        snapshots = load_split_manifest_snapshots(corpus_root)
    except Exception as error:
        raise EvaluationError(f"source corpus manifest or replay is invalid: {error}") from error

    actual_records: list[dict[str, Any]] = []
    for snapshot in snapshots:
        source = snapshot.source
        expected_path = corpus_root / source.split / f"{source.episode_id}.json"
        try:
            resolved_path = expected_path.resolve(strict=True)
        except OSError as error:
            raise EvaluationError(
                f"source corpus replay must resolve to a regular file path={expected_path}"
            ) from error
        if not resolved_path.is_file() or resolved_path.is_symlink():
            raise EvaluationError(
                f"source corpus replay must resolve to a regular file path={expected_path}"
            )
        if resolved_path != source.path:
            raise EvaluationError(
                f"source corpus replay provenance is not canonical path={expected_path}"
            )
        actual_records.append(
            {
                "episode_id": source.episode_id,
                "route_family": source.route_family,
                "shard_source_path": str(source.path),
                "source_date": source.source_date,
                "source_path": source.audit_source_path,
                "source_sha256": source.sha256,
                "split": source.split,
            }
        )
    audited_records = [
        {
            "episode_id": record["episode_id"],
            "route_family": record["route_family"],
            "shard_source_path": record["shard_source_path"],
            "source_date": record["source_date"],
            "source_path": record["source_path"],
            "source_sha256": record["source_sha256"],
            "split": record["split"],
        }
        for record in audit["shards"]
    ]
    key = lambda record: (record["split"], record["episode_id"])
    if sorted(actual_records, key=key) != sorted(audited_records, key=key):
        raise EvaluationError(
            "source corpus manifest/replay records do not exactly match the full audit"
        )
    return (
        {
            "corpus_root": str(corpus_root),
            "manifest_csv_sha256": expected_manifest["manifest_csv_sha256"],
            "split_summary_json_sha256": expected_manifest[
                "split_summary_json_sha256"
            ],
            "sources": len(actual_records),
        },
        {
            (snapshot.source.split, snapshot.source.episode_id): snapshot
            for snapshot in snapshots
        },
    )


def _reencode_source_game(
    snapshot: ReplaySnapshot, expected_module_version: str
) -> EncodedGame:
    source = snapshot.source
    try:
        replay = load_validated_replay_bytes(
            source, snapshot.content, expected_module_version
        )
        return encode_game(source, replay)
    except Exception as error:
        raise EvaluationError(
            "source replay cannot be deterministically re-encoded "
            f"split={source.split} episode={source.episode_id}: {error}"
        ) from error


def _snapshot_source_replay(source: SourceReplay) -> ReplaySnapshot:
    try:
        return read_replay_snapshot(source)
    except Exception as error:
        raise EvaluationError(
            "source replay cannot be snapshotted "
            f"split={source.split} episode={source.episode_id}: {error}"
        ) from error


def _safe_split_sources(
    audit: dict[str, Any], selection: dict[str, Any], split: str
) -> dict[tuple[str, str], ReplaySnapshot]:
    corpus_root = _source_corpus_root(selection)
    sources: dict[tuple[str, str], ReplaySnapshot] = {}
    for record in audit["shards"]:
        if record["split"] != split:
            continue
        path = Path(record["shard_source_path"])
        expected = corpus_root / split / f"{record['episode_id']}.json"
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() != expected.resolve()
        ):
            raise EvaluationError(
                "safe source replay does not match frozen training audit "
                f"split={split} episode={record['episode_id']}"
            )
        source = SourceReplay(
            split,
            str(record["episode_id"]),
            path.resolve(),
            str(record["source_sha256"]),
            str(record["source_date"]),
            str(record["route_family"]),
            str(record["source_path"]),
        )
        snapshot = _snapshot_source_replay(source)
        key = (split, source.episode_id)
        if key in sources:
            raise EvaluationError(
                f"duplicate safe source replay split={split} episode={source.episode_id}"
            )
        sources[key] = snapshot
    if not sources:
        raise EvaluationError(f"safe source replay split is empty split={split}")
    return sources


def _immutable_game(game: EncodedGame) -> EncodedGame:
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "grid",
        "global_features",
        "actor_features",
        "step_index",
        "label",
        "argument_item",
        "argument_quantity",
    ):
        array = np.array(getattr(game, name), copy=True)
        array.setflags(write=False)
        arrays[name] = array
    metadata = json.loads(json.dumps(game.metadata, sort_keys=True, allow_nan=False))
    return EncodedGame(
        arrays["grid"],
        arrays["global_features"],
        arrays["actor_features"],
        arrays["step_index"],
        arrays["label"],
        arrays["argument_item"],
        arrays["argument_quantity"],
        metadata,
    )


def _validate_row_identities(game: EncodedGame, path: Path) -> None:
    actor_indices = game.actor_features[:, 1].astype(np.float64) * 8.0
    rounded = np.rint(actor_indices)
    if (
        not np.all(np.isfinite(actor_indices))
        or not np.allclose(actor_indices, rounded, rtol=0.0, atol=1e-6)
        or np.any(rounded < 0)
    ):
        raise EvaluationError(f"invalid evaluation row actor identity path={path}")
    row_ids = [
        (int(step), int(actor))
        for step, actor in zip(game.step_index.tolist(), rounded.tolist())
    ]
    if len(set(row_ids)) != len(row_ids):
        raise EvaluationError(f"duplicate evaluation rows path={path}")
    if row_ids != sorted(row_ids):
        raise EvaluationError(f"evaluation rows are not in canonical order path={path}")


def _verified_split_shards(
    audit: dict[str, Any],
    data_root: Path,
    split: str,
    *,
    sources: dict[tuple[str, str], ReplaySnapshot] | None = None,
    expected_module_version: str | None = None,
    capture_snapshots: bool = False,
) -> tuple[list[Path], list[dict[str, Any]], list[EncodedGame]]:
    root = Path(data_root)
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError("evaluation data root must be a regular non-symlink directory")
    records = [record for record in audit["shards"] if record["split"] == split]
    if not records:
        raise EvaluationError(f"evaluation split {split} is empty")
    split_root = root / split
    if split_root.is_symlink() or not split_root.is_dir():
        raise EvaluationError(
            f"evaluation split directory must be regular and non-symlink path={split_root}"
        )
    expected_names = {f"{record['episode_id']}.npz" for record in records}
    actual_names: set[str] = set()
    for path in split_root.iterdir():
        if path.is_symlink():
            raise EvaluationError(f"evaluation shard tree contains a symlink path={path}")
        if not path.is_file():
            raise EvaluationError(f"evaluation shard tree contains an extra path={path}")
        if path.name in actual_names:
            raise EvaluationError(f"duplicate evaluation shard path={path.name}")
        actual_names.add(path.name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise EvaluationError(
            f"evaluation shard set mismatch missing={missing} extra={extra}"
        )

    paths: list[Path] = []
    identities: list[dict[str, Any]] = []
    snapshots: list[EncodedGame] = []
    seen_inodes: set[tuple[int, int]] = set()
    seen_games: set[str] = set()
    seen_logical: set[str] = set()
    seen_sources: set[str] = set()
    actual_counts = np.zeros(len(OPERATIONS), dtype=np.int64)
    actual_samples = 0
    for record in sorted(records, key=lambda item: item["episode_id"]):
        relative = Path(record["shard_path"])
        expected_relative = Path(split) / f"{record['episode_id']}.npz"
        if relative != expected_relative or relative.is_absolute() or ".." in relative.parts:
            raise EvaluationError(f"unsafe evaluation shard path={relative}")
        path = root / relative
        stat = path.stat(follow_symlinks=False)
        inode = (int(stat.st_dev), int(stat.st_ino))
        if inode in seen_inodes:
            raise EvaluationError(f"duplicate evaluation shard inode path={path}")
        seen_inodes.add(inode)
        try:
            game = read_shard(path)
            logical_identity = logical_shard_identity(game)
        except Exception as error:
            raise EvaluationError(f"invalid evaluation shard path={path}: {error}") from error
        metadata = game.metadata
        expected_metadata = {
            "split": split,
            "episode_id": record["episode_id"],
            "source_path": record["shard_source_path"],
            "source_sha256": record["source_sha256"],
            "source_date": record["source_date"],
            "route_family": record["route_family"],
        }
        actual_metadata = {name: metadata.get(name) for name in expected_metadata}
        label_counts = np.bincount(
            game.label, minlength=len(OPERATIONS)
        ).astype(np.int64).tolist()
        if (
            logical_identity != record["shard_identity"]
            or int(game.label.shape[0]) != record["sample_count"]
            or label_counts != record["label_counts"]
            or _array_shapes(game) != record["tensor_shapes"]
            or actual_metadata != expected_metadata
        ):
            raise EvaluationError(f"evaluation shard does not match full audit path={path}")
        _validate_row_identities(game, path)
        snapshot_game = game
        if sources is not None:
            if expected_module_version is None:
                raise EvaluationError(
                    "source replay module version is missing during shard authentication"
                )
            snapshot = sources.get((split, str(record["episode_id"])))
            if snapshot is None:
                raise EvaluationError(
                    "source replay is missing during prepared shard authentication "
                    f"split={split} episode={record['episode_id']}"
                )
            source_game = _reencode_source_game(snapshot, expected_module_version)
            source_identity = logical_shard_identity(source_game)
            if (
                source_identity != logical_identity
                or source_identity != record["shard_identity"]
                or int(source_game.label.shape[0]) != record["sample_count"]
                or np.bincount(
                    source_game.label, minlength=len(OPERATIONS)
                ).astype(np.int64).tolist()
                != record["label_counts"]
                or _array_shapes(source_game) != record["tensor_shapes"]
                or {
                    name: source_game.metadata.get(name) for name in expected_metadata
                }
                != expected_metadata
            ):
                raise EvaluationError(
                    "prepared shard does not match re-encoded source replay "
                    f"path={path}"
                )
            snapshot_game = source_game
        if capture_snapshots:
            snapshots.append(_immutable_game(snapshot_game))
        episode_id = str(metadata["episode_id"])
        source_identity = str(metadata["source_sha256"])
        if (
            episode_id in seen_games
            or logical_identity in seen_logical
            or source_identity in seen_sources
        ):
            raise EvaluationError("duplicate evaluation game or logical identity")
        seen_games.add(episode_id)
        seen_logical.add(logical_identity)
        seen_sources.add(source_identity)
        paths.append(path)
        identities.append(
            {
                "episode_id": episode_id,
                "logical_identity": logical_identity,
                "path": str(path.resolve()),
                "samples": int(game.label.shape[0]),
                "source_sha256": source_identity,
            }
        )
        actual_samples += int(game.label.shape[0])
        actual_counts += np.asarray(label_counts, dtype=np.int64)
    expected_summary = audit["splits"][split]
    if (
        len(paths) != expected_summary["games"]
        or actual_samples != expected_summary["samples"]
        or actual_counts.tolist() != expected_summary["label_counts"]
    ):
        raise EvaluationError("evaluation shard counts do not match the full audit")
    return paths, identities, snapshots


def _selected_validation_reports(
    run_dir: Path, selection: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for name in ("clock", "state"):
        epoch = selection["models"][name]["epoch"]
        path = run_dir / name / f"epoch-{epoch:03d}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationError(
                f"selected validation record is unreadable model={name}"
            ) from error
        reports[name] = {
            **record["validation_metrics"],
            "loss": record["validation_loss"],
            "slice_reports": record["validation_slice_reports"],
        }
    return reports


def _evaluation_systems(
    games: list[EncodedGame], selection: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str], np.ndarray, dict[str, np.ndarray]]:
    stats, _, _, majority, _ = load_train_artifacts(
        Path(selection["train_artifacts"]["path"])
    )
    dataset = EncodedGameDataset(games, stats)
    if len(dataset) == 0:
        raise EvaluationError("evaluation data is empty")
    config = selection["config"]["canonical"]
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_examples,
        generator=torch.Generator(device="cpu").manual_seed(int(config["seed"])),
    )
    clock = ClockOnlyModel()
    state = StateAwareModel()
    load_checkpoint(Path(selection["models"]["clock"]["checkpoint_path"]), clock)
    load_checkpoint(Path(selection["models"]["state"]["checkpoint_path"]), state)
    clock.eval()
    state.eval()
    global_row = ranking_to_logits(majority.global_ranking)
    farmer_row = ranking_to_logits(majority.farmer_ranking)
    hand_row = ranking_to_logits(majority.hand_ranking)
    logits_parts: dict[str, list[np.ndarray]] = {
        "majority_global": [],
        "majority_actor": [],
        "clock": [],
        "state": [],
    }
    label_parts: list[np.ndarray] = []
    game_ids: list[str] = []
    slices: list[dict[str, str]] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.label.detach().cpu().numpy().astype(np.int64, copy=False)
            clock_logits = clock(batch.clock_features).detach().cpu().numpy()
            state_logits = state(
                batch.grid, batch.global_features, batch.actor_features
            ).detach().cpu().numpy()
            expected_shape = (labels.shape[0], len(OPERATIONS))
            for name, values in (("clock", clock_logits), ("state", state_logits)):
                if values.shape != expected_shape or not np.all(np.isfinite(values)):
                    raise EvaluationError(
                        f"{name} evaluation produced non-finite logits or wrong shape"
                    )
            actor_logits = np.stack(
                [
                    farmer_row if row["actor_type"] == "farmer" else hand_row
                    for row in batch.slices
                ]
            )
            logits_parts["majority_global"].append(
                np.repeat(global_row[None, :], labels.shape[0], axis=0)
            )
            logits_parts["majority_actor"].append(actor_logits)
            logits_parts["clock"].append(clock_logits)
            logits_parts["state"].append(state_logits)
            label_parts.append(labels)
            game_ids.extend(batch.game_id)
            slices.extend(dict(row) for row in batch.slices)
    if not label_parts:
        raise EvaluationError("evaluation loader produced no rows")
    labels = np.concatenate(label_parts)
    logits = {
        name: np.concatenate(parts).astype(np.float32, copy=False)
        for name, parts in logits_parts.items()
    }
    if any(values.shape[0] != labels.shape[0] for values in logits.values()):
        raise EvaluationError("evaluation systems did not use an identical row order")
    reports = {
        name: {
            **classification_report(values, labels),
            "slice_reports": slice_reports(values, labels, slices),
        }
        for name, values in logits.items()
    }
    return reports, game_ids, labels, logits


def _diagnostics_complete(report: dict[str, dict[str, Any]], rows: int) -> bool:
    for name in _PRIMARY_SYSTEMS:
        system = report.get(name)
        if not isinstance(system, dict):
            return False
        if set(system.get("slice_reports", {})) != _SLICE_DIMENSIONS:
            return False
        per_class = system.get("per_class")
        confusion = system.get("confusion_matrix")
        if not isinstance(per_class, list) or len(per_class) != len(OPERATIONS):
            return False
        if (
            not isinstance(confusion, list)
            or len(confusion) != len(OPERATIONS)
            or any(not isinstance(row, list) or len(row) != len(OPERATIONS) for row in confusion)
        ):
            return False
        if sum(int(item["support"]) for item in per_class) != rows:
            return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise EvaluationError(f"cannot hash evaluation artifact path={path}") from error
    return digest.hexdigest()


def _capture_input_fingerprints(paths: list[Path]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for path in paths:
        candidate = Path(path)
        key = str(candidate)
        if key in fingerprints:
            continue
        if not candidate.exists() and not candidate.is_symlink():
            raise EvaluationError(f"frozen input is missing path={candidate}")
        if candidate.is_symlink() or not candidate.is_file():
            raise EvaluationError(
                f"frozen input must be a regular non-symlink file path={candidate}"
            )
        fingerprints[key] = _sha256_file(candidate)
    return fingerprints


def _assert_input_fingerprints(fingerprints: dict[str, str]) -> None:
    for value, expected in fingerprints.items():
        path = Path(value)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise EvaluationError(f"frozen input changed during evaluation path={path}")


def _selection_input_paths(
    run_dir: Path,
    selection: dict[str, Any],
    training_audit: dict[str, Any],
) -> list[Path]:
    paths = [
        run_dir / "selection.json",
        Path(selection["config"]["path"]),
        Path(selection["train_artifacts"]["path"]),
        Path(selection["training_audit"]["path"]),
    ]
    for name in ("clock", "state"):
        model_dir = run_dir / name
        if model_dir.is_symlink() or not model_dir.is_dir():
            raise EvaluationError(
                f"frozen model history directory is invalid model={name}"
            )
        for path in sorted(model_dir.iterdir(), key=lambda candidate: candidate.name):
            if path.is_symlink() or not path.is_file():
                raise EvaluationError(
                    f"frozen model history contains a non-regular path={path}"
                )
            paths.append(path)
    data_root = Path(selection["training_audit"]["path"]).parent
    for record in training_audit["shards"]:
        paths.append(data_root / str(record["shard_path"]))
        if record["split"] == "val":
            paths.append(Path(record["shard_source_path"]))
    return paths


def _corpus_input_paths(
    selection: dict[str, Any], full_audit: dict[str, Any]
) -> list[Path]:
    corpus_root = _source_corpus_root(selection)
    paths = [corpus_root / "manifest.csv", corpus_root / "split_summary.json"]
    paths.extend(Path(record["shard_source_path"]) for record in full_audit["shards"])
    return paths


def _revalidate_before_publication(
    *,
    run_dir: Path,
    data_root: Path,
    split: str,
    selection: dict[str, Any],
    training_audit: dict[str, Any],
    full_audit: dict[str, Any] | None,
    shard_identities: list[dict[str, Any]],
    fingerprints: dict[str, str],
) -> None:
    _assert_input_fingerprints(fingerprints)
    current_selection = verify_selection(run_dir)
    if _canonical_json_bytes(current_selection) != _canonical_json_bytes(selection):
        raise EvaluationError("frozen selection changed during evaluation")
    try:
        current_training = validate_training_audit(
            json.loads(
                Path(selection["training_audit"]["path"]).read_text(encoding="utf-8")
            ),
            selection["config"]["canonical"],
            require_trainable=True,
        )
    except Exception as error:
        raise EvaluationError(
            f"frozen training audit changed during evaluation: {error}"
        ) from error
    if _canonical_json_bytes(current_training) != _canonical_json_bytes(training_audit):
        raise EvaluationError("frozen training audit changed during evaluation")
    if split == "val":
        current_sources = _safe_split_sources(
            current_training, current_selection, "val"
        )
        _, current_identities, _ = _verified_split_shards(
            current_training,
            data_root,
            "val",
            sources=current_sources,
            expected_module_version=str(
                current_selection["config"]["canonical"]["module_version"]
            ),
        )
    else:
        if full_audit is None:
            raise EvaluationError("full audit snapshot is missing before publication")
        current_full_audit = _validate_full_audit(data_root / "audit.json", selection)
        if _canonical_json_bytes(current_full_audit) != _canonical_json_bytes(full_audit):
            raise EvaluationError("full audit changed during evaluation")
        current_projection = _training_projection(current_full_audit)
        if _canonical_json_bytes(current_projection) != _canonical_json_bytes(
            current_training
        ):
            raise EvaluationError(
                "full audit train/val projection changed during evaluation"
            )
        _, current_sources = _authenticate_source_corpus(
            current_full_audit, current_selection
        )
        _, current_identities, _ = _verified_split_shards(
            current_full_audit,
            data_root,
            "test",
            sources=current_sources,
            expected_module_version=str(
                current_selection["config"]["canonical"]["module_version"]
            ),
        )
    if _canonical_json_bytes(current_identities) != _canonical_json_bytes(
        shard_identities
    ):
        raise EvaluationError("evaluation shard identities changed during evaluation")
    _assert_input_fingerprints(fingerprints)


def _render_test_report(report: dict[str, Any]) -> str:
    gate = report["gate"]
    lines = [
        "# Ryo v0 frozen evaluation",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "## Corpus",
        "",
        "| Split | Games | Samples |",
        "|---|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        summary = report["corpus"][split]
        lines.append(f"| {split} | {summary['games']} | {summary['samples']} |")
    lines.extend(
        [
            "",
            "## Selected epochs",
            "",
            f"- Clock: {report['selected_epochs']['clock']}",
            f"- State: {report['selected_epochs']['state']}",
            "",
            "## Model comparison",
            "",
            "| System | Validation macro-F1 | Test top-1 | Test top-3 | Test macro-F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in _PRIMARY_SYSTEMS:
        validation = (
            "—"
            if name == "majority_actor"
            else f"{report['validation'][name]['macro_f1']:.6f}"
        )
        test = report["test"][name]
        lines.append(
            f"| {name} | {validation} | {test['top1']:.6f} | "
            f"{test['top3']:.6f} | {test['macro_f1']:.6f} |"
        )
    bootstrap = report["bootstrap"]
    lines.extend(
        [
            "",
            "## Paired game bootstrap",
            "",
            f"State minus clock top-1: {bootstrap['point_delta']:.6f}; "
            f"95% CI [{bootstrap['ci95_low']:.6f}, {bootstrap['ci95_high']:.6f}]; "
            f"{bootstrap['resamples']} resamples over {bootstrap['games']} games.",
            "",
            "## Slice diagnostics",
            "",
        ]
    )
    for family in _SLICE_FAMILY_ORDER:
        lines.extend(
            [
                f"### {family}",
                "",
                "| System | Observed value | Support | Top-1 | Top-3 | Macro-F1 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for name in _PRIMARY_SYSTEMS:
            slices = report["test"][name]["slice_reports"][family]
            for value in sorted(slices):
                metrics = slices[value]
                support = sum(
                    int(class_report["support"])
                    for class_report in metrics["per_class"]
                )
                rendered_value = str(value).replace("\\", "\\\\").replace("|", "\\|")
                lines.append(
                    f"| {name} | {rendered_value} | {support} | "
                    f"{metrics['top1']:.6f} | {metrics['top3']:.6f} | "
                    f"{metrics['macro_f1']:.6f} |"
                )
        lines.append("")
    lines.extend(["## Gate checks", ""])
    for name, passed in gate["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(["", "## Final decision", "", report["decision"], ""])
    return "\n".join(lines)


def _render_validation_report(report: dict[str, Any]) -> str:
    bootstrap = report["bootstrap"]
    lines = [
        "# Ryo v0 validation-only evaluation",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "This artifact is not a frozen test result and carries no go/no-go gate.",
        "",
        "| System | Top-1 | Top-3 | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name in _PRIMARY_SYSTEMS:
        metrics = report["validation"][name]
        lines.append(
            f"| {name} | {metrics['top1']:.6f} | {metrics['top3']:.6f} | "
            f"{metrics['macro_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"State minus clock top-1 95% CI: [{bootstrap['ci95_low']:.6f}, "
            f"{bootstrap['ci95_high']:.6f}] over {bootstrap['resamples']} resamples.",
            "",
            "## Evaluation status",
            "",
            report["decision"],
            "",
        ]
    )
    return "\n".join(lines)


def _preflight_output(path: Path, content: bytes) -> bool:
    if path.is_symlink():
        raise EvaluationError(f"evaluation output cannot be a symlink path={path}")
    if not path.exists():
        return False
    if not path.is_file():
        raise EvaluationError(f"evaluation output must be a regular file path={path}")
    try:
        existing = path.read_bytes()
    except OSError as error:
        raise EvaluationError(f"cannot read existing evaluation output path={path}") from error
    if existing != content:
        raise EvaluationError(f"refusing conflicting evaluation rerun path={path}")
    return True


def _publish_exclusive(path: Path, content: bytes) -> _OwnedOutput | None:
    temporary: Path | None = None
    owned: _OwnedOutput | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary_stat = temporary.stat(follow_symlinks=False)
        try:
            os.link(temporary, path)
            owned = _OwnedOutput(
                device=int(temporary_stat.st_dev), inode=int(temporary_stat.st_ino)
            )
        except FileExistsError:
            _preflight_output(path, content)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return owned


def _rollback_new_outputs(outputs: dict[Path, _OwnedOutput]) -> None:
    for path, owned in outputs.items():
        try:
            current = path.stat(follow_symlinks=False)
            if (int(current.st_dev), int(current.st_ino)) != (
                owned.device,
                owned.inode,
            ):
                continue
            path.unlink()
        except OSError:
            continue


def _publish_outputs(outputs: dict[Path, bytes]) -> dict[Path, _OwnedOutput]:
    existing = {
        path: _preflight_output(path, content) for path, content in outputs.items()
    }
    created: dict[Path, _OwnedOutput] = {}
    try:
        for path, content in outputs.items():
            if not existing[path]:
                owned = _publish_exclusive(path, content)
                if owned is not None:
                    created[path] = owned
    except Exception:
        _rollback_new_outputs(created)
        raise
    return created


def _assert_output_bytes(outputs: dict[Path, bytes]) -> None:
    for path, expected in outputs.items():
        if path.is_symlink() or not path.is_file():
            raise EvaluationError(f"evaluation output changed after publication path={path}")
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise EvaluationError(
                f"evaluation output changed after publication path={path}"
            ) from error
        if actual != expected:
            raise EvaluationError(f"evaluation output changed after publication path={path}")


def verify_selection(run_dir: Path) -> dict[str, Any]:
    """Verify that one run has an immutable validation selection."""
    run_dir = Path(run_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise EvaluationError("selected run must be a regular non-symlink directory")
    selection_path = run_dir / "selection.json"
    if selection_path.is_symlink() or not selection_path.is_file():
        raise EvaluationError(
            "selection.json must be a regular non-symlink file before evaluation"
        )
    try:
        raw = selection_path.read_bytes()
        selection = json.loads(raw.decode("utf-8"))
        selection = _validate_selection_document(run_dir, raw, selection)
        models = selection["models"]
        training_audit_path = Path(selection["training_audit"]["path"])
        training_audit = validate_training_audit(
            json.loads(training_audit_path.read_text(encoding="utf-8")),
            selection["config"]["canonical"],
            require_trainable=True,
        )
        fingerprints = _capture_input_fingerprints(
            _selection_input_paths(run_dir, selection, training_audit)
        )
        if fingerprints[str(selection_path)] != hashlib.sha256(raw).hexdigest():
            raise EvaluationError("selection.json changed during verification")
        verified = preflight_selection(
            run_dir,
            {
                "clock": Path(models["clock"]["checkpoint_path"]),
                "state": Path(models["state"]["checkpoint_path"]),
            },
            Path(selection["config"]["path"]),
            Path(selection["train_artifacts"]["path"]),
            Path(selection["training_audit"]["path"]),
        )
        checkpoint_metadata: dict[str, dict[str, Any]] = {}
        for name, model in (
            ("clock", ClockOnlyModel()),
            ("state", StateAwareModel()),
        ):
            payload = load_checkpoint(
                Path(models[name]["checkpoint_path"]), model
            )
            checkpoint_metadata[name] = payload["metadata"]
            preflight_model_identity(
                name,
                run_dir,
                verified["config"]["canonical"],
                payload["metadata"],
            )
        data_root = training_audit_path.parent
        config = verified["config"]["canonical"]
        train_paths, _, _ = _verified_split_shards(
            training_audit, data_root, "train"
        )
        val_sources = _safe_split_sources(training_audit, verified, "val")
        _, _, val_snapshots = _verified_split_shards(
            training_audit,
            data_root,
            "val",
            sources=val_sources,
            expected_module_version=str(config["module_version"]),
            capture_snapshots=True,
        )
        expected_artifact_metadata = {
            "schema_version": config["schema_version"],
            "feature_schema_version": config["feature_schema_version"],
            "operations": list(OPERATIONS),
            "train_shard_identities": [
                record["shard_identity"]
                for record in training_audit["shards"]
                if record["split"] == "train"
            ],
            "preparation_manifest_sha256": training_audit[
                "safe_manifest_sha256"
            ],
            "weight_cap": config["training"]["weight_cap"],
            "training_identity": training_audit["training_identity"],
            "preparation_identity": training_audit["training_identity"],
        }
        stats, _, class_weights, _, _, artifact_identity = validate_train_artifacts(
            Path(verified["train_artifacts"]["path"]),
            train_paths,
            expected_artifact_metadata,
            weight_cap=float(config["training"]["weight_cap"]),
        )
        if artifact_identity != verified["train_artifacts"]["logical_identity"]:
            raise EvaluationError("selection.json train artifact identity mismatch")
        validation_dataset = EncodedGameDataset(val_snapshots, stats)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_examples,
            generator=torch.Generator(device="cpu").manual_seed(int(config["seed"])),
        )
        for name in ("clock", "state"):
            history = preflight_resumed_model(
                name,
                run_dir,
                validation_loader,
                class_weights,
                config,
                checkpoint_metadata[name],
            )
            winner = history["best_checkpoint"]
            if (
                not history["complete"]
                or not isinstance(winner, Path)
                or winner.resolve() != Path(models[name]["checkpoint_path"]).resolve()
                or history["best_epoch"] != models[name]["epoch"]
            ):
                raise EvaluationError(
                    f"selection.json validation winner mismatch model={name}"
                )
        _assert_input_fingerprints(fingerprints)
    except EvaluationError:
        raise
    except Exception as error:
        raise EvaluationError(f"selection.json verification failed: {error}") from error
    return verified


def evaluate_frozen_run(
    run_dir: Path, data_root: Path, split: str = "test"
) -> dict[str, Any]:
    """Evaluate one split only after validating the frozen selection."""
    if split not in ("val", "test"):
        raise EvaluationError("evaluation split must be 'val' or 'test'")
    run_dir = Path(run_dir)
    selection = verify_selection(run_dir)
    data_root = Path(data_root)
    selected_data_root = Path(selection["training_audit"]["path"]).parent
    if (
        data_root.is_symlink()
        or not data_root.is_dir()
        or data_root.resolve() != selected_data_root.resolve()
    ):
        raise EvaluationError(
            "evaluation data root must equal the frozen training-audit directory"
        )
    try:
        training_audit = json.loads(
            Path(selection["training_audit"]["path"]).read_text(encoding="utf-8")
        )
        training_audit = validate_training_audit(
            training_audit,
            selection["config"]["canonical"],
            require_trainable=True,
        )
    except Exception as error:
        raise EvaluationError(f"frozen training audit is invalid: {error}") from error
    fingerprints = _capture_input_fingerprints(
        _selection_input_paths(run_dir, selection, training_audit)
    )
    if split == "val":
        val_sources = _safe_split_sources(training_audit, selection, "val")
        _, shard_identities, snapshots = _verified_split_shards(
            training_audit,
            data_root,
            "val",
            sources=val_sources,
            expected_module_version=str(
                selection["config"]["canonical"]["module_version"]
            ),
            capture_snapshots=True,
        )
        validation_reports, game_ids, labels, logits = _evaluation_systems(
            snapshots, selection
        )
        bootstrap = paired_game_bootstrap(
            np.argmax(logits["state"], axis=1) == labels,
            np.argmax(logits["clock"], axis=1) == labels,
            game_ids,
            resamples=int(selection["config"]["canonical"]["bootstrap_resamples"]),
            seed=int(selection["config"]["canonical"]["seed"]),
        )
        report = {
            "schema_version": "ryo-evaluation-v0",
            "run_id": run_dir.name,
            "split": "val",
            "selection_identity": selection["selection_identity"],
            "audit": {
                "all_checks_passed": training_audit["all_checks_passed"],
                "checks": training_audit["checks"],
                "operation_support_covered": training_audit["checks"][
                    "operation_support_covered"
                ],
                "training_identity": training_audit["training_identity"],
            },
            "artifacts": {
                "selection": {
                    "path": str((run_dir / "selection.json").resolve()),
                    "sha256": _sha256_file(run_dir / "selection.json"),
                    "logical_identity": selection["selection_identity"],
                },
                "shards": shard_identities,
            },
            "corpus": {
                name: dict(training_audit["splits"][name])
                for name in ("train", "val")
            },
            "selected_epochs": {
                name: selection["models"][name]["epoch"]
                for name in ("clock", "state")
            },
            "validation": validation_reports,
            "bootstrap": bootstrap,
            "row_order": {
                "rows": int(labels.shape[0]),
                "games": len(set(game_ids)),
                "systems": list(_PRIMARY_SYSTEMS),
            },
            "complete_diagnostics": _diagnostics_complete(
                validation_reports, int(labels.shape[0])
            ),
            "decision": "VALIDATION ONLY — NO FROZEN TEST DECISION",
        }
        _revalidate_before_publication(
            run_dir=run_dir,
            data_root=data_root,
            split="val",
            selection=selection,
            training_audit=training_audit,
            full_audit=None,
            shard_identities=shard_identities,
            fingerprints=fingerprints,
        )
        outputs = {
            run_dir / "evaluation.val.json": _canonical_json_bytes(
                report, document=True
            ),
            run_dir / "REPORT.val.md": _render_validation_report(report).encode(
                "utf-8"
            ),
        }
        created = _publish_outputs(outputs)
        try:
            _revalidate_before_publication(
                run_dir=run_dir,
                data_root=data_root,
                split="val",
                selection=selection,
                training_audit=training_audit,
                full_audit=None,
                shard_identities=shard_identities,
                fingerprints=fingerprints,
            )
            _assert_output_bytes(outputs)
        except Exception:
            _rollback_new_outputs(created)
            raise
        return report

    full_audit_path = data_root / "audit.json"
    full_audit = _validate_full_audit(full_audit_path, selection)
    projection = _training_projection(full_audit)
    if _canonical_json_bytes(projection) != _canonical_json_bytes(training_audit):
        raise EvaluationError(
            "full audit train/val projection does not match the frozen training audit"
        )
    source_corpus, test_sources = _authenticate_source_corpus(full_audit, selection)
    test_input_paths = [
        data_root / str(record["shard_path"])
        for record in full_audit["shards"]
        if record["split"] == "test"
    ]
    fingerprints.update(
        _capture_input_fingerprints(
            [full_audit_path, *_corpus_input_paths(selection, full_audit), *test_input_paths]
        )
    )
    _, shard_identities, snapshots = _verified_split_shards(
        full_audit,
        data_root,
        split,
        sources=test_sources,
        expected_module_version=str(
            selection["config"]["canonical"]["module_version"]
        ),
        capture_snapshots=True,
    )
    test_reports, game_ids, labels, logits = _evaluation_systems(
        snapshots, selection
    )
    validation_reports = _selected_validation_reports(run_dir, selection)
    state_correct = np.argmax(logits["state"], axis=1) == labels
    clock_correct = np.argmax(logits["clock"], axis=1) == labels
    config = selection["config"]["canonical"]
    bootstrap = paired_game_bootstrap(
        state_correct,
        clock_correct,
        game_ids,
        resamples=int(config["bootstrap_resamples"]),
        seed=int(config["seed"]),
    )
    complete_diagnostics = _diagnostics_complete(test_reports, int(labels.shape[0]))
    report: dict[str, Any] = {
        "schema_version": "ryo-evaluation-v0",
        "run_id": run_dir.name,
        "split": "test",
        "selection_identity": selection["selection_identity"],
        "audit": {
            "all_checks_passed": full_audit["all_checks_passed"],
            "checks": full_audit["checks"],
            "operation_support_covered": full_audit["checks"][
                "operation_support_covered"
            ],
            "preparation_identity": full_audit["preparation_identity"],
        },
        "artifacts": {
            "selection": {
                "path": str((run_dir / "selection.json").resolve()),
                "sha256": _sha256_file(run_dir / "selection.json"),
                "logical_identity": selection["selection_identity"],
            },
            "config": dict(selection["config"]),
            "train_artifacts": dict(selection["train_artifacts"]),
            "training_audit": dict(selection["training_audit"]),
            "full_audit": {
                "path": str(full_audit_path.resolve()),
                "sha256": _sha256_file(full_audit_path),
                "logical_identity": full_audit["preparation_identity"],
            },
            "source_corpus": source_corpus,
            "checkpoints": {
                name: {
                    "architecture": selection["models"][name]["architecture"],
                    "epoch": selection["models"][name]["epoch"],
                    "path": selection["models"][name]["checkpoint_path"],
                    "sha256": selection["models"][name]["sha256"],
                    "sidecar": dict(selection["models"][name]["sidecar"]),
                }
                for name in ("clock", "state")
            },
            "shards": shard_identities,
        },
        "corpus": {
            split_name: dict(full_audit["splits"][split_name])
            for split_name in ("train", "val", "test")
        },
        "selected_epochs": {
            name: selection["models"][name]["epoch"]
            for name in ("clock", "state")
        },
        "validation": validation_reports,
        "test": test_reports,
        "bootstrap": bootstrap,
        "row_order": {
            "rows": int(labels.shape[0]),
            "games": len(set(game_ids)),
            "systems": list(_PRIMARY_SYSTEMS),
        },
        "complete_diagnostics": complete_diagnostics,
    }
    report["gate"] = success_gate(report)
    report["decision"] = (
        "PROCEED TO MULTI-HEAD CLONING"
        if report["gate"]["pass"]
        else "STOP AND DIAGNOSE V0"
    )
    json_content = _canonical_json_bytes(report, document=True)
    markdown_content = _render_test_report(report).encode("utf-8")
    _revalidate_before_publication(
        run_dir=run_dir,
        data_root=data_root,
        split="test",
        selection=selection,
        training_audit=training_audit,
        full_audit=full_audit,
        shard_identities=shard_identities,
        fingerprints=fingerprints,
    )
    outputs = {
        run_dir / "evaluation.test.json": json_content,
        run_dir / "REPORT.md": markdown_content,
    }
    created = _publish_outputs(outputs)
    try:
        _revalidate_before_publication(
            run_dir=run_dir,
            data_root=data_root,
            split="test",
            selection=selection,
            training_audit=training_audit,
            full_audit=full_audit,
            shard_identities=shard_identities,
            fingerprints=fingerprints,
        )
        _assert_output_bytes(outputs)
    except Exception:
        _rollback_new_outputs(created)
        raise
    return report


def success_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the six fixed v0 go/no-go conditions."""
    try:
        checks = {
            "preparation_and_leakage_valid": report["audit"]["all_checks_passed"],
            "operation_support_covered": report["audit"][
                "operation_support_covered"
            ],
            "validation_state_macro_f1_gt_clock": report["validation"]["state"][
                "macro_f1"
            ]
            > report["validation"]["clock"]["macro_f1"],
            "test_state_macro_f1_gt_clock": report["test"]["state"]["macro_f1"]
            > report["test"]["clock"]["macro_f1"],
            "bootstrap_lower_bound_positive": report["bootstrap"]["ci95_low"]
            > 0.0,
            "complete_diagnostics": report["complete_diagnostics"],
        }
    except (KeyError, TypeError) as error:
        raise EvaluationError("evaluation report is missing a gate input") from error
    if any(type(value) is not bool for value in checks.values()):
        raise EvaluationError("evaluation gate inputs must resolve to booleans")
    failed = [name for name, passed in checks.items() if not passed]
    return {"checks": checks, "failed": failed, "pass": not failed}
