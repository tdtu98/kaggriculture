"""Exact train/validation-only preparation audit contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any

from bc_core.constants import ACTOR_DIM, GLOBAL_DIM, GRID_CHANNELS, OPERATIONS


TRAINING_AUDIT_SCHEMA = "ryo-training-preparation-v0"

_TOP_LEVEL_FIELDS = {
    "all_checks_passed",
    "checks",
    "config",
    "feature_schema_version",
    "model_schema_version",
    "operations",
    "safe_manifest_sha256",
    "schema_version",
    "shard_identities",
    "shards",
    "smoke_mode",
    "source_hashes",
    "splits",
    "totals",
    "trainable",
    "training_identity",
}
_CHECK_FIELDS = {
    "operation_support_covered",
    "safe_records_validated",
    "shard_identities_verified",
    "source_hashes_unique",
    "tensor_shapes_validated",
    "train_val_episode_ids_unique",
}
_RECORD_FIELDS = {
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


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON representation used by safe audit identities."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _fail(message: str) -> None:
    raise ValueError(f"training audit {message}")


def _exact_object(value: Any, fields: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{location} fields are not the exact safe schema")
    return value


def _exact_bool(value: Any, location: str) -> bool:
    if type(value) is not bool:
        _fail(f"{location} must be a boolean")
    return value


def _nonnegative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{location} must be a non-negative integer")
    return value


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{location} must be a non-empty string")
    return value


def _safe_source_path(value: Any, location: str) -> str:
    result = _nonempty_string(value, location)
    components = PurePosixPath(result.replace("\\", "/")).parts
    if any(component.casefold() == "test" for component in components):
        _fail(f"{location} contains a forbidden test path component")
    return result


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{location} must be a lowercase SHA-256")
    return value


def _counts(value: Any, location: str) -> list[int]:
    if not isinstance(value, list) or len(value) != len(OPERATIONS):
        _fail(f"{location} must contain exactly {len(OPERATIONS)} counts")
    return [
        _nonnegative_int(count, f"{location}[{index}]")
        for index, count in enumerate(value)
    ]


def _shape(value: Any, expected: list[int], location: str) -> None:
    if not isinstance(value, list) or value != expected or any(
        isinstance(dimension, bool) or not isinstance(dimension, int)
        for dimension in value
    ):
        _fail(f"{location} is incompatible")


def _validate_record(record: Any, index: int) -> dict[str, Any]:
    location = f"shards[{index}]"
    result = _exact_object(record, _RECORD_FIELDS, location)
    split = result["split"]
    if split not in ("train", "val"):
        _fail(f"{location}.split must be train or val")
    episode_id = _nonempty_string(result["episode_id"], f"{location}.episode_id")
    episode_path = PurePosixPath(episode_id)
    if episode_id in (".", "..") or episode_path.name != episode_id:
        _fail(f"{location}.episode_id must be one path component")
    expected_path = f"{split}/{episode_id}.npz"
    if result["shard_path"] != expected_path:
        _fail(f"{location}.shard_path must equal {expected_path}")
    for field in ("source_path", "shard_source_path"):
        _safe_source_path(result[field], f"{location}.{field}")
    for field in ("source_date", "route_family"):
        _nonempty_string(result[field], f"{location}.{field}")
    _sha256(result["source_sha256"], f"{location}.source_sha256")
    _sha256(result["shard_identity"], f"{location}.shard_identity")
    sample_count = _nonnegative_int(result["sample_count"], f"{location}.sample_count")
    if sample_count == 0:
        _fail(f"{location}.sample_count must be positive")
    label_counts = _counts(result["label_counts"], f"{location}.label_counts")
    if sum(label_counts) != sample_count:
        _fail(f"{location}.label_counts do not sum to sample_count")
    shapes = _exact_object(result["tensor_shapes"], _SHAPE_FIELDS, f"{location}.tensor_shapes")
    expected_shapes = {
        "actor_features": [sample_count, ACTOR_DIM],
        "argument_item": [sample_count],
        "argument_quantity": [sample_count],
        "global_features": [719, GLOBAL_DIM],
        "grid": [719, GRID_CHANNELS, 10, 10],
        "label": [sample_count],
        "step_index": [sample_count],
    }
    for field, expected in expected_shapes.items():
        _shape(shapes[field], expected, f"{location}.tensor_shapes.{field}")
    return result


def validate_training_audit(
    audit: Any,
    config: dict[str, Any] | None = None,
    *,
    require_trainable: bool = False,
) -> dict[str, Any]:
    """Validate every recursive field and recompute all safe audit identities."""
    result = _exact_object(audit, _TOP_LEVEL_FIELDS, "top-level")
    if result["schema_version"] != TRAINING_AUDIT_SCHEMA:
        _fail("schema_version is incompatible")
    smoke_mode = _exact_bool(result["smoke_mode"], "smoke_mode")
    all_checks_passed = _exact_bool(
        result["all_checks_passed"], "all_checks_passed"
    )
    trainable = _exact_bool(result["trainable"], "trainable")
    checks = _exact_object(result["checks"], _CHECK_FIELDS, "checks")
    for name, value in checks.items():
        _exact_bool(value, f"checks.{name}")
    if require_trainable and (
        smoke_mode or not trainable or not all_checks_passed or not all(checks.values())
    ):
        _fail("is not trainable with operation support")

    audit_config = _exact_object(result["config"], {"canonical", "sha256"}, "config")
    canonical_config = audit_config["canonical"]
    if not isinstance(canonical_config, dict):
        _fail("config.canonical must be an object")
    if config is not None and canonical_config != config:
        _fail("configuration identity is incompatible")
    expected_config_hash = hashlib.sha256(
        canonical_json_bytes(canonical_config)
    ).hexdigest()
    if audit_config["sha256"] != expected_config_hash:
        _fail("config.sha256 is incompatible")
    if result["model_schema_version"] != canonical_config.get("schema_version"):
        _fail("model_schema_version is incompatible")
    if result["feature_schema_version"] != canonical_config.get(
        "feature_schema_version"
    ):
        _fail("feature_schema_version is incompatible")
    if result["operations"] != list(OPERATIONS):
        _fail("operation vocabulary is incompatible")

    records_value = result["shards"]
    if not isinstance(records_value, list) or not records_value:
        _fail("shards must be a non-empty list")
    records = [
        _validate_record(record, index) for index, record in enumerate(records_value)
    ]
    if records != sorted(
        records, key=lambda record: (record["split"], record["episode_id"])
    ):
        _fail("shards are not in canonical order")
    episode_ids = [record["episode_id"] for record in records]
    shard_identities = [record["shard_identity"] for record in records]
    source_hashes = [record["source_sha256"] for record in records]
    if len(set(episode_ids)) != len(episode_ids):
        _fail("episode identities are not unique across train and val")
    if len(set(shard_identities)) != len(shard_identities):
        _fail("shard identities are not unique across train and val")
    if len(set(source_hashes)) != len(source_hashes):
        _fail("source identities are not unique across train and val")

    expected_splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        split_records = [record for record in records if record["split"] == split]
        counts = [0] * len(OPERATIONS)
        for record in split_records:
            counts = [
                left + right for left, right in zip(counts, record["label_counts"])
            ]
        expected_splits[split] = {
            "games": len(split_records),
            "samples": sum(record["sample_count"] for record in split_records),
            "label_counts": counts,
        }
    splits = _exact_object(result["splits"], {"train", "val"}, "splits")
    for split in ("train", "val"):
        summary = _exact_object(
            splits[split], {"games", "samples", "label_counts"}, f"splits.{split}"
        )
        _nonnegative_int(summary["games"], f"splits.{split}.games")
        _nonnegative_int(summary["samples"], f"splits.{split}.samples")
        _counts(summary["label_counts"], f"splits.{split}.label_counts")
        if summary != expected_splits[split]:
            _fail(f"splits.{split} does not equal the shard projection")
    if expected_splits["train"]["games"] == 0 or expected_splits["val"]["games"] == 0:
        _fail("requires at least one train and one val shard")

    totals = _exact_object(result["totals"], {"games", "samples"}, "totals")
    expected_totals = {
        "games": sum(summary["games"] for summary in expected_splits.values()),
        "samples": sum(summary["samples"] for summary in expected_splits.values()),
    }
    for field in ("games", "samples"):
        _nonnegative_int(totals[field], f"totals.{field}")
    if totals != expected_totals:
        _fail("totals do not equal the train/val shard projection")
    if result["shard_identities"] != sorted(shard_identities):
        _fail("shard_identities do not equal the safe shard projection")
    if result["source_hashes"] != sorted(source_hashes):
        _fail("source_hashes do not equal the safe shard projection")
    for index, value in enumerate(result["shard_identities"]):
        _sha256(value, f"shard_identities[{index}]")
    for index, value in enumerate(result["source_hashes"]):
        _sha256(value, f"source_hashes[{index}]")

    safe_manifest = {
        "schema_version": TRAINING_AUDIT_SCHEMA,
        "records": records,
    }
    expected_manifest = hashlib.sha256(canonical_json_bytes(safe_manifest)).hexdigest()
    if result["safe_manifest_sha256"] != expected_manifest:
        _fail("safe_manifest_sha256 is incompatible")
    train_support = {
        index
        for index, count in enumerate(expected_splits["train"]["label_counts"])
        if count
    }
    val_support = {
        index
        for index, count in enumerate(expected_splits["val"]["label_counts"])
        if count
    }
    expected_checks = {
        "operation_support_covered": not smoke_mode and val_support <= train_support,
        "safe_records_validated": True,
        "shard_identities_verified": True,
        "source_hashes_unique": True,
        "tensor_shapes_validated": True,
        "train_val_episode_ids_unique": True,
    }
    if checks != expected_checks:
        _fail("checks do not equal the canonical train/val projection")
    expected_all_checks = all(expected_checks.values())
    if all_checks_passed is not expected_all_checks:
        _fail("all_checks_passed is inconsistent")
    if trainable is not (not smoke_mode and expected_all_checks):
        _fail("trainable is inconsistent")
    if require_trainable and any(
        count <= 0 for count in expected_splits["train"]["label_counts"]
    ):
        _fail("requires a positive train count for all 17 operations")

    identity_payload = dict(result)
    recorded_identity = identity_payload.pop("training_identity")
    _sha256(recorded_identity, "training_identity")
    expected_identity = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
    if recorded_identity != expected_identity:
        _fail("identity is incompatible")
    return result
