#!/usr/bin/env python3
"""Train Ryo v0 baselines and freeze validation-selected checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bc_core.constants import OPERATIONS, load_config
from bc_core.dataset import (
    ShardDataset,
    fit_train_artifacts,
    save_train_artifacts,
    validate_train_artifacts,
)
from bc_core.features import logical_shard_identity, read_shard
from bc_core.checkpoints import architecture_metadata
from bc_core.paths import baseline_path
from bc_core.scripts_support import atomic_json_write
from bc_core.train import (
    evaluate_majority,
    fit_model,
    freeze_selection,
    make_loaders,
    preflight_model_identity,
    preflight_model_run,
    preflight_resumed_model,
    preflight_selection,
)
from bc_core.training_audit import validate_training_audit
from model.clock import ClockOnlyModel
from model.state import StateAwareModel


class _TrainingRunPreflight:
    __slots__ = (
        "state",
        "model_resumes",
        "stats",
        "class_counts",
        "class_weights",
        "majority",
        "artifact_identity",
        "artifact_sha256",
        "train_loader",
        "val_loader",
        "selection",
    )

    def __init__(
        self,
        *,
        state: str,
        model_resumes: dict[str, bool],
        stats: Any | None = None,
        class_counts: np.ndarray | None = None,
        class_weights: np.ndarray | None = None,
        majority: Any | None = None,
        artifact_identity: str | None = None,
        artifact_sha256: str | None = None,
        train_loader: Any | None = None,
        val_loader: Any | None = None,
        selection: dict[str, Any] | None = None,
    ) -> None:
        self.state = state
        self.model_resumes = model_resumes
        self.stats = stats
        self.class_counts = class_counts
        self.class_weights = class_weights
        self.majority = majority
        self.artifact_identity = artifact_identity
        self.artifact_sha256 = artifact_sha256
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.selection = selection


def _load_training_audit(
    audit_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read required training audit path={audit_path}") from error
    return validate_training_audit(audit, config, require_trainable=True)


def _array_shapes(game: Any) -> dict[str, list[int]]:
    return {
        "actor_features": list(game.actor_features.shape),
        "argument_item": list(game.argument_item.shape),
        "argument_quantity": list(game.argument_quantity.shape),
        "global_features": list(game.global_features.shape),
        "grid": list(game.grid.shape),
        "label": list(game.label.shape),
        "step_index": list(game.step_index.shape),
    }


def _verified_shards(audit: dict[str, Any], data_root: Path) -> dict[str, list[Path]]:
    root = data_root.resolve()
    records = audit["shards"]
    expected_paths: dict[Path, dict[str, Any]] = {}
    for record in records:
        relative = Path(str(record.get("shard_path", "")))
        expected_relative = Path(record["split"]) / f"{record['episode_id']}.npz"
        if relative != expected_relative or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe prepared shard path={relative}")
        if relative in expected_paths:
            raise ValueError(f"duplicate prepared shard path={relative}")
        expected_paths[relative] = record
    actual_paths: set[Path] = set()
    for split in ("train", "val"):
        split_root = root / split
        if split_root.is_symlink():
            raise ValueError(f"prepared shard split directory is a symlink path={split_root}")
        for directory, directory_names, file_names in os.walk(
            split_root, topdown=True, followlinks=False
        ):
            directory_path = Path(directory)
            for name in (*directory_names, *file_names):
                path = directory_path / name
                relative = path.relative_to(root)
                if path.is_symlink():
                    raise ValueError(f"prepared shard tree contains a symlink path={relative}")
            for name in directory_names:
                path = directory_path / name
                if path.suffix == ".npz":
                    raise ValueError(
                        f"prepared shard path is not a regular file path={path.relative_to(root)}"
                    )
            for name in file_names:
                path = directory_path / name
                if path.suffix != ".npz":
                    continue
                relative = path.relative_to(root)
                if not path.is_file():
                    raise ValueError(
                        f"prepared shard path is not a regular file path={relative}"
                    )
                if relative in actual_paths:
                    raise ValueError(f"duplicate prepared shard path={relative}")
                actual_paths.add(relative)
    if actual_paths != set(expected_paths):
        missing = sorted(str(path) for path in set(expected_paths) - actual_paths)
        extra = sorted(str(path) for path in actual_paths - set(expected_paths))
        raise ValueError(f"prepared shard set mismatch missing={missing} extra={extra}")

    seen_episode: set[str] = set()
    seen_identity: set[str] = set()
    seen_source: set[str] = set()
    verified: dict[str, list[Path]] = {"train": [], "val": []}
    actual_split_counts = {
        split: {"games": 0, "samples": 0, "label_counts": [0] * len(OPERATIONS)}
        for split in ("train", "val")
    }
    for relative, record in sorted(expected_paths.items(), key=lambda item: str(item[0])):
        path = root / relative
        game = read_shard(path)
        identity = logical_shard_identity(game)
        labels = np.bincount(game.label, minlength=len(OPERATIONS)).tolist()
        metadata = game.metadata
        expected_metadata = {
            "split": record["split"],
            "episode_id": record["episode_id"],
            "source_path": record["shard_source_path"],
            "source_sha256": record["source_sha256"],
            "source_date": record["source_date"],
            "route_family": record["route_family"],
        }
        actual_metadata = {name: metadata.get(name) for name in expected_metadata}
        if (
            identity != record.get("shard_identity")
            or int(game.label.shape[0]) != record.get("sample_count")
            or labels != record.get("label_counts")
            or _array_shapes(game) != record.get("tensor_shapes")
            or actual_metadata != expected_metadata
        ):
            raise ValueError(f"prepared shard does not match training audit path={path}")
        episode_id = str(record["episode_id"])
        source_identity = str(record["source_sha256"])
        if episode_id in seen_episode or identity in seen_identity or source_identity in seen_source:
            raise ValueError("prepared shard identities and sources must be unique")
        seen_episode.add(episode_id)
        seen_identity.add(identity)
        seen_source.add(source_identity)
        split = str(record["split"])
        verified[split].append(path)
        summary = actual_split_counts[split]
        summary["games"] += 1
        summary["samples"] += int(game.label.shape[0])
        summary["label_counts"] = (
            np.asarray(summary["label_counts"], dtype=np.int64)
            + np.asarray(labels, dtype=np.int64)
        ).tolist()
    if actual_split_counts != audit.get("splits"):
        raise ValueError("prepared shard counts do not match training audit")
    if sorted(seen_identity) != audit.get("shard_identities") or sorted(
        seen_source
    ) != audit.get("source_hashes"):
        raise ValueError("prepared shard identity sets do not match training audit")
    return verified


def _checkpoint_metadata(
    config: dict[str, Any],
    audit: dict[str, Any],
    stats: Any,
    class_weights: Any,
    train_artifact_identity: str,
    train_artifact_sha256: str,
    model: ClockOnlyModel | StateAwareModel,
) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "feature_schema_version": config["feature_schema_version"],
        "vocabularies": {"operations": list(OPERATIONS)},
        "normalization": {
            "global_mean": stats.global_mean,
            "global_std": stats.global_std,
            "actor_mean": stats.actor_mean,
            "actor_std": stats.actor_std,
        },
        "class_weights": class_weights,
        "manifest_sha256": audit["safe_manifest_sha256"],
        "architecture": architecture_metadata(model),
        "training_identity": audit["training_identity"],
        "preparation_identity": audit["training_identity"],
        "train_artifact_identity": train_artifact_identity,
        "train_artifact_sha256": train_artifact_sha256,
    }


def _artifact_metadata(
    config: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    train_records = sorted(
        (record for record in audit["shards"] if record["split"] == "train"),
        key=lambda record: str(record["shard_path"]),
    )
    return {
        "schema_version": config["schema_version"],
        "feature_schema_version": config["feature_schema_version"],
        "operations": list(OPERATIONS),
        "train_shard_identities": [
            record["shard_identity"] for record in train_records
        ],
        "preparation_manifest_sha256": audit["safe_manifest_sha256"],
        "weight_cap": config["training"]["weight_cap"],
        "training_identity": audit["training_identity"],
        "preparation_identity": audit["training_identity"],
    }


def _canonical_json_document(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _validate_majority_report(path: Path, expected: dict[str, Any]) -> None:
    try:
        raw = path.read_bytes()
        report = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("majority validation report is unreadable") from error
    if not isinstance(report, dict):
        raise ValueError("majority validation report must be a JSON object")
    try:
        canonical = _canonical_json_document(report)
        expected_canonical = _canonical_json_document(expected)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "majority validation report is not finite canonical JSON"
        ) from error
    if raw != canonical or canonical != expected_canonical:
        raise ValueError(
            "majority validation report does not match frozen artifacts and validation data"
        )


def _preflight_training_run(
    run_dir: Path,
    *,
    resume: bool,
    config: dict[str, Any],
    audit: dict[str, Any],
    train_shards: list[Path],
    val_shards: list[Path],
    config_path: Path,
    audit_path: Path,
) -> _TrainingRunPreflight:
    """Authorize one exact CLI state without creating, fitting, or publishing."""
    if run_dir.is_symlink() or (run_dir.exists() and not run_dir.is_dir()):
        raise ValueError(
            f"training run path must be a regular directory path={run_dir}"
        )
    if not run_dir.exists():
        if resume:
            raise ValueError("cannot resume without an existing training run")
        for name in ("clock", "state"):
            preflight_model_run(name, run_dir, resume=False)
        return _TrainingRunPreflight(
            state="fresh", model_resumes={"clock": False, "state": False}
        )

    entries = {path.name: path for path in run_dir.iterdir()}
    if not resume:
        for name in ("clock", "state"):
            preflight_model_run(name, run_dir, resume=False)
        if entries:
            raise ValueError("fresh training run directory must be empty")
        return _TrainingRunPreflight(
            state="fresh", model_resumes={"clock": False, "state": False}
        )

    allowed = {
        "clock",
        "state",
        "train_artifacts.npz",
        "majority.validation.json",
        "selection.json",
    }
    unexpected = sorted(set(entries) - allowed)
    if unexpected:
        raise ValueError(f"resume training run contains unexpected artifacts={unexpected}")
    for name in ("train_artifacts.npz", "majority.validation.json"):
        path = run_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"resume training run requires regular artifact path={path}")
    selection_path = run_dir / "selection.json"
    if selection_path.exists() and (
        selection_path.is_symlink() or not selection_path.is_file()
    ):
        raise ValueError("selection.json must be a regular non-symlink file")

    artifacts_path = run_dir / "train_artifacts.npz"
    (
        stats,
        class_counts,
        class_weights,
        majority,
        _,
        artifact_identity,
    ) = validate_train_artifacts(
        artifacts_path,
        train_shards,
        _artifact_metadata(config, audit),
        weight_cap=float(config["training"]["weight_cap"]),
    )
    artifact_sha256 = hashlib.sha256(artifacts_path.read_bytes()).hexdigest()
    train_dataset = ShardDataset(train_shards, stats)
    val_dataset = ShardDataset(val_shards, stats)
    train_loader, val_loader = make_loaders(
        train_dataset,
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        seed=int(config["seed"]),
    )
    expected_majority = evaluate_majority(val_loader, majority)
    _validate_majority_report(
        run_dir / "majority.validation.json", expected_majority
    )

    models = {"clock": ClockOnlyModel(), "state": StateAwareModel()}
    model_resumes: dict[str, bool] = {}
    checkpoint_metadatas: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        model_dir = run_dir / name
        present = model_dir.exists() or model_dir.is_symlink()
        model_resumes[name] = present
        if not present:
            preflight_model_run(name, run_dir, resume=False)
            continue
        checkpoint_metadata = _checkpoint_metadata(
            config,
            audit,
            stats,
            class_weights,
            artifact_identity,
            artifact_sha256,
            model,
        )
        checkpoint_metadatas[name] = checkpoint_metadata
        preflight_model_identity(name, run_dir, config, checkpoint_metadata)

    model_preflights: dict[str, dict[str, Any] | None] = {}
    for name in ("clock", "state"):
        if not model_resumes[name]:
            model_preflights[name] = None
            continue
        model_preflights[name] = preflight_resumed_model(
            name,
            run_dir,
            val_loader,
            class_weights,
            config,
            checkpoint_metadatas[name],
        )

    clock = model_preflights["clock"]
    state = model_preflights["state"]
    if state is not None and (clock is None or not clock["complete"]):
        raise ValueError(
            "resume state machine requires a completed clock run before state artifacts"
        )

    frozen_selection: dict[str, Any] | None = None
    if selection_path.exists() or selection_path.is_symlink():
        if (
            clock is None
            or state is None
            or not clock["complete"]
            or not state["complete"]
            or clock["best_checkpoint"] is None
            or state["best_checkpoint"] is None
        ):
            raise ValueError(
                "selection.json requires two completed model histories and winners"
            )
        frozen_selection = preflight_selection(
            run_dir,
            {
                "clock": Path(clock["best_checkpoint"]),
                "state": Path(state["best_checkpoint"]),
            },
            config_path,
            artifacts_path,
            audit_path,
        )
        state_name = "selected"
    elif clock is None:
        state_name = "baselines"
    elif not clock["complete"]:
        state_name = "clock-active"
    elif state is None:
        state_name = "clock-complete"
    elif not state["complete"]:
        state_name = "state-active"
    else:
        state_name = "models-complete"

    return _TrainingRunPreflight(
        state=state_name,
        model_resumes=model_resumes,
        stats=stats,
        class_counts=class_counts,
        class_weights=class_weights,
        majority=majority,
        artifact_identity=artifact_identity,
        artifact_sha256=artifact_sha256,
        train_loader=train_loader,
        val_loader=val_loader,
        selection=frozen_selection,
    )


def train_run(
    config_path: Path,
    run_id: str,
    data_root: Path,
    runs_root: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run train/validation-only fitting and freeze the selected inputs."""
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run-id must be one non-empty path component")
    config_path = config_path.resolve()
    data_root = data_root.resolve()
    runs_root = runs_root.resolve()
    config = load_config(config_path)
    audit_path = data_root / "training_audit.json"
    audit = _load_training_audit(audit_path, config)
    verified = _verified_shards(audit, data_root)
    train_shards = verified["train"]
    val_shards = verified["val"]

    run_dir = runs_root / run_id
    preflight = _preflight_training_run(
        run_dir,
        resume=resume,
        config=config,
        audit=audit,
        train_shards=train_shards,
        val_shards=val_shards,
        config_path=config_path,
        audit_path=audit_path,
    )
    if preflight.selection is not None:
        return preflight.selection

    artifacts_path = run_dir / "train_artifacts.npz"
    if resume:
        stats = preflight.stats
        class_counts = preflight.class_counts
        class_weights = preflight.class_weights
        majority = preflight.majority
        artifact_identity = preflight.artifact_identity
        artifact_sha256 = preflight.artifact_sha256
        train_loader = preflight.train_loader
        val_loader = preflight.val_loader
        if (
            stats is None
            or class_counts is None
            or class_weights is None
            or majority is None
            or artifact_identity is None
            or artifact_sha256 is None
            or train_loader is None
            or val_loader is None
        ):
            raise RuntimeError("resume preflight did not return its frozen inputs")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        stats, class_counts, class_weights, majority = fit_train_artifacts(
            train_shards, weight_cap=float(config["training"]["weight_cap"])
        )
        artifact_identity = save_train_artifacts(
            artifacts_path,
            stats,
            class_counts,
            class_weights,
            majority,
            _artifact_metadata(config, audit),
        )
        artifact_sha256 = hashlib.sha256(artifacts_path.read_bytes()).hexdigest()
        train_dataset = ShardDataset(train_shards, stats)
        val_dataset = ShardDataset(val_shards, stats)
        train_loader, val_loader = make_loaders(
            train_dataset,
            val_dataset,
            batch_size=int(config["training"]["batch_size"]),
            seed=int(config["seed"]),
        )
        majority_report = evaluate_majority(val_loader, majority)
        atomic_json_write(run_dir / "majority.validation.json", majority_report)

    results: dict[str, dict[str, Any]] = {}
    for name, model in (
        ("clock", ClockOnlyModel()),
        ("state", StateAwareModel()),
    ):
        results[name] = fit_model(
            name,
            train_loader,
            val_loader,
            class_weights,
            config,
            run_dir,
            _checkpoint_metadata(
                config,
                audit,
                stats,
                class_weights,
                artifact_identity,
                artifact_sha256,
                model,
            ),
            resume=preflight.model_resumes[name],
        )
    selection = freeze_selection(
        run_dir,
        {
            "clock": Path(results["clock"]["best_checkpoint"]),
            "state": Path(results["state"]["best_checkpoint"]),
        },
        config_path,
        artifacts_path,
        audit_path,
    )
    return selection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        selection = train_run(
            baseline_path(arguments.config),
            arguments.run_id,
            baseline_path(arguments.data_root),
            baseline_path(arguments.runs_root),
            resume=arguments.resume,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"selection={baseline_path(arguments.runs_root) / arguments.run_id / 'selection.json'} "
        f"identity={selection['selection_identity']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
