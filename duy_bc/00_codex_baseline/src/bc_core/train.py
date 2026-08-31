"""Deterministic validation-only training and immutable model selection."""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import random
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler

from bc_core.constants import OPERATIONS, load_config
from bc_core.dataset import (
    Batch,
    ShardDataset,
    collate_examples,
    load_train_artifacts,
    train_artifact_identity,
)
from bc_core.metrics import classification_report, slice_reports
from bc_core.checkpoints import (
    architecture_metadata,
    choose_device,
    load_checkpoint,
    save_checkpoint,
)
from bc_core.training_audit import validate_training_audit
from model.clock import ClockOnlyModel
from model.majority import MajorityRules, ranking_to_logits
from model.state import StateAwareModel


ModelName = Literal["clock", "state"]
_CLASS_COUNT = len(OPERATIONS)
_HISTORY_FIELDS = {
    "best_epoch",
    "checkpoint_path",
    "checkpoint_sha256",
    "device",
    "elapsed_seconds",
    "epoch",
    "model_name",
    "non_improving_epochs",
    "seed",
    "train_artifact_identity",
    "train_loss",
    "preparation_identity",
    "validation_loss",
    "validation_metrics",
    "validation_slice_reports",
}
_CLASSIFICATION_REPORT_FIELDS = {
    "confusion_matrix",
    "macro_f1",
    "per_class",
    "top1",
    "top3",
}
_PER_CLASS_REPORT_FIELDS = {
    "class_id",
    "f1",
    "operation",
    "precision",
    "recall",
    "support",
}
_SLICE_DIMENSIONS = {
    "actor_type",
    "day_band",
    "route_family",
    "seat",
    "source_date",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash required input path={path}") from error
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _publish_exclusive(path: Path, encoded: bytes, description: str) -> None:
    """Publish exact bytes without overwriting and verify a concurrent winner."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"refusing conflicting {description} path={path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and every available CUDA generator."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _ShardLocalSampler(Sampler[int]):
    """Shuffle shards and their rows while visiting each shard once per epoch."""

    def __init__(
        self, dataset: ShardDataset, generator: torch.Generator
    ) -> None:
        self.dataset = dataset
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        shard_order = torch.randperm(
            len(self.dataset.shard_paths), generator=self.generator
        ).tolist()
        for shard_index in shard_order:
            start = self.dataset._prefix[shard_index]
            stop = self.dataset._prefix[shard_index + 1]
            local_order = torch.randperm(
                stop - start, generator=self.generator
            ).tolist()
            for local_index in local_order:
                yield start + local_index

    def __len__(self) -> int:
        return len(self.dataset)


def make_loaders(
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any],
    *,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader[Batch], DataLoader[Batch]]:
    """Build deterministic loaders; only the training rows are shuffled."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    seed_everything(seed)
    train_generator = torch.Generator(device="cpu")
    validation_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(seed)
    validation_generator.manual_seed(seed)
    train_sampler = (
        _ShardLocalSampler(train_dataset, train_generator)
        if isinstance(train_dataset, ShardDataset)
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=train_generator,
        num_workers=0,
        collate_fn=collate_examples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=validation_generator,
        num_workers=0,
        collate_fn=collate_examples,
    )
    return train_loader, val_loader


def _forward(
    model: nn.Module, batch: Batch, device: torch.device, model_name: ModelName
) -> torch.Tensor:
    if model_name == "clock":
        return model(batch.clock_features.to(device))
    if model_name == "state":
        return model(
            batch.grid.to(device),
            batch.global_features.to(device),
            batch.actor_features.to(device),
        )
    raise ValueError(f"unsupported model_name={model_name!r}")


def _checked_loss(loss: torch.Tensor) -> None:
    if loss.ndim != 0:
        raise ValueError("criterion must return one scalar loss")
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("non-finite loss encountered")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_name: ModelName,
) -> float:
    """Train one epoch and return the sample-weighted mean batch loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, batch, device, model_name)
        loss = criterion(logits, batch.label.to(device))
        _checked_loss(loss)
        loss.backward()
        optimizer.step()
        samples = int(batch.label.shape[0])
        total_loss += float(loss.detach().cpu()) * samples
        total_samples += samples
    if total_samples == 0:
        raise ValueError("training loader produced no samples")
    result = total_loss / total_samples
    if not math.isfinite(result):
        raise ValueError("non-finite epoch loss encountered")
    return result


def evaluate_loader(
    model: nn.Module,
    loader: Iterable[Batch],
    criterion: nn.Module,
    device: torch.device,
    model_name: ModelName,
) -> dict[str, Any]:
    """Evaluate one loader without gradients and retain row identity metadata."""
    model.eval()
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    game_ids: list[str] = []
    slices: list[dict[str, str]] = []
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for batch in loader:
            logits = _forward(model, batch, device, model_name)
            loss = criterion(logits, batch.label.to(device))
            _checked_loss(loss)
            samples = int(batch.label.shape[0])
            logits_parts.append(logits.detach().cpu().numpy())
            label_parts.append(batch.label.detach().cpu().numpy())
            game_ids.extend(batch.game_id)
            slices.extend(dict(row) for row in batch.slices)
            total_loss += float(loss.detach().cpu()) * samples
            total_samples += samples
    if total_samples == 0:
        raise ValueError("evaluation loader produced no samples")
    mean_loss = total_loss / total_samples
    if not math.isfinite(mean_loss):
        raise ValueError("non-finite evaluation loss encountered")
    logits_array = np.concatenate(logits_parts).astype(np.float32, copy=False)
    labels_array = np.concatenate(label_parts).astype(np.int64, copy=False)
    if not np.all(np.isfinite(logits_array)):
        raise ValueError("evaluation logits contain non-finite values")
    return {
        "loss": mean_loss,
        "metrics": classification_report(logits_array, labels_array),
        "slice_reports": slice_reports(logits_array, labels_array, slices),
        "logits": logits_array,
        "labels": labels_array,
        "game_ids": game_ids,
        "slices": slices,
    }


def evaluate_majority(
    loader: Iterable[Batch], majority: MajorityRules
) -> dict[str, dict[str, Any]]:
    """Evaluate train-fitted global and actor-stratified majority rankings."""
    global_row = ranking_to_logits(majority.global_ranking)
    farmer_row = ranking_to_logits(majority.farmer_ranking)
    hand_row = ranking_to_logits(majority.hand_ranking)
    global_parts: list[np.ndarray] = []
    actor_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    slices: list[dict[str, str]] = []
    game_ids: list[str] = []
    for batch in loader:
        labels = batch.label.detach().cpu().numpy().astype(np.int64, copy=False)
        global_parts.append(np.repeat(global_row[None, :], labels.shape[0], axis=0))
        actor_parts.append(
            np.stack(
                [
                    farmer_row if row["actor_type"] == "farmer" else hand_row
                    for row in batch.slices
                ]
            )
        )
        label_parts.append(labels)
        slices.extend(dict(row) for row in batch.slices)
        game_ids.extend(batch.game_id)
    if not label_parts:
        raise ValueError("majority validation loader produced no samples")
    labels_array = np.concatenate(label_parts)
    systems = {
        "global": np.concatenate(global_parts).astype(np.float32, copy=False),
        "actor": np.concatenate(actor_parts).astype(np.float32, copy=False),
    }
    return {
        name: {
            "metrics": classification_report(logits, labels_array),
            "slice_reports": slice_reports(logits, labels_array, slices),
            "game_ids": list(game_ids),
        }
        for name, logits in systems.items()
    }


def is_better_epoch(
    macro_f1: float,
    validation_loss: float,
    best_macro_f1: float,
    best_validation_loss: float,
) -> bool:
    """Compare validation epochs by macro-F1, then by loss."""
    return macro_f1 > best_macro_f1 or (
        macro_f1 == best_macro_f1 and validation_loss < best_validation_loss
    )


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read training history path={path}") from error
    for index, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL training history line={index}") from error
        if line != _canonical_bytes(record).decode("utf-8"):
            raise ValueError(f"training history is not canonical line={index}")
        if not isinstance(record, dict) or record.get("epoch") != index:
            raise ValueError("training history epochs must be a contiguous one-based prefix")
        records.append(record)
    return records


def _append_history(path: Path, record: dict[str, Any]) -> None:
    encoded = _canonical_bytes(record) + b"\n"
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = _read_history(path)
        epoch = int(record["epoch"])
        if len(existing) >= epoch:
            if existing[epoch - 1] != record:
                raise RuntimeError(f"refusing forked training history epoch={epoch}")
            return
        if len(existing) != epoch - 1:
            raise RuntimeError("training history append is not a contiguous epoch")
        with path.open("ab") as history:
            history.write(encoded)
            history.flush()
            os.fsync(history.fileno())
        _fsync_directory(path.parent)


def _preflight_identity(path: Path, identity: dict[str, Any], resume: bool) -> None:
    encoded = _canonical_bytes(identity) + b"\n"
    if path.is_symlink():
        raise ValueError("run identity must be a regular non-symlink file")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise ValueError("run directory contains a different configuration or identity")
        return
    if resume:
        raise ValueError("cannot resume without an existing run identity")


def _write_identity(path: Path, identity: dict[str, Any], resume: bool) -> None:
    encoded = _canonical_bytes(identity) + b"\n"
    _preflight_identity(path, identity, resume)
    if path.exists():
        return
    _publish_exclusive(path, encoded, "run identity")


def _preflight_epoch_artifacts(model_dir: Path, completed_epochs: int) -> None:
    """Require one exact immutable artifact quartet for every history epoch."""
    expected_checkpoints = {
        model_dir / f"epoch-{epoch:03d}.pt"
        for epoch in range(1, completed_epochs + 1)
    }
    expected_sidecars = {
        model_dir / f"epoch-{epoch:03d}.pt.sha256"
        for epoch in range(1, completed_epochs + 1)
    }
    expected_records = {
        model_dir / f"epoch-{epoch:03d}.json"
        for epoch in range(1, completed_epochs + 1)
    }
    actual_checkpoints = set(model_dir.glob("epoch-*.pt"))
    actual_sidecars = set(model_dir.glob("epoch-*.pt.sha256"))
    actual_records = set(model_dir.glob("epoch-*.json"))
    history_path = model_dir / "epochs.jsonl"
    history_present = history_path.exists()
    if (
        actual_checkpoints != expected_checkpoints
        or actual_sidecars != expected_sidecars
        or actual_records != expected_records
        or history_present is not (completed_epochs > 0)
    ):
        raise ValueError(
            "checkpoint/sidecar/epoch-record artifact sets do not match "
            "the exact history prefix"
        )
    for path in (*actual_checkpoints, *actual_sidecars, *actual_records):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"epoch artifact must be a regular non-symlink file path={path}")
    if history_present and (history_path.is_symlink() or not history_path.is_file()):
        raise ValueError("epochs.jsonl must be a regular non-symlink file")


def preflight_model_run(
    model_name: ModelName,
    run_dir: Path,
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    """Inspect one model run's immutable prefix without creating or changing paths."""
    if model_name not in ("clock", "state"):
        raise ValueError(f"unsupported model_name={model_name!r}")
    model_dir = Path(run_dir) / model_name
    if model_dir.is_symlink() or (model_dir.exists() and not model_dir.is_dir()):
        raise ValueError(f"model run path must be a regular directory path={model_dir}")
    identity_path = model_dir / "run-identity.json"
    if identity_path.is_symlink() or (
        identity_path.exists() and not identity_path.is_file()
    ):
        raise ValueError("run identity must be a regular non-symlink file")
    records = _read_history(model_dir / "epochs.jsonl")
    _preflight_epoch_artifacts(model_dir, len(records))
    if model_dir.exists():
        allowed_names = {"run-identity.json"}
        if records:
            allowed_names.update({"epochs.jsonl", "epochs.jsonl.lock"})
            for epoch in range(1, len(records) + 1):
                allowed_names.update(
                    {
                        f"epoch-{epoch:03d}.pt",
                        f"epoch-{epoch:03d}.pt.sha256",
                        f"epoch-{epoch:03d}.json",
                    }
                )
        unexpected = sorted(
            path.name
            for path in model_dir.iterdir()
            if path.name not in allowed_names
        )
        if unexpected:
            raise ValueError(
                f"model run contains unexpected artifacts model={model_name} artifacts={unexpected}"
            )
        lock_path = model_dir / "epochs.jsonl.lock"
        if (lock_path.exists() or lock_path.is_symlink()) and (
            lock_path.is_symlink()
            or not lock_path.is_file()
            or lock_path.read_bytes() != b""
        ):
            raise ValueError(
                "epochs.jsonl.lock must be an empty regular non-symlink file"
            )
    if records and not resume:
        raise ValueError("refusing to overwrite an existing training history")
    if resume and not identity_path.is_file():
        raise ValueError("cannot resume without an existing run identity")
    return records


def _model_for_name(model_name: ModelName) -> nn.Module:
    if model_name == "clock":
        return ClockOnlyModel()
    if model_name == "state":
        return StateAwareModel()
    raise ValueError(f"unsupported model_name={model_name!r}")


def _validated_fit_settings(config: dict[str, Any]) -> tuple[int, int, int]:
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("configuration requires training settings")
    if float(training.get("learning_rate", -1)) != 1e-3:
        raise ValueError("v0 training requires AdamW learning_rate=0.001")
    max_epochs = training.get("max_epochs")
    patience = training.get("patience")
    seed = config.get("seed")
    if (
        isinstance(max_epochs, bool)
        or not isinstance(max_epochs, int)
        or not 1 <= max_epochs <= 50
    ):
        raise ValueError("v0 max_epochs must be a positive integer at most 50")
    if patience != 5:
        raise ValueError("v0 early-stopping patience must equal five")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("configuration seed must be a non-negative integer")
    return seed, max_epochs, patience


def _validated_class_weights(class_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(class_weights)
    if (
        weights.shape != (_CLASS_COUNT,)
        or weights.dtype != np.float32
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("class_weights must be 17 finite positive float32 values")
    return weights


def _history_number(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float:
        raise ValueError(
            f"training history {location} must be a JSON floating-point number"
        )
    result = value
    if not math.isfinite(result):
        raise ValueError(f"training history {location} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(
            f"training history {location} must be at least {minimum}"
        )
    if maximum is not None and result > maximum:
        raise ValueError(
            f"training history {location} must be at most {maximum}"
        )
    return result


def _history_integer(
    value: Any,
    location: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"training history {location} must be a JSON integer")
    if minimum is not None and value < minimum:
        raise ValueError(
            f"training history {location} must be at least {minimum}"
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            f"training history {location} must be at most {maximum}"
        )
    return value


def _history_sha256(value: Any, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"training history {location} must be a lowercase SHA-256 digest"
        )
    return value


def _validate_classification_history_report(value: Any, location: str) -> int:
    if type(value) is not dict or set(value) != _CLASSIFICATION_REPORT_FIELDS:
        raise ValueError(
            f"training history {location} classification report fields are invalid"
        )
    top1 = _history_number(value["top1"], f"{location}.top1", minimum=0, maximum=1)
    top3 = _history_number(value["top3"], f"{location}.top3", minimum=0, maximum=1)
    macro_f1 = _history_number(
        value["macro_f1"], f"{location}.macro_f1", minimum=0, maximum=1
    )
    if top3 < top1:
        raise ValueError(f"training history {location}.top3 cannot be below top1")

    per_class = value["per_class"]
    if type(per_class) is not list or len(per_class) != _CLASS_COUNT:
        raise ValueError(
            f"training history {location}.per_class must contain {_CLASS_COUNT} rows"
        )
    per_class_values: list[tuple[float, float, float, int]] = []
    for class_id, row in enumerate(per_class):
        row_location = f"{location}.per_class[{class_id}]"
        if type(row) is not dict or set(row) != _PER_CLASS_REPORT_FIELDS:
            raise ValueError(f"training history {row_location} fields are invalid")
        if _history_integer(
            row["class_id"], f"{row_location}.class_id", minimum=0
        ) != class_id:
            raise ValueError(f"training history {row_location}.class_id is invalid")
        if type(row["operation"]) is not str or row["operation"] != OPERATIONS[class_id]:
            raise ValueError(f"training history {row_location}.operation is invalid")
        per_class_values.append(
            (
                _history_number(
                    row["precision"],
                    f"{row_location}.precision",
                    minimum=0,
                    maximum=1,
                ),
                _history_number(
                    row["recall"],
                    f"{row_location}.recall",
                    minimum=0,
                    maximum=1,
                ),
                _history_number(
                    row["f1"], f"{row_location}.f1", minimum=0, maximum=1
                ),
                _history_integer(
                    row["support"], f"{row_location}.support", minimum=0
                ),
            )
        )

    confusion = value["confusion_matrix"]
    if type(confusion) is not list or len(confusion) != _CLASS_COUNT:
        raise ValueError(
            f"training history {location}.confusion_matrix must have {_CLASS_COUNT} rows"
        )
    matrix: list[list[int]] = []
    for row_index, row in enumerate(confusion):
        row_location = f"{location}.confusion_matrix[{row_index}]"
        if type(row) is not list or len(row) != _CLASS_COUNT:
            raise ValueError(
                f"training history {row_location} must have {_CLASS_COUNT} counts"
            )
        matrix.append(
            [
                _history_integer(
                    count, f"{row_location}[{column}]", minimum=0
                )
                for column, count in enumerate(row)
            ]
        )

    supports = [values[3] for values in per_class_values]
    row_totals = [sum(row) for row in matrix]
    if supports != row_totals:
        raise ValueError(
            f"training history {location} supports do not match confusion counts"
        )
    total = sum(row_totals)
    if total <= 0:
        raise ValueError(f"training history {location} must describe at least one row")
    predicted = [
        sum(matrix[row][column] for row in range(_CLASS_COUNT))
        for column in range(_CLASS_COUNT)
    ]
    expected_f1: list[float] = []
    for class_id, (precision, recall, f1, support) in enumerate(per_class_values):
        true_positive = matrix[class_id][class_id]
        expected_precision = (
            true_positive / predicted[class_id] if predicted[class_id] else 0.0
        )
        expected_recall = true_positive / support if support else 0.0
        expected_class_f1 = (
            2.0 * expected_precision * expected_recall
            / (expected_precision + expected_recall)
            if expected_precision + expected_recall
            else 0.0
        )
        if not (
            math.isclose(precision, expected_precision, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(recall, expected_recall, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(f1, expected_class_f1, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(
                f"training history {location} per-class metrics do not match confusion counts"
            )
        expected_f1.append(expected_class_f1)
    expected_top1 = sum(matrix[index][index] for index in range(_CLASS_COUNT)) / total
    expected_macro_f1 = sum(expected_f1) / _CLASS_COUNT
    if not math.isclose(top1, expected_top1, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"training history {location}.top1 does not match confusion counts"
        )
    if not math.isclose(
        macro_f1, expected_macro_f1, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(
            f"training history {location}.macro_f1 does not match per-class metrics"
        )
    return total


def _validate_slice_history_reports(
    value: Any, location: str, expected_support: int
) -> None:
    if type(value) is not dict or set(value) != _SLICE_DIMENSIONS:
        raise ValueError(f"training history {location} slice dimensions are invalid")
    for dimension in sorted(_SLICE_DIMENSIONS):
        groups = value[dimension]
        if type(groups) is not dict or not groups:
            raise ValueError(
                f"training history {location}.{dimension} must contain slice reports"
            )
        dimension_support = 0
        for group_name, report in groups.items():
            if type(group_name) is not str:
                raise ValueError(
                    f"training history {location}.{dimension} slice name is invalid"
                )
            dimension_support += _validate_classification_history_report(
                report, f"{location}.{dimension}[{group_name!r}]"
            )
        if dimension_support != expected_support:
            raise ValueError(
                f"training history {location}.{dimension} support does not match overall report"
            )


def _validate_epoch_history_record(
    record: dict[str, Any],
    *,
    epoch: int,
    model_name: ModelName,
    model_dir: Path,
    config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    expected_device: str | None,
    patience: int,
) -> dict[str, Any]:
    location = f"model={model_name} epoch={epoch}"
    if type(record) is not dict or set(record) != _HISTORY_FIELDS:
        raise ValueError(f"training history fields are invalid {location}")
    if _history_integer(record["epoch"], f"{location}.epoch", minimum=1) != epoch:
        raise ValueError(f"training history epoch identity is invalid {location}")
    best_epoch = _history_integer(
        record["best_epoch"], f"{location}.best_epoch", minimum=1, maximum=epoch
    )
    non_improving = _history_integer(
        record["non_improving_epochs"],
        f"{location}.non_improving_epochs",
        minimum=0,
        maximum=patience,
    )
    if _history_integer(record["seed"], f"{location}.seed", minimum=0) != config["seed"]:
        raise ValueError(f"training history seed is invalid {location}")
    if type(record["model_name"]) is not str or record["model_name"] != model_name:
        raise ValueError(f"training history model identity is invalid {location}")
    if type(record["device"]) is not str or not record["device"]:
        raise ValueError(f"training history device is invalid {location}")
    if expected_device is not None and record["device"] != expected_device:
        raise ValueError(f"training history device does not match resume device {location}")
    train_loss = _history_number(
        record["train_loss"], f"{location}.train_loss", minimum=0
    )
    validation_loss = _history_number(
        record["validation_loss"], f"{location}.validation_loss", minimum=0
    )
    elapsed_seconds = _history_number(
        record["elapsed_seconds"], f"{location}.elapsed_seconds", minimum=0
    )
    del train_loss, elapsed_seconds
    validation_support = _validate_classification_history_report(
        record["validation_metrics"], f"{location}.validation_metrics"
    )
    _validate_slice_history_reports(
        record["validation_slice_reports"],
        f"{location}.validation_slice_reports",
        validation_support,
    )

    expected_artifact_identity = checkpoint_metadata.get("train_artifact_identity")
    expected_preparation_identity = checkpoint_metadata.get("preparation_identity")
    if (
        _history_sha256(
            record["train_artifact_identity"],
            f"{location}.train_artifact_identity",
        )
        != expected_artifact_identity
        or _history_sha256(
            record["preparation_identity"], f"{location}.preparation_identity"
        )
        != expected_preparation_identity
    ):
        raise ValueError(f"training history data identity is invalid {location}")

    immutable_record = model_dir / f"epoch-{epoch:03d}.json"
    expected_record_bytes = _canonical_bytes(record) + b"\n"
    try:
        immutable_bytes = immutable_record.read_bytes()
    except OSError as error:
        raise ValueError(
            f"training history immutable record is unreadable {location}"
        ) from error
    if (
        immutable_record.is_symlink()
        or not immutable_record.is_file()
        or immutable_bytes != expected_record_bytes
    ):
        raise ValueError(f"training history immutable record mismatch {location}")

    checkpoint_path = model_dir / f"epoch-{epoch:03d}.pt"
    sidecar_path = checkpoint_path.with_name(checkpoint_path.name + ".sha256")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    try:
        sidecar_content = sidecar_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"training history sidecar is unreadable {location}") from error
    if (
        type(record["checkpoint_path"]) is not str
        or record["checkpoint_path"] != str(checkpoint_path.resolve())
        or _history_sha256(
            record["checkpoint_sha256"], f"{location}.checkpoint_sha256"
        )
        != checkpoint_sha256
        or sidecar_path.is_symlink()
        or not sidecar_path.is_file()
        or sidecar_content != f"{checkpoint_sha256}\n"
    ):
        raise ValueError(f"training history checkpoint identity mismatch {location}")
    probe = _model_for_name(model_name)
    payload = load_checkpoint(checkpoint_path, probe)
    if payload["epoch"] != epoch or _canonical_bytes(
        payload["metadata"]
    ) != _canonical_bytes(checkpoint_metadata):
        raise ValueError(f"training history checkpoint metadata mismatch {location}")
    return {
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
        "macro_f1": float(record["validation_metrics"]["macro_f1"]),
        "model": probe,
        "non_improving_epochs": non_improving,
        "validation_loss": validation_loss,
    }


def _validate_history(
    records: list[dict[str, Any]],
    model_name: ModelName,
    model_dir: Path,
    config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    *,
    val_loader: DataLoader[Batch] | Iterable[Batch] | None = None,
    criterion: nn.Module | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Strictly validate one immutable history prefix and reconstruct its winner."""
    _, max_epochs, patience = _validated_fit_settings(config)
    if len(records) > max_epochs:
        raise ValueError(f"training history exceeds max epochs model={model_name}")
    _preflight_epoch_artifacts(model_dir, len(records))
    validation_enabled = val_loader is not None or criterion is not None or device is not None
    if validation_enabled and (
        val_loader is None or criterion is None or device is None
    ):
        raise ValueError("history validation re-evaluation requires loader, criterion, and device")

    best_macro_f1 = -math.inf
    best_validation_loss = math.inf
    best_epoch = 0
    best_checkpoint: Path | None = None
    non_improving = 0
    for epoch, record in enumerate(records, 1):
        validated = _validate_epoch_history_record(
            record,
            epoch=epoch,
            model_name=model_name,
            model_dir=model_dir,
            config=config,
            checkpoint_metadata=checkpoint_metadata,
            expected_device=str(device) if device is not None else None,
            patience=patience,
        )
        macro_f1 = validated["macro_f1"]
        validation_loss = validated["validation_loss"]
        checkpoint_path = validated["checkpoint_path"]
        if validation_enabled:
            assert val_loader is not None and criterion is not None and device is not None
            probe = validated["model"]
            probe.to(device)
            validation = evaluate_loader(
                probe, val_loader, criterion, device, model_name
            )
            if (
                validation_loss != float(validation["loss"])
                or _canonical_bytes(record["validation_metrics"])
                != _canonical_bytes(validation["metrics"])
                or _canonical_bytes(record["validation_slice_reports"])
                != _canonical_bytes(validation["slice_reports"])
            ):
                raise ValueError(
                    f"training history validation evidence mismatch model={model_name} epoch={epoch}"
                )
        if is_better_epoch(
            macro_f1, validation_loss, best_macro_f1, best_validation_loss
        ):
            best_macro_f1 = macro_f1
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_checkpoint = checkpoint_path
            non_improving = 0
        else:
            non_improving += 1
        if (
            validated["best_epoch"] != best_epoch
            or validated["non_improving_epochs"] != non_improving
        ):
            raise ValueError(
                f"training history selection counters mismatch model={model_name} epoch={epoch}"
            )
        if non_improving >= patience and epoch != len(records):
            raise ValueError(f"training history continues past patience model={model_name}")
    return {
        "best_checkpoint": best_checkpoint,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_validation_loss": best_validation_loss,
        "complete": len(records) == max_epochs or non_improving == patience,
        "non_improving_epochs": non_improving,
        "records": records,
    }


def preflight_model_identity(
    model_name: ModelName,
    run_dir: Path,
    config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Verify one expected model identity and its topological prefix without writes."""
    records = preflight_model_run(model_name, run_dir, resume=True)
    _preflight_identity(
        Path(run_dir) / model_name / "run-identity.json",
        {
            "model_name": model_name,
            "config": config,
            "checkpoint_metadata": checkpoint_metadata,
        },
        True,
    )
    return records


def preflight_resumed_model(
    model_name: ModelName,
    run_dir: Path,
    val_loader: DataLoader[Batch] | Iterable[Batch],
    class_weights: np.ndarray,
    config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Read and semantically verify one resumable model prefix without writes."""
    _validated_fit_settings(config)
    weights = _validated_class_weights(class_weights)
    run_dir = Path(run_dir)
    model_dir = run_dir / model_name
    records = preflight_model_identity(
        model_name, run_dir, config, checkpoint_metadata
    )
    device = choose_device()
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights.copy()).to(device)
    )
    return _validate_history(
        records,
        model_name,
        model_dir,
        config,
        checkpoint_metadata,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )


def fit_model(
    model_name: ModelName,
    train_loader: DataLoader[Batch] | Iterable[Batch],
    val_loader: DataLoader[Batch] | Iterable[Batch],
    class_weights: np.ndarray,
    config: dict[str, Any],
    run_dir: Path,
    checkpoint_metadata: dict[str, Any],
    resume: bool = False,
) -> dict[str, Any]:
    """Fit one fixed v0 model and select using validation data only."""
    seed, max_epochs, patience = _validated_fit_settings(config)
    weights = _validated_class_weights(class_weights)

    run_dir = Path(run_dir)
    model_dir = run_dir / model_name
    identity = {
        "model_name": model_name,
        "config": config,
        "checkpoint_metadata": checkpoint_metadata,
    }
    records = preflight_model_run(model_name, run_dir, resume=resume)
    _preflight_identity(model_dir / "run-identity.json", identity, resume)
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_identity(model_dir / "run-identity.json", identity, resume)
    history_path = model_dir / "epochs.jsonl"

    seed_everything(seed)
    model = _model_for_name(model_name)
    if type(model) not in (ClockOnlyModel, StateAwareModel):
        raise ValueError("v0 fitting requires an untouched fixed model instance")
    device = choose_device()
    model.to(device)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("v0 models must retain standard float32 parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights.copy()).to(device)
    )

    best_macro_f1 = -math.inf
    best_validation_loss = math.inf
    best_epoch = 0
    best_checkpoint: Path | None = None
    non_improving = 0
    start_epoch = 1
    resume_epoch: int | None = None
    if resume and records:
        resume_epoch = len(records)
        expected_checkpoints = [
            model_dir / f"epoch-{epoch:03d}.pt" for epoch in range(1, resume_epoch + 1)
        ]
        validated_history = _validate_history(
            records,
            model_name,
            model_dir,
            config,
            checkpoint_metadata,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )
        best_macro_f1 = validated_history["best_macro_f1"]
        best_validation_loss = validated_history["best_validation_loss"]
        best_epoch = validated_history["best_epoch"]
        best_checkpoint = validated_history["best_checkpoint"]
        non_improving = validated_history["non_improving_epochs"]
        payload = load_checkpoint(expected_checkpoints[-1], model, optimizer)
        if payload["epoch"] != resume_epoch:
            raise ValueError("restored checkpoint epoch does not match history prefix")
        start_epoch = resume_epoch + 1

    started = time.monotonic()
    for epoch in range(start_epoch, max_epochs + 1):
        if non_improving >= patience:
            break
        generator = getattr(train_loader, "generator", None)
        if isinstance(generator, torch.Generator):
            generator.manual_seed(seed + epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, model_name
        )
        validation = evaluate_loader(model, val_loader, criterion, device, model_name)
        validation_loss = float(validation["loss"])
        macro_f1 = float(validation["metrics"]["macro_f1"])
        improved = is_better_epoch(
            macro_f1, validation_loss, best_macro_f1, best_validation_loss
        )
        if improved:
            best_macro_f1 = macro_f1
            best_validation_loss = validation_loss
            best_epoch = epoch
            non_improving = 0
        else:
            non_improving += 1

        checkpoint_path = model_dir / f"epoch-{epoch:03d}.pt"
        checkpoint_sha256 = save_checkpoint(
            checkpoint_path, model, optimizer, checkpoint_metadata, epoch
        )
        if improved:
            best_checkpoint = checkpoint_path
        record = {
            "best_epoch": best_epoch,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
            "epoch": epoch,
            "model_name": model_name,
            "non_improving_epochs": non_improving,
            "seed": seed,
            "train_artifact_identity": checkpoint_metadata.get(
                "train_artifact_identity"
            ),
            "train_loss": train_loss,
            "preparation_identity": checkpoint_metadata.get("preparation_identity"),
            "validation_loss": validation_loss,
            "validation_metrics": validation["metrics"],
            "validation_slice_reports": validation["slice_reports"],
        }
        _publish_exclusive(
            model_dir / f"epoch-{epoch:03d}.json",
            _canonical_bytes(record) + b"\n",
            f"immutable epoch record {epoch}",
        )
        _append_history(history_path, record)
        records.append(record)

    if best_checkpoint is None:
        raise ValueError("training produced no selectable validation checkpoint")
    return {
        "best_checkpoint": best_checkpoint,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(records),
        "history_path": history_path,
        "resume_epoch": resume_epoch,
        "resumed": resume_epoch is not None,
    }


def _selection_path(path: Path) -> str:
    return str(Path(path).resolve())


def _recorded_selection_winner(
    run_dir: Path,
    model_name: ModelName,
    config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
) -> Path:
    """Reconstruct the immutable validation winner for one model history."""
    model_dir = run_dir / model_name
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError(f"selection model history directory is invalid model={model_name}")
    history_path = model_dir / "epochs.jsonl"
    records = _read_history(history_path)
    if not records:
        raise ValueError(f"selection requires a non-empty model history model={model_name}")
    validated = _validate_history(
        records, model_name, model_dir, config, checkpoint_metadata
    )
    best_checkpoint = validated["best_checkpoint"]
    if not isinstance(best_checkpoint, Path):
        raise ValueError(f"selection history has no winner model={model_name}")
    return best_checkpoint


def _selection_payload(
    run_dir: Path,
    selected: dict[str, Path],
    config_path: Path,
    artifacts_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    """Build and verify the immutable selection payload without publishing it."""
    if set(selected) != {"clock", "state"}:
        raise ValueError("selection requires exactly clock and state checkpoints")
    run_dir = Path(run_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"run directory does not exist path={run_dir}")
    config_path = Path(config_path)
    artifacts_path = Path(artifacts_path)
    audit_path = Path(audit_path)
    for description, path in (
        ("configuration", config_path),
        ("train artifacts", artifacts_path),
        ("training audit", audit_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"selection {description} must be a regular non-symlink file")
    config_content = load_config(config_path)
    try:
        audit_content = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read training audit for selection") from error
    audit_content = validate_training_audit(
        audit_content, config_content, require_trainable=True
    )
    audit_identity = audit_content["training_identity"]
    audit_records = audit_content["shards"]
    stats, _, class_weights, _, artifact_metadata = load_train_artifacts(artifacts_path)
    artifact_identity = train_artifact_identity(artifacts_path)
    artifact_sha256 = _sha256_file(artifacts_path)
    expected_artifact_metadata = {
        "schema_version": config_content["schema_version"],
        "feature_schema_version": config_content["feature_schema_version"],
        "operations": list(OPERATIONS),
        "train_shard_identities": [
            record["shard_identity"]
            for record in audit_records
            if record.get("split") == "train"
        ],
        "preparation_manifest_sha256": audit_content["safe_manifest_sha256"],
        "weight_cap": config_content["training"]["weight_cap"],
        "training_identity": audit_identity,
        "preparation_identity": audit_identity,
    }
    if _canonical_bytes(artifact_metadata) != _canonical_bytes(expected_artifact_metadata):
        raise ValueError("train artifact metadata is not bound to frozen training inputs")

    loaded_models: dict[str, dict[str, Any]] = {}
    for name in ("clock", "state"):
        model = ClockOnlyModel() if name == "clock" else StateAwareModel()
        expected_metadata = {
            "schema_version": config_content["schema_version"],
            "feature_schema_version": config_content["feature_schema_version"],
            "vocabularies": {"operations": list(OPERATIONS)},
            "normalization": {
                "global_mean": stats.global_mean,
                "global_std": stats.global_std,
                "actor_mean": stats.actor_mean,
                "actor_std": stats.actor_std,
            },
            "class_weights": class_weights,
            "manifest_sha256": audit_content["safe_manifest_sha256"],
            "architecture": architecture_metadata(model),
            "training_identity": audit_identity,
            "preparation_identity": audit_identity,
            "train_artifact_identity": artifact_identity,
            "train_artifact_sha256": artifact_sha256,
        }
        winner_path = _recorded_selection_winner(
            run_dir, name, config_content, expected_metadata
        )
        checkpoint_path = Path(selected[name])
        if checkpoint_path.is_symlink() or checkpoint_path.resolve() != winner_path.resolve():
            raise ValueError(
                f"selected checkpoint is not the recorded validation winner model={name}"
            )
        payload_checkpoint = load_checkpoint(checkpoint_path, model)
        if _canonical_bytes(payload_checkpoint["metadata"]) != _canonical_bytes(
            expected_metadata
        ):
            raise ValueError(f"checkpoint metadata is not bound to frozen inputs model={name}")
        sidecar_path = checkpoint_path.with_name(checkpoint_path.name + ".sha256")
        sidecar_bytes = sidecar_path.read_bytes()
        canonical_sidecar = sidecar_bytes.decode("ascii").strip()
        loaded_models[name] = {
            "architecture": architecture_metadata(model),
            "checkpoint_path": _selection_path(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "epoch": payload_checkpoint["epoch"],
            "sidecar": {
                "path": _selection_path(sidecar_path),
                "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                "canonical": canonical_sidecar,
            },
        }

    payload: dict[str, Any] = {
        "schema_version": "ryo-selection-v0",
        "model_schema_version": config_content.get("schema_version"),
        "feature_schema_version": config_content.get("feature_schema_version"),
        "run_id": run_dir.name,
        "operations": list(OPERATIONS),
        "safe_manifest_sha256": audit_content["safe_manifest_sha256"],
        "training_identity": audit_identity,
        "models": loaded_models,
        "config": {
            "path": _selection_path(config_path),
            "sha256": _sha256_file(config_path),
            "canonical": config_content,
        },
        "train_artifacts": {
            "path": _selection_path(artifacts_path),
            "sha256": artifact_sha256,
            "logical_identity": artifact_identity,
        },
        "training_audit": {
            "path": _selection_path(audit_path),
            "sha256": _sha256_file(audit_path),
        },
    }
    return payload


def _selection_document(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _validate_existing_selection(
    selection_path: Path, expected_payload: dict[str, Any]
) -> dict[str, Any]:
    if selection_path.is_symlink() or not selection_path.is_file():
        raise RuntimeError("selection.json must be a regular non-symlink file")
    try:
        raw = selection_path.read_bytes()
        existing = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("existing selection.json is invalid") from error
    expected_fields = set(expected_payload) | {"selection_identity", "created_at"}
    if not isinstance(existing, dict) or set(existing) != expected_fields:
        raise RuntimeError("existing selection.json has non-canonical fields")
    try:
        canonical_document = _selection_document(existing)
    except (TypeError, ValueError) as error:
        raise RuntimeError("existing selection.json is not finite canonical JSON") from error
    if raw != canonical_document:
        raise RuntimeError("existing selection.json is not canonical JSON")
    created_at = existing["created_at"]
    try:
        timestamp = datetime.fromisoformat(created_at)
    except (TypeError, ValueError) as error:
        raise RuntimeError("existing selection.json has an invalid created_at") from error
    if (
        not isinstance(created_at, str)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
        or timestamp.isoformat() != created_at
    ):
        raise RuntimeError("existing selection.json has a non-canonical created_at")
    existing_payload = dict(existing)
    existing_identity = existing_payload.pop("selection_identity")
    existing_payload.pop("created_at")
    expected_identity = hashlib.sha256(_canonical_bytes(expected_payload)).hexdigest()
    if (
        existing_identity != expected_identity
        or _canonical_bytes(existing_payload) != _canonical_bytes(expected_payload)
    ):
        raise RuntimeError("refusing conflicting selection identity")
    return existing


def preflight_selection(
    run_dir: Path,
    selected: dict[str, Path],
    config_path: Path,
    artifacts_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    """Verify an existing frozen selection and all provenance without writes."""
    run_dir = Path(run_dir)
    payload = _selection_payload(
        run_dir, selected, config_path, artifacts_path, audit_path
    )
    return _validate_existing_selection(run_dir / "selection.json", payload)


def freeze_selection(
    run_dir: Path,
    selected: dict[str, Path],
    config_path: Path,
    artifacts_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    """Freeze the exact validation-selected checkpoints and all test inputs."""
    run_dir = Path(run_dir)
    payload = _selection_payload(
        run_dir, selected, config_path, artifacts_path, audit_path
    )
    identity = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    selection_path = run_dir / "selection.json"
    if selection_path.exists() or selection_path.is_symlink():
        return _validate_existing_selection(selection_path, payload)
    candidate = dict(payload)
    candidate["selection_identity"] = identity
    candidate["created_at"] = datetime.now(timezone.utc).isoformat()
    encoded = _selection_document(candidate)
    try:
        _publish_exclusive(selection_path, encoded, "selection identity")
    except RuntimeError:
        return _validate_existing_selection(selection_path, payload)
    return candidate
