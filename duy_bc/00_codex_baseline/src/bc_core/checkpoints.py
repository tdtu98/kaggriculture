"""Fixed behavior-cloning models and immutable verified checkpoints."""

from __future__ import annotations

import copy
import hashlib
import io
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from bc_core.constants import ACTOR_DIM, GLOBAL_DIM, OPERATIONS
from bc_core.dataset import NormalizationStats
from model.clock import ClockOnlyModel
from model.state import StateAwareModel


_CLASS_COUNT = len(OPERATIONS)
_CHECKPOINT_FIELDS = {
    "model_state_dict",
    "optimizer_state_dict",
    "metadata",
    "epoch",
    "rng_state",
}
_RNG_FIELDS = {"python", "numpy", "torch", "cuda"}


def architecture_metadata(model: nn.Module) -> dict[str, Any]:
    """Return the complete canonical architecture schema for a fixed v0 model."""
    if type(model) is ClockOnlyModel:
        _require_model_graph(model, _clock_model_graph())
        return {
            "model_kind": "clock_only_v0",
            "input_dim": 8,
            "hidden_dims": [64, 64],
            "classes": 17,
            "activation": "relu",
            "linear_bias": True,
        }
    if type(model) is StateAwareModel:
        _require_model_graph(model, _state_model_graph())
        return {
            "model_kind": "state_aware_v0",
            "classes": 17,
            "tile": {
                "input_channels": 44,
                "conv_channels": [32, 64],
                "kernel_size": 3,
                "padding": 1,
                "stride": 1,
                "dilation": 1,
                "groups": 1,
                "conv_bias": True,
                "pool_output_size": [2, 2],
                "pooling": "adaptive_avg_2d",
                "flattened_dim": 256,
                "flatten_start_dim": 1,
                "flatten_end_dim": -1,
                "activation": "relu",
            },
            "actor": {
                "input_dim": 38,
                "hidden_dims": [64, 64],
                "activation": "relu",
                "linear_bias": True,
            },
            "global": {
                "input_dim": 62,
                "hidden_dims": [128, 128],
                "activation": "relu",
                "linear_bias": True,
            },
            "concatenation_order": ["tile", "global", "actor"],
            "classifier": {
                "input_dim": 448,
                "hidden_dims": [256, 128],
                "activation": "relu",
                "linear_bias": True,
                "dropout": 0.1,
                "dropout_inplace": False,
                "dropout_after_hidden_index": 0,
            },
        }
    raise ValueError(f"unsupported checkpoint model type={type(model).__qualname__}")


def _require_model_graph(model: nn.Module, expected: dict[str, Any]) -> None:
    global_hooks = {
        name: value
        for name, value in vars(torch.nn.modules.module).items()
        if name.startswith("_global")
        and "hook" in name
        and isinstance(value, Mapping)
        and value
    }
    if global_hooks:
        raise ValueError(
            "model architecture forbids global runtime hooks: "
            f"hooks={sorted(global_hooks)}"
        )
    for module in model.modules():
        _require_clean_module_registrations(module)
    _require_complete_parameter_contract(model)
    actual = {
        "direct_parameters": sorted(model._parameters),
        "direct_buffers": sorted(model._buffers),
        "children": [
            {"name": name, "module": _module_signature(module)}
            for name, module in model.named_children()
        ],
    }
    if actual != expected:
        raise ValueError(
            "model architecture graph does not match fixed v0 topology: "
            f"actual={actual!r} expected={expected!r}"
        )


def _require_clean_module_registrations(module: nn.Module) -> None:
    hook_registries = {
        name: value
        for name, value in vars(module).items()
        if "hook" in name and isinstance(value, Mapping) and value
    }
    if hook_registries:
        raise ValueError(
            "model architecture forbids runtime hooks: "
            f"module={type(module).__qualname__} hooks={sorted(hook_registries)}"
        )
    for name, parameter in module._parameters.items():
        if parameter is None:
            continue
        parameter_hooks = {
            hook_name: value
            for hook_name in ("_backward_hooks", "_post_accumulate_grad_hooks")
            if (value := getattr(parameter, hook_name, None))
        }
        if parameter_hooks:
            raise ValueError(
                "model architecture forbids parameter hooks: "
                f"module={type(module).__qualname__} parameter={name} "
                f"hooks={sorted(parameter_hooks)}"
            )

    if type(module) is ClockOnlyModel:
        expected_parameters: set[str] = set()
        expected_buffers: set[str] = set()
        expected_children = {"network"}
    elif type(module) is StateAwareModel:
        expected_parameters = set()
        expected_buffers = set()
        expected_children = {"tile", "actor", "global_encoder", "classifier"}
    elif type(module) is nn.Sequential:
        expected_parameters = set()
        expected_buffers = set()
        expected_children = {str(index) for index in range(len(module))}
    elif type(module) is nn.Linear:
        expected_parameters = {"weight", "bias"}
        expected_buffers = set()
        expected_children = set()
        _require_fixed_parameter(
            module, "weight", (module.out_features, module.in_features)
        )
        _require_fixed_parameter(module, "bias", (module.out_features,))
    elif type(module) is nn.Conv2d:
        expected_parameters = {"weight", "bias"}
        expected_buffers = set()
        expected_children = set()
        _require_fixed_parameter(
            module,
            "weight",
            (
                module.out_channels,
                module.in_channels // module.groups,
                *module.kernel_size,
            ),
        )
        _require_fixed_parameter(module, "bias", (module.out_channels,))
    elif type(module) in (nn.ReLU, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Dropout):
        expected_parameters = set()
        expected_buffers = set()
        expected_children = set()
    else:
        raise ValueError(
            f"model architecture contains unsupported module={type(module).__qualname__}"
        )

    actual_parameters = set(module._parameters)
    actual_buffers = set(module._buffers)
    actual_children = set(module._modules)
    if (
        actual_parameters != expected_parameters
        or actual_buffers != expected_buffers
        or actual_children != expected_children
    ):
        raise ValueError(
            "model architecture has unexpected registrations: "
            f"module={type(module).__qualname__} "
            f"parameters={sorted(actual_parameters)} buffers={sorted(actual_buffers)} "
            f"children={sorted(actual_children)}"
        )


def _require_fixed_parameter(
    module: nn.Module, name: str, expected_shape: tuple[int, ...]
) -> None:
    parameter = module._parameters.get(name)
    actual_shape: tuple[int, ...] | None = None
    if type(parameter) is nn.Parameter:
        try:
            actual_shape = tuple(parameter.shape)
        except RuntimeError:
            pass
    if (
        type(parameter) is not nn.Parameter
        or actual_shape != expected_shape
        or not parameter.requires_grad
        or parameter.device.type not in {"cpu", "mps", "cuda"}
        or parameter.layout != torch.strided
        or not parameter.is_floating_point()
    ):
        raise ValueError(
            "model architecture has invalid fixed parameter: "
            f"module={type(module).__qualname__} parameter={name} "
            f"type={type(parameter).__qualname__} shape={actual_shape} "
            f"expected_shape={expected_shape} "
            f"requires_grad={getattr(parameter, 'requires_grad', None)} "
            f"device={getattr(parameter, 'device', None)} "
            f"dtype={getattr(parameter, 'dtype', None)} "
            f"layout={getattr(parameter, 'layout', None)}"
        )


def _require_complete_parameter_contract(model: nn.Module) -> None:
    object_owners: dict[int, str] = {}
    storage_owners: dict[tuple[torch.device, int], str] = {}
    expected_device: torch.device | None = None
    expected_dtype: torch.dtype | None = None
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if type(parameter) is not nn.Parameter:
            raise ValueError(
                "model architecture contains a non-standard parameter object: "
                f"parameter={name} type={type(parameter).__qualname__}"
            )
        if expected_device is None:
            expected_device = parameter.device
            expected_dtype = parameter.dtype
        elif parameter.device != expected_device or parameter.dtype != expected_dtype:
            raise ValueError(
                "model architecture parameters must share one dtype and device: "
                f"parameter={name} device={parameter.device} dtype={parameter.dtype} "
                f"expected_device={expected_device} expected_dtype={expected_dtype}"
            )

        object_identity = id(parameter)
        if owner := object_owners.get(object_identity):
            raise ValueError(
                "model architecture forbids tied parameter objects: "
                f"parameter={name} aliases={owner}"
            )
        object_owners[object_identity] = name

        storage_identity = (parameter.device, parameter.untyped_storage().data_ptr())
        if owner := storage_owners.get(storage_identity):
            raise ValueError(
                "model architecture forbids shared parameter storage: "
                f"parameter={name} aliases={owner}"
            )
        storage_owners[storage_identity] = name


def _module_signature(module: nn.Module) -> dict[str, Any]:
    if type(module) is nn.Sequential:
        return {
            "type": "Sequential",
            "children": [
                {"name": name, "module": _module_signature(child)}
                for name, child in module._modules.items()
            ],
        }
    if type(module) is nn.Linear:
        return {
            "type": "Linear",
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }
    if type(module) is nn.Conv2d:
        return {
            "type": "Conv2d",
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": module.kernel_size,
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation,
            "groups": module.groups,
            "bias": module.bias is not None,
            "padding_mode": module.padding_mode,
        }
    if type(module) is nn.ReLU:
        return {"type": "ReLU", "inplace": module.inplace}
    if type(module) is nn.AdaptiveAvgPool2d:
        return {"type": "AdaptiveAvgPool2d", "output_size": module.output_size}
    if type(module) is nn.Flatten:
        return {
            "type": "Flatten",
            "start_dim": module.start_dim,
            "end_dim": module.end_dim,
        }
    if type(module) is nn.Dropout:
        return {"type": "Dropout", "p": module.p, "inplace": module.inplace}
    return {"type": f"{type(module).__module__}.{type(module).__qualname__}"}


def _clock_model_graph() -> dict[str, Any]:
    return {
        "direct_parameters": [],
        "direct_buffers": [],
        "children": [
            {
                "name": "network",
                "module": _canonical_sequential_graph(
                    [
                        {"type": "Linear", "in_features": 8, "out_features": 64, "bias": True},
                        {"type": "ReLU", "inplace": False},
                        {"type": "Linear", "in_features": 64, "out_features": 64, "bias": True},
                        {"type": "ReLU", "inplace": False},
                        {"type": "Linear", "in_features": 64, "out_features": 17, "bias": True},
                    ]
                ),
            }
        ],
    }


def _state_model_graph() -> dict[str, Any]:
    convolution_44_32 = {
        "type": "Conv2d",
        "in_channels": 44,
        "out_channels": 32,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
        "bias": True,
        "padding_mode": "zeros",
    }
    convolution_32_64 = dict(convolution_44_32, in_channels=32, out_channels=64)
    return {
        "direct_parameters": [],
        "direct_buffers": [],
        "children": [
            {
                "name": "tile",
                "module": _canonical_sequential_graph(
                    [
                        convolution_44_32,
                        {"type": "ReLU", "inplace": False},
                        convolution_32_64,
                        {"type": "ReLU", "inplace": False},
                        {"type": "AdaptiveAvgPool2d", "output_size": (2, 2)},
                        {"type": "Flatten", "start_dim": 1, "end_dim": -1},
                    ]
                ),
            },
            {
                "name": "actor",
                "module": _two_layer_graph(38, 64, 64),
            },
            {
                "name": "global_encoder",
                "module": _two_layer_graph(62, 128, 128),
            },
            {
                "name": "classifier",
                "module": _canonical_sequential_graph(
                    [
                        {"type": "Linear", "in_features": 448, "out_features": 256, "bias": True},
                        {"type": "ReLU", "inplace": False},
                        {"type": "Dropout", "p": 0.1, "inplace": False},
                        {"type": "Linear", "in_features": 256, "out_features": 128, "bias": True},
                        {"type": "ReLU", "inplace": False},
                        {"type": "Linear", "in_features": 128, "out_features": 17, "bias": True},
                    ]
                ),
            },
        ],
    }


def _two_layer_graph(input_dim: int, first: int, second: int) -> dict[str, Any]:
    return _canonical_sequential_graph(
        [
            {"type": "Linear", "in_features": input_dim, "out_features": first, "bias": True},
            {"type": "ReLU", "inplace": False},
            {"type": "Linear", "in_features": first, "out_features": second, "bias": True},
            {"type": "ReLU", "inplace": False},
        ]
    )


def _canonical_sequential_graph(children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "Sequential",
        "children": [
            {"name": str(index), "module": child}
            for index, child in enumerate(children)
        ],
    }


def choose_device() -> torch.device:
    """Choose the preferred available local accelerator."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    metadata: dict[str, Any],
    epoch: int,
) -> str:
    """Atomically create an immutable checkpoint and exact-byte SHA-256 sidecar."""
    path = Path(path)
    sidecar = _sidecar_path(path)
    if not path.parent.is_dir():
        raise ValueError(f"checkpoint parent directory does not exist path={path.parent}")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"checkpoint target is immutable path={path}")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    _validate_metadata(metadata, model)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": metadata,
        "epoch": epoch,
        "rng_state": _capture_rng_state(),
    }
    checkpoint_temporary: Path | None = None
    sidecar_temporary: Path | None = None
    installed_checkpoint = False
    installed_sidecar = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as checkpoint_file:
            checkpoint_temporary = Path(checkpoint_file.name)
            torch.save(payload, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        digest = _sha256_file(checkpoint_temporary)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as sidecar_file:
            sidecar_temporary = Path(sidecar_file.name)
            sidecar_file.write(f"{digest}\n")
            sidecar_file.flush()
            os.fsync(sidecar_file.fileno())

        os.link(checkpoint_temporary, path)
        installed_checkpoint = True
        os.link(sidecar_temporary, sidecar)
        installed_sidecar = True
    except FileExistsError as error:
        raise FileExistsError(f"checkpoint target is immutable path={path}") from error
    finally:
        if not installed_sidecar and installed_checkpoint:
            path.unlink(missing_ok=True)
        if checkpoint_temporary is not None:
            checkpoint_temporary.unlink(missing_ok=True)
        if sidecar_temporary is not None:
            sidecar_temporary.unlink(missing_ok=True)
    return digest


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> dict[str, Any]:
    """Verify and load a checkpoint; restore RNG state only when resuming."""
    path = Path(path)
    evaluation_rng = _capture_rng_state() if optimizer is None else None
    try:
        expected_digest = _read_sidecar(_sidecar_path(path))
        checkpoint_bytes = _read_checkpoint_bytes(path)
        actual_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"checkpoint SHA-256 mismatch path={path} "
                f"expected={expected_digest} actual={actual_digest}"
            )
        try:
            payload = torch.load(
                io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
            )
        except (OSError, EOFError, RuntimeError) as error:
            raise ValueError(f"cannot deserialize checkpoint path={path}") from error
        _validate_payload(payload, model)
        _preflight_checkpoint_application(payload, model, optimizer)
        model_before = copy.deepcopy(model.state_dict())
        optimizer_before = (
            copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
        )
        resume_rng_before = _capture_rng_state() if optimizer is not None else None
        try:
            model.load_state_dict(payload["model_state_dict"])
            if optimizer is not None:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
                _restore_rng_state(payload["rng_state"])
        except Exception:
            model.load_state_dict(model_before)
            if optimizer is not None:
                assert optimizer_before is not None and resume_rng_before is not None
                optimizer.load_state_dict(optimizer_before)
                _restore_rng_state(resume_rng_before)
            raise
        return payload
    finally:
        if evaluation_rng is not None:
            _restore_rng_state(evaluation_rng)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as checkpoint_file:
            while chunk := checkpoint_file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot read checkpoint path={path}") from error
    return digest.hexdigest()


def _read_checkpoint_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read checkpoint path={path}") from error


def _read_sidecar(path: Path) -> str:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read checkpoint SHA-256 sidecar path={path}") from error
    digest = text.strip()
    if text != f"{digest}\n" or not _is_sha256(digest):
        raise ValueError(f"invalid checkpoint SHA-256 sidecar path={path}")
    return digest


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _capture_rng_state() -> dict[str, Any]:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    cuda_state = state["cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("cannot restore CUDA RNG state because CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _validate_payload(payload: Any, model: nn.Module) -> None:
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_FIELDS:
        fields = sorted(payload) if isinstance(payload, dict) else None
        raise ValueError(f"checkpoint fields={fields} expected={sorted(_CHECKPOINT_FIELDS)}")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    if not isinstance(payload["optimizer_state_dict"], dict):
        raise ValueError("checkpoint optimizer_state_dict must be an object")
    epoch = payload["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    _validate_metadata(payload["metadata"], model)
    _validate_rng_state(payload["rng_state"])


def _validate_metadata(metadata: Any, model: nn.Module) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    if metadata.get("schema_version") != "ryo-bc-v0":
        raise ValueError("checkpoint schema_version does not match ryo-bc-v0")
    if metadata.get("feature_schema_version") != "ryo-features-v0":
        raise ValueError(
            "checkpoint feature_schema_version does not match ryo-features-v0"
        )
    vocabularies = metadata.get("vocabularies")
    if not isinstance(vocabularies, dict) or vocabularies.get("operations") != list(
        OPERATIONS
    ):
        raise ValueError("checkpoint operations do not match the fixed vocabulary")
    _validate_normalization(metadata.get("normalization"))
    class_weights = np.asarray(metadata.get("class_weights"))
    if (
        class_weights.shape != (_CLASS_COUNT,)
        or not np.issubdtype(class_weights.dtype, np.number)
        or not np.all(np.isfinite(class_weights))
        or np.any(class_weights <= 0)
    ):
        raise ValueError("checkpoint class_weights must contain 17 finite positive values")
    if not _is_sha256(metadata.get("manifest_sha256")):
        raise ValueError("checkpoint metadata requires manifest_sha256")
    architecture = metadata.get("architecture")
    expected_architecture = architecture_metadata(model)
    if architecture != expected_architecture:
        raise ValueError(
            "checkpoint architecture does not match supplied model: "
            f"actual={architecture!r} expected={expected_architecture!r}"
        )


def _validate_normalization(normalization: Any) -> None:
    if isinstance(normalization, NormalizationStats):
        values = {
            "global_mean": normalization.global_mean,
            "global_std": normalization.global_std,
            "actor_mean": normalization.actor_mean,
            "actor_std": normalization.actor_std,
        }
    elif isinstance(normalization, Mapping):
        values = normalization
    else:
        raise ValueError("checkpoint metadata requires normalization statistics")
    expected_shapes = {
        "global_mean": (GLOBAL_DIM,),
        "global_std": (GLOBAL_DIM,),
        "actor_mean": (ACTOR_DIM,),
        "actor_std": (ACTOR_DIM,),
    }
    for name, shape in expected_shapes.items():
        array = np.asarray(values.get(name))
        if (
            array.shape != shape
            or not np.issubdtype(array.dtype, np.number)
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(f"checkpoint normalization {name} is invalid")
        if name.endswith("_std") and np.any(array <= 0):
            raise ValueError(f"checkpoint normalization {name} must be positive")


def _validate_rng_state(state: Any) -> None:
    if not isinstance(state, dict) or set(state) != _RNG_FIELDS:
        raise ValueError("checkpoint RNG state fields are invalid")
    try:
        random.Random().setstate(state["python"])
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint Python RNG state is invalid") from error
    try:
        np.random.RandomState().set_state(state["numpy"])
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint NumPy RNG state is invalid") from error
    try:
        torch.Generator(device="cpu").set_state(state["torch"])
    except (TypeError, RuntimeError) as error:
        raise ValueError("checkpoint Torch RNG state is invalid") from error
    cuda_state = state["cuda"]
    if cuda_state is not None:
        if not isinstance(cuda_state, list) or not cuda_state:
            raise ValueError("checkpoint CUDA RNG state is invalid")
        for value in cuda_state:
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.uint8
                or value.device.type != "cpu"
                or value.ndim != 1
                or value.numel() < 16
                or value.numel() % 8 != 0
                or not value.is_contiguous()
            ):
                raise ValueError("checkpoint CUDA RNG state is invalid")


def _preflight_checkpoint_application(
    payload: dict[str, Any], model: nn.Module, optimizer: Optimizer | None
) -> None:
    _preflight_cuda_rng(payload["rng_state"])
    try:
        model_probe = copy.deepcopy(model)
        model_probe.load_state_dict(payload["model_state_dict"])
        if optimizer is not None:
            optimizer_probe = copy.deepcopy(optimizer)
            optimizer_probe.load_state_dict(payload["optimizer_state_dict"])
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("checkpoint state cannot be applied without mutation") from error


def _preflight_cuda_rng(state: Mapping[str, Any]) -> None:
    cuda_state = state["cuda"]
    if cuda_state is None:
        return
    if not torch.cuda.is_available():
        raise ValueError("checkpoint CUDA RNG state cannot be restored without CUDA")
    device_count = torch.cuda.device_count()
    if len(cuda_state) != device_count:
        raise ValueError(
            "checkpoint CUDA RNG state device count does not match available CUDA devices"
        )
    try:
        for index, value in enumerate(cuda_state):
            torch.Generator(device=torch.device("cuda", index)).set_state(value)
    except (RuntimeError, TypeError) as error:
        raise ValueError("checkpoint CUDA RNG state is not restorable") from error
