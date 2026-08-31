import hashlib
import random
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import bc_core.checkpoints as model_module
from bc_core.constants import OPERATIONS
from bc_core.dataset import NormalizationStats
from bc_core.checkpoints import (
    choose_device,
    load_checkpoint,
    save_checkpoint,
)
from model.clock import ClockOnlyModel
from model.state import StateAwareModel


class ModelTest(unittest.TestCase):
    def test_state_aware_output_shape_and_parameter_budget(self) -> None:
        # Catches a broken three-encoder concatenation or an enlarged v0 model.
        model = StateAwareModel()
        logits = model(
            torch.zeros(4, 44, 10, 10),
            torch.zeros(4, 62),
            torch.zeros(4, 38),
        )
        self.assertEqual(tuple(logits.shape), (4, 17))
        parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
        self.assertGreater(parameters, 190_000)
        self.assertLess(parameters, 230_000)

    def test_clock_model_accepts_only_eight_features(self) -> None:
        # Catches accidentally exposing state-aware features to the clock baseline.
        model = ClockOnlyModel()
        self.assertEqual(tuple(model(torch.zeros(4, 8)).shape), (4, 17))
        with self.assertRaises(RuntimeError):
            model(torch.zeros(4, 9))

    def test_fixed_model_module_topologies_are_exact(self) -> None:
        # Catches changed layer order, dimensions, pooling, or dropout despite valid shapes.
        clock = ClockOnlyModel()
        self.assertEqual(
            [type(layer) for layer in clock.network],
            [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear],
        )
        self.assertEqual(
            [
                (clock.network[index].in_features, clock.network[index].out_features)
                for index in (0, 2, 4)
            ],
            [(8, 64), (64, 64), (64, 17)],
        )
        self.assertTrue(all(clock.network[index].bias is not None for index in (0, 2, 4)))
        self.assertFalse(clock.network[1].inplace)
        self.assertFalse(clock.network[3].inplace)

        state = StateAwareModel()
        self.assertEqual(
            [type(layer) for layer in state.tile],
            [
                torch.nn.Conv2d,
                torch.nn.ReLU,
                torch.nn.Conv2d,
                torch.nn.ReLU,
                torch.nn.AdaptiveAvgPool2d,
                torch.nn.Flatten,
            ],
        )
        self.assertEqual(
            [
                (
                    state.tile[index].in_channels,
                    state.tile[index].out_channels,
                    state.tile[index].kernel_size,
                    state.tile[index].padding,
                )
                for index in (0, 2)
            ],
            [(44, 32, (3, 3), (1, 1)), (32, 64, (3, 3), (1, 1))],
        )
        self.assertEqual(state.tile[4].output_size, (2, 2))
        for index in (0, 2):
            self.assertEqual(state.tile[index].stride, (1, 1))
            self.assertEqual(state.tile[index].dilation, (1, 1))
            self.assertEqual(state.tile[index].groups, 1)
            self.assertIsNotNone(state.tile[index].bias)
        self.assertFalse(state.tile[1].inplace)
        self.assertFalse(state.tile[3].inplace)
        self.assertEqual(state.tile[5].start_dim, 1)
        self.assertEqual(state.tile[5].end_dim, -1)
        self.assert_linear_stack(state.actor, [(38, 64), (64, 64)])
        self.assert_linear_stack(state.global_encoder, [(62, 128), (128, 128)])
        self.assertEqual(
            [type(layer) for layer in state.classifier],
            [
                torch.nn.Linear,
                torch.nn.ReLU,
                torch.nn.Dropout,
                torch.nn.Linear,
                torch.nn.ReLU,
                torch.nn.Linear,
            ],
        )
        self.assertEqual(
            [
                (state.classifier[index].in_features, state.classifier[index].out_features)
                for index in (0, 3, 5)
            ],
            [(448, 256), (256, 128), (128, 17)],
        )
        self.assertEqual(state.classifier[2].p, 0.1)
        self.assertFalse(state.classifier[2].inplace)
        self.assertTrue(
            all(state.classifier[index].bias is not None for index in (0, 3, 5))
        )
        self.assertFalse(state.classifier[1].inplace)
        self.assertFalse(state.classifier[4].inplace)

    def test_fixed_parameter_contract_accepts_uniform_real_floating_dtypes(self) -> None:
        # Catches overbinding the fixed topology to one valid floating precision.
        for factory in (ClockOnlyModel, StateAwareModel):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(model=factory.__name__, dtype=str(dtype)):
                    model = factory().to(dtype=dtype)
                    self.assertIsInstance(
                        model_module.architecture_metadata(model), dict
                    )

    def test_choose_device_prefers_mps_then_cuda_then_cpu(self) -> None:
        # Catches reversing the documented accelerator priority or omitting CPU fallback.
        cases = (
            (True, True, "mps"),
            (False, True, "cuda"),
            (False, False, "cpu"),
        )
        for mps_available, cuda_available, expected in cases:
            with self.subTest(expected=expected), patch(
                "torch.backends.mps.is_available", return_value=mps_available
            ), patch("torch.cuda.is_available", return_value=cuda_available):
                self.assertEqual(choose_device(), torch.device(expected))

    def test_checkpoint_round_trip_preserves_logits_and_complete_payload(self) -> None:
        # Catches omitted model/provenance state or loading weights other than the saved bytes.
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        model = ClockOnlyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 10
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        model.eval()
        expected_logits = model(inputs).detach().clone()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch-003.pt"
            digest = save_checkpoint(path, model, optimizer, self.metadata(), epoch=3)
            expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected_digest)
            self.assertEqual(path.with_name(path.name + ".sha256").read_text().strip(), digest)

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(100.0)

            restored = ClockOnlyModel()
            load_checkpoint(path, restored)
            restored.eval()
            torch.testing.assert_close(restored(inputs), expected_logits, rtol=0, atol=0)

            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(
                set(payload),
                {"model_state_dict", "optimizer_state_dict", "metadata", "epoch", "rng_state"},
            )
            self.assertEqual(payload["epoch"], 3)
            self.assertTrue(payload["optimizer_state_dict"]["state"])
            self.assertEqual(
                set(payload["rng_state"]), {"python", "numpy", "torch", "cuda"}
            )
            self.assertIsInstance(payload["rng_state"]["python"], tuple)
            self.assertIsInstance(payload["rng_state"]["numpy"], tuple)
            self.assertIsInstance(payload["rng_state"]["torch"], torch.Tensor)
            if torch.cuda.is_available():
                cuda_state = payload["rng_state"]["cuda"]
                self.assertIsInstance(cuda_state, list)
                self.assertEqual(len(cuda_state), torch.cuda.device_count())
                self.assertTrue(
                    all(
                        value.dtype == torch.uint8
                        and value.device.type == "cpu"
                        and value.ndim == 1
                        and value.numel() >= 16
                        for value in cuda_state
                    )
                )
            else:
                self.assertIsNone(payload["rng_state"]["cuda"])
            self.assertEqual(payload["metadata"]["schema_version"], "ryo-bc-v0")
            self.assertEqual(
                payload["metadata"]["feature_schema_version"], "ryo-features-v0"
            )
            self.assertEqual(payload["metadata"]["vocabularies"]["operations"], list(OPERATIONS))
            self.assertIsInstance(payload["metadata"]["normalization"], NormalizationStats)
            np.testing.assert_array_equal(
                payload["metadata"]["class_weights"], np.ones(17, dtype=np.float32)
            )
            self.assertEqual(payload["metadata"]["manifest_sha256"], "b" * 64)
            self.assertEqual(payload["metadata"]["architecture"]["classes"], 17)

    def test_checkpoint_target_is_immutable(self) -> None:
        # Catches silently replacing an epoch checkpoint with later model state.
        model = ClockOnlyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch-001.pt"
            save_checkpoint(path, model, optimizer, self.metadata(), epoch=1)
            original_checkpoint = path.read_bytes()
            original_sidecar = path.with_name(path.name + ".sha256").read_bytes()
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, optimizer, self.metadata(), epoch=1)
            self.assertEqual(path.read_bytes(), original_checkpoint)
            self.assertEqual(
                path.with_name(path.name + ".sha256").read_bytes(), original_sidecar
            )

    def test_corrupt_checkpoint_hash_is_rejected(self) -> None:
        # Catches deserializing bytes that no longer match the frozen checkpoint identity.
        model = ClockOnlyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch-001.pt"
            save_checkpoint(path, model, optimizer, self.metadata(), epoch=1)
            checkpoint_bytes = bytearray(path.read_bytes())
            checkpoint_bytes[len(checkpoint_bytes) // 2] ^= 1
            path.write_bytes(checkpoint_bytes)
            with patch("bc_core.checkpoints.torch.load") as mocked_load:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    load_checkpoint(path, ClockOnlyModel())
            mocked_load.assert_not_called()

    def test_second_atomic_link_failure_removes_checkpoint_and_temporary_files(self) -> None:
        # Catches leaving a loadable-looking checkpoint when sidecar publication fails.
        model = ClockOnlyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        real_link = model_module.os.link
        link_count = 0

        def fail_second_link(source: object, destination: object) -> None:
            nonlocal link_count
            link_count += 1
            if link_count == 2:
                raise OSError("injected sidecar link failure")
            real_link(source, destination)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch-001.pt"
            with patch("bc_core.checkpoints.os.link", side_effect=fail_second_link):
                with self.assertRaisesRegex(OSError, "sidecar link failure"):
                    save_checkpoint(path, model, optimizer, self.metadata(), epoch=1)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name(path.name + ".sha256").exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_load_deserializes_the_same_bytes_that_passed_hash_verification(self) -> None:
        # Catches reopening a path that can be substituted after its bytes were verified.
        inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        original_model = ClockOnlyModel()
        original_optimizer = torch.optim.AdamW(original_model.parameters(), lr=1e-3)
        original_model.eval()
        expected_logits = original_model(inputs).detach().clone()

        substitute_model = ClockOnlyModel()
        with torch.no_grad():
            for parameter in substitute_model.parameters():
                parameter.fill_(5.0)
        substitute_optimizer = torch.optim.AdamW(substitute_model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as directory:
            original_path = Path(directory) / "original.pt"
            substitute_path = Path(directory) / "substitute.pt"
            save_checkpoint(
                original_path, original_model, original_optimizer, self.metadata(), epoch=1
            )
            save_checkpoint(
                substitute_path,
                substitute_model,
                substitute_optimizer,
                self.metadata(),
                epoch=1,
            )
            substitute_bytes = substitute_path.read_bytes()
            real_torch_load = torch.load

            def swap_path_then_load(source: object, *args: object, **kwargs: object) -> object:
                original_path.write_bytes(substitute_bytes)
                return real_torch_load(source, *args, **kwargs)

            restored = ClockOnlyModel()
            with patch("bc_core.checkpoints.torch.load", side_effect=swap_path_then_load):
                load_checkpoint(original_path, restored)
            restored.eval()
            torch.testing.assert_close(restored(inputs), expected_logits, rtol=0, atol=0)

    def test_architecture_metadata_helper_returns_complete_model_specific_schemas(self) -> None:
        # Catches provenance that omits a fixed layer, pooling, activation, or dropout value.
        helper = getattr(model_module, "architecture_metadata", None)
        self.assertIsNotNone(helper)
        self.assertEqual(helper(ClockOnlyModel()), self.clock_architecture())
        self.assertEqual(helper(StateAwareModel()), self.state_architecture())

    def test_save_rejects_omitted_architecture_parameter(self) -> None:
        # Catches accepting incomplete provenance that cannot reconstruct the fixed model.
        architecture = self.clock_architecture()
        del architecture["activation"]
        model = ClockOnlyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "architecture"
        ):
            save_checkpoint(
                Path(directory) / "epoch-001.pt",
                model,
                optimizer,
                self.metadata(architecture),
                epoch=1,
            )

    def test_save_rejects_architecture_for_a_different_model(self) -> None:
        # Catches labeling a state-aware checkpoint with clock-only provenance.
        model = StateAwareModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "architecture"
        ):
            save_checkpoint(
                Path(directory) / "epoch-001.pt",
                model,
                optimizer,
                self.metadata(self.clock_architecture()),
                epoch=1,
            )

    def test_save_rejects_conflicting_architecture_parameter(self) -> None:
        # Catches accepting provenance whose hidden widths conflict with the saved weights.
        architecture = self.clock_architecture()
        architecture["hidden_dims"] = [32, 64]
        model = ClockOnlyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "architecture"
        ):
            save_checkpoint(
                Path(directory) / "epoch-001.pt",
                model,
                optimizer,
                self.metadata(architecture),
                epoch=1,
            )

    def test_load_rejects_checkpoint_architecture_for_supplied_model(self) -> None:
        # Catches deferring a wrong-model checkpoint to partial state-dict mutation.
        source = ClockOnlyModel()
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch-001.pt"
            save_checkpoint(
                path,
                source,
                source_optimizer,
                self.metadata(self.clock_architecture()),
                epoch=1,
            )
            target = StateAwareModel()
            before = self.clone_model_state(target)
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_checkpoint(path, target)
            self.assert_model_state_equal(target, before)

    def test_save_rejects_same_class_model_graph_mutations(self) -> None:
        # Catches trusting model type while activation, widths, dropout, or pool differ.
        cases = (
            (
                "clock-activation",
                ClockOnlyModel,
                lambda model: model.network.__setitem__(1, torch.nn.Sigmoid()),
            ),
            (
                "clock-linear-width",
                ClockOnlyModel,
                lambda model: model.network.__setitem__(2, torch.nn.Linear(64, 63)),
            ),
            (
                "state-dropout",
                StateAwareModel,
                lambda model: model.classifier.__setitem__(2, torch.nn.Dropout(0.2)),
            ),
            (
                "state-pool",
                StateAwareModel,
                lambda model: model.tile.__setitem__(
                    4, torch.nn.AdaptiveMaxPool2d((2, 2))
                ),
            ),
            (
                "state-convolution",
                StateAwareModel,
                lambda model: model.tile.__setitem__(
                    2, torch.nn.Conv2d(32, 64, 5, padding=2)
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, factory, mutate in cases:
                with self.subTest(name=name):
                    model = factory()
                    metadata = self.metadata(model_module.architecture_metadata(model))
                    mutate(model)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    path = Path(directory) / f"{name}.pt"
                    with self.assertRaisesRegex(ValueError, "architecture"):
                        save_checkpoint(path, model, optimizer, metadata, epoch=1)
                    self.assertFalse(path.exists())
                    self.assertFalse(path.with_name(path.name + ".sha256").exists())

    def test_load_rejects_same_class_mutated_graph_before_weight_mutation(self) -> None:
        # Catches same-class dropout changes bypassing preflight before state loading.
        source = StateAwareModel()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(4.0)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        metadata = self.metadata(model_module.architecture_metadata(source))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_checkpoint(path, source, source_optimizer, metadata, epoch=1)
            target = StateAwareModel()
            target.classifier[2] = torch.nn.Dropout(0.2)
            before = self.clone_model_state(target)

            with self.assertRaisesRegex(ValueError, "architecture"):
                load_checkpoint(path, target)

            self.assert_model_state_equal(target, before)

    def test_save_rejects_runtime_hook_and_nested_leaf_registration(self) -> None:
        # Catches behavior extensions hidden from parameterized layer signatures.
        cases = (
            (
                "forward-hook",
                lambda model: model.network[4].register_forward_hook(
                    lambda module, args, output: output + 1
                ),
            ),
            (
                "nested-child",
                lambda model: model.network[0].add_module("extra", torch.nn.Identity()),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, mutate in cases:
                with self.subTest(name=name):
                    model = ClockOnlyModel()
                    metadata = self.metadata(model_module.architecture_metadata(model))
                    inputs = torch.ones(2, 8)
                    output_before_hook = model(inputs).detach().clone()
                    mutate(model)
                    if name == "forward-hook":
                        torch.testing.assert_close(
                            model(inputs), output_before_hook + 1, rtol=0, atol=0
                        )
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    path = Path(directory) / f"{name}.pt"

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        save_checkpoint(path, model, optimizer, metadata, epoch=1)

                    self.assertFalse(path.exists())
                    self.assertFalse(path.with_name(path.name + ".sha256").exists())

    def test_load_rejects_runtime_hook_and_nested_leaf_before_weight_mutation(self) -> None:
        # Catches applying weights before validating runtime behavior registrations.
        source = ClockOnlyModel()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(8.0)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clock.pt"
            save_checkpoint(path, source, source_optimizer, self.metadata(), epoch=1)
            cases = (
                (
                    "forward-hook",
                    lambda model: model.network[4].register_forward_hook(
                        lambda module, args, output: output + 1
                    ),
                ),
                (
                    "nested-child",
                    lambda model: model.network[0].add_module(
                        "extra", torch.nn.Identity()
                    ),
                ),
            )
            for name, mutate in cases:
                with self.subTest(name=name):
                    target = ClockOnlyModel()
                    mutate(target)
                    before = self.clone_model_state(target)

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        load_checkpoint(path, target)

                    self.assert_model_state_equal(target, before)

    def test_save_rejects_direct_registrations_and_parametrizations(self) -> None:
        # Catches hidden checkpoint state and parameter transforms on fixed leaves.
        cases = (
            (
                "direct-parameter",
                lambda model: model.network[1].register_parameter(
                    "extra", torch.nn.Parameter(torch.ones(1))
                ),
            ),
            (
                "direct-buffer",
                lambda model: model.network[0].register_buffer(
                    "extra", torch.ones(1)
                ),
            ),
            (
                "parametrization",
                lambda model: torch.nn.utils.parametrizations.weight_norm(
                    model.network[0]
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, mutate in cases:
                with self.subTest(name=name):
                    model = ClockOnlyModel()
                    metadata = self.metadata(model_module.architecture_metadata(model))
                    mutate(model)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    path = Path(directory) / f"{name}.pt"

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        save_checkpoint(path, model, optimizer, metadata, epoch=1)

                    self.assertFalse(path.exists())
                    self.assertFalse(path.with_name(path.name + ".sha256").exists())

    def test_save_rejects_malformed_fixed_parameters_before_publication(self) -> None:
        # Catches trusting registration names while fixed layer tensors are unusable.
        cases = (
            (
                "clock-weight-shape",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "weight",
                    torch.nn.Parameter(torch.ones(63, 8)),
                ),
            ),
            (
                "clock-weight-none",
                ClockOnlyModel,
                lambda model: setattr(model.network[0], "weight", None),
            ),
            (
                "clock-bias-shape",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "bias",
                    torch.nn.Parameter(torch.ones(63)),
                ),
            ),
            (
                "clock-bias-none",
                ClockOnlyModel,
                lambda model: setattr(model.network[0], "bias", None),
            ),
            (
                "clock-non-parameter",
                ClockOnlyModel,
                lambda model: model.network[0]._parameters.__setitem__(
                    "weight", torch.ones(64, 8)
                ),
            ),
            (
                "clock-frozen-weight",
                ClockOnlyModel,
                lambda model: model.network[0].weight.requires_grad_(False),
            ),
            (
                "state-conv-weight-shape",
                StateAwareModel,
                lambda model: setattr(
                    model.tile[0],
                    "weight",
                    torch.nn.Parameter(torch.ones(31, 44, 3, 3)),
                ),
            ),
            (
                "state-linear-weight-shape",
                StateAwareModel,
                lambda model: setattr(
                    model.actor[0],
                    "weight",
                    torch.nn.Parameter(torch.ones(63, 38)),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, factory, mutate in cases:
                with self.subTest(name=name):
                    model = factory()
                    metadata = self.metadata(model_module.architecture_metadata(model))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    mutate(model)
                    path = Path(directory) / f"{name}.pt"

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        save_checkpoint(path, model, optimizer, metadata, epoch=1)

                    self.assertFalse(path.exists())
                    self.assertFalse(path.with_name(path.name + ".sha256").exists())

    def test_load_rejects_identically_malformed_fixed_parameters_before_mutation(
        self,
    ) -> None:
        # Catches malformed checkpoint and target tensors agreeing with each other.
        cases = (
            (
                "clock-weight-shape",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "weight",
                    torch.nn.Parameter(torch.zeros(63, 8)),
                ),
                "network.0.weight",
                torch.full((63, 8), 7.0),
            ),
            (
                "clock-weight-none",
                ClockOnlyModel,
                lambda model: setattr(model.network[0], "weight", None),
                "network.0.weight",
                None,
            ),
            (
                "clock-bias-shape",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "bias",
                    torch.nn.Parameter(torch.zeros(63)),
                ),
                "network.0.bias",
                torch.full((63,), 7.0),
            ),
            (
                "clock-bias-none",
                ClockOnlyModel,
                lambda model: setattr(model.network[0], "bias", None),
                "network.0.bias",
                None,
            ),
            (
                "clock-non-parameter",
                ClockOnlyModel,
                lambda model: model.network[0]._parameters.__setitem__(
                    "weight", torch.zeros(64, 8)
                ),
                None,
                None,
            ),
            (
                "clock-frozen-weight",
                ClockOnlyModel,
                lambda model: model.network[0].weight.requires_grad_(False),
                None,
                None,
            ),
            (
                "state-conv-weight-shape",
                StateAwareModel,
                lambda model: setattr(
                    model.tile[0],
                    "weight",
                    torch.nn.Parameter(torch.zeros(31, 44, 3, 3)),
                ),
                "tile.0.weight",
                torch.full((31, 44, 3, 3), 7.0),
            ),
            (
                "state-linear-weight-shape",
                StateAwareModel,
                lambda model: setattr(
                    model.actor[0],
                    "weight",
                    torch.nn.Parameter(torch.zeros(63, 38)),
                ),
                "actor.0.weight",
                torch.full((63, 38), 7.0),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, factory, mutate_target, state_name, state_value in cases:
                with self.subTest(name=name):
                    source = factory()
                    with torch.no_grad():
                        for parameter in source.parameters():
                            parameter.fill_(7.0)
                    source_optimizer = torch.optim.AdamW(
                        source.parameters(), lr=1e-3
                    )
                    path = Path(directory) / f"{name}.pt"
                    save_checkpoint(
                        path,
                        source,
                        source_optimizer,
                        self.metadata(model_module.architecture_metadata(source)),
                        epoch=1,
                    )
                    if state_name is not None:
                        self.replace_checkpoint_model_parameter(
                            path, state_name, state_value
                        )

                    target = factory()
                    mutate_target(target)
                    before = self.clone_model_state(target)

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        load_checkpoint(path, target)

                    self.assert_model_state_equal(target, before)

    def test_save_rejects_tied_parameter_objects_and_storage_before_publication(
        self,
    ) -> None:
        # Catches optimizer-visible parameter deduplication hidden by valid slot shapes.
        cases = (
            (
                "clock-object-alias",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[2], "bias", model.network[0].bias
                ),
            ),
            (
                "clock-storage-alias",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[2],
                    "bias",
                    torch.nn.Parameter(model.network[0].bias.detach()),
                ),
            ),
            (
                "clock-disjoint-storage-views",
                ClockOnlyModel,
                self.tie_clock_bias_storage_views,
            ),
            (
                "state-object-alias",
                StateAwareModel,
                lambda model: setattr(model.actor[2], "bias", model.actor[0].bias),
            ),
            (
                "state-storage-alias",
                StateAwareModel,
                lambda model: setattr(
                    model.actor[2],
                    "bias",
                    torch.nn.Parameter(model.actor[0].bias.detach()),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, factory, mutate in cases:
                with self.subTest(name=name):
                    model = factory()
                    metadata = self.metadata(model_module.architecture_metadata(model))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    mutate(model)
                    path = Path(directory) / f"{name}.pt"

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        save_checkpoint(path, model, optimizer, metadata, epoch=1)

                    self.assertFalse(path.exists())
                    self.assertFalse(path.with_name(path.name + ".sha256").exists())

    def test_load_rejects_tied_parameters_before_weight_mutation(self) -> None:
        # Catches accepting a tied target because checkpoint keys still appear complete.
        cases = (
            (
                "clock-object-alias",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[2], "bias", model.network[0].bias
                ),
            ),
            (
                "clock-storage-alias",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[2],
                    "bias",
                    torch.nn.Parameter(model.network[0].bias.detach()),
                ),
            ),
            (
                "clock-disjoint-storage-views",
                ClockOnlyModel,
                self.tie_clock_bias_storage_views,
            ),
            (
                "state-object-alias",
                StateAwareModel,
                lambda model: setattr(model.actor[2], "bias", model.actor[0].bias),
            ),
            (
                "state-storage-alias",
                StateAwareModel,
                lambda model: setattr(
                    model.actor[2],
                    "bias",
                    torch.nn.Parameter(model.actor[0].bias.detach()),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, factory, mutate in cases:
                with self.subTest(name=name):
                    source = factory()
                    with torch.no_grad():
                        for parameter in source.parameters():
                            parameter.fill_(7.0)
                    path = Path(directory) / f"{name}.pt"
                    save_checkpoint(
                        path,
                        source,
                        torch.optim.AdamW(source.parameters(), lr=1e-3),
                        self.metadata(model_module.architecture_metadata(source)),
                        epoch=1,
                    )
                    target = factory()
                    mutate(target)
                    before = self.clone_model_state(target)

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        load_checkpoint(path, target)

                    self.assert_model_state_equal(target, before)

    def test_save_rejects_unmaterialized_or_incompatible_parameters(self) -> None:
        # Catches correct shapes concealing meta, sparse, subclass, or dtype breaks.
        class ParameterSubclass(torch.nn.Parameter):
            pass

        cases = (
            (
                "clock-meta",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "weight",
                    torch.nn.Parameter(torch.empty(64, 8, device="meta")),
                ),
            ),
            (
                "state-meta",
                StateAwareModel,
                lambda model: setattr(
                    model.tile[0],
                    "weight",
                    torch.nn.Parameter(torch.empty(32, 44, 3, 3, device="meta")),
                ),
            ),
            (
                "clock-sparse",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "weight",
                    torch.nn.Parameter(torch.zeros(64, 8).to_sparse()),
                ),
            ),
            (
                "state-sparse",
                StateAwareModel,
                lambda model: setattr(
                    model.tile[0],
                    "weight",
                    torch.nn.Parameter(torch.zeros(32, 44, 3, 3).to_sparse()),
                ),
            ),
            (
                "clock-mixed-dtype",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "weight",
                    torch.nn.Parameter(torch.ones(64, 8, dtype=torch.float64)),
                ),
            ),
            (
                "state-complex",
                StateAwareModel,
                lambda model: setattr(
                    model.actor[0],
                    "weight",
                    torch.nn.Parameter(torch.ones(64, 38, dtype=torch.complex64)),
                ),
            ),
            (
                "clock-parameter-subclass",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "weight",
                    ParameterSubclass(torch.ones(64, 8)),
                ),
            ),
            (
                "state-uninitialized",
                StateAwareModel,
                lambda model: setattr(
                    model.actor[0],
                    "weight",
                    torch.nn.parameter.UninitializedParameter(),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, factory, mutate in cases:
                with self.subTest(name=name):
                    model = factory()
                    metadata = self.metadata(model_module.architecture_metadata(model))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    mutate(model)
                    path = Path(directory) / f"{name}.pt"

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        save_checkpoint(path, model, optimizer, metadata, epoch=1)

                    self.assertFalse(path.exists())
                    self.assertFalse(path.with_name(path.name + ".sha256").exists())

    def test_load_rejects_meta_sparse_and_incompatible_targets_before_mutation(
        self,
    ) -> None:
        # Catches meta no-op loads and matching non-strided/incompatible state.
        cases = (
            (
                "clock-meta",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "weight",
                    torch.nn.Parameter(torch.empty(64, 8, device="meta")),
                ),
                "network.0.weight",
                torch.empty(64, 8, device="meta"),
                "network.2.weight",
            ),
            (
                "state-meta",
                StateAwareModel,
                lambda model: setattr(
                    model.tile[0],
                    "weight",
                    torch.nn.Parameter(torch.empty(32, 44, 3, 3, device="meta")),
                ),
                "tile.0.weight",
                torch.empty(32, 44, 3, 3, device="meta"),
                "actor.0.weight",
            ),
            (
                "clock-sparse",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "weight",
                    torch.nn.Parameter(torch.zeros(64, 8).to_sparse()),
                ),
                "network.0.weight",
                torch.zeros(64, 8).to_sparse(),
                "network.2.weight",
            ),
            (
                "state-sparse",
                StateAwareModel,
                lambda model: setattr(
                    model.tile[0],
                    "weight",
                    torch.nn.Parameter(torch.zeros(32, 44, 3, 3).to_sparse()),
                ),
                "tile.0.weight",
                torch.zeros(32, 44, 3, 3).to_sparse(),
                "actor.0.weight",
            ),
            (
                "clock-mixed-dtype",
                ClockOnlyModel,
                lambda model: setattr(
                    model.network[0],
                    "weight",
                    torch.nn.Parameter(torch.zeros(64, 8, dtype=torch.float64)),
                ),
                "network.0.weight",
                torch.full((64, 8), 7.0, dtype=torch.float64),
                "network.2.weight",
            ),
            (
                "state-complex",
                StateAwareModel,
                lambda model: setattr(
                    model.actor[0],
                    "weight",
                    torch.nn.Parameter(torch.zeros(64, 38, dtype=torch.complex64)),
                ),
                "actor.0.weight",
                torch.full((64, 38), 7.0, dtype=torch.complex64),
                "classifier.0.weight",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for (
                name,
                factory,
                mutate,
                state_name,
                state_value,
                sentinel_name,
            ) in cases:
                with self.subTest(name=name):
                    source = factory()
                    with torch.no_grad():
                        for parameter in source.parameters():
                            parameter.fill_(7.0)
                    path = Path(directory) / f"{name}.pt"
                    save_checkpoint(
                        path,
                        source,
                        torch.optim.AdamW(source.parameters(), lr=1e-3),
                        self.metadata(model_module.architecture_metadata(source)),
                        epoch=1,
                    )
                    self.replace_checkpoint_model_parameter(
                        path, state_name, state_value
                    )
                    target = factory()
                    mutate(target)
                    sentinel_before = target.state_dict()[sentinel_name].clone()

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        load_checkpoint(path, target)

                    torch.testing.assert_close(
                        target.state_dict()[sentinel_name],
                        sentinel_before,
                        rtol=0,
                        atol=0,
                    )

    def test_save_rejects_renamed_sequential_children_before_publication(self) -> None:
        # Catches topology signatures that compare layers but omit registration names.
        cases = (
            (
                "clock",
                ClockOnlyModel,
                lambda model: setattr(
                    model,
                    "network",
                    self.renamed_sequential(
                        model.network,
                        ("dense0", "relu0", "dense1", "relu1", "output"),
                    ),
                ),
            ),
            (
                "state",
                StateAwareModel,
                lambda model: setattr(
                    model,
                    "actor",
                    self.renamed_sequential(
                        model.actor, ("dense0", "relu0", "dense1", "relu1")
                    ),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, factory, mutate in cases:
                with self.subTest(name=name):
                    model = factory()
                    metadata = self.metadata(model_module.architecture_metadata(model))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    mutate(model)
                    path = Path(directory) / f"{name}.pt"

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        save_checkpoint(path, model, optimizer, metadata, epoch=1)

                    self.assertFalse(path.exists())
                    self.assertFalse(path.with_name(path.name + ".sha256").exists())

    def test_load_rejects_matching_renamed_sequential_keys_before_mutation(self) -> None:
        # Catches renamed targets when checkpoint keys were crafted to match the rename.
        cases = (
            (
                "clock",
                ClockOnlyModel,
                lambda model: setattr(
                    model,
                    "network",
                    self.renamed_sequential(
                        model.network,
                        ("dense0", "relu0", "dense1", "relu1", "output"),
                    ),
                ),
                {
                    "network.0.": "network.dense0.",
                    "network.2.": "network.dense1.",
                    "network.4.": "network.output.",
                },
            ),
            (
                "state",
                StateAwareModel,
                lambda model: setattr(
                    model,
                    "actor",
                    self.renamed_sequential(
                        model.actor, ("dense0", "relu0", "dense1", "relu1")
                    ),
                ),
                {
                    "actor.0.": "actor.dense0.",
                    "actor.2.": "actor.dense1.",
                },
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, factory, mutate, replacements in cases:
                with self.subTest(name=name):
                    source = factory()
                    with torch.no_grad():
                        for parameter in source.parameters():
                            parameter.fill_(7.0)
                    path = Path(directory) / f"{name}.pt"
                    save_checkpoint(
                        path,
                        source,
                        torch.optim.AdamW(source.parameters(), lr=1e-3),
                        self.metadata(model_module.architecture_metadata(source)),
                        epoch=1,
                    )
                    self.rename_checkpoint_model_parameters(path, replacements)
                    target = factory()
                    mutate(target)
                    before = self.clone_model_state(target)

                    with self.assertRaisesRegex(ValueError, "architecture"):
                        load_checkpoint(path, target)

                    self.assert_model_state_equal(target, before)

    def test_save_rejects_parameter_gradient_hook(self) -> None:
        # Catches a tensor-level hook that changes training outside module hook registries.
        model = ClockOnlyModel()
        metadata = self.metadata(model_module.architecture_metadata(model))
        next(model.parameters()).register_hook(lambda gradient: torch.zeros_like(gradient))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameter-hook.pt"

            with self.assertRaisesRegex(ValueError, "architecture"):
                save_checkpoint(path, model, optimizer, metadata, epoch=1)

            self.assertFalse(path.exists())

    def test_save_rejects_global_module_hook(self) -> None:
        # Catches behavior-changing hooks whose registration lives outside instances.
        model = ClockOnlyModel()
        metadata = self.metadata(model_module.architecture_metadata(model))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def replace_relu_output(
            module: torch.nn.Module, args: tuple[object, ...], output: object
        ) -> object | None:
            if isinstance(module, torch.nn.ReLU):
                return torch.zeros_like(output)
            return None

        handle = torch.nn.modules.module.register_module_forward_hook(
            replace_relu_output
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "global-hook.pt"
                with self.assertRaisesRegex(ValueError, "architecture"):
                    save_checkpoint(path, model, optimizer, metadata, epoch=1)
                self.assertFalse(path.exists())
        finally:
            handle.remove()

    def test_evaluation_load_does_not_mutate_rng_state(self) -> None:
        # Catches evaluation changing subsequent Python, NumPy, or Torch randomness.
        model = ClockOnlyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch-001.pt"
            save_checkpoint(path, model, optimizer, self.metadata(), epoch=1)
            restored = ClockOnlyModel()
            random.seed(21)
            np.random.seed(22)
            torch.manual_seed(23)
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            before_torch = torch.get_rng_state().clone()

            load_checkpoint(path, restored)

            self.assertEqual(random.getstate(), before_python)
            self.assert_numpy_rng_equal(np.random.get_state(), before_numpy)
            torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)

    def test_resume_load_restores_optimizer_epoch_and_rng_state(self) -> None:
        # Catches a resume that restarts optimizer moments or any random stream.
        random.seed(31)
        np.random.seed(32)
        torch.manual_seed(33)
        model = ClockOnlyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.ones(2, 8)).sum()
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch-007.pt"
            save_checkpoint(path, model, optimizer, self.metadata(), epoch=7)
            expected_python = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = torch.rand(3)
            expected_cuda = (
                torch.rand(3, device="cuda") if torch.cuda.is_available() else None
            )

            random.seed(41)
            np.random.seed(42)
            torch.manual_seed(43)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(44)
            restored = ClockOnlyModel()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
            payload = load_checkpoint(path, restored, restored_optimizer)

            self.assertEqual(payload["epoch"], 7)
            self.assertAlmostEqual(restored_optimizer.param_groups[0]["lr"], 1e-3)
            self.assertTrue(restored_optimizer.state)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.random()), expected_numpy)
            torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
            if expected_cuda is not None:
                torch.testing.assert_close(
                    torch.rand(3, device="cuda"), expected_cuda, rtol=0, atol=0
                )

    def test_malformed_rng_states_are_rejected_before_evaluation_model_mutation(self) -> None:
        # Catches shape-only RNG checks and model loading before payload validation finishes.
        malformed_states = {
            "python": ("invalid-python-state",),
            "numpy": ("invalid-numpy-state",),
            "torch": torch.tensor([1], dtype=torch.int64),
            "cuda": [torch.tensor([1], dtype=torch.uint8)],
        }
        source = ClockOnlyModel()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(2.0)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as directory:
            for name, malformed in malformed_states.items():
                with self.subTest(name=name):
                    path = Path(directory) / f"bad-{name}.pt"
                    save_checkpoint(
                        path, source, source_optimizer, self.metadata(), epoch=1
                    )
                    self.replace_checkpoint_rng_state(path, name, malformed)
                    target = ClockOnlyModel()
                    before_model = self.clone_model_state(target)
                    random.seed(51)
                    np.random.seed(52)
                    torch.manual_seed(53)
                    before_python = random.getstate()
                    before_numpy = np.random.get_state()
                    before_torch = torch.get_rng_state().clone()
                    before_cuda = (
                        [value.clone() for value in torch.cuda.get_rng_state_all()]
                        if torch.cuda.is_available()
                        else None
                    )

                    with self.assertRaisesRegex(ValueError, "RNG"):
                        load_checkpoint(path, target)

                    self.assert_model_state_equal(target, before_model)
                    self.assertEqual(random.getstate(), before_python)
                    self.assert_numpy_rng_equal(np.random.get_state(), before_numpy)
                    torch.testing.assert_close(
                        torch.get_rng_state(), before_torch, rtol=0, atol=0
                    )
                    if before_cuda is not None:
                        self.assert_cuda_rng_equal(
                            torch.cuda.get_rng_state_all(), before_cuda
                        )

    def test_malformed_rng_state_is_rejected_before_resume_mutation(self) -> None:
        # Catches resume loading weights/optimizer moments before validating RNG payloads.
        source = ClockOnlyModel()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(3.0)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        source_loss = source(torch.ones(2, 8)).sum()
        source_loss.backward()
        source_optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-resume.pt"
            save_checkpoint(path, source, source_optimizer, self.metadata(), epoch=1)
            self.replace_checkpoint_rng_state(path, "python", ("invalid",))

            target = ClockOnlyModel()
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.5)
            target_loss = target(torch.ones(2, 8)).square().mean()
            target_loss.backward()
            target_optimizer.step()
            before_model = self.clone_model_state(target)
            before_lr = target_optimizer.param_groups[0]["lr"]
            before_steps = [
                state["step"].detach().clone() for state in target_optimizer.state.values()
            ]
            random.seed(61)
            np.random.seed(62)
            torch.manual_seed(63)
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            before_torch = torch.get_rng_state().clone()

            with self.assertRaisesRegex(ValueError, "RNG"):
                load_checkpoint(path, target, target_optimizer)

            self.assert_model_state_equal(target, before_model)
            self.assertEqual(target_optimizer.param_groups[0]["lr"], before_lr)
            after_steps = [state["step"] for state in target_optimizer.state.values()]
            self.assertEqual(len(after_steps), len(before_steps))
            for actual, expected in zip(after_steps, before_steps):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(random.getstate(), before_python)
            self.assert_numpy_rng_equal(np.random.get_state(), before_numpy)
            torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)

    def test_cpu_resume_rejects_cuda_rng_before_any_caller_mutation(self) -> None:
        # Catches restoring model/optimizer/CPU RNG before discovering CUDA is unavailable.
        source = ClockOnlyModel()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(6.0)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        source_loss = source(torch.ones(2, 8)).sum()
        source_loss.backward()
        source_optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda-resume.pt"
            save_checkpoint(path, source, source_optimizer, self.metadata(), epoch=1)
            self.replace_checkpoint_rng_state(
                path, "cuda", [torch.zeros(16, dtype=torch.uint8)]
            )

            target = ClockOnlyModel()
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.5)
            target_loss = target(torch.ones(2, 8)).square().mean()
            target_loss.backward()
            target_optimizer.step()
            before_model = self.clone_model_state(target)
            before_lr = target_optimizer.param_groups[0]["lr"]
            before_steps = [
                state["step"].detach().clone() for state in target_optimizer.state.values()
            ]
            random.seed(71)
            np.random.seed(72)
            torch.manual_seed(73)
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            before_torch = torch.get_rng_state().clone()

            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, "CUDA RNG"):
                    load_checkpoint(path, target, target_optimizer)

            self.assert_model_state_equal(target, before_model)
            self.assertEqual(target_optimizer.param_groups[0]["lr"], before_lr)
            after_steps = [state["step"] for state in target_optimizer.state.values()]
            self.assertEqual(len(after_steps), len(before_steps))
            for actual, expected in zip(after_steps, before_steps):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(random.getstate(), before_python)
            self.assert_numpy_rng_equal(np.random.get_state(), before_numpy)
            torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)

    def test_cpu_evaluation_rejects_unvalidated_cuda_rng_before_mutation(self) -> None:
        # Catches treating structurally plausible CUDA bytes as valid during evaluation.
        source = ClockOnlyModel()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(9.0)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda-evaluation.pt"
            save_checkpoint(path, source, source_optimizer, self.metadata(), epoch=1)
            self.replace_checkpoint_rng_state(
                path, "cuda", [torch.zeros(16, dtype=torch.uint8)]
            )
            target = ClockOnlyModel()
            before_model = self.clone_model_state(target)
            random.seed(91)
            np.random.seed(92)
            torch.manual_seed(93)
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            before_torch = torch.get_rng_state().clone()

            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, "CUDA RNG"):
                    load_checkpoint(path, target)

            self.assert_model_state_equal(target, before_model)
            self.assertEqual(random.getstate(), before_python)
            self.assert_numpy_rng_equal(np.random.get_state(), before_numpy)
            torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)

    def test_unexpected_resume_restore_failure_rolls_back_caller_state(self) -> None:
        # Catches an unforeseen restore error leaving model or optimizer partly replaced.
        source = ClockOnlyModel()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(7.0)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        source_loss = source(torch.ones(2, 8)).sum()
        source_loss.backward()
        source_optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restore-failure.pt"
            save_checkpoint(path, source, source_optimizer, self.metadata(), epoch=1)
            target = ClockOnlyModel()
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.5)
            target_loss = target(torch.ones(2, 8)).square().mean()
            target_loss.backward()
            target_optimizer.step()
            before_model = self.clone_model_state(target)
            before_lr = target_optimizer.param_groups[0]["lr"]
            before_steps = [
                state["step"].detach().clone() for state in target_optimizer.state.values()
            ]
            random.seed(81)
            np.random.seed(82)
            torch.manual_seed(83)
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            before_torch = torch.get_rng_state().clone()
            real_setstate = random.setstate
            calls = 0

            def fail_first_global_python_restore(state: object) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("injected RNG restore failure")
                real_setstate(state)

            with patch(
                "bc_core.checkpoints.random.setstate",
                side_effect=fail_first_global_python_restore,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected RNG restore"):
                    load_checkpoint(path, target, target_optimizer)

            self.assert_model_state_equal(target, before_model)
            self.assertEqual(target_optimizer.param_groups[0]["lr"], before_lr)
            after_steps = [state["step"] for state in target_optimizer.state.values()]
            self.assertEqual(len(after_steps), len(before_steps))
            for actual, expected in zip(after_steps, before_steps):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(random.getstate(), before_python)
            self.assert_numpy_rng_equal(np.random.get_state(), before_numpy)
            torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)

    @staticmethod
    def metadata(architecture: dict[str, object] | None = None) -> dict[str, object]:
        stats = NormalizationStats(
            global_mean=np.zeros(62, dtype=np.float32),
            global_std=np.ones(62, dtype=np.float32),
            actor_mean=np.zeros(38, dtype=np.float32),
            actor_std=np.ones(38, dtype=np.float32),
        )
        return {
            "schema_version": "ryo-bc-v0",
            "feature_schema_version": "ryo-features-v0",
            "vocabularies": {"operations": list(OPERATIONS)},
            "normalization": stats,
            "class_weights": np.ones(17, dtype=np.float32),
            "manifest_sha256": "b" * 64,
            "architecture": architecture
            if architecture is not None
            else ModelTest.clock_architecture(),
        }

    @staticmethod
    def clock_architecture() -> dict[str, object]:
        return {
            "model_kind": "clock_only_v0",
            "input_dim": 8,
            "hidden_dims": [64, 64],
            "classes": 17,
            "activation": "relu",
            "linear_bias": True,
        }

    @staticmethod
    def state_architecture() -> dict[str, object]:
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

    @staticmethod
    def clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in model.state_dict().items()}

    def assert_model_state_equal(
        self, model: torch.nn.Module, expected: dict[str, torch.Tensor]
    ) -> None:
        self.assertEqual(set(model.state_dict()), set(expected))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def assert_linear_stack(
        self, stack: torch.nn.Sequential, dimensions: list[tuple[int, int]]
    ) -> None:
        self.assertEqual(
            [type(layer) for layer in stack],
            [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear, torch.nn.ReLU],
        )
        self.assertEqual(
            [
                (stack[index].in_features, stack[index].out_features)
                for index in (0, 2)
            ],
            dimensions,
        )
        self.assertIsNotNone(stack[0].bias)
        self.assertIsNotNone(stack[2].bias)
        self.assertFalse(stack[1].inplace)
        self.assertFalse(stack[3].inplace)

    @staticmethod
    def replace_checkpoint_rng_state(path: Path, name: str, value: object) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["rng_state"][name] = value
        torch.save(payload, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(path.name + ".sha256").write_text(f"{digest}\n", encoding="ascii")

    @staticmethod
    def replace_checkpoint_model_parameter(
        path: Path, name: str, value: torch.Tensor | None
    ) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if value is None:
            del payload["model_state_dict"][name]
        else:
            payload["model_state_dict"][name] = value
        torch.save(payload, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(path.name + ".sha256").write_text(f"{digest}\n", encoding="ascii")

    @staticmethod
    def rename_checkpoint_model_parameters(
        path: Path, replacements: dict[str, str]
    ) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        renamed = {}
        for name, value in payload["model_state_dict"].items():
            for source, destination in replacements.items():
                if name.startswith(source):
                    name = destination + name[len(source) :]
                    break
            renamed[name] = value
        payload["model_state_dict"] = renamed
        torch.save(payload, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(path.name + ".sha256").write_text(f"{digest}\n", encoding="ascii")

    @staticmethod
    def renamed_sequential(
        sequential: torch.nn.Sequential, names: tuple[str, ...]
    ) -> torch.nn.Sequential:
        return torch.nn.Sequential(OrderedDict(zip(names, list(sequential))))

    @staticmethod
    def tie_clock_bias_storage_views(model: ClockOnlyModel) -> None:
        storage = torch.zeros(128)
        model.network[0].bias = torch.nn.Parameter(storage[:64])
        model.network[2].bias = torch.nn.Parameter(storage[64:])

    @staticmethod
    def assert_numpy_rng_equal(
        actual: tuple[object, ...], expected: tuple[object, ...]
    ) -> None:
        if actual[0] != expected[0]:
            raise AssertionError(f"NumPy RNG kind changed: {actual[0]} != {expected[0]}")
        np.testing.assert_array_equal(actual[1], expected[1])
        if actual[2:] != expected[2:]:
            raise AssertionError("NumPy RNG scalar state changed")

    @staticmethod
    def assert_cuda_rng_equal(
        actual: list[torch.Tensor], expected: list[torch.Tensor]
    ) -> None:
        if len(actual) != len(expected):
            raise AssertionError(
                f"CUDA RNG device count changed: {len(actual)} != {len(expected)}"
            )
        for actual_state, expected_state in zip(actual, expected):
            torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
