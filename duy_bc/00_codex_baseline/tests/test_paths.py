"""Location-independent configuration and output path contracts."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bc_core.constants import load_config
from bc_core.paths import baseline_path, baseline_root, corpus_path, repository_root


class PathContractTest(unittest.TestCase):
    def _prepare_cli(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
        spec = importlib.util.spec_from_file_location("handoff_prepare_cli", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_baseline_outputs_do_not_depend_on_the_callers_directory(self) -> None:
        """Catch generated outputs accidentally rooted at the process CWD."""
        expected = Path(__file__).resolve().parents[1]
        self.assertEqual(baseline_root(), expected)
        self.assertEqual(baseline_path("data"), expected / "data")
        self.assertEqual(baseline_path("runs"), expected / "runs")

    def test_relative_corpus_uses_repository_root_and_absolute_path_is_preserved(
        self,
    ) -> None:
        """Catch resolving Tu's data against the numbered baseline directory."""
        expected_repository = Path(__file__).resolve().parents[3]
        self.assertEqual(repository_root(), expected_repository)
        self.assertEqual(
            corpus_path("duy_explore/ryo_hasegawa_100_stratified"),
            (expected_repository / "duy_explore/ryo_hasegawa_100_stratified").resolve(),
        )
        with tempfile.TemporaryDirectory() as directory:
            absolute = Path(directory) / "tu-replays"
            self.assertEqual(corpus_path(absolute), absolute.resolve())

    def test_relative_corpus_follows_the_repository_containing_the_config(self) -> None:
        """Catch test or copied configs silently using Duy's checkout root."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            config = root / "duy_bc" / "00_codex_baseline" / "configs" / "v0.json"
            config.parent.mkdir(parents=True)
            expected = root / "replays" / "100-wins"
            self.assertEqual(
                corpus_path("replays/100-wins", config_path=config), expected.resolve()
            )

    def test_only_corpus_location_is_editable_in_v0_config(self) -> None:
        """Catch path portability weakening the fixed training contract."""
        source = Path(__file__).resolve().parents[1] / "configs" / "v0.json"
        with tempfile.TemporaryDirectory() as directory:
            edited = Path(directory) / "v0.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["corpus_root"] = str(Path(directory) / "tu-replays")
            edited.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                load_config(edited)["corpus_root"], payload["corpus_root"]
            )

            payload["training"]["batch_size"] = 7
            edited.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract"):
                load_config(edited)

    def test_prepare_cli_resolves_relative_inputs_from_the_baseline(self) -> None:
        """Catch nested handoff CLIs accidentally rooting paths at `duy_bc`."""
        cli = self._prepare_cli()
        captured: list[tuple[Path, Path]] = []

        def fake_prepare(config: Path, output: Path, **_kwargs):
            captured.append((config, output))
            return {
                "splits": {
                    split: {"games": 1, "samples": 2}
                    for split in ("train", "val", "test")
                }
            }

        with (
            patch.object(cli, "prepare", side_effect=fake_prepare),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = cli.main(
                ["--config", "configs/v0.json", "--output-root", "data"]
            )

        root = Path(__file__).resolve().parents[1]
        self.assertEqual(status, 0)
        self.assertEqual(captured, [(root / "configs/v0.json", root / "data")])

    def test_prepare_cli_defaults_to_the_handoff_config(self) -> None:
        """Catch a quick-start command that still requires a config argument."""
        cli = self._prepare_cli()
        captured: list[Path] = []

        def fake_prepare(config: Path, _output: Path, **_kwargs):
            captured.append(config)
            return {
                "splits": {
                    split: {"games": 0, "samples": 0}
                    for split in ("train", "val", "test")
                }
            }

        with (
            patch.object(cli, "prepare", side_effect=fake_prepare),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = cli.main([])

        self.assertEqual(status, 0)
        self.assertEqual(
            captured, [Path(__file__).resolve().parents[1] / "configs/v0.json"]
        )


if __name__ == "__main__":
    unittest.main()
