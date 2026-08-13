import csv
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from benchmarks import benchmark


class ResolutionAndScheduleTests(unittest.TestCase):
    def test_resolves_file_agent_with_hash_and_fresh_callable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent_one.py"
            path.write_text(
                "def agent(obs):\n"
                "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
            )

            ref = benchmark.resolve_agent(str(path))

            self.assertEqual(ref.label, "agent_one")
            self.assertEqual(ref.resolved_path, path.resolve())
            self.assertEqual(len(ref.sha256), 64)
            self.assertIsNot(ref.runner_value(), ref.runner_value())

    def test_accepts_supported_builtin(self):
        ref = benchmark.resolve_agent("starter")

        self.assertEqual(ref.runner_value(), "starter")
        self.assertIsNone(ref.resolved_path)

    def test_rejects_missing_file(self):
        with self.assertRaisesRegex(benchmark.BenchmarkError, "not found"):
            benchmark.resolve_agent("missing-agent.py")

    def test_builds_contiguous_seed_major_seat_pairs(self):
        self.assertEqual(
            benchmark.build_schedule(7, 2),
            [
                benchmark.MatchSpec(seed=7, agent_a_seat=0),
                benchmark.MatchSpec(seed=7, agent_a_seat=1),
                benchmark.MatchSpec(seed=8, agent_a_seat=0),
                benchmark.MatchSpec(seed=8, agent_a_seat=1),
            ],
        )

    def test_rejects_invalid_seed_ranges(self):
        with self.assertRaisesRegex(benchmark.BenchmarkError, "non-negative"):
            benchmark.build_schedule(-1, 50)
        with self.assertRaisesRegex(benchmark.BenchmarkError, "positive"):
            benchmark.build_schedule(0, 0)


class MatchExecutionTests(unittest.TestCase):
    @staticmethod
    def fake_make(name, configuration, debug):
        class FakeEnvironment:
            steps = []

            def run(self, runners):
                farms = [{"money": 4100.0}, {"money": 3600.0}]
                self.steps = [
                    [
                        SimpleNamespace(
                            status="DONE",
                            reward=4100.0,
                            observation={"farms": farms},
                        ),
                        SimpleNamespace(
                            status="DONE",
                            reward=3600.0,
                            observation={"farms": farms},
                        ),
                    ]
                ]

        return FakeEnvironment()

    def test_normalizes_agent_a_from_seat_zero(self):
        result = benchmark.run_match(
            benchmark.resolve_agent("pass"),
            benchmark.resolve_agent("starter"),
            benchmark.MatchSpec(12, 0),
            make_environment=self.fake_make,
        )

        self.assertEqual(result["agent_a_money"], 4100.0)
        self.assertEqual(result["agent_b_money"], 3600.0)
        self.assertEqual(result["outcome"], "win")
        self.assertEqual(result["margin"], 500.0)

    def test_normalizes_agent_a_from_seat_one(self):
        result = benchmark.run_match(
            benchmark.resolve_agent("pass"),
            benchmark.resolve_agent("starter"),
            benchmark.MatchSpec(12, 1),
            make_environment=self.fake_make,
        )

        self.assertEqual(result["agent_a_money"], 3600.0)
        self.assertEqual(result["agent_b_money"], 4100.0)
        self.assertEqual(result["outcome"], "loss")
        self.assertEqual(result["margin"], -500.0)

    def test_rejects_non_done_status(self):
        def failed_make(name, configuration, debug):
            environment = self.fake_make(name, configuration, debug)
            environment.run([])
            environment.steps[-1][1].status = "ERROR"
            environment.run = lambda runners: None
            return environment

        with self.assertRaisesRegex(benchmark.BenchmarkError, "statuses"):
            benchmark.run_match(
                benchmark.resolve_agent("pass"),
                benchmark.resolve_agent("starter"),
                benchmark.MatchSpec(2, 0),
                make_environment=failed_make,
            )

    def test_rejects_reward_money_mismatch(self):
        def mismatch_make(name, configuration, debug):
            environment = self.fake_make(name, configuration, debug)
            environment.run([])
            environment.steps[-1][0].reward = 1.0
            environment.run = lambda runners: None
            return environment

        with self.assertRaisesRegex(benchmark.BenchmarkError, "reward"):
            benchmark.run_match(
                benchmark.resolve_agent("pass"),
                benchmark.resolve_agent("starter"),
                benchmark.MatchSpec(2, 0),
                make_environment=mismatch_make,
            )

    def test_runs_schedule_in_order_and_reports_progress(self):
        progress = []

        results = benchmark.run_suite(
            benchmark.resolve_agent("pass"),
            benchmark.resolve_agent("starter"),
            benchmark.build_schedule(3, 1),
            make_environment=self.fake_make,
            progress=lambda index, total, result: progress.append(
                (index, total, result["seed"], result["agent_a_seat"])
            ),
        )

        self.assertEqual(
            [(row["seed"], row["agent_a_seat"]) for row in results],
            [(3, 0), (3, 1)],
        )
        self.assertEqual(progress, [(1, 2, 3, 0), (2, 2, 3, 1)])


class SummaryTests(unittest.TestCase):
    def test_summarizes_overall_and_each_agent_a_seat(self):
        results = [
            {
                "agent_a_seat": 0,
                "agent_a_money": 5000.0,
                "agent_b_money": 4000.0,
                "margin": 1000.0,
                "outcome": "win",
            },
            {
                "agent_a_seat": 1,
                "agent_a_money": 3000.0,
                "agent_b_money": 3500.0,
                "margin": -500.0,
                "outcome": "loss",
            },
            {
                "agent_a_seat": 0,
                "agent_a_money": 4200.0,
                "agent_b_money": 4200.0,
                "margin": 0.0,
                "outcome": "tie",
            },
        ]

        summary = benchmark.summarize(results)

        self.assertEqual(summary["games"], 3)
        self.assertEqual(
            (summary["wins"], summary["losses"], summary["ties"]),
            (1, 1, 1),
        )
        self.assertAlmostEqual(summary["win_rate"], 1 / 3)
        self.assertEqual(summary["agent_a_money"]["median"], 4200.0)
        self.assertAlmostEqual(summary["margin"]["mean"], 500.0 / 3)
        self.assertEqual(summary["by_agent_a_seat"]["0"]["games"], 2)
        self.assertEqual(summary["by_agent_a_seat"]["1"]["losses"], 1)


class ArtifactTests(unittest.TestCase):
    def test_writes_csv_json_and_text_with_protocol_metadata(self):
        results = [
            {
                "seed": 4,
                "agent_a_seat": 0,
                "agent_b_seat": 1,
                "seat_0_agent": "pass",
                "seat_1_agent": "starter",
                "seat_0_money": 4000.0,
                "seat_1_money": 3500.0,
                "seat_0_reward": 4000.0,
                "seat_1_reward": 3500.0,
                "seat_0_status": "DONE",
                "seat_1_status": "DONE",
                "agent_a_money": 4000.0,
                "agent_b_money": 3500.0,
                "margin": 500.0,
                "outcome": "win",
            },
            {
                "seed": 4,
                "agent_a_seat": 1,
                "agent_b_seat": 0,
                "seat_0_agent": "starter",
                "seat_1_agent": "pass",
                "seat_0_money": 3500.0,
                "seat_1_money": 4000.0,
                "seat_0_reward": 3500.0,
                "seat_1_reward": 4000.0,
                "seat_0_status": "DONE",
                "seat_1_status": "DONE",
                "agent_a_money": 4000.0,
                "agent_b_money": 3500.0,
                "margin": 500.0,
                "outcome": "win",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            metadata = benchmark.build_metadata(
                benchmark.resolve_agent("pass"),
                benchmark.resolve_agent("starter"),
                seed_start=4,
                seed_count=1,
                steps=720,
            )

            benchmark.write_artifacts(
                output, metadata, results, benchmark.summarize(results)
            )

            payload = json.loads((output / "summary.json").read_text())
            with (output / "games.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertNotIn(b"\r\n", (output / "games.csv").read_bytes())
            self.assertEqual(payload["metadata"]["seeds"], [4])
            self.assertEqual(payload["summary"]["games"], 2)
            self.assertEqual(len(rows), 2)
            self.assertIn(
                "Agent A in seat 0", (output / "summary.txt").read_text()
            )

    def test_cli_rejects_selecting_same_agent_twice(self):
        errors = StringIO()

        with redirect_stderr(errors):
            exit_code = benchmark.main(["pass", "pass"])

        self.assertEqual(exit_code, 1)
        self.assertIn("must be different", errors.getvalue())

    def test_generates_only_the_requested_replay(self):
        class ReplayEnvironment:
            def run(self, runners):
                farms = [{"money": 4100.0}, {"money": 3600.0}]
                self.steps = [
                    [
                        SimpleNamespace(
                            status="DONE",
                            reward=4100.0,
                            observation={"farms": farms},
                        ),
                        SimpleNamespace(
                            status="DONE",
                            reward=3600.0,
                            observation={"farms": farms},
                        ),
                    ]
                ]

            def toJSON(self):
                return {"requested_seed": 19, "step_count": len(self.steps)}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed-19-seat-1.json"
            result = benchmark.generate_replay(
                benchmark.resolve_agent("pass"),
                benchmark.resolve_agent("starter"),
                seed=19,
                agent_a_seat=1,
                output_path=output,
                make_environment=lambda *args, **kwargs: ReplayEnvironment(),
            )

            payload = json.loads(output.read_text())
            self.assertEqual(payload["requested_seed"], 19)
            self.assertEqual(result["agent_a_seat"], 1)
            self.assertEqual(result["agent_a_money"], 3600.0)


class EnvironmentIntegrationTests(unittest.TestCase):
    def test_agents_finish_one_short_seed_in_both_seats(self):
        results = benchmark.run_suite(
            benchmark.resolve_agent("pass"),
            benchmark.resolve_agent("starter"),
            benchmark.build_schedule(0, 1),
            steps=2,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            {result["agent_a_seat"] for result in results}, {0, 1}
        )
        self.assertTrue(
            all(result["seat_0_status"] == "DONE" for result in results)
        )
        self.assertTrue(
            all(result["seat_1_status"] == "DONE" for result in results)
        )


if __name__ == "__main__":
    unittest.main()
