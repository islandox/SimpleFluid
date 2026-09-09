#!/usr/bin/env python3
"""Check that MPI performance records cannot hide malformed or failed runs."""

import csv
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from compare_verification import ComparisonError
import run_performance as performance


class PerformanceOutputsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for rank in range(2):
            directory = self.root / f"rank{rank}"
            directory.mkdir()
            (directory / "fields.csv").write_text(f"time_s,sample,value\n0,{rank},{rank + 1}\n")
            (directory / "history.csv").write_text("time_s,sample,total\n0,global,3\n")
            (directory / "timing.json").write_text(json.dumps(
                {"rank": rank, "ranks": 2, "loop_wall_s": rank + 1, "loop_cpu_s": 0.5}))

    def test_recommended_policies_are_fixture_specific(self):
        for case in performance.CASES:
            arguments = performance.solver_arguments(case, self.root, "recommended")
            with self.subTest(case=case):
                self.assertEqual("--pressure-preconditioner" in arguments, not case.startswith("bubble-"))
                self.assertEqual("--transport-preconditioner" in arguments, not case.startswith("ale-"))
                if not case.startswith("bubble-"):
                    self.assertEqual(arguments[arguments.index("--pressure-preconditioner") + 1], "dic")
                if not case.startswith("ale-"):
                    self.assertEqual(arguments[arguments.index("--transport-preconditioner") + 1], "sgs")
                plain = performance.solver_arguments(case, self.root, "default")
                self.assertNotIn("--pressure-solver", plain)
                self.assertNotIn("--transport-solver", plain)
                pressure_only = performance.solver_arguments(case, self.root, "pressure-dic")
                self.assertNotIn("--transport-solver", pressure_only)

    def test_owned_rows_and_slowest_rank_are_retained(self):
        for rank in range(2):
            (self.root / f"rank{rank}/turbulence.csv").write_text(f"time_s,sample,k_m2_s2\n0,{rank},1e-8\n")
        timing = performance.merge_outputs(self.root, 2, 0)
        self.assertEqual(timing["loop_wall_s"], 2)
        with (self.root / "merged/turbulence.csv").open() as stream:
            self.assertEqual([row["sample"] for row in csv.DictReader(stream)], ["0", "1"])
        self.assertEqual(timing["loop_cpu_sum_s"], 1)
        with (self.root / "merged/fields.csv").open() as stream:
            self.assertEqual([row["sample"] for row in csv.DictReader(stream)], ["0", "1"])
        self.assertEqual((self.root / "merged/history.csv").read_text().count("global"), 1)

    def test_duplicate_partition_ownership_is_rejected(self):
        (self.root / "rank1/fields.csv").write_text("time_s,sample,value\n0,0,2\n")
        with self.assertRaisesRegex(ComparisonError, "duplicate partition samples"):
            performance.merge_outputs(self.root, 2, 0)

    def test_differing_global_histories_are_rejected(self):
        (self.root / "rank1/history.csv").write_text("time_s,sample,total\n0,global,4\n")
        with self.assertRaisesRegex(ComparisonError, "global diagnostics differ"):
            performance.merge_outputs(self.root, 2, 0)

    def test_missing_rank_output_is_rejected(self):
        (self.root / "rank1/history.csv").unlink()
        with self.assertRaisesRegex(ComparisonError, "CSV sets differ"):
            performance.merge_outputs(self.root, 2, 0)

    def test_invalid_rank_timing_is_rejected(self):
        (self.root / "rank1/timing.json").write_text(
            '{"rank":1,"ranks":2,"loop_wall_s":-1,"loop_cpu_s":0.5}')
        with self.assertRaisesRegex(ComparisonError, "invalid rank timing"):
            performance.merge_outputs(self.root, 2, 0)

    def test_timeout_keeps_failed_measurement_without_accepted_timing(self):
        executable = self.root / "binary"
        executable.write_bytes(b"unchanged executable fixture")
        args = SimpleNamespace(output=self.root, mpiexec="mpiexec", mpi_args="",
                               timeout=10, policy="recommended")
        with patch.object(performance.subprocess, "run", side_effect=subprocess.TimeoutExpired("mpiexec", 40)):
            with self.assertRaises(subprocess.TimeoutExpired):
                performance.run(args, "convection", "baseline", executable,
                                performance.digest(executable), self.root, 2, 1, {})
        record = json.loads((self.root / "measurements.jsonl").read_text())
        self.assertIn("error", record)
        self.assertNotIn("loop_wall_s", record)
        self.assertTrue((self.root / "convection/p2/baseline-1/measurement.json").is_file())
        self.assertTrue((self.root / "convection/p2/baseline-1/launcher.log").is_file())

    def test_rebuilt_library_invalidates_unchanged_executable(self):
        executable = self.root / "binary"
        executable.write_bytes(b"unchanged executable fixture")
        library = self.root / "libFixture.so"
        library.write_bytes(b"baseline library")
        hashes = performance.shared_library_hashes(self.root)
        library.write_bytes(b"rebuilt library")
        with self.assertRaisesRegex(RuntimeError, "shared library changed"):
            performance.check_artifacts(executable, performance.digest(executable), hashes)


if __name__ == "__main__":
    unittest.main()
