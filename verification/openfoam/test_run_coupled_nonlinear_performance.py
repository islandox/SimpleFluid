#!/usr/bin/env python3
"""Guard matched-build attribution and forcing-only performance probes."""
import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_coupled_nonlinear_performance as performance


class MatchedNonlinearPerformanceTest(unittest.TestCase):
    def test_matched_order_reverses_both_build_and_variant(self):
        forward = performance.run_plan(["openfoam", "nox-composite"], True, 1)
        self.assertEqual(forward, [("openfoam", None), ("nox-composite", "baseline"),
                                   ("nox-composite", "candidate")])
        self.assertEqual(performance.run_plan(["openfoam", "nox-composite"], True, 2),
                         list(reversed(forward)))

    def test_existing_single_build_order_is_preserved(self):
        self.assertEqual(performance.run_plan(["openfoam", "nox-composite"], False, 2),
                         [("nox-composite", None), ("openfoam", None)])

    def test_forcing_probe_preserves_physical_controls(self):
        base = performance.solver_arguments("nox-composite", Path("inputs"), Path("output"), 10, 80)
        tuned = performance.solver_arguments("nox-composite", Path("inputs"), Path("output"), 10, 80, 0.001)
        self.assertEqual(tuned, base + ["--nonlinear-forcing-initial", "0.001"])
        lagged = performance.solver_arguments("nox-composite", Path("inputs"), Path("output"), 10, 80,
                                              preconditioner_update="step")
        self.assertEqual(lagged, base + ["--nonlinear-preconditioner-update", "step"])
        for invalid in ["nan", "inf", "0", "1", "0.9", "1e-7"]:
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                performance.forcing_tolerance(invalid)

    def test_preserved_baseline_uses_its_own_executable_and_libraries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(output=root, fixtures=root, steps=10, nox_restart=80,
                                   nox_forcing_initial=0.001, nox_preconditioner_update="step", timeout=10,
                                   executable=root / "candidate", library_dir=root / "candidate-lib",
                                   libraries={"candidate-library": "candidate-hash"},
                                   baseline_executable=root / "baseline", baseline_library_dir=root / "baseline-lib",
                                   baseline_libraries={"baseline-library": "baseline-hash"})
            with patch.object(performance, "check_artifacts") as check, patch.object(
                    performance.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run:
                record = performance.run_one(args, 10752, "nox-composite", 1,
                                             {str(args.baseline_executable): "binary-hash"}, "baseline")
            command = run.call_args.args[0]
            self.assertIn(str(args.baseline_executable), command)
            self.assertNotIn(str(args.executable), command)
            self.assertNotIn("--nonlinear-forcing-initial", command)
            self.assertNotIn("--nonlinear-preconditioner-update", command)
            self.assertTrue(run.call_args.kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(args.baseline_library_dir) + ":"))
            self.assertEqual(check.call_count, 2)
            check.assert_called_with(args.baseline_executable, "binary-hash", args.baseline_libraries)
            self.assertEqual(record["build"], "baseline")
            self.assertFalse(record["qualified"])

    def test_failed_candidate_never_pools_with_qualified_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), cells=[10752], variants=["nox-composite"],
                                   baseline_build_dir=Path("baseline"))
            records = [dict(cells=10752, variant="nox-composite", build="baseline", qualified=True, loop_wall_s=12),
                       dict(cells=10752, variant="nox-composite", build="candidate", qualified=False, loop_wall_s=1)]
            summary = performance.summarize(args, records)
            self.assertEqual(summary[0]["median_s"], 12)
            self.assertEqual(summary[1]["qualified"], 0)
            self.assertNotIn("median_s", summary[1])

    def test_successful_process_without_nonlinear_statistics_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(output=root, fixtures=root, steps=10, nox_restart=80,
                                   nox_forcing_initial=None, timeout=10, executable=root / "candidate",
                                   library_dir=root / "candidate-lib", libraries={})
            with patch.object(performance, "check_artifacts"), patch.object(
                    performance.subprocess, "run", return_value=SimpleNamespace(returncode=0)), patch.object(
                    performance, "merge_outputs", return_value={"loop_wall_s": 1.0}):
                record = performance.run_one(args, 10752, "nox-composite", 1,
                                             {str(args.executable): "binary-hash"})
            self.assertFalse(record["qualified"])
            self.assertEqual(record["validation_error"], "missing nonlinear solver statistics")


if __name__ == "__main__":
    unittest.main()
