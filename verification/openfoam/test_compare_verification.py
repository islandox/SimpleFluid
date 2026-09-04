#!/usr/bin/env python3
"""Focused tests for the shared physical CSV comparison contract."""

from __future__ import annotations

import copy
import csv
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import compare_verification as comparator


class VerificationComparatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.openfoam = self.root / "openfoam.csv"
        self.simplefluid = self.root / "simplefluid.csv"
        self.manifest = {
            "schema_version": 1,
            "case": "manufactured_comparator_fixture",
            "mode": "transient",
            "expected_times_s": [0.0, 0.25, 0.5],
            "expected_samples": ["0", "1"],
            "quantities": {
                "temperature_K": {"units": "K", "absolute_tolerance": 0.1,
                                  "relative_tolerance": 0.001},
                "volume_m3": {"units": "m^3", "absolute_tolerance": 1.0e-12,
                              "relative_tolerance": 1.0e-8},
            },
            "conservation": {
                "mass_residual_kg": {"absolute_tolerance": 1.0e-9},
            },
        }
        self.rows = [
            {"time_s": time, "sample": sample,
             "temperature_K": 300.0 + time + int(sample),
             "volume_m3": 0.01, "mass_residual_kg": 0.0}
            for time in self.manifest["expected_times_s"]
            for sample in self.manifest["expected_samples"]
        ]
        self.write(self.openfoam, self.rows)
        self.write(self.simplefluid, self.rows)

    @staticmethod
    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def compare(self) -> dict:
        return comparator.compare(self.manifest, self.openfoam, self.simplefluid)

    def test_identical_full_transient_passes(self) -> None:
        report = self.compare()
        self.assertTrue(report["passed"])
        self.assertEqual(report["expected_rows_per_solver"], 6)
        self.assertEqual(report["quantities"]["temperature_K"]["max_absolute_error"], 0)
        self.assertEqual(report["conservation"]["mass_residual_kg"]["openfoam"][
            "failed_samples"], 0)

    def test_single_steady_snapshot_passes_and_rows_may_be_reordered(self) -> None:
        self.manifest["mode"] = "steady"
        self.manifest["expected_times_s"] = [0.5]
        self.write(self.openfoam, self.rows[-2:])
        self.write(self.simplefluid, list(reversed(self.rows[-2:])))
        self.assertTrue(self.compare()["passed"])

    def test_absolute_plus_relative_envelope_is_pointwise(self) -> None:
        self.rows[0]["temperature_K"] += 0.39  # 0.1 + 0.001 * 300 = 0.4 K.
        self.write(self.simplefluid, self.rows)
        self.assertTrue(self.compare()["passed"])
        self.rows[0]["temperature_K"] += 0.02
        self.write(self.simplefluid, self.rows)
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertEqual(report["quantities"]["temperature_K"]["failed_samples"], 1)
        self.assertEqual(report["quantities"]["volume_m3"]["failed_samples"], 0)

    def test_zero_reference_uses_absolute_tolerance(self) -> None:
        self.rows[0]["temperature_K"] = 0
        self.write(self.openfoam, self.rows)
        self.rows[0]["temperature_K"] = 0.11
        self.write(self.simplefluid, self.rows)
        self.assertFalse(self.compare()["passed"])

    def test_matching_conservation_errors_fail_for_each_solver(self) -> None:
        self.rows[1]["mass_residual_kg"] = -2.0e-9
        self.write(self.openfoam, self.rows)
        self.write(self.simplefluid, self.rows)
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["failures"]), 2)
        self.assertTrue(all("residual bound" in failure for failure in report["failures"]))

    def test_matching_truncated_runs_fail_expected_coverage(self) -> None:
        self.write(self.openfoam, self.rows[:-2])
        self.write(self.simplefluid, self.rows[:-2])
        with self.assertRaisesRegex(comparator.ComparisonError, "missing 2 expected"):
            self.compare()

    def test_missing_cell_fails_even_with_complete_time_axis(self) -> None:
        self.write(self.simplefluid, self.rows[:-1])
        with self.assertRaisesRegex(comparator.ComparisonError, "missing 1 expected"):
            self.compare()

    def test_duplicate_row_fails(self) -> None:
        self.write(self.simplefluid, self.rows + self.rows[:1])
        with self.assertRaisesRegex(comparator.ComparisonError, "duplicate time/sample"):
            self.compare()

    def test_unexpected_sample_fails(self) -> None:
        self.rows[0]["sample"] = "2"
        self.write(self.simplefluid, self.rows)
        with self.assertRaisesRegex(comparator.ComparisonError, "unexpected sample"):
            self.compare()

    def test_unexpected_time_fails(self) -> None:
        self.rows[-1]["time_s"] = 0.51
        self.write(self.simplefluid, self.rows)
        with self.assertRaisesRegex(comparator.ComparisonError, "unexpected time"):
            self.compare()

    def test_roundoff_time_matching_passes(self) -> None:
        self.rows[-1]["time_s"] += 1.0e-12
        self.write(self.simplefluid, self.rows)
        self.assertTrue(self.compare()["passed"])

    def test_actual_solver_times_must_also_match_each_other(self) -> None:
        self.rows[-1]["time_s"] = 0.5 - 0.9e-10
        self.write(self.openfoam, self.rows)
        self.rows[-1]["time_s"] = 0.5 + 0.9e-10
        self.write(self.simplefluid, self.rows)
        with self.assertRaisesRegex(comparator.ComparisonError, "solver time mismatch"):
            self.compare()

    def test_nonfinite_quantity_time_and_conservation_rejected(self) -> None:
        for column in ("temperature_K", "time_s", "mass_residual_kg"):
            for value in ("nan", "inf", "-inf", "1e9999", ""):
                with self.subTest(column=column, value=value):
                    rows = copy.deepcopy(self.rows)
                    rows[0][column] = value
                    self.write(self.simplefluid, rows)
                    with self.assertRaisesRegex(comparator.ComparisonError, "finite"):
                        self.compare()

    def test_bad_csv_header_and_ragged_row_rejected(self) -> None:
        original = self.simplefluid.read_text(encoding="utf-8")
        variants = [
            original.replace("temperature_K", "volume_m3", 1),
            original.replace("mass_residual_kg", "misspelled_residual", 1),
            original.replace("0.0,0,300.0,0.01,0.0", "0.0,0,300.0,0.01", 1),
            original.replace("0.0,0,300.0,0.01,0.0", "0.0,0,300.0,0.01,0.0,extra", 1),
            "",
        ]
        for text in variants:
            with self.subTest(text=text):
                self.simplefluid.write_text(text, encoding="utf-8")
                with self.assertRaises(comparator.ComparisonError):
                    self.compare()

    def test_stale_outputs_rejected_with_launch_timestamp(self) -> None:
        os.utime(self.openfoam, (1000.0, 1000.0))
        with self.assertRaisesRegex(comparator.ComparisonError, "stale CSV"):
            comparator.compare(self.manifest, self.openfoam, self.simplefluid,
                               not_before=1001.0)
        report = comparator.compare(self.manifest, self.openfoam, self.simplefluid,
                                    not_before=999.0)
        self.assertTrue(report["passed"])

    def test_same_input_file_or_symlink_rejected(self) -> None:
        alias = self.root / "alias.csv"
        alias.symlink_to(self.openfoam)
        for path in (self.openfoam, alias):
            with self.subTest(path=path):
                with self.assertRaisesRegex(comparator.ComparisonError, "distinct files"):
                    comparator.compare(self.manifest, self.openfoam, path)

    def test_manifest_rejects_invalid_axes_or_tolerances(self) -> None:
        invalid = [
            {"schema_version": True}, {"expected_times_s": []},
            {"expected_times_s": [0, 0]}, {"expected_times_s": [1, 0]},
            {"expected_times_s": [0, 1.0e-11]}, {"expected_times_s": [float("nan")]},
            {"expected_times_s": [-1]}, {"expected_times_s": [True]},
            {"expected_samples": ["0", "0"]}, {"expected_samples": [0]},
            {"expected_samples": [" 0"]}, {"quantities": {}},
            {"time_tolerance_s": -1}, {"mode": "unknown"},
            {"conservation": {"residual": {"absolute_tolerance": 1,
                                             "relative_tolerance": 0}}},
        ]
        for replacement in invalid:
            with self.subTest(replacement=replacement):
                with self.assertRaises(comparator.ComparisonError):
                    comparator.validate_manifest({**self.manifest, **replacement})
        for field in ("absolute_tolerance", "relative_tolerance"):
            for invalid_value in (-1, True, None, float("inf")):
                manifest = copy.deepcopy(self.manifest)
                manifest["quantities"]["temperature_K"][field] = invalid_value
                with self.assertRaises(comparator.ComparisonError):
                    comparator.validate_manifest(manifest)

    def test_duplicate_manifest_keys_rejected(self) -> None:
        path = self.root / "manifest.json"
        path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
        with self.assertRaisesRegex(comparator.ComparisonError, "duplicate manifest key"):
            comparator.load_manifest(path)

    def run_cli(self, report_path: Path) -> int:
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            return comparator.main([
                "--manifest", str(manifest_path), "--openfoam", str(self.openfoam),
                "--simplefluid", str(self.simplefluid), "--report", str(report_path)])

    def test_cli_writes_pass_failure_and_input_error_reports(self) -> None:
        report = self.root / "reports" / "result.json"
        self.assertEqual(self.run_cli(report), 0)
        self.assertTrue(json.loads(report.read_text())["passed"])
        self.rows[0]["temperature_K"] += 2
        self.write(self.simplefluid, self.rows)
        self.assertEqual(self.run_cli(report), 1)
        self.assertEqual(json.loads(report.read_text())["status"], "failed")
        self.write(self.simplefluid, self.rows[:-1])
        self.assertEqual(self.run_cli(report), 2)
        self.assertEqual(json.loads(report.read_text())["status"], "error")

    def test_report_cannot_overwrite_input(self) -> None:
        original = self.simplefluid.read_bytes()
        self.assertEqual(self.run_cli(self.simplefluid), 2)
        self.assertEqual(self.simplefluid.read_bytes(), original)

    def test_report_cannot_overwrite_input_via_hardlink(self) -> None:
        original = self.simplefluid.read_bytes()
        alias = self.root / "report.json"
        os.link(self.simplefluid, alias)
        self.assertEqual(self.run_cli(alias), 2)
        self.assertEqual(self.simplefluid.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
