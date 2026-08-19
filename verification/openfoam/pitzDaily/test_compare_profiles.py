#!/usr/bin/env python3
"""Deterministic tests for the pitzDaily profile-comparison contract."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import compare_profiles


CASE_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = CASE_DIR / "fixtures"
FIXTURE_MANIFEST = FIXTURE_DIR / "tolerances.json"
PHYSICAL_MANIFEST = CASE_DIR / "reference" / "physical_acceptance.json"


class PitzDailyComparatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = compare_profiles.collect_metrics(
            FIXTURE_DIR / "openfoam",
            [str(FIXTURE_DIR / "simplefluid_cells_rank*.csv")],
        )
        cls.manifest = compare_profiles.load_tolerance_manifest(
            FIXTURE_MANIFEST, "comparator_fixture")
        with FIXTURE_MANIFEST.open(encoding="utf-8") as stream:
            cls.raw_manifest = json.load(stream)

    def test_recorded_metrics_match_analytical_fixture(self) -> None:
        for station_name in compare_profiles.STATIONS:
            station_document = self.raw_manifest["stations"][station_name]
            for component_name in compare_profiles.COMPONENTS:
                component_document = station_document["components"][
                    component_name]
                component_metrics = self.metrics[station_name][component_name]
                self.assertTrue(math.isclose(
                    component_metrics["l2"],
                    component_document["observed_l2_m_per_s"],
                    rel_tol=0.0,
                    abs_tol=1.0e-14,
                ))
                self.assertTrue(math.isclose(
                    component_metrics["linf"],
                    component_document["observed_linf_m_per_s"],
                    rel_tol=0.0,
                    abs_tol=1.0e-14,
                ))

    def test_profile_sample_counts_follow_supported_mesh_divisors(self) -> None:
        self.assertEqual(compare_profiles.pitz_daily_profile_sample_count(820), 15)
        self.assertEqual(compare_profiles.pitz_daily_profile_sample_count(3122), 29)
        self.assertEqual(compare_profiles.pitz_daily_profile_sample_count(12225), 57)
        self.assertIsNone(compare_profiles.pitz_daily_profile_sample_count(6))

    def test_fixture_passes_station_component_limits(self) -> None:
        self.assertEqual(
            compare_profiles.evaluate_metrics(
                self.metrics, tolerance_manifest=self.manifest),
            [],
        )

    def test_tightened_component_limit_fails_target_metric(self) -> None:
        tightened = copy.deepcopy(self.manifest)
        tightened["stations"]["x100"]["components"]["ux"][
            "max_l2"] = 0.09
        failures = compare_profiles.evaluate_metrics(
            self.metrics, tolerance_manifest=tightened)
        self.assertEqual(len(failures), 1)
        self.assertIn("x100 ux l2", failures[0])

    def test_pending_physical_manifest_is_rejected(self) -> None:
        with self.assertRaisesRegex(
                compare_profiles.ToleranceManifestError,
                "not qualified"):
            compare_profiles.load_tolerance_manifest(
                PHYSICAL_MANIFEST, "physical_reference")

    def test_fixture_cannot_satisfy_physical_scope(self) -> None:
        with self.assertRaisesRegex(
                compare_profiles.ToleranceManifestError,
                "expected 'physical_reference'"):
            compare_profiles.load_tolerance_manifest(
                FIXTURE_MANIFEST, "physical_reference")

    def test_fixture_source_checksums_match_manifest(self) -> None:
        recorded = self.raw_manifest["qualification"][
            "source_files_sha256"]
        for relative_path, expected in recorded.items():
            digest = hashlib.sha256(
                (FIXTURE_DIR / relative_path).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, relative_path)

    def test_manifest_loader_rejects_tampered_qualified_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_fixture = Path(temporary_directory) / "fixture"
            shutil.copytree(FIXTURE_DIR, copied_fixture)
            source = copied_fixture / "simplefluid_cells_rank0.csv"
            source.write_text(
                source.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    compare_profiles.ToleranceManifestError,
                    "SHA-256 mismatch"):
                compare_profiles.load_tolerance_manifest(
                    copied_fixture / "tolerances.json",
                    "comparator_fixture",
                )

    def test_nonfinite_profile_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            simplefluid_path = temporary_path / "simplefluid.csv"
            simplefluid_path.write_text(
                "x,y,ux,uy\n0.05,0.0,NaN,0.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                compare_profiles.read_simplefluid([str(simplefluid_path)])

            openfoam_path = temporary_path / "openfoam.xy"
            openfoam_path.write_text(
                "-0.1 1.0 0.0\n0.1 inf 0.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                compare_profiles.read_openfoam(openfoam_path)

    def test_incomplete_profile_and_nonfinite_metrics_fail_closed(self) -> None:
        reference = [(-1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        with self.assertRaisesRegex(ValueError, "at least two"):
            compare_profiles.error_norms(
                reference, [(0.0, 0.5, 0.0)], 1)

        incomplete = copy.deepcopy(self.metrics)
        incomplete["x050"]["samples"] = 2
        incomplete["x050"]["reference_span_fraction"] = 0.25
        failures = compare_profiles.evaluate_metrics(
            incomplete, tolerance_manifest=self.manifest)
        self.assertTrue(any("requires at least 3" in item for item in failures))
        self.assertTrue(any("OpenFOAM y range" in item for item in failures))

        nonfinite = copy.deepcopy(self.metrics)
        nonfinite["x100"]["ux"]["l2"] = math.nan
        failures = compare_profiles.evaluate_metrics(
            nonfinite, tolerance_manifest=self.manifest)
        self.assertTrue(any("x100 ux l2 is non-finite" in item
                            for item in failures))

    def test_physical_manifest_prints_authenticated_run_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_fixture = Path(temporary_directory) / "physical"
            shutil.copytree(FIXTURE_DIR, copied_fixture)
            manifest_path = copied_fixture / "tolerances.json"
            with manifest_path.open(encoding="utf-8") as stream:
                document = json.load(stream)
            document["qualification"]["scope"] = "physical_reference"
            document["reference_definition"] = {
                "openfoam_case": "OpenFOAM.com v2606 pitzDaily",
                "simplefluid_source_revision": "0123456789abcdef",
                "simplefluid_run": {
                    "mesh_divisor": 1,
                    "steps": 1200,
                    "dt_s": 2.5e-6,
                    "mpi_ranks": 4,
                },
            }
            manifest_path.write_text(
                json.dumps(document), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                result = compare_profiles.main([
                    "--tolerances", str(manifest_path),
                    "--required-scope", "physical_reference",
                    "--print-simplefluid-settings",
                ])
            self.assertEqual(result, 0)
            self.assertEqual(output.getvalue().strip(), "1 1200 2.5e-06 4")

    def test_command_line_contract_passes_fixture(self) -> None:
        with redirect_stdout(io.StringIO()):
            result = compare_profiles.main([
                "--openfoam-case", str(FIXTURE_DIR / "openfoam"),
                "--simplefluid-glob",
                str(FIXTURE_DIR / "simplefluid_cells_rank*.csv"),
                "--tolerances", str(FIXTURE_MANIFEST),
                "--required-scope", "comparator_fixture",
            ])
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
