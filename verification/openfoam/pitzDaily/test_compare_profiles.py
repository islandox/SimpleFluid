#!/usr/bin/env python3
"""Deterministic tests for the pitzDaily profile-comparison contract."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import compare_profiles


CASE_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = CASE_DIR / "fixtures"
FIXTURE_MANIFEST = FIXTURE_DIR / "tolerances.json"
PHYSICAL_MANIFEST = CASE_DIR / "reference" / "physical_acceptance.json"
OPENFOAM_CASE = "OpenFOAM.com v2606 incompressible/simpleFoam/pitzDaily"
SIMPLEFLUID_REVISION = "0123456789abcdef0123456789abcdef01234567"


def qualified_simplefluid_run() -> dict[str, object]:
    """Return a complete deterministic physical-run contract for tests."""
    return {
        "mesh_divisor": 1,
        "steps": 1200,
        "dt_s": 2.5e-6,
        "mpi_ranks": 1,
        "steady_state": True,
        "linear_tolerance": 1.0e-9,
        "steady_consecutive_steps": 5,
        "steady_min_steps": 20,
        "steady_max_retries": 4,
        "steady_rejection_recovery_steps": 5,
        "steady_tolerance": 1.0e-4,
        "steady_min_dt_s": 1.0e-7,
        "steady_max_dt_s": 5.0e-2,
        "steady_target_courant": 0.8,
        "steady_dt_growth_factor": 1.5,
        "steady_dt_reduction_factor": 0.5,
        "steady_rejection_safety_factor": 0.9,
        "steady_relaxed_linear_tolerance": 1.0e-6,
        "steady_full_accuracy_update_ratio": 10.0,
        "steady_progress_interval": 1,
        "rank_csv_files": ["simplefluid_cells_rank0.csv"],
    }


def qualified_reference_definition(
    settings: dict[str, object] | None = None,
) -> dict[str, object]:
    """Bind the checked-in fixture outputs to a physical-style run record."""
    return {
        "openfoam_case": OPENFOAM_CASE,
        "openfoam_profile_files": {
            station: f"openfoam/postProcessing/profiles/2000/{station}_U.xy"
            for station in compare_profiles.STATIONS
        },
        "simplefluid_source_revision": SIMPLEFLUID_REVISION,
        "simplefluid_run": (
            qualified_simplefluid_run() if settings is None else settings),
    }


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

    def test_openfoam_profiles_use_latest_complete_common_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_openfoam = Path(temporary_directory) / "openfoam"
            shutil.copytree(FIXTURE_DIR / "openfoam", copied_openfoam)
            source_time = (
                copied_openfoam / "postProcessing" / "profiles" / "2000")
            complete_time = (
                copied_openfoam / "postProcessing" / "profiles" / "2500")
            shutil.copytree(source_time, complete_time)
            incomplete_time = (
                copied_openfoam / "postProcessing" / "profiles" / "3000")
            incomplete_time.mkdir()
            for station_name in ("x050", "x100"):
                shutil.copy(
                    source_time / f"{station_name}_U.xy",
                    incomplete_time,
                )

            profiles = compare_profiles.latest_common_profiles(copied_openfoam)
            self.assertEqual(set(profiles), set(compare_profiles.STATIONS))
            self.assertEqual(
                {profile.parent.name for profile in profiles.values()},
                {"2500"},
            )

    def test_openfoam_profiles_require_a_common_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            openfoam = Path(temporary_directory) / "openfoam"
            profiles_root = openfoam / "postProcessing" / "profiles"
            for index, station_name in enumerate(
                    compare_profiles.STATIONS, start=1):
                time_directory = profiles_root / str(index * 1000)
                time_directory.mkdir(parents=True)
                (time_directory / f"{station_name}_U.xy").write_text(
                    "-0.1 0 0\n0.1 0 0\n", encoding="utf-8")

            with self.assertRaisesRegex(
                    FileNotFoundError, "no common numeric OpenFOAM time"):
                compare_profiles.latest_common_profiles(openfoam)

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
            document["reference_definition"] = \
                qualified_reference_definition()
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
            self.assertEqual(
                output.getvalue().strip(),
                "1 1200 2.5e-06 1 1 1e-09 5 20 4 5 0.0001 1e-07 "
                "0.05 0.8 1.5 0.5 0.9 1e-06 10.0 1",
            )

    def test_physical_manifest_requires_every_solver_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_fixture = Path(temporary_directory) / "physical"
            shutil.copytree(FIXTURE_DIR, copied_fixture)
            manifest_path = copied_fixture / "tolerances.json"
            with manifest_path.open(encoding="utf-8") as stream:
                document = json.load(stream)
            document["qualification"]["scope"] = "physical_reference"
            settings = qualified_simplefluid_run()
            del settings["steady_target_courant"]
            document["reference_definition"] = \
                qualified_reference_definition(settings)
            manifest_path.write_text(
                json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                    compare_profiles.ToleranceManifestError,
                    "steady_target_courant must be a number"):
                compare_profiles.load_tolerance_manifest(
                    manifest_path, "physical_reference")

    def test_physical_manifest_rejects_inconsistent_solver_settings(
            self) -> None:
        invalid_settings = (
            ("initial timestep", {"dt_s": 1.0e-8},
             "dt_s must lie within"),
            ("growth factor", {"steady_dt_growth_factor": 1.0},
             "growth_factor must be greater than one"),
            ("reduction factor", {"steady_dt_reduction_factor": 1.0},
             r"reduction_factor must lie in \(0, 1\)"),
            ("rejection safety", {"steady_rejection_safety_factor": 1.0},
             r"safety_factor must lie in \(0, 1\)"),
            ("relaxed tolerance", {
                "linear_tolerance": 1.0e-6,
                "steady_relaxed_linear_tolerance": 1.0e-9,
            }, "cannot be stricter than linear_tolerance"),
            ("full accuracy ratio", {
                "steady_full_accuracy_update_ratio": 0.5,
            }, "must be at least one"),
            ("steady window", {
                "steady_min_steps": 1198,
                "steady_consecutive_steps": 5,
            }, "do not fit within steps"),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_fixture = Path(temporary_directory) / "physical"
            shutil.copytree(FIXTURE_DIR, copied_fixture)
            manifest_path = copied_fixture / "tolerances.json"
            with manifest_path.open(encoding="utf-8") as stream:
                base_document = json.load(stream)
            base_document["qualification"]["scope"] = "physical_reference"

            for case_name, updates, expected_message in invalid_settings:
                with self.subTest(case_name):
                    document = copy.deepcopy(base_document)
                    settings = qualified_simplefluid_run()
                    settings.update(updates)
                    document["reference_definition"] = \
                        qualified_reference_definition(settings)
                    manifest_path.write_text(
                        json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(
                            compare_profiles.ToleranceManifestError,
                            expected_message):
                        compare_profiles.load_tolerance_manifest(
                            manifest_path, "physical_reference")

    def test_physical_manifest_rejects_rank_relabeling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_fixture = Path(temporary_directory) / "physical"
            shutil.copytree(FIXTURE_DIR, copied_fixture)
            manifest_path = copied_fixture / "tolerances.json"
            with manifest_path.open(encoding="utf-8") as stream:
                document = json.load(stream)
            document["qualification"]["scope"] = "physical_reference"
            settings = qualified_simplefluid_run()
            settings["mpi_ranks"] = 4
            document["reference_definition"] = \
                qualified_reference_definition(settings)
            manifest_path.write_text(
                json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                    compare_profiles.ToleranceManifestError,
                    "one file per declared MPI rank"):
                compare_profiles.load_tolerance_manifest(
                    manifest_path, "physical_reference")

    def test_physical_manifest_requires_role_bound_output_checksums(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_fixture = Path(temporary_directory) / "physical"
            shutil.copytree(FIXTURE_DIR, copied_fixture)
            unbound_profile = copied_fixture / "unbound" / "x050_U.xy"
            unbound_profile.parent.mkdir()
            shutil.copy(
                copied_fixture
                / "openfoam/postProcessing/profiles/2000/x050_U.xy",
                unbound_profile,
            )
            manifest_path = copied_fixture / "tolerances.json"
            with manifest_path.open(encoding="utf-8") as stream:
                document = json.load(stream)
            document["qualification"]["scope"] = "physical_reference"
            definition = qualified_reference_definition()
            definition["openfoam_profile_files"]["x050"] = \
                "unbound/x050_U.xy"
            document["reference_definition"] = definition
            manifest_path.write_text(
                json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                    compare_profiles.ToleranceManifestError,
                    "has no source_files_sha256 entry"):
                compare_profiles.load_tolerance_manifest(
                    manifest_path, "physical_reference")

    def test_physical_manifest_rejects_ambiguous_provenance(self) -> None:
        invalid_provenance = (
            ("OpenFOAM case", {"openfoam_case": "some pitzDaily"},
             "must identify the OpenFOAM.com v2606"),
            ("source revision", {"simplefluid_source_revision": "deadbeef"},
             "must be a full 40-digit hexadecimal Git revision"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_fixture = Path(temporary_directory) / "physical"
            shutil.copytree(FIXTURE_DIR, copied_fixture)
            manifest_path = copied_fixture / "tolerances.json"
            with manifest_path.open(encoding="utf-8") as stream:
                base_document = json.load(stream)
            base_document["qualification"]["scope"] = "physical_reference"

            for case_name, updates, expected_message in invalid_provenance:
                with self.subTest(case_name):
                    document = copy.deepcopy(base_document)
                    definition = qualified_reference_definition()
                    definition.update(updates)
                    document["reference_definition"] = definition
                    manifest_path.write_text(
                        json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(
                            compare_profiles.ToleranceManifestError,
                            expected_message):
                        compare_profiles.load_tolerance_manifest(
                            manifest_path, "physical_reference")

    def test_acceptance_launcher_exports_contract_and_uses_output_dir(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            case_path = temporary_path / "case"
            openfoam_path = case_path / "openfoam"
            invocation_path = temporary_path / "invocation"
            output_path = invocation_path / "selected profiles"
            openfoam_path.mkdir(parents=True)
            invocation_path.mkdir()
            shutil.copy(CASE_DIR / "run_comparison.sh", case_path)

            (openfoam_path / "Allrun").write_text(
                "#!/bin/sh\nexit 0\n", encoding="utf-8")
            (case_path / "run_simplefluid.sh").write_text(
                "#!/bin/sh\nexit 0\n", encoding="utf-8")
            os.chmod(openfoam_path / "Allrun", 0o755)
            os.chmod(case_path / "run_simplefluid.sh", 0o755)

            recorded_path = temporary_path / "recorded.json"
            expected_settings = (
                "1 1200 2.5e-6 4 1 1e-9 5 20 4 5 1e-4 1e-7 5e-2 "
                "0.8 1.5 0.5 0.9 1e-6 10 7"
            )
            (case_path / "compare_profiles.py").write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if "--print-simplefluid-settings" in sys.argv:
    print(os.environ["EXPECTED_SETTINGS"])
else:
    names = [
        "SIMPLEFLUID_PITZ_MESH_DIVISOR",
        "SIMPLEFLUID_PITZ_STEPS",
        "SIMPLEFLUID_PITZ_DT",
        "SIMPLEFLUID_PITZ_STEADY_STATE",
        "SIMPLEFLUID_PITZ_LINEAR_TOLERANCE",
        "SIMPLEFLUID_PITZ_STEADY_CONSECUTIVE_STEPS",
        "SIMPLEFLUID_PITZ_STEADY_MIN_STEPS",
        "SIMPLEFLUID_PITZ_STEADY_MAX_RETRIES",
        "SIMPLEFLUID_PITZ_STEADY_REJECTION_RECOVERY_STEPS",
        "SIMPLEFLUID_PITZ_STEADY_TOLERANCE",
        "SIMPLEFLUID_PITZ_STEADY_MIN_DT",
        "SIMPLEFLUID_PITZ_STEADY_MAX_DT",
        "SIMPLEFLUID_PITZ_STEADY_TARGET_COURANT",
        "SIMPLEFLUID_PITZ_STEADY_DT_GROWTH",
        "SIMPLEFLUID_PITZ_STEADY_DT_REDUCTION",
        "SIMPLEFLUID_PITZ_STEADY_REJECTION_SAFETY",
        "SIMPLEFLUID_PITZ_STEADY_RELAXED_LINEAR_TOLERANCE",
        "SIMPLEFLUID_PITZ_STEADY_FULL_ACCURACY_UPDATE_RATIO",
        "SIMPLEFLUID_PITZ_STEADY_PROGRESS_INTERVAL",
        "SIMPLEFLUID_PITZ_OUTPUT_DIR",
    ]
    Path(os.environ["RECORDED_PATH"]).write_text(json.dumps({
        "arguments": sys.argv[1:],
        "environment": {name: os.environ[name] for name in names},
    }), encoding="utf-8")
""",
                encoding="utf-8",
            )

            environment = os.environ.copy()
            environment.update({
                "EXPECTED_SETTINGS": expected_settings,
                "RECORDED_PATH": str(recorded_path),
                "SIMPLEFLUID_PITZ_OUTPUT_DIR": "selected profiles",
                "SIMPLEFLUID_PITZ_STEADY_TARGET_COURANT": "999",
            })
            result = subprocess.run(
                ["/bin/sh", str(case_path / "run_comparison.sh"), "4"],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                cwd=invocation_path,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
            self.assertEqual(
                recorded["arguments"][recorded["arguments"].index(
                    "--simplefluid-glob") + 1],
                str(output_path.resolve() / "simplefluid_cells_rank*.csv"),
            )
            self.assertEqual(
                list(recorded["environment"].values()),
                expected_settings.split()[0:3]
                + expected_settings.split()[4:]
                + [str(output_path.resolve())],
            )

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
