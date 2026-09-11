#!/usr/bin/env python3
"""Exercise cavity selection and failure handling without running a solver."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class CavityLauncherTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = Path(__file__).resolve().parent
        case = self.root / "verification/openfoam/cavityFlow"
        case.mkdir(parents=True)
        self.launcher = case / "run_simplefluid.sh"
        shutil.copyfile(source / "run_simplefluid.sh", self.launcher)
        shutil.copyfile(source.parents[1] / "environments.sh",
                        self.root / "verification/environments.sh")
        self.build = self.root / "build"
        binary_dir = self.build / "bin/RelWithDebInfo"
        binary_dir.mkdir(parents=True)
        self.write_cache("ON")
        self.executable = binary_dir / "testVerificationCases"
        self.write_executable("""#!/bin/sh
printf '%s\\n' "$@" > "$TEST_INVOCATION_FILE"
for axis in X Y; do
    printf 'coordinate,ux,uy,uz\\n0,1,0,0\\n' > "$SIMPLEFLUID_PROFILE_OUTPUT_DIR/simplefluid_line${axis}.csv"
done
""")
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        cmake = fake_bin / "cmake"
        cmake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$TEST_BUILD_FILE"\n')
        cmake.chmod(0o755)
        self.output = self.root / "profiles"
        self.output.mkdir()
        self.profile = self.output / "simplefluid_lineX.csv"
        self.profile.write_text("retained profile\n")
        self.invocation = self.root / "invocation.txt"
        self.build_log = self.root / "build.txt"
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith("SIMPLEFLUID_")
        }
        self.environment.update({
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "SIMPLEFLUID_BUILD_DIR": str(self.build),
            "SIMPLEFLUID_PROFILE_OUTPUT_DIR": str(self.output),
            "SIMPLEFLUID_ENV_QUIET": "1",
            "TEST_INVOCATION_FILE": str(self.invocation),
            "TEST_BUILD_FILE": str(self.build_log),
        })

    def write_cache(self, enabled):
        (self.build / "CMakeCache.txt").write_text(
            "CMAKE_CONFIGURATION_TYPES:STRING=Debug;Release;RelWithDebInfo\n"
            f"SIMPLEFLUID_ENABLE_NOX:BOOL={enabled}\n")

    def write_executable(self, contents):
        self.executable.write_text(contents)
        self.executable.chmod(0o755)

    def run_launcher(self, *arguments):
        return subprocess.run(
            ["/bin/sh", str(self.launcher), *arguments], cwd=self.root,
            env=self.environment, text=True, capture_output=True, check=False)

    def assert_preserved(self):
        self.assertEqual(self.profile.read_text(), "retained profile\n")
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.invocation.exists())

    def test_default_preserves_piso_re1000_selection(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.invocation.read_text().strip(),
                         "--gtest_filter=VerificationCasesTest.LidDrivenCavityRe1000")

    def test_selects_each_supported_coupling_and_reynolds(self):
        for coupling in ("piso", "nox"):
            for reynolds in ("100", "1000"):
                with self.subTest(coupling=coupling, reynolds=reynolds):
                    result = self.run_launcher("--coupling", coupling,
                                               "--reynolds", reynolds)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    suffix = "Nox" if coupling == "nox" else ""
                    self.assertEqual(self.invocation.read_text().strip(),
                                     "--gtest_filter=VerificationCasesTest."
                                     f"LidDrivenCavityRe{reynolds}{suffix}")

    def test_cli_overrides_environment_coupling(self):
        self.environment["SIMPLEFLUID_CAVITY_COUPLING"] = "nox"
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LidDrivenCavityRe1000Nox", self.invocation.read_text())
        result = self.run_launcher("--coupling", "piso")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Nox", self.invocation.read_text())

    def test_disabled_nox_preserves_profiles_without_building(self):
        self.write_cache("OFF")
        result = self.run_launcher("--coupling", "nox")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SIMPLEFLUID_ENABLE_NOX=ON", result.stderr)
        self.assert_preserved()

    def test_unconfigured_nox_preserves_profiles_without_configuring(self):
        (self.build / "CMakeCache.txt").unlink()
        result = self.run_launcher("--coupling", "nox")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("configure that build tree explicitly", result.stderr)
        self.assert_preserved()

    def test_invalid_arguments_preserve_profiles_without_building(self):
        for arguments in (("--unknown",), ("--coupling",),
                          ("--coupling", "newton"), ("--reynolds",),
                          ("--reynolds", "200")):
            with self.subTest(arguments=arguments):
                result = self.run_launcher(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assert_preserved()

    def test_skipped_test_cannot_reuse_existing_profiles(self):
        self.write_executable("#!/bin/sh\nexit 0\n")
        result = self.run_launcher("--coupling", "nox")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not produce", result.stderr)
        self.assertFalse(self.profile.exists())

    def test_coupled_policy_controls_require_nox(self):
        for variable in ("SIMPLEFLUID_CAVITY_COUPLED_OPERATOR",
                         "SIMPLEFLUID_CAVITY_COUPLED_WORKSPACE"):
            with self.subTest(variable=variable):
                self.environment[variable] = "assembled"
                result = self.run_launcher()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("require --coupling nox", result.stderr)
                self.assert_preserved()
                del self.environment[variable]

    def test_invalid_coupled_policies_fail_before_building(self):
        for variable in ("SIMPLEFLUID_CAVITY_COUPLED_OPERATOR",
                         "SIMPLEFLUID_CAVITY_COUPLED_WORKSPACE"):
            with self.subTest(variable=variable):
                self.environment[variable] = "invalid"
                result = self.run_launcher("--coupling", "nox")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(variable, result.stderr)
                self.assert_preserved()
                del self.environment[variable]

    def test_both_profiles_are_required(self):
        self.write_executable("""#!/bin/sh
printf 'coordinate,ux,uy,uz\\n0,1,0,0\\n' > "$SIMPLEFLUID_PROFILE_OUTPUT_DIR/simplefluid_lineX.csv"
""")
        result = self.run_launcher("--coupling", "nox")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("simplefluid_lineY.csv", result.stderr)

    def test_solver_failure_is_propagated(self):
        self.write_executable("#!/bin/sh\nexit 7\n")
        result = self.run_launcher("--coupling", "nox")
        self.assertEqual(result.returncode, 7)


if __name__ == "__main__":
    unittest.main()
