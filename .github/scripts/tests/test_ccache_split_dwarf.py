"""Focused real-compiler checks; run with python3 -m unittest discover -s .../tests."""

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "ccache-split-dwarf.py"
SPEC = importlib.util.spec_from_file_location("split_dwarf", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(shutil.which("ccache") and shutil.which("zstd"), "requires ccache and zstd")
class SplitDwarfTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ccache-dwo-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = self.root / "symbols"
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("CCACHE_")}
        self.env.update(
            CCACHE_DIR=str(self.root / "ccache"),
            CCACHE_CONFIGPATH=str(self.root / "ccache.conf"),
            CCACHE_SLOPPINESS="pch_defines,time_macros",
            SIMPLEFLUID_DWO_CACHE_DIR=str(self.store),
        )

    def run_command(self, command, **kwargs):
        return subprocess.run(command, env=self.env, text=True, capture_output=True, check=True, **kwargs)

    def compile(self, compiler, source, output, flags=None):
        return self.run_command([
            "python3", str(SCRIPT), "ccache", compiler,
            *(flags if flags is not None else ["-g", "-gsplit-dwarf"]),
            "-c", str(source), "-o", str(output),
        ])

    def hits(self):
        stats = self.run_command(["ccache", "--print-stats"]).stdout
        return sum(int(line.split()[1]) for line in stats.splitlines()
                   if line.startswith(("direct_cache_hit\t", "preprocessed_cache_hit\t")))

    def test_compiler_roundtrips_and_recovery(self):
        compilers = [compiler for compiler in ("g++", "clang++") if shutil.which(compiler)]
        self.assertTrue(compilers, "requires GCC or Clang")
        for compiler in compilers:
            with self.subTest(compiler=compiler):
                work = self.root / compiler
                work.mkdir()
                source = work / "sample.cpp"
                source.write_text("int twice(int value) { return value * 2; }\n")
                output = work / "sample.cpp.o"
                dwo = output.with_suffix(".dwo")
                self.compile(compiler, source, output)
                symbols = dwo.read_bytes()
                original_object = output.read_bytes()
                record = self.store / hashlib.sha256(original_object).hexdigest()
                self.assertTrue(record.is_file())
                # Restored bundle is the only symbol source after a clean build.
                bundle = work / "symbols.tar.zst"
                self.run_command(["python3", str(SCRIPT), "pack", str(bundle)])
                shutil.rmtree(self.store)
                self.run_command(["python3", str(SCRIPT), "unpack", str(bundle)])
                output.unlink()
                dwo.unlink()
                hits = self.hits()
                self.compile(compiler, source, output)
                self.assertGreater(self.hits(), hits)
                self.assertEqual(original_object, output.read_bytes())
                self.assertEqual(symbols, dwo.read_bytes())
                # A stale build-tree sidecar must never override matched symbols.
                dwo.write_bytes(b"stale debug information")
                self.compile(compiler, source, output)
                self.assertEqual(symbols, dwo.read_bytes())
                for damage in ("missing", "truncated", "checksum"):
                    with self.subTest(damage=damage):
                        record = self.store / hashlib.sha256(output.read_bytes()).hexdigest()
                        if damage == "missing":
                            record.unlink()
                        elif damage == "truncated":
                            record.write_bytes(b"broken")
                        else:
                            record.write_bytes(b"0" * 64 + b"\nwrong symbols")
                        dwo.unlink()
                        result = self.compile(compiler, source, output)
                        self.assertIn("regenerating", result.stderr)
                        self.assertEqual(symbols, dwo.read_bytes())
                # Output names are part of skeleton DWARF identity.
                renamed = work / "other" / "renamed.o"
                renamed.parent.mkdir()
                self.compile(compiler, source, renamed)
                info = self.run_command(["readelf", "--debug-dump=info", str(renamed)]).stdout
                self.assertIn("renamed.dwo", info)
                self.assertTrue(renamed.with_suffix(".dwo").is_file())
                # A restored object and its symbols still link and run.
                main = work / "main.cpp"
                main.write_text("int twice(int); int main() { return twice(4) == 8 ? 0 : 1; }\n")
                binary = work / "consumer"
                self.run_command([compiler, str(main), str(output), "-o", str(binary)])
                self.run_command([str(binary)])
                if shutil.which("gdb"):
                    debug = self.run_command([
                        "gdb", "-batch", "-ex", "set debuginfod enabled off",
                        "-ex", "info functions twice", "-ex", "list twice", "-ex", "ptype twice", str(binary),
                    ]).stdout
                    self.assertIn("sample.cpp", debug)
                    self.assertIn("return value * 2", debug)
                    self.assertIn("int (int)", debug)
                # Valid no-symbol compilations must also have warm behavior.
                quiet = work / "quiet.o"
                self.compile(compiler, source, quiet, ["-g0", "-gsplit-dwarf"])
                self.assertNotIn("regenerating", self.compile(
                    compiler, source, quiet, ["-g0", "-gsplit-dwarf"]).stderr)
                # Ordinary non-split / LTO calls continue through regular ccache.
                for flags in (["-g"], ["-g", "-flto"]):
                    self.compile(compiler, source, work / "ordinary.o", flags)
        # Inspect the actual cache format, not only the files in the build tree.
        results = 0
        for path in (self.root / "ccache").rglob("*"):
            if path.is_file():
                inspection = subprocess.run(["ccache", "--inspect", str(path)], text=True, capture_output=True)
                if "Entry type: 0 (result)" in inspection.stdout:
                    results += 1
                    self.assertNotIn(".dwo", inspection.stdout)
        self.assertGreater(results, 0)

    @unittest.skipUnless(shutil.which("cmake") and shutil.which("ninja"), "requires CMake and Ninja")
    def test_cmake_pch_clean_build_and_warm_build(self):
        for compiler in ("g++", "clang++"):
            if not shutil.which(compiler):
                continue
            with self.subTest(compiler=compiler):
                project = self.root / compiler
                project.mkdir()
                (project / "pch.hh").write_text("#include <vector>\n")
                (project / "main.cpp").write_text(
                    "int main() { std::vector<int> values{3, 5}; return values[0] + values[1] == 8 ? 0 : 1; }\n")
                (project / "CMakeLists.txt").write_text(
                    "cmake_minimum_required(VERSION 3.21)\nproject(DwoFixture CXX)\n"
                    "add_executable(consumer main.cpp)\n"
                    "target_compile_options(consumer PRIVATE -gsplit-dwarf)\n"
                    "target_precompile_headers(consumer PRIVATE pch.hh)\n")
                build = project / "build"
                self.run_command([
                    "cmake", "-S", str(project), "-B", str(build), "-G", "Ninja Multi-Config",
                    "-DCMAKE_CXX_COMPILER=" + compiler,
                    "-DCMAKE_CXX_COMPILER_LAUNCHER=python3;" + str(SCRIPT) + ";ccache",
                    *(["-DCMAKE_CXX_FLAGS=-Xclang -fno-pch-timestamp"] if compiler == "clang++" else []),
                ])
                command = ["cmake", "--build", str(build), "--config", "Debug", "--parallel", "2"]
                self.run_command(command)
                original = {path.relative_to(build): path.read_bytes() for path in build.rglob("*.dwo")}
                self.assertTrue(original)
                bundle = project / "bundle.tar.zst"
                self.run_command(["python3", str(SCRIPT), "pack", str(bundle)])
                self.run_command(["cmake", "--build", str(build), "--config", "Debug", "--target", "clean"])
                for path in build.rglob("*.dwo"):
                    path.unlink()
                self.run_command(["python3", str(SCRIPT), "unpack", str(bundle)])
                hits = self.hits()
                self.run_command(command)
                self.assertGreater(self.hits(), hits)
                for relative, data in original.items():
                    self.assertEqual(data, (build / relative).read_bytes())
                self.run_command([str(build / "Debug" / "consumer")])

    def test_bundle_rejects_unsafe_members_and_missing_bundle_is_empty(self):
        self.store.mkdir()
        (self.store / "leftover").write_text("stale")
        MODULE.unpack(self.store, self.root / "absent.tar.zst")
        self.assertEqual(list(self.store.iterdir()), [])
        tar = self.root / "unsafe.tar"
        with tarfile.open(tar, "w") as archive:
            member = tarfile.TarInfo("../escape")
            member.size = 4
            archive.addfile(member, io.BytesIO(b"oops"))
        bundle = self.root / "unsafe.tar.zst"
        self.run_command(["zstd", "-q", str(tar), "-o", str(bundle)])
        with self.assertRaises(ValueError):
            MODULE.unpack(self.store, bundle)
        self.assertFalse((self.root / "escape").exists())
        self.assertFalse(self.store.exists())

    def test_prune_and_failed_pack_preserve_previous_bundle(self):
        self.store.mkdir()
        recent = self.store / ("1" * 64)
        stale = self.store / ("2" * 64)
        recent.write_bytes(b"none\n")
        stale.write_bytes(b"none\n")
        os.utime(stale, (0, 0))
        bundle = self.root / "bundle.tar.zst"
        MODULE.pack(self.store, bundle)
        self.assertTrue(recent.exists())
        self.assertFalse(stale.exists())
        previous = bundle.read_bytes()
        failing = self.root / "bin"
        failing.mkdir()
        executable = failing / "zstd"
        executable.write_text("#!/bin/sh\ncat >/dev/null\nexit 1\n")
        executable.chmod(0o755)
        from unittest.mock import patch
        with patch.dict(os.environ, {"PATH": str(failing) + os.pathsep + os.environ["PATH"]}):
            with self.assertRaises(RuntimeError):
                MODULE.pack(self.store, bundle)
        self.assertEqual(previous, bundle.read_bytes())

    def test_workflow_cache_replacement_conditions(self):
        workflow = SCRIPT.parents[1] / "workflows" / "cmake-multi-platform.yml"
        text = workflow.read_text()
        conditions = []
        for name in ("Delete previous SimpleFluid compiler cache", "Save SimpleFluid compiler cache"):
            step = text.split("- name: " + name + "\n", 1)[1].split("\n    - ", 1)[0]
            condition = re.search(r"^      if: >-\n((?:        .*\n)+)", step, re.M)
            self.assertIsNotNone(condition)
            conditions.append(" ".join(condition[1].split()).replace("&&", "and").replace("||", "or"))
        values = {
            "steps.build.outcome": "success", "steps.dwo_pack.outcome": "success",
            "steps.ccache.outputs.cache-primary-key": "primary",
            "github.event_name": "push",
            "github.event.pull_request.head.repo.full_name": "owner/repo",
            "github.repository": "owner/repo", "steps.ccache_delete.outcome": "success",
            "steps.ccache.outputs.cache-hit": "true",
        }
        cases = [
            ({}, (True, True)),
            ({"steps.dwo_pack.outcome": "failure"}, (False, False)),
            ({"steps.dwo_pack.outcome": "skipped"}, (False, False)),
            ({"steps.build.outcome": "failure"}, (False, False)),
            ({"steps.ccache.outputs.cache-primary-key": ""}, (False, False)),
            ({"steps.ccache_delete.outcome": "failure"}, (True, False)),
            ({"github.event_name": "pull_request",
              "github.event.pull_request.head.repo.full_name": "fork/repo",
              "steps.ccache_delete.outcome": "skipped"}, (False, False)),
            ({"github.event_name": "pull_request",
              "github.event.pull_request.head.repo.full_name": "fork/repo",
              "steps.ccache_delete.outcome": "skipped",
              "steps.ccache.outputs.cache-hit": "false"}, (False, True)),
        ]
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                actual = []
                for condition in conditions:
                    for key, value in {**values, **overrides}.items():
                        condition = condition.replace(key, repr(value))
                    actual.append(bool(eval(condition, {"__builtins__": {}})))
                self.assertEqual(tuple(actual), expected)
        self.assertLess(text.index("- name: Pack split DWARF bundle"),
                        text.index("- name: Delete previous SimpleFluid compiler cache"))


if __name__ == "__main__":
    unittest.main()
