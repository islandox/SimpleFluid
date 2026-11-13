#!/usr/bin/env python3
"""Focused standalone checks for split and compressed LTO debug information."""

import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import zlib


SOURCE = Path(__file__).resolve().parent
LINKED = {
    "ipo_all", "ipo_debug_off", "ipo_debug_on", "ipo_relwithdebinfo_off", "ipo_relwithdebinfo_on", "plain",
    "ipo_shared", "ipo_module",
}
TARGETS = sorted(LINKED | {"ipo_static", "ipo_object", "plain_static", "plain_object"})
GENERAL_IPO = {"ipo_all", "ipo_debug_off", "ipo_relwithdebinfo_off", "ipo_shared", "ipo_module", "ipo_static", "ipo_object"}
DEBUG_CONFIGS = {"Debug", "RelWithDebInfo"}


def expected_ipo(target, config):
    # These expectations are independent of the production helper's generator expressions.
    if target == "ipo_debug_off" and config == "Debug":
        return False
    if target == "ipo_debug_on" and config == "Debug":
        return True
    if target == "ipo_relwithdebinfo_off" and config == "RelWithDebInfo":
        return False
    if target == "ipo_relwithdebinfo_on" and config == "RelWithDebInfo":
        return True
    return target in GENERAL_IPO


def report(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def elf_debug_sections(path):
    """Read ELF directly: SHF_COMPRESSED plus its zlib header, not readelf wording."""
    raw = path.read_bytes()
    if raw[:4] != b"\x7fELF":
        raise AssertionError(f"Not an ELF artifact: {path}")
    bits = raw[4]
    endian = {1: "<", 2: ">"}[raw[5]]
    if bits == 2:
        section_offset = struct.unpack_from(endian + "Q", raw, 40)[0]
        entry_size, count, names_index = struct.unpack_from(endian + "HHH", raw, 58)
        layout = endian + "IIQQQQIIQQ"
    elif bits == 1:
        section_offset = struct.unpack_from(endian + "I", raw, 32)[0]
        entry_size, count, names_index = struct.unpack_from(endian + "HHH", raw, 46)
        layout = endian + "IIIIIIIIII"
    else:
        raise AssertionError(f"Unsupported ELF class {bits}: {path}")
    first = struct.unpack_from(layout, raw, section_offset)
    if count == 0:
        count = first[5]
    if names_index == 0xFFFF:
        names_index = first[6]
    headers = [struct.unpack_from(layout, raw, section_offset + i * entry_size) for i in range(count)]
    name_header = headers[names_index]
    names = raw[name_header[4]:name_header[4] + name_header[5]]
    sections = {}
    for header in headers:
        name = names[header[0]:].split(b"\0", 1)[0].decode("ascii")
        if not name.startswith((".debug_", ".zdebug_")):
            continue
        payload = raw[header[4]:header[4] + header[5]]
        item = {"compressed": bool(header[2] & 0x800), "stored_bytes": len(payload)}
        if item["compressed"]:
            if bits == 2:
                algorithm, reserved, size, alignment = struct.unpack_from(endian + "IIQQ", payload)
                header_bytes = 24
                if reserved != 0:
                    raise AssertionError(f"Nonzero ELF compression reserved field: {path}:{name}")
            else:
                algorithm, size, alignment = struct.unpack_from(endian + "III", payload)
                header_bytes = 12
            if algorithm != 1:
                raise AssertionError(f"Expected ELF zlib type 1, got {algorithm}: {path}:{name}")
            decoded = zlib.decompress(payload[header_bytes:])
            if len(decoded) != size:
                raise AssertionError(f"Compressed section size mismatch: {path}:{name}")
            item.update(algorithm=algorithm, uncompressed_bytes=size, alignment=alignment)
        sections[name] = item
    return sections


class Runner:
    def __init__(self, args):
        self.args = args
        self.root = args.build_root.resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError(f"Build root must be new or empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.summary = {"source": str(SOURCE), "commands": [], "checks": [], "skips": [], "artifacts": {}}

    def check(self, condition, message):
        if not condition:
            raise AssertionError(message)
        self.summary["checks"].append(message)

    def command(self, label, command, *, cwd=None, success=True):
        command = [str(part) for part in command]
        completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = self.root / f"{label}.log"
        log.write_text("$ " + shlex.join(command) + "\n" + completed.stdout)
        self.summary["commands"].append({"command": command, "cwd": str(cwd) if cwd else None,
                                         "returncode": completed.returncode, "log": str(log)})
        if success and completed.returncode:
            raise RuntimeError(f"{label} failed; see {log}\n{completed.stdout[-4000:]}")
        if not success and not completed.returncode:
            raise AssertionError(f"{label} unexpectedly succeeded; see {log}")
        return completed.stdout

    def configure(self, label, compiler, options=(), *, success=True):
        directory = self.root / label
        command = [self.args.cmake, "-S", SOURCE, "-B", directory, "-G", "Ninja Multi-Config",
                   f"-DCMAKE_CXX_COMPILER={compiler}", "-DCMAKE_CONFIGURATION_TYPES=Debug;RelWithDebInfo;Release"]
        output = self.command(label + "-configure", command + list(options), success=success)
        return directory, output

    def negative_checks(self, compiler):
        common = ["-DCMAKE_SYSTEM_NAME=Generic", "-DFIXTURE_PLATFORM_CHECK=ON"]
        directory, _ = self.configure("unsupported-default", compiler, common)
        options = report(directory / "platform-options.txt")
        self.check(options == {"compress": "OFF", "split": "OFF"}, "Unsupported platform defaults disable both policies")
        _, output = self.configure("unsupported-opt-in", compiler,
                                   common + ["-DSIMPLEFLUID_COMPRESS_LTO_DEBUG_INFO=ON"], success=False)
        self.check("SIMPLEFLUID_COMPRESS_LTO_DEBUG_INFO" in output, "Unsupported opt-in reports the controlling option")

        # Exercise the real linker flag probe: this compiler driver refuses only
        # that flag and delegates every other operation to the selected compiler.
        wrapper = self.root / "reject-compression-compiler.py"
        wrapper.write_text("#!" + sys.executable + "\nimport os, sys\n"
                           "if any('--compress-debug-sections' in arg for arg in sys.argv[1:]):\n"
                           "    print('fixture: compression linker flag rejected', file=sys.stderr)\n"
                           "    sys.exit(1)\n"
                           f"os.execv({str(compiler)!r}, [{str(compiler)!r}] + sys.argv[1:])\n")
        wrapper.chmod(0o755)
        _, output = self.configure("unsupported-linker-flag", wrapper,
                                   ["-DFIXTURE_PLATFORM_CHECK=ON", "-DSIMPLEFLUID_SPLIT_DEBUG_INFO=OFF",
                                    "-DSIMPLEFLUID_COMPRESS_LTO_DEBUG_INFO=ON"], success=False)
        self.check("SIMPLEFLUID_COMPRESS_LTO_DEBUG_INFO" in output, "Rejected real linker probe reports the controlling option")

    def debugger(self, label, artifact):
        if not self.args.gdb:
            return
        source = SOURCE / "fixture.cc"
        line = next(i for i, text in enumerate(source.read_text().splitlines(), 1) if "SOURCE_BREAKPOINT" in text)
        output = self.command(label + "-gdb", [self.args.gdb, "--batch", "--nx", "--quiet",
                              "-ex", "set debuginfod enabled off", "-ex", f"file {artifact}",
                              "-ex", f"info line {source}:{line}", "-ex", f"break {source}:{line}"])
        self.check(bool(re.search(r"Breakpoint 1 at .*file .*fixture\.cc, line \d+", output)),
                   f"{label}: GDB resolves a source breakpoint")
        self.check(bool(re.search(r"Line \d+ of .*fixture\.cc.*(?:starts at address|is at address)", output)),
                   f"{label}: GDB reads the source line table")

    def inspect(self, toolchain, variant, directory, config, split, compression):
        label = f"{toolchain}-{variant}-{config}"
        self.command(label + "-build", [self.args.cmake, "--build", directory, "--config", config,
                                        "--parallel", self.args.parallel])
        commands = self.command(label + "-commands", [self.args.ninja, "-f", f"build-{config}.ninja", "-t", "commands"],
                                cwd=directory).splitlines()
        compile_commands = json.loads((directory / "compile_commands.json").read_text())
        for target in TARGETS:
            key = f"{label}/{target}"
            info = report(directory / "reports" / config / (target + ".txt"))
            ipo = expected_ipo(target, config)
            want_split = split and not ipo and config in DEBUG_CONFIGS
            want_compression = compression and ipo and config in DEBUG_CONFIGS and target in LINKED
            self.check(info["general_ipo"] == ("ON" if target in GENERAL_IPO else "OFF"),
                       key + ": general IPO setting preserved")
            self.check(info["debug_ipo"] == ({"ipo_debug_off": "OFF", "ipo_debug_on": "ON"}.get(target, "debug_ipo-NOTFOUND")),
                       key + ": Debug IPO override preserved")
            self.check(info["relwithdebinfo_ipo"] == ({"ipo_relwithdebinfo_off": "OFF", "ipo_relwithdebinfo_on": "ON"}.get(target, "relwithdebinfo_ipo-NOTFOUND")),
                       key + ": RelWithDebInfo IPO override preserved")
            self.check(("--compress-debug-sections=zlib" in info["link"]) == want_compression,
                       key + ": compression link policy")
            self.check(("-gsplit-dwarf" in info["compile"]) == want_split, key + ": split compile policy")
            marker = f"CMakeFiles/{target}.dir/{config}/"
            matching = [entry["command"] for entry in compile_commands if marker in entry["command"]]
            self.check(bool(matching), key + ": actual compile command recorded")
            for command in matching:
                self.check(("-flto" in command) == ipo, key + ": actual compile preserves LTO selection")
                self.check(("-gsplit-dwarf" in command) == want_split, key + ": actual compile split selection")
            dwo = list((directory / "CMakeFiles" / (target + ".dir") / config).rglob("*.dwo"))
            self.check(bool(dwo) == want_split, key + ": actual reusable dwo presence")
            self.check(all(path.stat().st_size > 0 for path in dwo), key + ": dwo files are nonempty")
            if target not in LINKED:
                self.check(not info["link"], key + ": archive/object has no debug link options")
                continue
            artifact = Path(info["artifact"])
            link_commands = [command for command in commands if re.search(r"(?:^|\s)-o\s+", command)
                             and artifact.name in command and " -c " not in command]
            self.check(bool(link_commands), key + ": actual link command recorded")
            for command in link_commands:
                self.check(("-flto" in command) == ipo, key + ": actual link preserves LTO selection")
                self.check(("--compress-debug-sections=zlib" in command) == want_compression,
                           key + ": actual link compression selection")
            sections = elf_debug_sections(artifact)
            self.summary["artifacts"][key] = {"path": str(artifact), "sections": sections}
            self.check(not any(name.startswith(".zdebug_") for name in sections), key + ": no legacy GNU zdebug sections")
            if want_compression:
                self.check(sections.get(".debug_info", {}).get("compressed", False),
                           key + ": debug_info has standard ELF zlib compression")
            else:
                self.check(not any(section["compressed"] for section in sections.values()),
                           key + ": no unsolicited compressed sections")
            if config in DEBUG_CONFIGS:
                self.check(".debug_info" in sections, key + ": debug information retained")
            if info["kind"] == "EXECUTABLE":
                output = self.command(label + "-" + target + "-run", [artifact])
                self.check(output.strip() == "debug-symbol-fixture=42", key + ": executable result remains 42")
                if config in DEBUG_CONFIGS:
                    self.debugger(label + "-" + target, artifact)
        imported = report(directory / "reports" / config / "fixture_imported.txt")
        self.check(imported["compile"] == "imported-compile-sentinel" and imported["link"] == "imported-link-sentinel",
                   label + ": imported target remains untouched")

    def run(self):
        if not self.args.gdb:
            self.summary["skips"].append("GDB unavailable: debug loading and source-breakpoint checks not run")
        for toolchain in self.args.toolchain:
            compiler = self.args.gcc_cxx if toolchain == "gcc" else self.args.clang_cxx
            compiler = shutil.which(compiler)
            if not compiler:
                raise RuntimeError(f"Requested compiler unavailable: {toolchain}")
            compiler = Path(compiler).absolute()
            self.command(toolchain + "-compiler-version", [compiler, "--version"])
            options = []
            if toolchain == "clang-lld":
                if not shutil.which("ld.lld"):
                    raise RuntimeError("Requested Clang/LLD validation requires ld.lld in PATH")
                options = ["-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld", "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
                           "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld"]
            for variant, split, compression in [("default", True, True), ("split-off", False, True), ("compression-off", True, False)]:
                overrides = []
                if not split:
                    overrides.append("-DSIMPLEFLUID_SPLIT_DEBUG_INFO=OFF")
                if not compression:
                    overrides.append("-DSIMPLEFLUID_COMPRESS_LTO_DEBUG_INFO=OFF")
                directory, _ = self.configure(toolchain + "-" + variant, compiler, options + overrides)
                configs = ["Debug", "RelWithDebInfo", "Release"] if variant == "default" else ["Debug", "RelWithDebInfo"]
                for config in configs:
                    self.inspect(toolchain, variant, directory, config, split, compression)
            if toolchain == self.args.toolchain[0]:
                self.negative_checks(compiler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--toolchain", choices=["gcc", "clang-lld"], action="append")
    parser.add_argument("--gcc-cxx", default="g++")
    parser.add_argument("--clang-cxx", default="clang++")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ninja", default="ninja")
    parser.add_argument("--gdb", default=shutil.which("gdb"))
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    args.toolchain = args.toolchain or ["gcc", "clang-lld"]
    runner = Runner(args)
    try:
        runner.run()
        runner.summary["status"] = "passed"
    except Exception as error:
        runner.summary.update(status="failed", error=str(error))
        print(str(error), file=sys.stderr)
    finally:
        (runner.root / "summary.json").write_text(json.dumps(runner.summary, indent=2) + "\n")
    print(f"{runner.summary['status']}: {len(runner.summary['checks'])} checks; {runner.root / 'summary.json'}")
    for skip in runner.summary["skips"]:
        print("SKIP: " + skip)
    return 0 if runner.summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
