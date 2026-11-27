#!/usr/bin/env python3
"""Check IPO, PCH reuse, and Linux debug artifacts without building Trilinos."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ("Debug", "Release", "RelWithDebInfo", "Audit")
FIXTURE = r'''
cmake_minimum_required(VERSION 3.21)
project(TestIpoPolicy LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(PROJECT_SOURCE_DIR "@ROOT@")
include("${PROJECT_SOURCE_DIR}/cmake/Options.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/targets/Helpers.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/targets/PrecompiledHeaders.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/testing/Registration.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/profiling/DebugSymbols.cmake")
if(SIMPLEFLUID_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_AUDIT ON)
endif()
if(NOT DEFINED SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME)
    set(SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME ON)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(ipo_linker_option -fuse-ld=lld)
    add_link_options("$<$<CONFIG:Release,RelWithDebInfo,Audit>:${ipo_linker_option}>")
endif()
simplefluid_configure_lto_debug_compression("${ipo_linker_option}")
set(SIMPLEFLUID_PCH_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/pch.hh")
set(SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET hidden)
set(SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN YES)
simplefluid_add_precompiled_headers()
add_library(component STATIC component.cpp)
simplefluid_configure_component(component)
add_library(shared_fixture SHARED shared.cpp)
simplefluid_configure_component(shared_fixture)
target_compile_options(shared_fixture PRIVATE -Werror=invalid-pch)
add_executable(application application.cpp)
target_link_libraries(application PRIVATE component shared_fixture)
target_compile_options(application PRIVATE -Werror=invalid-pch)
simplefluid_exe_reuse_precompiled_header(application)
if(BUILD_TESTING)
    add_executable(unit unit.cpp)
    target_link_libraries(unit PRIVATE component)
    target_compile_options(unit PRIVATE -Werror=invalid-pch)
    simplefluid_configure_test_executable(unit)
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/unit-$<CONFIG>.txt" CONTENT
        "ipo=$<BOOL:$<TARGET_PROPERTY:unit,INTERPROCEDURAL_OPTIMIZATION_$<UPPER_CASE:$<CONFIG>>>>\npch=$<TARGET_PROPERTY:unit,PRECOMPILE_HEADERS_REUSE_FROM>\n")
endif()
if(TARGET SimpleFluidTestPch)
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/test-pch-$<CONFIG>.txt" CONTENT
        "$<BOOL:$<TARGET_PROPERTY:SimpleFluidTestPch,INTERPROCEDURAL_OPTIMIZATION_$<UPPER_CASE:$<CONFIG>>>>")
endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/production-$<CONFIG>.txt" CONTENT
    "component=$<BOOL:$<TARGET_PROPERTY:component,INTERPROCEDURAL_OPTIMIZATION_$<UPPER_CASE:$<CONFIG>>>>\napplication=$<BOOL:$<TARGET_PROPERTY:application,INTERPROCEDURAL_OPTIMIZATION_$<UPPER_CASE:$<CONFIG>>>>\nshared=$<BOOL:$<TARGET_PROPERTY:shared_fixture,INTERPROCEDURAL_OPTIMIZATION_$<UPPER_CASE:$<CONFIG>>>>\npch=$<BOOL:$<TARGET_PROPERTY:SimpleFluidExecutablePch,INTERPROCEDURAL_OPTIMIZATION_$<UPPER_CASE:$<CONFIG>>>>\n")
simplefluid_finalize_debug_symbols("${CMAKE_CURRENT_SOURCE_DIR}" "${ipo_linker_option}")
'''

APPLICATION = '''extern "C" int shared_answer(int);
extern "C" [[gnu::noinline, gnu::used]] int application_answer(int value) {
    return shared_answer(value) + 1;
}
int answer();
int main(int argc, char **) {
    std::vector<int> values{answer(), application_answer(argc)};
    return values[0] + values[1] == argc + 14 ? 0 : 1;
}
'''
SHARED = '''extern "C" [[gnu::noinline, gnu::visibility("default")]] int shared_answer(int input) {
    volatile int value = input;
    std::vector<int> values{value, 6};
    return values[0] + values[1];
}
'''
UNIT = '''extern "C" [[gnu::noinline, gnu::used]] int unit_answer(int value) {
    return value + 5;
}
int answer();
int main(int argc, char **) {
    std::vector<int> values{answer(), unit_answer(argc)};
    return values[0] + values[1] == argc + 12 ? 0 : 1;
}
'''


def run(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=dict(os.environ, LC_ALL="C", DEBUGINFOD_URLS=""))
    if result.returncode:
        raise RuntimeError("Command failed: " + repr(command) + "\n" + result.stdout)
    return result.stdout


def has_lto(command):
    return bool(re.search(r"(?:^|\s)-flto(?:\b|=)", command))


def compile_command(entries, target, config, filename):
    commands = [entry["command"] for entry in entries
                if f"CMakeFiles/{target}.dir/{config}/" in entry["command"]
                and Path(entry["file"]).name == filename]
    assert len(commands) == 1, (target, config, filename, commands)
    return commands[0]


def link_command(output, config, target):
    commands = run(["ninja", "-C", str(output), "-f", f"build-{config}.ninja", "-t", "commands", target])
    return commands.splitlines()[-1]


def check_debug_flags(command, ipo, split, compressed):
    assert has_lto(command) == ipo, command
    assert ("-gsplit-dwarf" in command) == split, command
    assert ("-gz=zstd" in command) == compressed, command


def check_source_line(binary, symbol, filename):
    symbols = run(["readelf", "--symbols", "--wide", str(binary)])
    addresses = {match.group(1) for match in re.finditer(
        rf"^\s*\d+:\s+([0-9a-f]+)\s+\d+\s+FUNC\s+\w+\s+\w+\s+\d+\s+{symbol}$", symbols, re.M)}
    assert len(addresses) == 1, (binary, symbol, addresses)
    address = "0x" + addresses.pop()
    results = []
    for tool in ("addr2line", "llvm-addr2line", "eu-addr2line", "gdb"):
        if not shutil.which(tool):
            continue
        if tool == "gdb":
            command = [tool, "-nx", "-batch", "-ex", f"info line *{address}", str(binary)]
            pattern = rf'Line [1-9]\d* of ".*{re.escape(filename)}"'
        else:
            command = [tool, "-f", "-e", str(binary), address]
            pattern = rf"{re.escape(filename)}:[1-9]\d*\b"
        try:
            result = run(command)
        except RuntimeError as error:
            results.append((tool, str(error)))
            continue
        if re.search(pattern, result):
            return tool
        results.append((tool, result))
    raise AssertionError((binary, symbol, filename, results))


def check_split_debug(output, config):
    unit = output / config / "unit"
    sidecars = list((output / "CMakeFiles" / "unit.dir" / config).rglob("*.dwo"))
    assert len(sidecars) == 1, sidecars
    sidecar = sidecars[0]
    skeleton = run(["readelf", "--debug-dump=info", "--debug-dump=no-follow-links", "--dwarf-depth=1", str(unit)])
    assert re.search(rf"DW_AT_(?:GNU_)?dwo_name\b.*{re.escape(sidecar.name)}$", skeleton, re.M), skeleton
    links = run(["readelf", "--debug-dump=links", str(unit)])
    assert f"Found separate debug object file: {sidecar}" in links, links
    sections = run(["readelf", "--section-headers", "--wide", str(sidecar)])
    assert ".debug_info.dwo" in sections, sections
    return check_source_line(unit, "unit_answer", "unit.cpp")


def check_compression(binary, compressed, symbol, filename):
    sections = run(["readelf", "--section-details", "--wide", str(binary)])
    debug_sections = re.split(r"(?m)^\s*\[\s*\d+\]\s+", sections)
    info = next((section for section in debug_sections if section.startswith(".debug_info\n")), "")
    assert info, sections
    if compressed:
        assert "COMPRESSED" in info and "ZSTD" in info, info
    else:
        for section in debug_sections:
            if section.startswith((".debug_", ".zdebug_")):
                assert not section.startswith(".zdebug_") and "COMPRESSED" not in section, section
    check_source_line(binary, symbol, filename)
    return binary.stat().st_size


def build_and_run(output, config):
    run(["cmake", "--build", str(output), "--config", config,
         "--target", "unit", "application", "--parallel", "2"])
    run([str(output / config / "unit")])
    run([str(output / config / "application")])


def check(compiler, require_zstd):
    with tempfile.TemporaryDirectory(prefix="simplefluid-test-ipo-") as directory:
        source = Path(directory)
        (source / "CMakeLists.txt").write_text(FIXTURE.replace("@ROOT@", ROOT.as_posix()))
        (source / "pch.hh").write_text("#include <vector>\n")
        (source / "component.cpp").write_text("int answer() { return 7; }\n")
        (source / "application.cpp").write_text(APPLICATION)
        (source / "shared.cpp").write_text(SHARED)
        (source / "unit.cpp").write_text(UNIT)
        cases = [
            ("default", [], False, True, True, False),
            ("off", ["-DSIMPLEFLUID_DISABLE_TEST_LTO=OFF"], False, True, True, False),
            ("on", ["-DSIMPLEFLUID_DISABLE_TEST_LTO=ON"], True, True, True, True),
            ("global-off", ["-DSIMPLEFLUID_DISABLE_TEST_LTO=ON", "-DSIMPLEFLUID_ENABLE_LTO=OFF"], True, False, True, False),
            ("no-tests", ["-DSIMPLEFLUID_DISABLE_TEST_LTO=ON", "-DBUILD_TESTING=OFF"], True, True, False, False),
        ]
        linux = sys.platform.startswith("linux")
        if linux:
            compression_off = "-DSIMPLEFLUID_COMPRESS_LTO_DEBUG_INFO=OFF"
            debug_ipo = "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG=ON"
            cases += [
                ("compression-off", [compression_off], False, True, True, False),
                ("debug-ipo", [debug_ipo], False, True, True, False),
                ("debug-ipo-compression-off", [debug_ipo, compression_off], False, True, True, False),
            ]
        is_gnu = "clang" not in run([compiler, "--version"]).lower()
        if is_gnu and linux:
            cases += [
                ("gnu-native-off", ["-DSIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME=OFF"], False, True, True, False),
                ("gnu-native-on", ["-DSIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME=OFF", "-DSIMPLEFLUID_DISABLE_TEST_LTO=ON"], True, True, True, True),
            ]
        artifact_sizes = {}
        zstd_supported = False
        for name, options, disabled, lto, testing, build in cases:
            output = source / name
            run(["cmake", "-S", str(source), "-B", str(output), "-G", "Ninja Multi-Config",
                 "-DCMAKE_CXX_COMPILER=" + compiler, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 "-DCMAKE_CONFIGURATION_TYPES=" + ";".join(CONFIGS), *options])
            cache = (output / "CMakeCache.txt").read_text()
            if name == "default" and linux:
                zstd_supported = bool(re.search(
                    r"^SIMPLEFLUID_CXX_HAS_GZ_ZSTD:INTERNAL=1$", cache, re.M
                ) and re.search(r"^SIMPLEFLUID_CXX_HAS_ZSTD_DEBUG_LINK_FLAGS\w*:INTERNAL=1$", cache, re.M))
                if not zstd_supported:
                    message = f"{compiler}: zstd debug compression requires compiler and linker support"
                    if require_zstd:
                        raise AssertionError(message)
                    print(f"{message}. Skipping zstd artifact comparison.")
            compression_enabled = zstd_supported and not name.endswith("compression-off")
            assert f"SIMPLEFLUID_COMPRESS_LTO_DEBUG_INFO:BOOL={'ON' if compression_enabled else 'OFF'}" in cache
            entries = json.loads((output / "compile_commands.json").read_text())
            native_exception = name.startswith("gnu-native")
            for config in CONFIGS:
                production_ipo = lto and (config != "Debug" or name.startswith("debug-ipo"))
                symbol_config = linux and config in ("Debug", "RelWithDebInfo")
                production_compression = symbol_config and production_ipo and compression_enabled
                expected = int(production_ipo)
                assert (output / f"production-{config}.txt").read_text() == (
                    f"component={expected}\napplication={expected}\nshared={expected}\npch={expected}\n"), (name, config)
                for target, filename, pch in (
                    ("application", "application.cpp", "SimpleFluidExecutablePch"),
                    ("shared_fixture", "shared.cpp", "SimpleFluidSharedPch"),
                ):
                    consumer = compile_command(entries, target, config, filename)
                    provider_command = compile_command(entries, pch, config, "cmake_pch.hxx.cxx")
                    for command in (consumer, provider_command):
                        check_debug_flags(command, production_ipo, symbol_config and not production_ipo,
                                          production_compression)
                    assert f"{pch}.dir/{config}/cmake_pch.hxx" in consumer, consumer
                    assert "-Werror=invalid-pch" in consumer, consumer
                    link = link_command(output, config, target)
                    assert ("--compress-debug-sections=zstd" in link) == production_compression, link
                    if linux and not is_gnu and config == "Debug":
                        assert ("-fuse-ld=lld" in link) == production_compression, link
                assert (output / f"test-pch-{config}.txt").exists() == (disabled and testing)
                if not testing:
                    assert not (output / f"unit-{config}.txt").exists()
                    continue
                test_ipo = production_ipo and not disabled and not (
                    native_exception and config in ("Release", "RelWithDebInfo"))
                provider = "SimpleFluidTestPch" if disabled else "SimpleFluidExecutablePch"
                assert (output / f"unit-{config}.txt").read_text() == f"ipo={int(test_ipo)}\npch={provider}\n"
                if disabled:
                    assert (output / f"test-pch-{config}.txt").read_text() == "0"
                consumer = compile_command(entries, "unit", config, "unit.cpp")
                link = link_command(output, config, "unit")
                test_compression = test_ipo and production_compression
                check_debug_flags(consumer, test_ipo, symbol_config and not test_ipo, test_compression)
                assert has_lto(link) == test_ipo, (name, config, link)
                assert ("--compress-debug-sections=zstd" in link) == test_compression, link
                if linux and not is_gnu and config == "Debug":
                    assert ("-fuse-ld=lld" in link) == test_compression, link
                assert f"{provider}.dir/{config}/cmake_pch.hxx" in consumer, consumer
                assert "-Werror=invalid-pch" in consumer, consumer
                if native_exception and config in ("Release", "RelWithDebInfo"):
                    assert "-fno-lto" in consumer and "-fno-use-linker-plugin" in link
                # The GNU native archive exception can reuse the production IPO PCH.
                if not native_exception or disabled or config not in ("Release", "RelWithDebInfo"):
                    provider_command = compile_command(entries, provider, config, "cmake_pch.hxx.cxx")
                    check_debug_flags(provider_command, test_ipo, symbol_config and not test_ipo, test_compression)
            verify_compression = zstd_supported and name in ("default", "compression-off", "debug-ipo")
            artifact_config = "Debug" if name == "debug-ipo" else "RelWithDebInfo"
            build_configs = CONFIGS[:-1] if build else ()
            if name == "gnu-native-off" or verify_compression:
                build_configs = (artifact_config,)
            for config in build_configs:
                build_and_run(output, config)
            if verify_compression:
                artifact_sizes[name] = [
                    check_compression(output / artifact_config / artifact, compression_enabled, symbol, filename)
                    for artifact, symbol, filename in (
                        ("application", "application_answer", "application.cpp"),
                        ("libshared_fixture.so", "shared_answer", "shared.cpp"),
                    )
                ]
            if linux and name == "on":
                symbolizer = check_split_debug(output, "RelWithDebInfo")
                print(f"{compiler}: RelWithDebInfo split DWARF skeleton, sidecar, and {symbolizer} source line verified")
            build_result = ". PCH builds and consumers passed." if build_configs else "."
            print(f"{compiler}: {name} policy checks passed{build_result}")

        if zstd_supported:
            for artifact, compressed, plain in zip(("application", "shared library"),
                                                   artifact_sizes["default"], artifact_sizes["compression-off"]):
                assert compressed < plain, (artifact, compressed, plain)
                print(f"{compiler}: RelWithDebInfo {artifact} with/without zstd: {compressed}/{plain} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--require-zstd", action="store_true", help="fail if the Linux zstd probes do not pass")
    args = parser.parse_args()
    if not shutil.which(args.compiler):
        parser.error("compiler not found: " + args.compiler)
    if args.require_zstd and not sys.platform.startswith("linux"):
        parser.error("--require-zstd requires Linux")
    check(args.compiler, args.require_zstd)
