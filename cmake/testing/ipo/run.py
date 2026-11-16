#!/usr/bin/env python3
"""Check unit-test IPO policy with real CMake/PCH builds and no Trilinos dependency."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
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
    add_link_options(-fuse-ld=lld)
endif()
set(SIMPLEFLUID_PCH_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/pch.hh")
set(SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET hidden)
set(SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN YES)
simplefluid_add_precompiled_headers()
add_library(component STATIC component.cpp)
simplefluid_configure_component(component)
add_executable(application application.cpp)
target_link_libraries(application PRIVATE component)
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
    "component=$<BOOL:$<TARGET_PROPERTY:component,INTERPROCEDURAL_OPTIMIZATION_$<UPPER_CASE:$<CONFIG>>>>\napplication=$<BOOL:$<TARGET_PROPERTY:application,INTERPROCEDURAL_OPTIMIZATION_$<UPPER_CASE:$<CONFIG>>>>\npch=$<BOOL:$<TARGET_PROPERTY:SimpleFluidExecutablePch,INTERPROCEDURAL_OPTIMIZATION_$<UPPER_CASE:$<CONFIG>>>>\n")
simplefluid_finalize_debug_symbols("${CMAKE_CURRENT_SOURCE_DIR}")
'''


def run(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Command failed: " + repr(command) + "\n" + result.stdout)
    return result.stdout


def has_lto(command):
    return bool(re.search(r"(?:^|\s)-flto(?:\b|=)", command))


def check(compiler):
    with tempfile.TemporaryDirectory(prefix="simplefluid-test-ipo-") as directory:
        source = Path(directory)
        (source / "CMakeLists.txt").write_text(FIXTURE.replace("@ROOT@", ROOT.as_posix()))
        (source / "pch.hh").write_text("#include <vector>\n")
        (source / "component.cpp").write_text("int answer() { return 7; }\n")
        (source / "application.cpp").write_text("int answer(); int main() { return answer() == 7 ? 0 : 1; }\n")
        (source / "unit.cpp").write_text(
            "int answer(); int main() { std::vector<int> v{answer(), 5}; return v[0] + v[1] == 12 ? 0 : 1; }\n")
        cases = [
            ("default", [], False, True, True, False),
            ("off", ["-DSIMPLEFLUID_DISABLE_TEST_LTO=OFF"], False, True, True, False),
            ("on", ["-DSIMPLEFLUID_DISABLE_TEST_LTO=ON"], True, True, True, True),
            ("global-off", ["-DSIMPLEFLUID_DISABLE_TEST_LTO=ON", "-DSIMPLEFLUID_ENABLE_LTO=OFF"], True, False, True, False),
            ("no-tests", ["-DSIMPLEFLUID_DISABLE_TEST_LTO=ON", "-DBUILD_TESTING=OFF"], True, True, False, False),
        ]
        is_gnu = "clang" not in run([compiler, "--version"]).lower()
        if is_gnu and sys.platform.startswith("linux"):
            cases += [
                ("gnu-native-off", ["-DSIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME=OFF"], False, True, True, False),
                ("gnu-native-on", ["-DSIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME=OFF", "-DSIMPLEFLUID_DISABLE_TEST_LTO=ON"], True, True, True, True),
            ]
        for name, options, disabled, lto, testing, build in cases:
            output = source / name
            run(["cmake", "-S", str(source), "-B", str(output), "-G", "Ninja Multi-Config",
                 "-DCMAKE_CXX_COMPILER=" + compiler, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 "-DCMAKE_CONFIGURATION_TYPES=Debug;Release;RelWithDebInfo;Audit", *options])
            native_exception = name.startswith("gnu-native")
            for config in ("Debug", "Release", "RelWithDebInfo", "Audit"):
                production_ipo = lto and config != "Debug"
                expected = int(production_ipo)
                assert (output / f"production-{config}.txt").read_text() == (
                    f"component={expected}\napplication={expected}\npch={expected}\n"), (name, config)
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
                commands = run(["ninja", "-C", str(output), "-f", f"build-{config}.ninja", "-t", "commands", "unit"])
                compile_command = next(line for line in commands.splitlines() if "unit.cpp" in line and " -c " in line)
                link_command = commands.splitlines()[-1]
                assert has_lto(compile_command) == bool(test_ipo), (name, config, compile_command)
                assert has_lto(link_command) == bool(test_ipo), (name, config, link_command)
                if native_exception and config in ("Release", "RelWithDebInfo"):
                    assert "-fno-lto" in compile_command and "-fno-use-linker-plugin" in link_command
            if disabled and testing and lto and sys.platform.startswith("linux"):
                entries = json.loads((output / "compile_commands.json").read_text())
                test_pch = [entry["command"] for entry in entries
                            if "SimpleFluidTestPch.dir/RelWithDebInfo/" in entry["command"]]
                assert test_pch and all("-gsplit-dwarf" in command for command in test_pch)
                assert all(not has_lto(command) for command in test_pch)
            if build:
                for config in ("Debug", "Release", "RelWithDebInfo"):
                    run(["cmake", "--build", str(output), "--config", config, "--target", "unit", "application", "--parallel", "2"])
                    run([str(output / config / "unit")])
                    run([str(output / config / "application")])
            print(f"{compiler}: {name} policy checks passed" + ("; PCH builds and consumers passed" if build else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    args = parser.parse_args()
    if not shutil.which(args.compiler):
        parser.error("compiler not found: " + args.compiler)
    check(args.compiler)
