cmake_minimum_required(VERSION 3.21)
project(SimpleFluidDebugSymbolsFixture LANGUAGES CXX)

# This project deliberately has no SimpleFluid/Trilinos/MPI dependency.
get_filename_component(SIMPLEFLUID_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include("${SIMPLEFLUID_SOURCE_DIR}/cmake/profiling/DebugSymbols.cmake")

if(FIXTURE_PLATFORM_CHECK)
    file(WRITE "${CMAKE_BINARY_DIR}/platform-options.txt"
        "compress=${SIMPLEFLUID_COMPRESS_LTO_DEBUG_INFO}\nsplit=${SIMPLEFLUID_SPLIT_DEBUG_INFO}\n")
    return()
endif()

# The actual fixture links verify IPO with the requested linker. A separate
# CheckIPOSupported subproject would not inherit linker flags on older CMake.
foreach(target IN ITEMS ipo_all ipo_debug_off ipo_debug_on ipo_relwithdebinfo_off ipo_relwithdebinfo_on plain)
    add_executable(${target} fixture.cc)
    target_compile_definitions(${target} PRIVATE FIXTURE_EXECUTABLE=1)
endforeach()
add_library(ipo_shared SHARED fixture.cc)
add_library(ipo_module MODULE fixture.cc)
add_library(ipo_static STATIC fixture.cc)
add_library(ipo_object OBJECT fixture.cc)
add_library(plain_static STATIC fixture.cc)
add_library(plain_object OBJECT fixture.cc)

set(ipo_targets ipo_all ipo_debug_off ipo_relwithdebinfo_off ipo_shared ipo_module ipo_static ipo_object)
set(nonipo_targets ipo_debug_on ipo_relwithdebinfo_on plain plain_static plain_object)
set_property(TARGET ${ipo_targets} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
set_property(TARGET ${nonipo_targets} PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
set_property(TARGET ipo_debug_off PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)
set_property(TARGET ipo_debug_on PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUG ON)
set_property(TARGET ipo_relwithdebinfo_off PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO OFF)
set_property(TARGET ipo_relwithdebinfo_on PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)

# Imported targets must never acquire private options from the finalizer.
add_library(fixture_imported STATIC IMPORTED)
set_target_properties(fixture_imported PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/unused-imported.a"
    INTERPROCEDURAL_OPTIMIZATION ON
    COMPILE_OPTIONS "imported-compile-sentinel"
    LINK_OPTIONS "imported-link-sentinel")

simplefluid_finalize_debug_symbols("${CMAKE_CURRENT_SOURCE_DIR}")

get_property(all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(APPEND all_targets fixture_imported)
foreach(target IN LISTS all_targets)
    get_target_property(kind ${target} TYPE)
    get_target_property(general_ipo ${target} INTERPROCEDURAL_OPTIMIZATION)
    get_target_property(debug_ipo ${target} INTERPROCEDURAL_OPTIMIZATION_DEBUG)
    get_target_property(relwithdebinfo_ipo ${target} INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO)
    set(artifact "")
    if(NOT kind STREQUAL "OBJECT_LIBRARY" AND NOT target STREQUAL "fixture_imported")
        set(artifact "$<TARGET_FILE:${target}>")
    endif()
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/reports/$<CONFIG>/${target}.txt" CONTENT
        "kind=${kind}\ngeneral_ipo=${general_ipo}\ndebug_ipo=${debug_ipo}\nrelwithdebinfo_ipo=${relwithdebinfo_ipo}\nartifact=${artifact}\ncompile=$<JOIN:$<TARGET_GENEX_EVAL:${target},$<TARGET_PROPERTY:${target},COMPILE_OPTIONS>>,|>\nlink=$<JOIN:$<TARGET_GENEX_EVAL:${target},$<TARGET_PROPERTY:${target},LINK_OPTIONS>>,|>\n")
endforeach()
