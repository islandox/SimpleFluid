# Select debug information without changing optimization or numerical policy.
include_guard(GLOBAL)

# Keep each translation unit's DWARF beside its object, so static-library
# consumers reuse the same debug data instead of copying it into every binary.
set(_simplefluid_split_debug_default OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
    set(_simplefluid_split_debug_default ON)
endif()
option(SIMPLEFLUID_SPLIT_DEBUG_INFO
    "Keep Debug/RelWithDebInfo symbols in reusable .dwo files (Linux GCC/Clang)"
    ${_simplefluid_split_debug_default})
unset(_simplefluid_split_debug_default)
if(SIMPLEFLUID_SPLIT_DEBUG_INFO)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "SIMPLEFLUID_SPLIT_DEBUG_INFO requires Linux GCC or Clang")
    endif()
    include(CheckCompilerFlag)
    include(CheckLinkerFlag)
    foreach(language IN ITEMS C CXX)
        if(CMAKE_${language}_COMPILER_LOADED)
            if(NOT CMAKE_${language}_COMPILER_ID MATCHES "^(GNU|Clang)$")
                message(FATAL_ERROR "SIMPLEFLUID_SPLIT_DEBUG_INFO requires GCC or Clang (${language})")
            endif()
            check_compiler_flag(${language} "-gsplit-dwarf" SIMPLEFLUID_${language}_HAS_SPLIT_DWARF)
            if(NOT SIMPLEFLUID_${language}_HAS_SPLIT_DWARF)
                message(FATAL_ERROR "SIMPLEFLUID_SPLIT_DEBUG_INFO requires -gsplit-dwarf support (${language})")
            endif()
            check_linker_flag(${language}
                "LINKER:-T,${CMAKE_CURRENT_LIST_DIR}/../linkage/SplitDebugSections.ld"
                SIMPLEFLUID_${language}_HAS_SPLIT_DEBUG_LINKER_SCRIPT)
            if(NOT SIMPLEFLUID_${language}_HAS_SPLIT_DEBUG_LINKER_SCRIPT)
                message(FATAL_ERROR
                    "SIMPLEFLUID_SPLIT_DEBUG_INFO requires GNU ld.bfd or LLD linker-script support (${language}); "
                    "select a compatible linker or set SIMPLEFLUID_SPLIT_DEBUG_INFO=OFF")
            endif()
        endif()
    endforeach()
    unset(language)
endif()

set(_simplefluid_debug_symbol_modules
    CORE IO PARALLEL MESH PLIC SWEPT_VOLUME FIELDS FVM EQUATIONS SOLVERS
    MATERIALS IF97 NOX EXPERIMENTAL_VOF TESTS BENCHMARKS EXAMPLES)
foreach(module IN ITEMS "" ${_simplefluid_debug_symbol_modules})
    set(option_name SIMPLEFLUID_DEBUG_SYMBOLS)
    if(NOT module STREQUAL "")
        string(APPEND option_name "_${module}")
    endif()
    set(${option_name} INHERIT CACHE STRING
        "Debug/RelWithDebInfo debug symbols: INHERIT, ON (-g), or OFF (-g0)")
    set_property(CACHE ${option_name} PROPERTY STRINGS INHERIT ON OFF)
    if(NOT "${${option_name}}" MATCHES "^(INHERIT|ON|OFF)$")
        message(FATAL_ERROR "${option_name} must be INHERIT, ON, or OFF; got '${${option_name}}'")
    endif()
    if(NOT "${${option_name}}" STREQUAL "INHERIT")
        foreach(language IN ITEMS C CXX)
            if(CMAKE_${language}_COMPILER_LOADED AND
               NOT CMAKE_${language}_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
                message(FATAL_ERROR "${option_name} overrides require GCC or Clang (${language})")
            endif()
        endforeach()
    endif()
endforeach()
unset(_simplefluid_debug_symbol_modules)
unset(module)
unset(option_name)

function(simplefluid_debug_symbols_target target_name modules)
    set_property(TARGET ${target_name} PROPERTY SIMPLEFLUID_DEBUG_SYMBOL_MODULES "${modules}")
endfunction()

# Register logical ownership before sibling directories consume source lists.
# The ordered module chain is independent of the physical archive/DSO owner.
function(simplefluid_debug_symbols_sources modules)
    foreach(source IN LISTS ARGN)
        get_filename_component(source "${source}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        string(SHA256 source_key "${source}")
        set_property(GLOBAL PROPERTY SIMPLEFLUID_DEBUG_SYMBOL_SOURCE_${source_key} "${modules}")
    endforeach()
endfunction()

function(_simplefluid_debug_symbol_flag state result)
    if(state STREQUAL "ON")
        set(${result} -g PARENT_SCOPE)
    else()
        set(${result} -g0 PARENT_SCOPE)
    endif()
endfunction()

# Call once after all component numerical/source properties have been applied.
# Source properties must be written in the target's directory scope, including
# sources exported from sibling directories (IO, parallel and equations).
function(simplefluid_finalize_debug_symbols directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target_name IN LISTS targets)
        get_target_property(target_type ${target_name} TYPE)
        get_target_property(imported ${target_name} IMPORTED)
        if(imported OR NOT target_type MATCHES "^(STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY|EXECUTABLE)$")
            continue()
        endif()
        if(SIMPLEFLUID_SPLIT_DEBUG_INFO)
            # GCC LTO does not support reusable split DWARF. Clang LTO emits
            # per-link bundles instead. Preserve IPO and its embedded symbols.
            foreach(config IN ITEMS Debug RelWithDebInfo)
                string(TOUPPER "${config}" config_upper)
                get_property(ipo_config_set TARGET ${target_name}
                    PROPERTY INTERPROCEDURAL_OPTIMIZATION_${config_upper} SET)
                if(ipo_config_set)
                    get_property(ipo TARGET ${target_name}
                        PROPERTY INTERPROCEDURAL_OPTIMIZATION_${config_upper})
                else()
                    get_property(ipo TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION)
                endif()
                if(NOT ipo)
                    target_compile_options(${target_name} PRIVATE
                        "$<$<AND:$<CONFIG:${config}>,$<COMPILE_LANGUAGE:C,CXX>>:-gsplit-dwarf>")
                    if(target_type MATCHES "^(SHARED_LIBRARY|MODULE_LIBRARY|EXECUTABLE)$")
                        set(split_debug_script
                            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../linkage/SplitDebugSections.ld")
                        target_link_options(${target_name} PRIVATE
                            "$<$<CONFIG:${config}>:LINKER:-T,${split_debug_script}>")
                        set_property(TARGET ${target_name} APPEND PROPERTY
                            LINK_DEPENDS "$<$<CONFIG:${config}>:${split_debug_script}>")
                    endif()
                endif()
            endforeach()
        endif()
        if(NOT SIMPLEFLUID_DEBUG_SYMBOLS STREQUAL "INHERIT")
            _simplefluid_debug_symbol_flag("${SIMPLEFLUID_DEBUG_SYMBOLS}" flag)
            target_compile_options(${target_name} PRIVATE
                "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:C,CXX>>:${flag}>")
        endif()

        # Shared PCH providers follow the global setting, matching ordinary
        # consumers. Only sources differing from it need to bypass PCH reuse.
        get_target_property(pch ${target_name} PRECOMPILE_HEADERS)
        if(pch)
            continue()
        endif()
        get_target_property(target_modules ${target_name} SIMPLEFLUID_DEBUG_SYMBOL_MODULES)
        if(NOT target_modules)
            if(directory MATCHES "/unitTests$")
                set(target_modules TESTS)
            elseif(directory STREQUAL "${PROJECT_SOURCE_DIR}/src/examples")
                set(target_modules EXAMPLES)
            elseif(directory STREQUAL "${PROJECT_SOURCE_DIR}/src/benchmarks")
                set(target_modules BENCHMARKS)
            else()
                # Project runtime/ABI bridge objects and the optional main.
                set(target_modules CORE)
            endif()
        endif()
        get_target_property(sources ${target_name} SOURCES)
        foreach(source IN LISTS sources)
            # Headers, implementation fragments and target-object generator
            # expressions do not compile as independent translation units.
            if(source MATCHES "\\$<" OR NOT source MATCHES "\\.(c|cc|cpp|cxx|C)$")
                continue()
            endif()
            get_filename_component(source "${source}" ABSOLUTE BASE_DIR "${directory}")
            string(SHA256 source_key "${source}")
            get_property(source_modules GLOBAL PROPERTY SIMPLEFLUID_DEBUG_SYMBOL_SOURCE_${source_key})
            set(state "${SIMPLEFLUID_DEBUG_SYMBOLS}")
            foreach(module IN LISTS source_modules target_modules)
                if(NOT SIMPLEFLUID_DEBUG_SYMBOLS_${module} STREQUAL "INHERIT")
                    set(state "${SIMPLEFLUID_DEBUG_SYMBOLS_${module}}")
                    break()
                endif()
            endforeach()
            if(NOT state STREQUAL SIMPLEFLUID_DEBUG_SYMBOLS)
                _simplefluid_debug_symbol_flag("${state}" flag)
                set_property(SOURCE "${source}" TARGET_DIRECTORY ${target_name}
                    APPEND PROPERTY COMPILE_OPTIONS "$<$<CONFIG:Debug,RelWithDebInfo>:${flag}>")
                # CMake 3.21 cannot make this source property config-specific.
                set_property(SOURCE "${source}" TARGET_DIRECTORY ${target_name}
                    PROPERTY SKIP_PRECOMPILE_HEADERS TRUE)
            endif()
        endforeach()
    endforeach()
    get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(subdirectory IN LISTS subdirectories)
        simplefluid_finalize_debug_symbols("${subdirectory}")
    endforeach()
endfunction()
