set(SIMPLEFLUID_PROFILE_GUIDED_OPTIMIZATION_DIR
    "${CMAKE_CURRENT_LIST_DIR}")

set(SIMPLEFLUID_PGO_PHASE "OFF" CACHE STRING
    "Profile-guided optimization phase: OFF, GENERATE, or USE")
set_property(CACHE SIMPLEFLUID_PGO_PHASE
             PROPERTY STRINGS OFF GENERATE USE)
set(SIMPLEFLUID_PGO_DIR "${CMAKE_BINARY_DIR}/pgo" CACHE PATH
    "Compiler-specific profile data directory")
set(SIMPLEFLUID_PGO_CONFIG "RelWithDebInfo" CACHE STRING
    "Single optimized configuration trained by PGO")
set_property(CACHE SIMPLEFLUID_PGO_CONFIG
             PROPERTY STRINGS Release RelWithDebInfo)
set(SIMPLEFLUID_PGO_WORKLOAD "natural_convection_shiri" CACHE STRING
    "Production workload represented by the profile data")
set_property(CACHE SIMPLEFLUID_PGO_WORKLOAD
             PROPERTY STRINGS natural_convection_shiri)

macro(simplefluid_configure_profile_guided_optimization)
    string(TOUPPER "${SIMPLEFLUID_PGO_PHASE}"
           SIMPLEFLUID_PGO_PHASE_NORMALIZED)
    set(SIMPLEFLUID_PGO_PHASES OFF GENERATE USE)
    if(NOT SIMPLEFLUID_PGO_PHASE_NORMALIZED IN_LIST SIMPLEFLUID_PGO_PHASES)
        message(FATAL_ERROR
            "SIMPLEFLUID_PGO_PHASE must be OFF, GENERATE, or USE")
    endif()

    set(SIMPLEFLUID_PGO_CONFIGS Release RelWithDebInfo)
    if(NOT SIMPLEFLUID_PGO_CONFIG IN_LIST SIMPLEFLUID_PGO_CONFIGS)
        message(FATAL_ERROR
            "SIMPLEFLUID_PGO_CONFIG must be Release or RelWithDebInfo")
    endif()
    if(NOT SIMPLEFLUID_PGO_WORKLOAD STREQUAL "natural_convection_shiri")
        message(FATAL_ERROR
            "The supported PGO workload is natural_convection_shiri")
    endif()

    if(NOT SIMPLEFLUID_PGO_PHASE_NORMALIZED STREQUAL "OFF")
        if(SIMPLEFLUID_ENABLE_COVERAGE)
            message(FATAL_ERROR
                "SIMPLEFLUID_PGO_PHASE and SIMPLEFLUID_ENABLE_COVERAGE "
                "cannot be enabled together")
        endif()

        get_filename_component(
            SIMPLEFLUID_PGO_DIR "${SIMPLEFLUID_PGO_DIR}"
            ABSOLUTE BASE_DIR "${CMAKE_BINARY_DIR}")
        file(MAKE_DIRECTORY "${SIMPLEFLUID_PGO_DIR}")
        set(SIMPLEFLUID_PGO_CONFIG_EXPRESSION
            "$<CONFIG:${SIMPLEFLUID_PGO_CONFIG}>")

        if(CMAKE_CONFIGURATION_TYPES)
            if(NOT SIMPLEFLUID_PGO_CONFIG IN_LIST CMAKE_CONFIGURATION_TYPES)
                message(FATAL_ERROR
                    "SIMPLEFLUID_PGO_CONFIG=${SIMPLEFLUID_PGO_CONFIG} is not "
                    "available from this multi-config generator")
            endif()
        elseif(NOT CMAKE_BUILD_TYPE
               OR NOT CMAKE_BUILD_TYPE STREQUAL SIMPLEFLUID_PGO_CONFIG)
            message(FATAL_ERROR
                "A single-config PGO build requires "
                "CMAKE_BUILD_TYPE=${SIMPLEFLUID_PGO_CONFIG}")
        endif()

        # Bind profile data to the compiler, configuration, production source
        # closure, and declared workload. Generate builds may accumulate
        # multiple runs of the same compatible workload, but cannot silently
        # adopt data produced by another source tree or configuration.
        file(GLOB_RECURSE SIMPLEFLUID_PGO_SOURCE_INPUTS
            CONFIGURE_DEPENDS
            RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.tcc"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cppm"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.hh"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.map"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.json.in")
        list(FILTER SIMPLEFLUID_PGO_SOURCE_INPUTS
             EXCLUDE REGEX "/unitTests/")
        list(FILTER SIMPLEFLUID_PGO_SOURCE_INPUTS
             EXCLUDE REGEX "^src/benchmarks/")
        list(FILTER SIMPLEFLUID_PGO_SOURCE_INPUTS
             EXCLUDE REGEX "^src/examples/")
        list(APPEND SIMPLEFLUID_PGO_SOURCE_INPUTS
            CMakeLists.txt
            CMakePresets.json
            src/CMakeLists.txt
            src/examples/natural_convection_shiri.cc)
        list(REMOVE_DUPLICATES SIMPLEFLUID_PGO_SOURCE_INPUTS)
        list(SORT SIMPLEFLUID_PGO_SOURCE_INPUTS)
        set(SIMPLEFLUID_PGO_SOURCE_MANIFEST)
        foreach(SIMPLEFLUID_PGO_SOURCE IN LISTS SIMPLEFLUID_PGO_SOURCE_INPUTS)
            file(SHA256
                "${CMAKE_CURRENT_SOURCE_DIR}/${SIMPLEFLUID_PGO_SOURCE}"
                SIMPLEFLUID_PGO_SOURCE_HASH)
            string(APPEND SIMPLEFLUID_PGO_SOURCE_MANIFEST
                "${SIMPLEFLUID_PGO_SOURCE}:${SIMPLEFLUID_PGO_SOURCE_HASH}\n")
        endforeach()
        string(SHA256 SIMPLEFLUID_PGO_SOURCE_FINGERPRINT
               "${SIMPLEFLUID_PGO_SOURCE_MANIFEST}")

        get_filename_component(
            SIMPLEFLUID_PGO_COMPILER_PATH "${CMAKE_CXX_COMPILER}" REALPATH)
        string(TOUPPER "${SIMPLEFLUID_PGO_CONFIG}"
               SIMPLEFLUID_PGO_CONFIG_UPPER)
        set(SIMPLEFLUID_PGO_CONFIG_FLAGS_VARIABLE
            "CMAKE_CXX_FLAGS_${SIMPLEFLUID_PGO_CONFIG_UPPER}")
        set(SIMPLEFLUID_PGO_EXPECTED_MANIFEST
            "format=1\n"
            "workload=${SIMPLEFLUID_PGO_WORKLOAD}\n"
            "compiler_id=${CMAKE_CXX_COMPILER_ID}\n"
            "compiler_version=${CMAKE_CXX_COMPILER_VERSION}\n"
            "compiler_path=${SIMPLEFLUID_PGO_COMPILER_PATH}\n"
            "configuration=${SIMPLEFLUID_PGO_CONFIG}\n"
            "source_fingerprint=${SIMPLEFLUID_PGO_SOURCE_FINGERPRINT}\n"
            "lto=${SIMPLEFLUID_ENABLE_LTO}\n"
            "cxx_flags=${CMAKE_CXX_FLAGS}|${${SIMPLEFLUID_PGO_CONFIG_FLAGS_VARIABLE}}\n")
        string(JOIN "" SIMPLEFLUID_PGO_EXPECTED_MANIFEST
               ${SIMPLEFLUID_PGO_EXPECTED_MANIFEST})
        set(SIMPLEFLUID_PGO_MANIFEST
            "${SIMPLEFLUID_PGO_DIR}/SimpleFluidPGO.manifest")
        file(GLOB_RECURSE SIMPLEFLUID_EXISTING_PGO_DATA
            LIST_DIRECTORIES FALSE
            "${SIMPLEFLUID_PGO_DIR}/*.gcda"
            "${SIMPLEFLUID_PGO_DIR}/*.profraw"
            "${SIMPLEFLUID_PGO_DIR}/*.profdata")

        if(SIMPLEFLUID_PGO_PHASE_NORMALIZED STREQUAL "GENERATE")
            if(EXISTS "${SIMPLEFLUID_PGO_MANIFEST}")
                file(READ "${SIMPLEFLUID_PGO_MANIFEST}"
                     SIMPLEFLUID_PGO_ACTUAL_MANIFEST)
                if(NOT SIMPLEFLUID_PGO_ACTUAL_MANIFEST STREQUAL
                       SIMPLEFLUID_PGO_EXPECTED_MANIFEST)
                    if(SIMPLEFLUID_EXISTING_PGO_DATA)
                        message(FATAL_ERROR
                            "${SIMPLEFLUID_PGO_DIR} contains profile data for "
                            "a different compiler, source, configuration, or "
                            "workload. Select a fresh SIMPLEFLUID_PGO_DIR.")
                    endif()
                    file(WRITE "${SIMPLEFLUID_PGO_MANIFEST}"
                         "${SIMPLEFLUID_PGO_EXPECTED_MANIFEST}")
                endif()
            elseif(SIMPLEFLUID_EXISTING_PGO_DATA)
                message(FATAL_ERROR
                    "${SIMPLEFLUID_PGO_DIR} contains unowned legacy profile "
                    "data. Select a fresh SIMPLEFLUID_PGO_DIR.")
            else()
                file(WRITE "${SIMPLEFLUID_PGO_MANIFEST}"
                     "${SIMPLEFLUID_PGO_EXPECTED_MANIFEST}")
            endif()
        else()
            if(NOT EXISTS "${SIMPLEFLUID_PGO_MANIFEST}")
                message(FATAL_ERROR
                    "SIMPLEFLUID_PGO_PHASE=USE requires provenance manifest "
                    "${SIMPLEFLUID_PGO_MANIFEST}")
            endif()
            file(READ "${SIMPLEFLUID_PGO_MANIFEST}"
                 SIMPLEFLUID_PGO_ACTUAL_MANIFEST)
            if(NOT SIMPLEFLUID_PGO_ACTUAL_MANIFEST STREQUAL
                   SIMPLEFLUID_PGO_EXPECTED_MANIFEST)
                message(FATAL_ERROR
                    "${SIMPLEFLUID_PGO_MANIFEST} does not match this "
                    "compiler, source, configuration, and workload")
            endif()
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                "${SIMPLEFLUID_PGO_MANIFEST}")
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(SIMPLEFLUID_PGO_PHASE_NORMALIZED STREQUAL "GENERATE")
                add_compile_options(
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-generate=${SIMPLEFLUID_PGO_DIR}>"
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-prefix-path=${CMAKE_BINARY_DIR}>"
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-update=atomic>")
                add_link_options(
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-generate=${SIMPLEFLUID_PGO_DIR}>")
            else()
                file(GLOB_RECURSE SIMPLEFLUID_GCC_PROFILE_DATA
                    CONFIGURE_DEPENDS
                    "${SIMPLEFLUID_PGO_DIR}/*.gcda")
                if(NOT SIMPLEFLUID_GCC_PROFILE_DATA)
                    message(FATAL_ERROR
                        "SIMPLEFLUID_PGO_PHASE=USE found no GCC .gcda data in "
                        "${SIMPLEFLUID_PGO_DIR}")
                endif()
                list(SORT SIMPLEFLUID_GCC_PROFILE_DATA)
                set(SIMPLEFLUID_PGO_PROFILE_MANIFEST)
                foreach(SIMPLEFLUID_GCC_PROFILE
                        IN LISTS SIMPLEFLUID_GCC_PROFILE_DATA)
                    file(SHA256 "${SIMPLEFLUID_GCC_PROFILE}"
                         SIMPLEFLUID_GCC_PROFILE_HASH)
                    string(APPEND SIMPLEFLUID_PGO_PROFILE_MANIFEST
                        "${SIMPLEFLUID_GCC_PROFILE}:"
                        "${SIMPLEFLUID_GCC_PROFILE_HASH}\n")
                endforeach()
                string(SHA256 SIMPLEFLUID_PGO_PROFILE_FINGERPRINT
                       "${SIMPLEFLUID_PGO_PROFILE_MANIFEST}")
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                    ${SIMPLEFLUID_GCC_PROFILE_DATA})
                add_compile_options(
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-use=${SIMPLEFLUID_PGO_DIR}>"
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-prefix-path=${CMAKE_BINARY_DIR}>"
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-Wno-missing-profile>"
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-Wno-error=coverage-mismatch>")
                add_link_options(
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-use=${SIMPLEFLUID_PGO_DIR}>")
            endif()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(SIMPLEFLUID_CLANG_PROFILE_DATA
                "${SIMPLEFLUID_PGO_DIR}/simplefluid.profdata")
            if(SIMPLEFLUID_PGO_PHASE_NORMALIZED STREQUAL "GENERATE")
                get_filename_component(
                    SIMPLEFLUID_CLANG_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
                string(REGEX MATCH "^[0-9]+"
                    SIMPLEFLUID_CLANG_MAJOR "${CMAKE_CXX_COMPILER_VERSION}")
                find_program(SIMPLEFLUID_LLVM_PROFDATA
                    NAMES
                        "llvm-profdata-${SIMPLEFLUID_CLANG_MAJOR}"
                        llvm-profdata
                    HINTS "${SIMPLEFLUID_CLANG_BIN_DIR}"
                    REQUIRED)
                execute_process(
                    COMMAND "${SIMPLEFLUID_LLVM_PROFDATA}" --version
                    OUTPUT_VARIABLE SIMPLEFLUID_LLVM_PROFDATA_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                if(NOT SIMPLEFLUID_LLVM_PROFDATA_VERSION
                   MATCHES "LLVM version ${SIMPLEFLUID_CLANG_MAJOR}\\.")
                    message(FATAL_ERROR
                        "${SIMPLEFLUID_LLVM_PROFDATA} does not match "
                        "Clang ${CMAKE_CXX_COMPILER_VERSION}")
                endif()
                set(SIMPLEFLUID_CLANG_RAW_PROFILE
                    "${SIMPLEFLUID_PGO_DIR}/simplefluid-%m-%h-%p.profraw")
                add_compile_options(
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-instr-generate=${SIMPLEFLUID_CLANG_RAW_PROFILE}>")
                add_link_options(
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-instr-generate=${SIMPLEFLUID_CLANG_RAW_PROFILE}>")
                add_custom_target(simplefluid-pgo-merge
                    COMMAND "${CMAKE_COMMAND}"
                        "-DSIMPLEFLUID_LLVM_PROFDATA=${SIMPLEFLUID_LLVM_PROFDATA}"
                        "-DSIMPLEFLUID_PGO_DIR=${SIMPLEFLUID_PGO_DIR}"
                        "-DSIMPLEFLUID_PGO_OUTPUT=${SIMPLEFLUID_CLANG_PROFILE_DATA}"
                        -P "${SIMPLEFLUID_PROFILE_GUIDED_OPTIMIZATION_DIR}/MergeClangProfiles.cmake"
                    COMMENT "Merging Clang PGO profiles"
                    VERBATIM)
            else()
                if(NOT EXISTS "${SIMPLEFLUID_CLANG_PROFILE_DATA}")
                    message(FATAL_ERROR
                        "SIMPLEFLUID_PGO_PHASE=USE requires "
                        "${SIMPLEFLUID_CLANG_PROFILE_DATA}")
                endif()
                file(SHA256 "${SIMPLEFLUID_CLANG_PROFILE_DATA}"
                     SIMPLEFLUID_PGO_PROFILE_FINGERPRINT)
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                    "${SIMPLEFLUID_CLANG_PROFILE_DATA}")
                add_compile_options(
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-instr-use=${SIMPLEFLUID_CLANG_PROFILE_DATA}>"
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-Wno-profile-instr-unprofiled>"
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-Wno-profile-instr-missing>"
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-Wno-profile-instr-out-of-date>")
                add_link_options(
                    "$<${SIMPLEFLUID_PGO_CONFIG_EXPRESSION}:-fprofile-instr-use=${SIMPLEFLUID_CLANG_PROFILE_DATA}>")
            endif()
        else()
            message(FATAL_ERROR
                "SIMPLEFLUID_PGO_PHASE supports GCC and Clang-family compilers")
        endif()

        if(SIMPLEFLUID_PGO_PHASE_NORMALIZED STREQUAL "USE")
            add_compile_definitions(
                "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:${SIMPLEFLUID_PGO_CONFIG}>>:SIMPLEFLUID_PGO_PROFILE_FINGERPRINT_${SIMPLEFLUID_PGO_PROFILE_FINGERPRINT}=1>")
            message(STATUS
                "PGO profile fingerprint: "
                "${SIMPLEFLUID_PGO_PROFILE_FINGERPRINT}")
        endif()
    endif()
endmacro()

# PGO training deliberately covers production targets, not every PCH, test, or
# optional static-archive member in the build. Opt production code into fatal
# profile-completeness diagnostics target by target or, for a partially linked
# static archive, source by source. A fingerprint definition makes every
# covered object depend on the profile contents, so retraining cannot leave a
# successful no-op USE build optimized with stale data.
function(simplefluid_pgo_completeness_flags output_name)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(simplefluid_pgo_flags
            -Werror=missing-profile
            -Werror=coverage-mismatch)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(simplefluid_pgo_flags
            -Werror=profile-instr-unprofiled
            -Werror=profile-instr-missing
            -Werror=profile-instr-out-of-date)
    else()
        set(simplefluid_pgo_flags)
    endif()
    set(${output_name} "${simplefluid_pgo_flags}" PARENT_SCOPE)
endfunction()

function(simplefluid_require_complete_pgo target_name)
    if(NOT SIMPLEFLUID_PGO_PHASE_NORMALIZED STREQUAL "USE")
        return()
    endif()

    simplefluid_pgo_completeness_flags(simplefluid_pgo_strict_flags)
    foreach(simplefluid_pgo_flag IN LISTS simplefluid_pgo_strict_flags)
        target_compile_options(${target_name} PRIVATE
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:${SIMPLEFLUID_PGO_CONFIG}>>:${simplefluid_pgo_flag}>")
    endforeach()
endfunction()

function(simplefluid_require_complete_pgo_sources)
    if(NOT SIMPLEFLUID_PGO_PHASE_NORMALIZED STREQUAL "USE")
        return()
    endif()

    simplefluid_pgo_completeness_flags(simplefluid_pgo_strict_flags)
    foreach(simplefluid_pgo_source IN LISTS ARGN)
        foreach(simplefluid_pgo_flag IN LISTS simplefluid_pgo_strict_flags)
            set_property(SOURCE "${simplefluid_pgo_source}" APPEND PROPERTY
                COMPILE_OPTIONS
                "$<$<CONFIG:${SIMPLEFLUID_PGO_CONFIG}>:${simplefluid_pgo_flag}>")
        endforeach()
    endforeach()
endfunction()

function(simplefluid_exclude_from_pgo_sources)
    if(NOT SIMPLEFLUID_PGO_PHASE_NORMALIZED STREQUAL "USE")
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(simplefluid_pgo_disable_flag -fno-profile-use)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(simplefluid_pgo_disable_flag -fno-profile-instr-use)
    else()
        return()
    endif()
    foreach(simplefluid_pgo_source IN LISTS ARGN)
        set_property(SOURCE "${simplefluid_pgo_source}" APPEND PROPERTY
            COMPILE_OPTIONS
            "$<$<CONFIG:${SIMPLEFLUID_PGO_CONFIG}>:${simplefluid_pgo_disable_flag}>")
    endforeach()
endfunction()
