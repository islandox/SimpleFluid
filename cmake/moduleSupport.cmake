set(SIMPLEFLUID_MODULE_SUPPORT_DIR "${CMAKE_CURRENT_LIST_DIR}")

set(SIMPLEFLUID_ENABLE_CXX_MODULES_DEFAULT ON)
option(SIMPLEFLUID_ENABLE_CXX_MODULES
       "Build C++ module interfaces for module-capable toolchains"
       ${SIMPLEFLUID_ENABLE_CXX_MODULES_DEFAULT})
option(SIMPLEFLUID_ENABLE_STD_MODULE
       "Use the C++23 standard-library module in module consumers"
       ON)

# CMake's import-std support is still experimental and its activation token
# changes between CMake releases. Keep older/newer CMake versions on the
# existing standard-header path until their token is deliberately validated.
# This must be set before project() enables CXX.
if(SIMPLEFLUID_ENABLE_CXX_MODULES
   AND SIMPLEFLUID_ENABLE_STD_MODULE
   AND CMAKE_VERSION VERSION_GREATER_EQUAL 4.4
   AND CMAKE_VERSION VERSION_LESS 4.5)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
        "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
endif()

macro(simplefluid_configure_module_support)
    set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
    set(SIMPLEFLUID_USE_STD_MODULE OFF)
    set(SIMPLEFLUID_USE_BENCHMARK_MODULES OFF)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Trilinos 17 has fallback template bodies that Clang diagnoses eagerly
        # against libstdc++'s C++23 string(nullptr_t) deletion.
        add_compile_options(
            "$<$<COMPILE_LANGUAGE:CXX>:-fdelayed-template-parsing>"
            "$<$<COMPILE_LANGUAGE:CXX>:-Wno-delayed-template-parsing-in-cxx20>"
            "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated>")
    endif()

    if(SIMPLEFLUID_ENABLE_CXX_MODULES
       AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(SIMPLEFLUID_USE_BENCHMARK_MODULES ON)

        if(SIMPLEFLUID_ENABLE_STD_MODULE
           AND 23 IN_LIST CMAKE_CXX_COMPILER_IMPORT_STD)
            set(SIMPLEFLUID_USE_STD_MODULE ON)

            # GCC 16 installations may ship libstdc++.modules.json beside
            # libgcc even though its relative source paths assume the C++
            # include root. Generate a corrected build-local copy based on the
            # selected standard library.
            if(CMAKE_CXX_STDLIB_MODULES_JSON
               MATCHES "libstdc\\+\\+\\.modules\\.json$")
                find_file(SIMPLEFLUID_LIBSTDCXX_STD_MODULE_SOURCE
                    NAMES bits/std.cc
                    PATHS ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}
                    NO_DEFAULT_PATH)
                find_file(SIMPLEFLUID_LIBSTDCXX_STD_COMPAT_MODULE_SOURCE
                    NAMES bits/std.compat.cc
                    PATHS ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}
                    NO_DEFAULT_PATH)
                if(SIMPLEFLUID_LIBSTDCXX_STD_MODULE_SOURCE
                   AND SIMPLEFLUID_LIBSTDCXX_STD_COMPAT_MODULE_SOURCE)
                    configure_file(
                        "${SIMPLEFLUID_MODULE_SUPPORT_DIR}/libstdc++.modules.json.in"
                        "${CMAKE_BINARY_DIR}/libstdc++.modules.json"
                        @ONLY)
                    set(CMAKE_CXX_STDLIB_MODULES_JSON
                        "${CMAKE_BINARY_DIR}/libstdc++.modules.json")
                else()
                    message(WARNING
                        "The toolchain advertises import std support, but the "
                        "libstdc++ module sources were not found; retaining "
                        "standard headers")
                    set(SIMPLEFLUID_USE_STD_MODULE OFF)
                endif()
            endif()
        endif()
    endif()

    if(SIMPLEFLUID_USE_STD_MODULE)
        message(STATUS "SimpleFluid module consumers use import std")
    elseif(SIMPLEFLUID_ENABLE_CXX_MODULES
           AND SIMPLEFLUID_ENABLE_STD_MODULE
           AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS
            "Clang retains standard headers because import std triggers a "
            "Trilinos late-template compiler crash")
    elseif(SIMPLEFLUID_ENABLE_CXX_MODULES
           AND SIMPLEFLUID_ENABLE_STD_MODULE)
        message(STATUS
            "import std is unavailable; SimpleFluid modules retain standard "
            "headers")
    elseif(SIMPLEFLUID_ENABLE_CXX_MODULES)
        message(STATUS "SimpleFluid import std support is disabled")
    endif()
endmacro()

function(simplefluid_enable_std_module target_name)
    if(SIMPLEFLUID_USE_STD_MODULE)
        set_target_properties(${target_name} PROPERTIES CXX_MODULE_STD ON)
    endif()
endfunction()

# Configure ordinary translation units that consume named modules. These
# sources must not inherit the Trilinos-heavy PCH: mixing textual declarations
# with declarations reachable through a BMI is ill-formed.
function(simplefluid_configure_module_consumers target_name)
    if(NOT SIMPLEFLUID_ENABLE_CXX_MODULES)
        return()
    endif()

    set_target_properties(${target_name} PROPERTIES
        CXX_SCAN_FOR_MODULES ON)
    simplefluid_enable_std_module(${target_name})
    set_source_files_properties(${ARGN}
        PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON
        COMPILE_DEFINITIONS SIMPLEFLUID_USE_CXX_MODULES=1
        COMPILE_OPTIONS
        "$<$<CXX_COMPILER_ID:Clang>:-fno-delayed-template-parsing>")
    if(SIMPLEFLUID_USE_STD_MODULE)
        set_property(SOURCE ${ARGN} APPEND PROPERTY
            COMPILE_DEFINITIONS SIMPLEFLUID_USE_STD_MODULE=1)
    endif()
endfunction()

function(simplefluid_add_trilinos_modules)
    cmake_parse_arguments(PARSE_ARGV 0 _module "" "" "HEADERS;MODULES")
    if(NOT _module_HEADERS OR NOT _module_MODULES)
        message(FATAL_ERROR
            "simplefluid_add_trilinos_modules requires HEADERS and MODULES")
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR
            "SimpleFluid Trilinos wrapper modules require GCC or Clang")
    endif()

    set(_header_unit_include_options)
    set(_header_unit_compiler_options)
    if(CMAKE_CXX_FLAGS)
        # Custom header-unit commands do not inherit target compile options.
        # Preserve global ABI selections such as Clang's -stdlib=libc++ so the
        # BMI and every translation unit importing it use the same standard
        # library.
        separate_arguments(_header_unit_compiler_options
            NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
    endif()
    set(_header_unit_include_dirs
        ${SIMPLEFLUID_PUBLIC_INCLUDE_DIRS}
        ${Trilinos_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS})
    foreach(_dependency_target IN ITEMS
            Kokkos::kokkos
            METIS::all_libs
            ParMETIS::all_libs)
        if(TARGET ${_dependency_target})
            get_target_property(_dependency_include_dirs
                ${_dependency_target} INTERFACE_INCLUDE_DIRECTORIES)
            if(_dependency_include_dirs)
                list(APPEND _header_unit_include_dirs
                    ${_dependency_include_dirs})
            endif()
        endif()
    endforeach()
    foreach(_include_dir IN LISTS _header_unit_include_dirs)
        list(APPEND _header_unit_include_options "-I${_include_dir}")
    endforeach()
    list(REMOVE_DUPLICATES _header_unit_include_options)

    set(_header_unit_outputs)
    set(_clang_header_unit_options)
    set(_gcc_header_unit_map_contents "")
    set(_header_unit_work_dir
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/SimpleFluidHeaderUnits")
    file(MAKE_DIRECTORY "${_header_unit_work_dir}")

    foreach(_header IN LISTS _module_HEADERS)
        get_filename_component(_header_name "${_header}" NAME_WE)
        set(_depfile "${_header_unit_work_dir}/${_header_name}.d")

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(_stamp "${_header_unit_work_dir}/${_header_name}.stamp")
            set(_gcm "${CMAKE_BINARY_DIR}/gcm.cache${_header}.gcm")
            add_custom_command(
                OUTPUT "${_stamp}"
                BYPRODUCTS "${_gcm}"
                COMMAND ${CMAKE_CXX_COMPILER}
                    ${_header_unit_compiler_options}
                    -DKOKKOS_DEPENDENCE
                    ${_header_unit_include_options}
                    -std=c++23
                    -fmodules
                    -x c++-header
                    -MD -Mno-modules -MF "${_depfile}"
                    "${_header}"
                COMMAND ${CMAKE_COMMAND} -E touch "${_stamp}"
                DEPENDS "${_header}"
                DEPFILE "${_depfile}"
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                COMMAND_EXPAND_LISTS
                VERBATIM)
            list(APPEND _header_unit_outputs "${_stamp}")
            string(APPEND _gcc_header_unit_map_contents
                "${_header} ${_gcm}\n")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(_pcm "${_header_unit_work_dir}/${_header_name}.pcm")
            add_custom_command(
                OUTPUT "${_pcm}"
                COMMAND ${CMAKE_CXX_COMPILER}
                    ${_header_unit_compiler_options}
                    -DKOKKOS_DEPENDENCE
                    ${_header_unit_include_options}
                    -std=c++23
                    -fdelayed-template-parsing
                    -Wno-delayed-template-parsing-in-cxx20
                    -Wno-deprecated
                    -fmodule-header=user
                    -MD -MF "${_depfile}"
                    "${_header}"
                    -o "${_pcm}"
                DEPENDS "${_header}"
                DEPFILE "${_depfile}"
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                COMMAND_EXPAND_LISTS
                VERBATIM)
            list(APPEND _header_unit_outputs "${_pcm}")
            list(APPEND _clang_header_unit_options "-fmodule-file=${_pcm}")
        endif()
    endforeach()

    add_custom_target(SimpleFluidHeaderUnits
        DEPENDS ${_header_unit_outputs})

    add_library(SimpleFluidTrilinosModules STATIC)
    add_library(SimpleFluid::TrilinosModules ALIAS SimpleFluidTrilinosModules)
    simplefluid_configure_library(SimpleFluidTrilinosModules)
    simplefluid_enable_native_archive_fallback(SimpleFluidTrilinosModules)
    set_target_properties(SimpleFluidTrilinosModules PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_sources(SimpleFluidTrilinosModules PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${_module_MODULES})
    set_source_files_properties(${_module_MODULES}
        PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
    add_dependencies(SimpleFluidTrilinosModules SimpleFluidHeaderUnits)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(_gcc_header_unit_map
            "${_header_unit_work_dir}/gcc-header-units.map")
        file(GENERATE
            OUTPUT "${_gcc_header_unit_map}"
            CONTENT "${_gcc_header_unit_map_contents}")

        # CMake 4.4 may compile a module interface again in a synthetic target
        # when a consumer has different compile options. Synthetic targets do
        # not copy the provider target's CXX_COMPILER_LAUNCHER, so intercept
        # every compile rule in this directory, including CMake's synthetic
        # ones. Keep any existing rule launcher behind this mapper launcher.
        get_property(_existing_rule_launcher DIRECTORY PROPERTY
            RULE_LAUNCH_COMPILE)
        set(_gcc_rule_launcher
            "${CMAKE_COMMAND} -DSIMPLEFLUID_HEADER_UNIT_MAP=${_gcc_header_unit_map} -P ${SIMPLEFLUID_MODULE_SUPPORT_DIR}/LaunchGccWithHeaderUnits.cmake --")
        if(_existing_rule_launcher)
            string(APPEND _gcc_rule_launcher
                " ${_existing_rule_launcher}")
        endif()
        set_property(DIRECTORY PROPERTY RULE_LAUNCH_COMPILE
            "${_gcc_rule_launcher}")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(SimpleFluidTrilinosModules PUBLIC
            ${_clang_header_unit_options})
    endif()
endfunction()

function(simplefluid_configure_module_target target_name)
    add_dependencies(${target_name} SimpleFluidHeaderUnits)
    # A public provider must not carry a consumer-specific synthetic std BMI.
    # Its interface uses StandardHeaders.hh; leaf consumers opt into import std.
    set_target_properties(${target_name} PROPERTIES CXX_MODULE_STD OFF)
endfunction()

function(simplefluid_configure_project_modules)
    cmake_parse_arguments(PARSE_ARGV 0 _module "" "" "TARGETS;SOURCES")
    if(NOT _module_TARGETS OR NOT _module_SOURCES)
        message(FATAL_ERROR
            "simplefluid_configure_project_modules requires TARGETS and SOURCES")
    endif()

    foreach(_module_target IN LISTS _module_TARGETS)
        simplefluid_configure_module_target(${_module_target})
    endforeach()

    set_source_files_properties(${_module_SOURCES}
        PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
    set_source_files_properties(${_module_SOURCES}
        PROPERTIES COMPILE_DEFINITIONS
        "$<$<CXX_COMPILER_ID:GNU>:SIMPLEFLUID_USE_TRILINOS_MODULES=1>")
    set_source_files_properties(${_module_SOURCES}
        PROPERTIES COMPILE_OPTIONS
        "$<$<CXX_COMPILER_ID:GNU,Clang>:-fvisibility=default>;$<$<CXX_COMPILER_ID:Clang>:-fno-delayed-template-parsing>")
endfunction()

# Clang with libstdc++ safely consumes project BMIs from module-aware
# implementation TUs. GCC retains the textual PCH path to avoid interposable
# Trilinos template bodies being emitted into the shared library. Clang with
# libc++ uses that same implementation-only fallback: importing the aggregate
# solver BMI into its explicit-instantiation TUs otherwise causes pathological
# compile times, while public module providers and consumers remain enabled.
function(simplefluid_configure_clang_module_consumers target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang"
       AND NOT SIMPLEFLUID_CXX_USES_LIBCXX)
        simplefluid_configure_module_consumers(${target_name} ${ARGN})
    endif()
endfunction()

function(simplefluid_configure_example_module target_name source_file)
    simplefluid_configure_module_target(${target_name})
    set_source_files_properties(${source_file} PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON
        COMPILE_DEFINITIONS
        "$<$<CXX_COMPILER_ID:GNU>:SIMPLEFLUID_USE_TRILINOS_MODULES=1>"
        COMPILE_OPTIONS
        "$<$<CXX_COMPILER_ID:Clang>:-fno-delayed-template-parsing>")
endfunction()

function(simplefluid_configure_benchmark_module target_name source_file)
    set_target_properties(${target_name} PROPERTIES CXX_MODULE_STD OFF)
    set_source_files_properties(${source_file} PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON)
endfunction()

function(simplefluid_configure_module_test target_name source_file)
    set_target_properties(${target_name} PROPERTIES CXX_SCAN_FOR_MODULES ON)
    simplefluid_enable_std_module(${target_name})
    set_source_files_properties(${source_file}
        PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON
        COMPILE_OPTIONS
        "$<$<CXX_COMPILER_ID:Clang>:-fno-delayed-template-parsing>")
    if(SIMPLEFLUID_USE_STD_MODULE)
        set_property(SOURCE ${source_file} APPEND PROPERTY
            COMPILE_DEFINITIONS SIMPLEFLUID_USE_STD_MODULE=1)
    endif()
endfunction()
