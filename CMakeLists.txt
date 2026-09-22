cmake_minimum_required(VERSION 3.21)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(SimpleFluid C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/Options.cmake")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(
    "-fno-semantic-interposition"
    SIMPLEFLUID_CXX_HAS_FNO_SEMANTIC_INTERPOSITION)

include(CTest)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/Profiling.cmake")

if(SIMPLEFLUID_ENABLE_LTO AND NOT SIMPLEFLUID_ENABLE_COVERAGE)
    if(POLICY CMP0138)
        cmake_policy(SET CMP0138 NEW)
    endif()
    include(CheckIPOSupported)
    check_ipo_supported(
        RESULT SIMPLEFLUID_CXX_IPO_SUPPORTED
        OUTPUT SIMPLEFLUID_CXX_IPO_ERROR
        LANGUAGES CXX)
    if(NOT SIMPLEFLUID_CXX_IPO_SUPPORTED)
        message(FATAL_ERROR
            "SIMPLEFLUID_ENABLE_LTO requires compiler IPO support: "
            "${SIMPLEFLUID_CXX_IPO_ERROR}")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)

    # Use LLD for Clang ThinLTO. Legacy direct-static Trilinos linking also
    # needs the non-LTO compatibility object in src/CMakeLists.txt.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        check_cxx_compiler_flag(
            "-fuse-ld=lld" SIMPLEFLUID_CXX_HAS_LLD_LINKER)
        if(NOT SIMPLEFLUID_CXX_HAS_LLD_LINKER)
            message(FATAL_ERROR
                "Clang LTO with the static Trilinos build requires LLD")
        endif()
        add_link_options(
            "$<$<CONFIG:Release,RelWithDebInfo>:-fuse-ld=lld>")
    endif()
elseif(SIMPLEFLUID_ENABLE_LTO AND SIMPLEFLUID_ENABLE_COVERAGE)
    message(STATUS "LTO is disabled for the coverage-instrumented build")
endif()

simplefluid_configure_profile_guided_optimization()

cmake_host_system_information(RESULT NCPU QUERY NUMBER_OF_LOGICAL_CORES)

# parallel job pool settings, only available for ninja build.
math(EXPR SIMPLEFLUID_COMPILE_JOBS "${NCPU} / 2")
math(EXPR SIMPLEFLUID_LINK_JOBS "${NCPU} / 5")
if(SIMPLEFLUID_COMPILE_JOBS LESS 1)
    set(SIMPLEFLUID_COMPILE_JOBS 1)
endif()
if(SIMPLEFLUID_LINK_JOBS LESS 1)
    set(SIMPLEFLUID_LINK_JOBS 1)
endif()
set_property(GLOBAL PROPERTY JOB_POOLS
    compile_pool=${SIMPLEFLUID_COMPILE_JOBS}
    link_pool=${SIMPLEFLUID_LINK_JOBS})
set(CMAKE_JOB_POOL_COMPILE compile_pool)
set(CMAKE_JOB_POOL_LINK link_pool)

if(NOT SIMPLEFLUID_MAX_TEST_PROCS MATCHES "^(0|[1-9][0-9]*)$")
    message(FATAL_ERROR
        "SIMPLEFLUID_MAX_TEST_PROCS must be a non-negative integer; got "
        "'${SIMPLEFLUID_MAX_TEST_PROCS}'")
endif()
if(SIMPLEFLUID_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(--coverage -O0 -g)
        add_link_options(--coverage)
    else()
        message(FATAL_ERROR
            "SIMPLEFLUID_ENABLE_COVERAGE requires GCC or Clang")
    endif()
endif()

# Profiling flags for RelWithDebInfo: preserve frame pointers for accurate
# stack unwinding in performance profilers (Instruments, perf, etc.)
# Also define NDEBUG to disable expensive debug-only bounds checks while
# keeping debug symbols for the profiler.
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")

set(SIMPLEFLUID_TRILINOS_COMPONENTS
    Kokkos
    Teuchos
    Tpetra
    STKIO
    STKMesh
    STKTopology
    STKUtil
    Belos
    Ifpack2
    MueLu
    Zoltan2
)
if(SIMPLEFLUID_ENABLE_NOX)
    list(APPEND SIMPLEFLUID_TRILINOS_COMPONENTS NOX ThyraCore ThyraTpetraAdapters)
endif()
find_package(Trilinos REQUIRED COMPONENTS ${SIMPLEFLUID_TRILINOS_COMPONENTS})
find_package(MPI REQUIRED COMPONENTS CXX)
if(SIMPLEFLUID_ENABLE_NOX)
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES
        ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})
    set(_simplefluid_previous_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    # Recheck when the selected Trilinos installation changes in this tree.
    unset(SIMPLEFLUID_TRILINOS_HAS_NOX_THYRA CACHE)
    check_cxx_source_compiles([[
        #include <NOX_Config.h>
        #ifndef HAVE_NOX_THYRA
        #error "NOX was built without Thyra support"
        #endif
        int main() { return 0; }
    ]] SIMPLEFLUID_TRILINOS_HAS_NOX_THYRA)
    if(_simplefluid_previous_try_compile_target_type)
        set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_simplefluid_previous_try_compile_target_type}")
    else()
        unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
    endif()
    unset(_simplefluid_previous_try_compile_target_type)
    cmake_pop_check_state()
    if(NOT SIMPLEFLUID_TRILINOS_HAS_NOX_THYRA)
        message(FATAL_ERROR
            "SIMPLEFLUID_ENABLE_NOX requires NOX with Thyra support and Tpetra adapters "
            "in the selected Trilinos installation.")
    endif()
endif()

if(SIMPLEFLUID_DOCS_ENABLE_DIAGRAMS AND NOT SIMPLEFLUID_BUILD_DOCS)
    message(FATAL_ERROR
        "SIMPLEFLUID_DOCS_ENABLE_DIAGRAMS requires SIMPLEFLUID_BUILD_DOCS=ON.")
endif()

if(SIMPLEFLUID_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(SIMPLEFLUID_DOXYGEN_ENABLED_SECTIONS "")
    set(SIMPLEFLUID_DOXYGEN_HAVE_DOT "NO")
    set(SIMPLEFLUID_DOXYGEN_DOT_PATH "")
    set(SIMPLEFLUID_DOXYGEN_DOT_IMAGE_FORMAT "png")
    set(SIMPLEFLUID_DOXYGEN_PLANTUML_JAR_PATH "")
    set(_simplefluid_doxygen_command Doxygen::doxygen "${CMAKE_BINARY_DIR}/Doxyfile")
    set(_simplefluid_docs_smoke_arguments "${CMAKE_BINARY_DIR}/docs/doxygen/html")

    if(SIMPLEFLUID_DOCS_ENABLE_DIAGRAMS)
        execute_process(
            COMMAND "${DOXYGEN_EXECUTABLE}" -g -
            OUTPUT_VARIABLE _simplefluid_doxygen_defaults
            ERROR_QUIET
            RESULT_VARIABLE _simplefluid_doxygen_defaults_result
        )
        if(NOT _simplefluid_doxygen_defaults_result EQUAL 0 OR
           NOT _simplefluid_doxygen_defaults MATCHES "PLANTUML_JAR_PATH" OR
           NOT _simplefluid_doxygen_defaults MATCHES "PLANTUML_CFG_FILE" OR
           NOT _simplefluid_doxygen_defaults MATCHES "DOT_IMAGE_FORMAT")
            message(FATAL_ERROR
                "The selected Doxygen does not expose the PlantUML/SVG "
                "configuration used by diagram-enabled documentation.")
        endif()
        find_package(Java REQUIRED COMPONENTS Runtime)
        find_program(_simplefluid_dot_executable NAMES dot)
        if(NOT _simplefluid_dot_executable)
            message(FATAL_ERROR
                "Diagram-enabled documentation requires Graphviz 'dot'.")
        endif()

        if(SIMPLEFLUID_PLANTUML_JAR)
            if(NOT EXISTS "${SIMPLEFLUID_PLANTUML_JAR}")
                message(FATAL_ERROR
                    "SIMPLEFLUID_PLANTUML_JAR does not exist: "
                    "${SIMPLEFLUID_PLANTUML_JAR}")
            endif()
            set(_simplefluid_plantuml_jar "${SIMPLEFLUID_PLANTUML_JAR}")
        else()
            find_file(_simplefluid_plantuml_jar
                NAMES plantuml.jar
                HINTS ENV PLANTUML_HOME
                PATHS
                    /usr/share/java
                    /usr/local/share/java
                    /usr/share/plantuml
                    /usr/local/share/plantuml
                    /opt/plantuml
                PATH_SUFFIXES plantuml
            )
        endif()
        if(NOT _simplefluid_plantuml_jar OR
           NOT EXISTS "${_simplefluid_plantuml_jar}" OR
           IS_DIRECTORY "${_simplefluid_plantuml_jar}")
            message(FATAL_ERROR
                "Diagram-enabled documentation requires a local PlantUML JAR. "
                "Set SIMPLEFLUID_PLANTUML_JAR=/path/to/plantuml.jar.")
        endif()

        get_filename_component(
            SIMPLEFLUID_DOXYGEN_PLANTUML_JAR_PATH
            "${_simplefluid_plantuml_jar}" DIRECTORY)
        get_filename_component(
            SIMPLEFLUID_DOXYGEN_DOT_PATH
            "${_simplefluid_dot_executable}" DIRECTORY)
        set(SIMPLEFLUID_DOXYGEN_ENABLED_SECTIONS "SIMPLEFLUID_DIAGRAMS")
        set(SIMPLEFLUID_DOXYGEN_HAVE_DOT "YES")
        # The selected Doxygen was checked for this setting; PlantUML output is
        # validated as genuine SVG by the post-build documentation smoke test.
        set(SIMPLEFLUID_DOXYGEN_DOT_IMAGE_FORMAT "svg")
        set(_simplefluid_doxygen_command
            "${CMAKE_COMMAND}" -E env
            "JAVA_TOOL_OPTIONS=-Djava.awt.headless=true"
            "${DOXYGEN_EXECUTABLE}" "${CMAKE_BINARY_DIR}/Doxyfile")
        list(APPEND _simplefluid_docs_smoke_arguments --expect-diagrams)
    endif()

    configure_file(
        "${PROJECT_SOURCE_DIR}/docs/Doxyfile.in"
        "${CMAKE_BINARY_DIR}/Doxyfile"
        @ONLY
    )
    add_custom_target(docs
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${CMAKE_BINARY_DIR}/docs/doxygen"
        COMMAND ${_simplefluid_doxygen_command}
        COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/docs/check_doxygen_output.py"
                ${_simplefluid_docs_smoke_arguments}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Generating SimpleFluid API documentation"
        VERBATIM
    )
endif()

if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    include(GoogleTest)
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/Testing.cmake")
endif()

add_subdirectory(src)
