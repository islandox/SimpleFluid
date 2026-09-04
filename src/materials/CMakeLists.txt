# The upstream dependency is private to one translation unit. Existing builds
# neither search for nor download IF97 unless SIMPLEFLUID_ENABLE_IF97 is ON.
find_path(SIMPLEFLUID_IF97_INCLUDE_DIR NAMES IF97.h PATH_SUFFIXES IF97
    DOC "Directory containing CoolProp's IF97.h (optional offline dependency)")
if(SIMPLEFLUID_IF97_INCLUDE_DIR)
    if(NOT EXISTS "${SIMPLEFLUID_IF97_INCLUDE_DIR}/IF97.h")
        message(FATAL_ERROR "SIMPLEFLUID_IF97_INCLUDE_DIR must contain IF97.h")
    endif()
    set(simplefluid_if97_headers "${SIMPLEFLUID_IF97_INCLUDE_DIR}")
else()
    include(FetchContent)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    FetchContent_Declare(simplefluid_if97
        URL https://codeload.github.com/CoolProp/IF97/tar.gz/refs/tags/v2.2.1
        URL_HASH SHA256=6ea212c17773be24d198659494b8ac226f4ccd6bfbbc69092d9b2f01fe211bac
        # Consume the header without running upstream's wrapper/test project.
        SOURCE_SUBDIR simplefluid_header_only)
    FetchContent_MakeAvailable(simplefluid_if97)
    set(simplefluid_if97_headers "${simplefluid_if97_SOURCE_DIR}")
endif()

# Earlier releases use shared scratch storage in saturation evaluation.
file(STRINGS "${simplefluid_if97_headers}/IF97.h" simplefluid_if97_version_line
    REGEX "IF97VERSION.*=")
if(NOT simplefluid_if97_version_line MATCHES "v([0-9]+\\.[0-9]+\\.[0-9]+)"
   OR CMAKE_MATCH_1 VERSION_LESS "2.2.1")
    message(FATAL_ERROR "SimpleFluid requires CoolProp IF97 2.2.1 or newer")
endif()

add_library(SimpleFluidIF97 STATIC IF97Water.cc)
add_library(SimpleFluid::IF97 ALIAS SimpleFluidIF97)
target_include_directories(SimpleFluidIF97 PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_include_directories(SimpleFluidIF97 SYSTEM PRIVATE "${simplefluid_if97_headers}")
set_target_properties(SimpleFluidIF97 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
simplefluid_enable_native_archive_fallback(SimpleFluidIF97)
