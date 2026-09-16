include_guard(GLOBAL)

function(simplefluid_configure_library target_name)
    target_include_directories(${target_name} PUBLIC
        ${SIMPLEFLUID_PUBLIC_INCLUDE_DIRS}
    )
    target_link_libraries(${target_name} PUBLIC
        ${SIMPLEFLUID_EXTERNAL_LIBRARIES}
    )
endfunction()

# GCC's linker plugin can discard weak definitions from the non-LTO static
# Trilinos archives when a unit test also contributes LTO objects.  Preserve a
# native copy in SimpleFluid's static archives so GNU test executables can use a
# normal non-plugin link while production libraries and applications retain LTO.
function(simplefluid_enable_native_archive_fallback target_name)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND SIMPLEFLUID_ENABLE_LTO
       AND NOT SIMPLEFLUID_ENABLE_COVERAGE
       AND NOT SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME)
        target_compile_options(${target_name} PRIVATE
            "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release,RelWithDebInfo>>:-ffat-lto-objects>")
    endif()
endfunction()

function(simplefluid_lib_reuse_precompiled_header target_name)
    target_precompile_headers(${target_name} REUSE_FROM SimpleFluidPch)
endfunction()

function(simplefluid_exe_reuse_precompiled_header target_name)
    target_precompile_headers(
        ${target_name} REUSE_FROM SimpleFluidExecutablePch)
endfunction()

# Apply the common binary-library policy without changing target ownership.
function(simplefluid_configure_component target_name)
    simplefluid_configure_library(${target_name})
    get_target_property(component_type ${target_name} TYPE)
    if(component_type STREQUAL "STATIC_LIBRARY")
        simplefluid_enable_native_archive_fallback(${target_name})
    endif()
    set_target_properties(${target_name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET ${SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET}
        VISIBILITY_INLINES_HIDDEN ${SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN})
    target_precompile_headers(${target_name} REUSE_FROM SimpleFluidSharedPch)
endfunction()

# Source lists exported by a sibling component must not depend on its caller's
# source directory. Keep ordering intact while resolving those local paths.
function(simplefluid_absolute_sources list_name)
    set(absolute_sources)
    foreach(source IN LISTS ${list_name})
        get_filename_component(source "${source}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        list(APPEND absolute_sources "${source}")
    endforeach()
    set(${list_name} "${absolute_sources}" PARENT_SCOPE)
endfunction()

function(simplefluid_add_example target_name source)
    add_executable(${target_name} "${source}")
    target_link_libraries(${target_name} PRIVATE SimpleFluid ${ARGN})
    simplefluid_exe_reuse_precompiled_header(${target_name})
endfunction()
