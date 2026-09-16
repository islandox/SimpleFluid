include_guard(GLOBAL)

function(simplefluid_add_pch target_name)
    cmake_parse_arguments(PCH "SHARED;EXECUTABLE" "" "" ${ARGN})
    add_library(${target_name} STATIC "${PROJECT_SOURCE_DIR}/cmake/targets/PrecompiledHeaders.cc")
    simplefluid_configure_library(${target_name})
    target_precompile_headers(${target_name} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:${SIMPLEFLUID_PCH_HEADER}>")
    if(PCH_EXECUTABLE)
        set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE OFF)
    else()
        set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    if(PCH_SHARED)
        set_target_properties(${target_name} PROPERTIES
            CXX_VISIBILITY_PRESET ${SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET}
            VISIBILITY_INLINES_HIDDEN ${SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN})
    endif()
endfunction()

# PCH ownership stays independent of component dependencies. Executables and
# shared objects need separate PCHs because their PIC/visibility flags differ.
function(simplefluid_add_precompiled_headers)
    simplefluid_add_pch(SimpleFluidPch)
    simplefluid_add_pch(SimpleFluidSharedPch SHARED)
    simplefluid_add_pch(SimpleFluidExecutablePch EXECUTABLE)
endfunction()
