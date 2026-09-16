include_guard(GLOBAL)

function(simplefluid_add_lto_compatibility)
    # Clang ThinLTO can select an incomplete consumer-emitted MueLu vtable COMDAT
    # before it sees the complete GCC-built archive definition.  Put a non-LTO
    # explicit instantiation ahead of every ThinLTO input on the DSO link line.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
       AND SIMPLEFLUID_ENABLE_LTO
       AND NOT SIMPLEFLUID_ENABLE_COVERAGE
       AND NOT SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME)
        add_library(SimpleFluidClangLtoCompatibility OBJECT
            ClangLtoTrilinosCompatibility.cc)
        simplefluid_configure_library(SimpleFluidClangLtoCompatibility)
        set_target_properties(SimpleFluidClangLtoCompatibility PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION FALSE
            INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
            INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE
            POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES)
        simplefluid_exclude_from_pgo_sources(
            ClangLtoTrilinosCompatibility.cc)
    endif()

    # GCC's LTO plugin can select, then discard, weak libstdc++ COMDAT definitions
    # that are also referenced by the non-LTO STK/Ioss archives.  Supply those
    # definitions from one deliberately non-LTO object instead of forcing large
    # archive members into each shared library.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND SIMPLEFLUID_ENABLE_LTO
       AND NOT SIMPLEFLUID_ENABLE_COVERAGE
       AND NOT SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME)
        add_library(SimpleFluidGccLtoCompatibility OBJECT
            GccLtoStdlibCompatibility.cc)
        set_target_properties(SimpleFluidGccLtoCompatibility PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION FALSE
            INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
            INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE
            POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES)
        # Propagate the object from the root static-library interface so Core-only
        # tests as well as DSO consumers receive it before their LTO inputs.
        target_sources(SimpleFluidCore PUBLIC
            "$<$<CONFIG:Release,RelWithDebInfo>:$<TARGET_OBJECTS:SimpleFluidGccLtoCompatibility>>")
        simplefluid_require_complete_pgo(SimpleFluidGccLtoCompatibility)
    endif()

    if(TARGET SimpleFluidClangLtoCompatibility)
        add_dependencies(SimpleFluidFVM SimpleFluidClangLtoCompatibility)
        target_link_options(SimpleFluidFVM PRIVATE
            "$<$<CONFIG:Release,RelWithDebInfo>:$<TARGET_OBJECTS:SimpleFluidClangLtoCompatibility>>")
        add_dependencies(
            SimpleFluidSolvers SimpleFluidClangLtoCompatibility)
        target_link_options(SimpleFluidSolvers PRIVATE
            "$<$<CONFIG:Release,RelWithDebInfo>:$<TARGET_OBJECTS:SimpleFluidClangLtoCompatibility>>")
    endif()

endfunction()

function(simplefluid_configure_shared_linkage)
    # Source visibility attributes select the public SimpleFluid API.  The ELF
    # version script localizes embedded vendor archives except for a separately
    # versioned Kokkos runtime bridge: executables and the DSO must share Kokkos
    # initialization, allocation, and callback state.  Bind archive functions
    # locally while leaving runtime data canonical across the process.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(SIMPLEFLUID_ELF_VERSION_SCRIPT
            "${PROJECT_SOURCE_DIR}/cmake/linkage/maps/Linux.map")
        foreach(shared_target IN LISTS ARGN)
            target_link_options(${shared_target} PRIVATE
                "LINKER:--version-script=${SIMPLEFLUID_ELF_VERSION_SCRIPT}"
                "LINKER:-Bsymbolic-functions")
            if(SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME)
                # Shared dependencies can supply entries named by the static
                # runtime bridge, so those entries need not be defined here.
                target_link_options(${shared_target} PRIVATE "LINKER:--undefined-version")
                set(shared_trilinos_version_script
                    "${PROJECT_SOURCE_DIR}/cmake/linkage/maps/SharedTrilinosLinux.map")
                target_link_options(${shared_target} PRIVATE
                    "LINKER:--version-script=${shared_trilinos_version_script}")
                set_property(TARGET ${shared_target} APPEND PROPERTY
                    LINK_DEPENDS "${shared_trilinos_version_script}")
            endif()
            set_property(TARGET ${shared_target} APPEND PROPERTY
                LINK_DEPENDS "${SIMPLEFLUID_ELF_VERSION_SCRIPT}")
        endforeach()
    endif()

endfunction()

function(simplefluid_add_equations_linker_alias)
    # Preserve the former linker names for source-level consumers.  New binaries
    # record libSimpleFluidSolvers as the single implementation SONAME.
    if(UNIX)
        set(SIMPLEFLUID_EQUATIONS_LINKER_ALIAS
            "$<TARGET_FILE_DIR:SimpleFluidSolvers>/${CMAKE_SHARED_LIBRARY_PREFIX}SimpleFluidEquations${CMAKE_SHARED_LIBRARY_SUFFIX}")
        add_custom_command(TARGET SimpleFluidSolvers POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E rm -f
                    "${SIMPLEFLUID_EQUATIONS_LINKER_ALIAS}"
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "$<TARGET_FILE_NAME:SimpleFluidSolvers>"
                    "${SIMPLEFLUID_EQUATIONS_LINKER_ALIAS}"
            VERBATIM)
    endif()

endfunction()
