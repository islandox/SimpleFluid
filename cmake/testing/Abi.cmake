include_guard(GLOBAL)

function(simplefluid_add_abi_tests)
    # Keep the export map enforceable as the static Trilinos dependency set evolves.
    if(BUILD_TESTING
       AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Clang-built Trilinos emits substantially more weak Kokkos template
        # definitions in Debug than GCC-built archives.  Both sets are part of
        # the intentional, versioned runtime bridge; retain a tighter bound for
        # GCC while allowing the validated Clang archive shape.
        set(SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING 7000)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING 20000)
        endif()
        set(SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING 64)
        find_program(SIMPLEFLUID_CXXFILT
            NAMES c++filt llvm-cxxfilt
            REQUIRED)
        set(SIMPLEFLUID_RUNTIME_TARGET SimpleFluidFVM)
        if(TARGET SimpleFluidTrilinosBackend)
            set(SIMPLEFLUID_RUNTIME_TARGET SimpleFluidTrilinosBackend)
        elseif(Trilinos_BUILD_SHARED_LIBS)
            set(SIMPLEFLUID_RUNTIME_TARGET Kokkos::kokkoscore)
        endif()
        add_test(NAME simplefluid_elf_export_boundary
            COMMAND "${CMAKE_COMMAND}"
                "-DSIMPLEFLUID_LIBRARY=$<TARGET_FILE:SimpleFluidSolvers>"
                "-DSIMPLEFLUID_RUNTIME_LIBRARY=$<TARGET_FILE:${SIMPLEFLUID_RUNTIME_TARGET}>"
                "-DSIMPLEFLUID_RUNTIME_SONAME=$<TARGET_SONAME_FILE_NAME:${SIMPLEFLUID_RUNTIME_TARGET}>"
                "-DSIMPLEFLUID_TRILINOS_SHARED=$<BOOL:${Trilinos_BUILD_SHARED_LIBS}>"
                "-DSIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME=$<BOOL:${SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME}>"
                "-DSIMPLEFLUID_NM=${CMAKE_NM}"
                "-DSIMPLEFLUID_READELF=${CMAKE_READELF}"
                "-DSIMPLEFLUID_CXXFILT=${SIMPLEFLUID_CXXFILT}"
                "-DSIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING=${SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING}"
                "-DSIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING=${SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING}"
                "-DSIMPLEFLUID_REQUIRE_STATIC_LIBCXX_RTTI_BRIDGES=$<AND:$<BOOL:${SIMPLEFLUID_CXX_USES_LIBCXX}>,$<NOT:$<BOOL:${SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME}>>>"
                "-DSIMPLEFLUID_REQUIRE_SHARED_LIBCXX_RTTI_BRIDGES=$<AND:$<BOOL:${SIMPLEFLUID_CXX_USES_LIBCXX}>,$<BOOL:${SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME}>>"
                -P "${PROJECT_SOURCE_DIR}/cmake/testing/abi/CheckElfExports.cmake")
        set_tests_properties(simplefluid_elf_export_boundary PROPERTIES
            LABELS abi)
        # LLVM's demangler understands Clang's constrained-template encodings.
        find_program(SIMPLEFLUID_FVM_CXXFILT
            NAMES llvm-cxxfilt c++filt
            REQUIRED)
        add_test(NAME simplefluid_fvm_elf_export_boundary
            COMMAND "${CMAKE_COMMAND}"
                "-DSIMPLEFLUID_LIBRARY=$<TARGET_FILE:SimpleFluidFVM>"
                "-DSIMPLEFLUID_NM=${CMAKE_NM}"
                "-DSIMPLEFLUID_CXXFILT=${SIMPLEFLUID_FVM_CXXFILT}"
                "-DSIMPLEFLUID_REQUIRE_SHARED_LIBCXX_RTTI_BRIDGES=$<AND:$<BOOL:${SIMPLEFLUID_CXX_USES_LIBCXX}>,$<BOOL:${SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME}>>"
                "-DSIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING=${SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING}"
                "-DSIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING=${SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING}"
                -P "${PROJECT_SOURCE_DIR}/cmake/testing/abi/CheckFvmExports.cmake")
        set_tests_properties(simplefluid_fvm_elf_export_boundary PROPERTIES
            LABELS abi)
        if(TARGET SimpleFluidTrilinosBackend)
            set(backend_link_rules "")
            if(CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
                set(backend_link_rules "${CMAKE_BINARY_DIR}/CMakeFiles/impl-$<CONFIG>.ninja")
            elseif(CMAKE_GENERATOR STREQUAL "Ninja")
                set(backend_link_rules "${CMAKE_BINARY_DIR}/build.ninja")
            endif()
            get_filename_component(backend_trilinos_prefix "${Trilinos_DIR}/../../.." ABSOLUTE)
            add_test(NAME simplefluid_trilinos_backend_boundary
                COMMAND "${CMAKE_COMMAND}"
                    "-DSIMPLEFLUID_BACKEND_LIBRARY=$<TARGET_FILE:SimpleFluidTrilinosBackend>"
                    "-DSIMPLEFLUID_NM=${CMAKE_NM}"
                    "-DSIMPLEFLUID_READELF=${CMAKE_READELF}"
                    "-DSIMPLEFLUID_LINK_RULES=${backend_link_rules}"
                    "-DSIMPLEFLUID_TRILINOS_PREFIX=${backend_trilinos_prefix}"
                    -P "${PROJECT_SOURCE_DIR}/cmake/testing/abi/CheckTrilinosBackend.cmake")
            set_tests_properties(simplefluid_trilinos_backend_boundary PROPERTIES LABELS abi)
        endif()
    endif()

endfunction()
