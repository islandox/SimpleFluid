if(NOT EXISTS "${SIMPLEFLUID_BACKEND_LIBRARY}")
    message(FATAL_ERROR "The static Trilinos backend shared library is missing")
endif()
execute_process(
    COMMAND "${SIMPLEFLUID_NM}" -D --defined-only -j "${SIMPLEFLUID_BACKEND_LIBRARY}"
    RESULT_VARIABLE nm_result
    OUTPUT_VARIABLE symbols)
if(NOT nm_result EQUAL 0 OR NOT symbols MATCHES
   "simplefluid_trilinos_backend_abi_version@@SIMPLEFLUID_TRILINOS_BACKEND_1[.]0")
    message(FATAL_ERROR "The backend must export its versioned ownership marker")
endif()
execute_process(
    COMMAND "${SIMPLEFLUID_READELF}" --dynamic "${SIMPLEFLUID_BACKEND_LIBRARY}"
    RESULT_VARIABLE readelf_result
    OUTPUT_VARIABLE dynamic)
if(NOT readelf_result EQUAL 0)
    message(FATAL_ERROR "Cannot inspect the Trilinos backend dependencies")
endif()
if(dynamic MATCHES "NEEDED[^\n]*lib(tpetra|teuchos|kokkos|muelu|xpetra|belos|ifpack2|nox|stk_)[^\n]*[.]so")
    message(FATAL_ERROR "The static backend unexpectedly depends on shared Trilinos libraries")
endif()

if(DEFINED SIMPLEFLUID_LINK_RULES AND NOT SIMPLEFLUID_LINK_RULES STREQUAL "")
    file(READ "${SIMPLEFLUID_LINK_RULES}" rules)
    string(REGEX MATCHALL "LINK_LIBRARIES = [^\n]*" link_lines "${rules}")
    set(archive_owners 0)
    foreach(line IN LISTS link_lines)
        string(FIND "${line}" "${SIMPLEFLUID_TRILINOS_PREFIX}/" has_trilinos)
        if(NOT has_trilinos EQUAL -1)
            if(NOT line MATCHES "--whole-archive" OR NOT line MATCHES "--no-whole-archive")
                message(FATAL_ERROR "A consumer link rule still receives static Trilinos: ${line}")
            endif()
            math(EXPR archive_owners "${archive_owners} + 1")
        endif()
    endforeach()
    if(NOT archive_owners EQUAL 1)
        message(FATAL_ERROR "Expected one Trilinos archive owner; found ${archive_owners}")
    endif()
endif()
message(STATUS "Static Trilinos is owned and exported by the SimpleFluid backend")
