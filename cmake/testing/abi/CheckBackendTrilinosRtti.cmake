# Select the core families required by CheckSharedTrilinosRtti.cmake without
# applying a consumer's bridge limit to the backend's complete vendor ABI.
function(simplefluid_is_core_trilinos_rtti symbol output)
    if(symbol MATCHES "^_ZT[IS]N7Teuchos3any(11placeholderE|6holderIiEE)$"
       OR symbol MATCHES "^_ZT[IS]N6Tpetra(8Operator|9RowMatrix|9CrsMatrix)I"
       OR symbol MATCHES "^_ZT[IS]NSt3__112basic_stringIc")
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(simplefluid_check_backend_trilinos_rtti consumer_symbols)
    if(NOT EXISTS "${SIMPLEFLUID_RTTI_BACKEND_LIBRARY}"
       OR NOT EXISTS "${SIMPLEFLUID_READELF}"
       OR NOT DEFINED SIMPLEFLUID_RTTI_BACKEND_SONAME
       OR SIMPLEFLUID_RTTI_BACKEND_SONAME STREQUAL "")
        message(FATAL_ERROR "The RTTI backend, SONAME and ELF inspection tool must be available")
    endif()
    execute_process(
        COMMAND "${SIMPLEFLUID_READELF}" --dynamic "${SIMPLEFLUID_LIBRARY}"
        RESULT_VARIABLE result OUTPUT_VARIABLE dynamic ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Cannot inspect the RTTI consumer dependencies: ${error}")
    endif()
    string(REPLACE "\n" ";" dynamic_lines "${dynamic}")
    set(found_dependency FALSE)
    foreach(line IN LISTS dynamic_lines)
        if(line MATCHES "[(]NEEDED[)]")
            string(FIND "${line}" "[${SIMPLEFLUID_RTTI_BACKEND_SONAME}]" dependency)
            if(NOT dependency EQUAL -1)
                set(found_dependency TRUE)
            endif()
        endif()
    endforeach()
    if(NOT found_dependency)
        message(FATAL_ERROR "FVM does not depend on its declared RTTI backend: ${SIMPLEFLUID_RTTI_BACKEND_SONAME}")
    endif()
    execute_process(
        COMMAND "${SIMPLEFLUID_READELF}" --dynamic "${SIMPLEFLUID_RTTI_BACKEND_LIBRARY}"
        RESULT_VARIABLE result OUTPUT_VARIABLE dynamic ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Cannot inspect the RTTI backend SONAME: ${error}")
    endif()
    string(REGEX MATCH "[(]SONAME[)][^\n]*\\[([^]]+)\\]" soname_line "${dynamic}")
    if(NOT CMAKE_MATCH_1 STREQUAL SIMPLEFLUID_RTTI_BACKEND_SONAME)
        message(FATAL_ERROR "The RTTI backend SONAME does not match the declared provider")
    endif()

    execute_process(
        COMMAND "${SIMPLEFLUID_NM}" -D --defined-only -j "${SIMPLEFLUID_RTTI_BACKEND_LIBRARY}"
        RESULT_VARIABLE result OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Cannot inspect the RTTI backend exports: ${error}")
    endif()
    string(REPLACE "\n" ";" symbols "${symbols}")
    if(NOT "simplefluid_trilinos_backend_abi_version@@SIMPLEFLUID_TRILINOS_BACKEND_1.0" IN_LIST symbols)
        message(FATAL_ERROR "The RTTI backend must export its versioned ownership marker")
    endif()
    set(core_symbols)
    foreach(symbol IN LISTS symbols)
        string(REGEX REPLACE "@.*$" "" symbol "${symbol}")
        simplefluid_is_core_trilinos_rtti("${symbol}" is_core)
        if(is_core)
            list(APPEND core_symbols "${symbol}")
        endif()
    endforeach()
    simplefluid_check_shared_trilinos_rtti("${core_symbols}" TRUE)

    # A runtime provider must not conceal a private duplicate in the consumer.
    # Undefined default-visible references are imports; every local definition
    # in these families must also have the reviewed dynamic bridge export.
    execute_process(
        COMMAND "${SIMPLEFLUID_READELF}" --wide --symbols "${SIMPLEFLUID_LIBRARY}"
        RESULT_VARIABLE result OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Cannot inspect the RTTI consumer symbol table: ${error}")
    endif()
    if(NOT symbols MATCHES "Symbol table '[.]symtab'")
        message(FATAL_ERROR "The RTTI consumer needs its full .symtab to audit private core definitions")
    endif()
    string(REPLACE "\n" ";" symbols "${symbols}")
    foreach(line IN LISTS symbols)
        if(line MATCHES "^[ \t]*[0-9]+:[ \t]+[0-9A-Fa-f]+[ \t]+[0-9]+[ \t]+[^ \t]+[ \t]+([^ \t]+)[ \t]+([^ \t]+)[ \t]+([^ \t]+)[ \t]+([^ \t]+)")
            set(binding "${CMAKE_MATCH_1}")
            set(visibility "${CMAKE_MATCH_2}")
            set(section "${CMAKE_MATCH_3}")
            string(REGEX REPLACE "@.*$" "" symbol "${CMAKE_MATCH_4}")
            simplefluid_is_core_trilinos_rtti("${symbol}" is_core)
            if(is_core)
                if(NOT visibility STREQUAL "DEFAULT")
                    message(FATAL_ERROR "Private core Trilinos RTTI in FVM: ${line}")
                endif()
                if(NOT section STREQUAL "UND"
                   AND (NOT binding MATCHES "^(GLOBAL|WEAK|UNIQUE)$" OR NOT symbol IN_LIST consumer_symbols))
                    message(FATAL_ERROR "Core Trilinos RTTI lacks its shared FVM bridge export: ${line}")
                endif()
            endif()
        endif()
    endforeach()
endfunction()
