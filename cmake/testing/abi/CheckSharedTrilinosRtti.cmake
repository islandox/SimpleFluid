function(simplefluid_check_shared_trilinos_rtti symbols required)
    list(LENGTH symbols symbol_count)
    if(symbol_count GREATER 512)
        message(FATAL_ERROR "Shared Trilinos RTTI exceeds the 512-symbol export limit")
    endif()
    set(demangled)
    if(symbols)
        execute_process(
            COMMAND "${SIMPLEFLUID_CXXFILT}" ${symbols}
            RESULT_VARIABLE result
            OUTPUT_VARIABLE demangled
            ERROR_VARIABLE error
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "Cannot demangle shared Trilinos RTTI: ${error}")
        endif()
    endif()
    string(REPLACE "\n" ";" names "${demangled}")
    foreach(name IN LISTS names)
        if(NOT name MATCHES "^typeinfo( name)? for (Teuchos|Tpetra|Xpetra|MueLu|Ifpack2|Belos|Thyra|NOX)::"
           AND NOT name MATCHES "^typeinfo( name)? for std::(__1::|__cxx11::)?basic_string<")
            message(FATAL_ERROR "Unexpected shared Trilinos RTTI export: ${name}")
        endif()
    endforeach()
    if(required)
        foreach(name IN ITEMS
                "typeinfo for Teuchos::any::placeholder"
                "typeinfo name for Teuchos::any::placeholder"
                "typeinfo for Teuchos::any::holder<int>"
                "typeinfo name for Teuchos::any::holder<int>")
            if(NOT name IN_LIST names)
                message(FATAL_ERROR "Missing shared Teuchos RTTI: ${name}")
            endif()
        endforeach()
        foreach(type IN ITEMS Operator RowMatrix CrsMatrix)
            if(NOT demangled MATCHES "typeinfo for Tpetra::${type}<")
                message(FATAL_ERROR "Missing shared Tpetra RTTI: ${type}")
            endif()
        endforeach()
        foreach(prefix IN ITEMS "typeinfo for" "typeinfo name for")
            if(NOT demangled MATCHES "${prefix} std::__1::basic_string<char,")
                message(FATAL_ERROR "Missing shared string RTTI: ${prefix} std::string")
            endif()
        endforeach()
    endif()
endfunction()
