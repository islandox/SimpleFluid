include_guard(GLOBAL)

# Register an MPI executable with CTest while applying the project-wide process
# cap and the launcher flags reported by FindMPI. Additional PREFLAGS and
# POSTFLAGS augment, rather than replace, the launcher configuration.
function(simplefluid_add_mpi_test)
    set(_options)
    set(_one_value_args NAME TARGET PROCS TIMEOUT WORKING_DIRECTORY)
    set(_multi_value_args ARGS LABELS PREFLAGS POSTFLAGS)
    cmake_parse_arguments(PARSE_ARGV 0 SF_MPI
        "${_options}" "${_one_value_args}" "${_multi_value_args}")

    if(SF_MPI_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "simplefluid_add_mpi_test received unknown arguments: "
            "${SF_MPI_UNPARSED_ARGUMENTS}")
    endif()
    foreach(_required_arg IN ITEMS NAME TARGET PROCS)
        if(NOT SF_MPI_${_required_arg})
            message(FATAL_ERROR
                "simplefluid_add_mpi_test requires ${_required_arg}")
        endif()
    endforeach()
    if(NOT SF_MPI_PROCS MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR
            "simplefluid_add_mpi_test PROCS must be a positive integer; got "
            "'${SF_MPI_PROCS}' for ${SF_MPI_NAME}")
    endif()
    if(NOT TARGET "${SF_MPI_TARGET}")
        message(FATAL_ERROR
            "simplefluid_add_mpi_test target does not exist: "
            "${SF_MPI_TARGET}")
    endif()

    if(NOT MPIEXEC_EXECUTABLE)
        message(STATUS
            "Skipping MPI test ${SF_MPI_NAME}: MPI launcher is unavailable")
        return()
    endif()
    if(NOT SIMPLEFLUID_MAX_TEST_PROCS EQUAL 0
       AND SF_MPI_PROCS GREATER SIMPLEFLUID_MAX_TEST_PROCS)
        message(STATUS
            "Skipping MPI test ${SF_MPI_NAME}: requires ${SF_MPI_PROCS} "
            "processes, cap is ${SIMPLEFLUID_MAX_TEST_PROCS}")
        return()
    endif()

    add_test(
        NAME "${SF_MPI_NAME}"
        COMMAND "${MPIEXEC_EXECUTABLE}"
            "${MPIEXEC_NUMPROC_FLAG}" "${SF_MPI_PROCS}"
            ${MPIEXEC_PREFLAGS} ${SF_MPI_PREFLAGS}
            "$<TARGET_FILE:${SF_MPI_TARGET}>"
            ${MPIEXEC_POSTFLAGS} ${SF_MPI_POSTFLAGS}
            ${SF_MPI_ARGS})

    set(_labels ${SF_MPI_LABELS})
    list(APPEND _labels mpi)
    list(REMOVE_DUPLICATES _labels)
    set_tests_properties("${SF_MPI_NAME}" PROPERTIES
        LABELS "${_labels}"
        PROCESSORS "${SF_MPI_PROCS}")
    if(SF_MPI_TIMEOUT)
        set_tests_properties("${SF_MPI_NAME}" PROPERTIES
            TIMEOUT "${SF_MPI_TIMEOUT}")
    endif()
    if(SF_MPI_WORKING_DIRECTORY)
        set_tests_properties("${SF_MPI_NAME}" PROPERTIES
            WORKING_DIRECTORY "${SF_MPI_WORKING_DIRECTORY}")
    endif()
endfunction()
