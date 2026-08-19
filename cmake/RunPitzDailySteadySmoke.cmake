if(NOT DEFINED SIMPLEFLUID_PITZ_DAILY_EXECUTABLE OR
   SIMPLEFLUID_PITZ_DAILY_EXECUTABLE STREQUAL "")
  message(FATAL_ERROR "pitzDaily steady smoke requires an executable")
endif()
if(NOT DEFINED SIMPLEFLUID_PITZ_WORKING_DIRECTORY OR
   SIMPLEFLUID_PITZ_WORKING_DIRECTORY STREQUAL "")
  message(FATAL_ERROR "pitzDaily steady smoke requires a working directory")
endif()
file(MAKE_DIRECTORY "${SIMPLEFLUID_PITZ_WORKING_DIRECTORY}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
          SIMPLEFLUID_PITZ_MESH_DIVISOR=40
          SIMPLEFLUID_PITZ_STEPS=2
          SIMPLEFLUID_PITZ_DT=1e-6
          SIMPLEFLUID_PITZ_STEADY_STATE=1
          SIMPLEFLUID_PITZ_STEADY_TOLERANCE=1e20
          SIMPLEFLUID_PITZ_STEADY_PROGRESS_INTERVAL=1
          "${SIMPLEFLUID_PITZ_DAILY_EXECUTABLE}"
  WORKING_DIRECTORY "${SIMPLEFLUID_PITZ_WORKING_DIRECTORY}"
  RESULT_VARIABLE simplefluid_pitz_result
  OUTPUT_VARIABLE simplefluid_pitz_output
  ERROR_VARIABLE simplefluid_pitz_error)

if(NOT simplefluid_pitz_result STREQUAL "0")
  message(FATAL_ERROR
    "pitzDaily steady smoke exited ${simplefluid_pitz_result}\n"
    "stdout:\n${simplefluid_pitz_output}\n"
    "stderr:\n${simplefluid_pitz_error}")
endif()

if(NOT simplefluid_pitz_output MATCHES
   "steady_step=1/2[^\n]* requested_linear_tolerance=1\\.000000e-06[^\n]* next_requested_linear_tolerance=1\\.000000e-09")
  message(FATAL_ERROR
    "pitzDaily did not tighten the first accepted step to the final tolerance\n"
    "stdout:\n${simplefluid_pitz_output}")
endif()

if(NOT simplefluid_pitz_output MATCHES
   "steady_step=2/2[^\n]* requested_linear_tolerance=1\\.000000e-09")
  message(FATAL_ERROR
    "pitzDaily did not apply the tightened tolerance on the next step\n"
    "stdout:\n${simplefluid_pitz_output}")
endif()
