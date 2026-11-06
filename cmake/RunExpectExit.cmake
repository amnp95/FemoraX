# Run CMD with ARGS (a ;-list) and fail unless it exits with EXPECTED.
#   cmake -DCMD=<exe> -DARGS=<a;b> -DEXPECTED=<code> -P RunExpectExit.cmake
execute_process(COMMAND ${CMD} ${ARGS} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc STREQUAL EXPECTED)
  message(FATAL_ERROR "expected exit ${EXPECTED}, got ${rc}\nstdout:\n${out}\nstderr:\n${err}")
endif()
message(STATUS "exit ${rc} as expected\n${out}${err}")
