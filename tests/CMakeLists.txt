include(GoogleTest)

# ---------------------------------------------------------------------------
# Unit tests: one executable, run by CTest per test case and, in addition, as a whole
# with 1 and 8 OpenMP threads so thread-count dependence shows up.
# ---------------------------------------------------------------------------
add_executable(femorax_unit_tests
  unit/main.cpp
  unit/test_contracts.cpp
  unit/test_elastic_isotropic.cpp
  unit/test_elastic_isotropic_reference.cpp
  unit/test_std_brick.cpp
  unit/test_std_brick_reference.cpp
  unit/test_runtime.cpp)
target_link_libraries(femorax_unit_tests PRIVATE femorax::exec GTest::gtest)
target_compile_definitions(femorax_unit_tests PRIVATE
  FEMORAX_REFERENCE_DATA="${PROJECT_SOURCE_DIR}/tests/reference/data")
femorax_set_warnings(femorax_unit_tests)

gtest_discover_tests(femorax_unit_tests
  TEST_PREFIX "unit."
  DISCOVERY_MODE PRE_TEST)

foreach(nthreads 1 8)
  add_test(NAME unit_all.threads${nthreads}
           COMMAND femorax_unit_tests --kokkos-num-threads=${nthreads})
endforeach()

# ---------------------------------------------------------------------------
# Physics purity (CLAUDE.md rule 2)
# 1. Source scan for forbidden constructs and includes.
# 2. Each physics header compiles on its own, without exceptions or RTTI and without
#    Kokkos on the include path (part of the default build, so a violation breaks it).
# ---------------------------------------------------------------------------
add_test(NAME physics_purity.scan
         COMMAND ${CMAKE_COMMAND} -DPHYSICS_DIR=${PROJECT_SOURCE_DIR}/src/physics
                 -P ${PROJECT_SOURCE_DIR}/cmake/CheckPhysicsPurity.cmake)

# The scanner must find every planted violation in the fixture (and nothing else:
# the fixture also holds allowed constructs).
add_test(NAME physics_purity.scan_rejects_violations
         COMMAND ${CMAKE_COMMAND}
                 -DPHYSICS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/purity/bad_physics
                 -P ${PROJECT_SOURCE_DIR}/cmake/CheckPhysicsPurity.cmake)
set_tests_properties(physics_purity.scan_rejects_violations PROPERTIES
  PASS_REGULAR_EXPRESSION "physics purity check: 14 violation\\(s\\)")

file(GLOB_RECURSE FEMORAX_PHYSICS_HEADERS CONFIGURE_DEPENDS
     "${PROJECT_SOURCE_DIR}/src/physics/*.hpp")
set(purity_sources "")
foreach(header IN LISTS FEMORAX_PHYSICS_HEADERS)
  file(RELATIVE_PATH rel "${PROJECT_SOURCE_DIR}/src" "${header}")
  string(MAKE_C_IDENTIFIER "${rel}" ident)
  set(tu "${CMAKE_CURRENT_BINARY_DIR}/purity/${ident}.cpp")
  file(CONFIGURE OUTPUT "${tu}" CONTENT "#include \"${rel}\"\n" @ONLY)
  list(APPEND purity_sources "${tu}")
endforeach()
add_library(femorax_physics_standalone OBJECT ${purity_sources})
target_link_libraries(femorax_physics_standalone PRIVATE femorax::physics)  # no Kokkos
target_compile_options(femorax_physics_standalone PRIVATE -fno-exceptions -fno-rtti)
femorax_set_warnings(femorax_physics_standalone)

# ---------------------------------------------------------------------------
# Command-line program
# ---------------------------------------------------------------------------
add_test(NAME app.version COMMAND femorax --version)
set_tests_properties(app.version PROPERTIES
  PASS_REGULAR_EXPRESSION "^femorax ${PROJECT_VERSION}")
add_test(NAME app.info COMMAND femorax info)
set_tests_properties(app.info PROPERTIES
  PASS_REGULAR_EXPRESSION "Kokkos ${FEMORAX_KOKKOS_VERSION}.*execution spaces:.*Serial.*OpenMP")
add_test(NAME app.unknown_command COMMAND femorax frobnicate)
set_tests_properties(app.unknown_command PROPERTIES WILL_FAIL TRUE)

# ---------------------------------------------------------------------------
# Model reader: fixtures written by Python (h5py, from .venv), read by C++.
# ---------------------------------------------------------------------------
set(FEMORAX_IO_FIXTURES "${CMAKE_CURRENT_BINARY_DIR}/io_fixtures")
add_test(NAME io.make_fixtures
         COMMAND ${FEMORAX_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/io/make_fixtures.py
                 ${FEMORAX_IO_FIXTURES})
set_tests_properties(io.make_fixtures PROPERTIES FIXTURES_SETUP io_fixtures)

add_executable(femorax_io_tests
  io/test_model_reader.cpp)
target_link_libraries(femorax_io_tests PRIVATE femorax::io GTest::gtest_main)
femorax_set_warnings(femorax_io_tests)
gtest_discover_tests(femorax_io_tests
  TEST_PREFIX "io."
  DISCOVERY_MODE PRE_TEST
  PROPERTIES FIXTURES_REQUIRED io_fixtures
             ENVIRONMENT "FEMORAX_IO_FIXTURES=${FEMORAX_IO_FIXTURES}")

# The command-line check: exit 0 for a valid model, 3 unsupported, 4 invalid.
foreach(case IN ITEMS "explicit;0" "implicit_newmark;3" "duplicate_node;4")
  list(GET case 0 name)
  list(GET case 1 code)
  add_test(NAME app.check.${name}
           COMMAND ${CMAKE_COMMAND} -DCMD=$<TARGET_FILE:femorax>
                   -DARGS=check\;${FEMORAX_IO_FIXTURES}/${name}.femorax.h5
                   -DEXPECTED=${code}
                   -P ${PROJECT_SOURCE_DIR}/cmake/RunExpectExit.cmake)
  set_tests_properties(app.check.${name} PROPERTIES FIXTURES_REQUIRED io_fixtures)
endforeach()
