include(GoogleTest)

# ---------------------------------------------------------------------------
# Unit tests: one executable, run by CTest per test case and, in addition, as a whole
# with 1 and 8 OpenMP threads so thread-count dependence shows up.
# ---------------------------------------------------------------------------
add_executable(femorax_unit_tests
  unit/main.cpp
  unit/test_contracts.cpp
  unit/test_runtime.cpp)
target_link_libraries(femorax_unit_tests PRIVATE femorax::exec GTest::gtest)
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
