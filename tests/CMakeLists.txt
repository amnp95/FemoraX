include(GoogleTest)

# Eigen: test-only reference solver (docs/design.md, "Linear solvers"; docs/dependencies.md).
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

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
  unit/test_ssp_brick.cpp
  unit/test_ssp_brick_reference.cpp
  unit/test_pdmy.cpp
  unit/test_pdmy_reference.cpp
  unit/test_uniform_damping.cpp
  unit/test_memory_estimate.cpp
  unit/test_pimy.cpp
  unit/test_pimy_reference.cpp
  unit/test_material_state.cpp
  unit/test_memory_contract.cpp
  unit/test_dof_numbering.cpp
  unit/test_renumber.cpp
  unit/test_internal_force.cpp
  unit/test_ground_motion.cpp
  unit/test_vector_ops.cpp
  unit/test_sparsity.cpp
  unit/test_reference_assembly.cpp
  unit/test_matrix_free_operator.cpp
  unit/test_cg.cpp
  unit/test_gmres.cpp
  unit/test_implicit.cpp
  unit/test_runtime.cpp
  unit/test_drm_setup.cpp
  unit/test_drm_free_field.cpp
  unit/test_local_time_stepping.cpp
  unit/test_lts_levels.cpp
  unit/test_lts_3d.cpp
  unit/test_elastic_section_3d.cpp
  unit/test_crd_transf_3d.cpp
  unit/test_disp_beam_3d.cpp
  unit/test_force_beam_3d.cpp
  unit/test_uniaxial.cpp
  unit/test_fiber_section_3d.cpp
  unit/test_beam_kernels.cpp
  unit/test_linear_constraints.cpp)
target_link_libraries(femorax_unit_tests PRIVATE femorax::exec femorax::solvers femorax::compile GTest::gtest
                      Eigen3::Eigen HDF5::HDF5)
# test_contracts.cpp declares mock elements and materials whose traits only the concepts
# read; Clang-based compilers (Intel icpx) report them as unused constants.
set_source_files_properties(unit/test_contracts.cpp PROPERTIES COMPILE_OPTIONS
  "$<$<CXX_COMPILER_ID:Clang,IntelLLVM>:-Wno-unused-const-variable;-Wno-unneeded-internal-declaration>")
target_compile_definitions(femorax_unit_tests PRIVATE
  FEMORAX_REFERENCE_DATA="${PROJECT_SOURCE_DIR}/tests/reference/data")
femorax_set_warnings(femorax_unit_tests)

# ctest runs these cases in parallel (-j); with every case on all hardware threads the
# OpenMP threads oversubscribe the machine and spin-wait each other to a crawl (the implicit
# tests launch many small kernels). Each case gets 2 threads; thread-count dependence is
# covered by the whole-binary runs below.
gtest_discover_tests(femorax_unit_tests
  TEST_PREFIX "unit."
  DISCOVERY_MODE PRE_TEST
  TEST_FILTER "-MemoryEstimate.*"
  PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
# The memory estimate compares with the device's free memory: alone on the machine, because
# other tests on the same GPU change what it reports (Intel's runtime then reported 0 bytes).
add_test(NAME unit.MemoryEstimate.WithinTenPercentOfTheAllocatedPeak
         COMMAND femorax_unit_tests --gtest_filter=MemoryEstimate.*)
set_tests_properties(unit.MemoryEstimate.WithinTenPercentOfTheAllocatedPeak PROPERTIES
  RUN_SERIAL TRUE ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# OpenMP release builds: the whole binary again with 1 and 8 threads (thread-count
# dependence). Not in Debug builds (debug, UBSan, ASan): there every case already ran on its
# own, and the repeats cost most of their test time (single-threaded UBSan: 23 minutes).
if(Kokkos_ENABLE_OPENMP AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  foreach(nthreads 1 8)
    add_test(NAME unit_all.threads${nthreads}
             COMMAND femorax_unit_tests --kokkos-num-threads=${nthreads})
    # The 8-thread run alone on the machine, for the same reason.
    if(nthreads GREATER 1)
      set_tests_properties(unit_all.threads${nthreads} PROPERTIES RUN_SERIAL TRUE)
    endif()
  endforeach()
endif()

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
set(expected_spaces "Serial")
if(Kokkos_ENABLE_OPENMP)
  string(APPEND expected_spaces ".*OpenMP")
endif()
if(Kokkos_ENABLE_CUDA)
  string(APPEND expected_spaces ".*Cuda")
endif()
if(Kokkos_ENABLE_SYCL)
  string(APPEND expected_spaces ".*SYCL")
endif()
set_tests_properties(app.info PROPERTIES
  PASS_REGULAR_EXPRESSION "Kokkos ${FEMORAX_KOKKOS_VERSION}.*execution spaces:.*${expected_spaces}")
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
foreach(case IN ITEMS "explicit;0" "implicit_newmark;0" "element_type;3" "duplicate_node;4")
  list(GET case 0 name)
  list(GET case 1 code)
  add_test(NAME app.check.${name}
           COMMAND ${CMAKE_COMMAND} -DCMD=$<TARGET_FILE:femorax>
                   -DARGS=check\;${FEMORAX_IO_FIXTURES}/${name}.femorax.h5
                   -DEXPECTED=${code}
                   -P ${PROJECT_SOURCE_DIR}/cmake/RunExpectExit.cmake)
  set_tests_properties(app.check.${name} PROPERTIES FIXTURES_REQUIRED io_fixtures)
endforeach()

# System test: VTKHDF output read by femora.results, against OpenSees. Needs the reference
# build with HDF5 >= 1.12 (tests/reference/README.md) and Femora in .venv
# (docs/dependencies.md); skipped otherwise.
set(FEMORAX_OPENSEES "${PROJECT_SOURCE_DIR}/build/reference/opensees/OpenSees" CACHE FILEPATH
    "OpenSees executable of the reference build")
set(FEMORAX_VENV_PYTHON "${PROJECT_SOURCE_DIR}/.venv/bin/python")
# Machines whose presets name another interpreter (Vista: .venv-vista; /work is shared with
# Stampede3, whose .venv does not resolve there): the stored-reference system tests use it. Before
# 2026-11-21 they were silently not registered on Vista.
if(NOT EXISTS "${FEMORAX_VENV_PYTHON}" AND EXISTS "${FEMORAX_PYTHON}")
  set(FEMORAX_VENV_PYTHON "${FEMORAX_PYTHON}")
endif()
# The tests that read results with femora.results or export with Femora need Femora in that
# interpreter (docs/dependencies.md); the others need numpy and h5py only.
set(FEMORAX_HAVE_FEMORA FALSE)
if(EXISTS "${FEMORAX_VENV_PYTHON}")
  execute_process(COMMAND ${FEMORAX_VENV_PYTHON} -c "import femora" RESULT_VARIABLE _femora_rc
                  OUTPUT_QUIET ERROR_QUIET)
  if(_femora_rc EQUAL 0)
    set(FEMORAX_HAVE_FEMORA TRUE)
  else()
    message(STATUS "FemoraX tests: no femora in ${FEMORAX_VENV_PYTHON}: the system tests that need "
                   "it are not registered (vtkhdf_single_precision, nonlinear, wide_column, femora_example)")
  endif()
endif()
if(EXISTS "${FEMORAX_OPENSEES}" AND EXISTS "${FEMORAX_VENV_PYTHON}")
  add_test(NAME system.vtkhdf_vs_opensees
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/soil_column/vtkhdf_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_OPENSEES} ${FEMORAX_VENV_PYTHON}
                   ${CMAKE_CURRENT_BINARY_DIR}/system_vtkhdf)
  set_tests_properties(system.vtkhdf_vs_opensees PROPERTIES TIMEOUT 300)
  # Implicit stages (consistent mass; dynamic and static gravity; Linear and Newton).
  add_test(NAME system.implicit_vs_opensees
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/soil_column/implicit_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_OPENSEES} ${FEMORAX_VENV_PYTHON}
                   ${CMAKE_CURRENT_BINARY_DIR}/system_implicit)
  set_tests_properties(system.implicit_vs_opensees PROPERTIES TIMEOUT 600)
endif()
# Single-precision VTKHDF output against the double output of the same run (no OpenSees).
if(FEMORAX_HAVE_FEMORA)
  add_test(NAME system.vtkhdf_single_precision
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/soil_column/single_precision_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_VENV_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/system_single_precision)
  set_tests_properties(system.vtkhdf_single_precision PROPERTIES TIMEOUT 300)
endif()
# Nonlinear (PressureDependMultiYield) cases against stored OpenSees output: OpenSees does
# not run (tests/system/soil_column/nonlinear_references.sh stores it).
if(EXISTS "${FEMORAX_VENV_PYTHON}")
  # A device build (CUDA, HIP or SYCL) compares at the GPU tolerances.
  if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    set(femorax_device_build TRUE)
  else()
    set(femorax_device_build FALSE)
  endif()
  if(femorax_device_build)
    set(femorax_backend gpu)
  else()
    set(femorax_backend host)
  endif()
  if(FEMORAX_HAVE_FEMORA)
  add_test(NAME system.nonlinear_vs_stored_opensees
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/soil_column/nonlinear_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_VENV_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/system_nonlinear
                   ${femorax_backend})
  set_tests_properties(system.nonlinear_vs_stored_opensees PROPERTIES TIMEOUT 900)
  # A wide column (n x n x 25, laminar sides) against the stored 1 x 1 x 25 OpenSees output:
  # every vertical line must move as the single column does (docs/deviations.md, "Wide
  # columns"). 30 x 30 on GPUs (about 2 minutes on the laptop); 6 x 6 in host Release builds
  # (about 2 minutes; the 30 x 30 run takes 15). Not in Debug builds (cpu-debug, cpu-asan),
  # where it would take far longer than the rest of the suite.
  if(femorax_device_build)
    set(femorax_wide_n 30)
  else()
    set(femorax_wide_n 6)
  endif()
  if(femorax_device_build OR NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_test(NAME system.wide_column_vs_stored_opensees
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/soil_column/wide_column_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_VENV_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/system_wide_column
                   ${femorax_wide_n} 2e-5)
  set_tests_properties(system.wide_column_vs_stored_opensees PROPERTIES TIMEOUT 900)
  endif()
  # Femora's own nonlinear layered soil column (its example, exported by Femora, on a 2 s
  # window of its motion) against stored OpenSees output (femora_example_reference.sh).
  add_test(NAME system.femora_example_vs_stored_opensees
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/femora_example/femora_example_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_VENV_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/system_femora_example
                   ${femorax_backend})
  set_tests_properties(system.femora_example_vs_stored_opensees PROPERTIES TIMEOUT 900)
  endif()  # FEMORAX_HAVE_FEMORA
  # The converter's beam commands (schema 1.1) end to end: a Tcl cantilever converted, checked
  # and run implicit and explicit (tests/system/beam_cantilever/beam_check.sh).
  add_test(NAME system.beam_cantilever_conversion
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/beam_cantilever/beam_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_VENV_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/system_beam_cantilever)
  # The converter's force-beam command (schema 1.3) end to end: the same cantilever as
  # nonlinearBeamColumn elements, against its displacement-beam run (an elastic prismatic member:
  # the same stiffness and mass; tests/system/beam_cantilever/force_beam_check.sh). GPUs: rounding
  # of the solves.
  if(femorax_device_build)
    set(femorax_force_beam_tol 1e-7)
  else()
    set(femorax_force_beam_tol 1e-9)
  endif()
  add_test(NAME system.force_beam_cantilever_conversion
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/beam_cantilever/force_beam_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_VENV_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/system_force_beam_cantilever
                   ${femorax_force_beam_tol})
  # A pile tied into a soil column against stored OpenSees output (beams, bricks, equalDOF,
  # Newmark; tests/system/pile_in_soil/pile_check.sh). GPUs: rounding of the solves.
  if(femorax_device_build)
    set(femorax_pile_tol 1e-8)
  else()
    set(femorax_pile_tol 1e-12)
  endif()
  add_test(NAME system.pile_in_soil_vs_stored_opensees
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/pile_in_soil/pile_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_VENV_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/system_pile_in_soil
                   ${femorax_pile_tol})
  # The same pile as force-based beams (schema 1.3) against OpenSees' own nonlinearBeamColumn run
  # of it (stored; pile_check.sh, "force").
  add_test(NAME system.force_beam_pile_in_soil_vs_stored_opensees
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/pile_in_soil/pile_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_VENV_PYTHON}
                   ${CMAKE_CURRENT_BINARY_DIR}/system_force_beam_pile_in_soil ${femorax_pile_tol} force)
  # Rigid links, a rigid diaphragm and embedded nodes against stored OpenSees output: the frame
  # to rounding, the embedded pile converging to FemoraX's exact tie as 1 / K
  # (tests/system/constraints/constraints_check.sh).
  add_test(NAME system.constraints_vs_stored_opensees
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/constraints/constraints_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_VENV_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/system_constraints
                   ${femorax_pile_tol})
  # A two-storey 3D frame, one force-based beam per member (PDelta columns, joint offsets, 3 to 7
  # Lobatto points), against OpenSees' stored nonlinearBeamColumn run
  # (tests/system/force_beam_frame/frame_check.sh).
  add_test(NAME system.force_beam_frame_vs_stored_opensees
           COMMAND ${PROJECT_SOURCE_DIR}/tests/system/force_beam_frame/frame_check.sh
                   $<TARGET_FILE:femorax> ${FEMORAX_VENV_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/system_force_beam_frame
                   ${femorax_pile_tol})
endif()
