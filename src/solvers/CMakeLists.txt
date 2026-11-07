# Linear solvers behind femorax::LinearSolver (docs/design.md, "Linear solvers"). Kokkos is
# allowed here and in src/exec/ only (CLAUDE.md rule 3); solver-library types never leave
# this directory (rule 6).
add_library(femorax_solvers
  vector_ops.cpp)
add_library(femorax::solvers ALIAS femorax_solvers)
target_include_directories(femorax_solvers PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(femorax_solvers PUBLIC Kokkos::kokkos)
femorax_set_warnings(femorax_solvers)
