# Launch layer: the only place Kokkos is used in the solver (CLAUDE.md rule 3).
add_library(femorax_exec
  assembly.cpp
  device_memory.cpp
  explicit_analysis.cpp
  drm_load.cpp
  ground_motion.cpp
  implicit_stage.cpp
  recorders.cpp
  force_settings.cpp
  internal_force.cpp
  matrix_free_operator.cpp
  runtime.cpp)
add_library(femorax::exec ALIAS femorax_exec)
target_include_directories(femorax_exec PUBLIC
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_BINARY_DIR}/generated")
target_link_libraries(femorax_exec PUBLIC femorax::physics femorax::compile femorax::io femorax::solvers Kokkos::kokkos
                      PRIVATE femorax::native)
femorax_set_warnings(femorax_exec)
