# Launch layer: the only place Kokkos is used in the solver (CLAUDE.md rule 3).
# The typed batches of each material entry in a translation unit of their own, compiled in
# parallel (internal_force_batch.hpp; nvcc compiled them in one file for 14 minutes). The count
# matches batch_impl::kSplitUnits.
set(_femorax_pair_units)
foreach(k RANGE 0 15)
  set(_unit "${CMAKE_CURRENT_BINARY_DIR}/internal_force_pairs_${k}.cpp")
  file(CONFIGURE OUTPUT "${_unit}" CONTENT
       "#include \"exec/internal_force_batch.hpp\"\nnamespace femorax::exec::batch_impl {\ntemplate void register_batches<${k}>(BatchFactory*);\n}\n")
  list(APPEND _femorax_pair_units "${_unit}")
endforeach()

add_library(femorax_exec
  ${_femorax_pair_units}
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
