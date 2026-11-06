# Launch layer: the only place Kokkos is used (CLAUDE.md rule 3).
add_library(femorax_exec
  device_memory.cpp
  explicit_force.cpp
  force_settings.cpp
  peak.cpp
  runtime.cpp
  stream.cpp
  timing.cpp)
add_library(femorax::exec ALIAS femorax_exec)
target_include_directories(femorax_exec PUBLIC
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_BINARY_DIR}/generated")
target_link_libraries(femorax_exec PUBLIC femorax::physics femorax::compile Kokkos::kokkos
                      PRIVATE femorax::native)
femorax_set_warnings(femorax_exec)
