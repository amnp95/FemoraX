# Compile step: host code run once per analysis stage (DOF numbering, renumbering,
# batches, assembly maps, memory estimate). Plain C++, no Kokkos.
add_library(femorax_compile
  dof_numbering.cpp
  renumber.cpp
  batches.cpp
  sparsity.cpp
  drm.cpp)
add_library(femorax::compile ALIAS femorax_compile)
target_include_directories(femorax_compile PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(femorax_compile PUBLIC femorax::io femorax::physics)
femorax_set_warnings(femorax_compile)
