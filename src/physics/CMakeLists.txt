# Physics layer: header-only, framework-free (CLAUDE.md rule 2).
# This target must never link Kokkos or any other library.
add_library(femorax_physics INTERFACE)
add_library(femorax::physics ALIAS femorax_physics)
target_include_directories(femorax_physics INTERFACE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(femorax_physics INTERFACE cxx_std_20)
