# Model reader and (later) recorders. Host code; no Kokkos (CLAUDE.md rule 3).
add_library(femorax_io
  h5.cpp
  model_reader.cpp
  support.cpp)
add_library(femorax::io ALIAS femorax_io)
target_include_directories(femorax_io PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(femorax_io PRIVATE HDF5::HDF5)
femorax_set_warnings(femorax_io)
