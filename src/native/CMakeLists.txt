# Native vendor code (CLAUDE.md rule 3): CUDA/HIP kernels that a profile shows are needed,
# with portable fallbacks, and device queries Kokkos does not expose.
if(Kokkos_ENABLE_CUDA)
  add_library(femorax_native device_memory_cuda.cpp)
elseif(Kokkos_ENABLE_HIP)
  # Not built or tested yet: no ROCm machine. hipMemGetInfo has the same contract.
  add_library(femorax_native device_memory_hip.cpp)
else()
  add_library(femorax_native INTERFACE)
endif()
if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
  target_include_directories(femorax_native PUBLIC "${PROJECT_SOURCE_DIR}/src")
  femorax_set_warnings(femorax_native)
endif()
add_library(femorax::native ALIAS femorax_native)
