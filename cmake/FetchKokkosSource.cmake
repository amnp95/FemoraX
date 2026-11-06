# Download the pinned Kokkos source once into build/_kokkos/, for builds that need Kokkos'
# nvcc_wrapper as their C++ compiler before configuring (CUDA presets).
#   cmake -P cmake/FetchKokkosSource.cmake
# The CUDA presets point CMAKE_CXX_COMPILER at its bin/nvcc_wrapper and
# FETCHCONTENT_SOURCE_DIR_KOKKOS at the same tree. Version and hash: cmake/FemoraXDeps.cmake.
set(version 5.1.0)
set(sha256 7bdbdfc88033ed7d940c7940ed8919e1f2b78a9656c69276beb76ad45c41ec4e)
get_filename_component(root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(dest "${root}/build/_kokkos")
if(EXISTS "${dest}/kokkos-${version}/bin/nvcc_wrapper")
  message(STATUS "Kokkos ${version} source present: ${dest}/kokkos-${version}")
  return()
endif()
file(MAKE_DIRECTORY "${dest}")
file(DOWNLOAD "https://github.com/kokkos/kokkos/releases/download/${version}/kokkos-${version}.tar.gz"
     "${dest}/kokkos-${version}.tar.gz" EXPECTED_HASH SHA256=${sha256} STATUS st)
list(GET st 0 code)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "download of Kokkos ${version} failed: ${st}")
endif()
file(ARCHIVE_EXTRACT INPUT "${dest}/kokkos-${version}.tar.gz" DESTINATION "${dest}")
message(STATUS "Kokkos ${version} source: ${dest}/kokkos-${version}")
