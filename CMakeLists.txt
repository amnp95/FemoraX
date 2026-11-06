cmake_minimum_required(VERSION 3.25)

project(FemoraX
  VERSION 0.0.1
  DESCRIPTION "GPU-native finite element solver for dynamic geotechnical analysis"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FEMORAX_BUILD_TESTS "Build the FemoraX test suite" ON)
option(FEMORAX_WERROR "Treat compiler warnings as errors in FemoraX targets" OFF)
option(FEMORAX_BUILD_BENCHMARKS "Build the benchmark suite" ON)
option(FEMORAX_USE_SYSTEM_KOKKOS "Use an installed Kokkos (must be the pinned version)" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FemoraXWarnings)
include(FemoraXDeps)

configure_file(src/femorax_version.hpp.in
               "${CMAKE_CURRENT_BINARY_DIR}/generated/femorax_version.hpp" @ONLY)

if(FEMORAX_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(src/physics)
add_subdirectory(src/io)
add_subdirectory(src/compile)
add_subdirectory(src/native)
add_subdirectory(src/exec)
add_subdirectory(src/app)
if(FEMORAX_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(FEMORAX_BUILD_TESTS)
  add_subdirectory(tests)
endif()
