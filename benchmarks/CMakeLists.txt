# Benchmark suite. Build it in a Release tree (cpu-release); timings from Debug trees are
# meaningless, and benchmarks/check.py refuses them.
# Benchmark-only Kokkos kernels (bandwidth and FP64 references, timing): not solver code.
add_library(femorax_bench_kernels
  kernels/stream.cpp
  kernels/peak.cpp
  kernels/timing.cpp)
target_include_directories(femorax_bench_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(femorax_bench_kernels PUBLIC Kokkos::kokkos)
femorax_set_warnings(femorax_bench_kernels)

add_executable(femorax_bench
  bench_main.cpp
  harness.cpp
  bench_stream.cpp
  bench_explicit.cpp
  bench_solvers.cpp
  flop_count.cpp)
target_link_libraries(femorax_bench PRIVATE femorax::exec femorax::solvers femorax::compile femorax_bench_kernels)
target_compile_definitions(femorax_bench PRIVATE FEMORAX_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
femorax_set_warnings(femorax_bench)

if(FEMORAX_BUILD_TESTS)
  # Smoke run: every benchmark runs and writes valid JSON. Not a timing check.
  add_test(NAME bench.smoke
           COMMAND femorax_bench --quick --reps 2 --warmup 1
                   --json ${CMAKE_CURRENT_BINARY_DIR}/smoke.json)
endif()
