# Benchmark suite. Build it in a Release tree (cpu-release); timings from Debug trees are
# meaningless, and benchmarks/check.py refuses them.
add_executable(femorax_bench
  bench_main.cpp
  harness.cpp
  bench_stream.cpp
  bench_explicit.cpp
  flop_count.cpp)
target_link_libraries(femorax_bench PRIVATE femorax::exec femorax::compile)
target_compile_definitions(femorax_bench PRIVATE FEMORAX_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
femorax_set_warnings(femorax_bench)

if(FEMORAX_BUILD_TESTS)
  # Smoke run: every benchmark runs and writes valid JSON. Not a timing check.
  add_test(NAME bench.smoke
           COMMAND femorax_bench --quick --reps 2 --warmup 1
                   --json ${CMAKE_CURRENT_BINARY_DIR}/smoke.json)
endif()
