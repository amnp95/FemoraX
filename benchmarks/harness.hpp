#pragma once

// Benchmark harness: times kernels (or whole time steps), reports time per call and the
// achieved memory bandwidth, and writes JSON for benchmarks/check.py to compare against a
// baseline. No Kokkos here: kernels come from src/exec/, timing goes through
// femorax::fence() and femorax::wall_time().

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace femorax::bench {

struct Benchmark {
  std::string name;          // unique, "group/case", e.g. "stream/triad"
  std::string unit = "call"; // what one run() is: "call" for a kernel, "step" for a step
  // Minimum memory traffic of one run() in bytes (compulsory loads and stores, counted from
  // the data structures, not measured). Achieved bandwidth = bytes / time.
  std::int64_t bytes = 0;
  double flops = 0.0;        // floating-point operations of one run(), 0 if not counted
  std::int64_t items = 0;    // work items of one run() (elements, entries, DOFs)
  std::function<void()> run; // one call or step; may launch asynchronously
  // Whether check.py fails on a slowdown of this case. Gate the paths that run (the
  // backend's default layout, the deterministic validation path, the bandwidth
  // reference); variants that only inform the layout choice are reported, not gated.
  bool gated = true;
};

struct Result {
  Benchmark bench;
  int reps = 0;
  double min_s = 0.0, median_s = 0.0, max_s = 0.0;
  double gbytes_per_s() const { return min_s > 0.0 ? bench.bytes / min_s * 1e-9 : 0.0; }
  double gflops() const { return min_s > 0.0 ? bench.flops / min_s * 1e-9 : 0.0; }
};

struct Options {
  int warmup = 3;
  int reps = 20;
  std::string filter;  // run only benchmarks whose name contains this
  bool quick = false;  // small problem sizes (for a fast smoke run in the test suite)
};

class Registry {
 public:
  void add(Benchmark b) { benchmarks_.push_back(std::move(b)); }
  const std::vector<Benchmark>& all() const { return benchmarks_; }

 private:
  std::vector<Benchmark> benchmarks_;
};

// Runs every registered benchmark that matches the filter: warmup runs, then reps timed
// runs, each bracketed by fences.
std::vector<Result> run_all(const Registry& registry, const Options& options);

void print_table(const std::vector<Result>& results);

// JSON with the machine description and every result (schema: benchmarks/README.md).
std::string to_json(const std::vector<Result>& results, const Options& options);

// Each benchmark source file defines one of these; bench_main.cpp calls them all.
void register_stream(Registry& registry, const Options& options);
void register_peak(Registry& registry, const Options& options);
void register_explicit(Registry& registry, const Options& options);
void register_solvers(Registry& registry, const Options& options);
void register_operator(Registry& registry, const Options& options);

}  // namespace femorax::bench
