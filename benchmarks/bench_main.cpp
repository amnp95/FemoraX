// femorax_bench: run the benchmark suite.
//
//   femorax_bench [--json out.json] [--filter text] [--reps N] [--warmup N] [--quick]
//                 [--kokkos-...]
//
// Use benchmarks/check.py to run it against a baseline; see benchmarks/README.md.

#include "exec/runtime.hpp"
#include "harness.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>

int main(int argc, char* argv[]) {
  femorax::Runtime runtime(argc, argv);

  femorax::bench::Options options;
  std::string json_path;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "femorax_bench: %s needs a value\n", argv[i]);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--json") json_path = next();
    else if (a == "--filter") options.filter = next();
    else if (a == "--reps") options.reps = std::atoi(next());
    else if (a == "--warmup") options.warmup = std::atoi(next());
    else if (a == "--quick") options.quick = true;
    else {
      std::fprintf(stderr, "femorax_bench: unknown option '%s'\n", argv[i]);
      return 1;
    }
  }
  if (options.reps < 1) {
    std::fprintf(stderr, "femorax_bench: --reps must be at least 1\n");
    return 1;
  }

  int status = 0;
  {
    femorax::bench::Registry registry;
    femorax::bench::register_stream(registry, options);

    const auto results = femorax::bench::run_all(registry, options);
    if (results.empty()) {
      std::fprintf(stderr, "femorax_bench: no benchmark matches '%s'\n", options.filter.c_str());
      status = 1;
    }
    femorax::bench::print_table(results);
    if (!json_path.empty()) {
      std::ofstream(json_path) << femorax::bench::to_json(results, options);
    }
  }  // kernels (and their Kokkos Views) are released before Kokkos finalizes
  return status;
}
