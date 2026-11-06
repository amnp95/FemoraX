#include "harness.hpp"

#include "exec/runtime.hpp"
#include "exec/timing.hpp"
#include "femorax_version.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

#ifndef FEMORAX_BUILD_TYPE
#define FEMORAX_BUILD_TYPE "unknown"
#endif

namespace femorax::bench {

namespace {

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    if (c == '\n') {
      out += "\\n";
      continue;
    }
    out += c;
  }
  return out;
}

std::string cpu_model() {
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("model name", 0) == 0) {
      const auto pos = line.find(':');
      if (pos != std::string::npos) return line.substr(pos + 2);
    }
  }
  return "unknown";
}

std::string host_name() {
  char buf[256] = {};
  if (gethostname(buf, sizeof(buf) - 1) != 0) return "unknown";
  return buf;
}

}  // namespace

std::vector<Result> run_all(const Registry& registry, const Options& options) {
  std::vector<Result> results;
  for (const auto& b : registry.all()) {
    if (!options.filter.empty() && b.name.find(options.filter) == std::string::npos) continue;
    for (int i = 0; i < options.warmup; ++i) b.run();
    fence();
    std::vector<double> t(static_cast<std::size_t>(options.reps));
    for (auto& ti : t) {
      fence();
      const double t0 = wall_time();
      b.run();
      fence();
      ti = wall_time() - t0;
    }
    std::sort(t.begin(), t.end());
    Result r;
    r.bench = b;
    r.reps = options.reps;
    r.min_s = t.front();
    r.median_s = t[t.size() / 2];
    r.max_s = t.back();
    results.push_back(std::move(r));
  }
  return results;
}

void print_table(const std::vector<Result>& results) {
  std::printf("%-36s %6s %12s %12s %10s %10s\n", "benchmark", "per", "min [us]", "median [us]",
              "GB/s", "GFLOP/s");
  for (const auto& r : results) {
    std::printf("%-36s %6s %12.2f %12.2f %10.2f %10.2f\n", r.bench.name.c_str(),
                r.bench.unit.c_str(), r.min_s * 1e6, r.median_s * 1e6, r.gbytes_per_s(),
                r.gflops());
  }
}

std::string to_json(const std::vector<Result>& results, const Options& options) {
  std::ostringstream o;
  o.precision(9);
  o << "{\n";
  o << "  \"format\": 1,\n";
  o << "  \"machine\": {\n";
  o << "    \"host\": \"" << json_escape(host_name()) << "\",\n";
  o << "    \"cpu\": \"" << json_escape(cpu_model()) << "\",\n";
  o << "    \"execution_space\": \"" << default_execution_space() << "\",\n";
  o << "    \"concurrency\": " << default_concurrency() << ",\n";
  o << "    \"kokkos\": \"" << kokkos_version() << "\",\n";
  o << "    \"femorax\": \"" << version_string << "\",\n";
  o << "    \"build_type\": \"" << FEMORAX_BUILD_TYPE << "\"\n";
  o << "  },\n";
  o << "  \"options\": {\"warmup\": " << options.warmup << ", \"reps\": " << options.reps
    << ", \"quick\": " << (options.quick ? "true" : "false") << "},\n";
  o << "  \"results\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    o << "    {\"name\": \"" << json_escape(r.bench.name) << "\", \"unit\": \"" << r.bench.unit
      << "\", \"bytes\": " << r.bench.bytes << ", \"flops\": " << r.bench.flops
      << ", \"items\": " << r.bench.items << ", \"gated\": " << (r.bench.gated ? "true" : "false")
      << ", \"reps\": " << r.reps
      << ", \"min_s\": " << r.min_s << ", \"median_s\": " << r.median_s
      << ", \"max_s\": " << r.max_s << ", \"gbytes_per_s\": " << r.gbytes_per_s()
      << ", \"gflops\": " << r.gflops() << "}" << (i + 1 < results.size() ? "," : "") << "\n";
  }
  o << "  ]\n}\n";
  return o.str();
}

}  // namespace femorax::bench
