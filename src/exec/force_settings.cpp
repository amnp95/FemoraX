#include "exec/force_settings.hpp"

#include <Kokkos_Core.hpp>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <type_traits>

#include "compile/registry.hpp"
#include "exec/runtime.hpp"
#include "femorax_version.hpp"

namespace femorax::exec {

namespace {
constexpr bool kDevice = !std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::filesystem::path tune_cache_path() {
  if (const char* p = std::getenv("FEMORAX_TUNE_CACHE"); p && *p) return p;
  if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x)
    return std::filesystem::path(x) / "femorax" / "autotune.txt";
  if (const char* h = std::getenv("HOME"); h && *h)
    return std::filesystem::path(h) / ".cache" / "femorax" / "autotune.txt";
  return {};
}

std::map<std::string, std::string> read_cache(const std::filesystem::path& p) {
  std::map<std::string, std::string> m;
  std::ifstream in(p);
  for (std::string line; std::getline(in, line);) {
    const auto tab = line.find('\t');
    if (tab != std::string::npos) m[line.substr(0, tab)] = line.substr(tab + 1);
  }
  return m;
}

void write_cache(const std::filesystem::path& p, const std::map<std::string, std::string>& m) {
  std::error_code ec;
  std::filesystem::create_directories(p.parent_path(), ec);
  const auto tmp = p.string() + ".tmp";
  {
    std::ofstream out(tmp);
    if (!out) return;  // no writable cache: tune again next time
    for (const auto& [k, v] : m) out << k << '\t' << v << '\n';
  }
  std::filesystem::rename(tmp, p, ec);
}
}  // namespace

ForceSettings default_force_settings() {
  // CPU (i7-11370H, 8 OpenMP threads), 64k stdBrick, 2026-11-06: flat gather cached 18.7 ms
  // (flat atomic cached 18.2, team 178 ms or more). GPU (RTX 3050 Ti Laptop): team gather
  // ept4 cached 6.07 ms, the fastest variant of all (flat atomic cached 6.48).
  if (kDevice) return {Layout::Team, Scatter::Gather, 4, Geometry::Cached};
  return {Layout::Flat, Scatter::Gather, 0, Geometry::Cached};
}

bool layouts_share_arithmetic() { return !kDevice; }

std::vector<ForceSettings> tunable_variants(const InternalForce& force) {
  // On GPU backends the compiler fuses multiply-adds differently in the flat and the team
  // kernels, so the two layouts can differ in the last bit; within a layout every variant
  // is bit-identical. There the tuner keeps to the default's layout. Host compilers do not
  // fuse (GCC, -std=c++20), and every variant is bit-identical.
  const Layout family = default_force_settings().layout;
  std::vector<ForceSettings> out;
  std::vector<Geometry> geometries = {Geometry::Recompute};
  if (force.has_geometry_cache()) geometries.push_back(Geometry::Cached);
  for (auto g : geometries) {
    if (layouts_share_arithmetic() || family == Layout::Flat)
      out.push_back({Layout::Flat, Scatter::Gather, 0, g});
    if (layouts_share_arithmetic() || family == Layout::Team)
      for (int ept : {1, 2, 4, 8})
        if (ept <= force.max_elements_per_team()) out.push_back({Layout::Team, Scatter::Gather, ept, g});
  }
  // The compacted corrector path: bit-identical to the fused one where the compiler does not
  // contract (the same condition as the layouts).
  if (force.has_split_material() && layouts_share_arithmetic()) {
    const std::size_t n = out.size();
    for (std::size_t i = 0; i < n; ++i) {
      ForceSettings c = out[i];
      c.material = MaterialPath::Compacted;
      out.push_back(c);
    }
  }
  return out;
}

std::vector<ForceSettings> all_variants(const InternalForce& force) {
  std::vector<ForceSettings> out;
  std::vector<Scatter> scatters = {Scatter::Gather, Scatter::Atomic};
  if (force.has_coloring()) scatters.push_back(Scatter::Colored);
  std::vector<Geometry> geometries = {Geometry::Recompute};
  if (force.has_geometry_cache()) geometries.push_back(Geometry::Cached);
  for (auto g : geometries)
    for (auto sc : scatters) {
      out.push_back({Layout::Flat, sc, 0, g});
      for (int ept : {1, 2, 4, 8})
        if (ept <= force.max_elements_per_team()) out.push_back({Layout::Team, sc, ept, g});
    }
  if (force.has_split_material()) {
    const std::size_t n = out.size();
    for (std::size_t i = 0; i < n; ++i) {
      ForceSettings c = out[i];
      c.material = MaterialPath::Compacted;
      out.push_back(c);
    }
  }
  return out;
}

TuneResult autotune(InternalForce& force, const std::vector<ForceSettings>& variants, int reps) {
  TuneResult r;
  double best = 1e300;
  for (const auto& s : variants) {
    force.compute(s);
    Kokkos::fence();
    double t_min = 1e300;
    for (int i = 0; i < reps; ++i) {
      const double t0 = now();
      force.compute(s);
      Kokkos::fence();
      t_min = std::min(t_min, now() - t0);
    }
    r.seconds.emplace_back(s, t_min);
    if (t_min < best) {
      best = t_min;
      r.best = s;
    }
  }
  return r;
}

std::string tune_key(const compile::Compiled& c) {
  std::set<std::string> pairs;
  bool mass_damping = false, stiffness_damping = false;
  std::int64_t n = 0;
  for (const auto& b : c.batches) {
    registry::for_each_pair([&]<class E, class M>(int index) {
      if (index == b.pair) pairs.insert(std::string(E::name) + "+" + M::name);
    });
    for (const auto& p : b.params) {
      mass_damping |= p.damping.alphaM != 0.0;
      stiffness_damping |= p.damping.betaK != 0.0 || p.damping.betaKinit != 0.0 || p.damping.betaKcomm != 0.0;
    }
    n += b.count;
  }
  std::string key;
  for (const auto& p : pairs) key += p + ",";
  key += std::string("damping=") + (stiffness_damping ? "K" : "") + (mass_damping ? "M" : "") +
         (mass_damping || stiffness_damping ? "" : "none");
  key += ",size=2^" + std::to_string(n > 0 ? static_cast<int>(std::floor(std::log2(static_cast<double>(n)))) : 0);
  return key;
}

std::string machine_key() {
  char host[256] = {};
  gethostname(host, sizeof(host) - 1);
  return std::string(host) + "," + default_execution_space() + std::to_string(default_concurrency()) +
         ",simd" + std::to_string(simd_width()) + ",femorax" + version_string;
}

Choice choose_force_settings(InternalForce& force, const compile::Compiled& c) {
  if (const char* o = std::getenv("FEMORAX_FORCE_SETTINGS"); o && *o) {
    ForceSettings s;
    if (parse_settings(o, s)) return {force.resolve(s), "override"};
  }
  const auto path = tune_cache_path();
  const std::string key = machine_key() + "|" + tune_key(c);
  auto cache = path.empty() ? std::map<std::string, std::string>{} : read_cache(path);
  if (const auto it = cache.find(key); it != cache.end()) {
    ForceSettings s;
    if (parse_settings(it->second, s)) return {force.resolve(s), "saved"};
  }
  const TuneResult t = autotune(force, tunable_variants(force));
  const ForceSettings best = force.resolve(t.best);
  if (!path.empty()) {
    cache[key] = to_string(best);
    write_cache(path, cache);
  }
  return {best, "tuned"};
}

}  // namespace femorax::exec
