// The device memory estimate (exec::RunReport::device_bytes_estimate, checked against the
// device's free memory before anything is allocated) against what a run actually allocates:
// Kokkos' allocation hooks count every allocation in the default execution space's memory
// space, and the peak of the live bytes over the run is the actual use. The estimate must be
// within 10% of it, from above or below, for explicit and implicit runs, both elements, an
// elastic and a history material, with and without the geometry cache's room, and with the
// recorders (CUDA's own context, about 200 MiB on the laptop, is not an allocation and is
// covered by the margin the check leaves).

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>

#include "../common/synthetic_models.hpp"
#include "exec/explicit_analysis.hpp"

namespace {

using namespace femorax;
namespace fs = std::filesystem;

struct Tracker {
  std::int64_t live = 0, peak = 0;
  std::map<const void*, std::pair<std::string, std::int64_t>> blocks;
  std::map<std::string, std::int64_t> at_peak;  // label -> bytes, at the peak
  bool on = false;
};
Tracker g_track;

bool counts(const Kokkos::Profiling::SpaceHandle h) {
  return std::strcmp(h.name, Kokkos::DefaultExecutionSpace::memory_space::name()) == 0;
}

void on_alloc(const Kokkos::Profiling::SpaceHandle h, const char* label, const void* ptr, const std::uint64_t size) {
  if (!g_track.on || !counts(h)) return;
  g_track.blocks[ptr] = {label, static_cast<std::int64_t>(size)};
  g_track.live += static_cast<std::int64_t>(size);
  if (g_track.live > g_track.peak) {
    g_track.peak = g_track.live;
    g_track.at_peak.clear();
    for (const auto& [p, b] : g_track.blocks) g_track.at_peak[b.first] += b.second;
  }
}

void on_free(const Kokkos::Profiling::SpaceHandle h, const char*, const void* ptr, const std::uint64_t) {
  if (!g_track.on || !counts(h)) return;
  const auto it = g_track.blocks.find(ptr);
  if (it == g_track.blocks.end()) return;
  g_track.live -= it->second.second;
  g_track.blocks.erase(it);
}

std::string temp_dir() {
  const fs::path d = fs::temp_directory_path() / ("femorax_test_memory_" + std::to_string(::getpid()));
  fs::create_directories(d);
  return d.string();
}

io::PressureDependMultiYieldData soil() {
  io::PressureDependMultiYieldData d;
  d.rho = 19.9 / 9.81;
  d.refShearModul = 145.0e3;
  d.refBulkModul = 2.0 * 145.0e3 * 1.3 / 1.2;
  d.frictionAng = 40.0;
  d.peakShearStra = 0.1;
  d.refPress = 80.0;
  d.pressDependCoe = 0.5;
  d.PTAng = 27.0;
  d.contrac = 0.03;
  d.dilat1 = 0.8;
  d.dilat2 = 5.0;
  d.noYieldSurf = 20;
  d.e = 0.45;
  return d;
}

io::AnalysisStep explicit_step(int steps) {
  io::AnalysisStep a;
  a.name = "shake";
  a.dt = 1e-5;
  a.num_steps = steps;
  a.integrator = io::Integrator::ExplicitDifference;
  return a;
}

io::AnalysisStep newmark_step(int steps) {
  io::AnalysisStep a;
  a.name = "shake";
  a.dt = 1e-3;
  a.num_steps = steps;
  a.integrator = io::Integrator::Newmark;
  a.gamma = 0.5;
  a.beta = 0.25;
  a.algorithm = io::Algorithm::Newton;
  a.test_type = io::TestType::NormDispIncr;
  a.test_tol = 1e-10;
  a.test_max_iter = 10;
  return a;
}

struct Case {
  const char* name;
  io::ElementType element;
  bool soil, implicit, vtkhdf;
};

TEST(MemoryEstimate, WithinTenPercentOfTheAllocatedPeak) {
  Kokkos::Tools::Experimental::set_allocate_data_callback(on_alloc);
  Kokkos::Tools::Experimental::set_deallocate_data_callback(on_free);
  const Case cases[] = {
      {"stdBrick elastic explicit", io::ElementType::StdBrick, false, false, false},
      {"SSPbrick elastic explicit, VTKHDF", io::ElementType::SSPbrick, false, false, true},
      {"stdBrick elastic implicit", io::ElementType::StdBrick, false, true, false},
      {"stdBrick PDMY implicit, VTKHDF", io::ElementType::StdBrick, true, true, true},
      {"SSPbrick PDMY explicit, VTKHDF", io::ElementType::SSPbrick, true, false, true},
  };
  const std::string dir = temp_dir();
  for (const auto& cs : cases) {
    SCOPED_TRACE(cs.name);
    femorax::testing::ColumnSpec s;
    s.nx = s.ny = 16;  // large enough that fixed costs (OpenMP's scratch pool, ~0.3 MB) are small
    s.nz = 12;
    s.element = cs.element;
    s.lumped = !cs.implicit || cs.element == io::ElementType::SSPbrick;
    io::Model m = femorax::testing::soil_column(s);
    if (cs.soil) {
      m.materials[0].data = soil();
      m.materials[1].data = soil();
    }
    if (cs.vtkhdf) {
      io::VtkhdfRecorder r;
      r.file = dir + "/memory.vtkhdf";
      r.dT = 0.0;
      r.responses = {"disp", "vel", "accel"};
      if (cs.soil) {
        r.responses.push_back("stress3D6");
        r.responses.push_back("strain3D6");
      }
      m.recorders.push_back({"vtkhdf", r, {}});
      m.process.push_back({"000", "", io::RecorderStep{"vtkhdf"}});
    }
    m.process.push_back({"001", "", cs.implicit ? newmark_step(3) : explicit_step(3)});
    exec::RunOptions o;
    o.verbose = false;
    g_track = Tracker{};
    g_track.on = true;
    const auto rep = exec::run_model(m, o);
    g_track.on = false;
    const double est = static_cast<double>(rep.device_bytes_estimate), peak = static_cast<double>(g_track.peak);
    std::printf("[ %-36s estimate %9.3f MB, allocated peak %9.3f MB, ratio %.3f ]\n", cs.name, est / 1e6, peak / 1e6,
                est / peak);
    if (std::abs(est / peak - 1.0) > 0.02) {
      std::vector<std::pair<std::int64_t, std::string>> top;
      for (const auto& [label, bytes] : g_track.at_peak) top.emplace_back(bytes, label);
      std::sort(top.rbegin(), top.rend());
      for (std::size_t i = 0; i < std::min<std::size_t>(top.size(), 12); ++i)
        std::printf("    %-32s %10.3f MB\n", top[i].second.c_str(), top[i].first / 1e6);
    }
    EXPECT_NEAR(est / peak, 1.0, 0.10);
  }
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  Kokkos::Tools::Experimental::set_deallocate_data_callback(nullptr);
  std::error_code ec;
  fs::remove_all(dir, ec);
}

}  // namespace
