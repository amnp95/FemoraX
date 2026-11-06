#include "exec/force_settings.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <type_traits>

#include "exec/timing.hpp"

namespace femorax::exec {

ForceSettings default_force_settings(Purpose purpose) {
  constexpr bool device =
      !std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;
  if (purpose == Purpose::Production) {
    // OpenMP (i7-11370H, 8 threads) and CUDA (RTX 3050 Ti Laptop), 64k stdBrick elements,
    // 2026-11-06: flat atomic cached 19.3 ms / 6.47 ms, the fastest on both.
    return {Layout::Flat, Scatter::Atomic, 0, Geometry::Cached};
  }
  if (!device) {
    // OpenMP: flat colored recompute 22.1 ms (cached 23.9, team 195 ms or more).
    return {Layout::Flat, Scatter::Colored, 0, Geometry::Recompute};
  }
  // CUDA: team colored ept4 cached 15.5 ms (flat colored 46.8-51.2 ms).
  return {Layout::Team, Scatter::Colored, 4, Geometry::Cached};
}

std::vector<ForceSettings> force_variants(Purpose purpose, const ExplicitForce& force) {
  std::vector<ForceSettings> out;
  std::vector<Scatter> scatters;
  if (purpose == Purpose::Production) scatters = {Scatter::Atomic, Scatter::Colored, Scatter::Gather};
  else scatters = {Scatter::Colored, Scatter::Gather};
  std::vector<Geometry> geometries = {Geometry::Recompute};
  if (force.has_geometry_cache()) geometries.push_back(Geometry::Cached);
  for (auto g : geometries)
    for (auto sc : scatters) {
      out.push_back({Layout::Flat, sc, 0, g});
      for (int ept : {1, 2, 4, 8})
        if (ept <= ExplicitForce::max_elements_per_team()) out.push_back({Layout::Team, sc, ept, g});
    }
  return out;
}

TuneResult autotune(ExplicitForce& force, Purpose purpose, int reps) {
  TuneResult r;
  double best = 1e300;
  for (const auto& s : force_variants(purpose, force)) {
    force.compute(s);
    fence();
    double t_min = 1e300;
    for (int i = 0; i < reps; ++i) {
      const double t0 = wall_time();
      force.compute(s);
      fence();
      t_min = std::min(t_min, wall_time() - t0);
    }
    r.seconds.emplace_back(s, t_min);
    if (t_min < best) {
      best = t_min;
      r.best = s;
    }
  }
  return r;
}

}  // namespace femorax::exec
