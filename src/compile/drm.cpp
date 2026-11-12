#include "compile/drm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "compile/registry.hpp"
#include "io/h5.hpp"

namespace femorax::compile {

namespace {

// Cell coordinates of a point in a grid of cell size h.
struct Key {
  std::int64_t i, j, k;
  bool operator==(const Key& o) const { return i == o.i && j == o.j && k == o.k; }
};
struct KeyHash {
  std::size_t operator()(const Key& k) const {
    return static_cast<std::size_t>(k.i * 73856093LL ^ k.j * 19349663LL ^ k.k * 83492791LL);
  }
};

}  // namespace

DrmSetup setup_drm(const io::H5DrmPattern& p, const io::DrmFile& f, const Compiled& c) {
  DrmSetup s;
  s.pattern = p.tag;
  const std::int32_t S = f.stations(), N = c.dofs.num_nodes();

  // Station coordinates in the model: OpenSees' lam_transform,
  //   x <- T (crd_scale (x - drmbox_x0)) + x0,
  // in its operation order (copy -= x0; copy *= crd_scale; point = T copy; point += x0).
  s.station_xyz = f.xyz;
  if (p.do_coordinate_transformation) {
    for (std::int32_t st = 0; st < S; ++st) {
      double* x = &s.station_xyz[3 * static_cast<std::size_t>(st)];
      double copy[3];
      for (int d = 0; d < 3; ++d) copy[d] = (x[d] - f.x0[d]) * p.crd_scale;
      for (int d = 0; d < 3; ++d) {
        double v = 0.0;
        for (int k = 0; k < 3; ++k) v += p.T[d][k] * copy[k];
        x[d] = v + p.x0[d];
      }
    }
  }

  // Nearest station of each node within the tolerance. OpenSees takes the nearest station
  // over all of them (the first one on a tie) and accepts it if it is closer than the
  // tolerance; any station closer than the tolerance lies in the 27 cells around the node.
  const double h = p.distance_tolerance;
  auto key = [&](const double* x) {
    return Key{static_cast<std::int64_t>(std::floor(x[0] / h)), static_cast<std::int64_t>(std::floor(x[1] / h)),
               static_cast<std::int64_t>(std::floor(x[2] / h))};
  };
  std::unordered_map<Key, std::vector<std::int32_t>, KeyHash> grid;
  for (std::int32_t st = 0; st < S; ++st) grid[key(&s.station_xyz[3 * static_cast<std::size_t>(st)])].push_back(st);
  s.node_slot.assign(static_cast<std::size_t>(N), -1);
  std::vector<std::uint8_t> station_used(static_cast<std::size_t>(S), 0);
  for (std::int32_t n = 0; n < N; ++n) {
    const double x[3] = {c.coords[static_cast<std::size_t>(n)], c.coords[static_cast<std::size_t>(N + n)],
                         c.coords[static_cast<std::size_t>(2 * N + n)]};
    const Key k0 = key(x);
    double dmin = std::numeric_limits<double>::infinity();
    std::int32_t best = -1;
    for (std::int64_t di = -1; di <= 1; ++di)
      for (std::int64_t dj = -1; dj <= 1; ++dj)
        for (std::int64_t dk = -1; dk <= 1; ++dk) {
          const auto it = grid.find({k0.i + di, k0.j + dj, k0.k + dk});
          if (it == grid.end()) continue;
          for (const auto st : it->second) {
            const double* y = &s.station_xyz[3 * static_cast<std::size_t>(st)];
            const double d0 = x[0] - y[0], d1 = x[1] - y[1], d2 = x[2] - y[2];
            const double d = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);  // Vector::Norm
            if (d < dmin || (d == dmin && st < best)) {
              dmin = d;
              best = st;
            }
          }
        }
    if (best < 0 || !(std::fabs(dmin) < h)) continue;
    s.node_slot[static_cast<std::size_t>(n)] = static_cast<std::int32_t>(s.node.size());
    s.node.push_back(n);
    s.station.push_back(best);
    s.row.push_back(f.data_location[static_cast<std::size_t>(best)]);
    s.boundary.push_back(static_cast<std::uint8_t>(f.internal[static_cast<std::size_t>(best)]));
    s.max_match_distance = std::max(s.max_match_distance, dmin);
    station_used[static_cast<std::size_t>(best)] = 1;
  }
  for (const auto u : station_used) s.stations_matched += u;

  // DRM elements: all nodes matched. Those with both kinds of node receive forces.
  for (std::size_t bi = 0; bi < c.batches.size(); ++bi) {
    const Batch& b = c.batches[bi];
    const auto P = static_cast<std::size_t>(b.padded);
    for (std::int32_t e = 0; e < b.count; ++e) {
      std::uint32_t mask = 0;
      int matched = 0, nb = 0;
      for (int a = 0; a < b.nnode; ++a) {
        const std::int32_t n = b.node[static_cast<std::size_t>(a) * P + static_cast<std::size_t>(e)];
        const std::int32_t slot = s.node_slot[static_cast<std::size_t>(n)];
        if (slot < 0) break;
        ++matched;
        if (s.boundary[static_cast<std::size_t>(slot)]) {
          mask |= 1u << a;
          ++nb;
        }
      }
      if (matched != b.nnode) continue;
      ++s.drm_elements;
      if (nb == 0 || nb == b.nnode) continue;
      if (b.nstate > 0) {
        // A damped elastic element is linear (its state is the damping's filters), but the
        // DRM forces use a constant elastic stiffness: not supported yet. A material that can
        // yield makes the method invalid.
        bool damped_linear = false;
        registry::for_each_pair([&]<class E, class M>(int index) {
          if constexpr (registry::is_damped_entry<M>)
            if (index == b.pair) damped_linear = M::Material::Base::NSTATE == 0;
        });
        if (damped_linear) {
          throw io::UnsupportedComponent(
              "unsupported component: H5DRM pattern " + std::to_string(p.tag) + ": element " +
              std::to_string(b.tag[static_cast<std::size_t>(e)]) +
              " lies on the DRM boundary and has Uniform damping (/damping/uniform). The DRM forces are "
              "computed with the layer's constant elastic stiffness, which does not include the damping yet: "
              "leave the elements of the DRM layer out of the Uniform damping regions, or run this model with "
              "the OpenSees backend.");
        }
        throw io::ModelError("H5DRM pattern " + std::to_string(p.tag) + ": element " +
                             std::to_string(b.tag[static_cast<std::size_t>(e)]) +
                             " lies on the DRM boundary and its material has history state (it can yield). "
                             "The Domain Reduction Method is valid only if the DRM layer stays linear: give "
                             "the elements of the DRM layer a linear elastic material.");
      }
      s.elements.push_back({static_cast<std::int32_t>(bi), e, mask});
    }
  }
  if (s.elements.empty()) {
    throw io::ModelError("H5DRM pattern " + std::to_string(p.tag) + ": no element has both boundary and exterior "
                         "DRM nodes (" + std::to_string(s.node.size()) + " nodes matched " +
                         std::to_string(s.stations_matched) + " of " + std::to_string(S) +
                         " stations within " + std::to_string(h) + "); check the coordinate transformation and "
                         "the distance tolerance");
  }
  return s;
}

}  // namespace femorax::compile
