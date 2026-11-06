// FLOP counting: compile the real stdBrick + ElasticIsotropic physics with `double` replaced
// by a type that counts every arithmetic operation. The physics is wrapped in its own
// namespace (counting::femorax) so it never meets the real build's types, and built host
// only. The physics has no data-dependent branches except the damping flags and the mass
// type, so one evaluation per case gives the exact count.

#include "flop_count.hpp"

// Standard headers first: the macro below must not reach them.
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace counting {

inline std::uint64_t flops = 0;

struct Counted {
  double v;
  Counted() = default;
  constexpr Counted(double x) : v(x) {}
};
inline Counted operator+(Counted a, Counted b) { ++flops; return a.v + b.v; }
inline Counted operator-(Counted a, Counted b) { ++flops; return a.v - b.v; }
inline Counted operator*(Counted a, Counted b) { ++flops; return a.v * b.v; }
inline Counted operator/(Counted a, Counted b) { ++flops; return a.v / b.v; }
inline Counted operator-(Counted a) { return -a.v; }  // a sign flip, no arithmetic
inline Counted& operator+=(Counted& a, Counted b) { return a = a + b; }
inline Counted& operator-=(Counted& a, Counted b) { return a = a - b; }
inline Counted& operator*=(Counted& a, Counted b) { return a = a * b; }
inline Counted& operator/=(Counted& a, Counted b) { return a = a / b; }
inline bool operator<(Counted a, Counted b) { return a.v < b.v; }
inline bool operator>(Counted a, Counted b) { return a.v > b.v; }
inline bool operator<=(Counted a, Counted b) { return a.v <= b.v; }
inline bool operator>=(Counted a, Counted b) { return a.v >= b.v; }
inline bool operator==(Counted a, Counted b) { return a.v == b.v; }
inline bool operator!=(Counted a, Counted b) { return a.v != b.v; }
inline bool operator!(Counted a) { return !a.v; }

}  // namespace counting

namespace std {
inline counting::Counted sqrt(counting::Counted x) { ++counting::flops; return std::sqrt(x.v); }
inline counting::Counted abs(counting::Counted x) { return std::abs(x.v); }
}  // namespace std

#define FEMORAX_HD
#define double ::counting::Counted
namespace counting {
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"
}  // namespace counting
#undef double

namespace femorax::bench {

double brick_flops_per_element(const BrickFlopCase& c, bool team, bool cached) {
  using namespace counting::femorax;
  using C = counting::Counted;
  using Brick = StdBrick<ElasticIsotropic>;
  const Brick::Params p{{377000.0, 0.3, 2.0}, {0.0, 0.0, -19.9}, c.massType};
  const Rayleigh d{c.alphaM, c.betaK, c.betaKinit, c.betaKcomm};
  C xe[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  C ue[24], ve[24], fe[24];
  for (int k = 0; k < 24; ++k) {
    ue[k] = 1e-4 * (k + 1);
    ve[k] = 1e-3 * (k + 2);
  }
  C xl[3][8];
  Brick::local_coords(xe, xl);
  C geom[Brick::NGP * Brick::NGEOM];
  const PointArray<C> g{geom, Brick::NGP, 1};
  for (int gp = 0; gp < Brick::NGP; ++gp) Brick::point_store_geometry(xl, gp, g.point(gp));

  counting::flops = 0;
  if (!team) {
    if (cached) Brick::compute_cached<false>(p, d, g, ue, ve, {}, {}, fe, nullptr);
    else Brick::compute<false>(p, d, xe, ue, ve, {}, {}, fe, nullptr);
  } else {
    // As launch_team: per Gauss point, then the per-node reduction.
    C contrib[8][8][9];
    for (int gp = 0; gp < 8; ++gp) {
      Brick::PointShape s;
      if (cached) Brick::point_load_geometry(gp, g.point(gp), s);
      else Brick::point_geometry(xl, gp, s);
      C D0[6][6];
      if (d.betaKinit != 0.0) ElasticIsotropic::initial_tangent(p.mat, D0);
      Brick::PointResult r;
      C dd[6][6];
      Brick::point_update_shape(p, d, s, ue, ve, {}, {}, D0, r, dd);
      for (int j = 0; j < 8; ++j)
        Brick::point_node_contribution(p, d, r, ve, j, contrib[gp][j], contrib[gp][j] + 3,
                                       contrib[gp][j] + 6);
    }
    for (int j = 0; j < 8; ++j)
      for (int q = 0; q < 3; ++q) {
        C resid = 0.0, fdamp = 0.0;
        for (int gp = 0; gp < 8; ++gp) {
          resid += contrib[gp][j][q];
          resid -= contrib[gp][j][3 + q];
          fdamp += contrib[gp][j][6 + q];
        }
        fe[3 * j + q] = resid + fdamp;
      }
  }
  return static_cast<double>(counting::flops);
}

}  // namespace femorax::bench
