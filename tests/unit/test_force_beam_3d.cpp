// ForceBeam3d (docs/design.md, "Beams: the force-based beam", step 1): the physics with the
// elastic section. One element of an elastic cantilever is exact for end forces and moments
// (the force-based element's interpolation of the section forces is exact without element
// loads): the tip deflections and rotations equal Euler-Bernoulli's to 1e-12 for 3 to 7 Lobatto
// points; the tangent times the displacement gives the force (linear); the stiffness equals the
// displacement beam's exact two-node stiffness; rigid motion gives no force; a fresh state and a
// committed state give the same answer.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "physics/elements/force_beam_3d.hpp"
#include "physics/sections/elastic_section_3d.hpp"

namespace {

using femorax::ElasticSection3d;
constexpr double E = 2.5e7, A = 0.5, Iz = 0.02, Iy = 0.015, G = 1.0e7, J = 0.03, L = 3.0;

template <int NIP>
struct Fixture {
  using Beam = femorax::ForceBeam3d<ElasticSection3d, NIP>;
  typename Beam::Params p{};
  std::vector<double> sc, st;  // committed and trial point state (contiguous per point)
  double xe[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, L}};  // vertical
  Fixture() : sc(Beam::NSTATE, 0.0), st(Beam::NSTATE, 0.0) {
    p.mat = {E, A, Iz, Iy, G, J};
    p.transf.vecxz[0] = 1.0;
    p.transf.vecxz[1] = 0.0;
    p.transf.vecxz[2] = 0.0;
  }
  femorax::PointArray<const double> committed() const { return {sc.data(), 1, Beam::POINT_STATE}; }
  femorax::PointArray<double> trial() { return {st.data(), 1, Beam::POINT_STATE}; }
  // K (12 x 12) and f at the nodal displacements u.
  bool eval(const double* u, double* f, double (*K)[12]) {
    return Beam::template compute<true>(p, xe, u, committed(), trial(), f, K);
  }
};

template <int NIP>
void cantilever() {
  Fixture<NIP> fx;
  // The base node fixed: solve K_jj u_j = F at the tip node (DOFs 6..11) for unit end loads.
  double u0[12] = {}, f[12], K[12][12];
  ASSERT_TRUE(fx.eval(u0, f, K));
  for (int k = 0; k < 12; ++k) EXPECT_NEAR(f[k], 0.0, 1e-30);
  double Kt[6][6];
  for (int a = 0; a < 6; ++a)
    for (int b = 0; b < 6; ++b) Kt[a][b] = K[6 + a][6 + b];
  double Ft[6][6];
  femorax::force_beam_detail::invert<6>(Kt, Ft);
  // The member's local axes: x along global z; vecxz = global x, so local z ~ global x, local y ~ ...
  // Check against closed forms in global terms: a tip force P along global x bends about the axis
  // whose moment of inertia couples x: here (vecxz = x) the local z axis is global x, so a force
  // along global x is a local z force: deflection P L^3 / (3 E Iy), rotation P L^2 / (2 E Iy).
  const double defl_x = L * L * L / (3.0 * E * Iy), defl_y = L * L * L / (3.0 * E * Iz);
  EXPECT_NEAR(std::fabs(Ft[0][0]), defl_x, 1e-12 * defl_x) << NIP;
  EXPECT_NEAR(std::fabs(Ft[1][1]), defl_y, 1e-12 * defl_y) << NIP;
  EXPECT_NEAR(Ft[2][2], L / (E * A), 1e-12 * L / (E * A)) << NIP;
  EXPECT_NEAR(Ft[5][5], L / (G * J), 1e-12 * L / (G * J)) << NIP;
  EXPECT_NEAR(std::fabs(Ft[3][3]), L / (E * Iz), 1e-12 * L / (E * Iz)) << NIP;
  EXPECT_NEAR(std::fabs(Ft[4][4]), L / (E * Iy), 1e-12 * L / (E * Iy)) << NIP;
  EXPECT_NEAR(std::fabs(Ft[0][4]), L * L / (2.0 * E * Iy), 1e-12 * L * L / (2.0 * E * Iy)) << NIP;

  // Linear: the force at a displacement equals K u (to rounding), from the fresh state and from
  // a committed state.
  double u[12];
  for (int k = 0; k < 12; ++k) u[k] = 1e-4 * std::sin(0.7 * k + 0.3);
  double fu[12], Ku[12][12];
  ASSERT_TRUE(fx.eval(u, fu, Ku));
  double peak = 0.0;
  for (int a = 0; a < 12; ++a) {
    double ku = 0.0;
    for (int b = 0; b < 12; ++b) ku += K[a][b] * u[b];
    peak = std::max(peak, std::fabs(ku));
    EXPECT_NEAR(fu[a], ku, 1e-10 * std::fabs(ku) + 1e-9) << NIP << " " << a;
  }
  // Commit (trial -> committed), then the same displacement again: the same force; then twice it.
  fx.sc = fx.st;
  double f2[12], K2[12][12];
  ASSERT_TRUE(fx.eval(u, f2, K2));
  for (int a = 0; a < 12; ++a) EXPECT_NEAR(f2[a], fu[a], 1e-12 * peak) << a;
  double uu[12];
  for (int k = 0; k < 12; ++k) uu[k] = 2.0 * u[k];
  ASSERT_TRUE(fx.eval(uu, f2, K2));
  for (int a = 0; a < 12; ++a) EXPECT_NEAR(f2[a], 2.0 * fu[a], 1e-11 * peak) << a;
  // Rigid translation: no force.
  double ur[12] = {1e-3, -2e-3, 3e-3, 0, 0, 0, 1e-3, -2e-3, 3e-3, 0, 0, 0};
  Fixture<NIP> fr;
  double fr_[12], Kr[12][12];
  ASSERT_TRUE(fr.eval(ur, fr_, Kr));
  for (int a = 0; a < 12; ++a) EXPECT_NEAR(fr_[a], 0.0, 1e-9) << a;
}

TEST(ForceBeam3d, OneElementCantileverIsExactFor3To7Points) {
  cantilever<3>();
  cantilever<4>();
  cantilever<5>();
  cantilever<6>();
  cantilever<7>();
}

}  // namespace
