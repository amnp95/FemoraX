// DispBeam3d<ElasticSection3d, NIP> (physics/elements/disp_beam_3d.hpp), physics only, against
// mechanics:
//   - a cantilever (one element, node I fixed) under tip forces and moments in every direction
//     reproduces the Euler-Bernoulli tip deflections and rotations exactly (cubic Hermite
//     interpolation is exact for end loads): P L^3 / (3 EI), P L^2 / (2 EI), M L^2 / (2 EI),
//     M L / EI, N L / EA, T L / GJ;
//   - the internal force of an elastic member is its tangent times the displacements, for every
//     number of integration points;
//   - rotating the whole member (nodes, vecxz, displacements) rotates its forces;
//   - the lumped mass: rhoA L / 2 per translation and the rotary inertia per length times L / 2;
//   - the stable step's omega^2 against closed forms (axial only: 4 E / (rho L^2); torsion only:
//     4 G J / (rho Ip L^2)).
// The OpenSees element comparison (a reference driver for DispBeamColumn3d) is the next step.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

#include "physics/elements/disp_beam_3d.hpp"
#include "physics/sections/elastic_section_3d.hpp"

namespace {

using femorax::ElasticSection3d;
template <int NIP>
using Beam = femorax::DispBeam3d<ElasticSection3d, NIP>;

constexpr double E = 2.0e8, A = 0.05, Iz = 4.0e-4, Iy = 2.5e-4, G = 7.7e7, J = 3.1e-4, L = 3.0;

template <int NIP>
typename Beam<NIP>::Params params() {
  typename Beam<NIP>::Params p;
  p.sec = {E, A, Iz, Iy, G, J};
  p.transf.vecxz[2] = 1.0;  // member along x: local axes = global axes
  return p;
}

femorax::PointArray<const double> no_state() { return {nullptr, 1, 0}; }
femorax::PointArray<double> no_trial() { return {nullptr, 1, 0}; }

// Solves the 6 x 6 free block (node J) of the tangent for a tip load.
void solve6(double K[6][6], double b[6]) {
  for (int c = 0; c < 6; ++c) {
    int piv = c;
    for (int r = c + 1; r < 6; ++r)
      if (std::abs(K[r][c]) > std::abs(K[piv][c])) piv = r;
    for (int k = 0; k < 6; ++k) std::swap(K[c][k], K[piv][k]);
    std::swap(b[c], b[piv]);
    for (int r = c + 1; r < 6; ++r) {
      const double f = K[r][c] / K[c][c];
      for (int k = c; k < 6; ++k) K[r][k] -= f * K[c][k];
      b[r] -= f * b[c];
    }
  }
  for (int c = 5; c >= 0; --c) {
    for (int k = c + 1; k < 6; ++k) b[c] -= K[c][k] * b[k];
    b[c] /= K[c][c];
  }
}

template <int NIP>
void cantilever() {
  const auto p = params<NIP>();
  const double xe[2][3] = {{0, 0, 0}, {L, 0, 0}};
  const double u0[12] = {};
  double fe[12], Ke[12][12];
  Beam<NIP>::template compute<true>(p, xe, u0, no_state(), no_trial(), fe, Ke);
  auto tip = [&](int dof, double load, double out[6]) {
    double K[6][6];
    for (int a = 0; a < 6; ++a)
      for (int b = 0; b < 6; ++b) K[a][b] = Ke[6 + a][6 + b];
    for (int k = 0; k < 6; ++k) out[k] = 0.0;
    out[dof] = load;
    solve6(K, out);
  };
  const double P = 1.3e3, M = 2.1e3, N = 4.0e4, T = 900.0;
  double u[6];
  tip(1, P, u);  // force in y: bending about z
  EXPECT_NEAR(u[1], P * L * L * L / (3 * E * Iz), 1e-12 * std::abs(u[1]));
  EXPECT_NEAR(u[5], P * L * L / (2 * E * Iz), 1e-12 * std::abs(u[5]));
  tip(2, P, u);  // force in z: bending about y (rotation about y is -dw/dx)
  EXPECT_NEAR(u[2], P * L * L * L / (3 * E * Iy), 1e-12 * std::abs(u[2]));
  EXPECT_NEAR(u[4], -P * L * L / (2 * E * Iy), 1e-12 * std::abs(u[4]));
  tip(5, M, u);  // moment about z
  EXPECT_NEAR(u[5], M * L / (E * Iz), 1e-12 * std::abs(u[5]));
  EXPECT_NEAR(u[1], M * L * L / (2 * E * Iz), 1e-12 * std::abs(u[1]));
  tip(4, M, u);  // moment about y
  EXPECT_NEAR(u[4], M * L / (E * Iy), 1e-12 * std::abs(u[4]));
  EXPECT_NEAR(u[2], -M * L * L / (2 * E * Iy), 1e-12 * std::abs(u[2]));
  tip(0, N, u);
  EXPECT_NEAR(u[0], N * L / (E * A), 1e-12 * std::abs(u[0]));
  tip(3, T, u);
  EXPECT_NEAR(u[3], T * L / (G * J), 1e-12 * std::abs(u[3]));
}

TEST(DispBeam3d, CantileverIsExactForEndLoads) {
  cantilever<2>();
  cantilever<3>();
  cantilever<4>();
  cantilever<5>();
}

template <int NIP>
void force_is_tangent_times_displacement(femorax::CrdTransfType type) {
  auto p = params<NIP>();
  p.transf.type = type;
  p.transf.vecxz[0] = -0.2, p.transf.vecxz[1] = 0.3, p.transf.vecxz[2] = 1.0;
  p.transf.nodeIOffset[0] = 0.2, p.transf.nodeJOffset[2] = -0.1;
  const double xe[2][3] = {{0.3, -1.2, 0.7}, {2.9, 0.4, 2.2}};
  double u[12];
  for (int k = 0; k < 12; ++k) u[k] = 1e-3 * std::sin(0.7 * k + 0.2);
  double fe[12], Ke[12][12];
  Beam<NIP>::template compute<true>(p, xe, u, no_state(), no_trial(), fe, Ke);
  double d = 0.0, m = 0.0;
  for (int a = 0; a < 12; ++a) {
    double ku = 0.0;
    for (int b = 0; b < 12; ++b) ku += Ke[a][b] * u[b];
    if (type == femorax::CrdTransfType::Linear) d = std::max(d, std::abs(ku - fe[a]));
    m = std::max(m, std::abs(fe[a]));
    for (int b = 0; b < 12; ++b) EXPECT_NEAR(Ke[a][b], Ke[b][a], 1e-9 * std::abs(Ke[a][a]) + 1e-9);
  }
  EXPECT_GT(m, 0.0);
  EXPECT_LE(d, 1e-12 * m);
}

TEST(DispBeam3d, ElasticForceIsTheTangentTimesTheDisplacement) {
  for (const auto t : {femorax::CrdTransfType::Linear, femorax::CrdTransfType::PDelta}) {
    force_is_tangent_times_displacement<2>(t);
    force_is_tangent_times_displacement<5>(t);
  }
}

TEST(DispBeam3d, RotatingTheMemberRotatesItsForces) {
  auto p = params<3>();
  const double xe[2][3] = {{0, 0, 0}, {L, 0, 0}};
  double u[12];
  for (int k = 0; k < 12; ++k) u[k] = 1e-3 * std::cos(1.1 * k);
  double f[12], K[12][12];
  Beam<3>::compute<true>(p, xe, u, no_state(), no_trial(), f, K);
  // Rotation Q about an oblique axis; Q applied to nodes, vecxz, and every 3-vector of u.
  const double c1 = std::cos(0.6), s1 = std::sin(0.6), c2 = std::cos(-0.9), s2 = std::sin(-0.9);
  const double Qz[3][3] = {{c1, -s1, 0}, {s1, c1, 0}, {0, 0, 1}}, Qx[3][3] = {{1, 0, 0}, {0, c2, -s2}, {0, s2, c2}};
  double Q[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Q[i][j] = 0.0;
      for (int k = 0; k < 3; ++k) Q[i][j] += Qx[i][k] * Qz[k][j];
    }
  auto rot = [&](const double* a, double* b) {
    for (int i = 0; i < 3; ++i) b[i] = Q[i][0] * a[0] + Q[i][1] * a[1] + Q[i][2] * a[2];
  };
  auto pr = p;
  rot(p.transf.vecxz, pr.transf.vecxz);
  double xr[2][3], ur[12];
  rot(xe[0], xr[0]);
  rot(xe[1], xr[1]);
  for (int b = 0; b < 4; ++b) rot(u + 3 * b, ur + 3 * b);
  double fr[12], Kr[12][12];
  Beam<3>::compute<true>(pr, xr, ur, no_state(), no_trial(), fr, Kr);
  double want[12], m = 0.0;
  for (int b = 0; b < 4; ++b) rot(f + 3 * b, want + 3 * b);
  for (int k = 0; k < 12; ++k) m = std::max(m, std::abs(want[k]));
  for (int k = 0; k < 12; ++k) EXPECT_NEAR(fr[k], want[k], 1e-12 * m) << k;
}

TEST(DispBeam3d, LumpedMassHasTheRotaryInertia) {
  auto p = params<2>();
  const double rho = 2.4;
  p.rhoA = rho * A;
  p.rot[0] = rho * (Iy + Iz), p.rot[1] = rho * Iy, p.rot[2] = rho * Iz;
  const double xe[2][3] = {{0, 0, 0}, {L, 0, 0}};
  double m[12];
  Beam<2>::lumped_mass(p, xe, m);
  for (int n = 0; n < 2; ++n) {
    for (int d = 0; d < 3; ++d) EXPECT_DOUBLE_EQ(m[6 * n + d], 0.5 * L * rho * A);
    EXPECT_DOUBLE_EQ(m[6 * n + 3], 0.5 * L * rho * (Iy + Iz));
    EXPECT_DOUBLE_EQ(m[6 * n + 4], 0.5 * L * rho * Iy);
    EXPECT_DOUBLE_EQ(m[6 * n + 5], 0.5 * L * rho * Iz);
  }
  // A vertical member (local x = global z, vecxz = global x: local y = global y... rotated):
  // the trace of the rotary inertia is kept, and translations stay rhoA L / 2.
  auto pv = p;
  pv.transf.vecxz[0] = 1.0, pv.transf.vecxz[2] = 0.0;
  const double xv[2][3] = {{1, 2, -L}, {1, 2, 0}};
  double mv[12];
  Beam<2>::lumped_mass(pv, xv, mv);
  EXPECT_DOUBLE_EQ(mv[2], 0.5 * L * rho * A);
  EXPECT_NEAR(mv[3] + mv[4] + mv[5], 0.5 * L * rho * (2 * Iy + 2 * Iz), 1e-15);
  EXPECT_NEAR(mv[5], 0.5 * L * rho * (Iy + Iz), 1e-15);  // torsion about global z
}

TEST(DispBeam3d, StableStepMatchesClosedForms) {
  const double rho = 2.4;
  const double xe[2][3] = {{0, 0, 0}, {L, 0, 0}};
  {  // axial only: two masses rhoA L / 2 on a spring EA / L
    typename Beam<2>::Params p;
    p.sec = {E, A, 0.0, 0.0, G, 0.0};
    p.transf.vecxz[2] = 1.0;
    p.rhoA = rho * A;
    const double w2 = Beam<2>::max_omega_sq(p, xe, no_state());
    EXPECT_NEAR(w2, 4.0 * E / (rho * L * L), 1e-12 * w2);
    const femorax::Rayleigh none{};
    EXPECT_NEAR(Beam<2>::critical_dt(p, none, femorax::ExplicitScheme::CentralDifference, xe, no_state()),
                L / std::sqrt(E / rho), 1e-12 * L / std::sqrt(E / rho));
  }
  {  // torsion only: two rotary inertias rho Ip L / 2 on a spring GJ / L
    typename Beam<2>::Params p;
    p.sec = {E, 0.0, 0.0, 0.0, G, J};
    p.transf.vecxz[2] = 1.0;
    p.rot[0] = rho * (Iy + Iz);
    const double w2 = Beam<2>::max_omega_sq(p, xe, no_state());
    EXPECT_NEAR(w2, 4.0 * G * J / (rho * (Iy + Iz) * L * L), 1e-12 * w2);
  }
  {  // the full section with physical rotary inertia: the largest mode is a rotation one,
     // within sqrt(2)-ish of the axial step, never above the axial omega^2 by more than 8x
    auto p = params<2>();
    p.rhoA = rho * A;
    p.rot[0] = rho * (Iy + Iz), p.rot[1] = rho * Iy, p.rot[2] = rho * Iz;
    const double w2 = Beam<2>::max_omega_sq(p, xe, no_state());
    const double axial = 4.0 * E / (rho * L * L);
    std::printf("[ full section: omega^2 %.4e = %.3f x the axial one; dt %.4e against L/c %.4e ]\n", w2, w2 / axial,
                2.0 / std::sqrt(w2), L / std::sqrt(E / rho));
    EXPECT_GE(w2, axial * (1.0 - 1e-12));
    EXPECT_LE(w2, 8.0 * axial);
  }
}

}  // namespace
