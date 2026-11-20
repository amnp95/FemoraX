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
// And against OpenSees' DispBeamColumn3d with ElasticSection3d (tests/reference/drivers/
// disp_beam_column.cpp, commit 955ceb0): force, tangent and initial stiffness for four
// geometries (joint offsets included), 2, 3 and 5 points, Linear and PDelta; the lumped mass
// equals OpenSees' (translations only) when the rotary inertia is zero.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "physics/elements/disp_beam_3d.hpp"
#include "physics/sections/elastic_section_3d.hpp"
#include "reference_data.hpp"

namespace {

using femorax::ElasticSection3d;
template <int NIP>
using Beam = femorax::DispBeam3d<ElasticSection3d, NIP>;

constexpr double E = 2.0e8, A = 0.05, Iz = 4.0e-4, Iy = 2.5e-4, G = 7.7e7, J = 3.1e-4, L = 3.0;

template <int NIP>
typename Beam<NIP>::Params params() {
  typename Beam<NIP>::Params p;
  p.mat = {E, A, Iz, Iy, G, J};
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
    p.mat = {E, A, 0.0, 0.0, G, 0.0};
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
    p.mat = {E, 0.0, 0.0, 0.0, G, J};
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

// The consistent mass gives the exact kinetic energy of rigid motions: a translation
// (rhoA L), and rotations about the member axis (rho Ip L, the torsional mass) and about each
// transverse axis through node I (rhoA L^3 / 3 from the translations plus rho I L from the
// rotary inertia), for a member along x and the same member skewed in space.
TEST(DispBeam3d, ConsistentMassGivesTheRigidKineticEnergy) {
  const double rho = 2.4;
  for (const bool skew : {false, true}) {
    auto p = params<2>();
    p.cMass = 1;
    p.rhoA = rho * A;
    p.rot[0] = rho * (Iy + Iz), p.rot[1] = rho * Iy, p.rot[2] = rho * Iz;
    double xe[2][3] = {{0, 0, 0}, {L, 0, 0}};
    if (skew) {
      const double d[3] = {2.0, -1.0, 2.0};  // length 3 = L
      for (int i = 0; i < 3; ++i) xe[0][i] = 0.4 * (i + 1), xe[1][i] = xe[0][i] + d[i];
      p.transf.vecxz[0] = 0.3, p.transf.vecxz[1] = 0.5, p.transf.vecxz[2] = 0.8;
    }
    femorax::CrdTransf3dGeometry g;
    ASSERT_EQ(Beam<2>::geometry(p, xe, g), 0);
    double M[12][12];
    Beam<2>::consistent_mass(p, g, M);
    auto energy = [&](const double v[12]) {
      double e = 0.0;
      for (int a = 0; a < 12; ++a)
        for (int b = 0; b < 12; ++b) e += 0.5 * v[a] * M[a][b] * v[b];
      return e;
    };
    // Rigid velocity field: translation t plus rotation w about node I.
    auto rigid = [&](const double t[3], const double w[3], double v[12]) {
      for (int n = 0; n < 2; ++n) {
        const double r[3] = {xe[n][0] - xe[0][0], xe[n][1] - xe[0][1], xe[n][2] - xe[0][2]};
        v[6 * n + 0] = t[0] + w[1] * r[2] - w[2] * r[1];
        v[6 * n + 1] = t[1] + w[2] * r[0] - w[0] * r[2];
        v[6 * n + 2] = t[2] + w[0] * r[1] - w[1] * r[0];
        for (int d = 0; d < 3; ++d) v[6 * n + 3 + d] = w[d];
      }
    };
    const double zero[3] = {0, 0, 0};
    double v[12];
    const double t[3] = {0.3, -0.2, 0.5};
    rigid(t, zero, v);
    const double tt = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
    EXPECT_NEAR(energy(v), 0.5 * rho * A * L * tt, 1e-13);
    // Rotations about the local axes (rows of R), unit rate.
    const double I_local[3] = {Iy + Iz, Iy, Iz};
    for (int a = 0; a < 3; ++a) {
      const double w[3] = {g.R[a][0], g.R[a][1], g.R[a][2]};
      rigid(zero, w, v);
      const double want = 0.5 * rho * (I_local[a] * L + (a == 0 ? 0.0 : A * L * L * L / 3.0));
      EXPECT_NEAR(energy(v), want, 1e-12 * want) << "skew " << skew << " axis " << a;
    }
  }
}

// --- Against OpenSees ----------------------------------------------------------------

double rel_diff(const double* a, const double* b, int n) {
  double d = 0.0, m = 0.0;
  for (int i = 0; i < n; ++i) {
    d = std::max(d, std::abs(a[i] - b[i]));
    m = std::max(m, std::abs(b[i]));
  }
  return m > 0.0 ? d / m : d;
}

// worst: force, tangent, initial stiffness, lumped mass, consistent mass, then the Rayleigh
// damping forces alphaM, betaK, betaK0, betaKc.
template <int NIP>
void compare_row(const femorax::test::ReferenceFile& ref, const std::vector<double>& r, femorax::CrdTransfType type,
                 double worst[9]) {
  auto at = [&](const std::string& col) { return r[static_cast<std::size_t>(ref.column(col))]; };
  auto atk = [&](const std::string& col, int k) { return at(col + std::to_string(k)); };
  typename Beam<NIP>::Params p;
  p.mat = {at("E"), at("A"), at("Iz"), at("Iy"), at("G"), at("J")};
  p.transf.type = type;
  p.rhoA = at("rhoA");
  double xe[2][3];
  for (int i = 0; i < 3; ++i) {
    xe[0][i] = atk("xi", i);
    xe[1][i] = atk("xj", i);
    p.transf.vecxz[i] = atk("vecxz", i);
    p.transf.nodeIOffset[i] = atk("offi", i);
    p.transf.nodeJOffset[i] = atk("offj", i);
  }
  double ug[12], os_f[12], os_K[144], os_K0[144], os_M[144];
  for (int k = 0; k < 12; ++k) ug[k] = atk("ug", k), os_f[k] = atk("f", k);
  for (int k = 0; k < 144; ++k) os_K[k] = atk("K", k), os_K0[k] = atk("K0", k), os_M[k] = atk("M", k);
  double f[12], K[12][12], f0[12], K0[12][12];
  Beam<NIP>::template compute<true>(p, xe, ug, no_state(), no_trial(), f, K);
  const double zero[12] = {};
  Beam<NIP>::template compute<true>(p, xe, zero, no_state(), no_trial(), f0, K0);  // the initial stiffness
  double m[12], M[144] = {};
  Beam<NIP>::lumped_mass(p, xe, m);
  for (int k = 0; k < 12; ++k) M[13 * k] = m[k];
  worst[0] = std::max(worst[0], rel_diff(f, os_f, 12));
  worst[1] = std::max(worst[1], rel_diff(&K[0][0], os_K, 144));
  worst[2] = std::max(worst[2], rel_diff(&K0[0][0], os_K0, 144));
  worst[3] = std::max(worst[3], rel_diff(M, os_M, 144));
  // The consistent mass (OpenSees': no rotary or torsional inertia).
  {
    auto pc = p;
    pc.cMass = 1;
    femorax::CrdTransf3dGeometry g;
    ASSERT_EQ(Beam<NIP>::geometry(pc, xe, g), 0);
    double Mc[12][12], os_Mc[144];
    Beam<NIP>::consistent_mass(pc, g, Mc);
    for (int k = 0; k < 144; ++k) os_Mc[k] = atk("Mc", k);
    worst[4] = std::max(worst[4], rel_diff(&Mc[0][0], os_Mc, 144));
  }
  // Rayleigh damping, one factor at a time (the lumped mass for alphaM).
  double vg[12];
  for (int k = 0; k < 12; ++k) vg[k] = atk("vg", k);
  const char* cols[4] = {"dA", "dK", "dK0", "dKc"};
  const double factors[4][4] = {{0.7, 0, 0, 0}, {0, 2e-3, 0, 0}, {0, 0, 1e-3, 0}, {0, 0, 0, 1.5e-3}};
  for (int c = 0; c < 4; ++c) {
    femorax::Rayleigh d;
    d.alphaM = factors[c][0], d.betaK = factors[c][1], d.betaKinit = factors[c][2], d.betaKcomm = factors[c][3];
    // The damping force is the difference of two force evaluations; alphaM's is about 1e-6 of
    // the internal force, so it is taken at zero displacement (its value does not depend on
    // it), where the internal force is exactly zero.
    const double zero_u[12] = {};
    const double* u_used = c == 0 ? zero_u : ug;
    double fd[12], fu[12], os_d[12];
    Beam<NIP>::template compute<false>(p, d, xe, u_used, vg, no_state(), no_trial(), fd, nullptr);
    Beam<NIP>::template compute<false>(p, femorax::Rayleigh{}, xe, u_used, vg, no_state(), no_trial(), fu, nullptr);
    for (int k = 0; k < 12; ++k) fd[k] -= fu[k], os_d[k] = atk(cols[c], k);
    const double e = rel_diff(fd, os_d, 12);
    if (c == 3 && type == femorax::CrdTransfType::PDelta) {
      worst[8] = std::max(worst[8], e);  // the documented difference: no geometric term in Kc
    } else {
      worst[5 + c] = std::max(worst[5 + c], e);
    }
  }
}

class DispBeam3dReference : public ::testing::TestWithParam<femorax::CrdTransfType> {};

TEST_P(DispBeam3dReference, MatchesOpenSees) {
  const auto type = GetParam();
  const std::string name = type == femorax::CrdTransfType::Linear ? "Linear" : "PDelta";
  const auto ref = femorax::test::read_reference(std::string(FEMORAX_REFERENCE_DATA) + "/element/DispBeamColumn3d_" +
                                                 name + ".txt");
  ASSERT_EQ(ref.rows.size(), 24u);
  double worst[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (const auto& r : ref.rows) {
    const int nip = static_cast<int>(r[static_cast<std::size_t>(ref.column("nip"))]);
    if (nip == 2) compare_row<2>(ref, r, type, worst);
    else if (nip == 3) compare_row<3>(ref, r, type, worst);
    else compare_row<5>(ref, r, type, worst);
  }
  std::printf("[ DispBeamColumn3d %s against OpenSees: f %.2e, K %.2e, K0 %.2e, M %.2e, Mc %.2e; damping alphaM %.2e, "
              "betaK %.2e, betaK0 %.2e, betaKc %.2e ]\n",
              name.c_str(), worst[0], worst[1], worst[2], worst[3], worst[4], worst[5], worst[6], worst[7], worst[8]);
  for (int k = 0; k < 5; ++k) EXPECT_LE(worst[k], 1e-14) << k;
  // The damping forces are formed in the basic system, OpenSees' as a 12 x 12 matrix times v.
  for (int k = 5; k < 8; ++k) EXPECT_LE(worst[k], 1e-13) << k;
  if (type == femorax::CrdTransfType::Linear) {
    EXPECT_LE(worst[8], 1e-13);
  }
}

INSTANTIATE_TEST_SUITE_P(Types, DispBeam3dReference,
                         ::testing::Values(femorax::CrdTransfType::Linear, femorax::CrdTransfType::PDelta),
                         [](const auto& tp) { return tp.param == femorax::CrdTransfType::Linear ? "Linear" : "PDelta"; });

}  // namespace
