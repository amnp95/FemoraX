// The 3D Linear and PDelta coordinate transformations (physics/transformations/
// crd_transf_3d.hpp):
//   - against OpenSees' LinearCrdTransf3d and PDeltaCrdTransf3d (tests/reference/drivers/
//     crd_transf.cpp, commit 955ceb0): basic displacements, global forces, tangent and initial
//     stiffness, for six geometries (axis-aligned, vertical, skewed, rigid joint offsets);
//   - mechanics: rigid-body motion (translation and small rotation, offsets included) gives no
//     basic deformation; the force and stiffness are the transpose of the same map
//     (pg = T^T pb and kg = T^T kb T for kb, pb = kb ub); PDelta's geometric stiffness is the
//     exact derivative of its force.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "physics/transformations/crd_transf_3d.hpp"
#include "reference_data.hpp"

namespace {

using femorax::CrdTransf3dGeometry;
using femorax::CrdTransf3dParams;
using femorax::CrdTransfType;

struct Case {
  CrdTransf3dParams p;
  double xi[3], xj[3];
};

Case case_of(const femorax::test::ReferenceFile& ref, const std::vector<double>& r, CrdTransfType type) {
  Case c;
  c.p.type = type;
  for (int i = 0; i < 3; ++i) {
    c.xi[i] = r[static_cast<std::size_t>(ref.column("xi" + std::to_string(i)))];
    c.xj[i] = r[static_cast<std::size_t>(ref.column("xj" + std::to_string(i)))];
    c.p.vecxz[i] = r[static_cast<std::size_t>(ref.column("vecxz" + std::to_string(i)))];
    c.p.nodeIOffset[i] = r[static_cast<std::size_t>(ref.column("offi" + std::to_string(i)))];
    c.p.nodeJOffset[i] = r[static_cast<std::size_t>(ref.column("offj" + std::to_string(i)))];
  }
  return c;
}

// Largest |a - b| over n values, relative to the largest |b|.
double rel_diff(const double* a, const double* b, int n) {
  double d = 0.0, m = 0.0;
  for (int i = 0; i < n; ++i) {
    d = std::max(d, std::abs(a[i] - b[i]));
    m = std::max(m, std::abs(b[i]));
  }
  return m > 0.0 ? d / m : d;
}

class CrdTransf3dReference : public ::testing::TestWithParam<CrdTransfType> {};

TEST_P(CrdTransf3dReference, MatchesOpenSees) {
  const CrdTransfType type = GetParam();
  const std::string name = type == CrdTransfType::Linear ? "Linear" : "PDelta";
  const auto ref = femorax::test::read_reference(std::string(FEMORAX_REFERENCE_DATA) + "/transf/CrdTransf3d_" +
                                                    name + ".txt");
  ASSERT_EQ(ref.rows.size(), 18u);
  double worst[4] = {0, 0, 0, 0};
  for (const auto& r : ref.rows) {
    const Case c = case_of(ref, r, type);
    CrdTransf3dGeometry g;
    ASSERT_EQ(femorax::crd_transf_3d_geometry(c.p, c.xi, c.xj, g), 0);
    auto at = [&](const std::string& col, int k) { return r[static_cast<std::size_t>(ref.column(col + std::to_string(k)))]; };
    double ug[12], pb[6], p0[5], kb[6][6];
    for (int k = 0; k < 12; ++k) ug[k] = at("ug", k);
    for (int k = 0; k < 6; ++k) pb[k] = at("pb", k);
    for (int k = 0; k < 5; ++k) p0[k] = at("p0", k);
    for (int a = 0; a < 6; ++a)
      for (int b = 0; b < 6; ++b) kb[a][b] = at("kb", 6 * a + b);
    double ub[6], pg[12], kg[12][12], k0[12][12];
    femorax::crd_transf_3d_basic_disp(c.p, g, ug, ub);
    femorax::crd_transf_3d_global_force(c.p, g, ug, pb, p0, pg);
    femorax::crd_transf_3d_global_stiff(c.p, g, kb, pb, true, kg);
    femorax::crd_transf_3d_global_stiff(c.p, g, kb, pb, false, k0);
    double os_ub[6], os_pg[12], os_kg[144], os_k0[144];
    for (int k = 0; k < 6; ++k) os_ub[k] = at("ub", k);
    for (int k = 0; k < 12; ++k) os_pg[k] = at("pg", k);
    for (int k = 0; k < 144; ++k) os_kg[k] = at("kg", k), os_k0[k] = at("k0", k);
    worst[0] = std::max(worst[0], rel_diff(ub, os_ub, 6));
    worst[1] = std::max(worst[1], rel_diff(pg, os_pg, 12));
    worst[2] = std::max(worst[2], rel_diff(&kg[0][0], os_kg, 144));
    worst[3] = std::max(worst[3], rel_diff(&k0[0][0], os_k0, 144));
  }
  std::printf("[ %s against OpenSees: ub %.2e, pg %.2e, kg %.2e, k0 %.2e (relative to each quantity's peak) ]\n",
              name.c_str(), worst[0], worst[1], worst[2], worst[3]);
  for (const double w : worst) EXPECT_LE(w, 1e-14);
}

INSTANTIATE_TEST_SUITE_P(Types, CrdTransf3dReference, ::testing::Values(CrdTransfType::Linear, CrdTransfType::PDelta),
                         [](const auto& tp) { return tp.param == CrdTransfType::Linear ? "Linear" : "PDelta"; });

// A skewed member with offsets at both ends.
Case skewed(CrdTransfType type) {
  Case c;
  c.p.type = type;
  const double xi[3] = {0.3, -1.2, 0.7}, xj[3] = {2.9, 0.4, 2.2};
  for (int i = 0; i < 3; ++i) c.xi[i] = xi[i], c.xj[i] = xj[i];
  const double v[3] = {-0.2, 0.3, 1.0}, oi[3] = {0.25, -0.1, 0.05}, oj[3] = {-0.3, 0.2, -0.15};
  for (int i = 0; i < 3; ++i) c.p.vecxz[i] = v[i], c.p.nodeIOffset[i] = oi[i], c.p.nodeJOffset[i] = oj[i];
  return c;
}

TEST(CrdTransf3d, RigidBodyMotionGivesNoBasicDeformation) {
  for (const auto type : {CrdTransfType::Linear, CrdTransfType::PDelta}) {
    const Case c = skewed(type);
    CrdTransf3dGeometry g;
    ASSERT_EQ(femorax::crd_transf_3d_geometry(c.p, c.xi, c.xj, g), 0);
    // u(x) = a + theta x (x - xi) at each node; rotations theta at both nodes.
    const double a[3] = {0.013, -0.007, 0.021}, th[3] = {2e-3, -1.5e-3, 3e-3};
    double ug[12];
    for (int n = 0; n < 2; ++n) {
      const double* x = n == 0 ? c.xi : c.xj;
      const double r[3] = {x[0] - c.xi[0], x[1] - c.xi[1], x[2] - c.xi[2]};
      ug[6 * n + 0] = a[0] + th[1] * r[2] - th[2] * r[1];
      ug[6 * n + 1] = a[1] + th[2] * r[0] - th[0] * r[2];
      ug[6 * n + 2] = a[2] + th[0] * r[1] - th[1] * r[0];
      for (int d = 0; d < 3; ++d) ug[6 * n + 3 + d] = th[d];
    }
    double ub[6];
    femorax::crd_transf_3d_basic_disp(c.p, g, ug, ub);
    for (int k = 0; k < 6; ++k) EXPECT_NEAR(ub[k], 0.0, 1e-17) << k;
  }
}

TEST(CrdTransf3d, ForceAndStiffnessAreTheTransposeOfTheKinematics) {
  const Case c = skewed(CrdTransfType::Linear);
  CrdTransf3dGeometry g;
  ASSERT_EQ(femorax::crd_transf_3d_geometry(c.p, c.xi, c.xj, g), 0);
  // T column by column: ub of unit global displacements.
  double T[6][12];
  for (int j = 0; j < 12; ++j) {
    double e[12] = {};
    e[j] = 1.0;
    double ub[6];
    femorax::crd_transf_3d_basic_disp(c.p, g, e, ub);
    for (int i = 0; i < 6; ++i) T[i][j] = ub[i];
  }
  const double pb[6] = {120.0, -35.0, 48.0, 17.0, -22.0, 9.0}, p0[5] = {0, 0, 0, 0, 0};
  double kb[6][6];
  for (int a = 0; a < 6; ++a)
    for (int b = 0; b < 6; ++b) kb[a][b] = (a == b ? 1e4 : 0.0) + 300.0 * (a + 1) * (b + 1);
  double ug[12] = {};
  double pg[12], kg[12][12];
  femorax::crd_transf_3d_global_force(c.p, g, ug, pb, p0, pg);
  femorax::crd_transf_3d_global_stiff(c.p, g, kb, pb, true, kg);
  double dp = 0.0, dk = 0.0, mp = 0.0, mk = 0.0;
  for (int i = 0; i < 12; ++i) {
    double t = 0.0;
    for (int a = 0; a < 6; ++a) t += T[a][i] * pb[a];
    dp = std::max(dp, std::abs(t - pg[i]));
    mp = std::max(mp, std::abs(pg[i]));
    for (int j = 0; j < 12; ++j) {
      double s = 0.0;
      for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b) s += T[a][i] * kb[a][b] * T[b][j];
      dk = std::max(dk, std::abs(s - kg[i][j]));
      mk = std::max(mk, std::abs(kg[i][j]));
    }
  }
  EXPECT_LE(dp, 1e-13 * mp);
  EXPECT_LE(dk, 1e-13 * mk);
}

// PDelta's force is linear in the trial displacements for fixed basic forces; the change of the
// force along any direction is the geometric part of the tangent times that direction.
TEST(CrdTransf3d, PDeltaGeometricStiffnessIsTheDerivativeOfItsForce) {
  const Case c = skewed(CrdTransfType::PDelta);
  CrdTransf3dGeometry g;
  ASSERT_EQ(femorax::crd_transf_3d_geometry(c.p, c.xi, c.xj, g), 0);
  const double pb[6] = {-850.0, 12.0, -7.0, 4.0, 3.0, -2.0}, p0[5] = {0, 0, 0, 0, 0};
  const double zero_kb[6][6] = {};
  double kgeo[12][12];
  femorax::crd_transf_3d_global_stiff(c.p, g, zero_kb, pb, true, kgeo);
  double ug[12], du[12];
  for (int k = 0; k < 12; ++k) ug[k] = 1e-3 * std::sin(1.0 + k), du[k] = 1e-4 * std::cos(2.0 + 3 * k);
  double ug2[12];
  for (int k = 0; k < 12; ++k) ug2[k] = ug[k] + du[k];
  double f1[12], f2[12];
  femorax::crd_transf_3d_global_force(c.p, g, ug, pb, p0, f1);
  femorax::crd_transf_3d_global_force(c.p, g, ug2, pb, p0, f2);
  double d = 0.0, m = 0.0;
  for (int i = 0; i < 12; ++i) {
    double kdu = 0.0;
    for (int j = 0; j < 12; ++j) kdu += kgeo[i][j] * du[j];
    d = std::max(d, std::abs((f2[i] - f1[i]) - kdu));
    m = std::max(m, std::abs(kdu));
  }
  ASSERT_GT(m, 0.0);
  EXPECT_LE(d, 1e-9 * m);  // the difference of two forces of size |pb|: rounding of |pb| eps
  // Linear has no geometric term.
  Case l = c;
  l.p.type = CrdTransfType::Linear;
  double klin[12][12];
  femorax::crd_transf_3d_global_stiff(l.p, g, zero_kb, pb, true, klin);
  for (int i = 0; i < 12; ++i)
    for (int j = 0; j < 12; ++j) EXPECT_EQ(klin[i][j], 0.0);
}

TEST(CrdTransf3d, RefusesAZeroLengthAndAnAxisParallelVecxz) {
  CrdTransf3dParams p;
  p.vecxz[2] = 1.0;
  CrdTransf3dGeometry g;
  const double a[3] = {1, 2, 3}, b[3] = {1, 2, 5};
  EXPECT_EQ(femorax::crd_transf_3d_geometry(p, a, a, g), -2);
  EXPECT_EQ(femorax::crd_transf_3d_geometry(p, a, b, g), -3);  // vecxz along the axis
}

}  // namespace
