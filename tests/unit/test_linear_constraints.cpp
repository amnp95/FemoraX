// Exact linear constraints u = T q, compile step (compile/linear_constraints.hpp; docs/design.md,
// "Exact constraints u = T q", step 1): each generator against mechanics, and the resolver's rules.
//   - rigidLink beam: any small rigid motion of the master gives the slave the same rigid motion;
//     bar: equal translations;
//   - rigidDiaphragm (each normal direction): an in-plane rigid motion of the master moves the
//     slave rigidly in the plane, the other DOFs untouched;
//   - the embedded node in a distorted brick: its natural coordinates map back to the node, the
//     weights sum to one and reproduce any linear field exactly (the patch test of the tie); with
//     rot, a rigid rotation of the brick gives the node that rotation and a pure strain none;
//   - the resolver: a chain is substituted, a cycle and a DOF constrained twice are refused.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "compile/linear_constraints.hpp"
#include "io/h5.hpp"

namespace {

using femorax::compile::LinearConstraint;

// The constrained values from the resolved rows: u[dof] = sum w u[term].
std::vector<double> apply(const femorax::compile::ResolvedConstraints& r, std::vector<double> u) {
  for (std::size_t i = 0; i < r.rows(); ++i) {
    double s = 0.0;
    for (auto k = r.start[i]; k < r.start[i + 1]; ++k)
      s += r.term_weight[static_cast<std::size_t>(k)] * u[static_cast<std::size_t>(r.term_dof[static_cast<std::size_t>(k)])];
    u[static_cast<std::size_t>(r.dof[i])] = s;
  }
  return u;
}

// A small rigid motion (translation t, rotation w about the origin) at point x: u and theta.
void rigid(const double t[3], const double w[3], const std::array<double, 3>& x, double* u6) {
  u6[0] = t[0] + w[1] * x[2] - w[2] * x[1];
  u6[1] = t[1] + w[2] * x[0] - w[0] * x[2];
  u6[2] = t[2] + w[0] * x[1] - w[1] * x[0];
  u6[3] = w[0], u6[4] = w[1], u6[5] = w[2];
}

TEST(LinearConstraints, RigidLinkBeamMovesTheSlaveRigidly) {
  const std::array<double, 3> xm{0.3, -1.1, 2.0}, xs{1.7, 0.4, 2.9};
  std::vector<LinearConstraint> c;
  femorax::compile::rigid_link_beam(0, 6, xm, xs, c, "rigidLink beam 1 2");
  const auto r = femorax::compile::resolve_constraints(12, c);
  ASSERT_EQ(r.rows(), 6u);
  const double t[3] = {0.01, -0.02, 0.005}, w[3] = {2e-3, -1e-3, 3e-3};
  std::vector<double> u(12, 0.0);
  rigid(t, w, xm, u.data());
  const auto v = apply(r, u);
  double want[6];
  rigid(t, w, xs, want);
  for (int d = 0; d < 6; ++d) EXPECT_NEAR(v[static_cast<std::size_t>(6 + d)], want[d], 1e-16) << d;
  // bar: translations only.
  std::vector<LinearConstraint> b;
  femorax::compile::rigid_link_bar(0, 6, b, "rigidLink bar 1 2");
  const auto rb = femorax::compile::resolve_constraints(12, b);
  ASSERT_EQ(rb.rows(), 3u);
  const auto vb = apply(rb, u);
  for (int d = 0; d < 3; ++d) EXPECT_EQ(vb[static_cast<std::size_t>(6 + d)], u[static_cast<std::size_t>(d)]);
}

TEST(LinearConstraints, RigidDiaphragmMovesTheSlaveRigidlyInItsPlane) {
  for (int perp = 1; perp <= 3; ++perp) {
    const int n = perp - 1;
    std::array<double, 3> xm{1.0, 2.0, 3.0}, xs{2.5, 0.5, 4.0};
    xs[static_cast<std::size_t>(n)] = xm[static_cast<std::size_t>(n)];  // in the plane
    std::vector<LinearConstraint> c;
    femorax::compile::rigid_diaphragm(perp, 0, 6, xm, xs, c, "rigidDiaphragm");
    const auto r = femorax::compile::resolve_constraints(12, c);
    ASSERT_EQ(r.rows(), 3u);
    // An in-plane rigid motion: translation in the plane, rotation about the normal.
    double t[3] = {0.01, 0.02, 0.03}, w[3] = {0, 0, 0};
    t[n] = 0.0;
    w[n] = 2e-3;
    std::vector<double> u(12, 0.0);
    rigid(t, w, xm, u.data());
    for (int d = 0; d < 6; ++d) u[static_cast<std::size_t>(6 + d)] = 7.0;  // slave's own values
    const auto v = apply(r, u);
    double want[6];
    rigid(t, w, xs, want);
    for (int d = 0; d < 6; ++d) {
      const bool constrained = d == (n + 1) % 3 || d == (n + 2) % 3 || d == 3 + n;
      EXPECT_NEAR(v[static_cast<std::size_t>(6 + d)], constrained ? want[d] : 7.0, 1e-16) << "perp " << perp << " dof " << d;
    }
  }
}

// A distorted brick (corner order of stdBrick).
void distorted(double (&xe)[8][3]) {
  const double c[8][3] = {{0, 0, 0}, {1.2, 0.1, 0}, {1.3, 1.1, 0.1}, {-0.1, 0.9, 0},
                          {0.1, 0, 1.0}, {1.1, -0.1, 1.2}, {1.2, 1.0, 1.1}, {0, 1.1, 0.9}};
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 3; ++j) xe[i][j] = c[i][j];
}

TEST(LinearConstraints, EmbeddedNodeInABrickReproducesLinearFields) {
  double xe[8][3];
  distorted(xe);
  // A point inside: the image of a known natural point.
  const std::array<double, 3> xi0{0.3, -0.45, 0.6};
  double N[8], dN[8][3];
  femorax::compile::hex8_shape(xi0, N, dN);
  std::array<double, 3> x{0, 0, 0};
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 3; ++j) x[static_cast<std::size_t>(j)] += N[i] * xe[i][j];
  std::array<double, 3> xi{};
  ASSERT_TRUE(femorax::compile::hex8_natural_coordinates(xe, x, xi));
  for (int k = 0; k < 3; ++k) EXPECT_NEAR(xi[static_cast<std::size_t>(k)], xi0[static_cast<std::size_t>(k)], 1e-12);
  // A point outside is reported.
  std::array<double, 3> far{3.0, 3.0, 3.0}, xi2{};
  EXPECT_FALSE(femorax::compile::hex8_natural_coordinates(xe, far, xi2));

  // The tie: host nodes 3 DOFs each at offsets 0, 3, ..., 21; the embedded node (6 DOFs) at 24.
  const std::int32_t host[8] = {0, 3, 6, 9, 12, 15, 18, 21};
  std::vector<LinearConstraint> c;
  femorax::compile::embedded_in_hex8(host, xe, 24, x, true, c, "embedded 100 in brick 1");
  const auto r = femorax::compile::resolve_constraints(30, c);
  ASSERT_EQ(r.rows(), 6u);
  // A linear displacement field u(x) = a + G x at the host nodes.
  auto field = [](const double G[3][3], const double a[3], std::vector<double>& u, const double (&p)[8][3]) {
    for (int i = 0; i < 8; ++i)
      for (int d = 0; d < 3; ++d) u[static_cast<std::size_t>(3 * i + d)] = a[d] + G[d][0] * p[i][0] + G[d][1] * p[i][1] + G[d][2] * p[i][2];
  };
  const double a[3] = {0.01, -0.02, 0.03};
  {  // a general linear field: the node gets u(x) exactly
    const double G[3][3] = {{1e-3, 2e-3, -1e-3}, {0.5e-3, -2e-3, 3e-3}, {-1e-3, 1e-3, 2e-3}};
    std::vector<double> u(30, 0.0);
    field(G, a, u, xe);
    const auto v = apply(r, u);
    for (int d = 0; d < 3; ++d)
      EXPECT_NEAR(v[static_cast<std::size_t>(24 + d)], a[d] + G[d][0] * x[0] + G[d][1] * x[1] + G[d][2] * x[2], 1e-15) << d;
    // rot: half the curl of G
    const double th[3] = {0.5 * (G[2][1] - G[1][2]), 0.5 * (G[0][2] - G[2][0]), 0.5 * (G[1][0] - G[0][1])};
    for (int d = 0; d < 3; ++d) EXPECT_NEAR(v[static_cast<std::size_t>(27 + d)], th[d], 1e-15) << d;
  }
  {  // a rigid rotation w: the node rotates by w
    const double w[3] = {2e-3, -1e-3, 3e-3};
    const double G[3][3] = {{0, -w[2], w[1]}, {w[2], 0, -w[0]}, {-w[1], w[0], 0}};
    std::vector<double> u(30, 0.0);
    field(G, a, u, xe);
    const auto v = apply(r, u);
    for (int d = 0; d < 3; ++d) EXPECT_NEAR(v[static_cast<std::size_t>(27 + d)], w[d], 1e-15) << d;
  }
  {  // a pure (symmetric) strain: no rotation
    const double G[3][3] = {{1e-3, 2e-3, -1e-3}, {2e-3, -2e-3, 3e-3}, {-1e-3, 3e-3, 2e-3}};
    std::vector<double> u(30, 0.0);
    field(G, a, u, xe);
    const auto v = apply(r, u);
    for (int d = 0; d < 3; ++d) EXPECT_NEAR(v[static_cast<std::size_t>(27 + d)], 0.0, 1e-15) << d;
  }
  // Partition of unity: the translation weights sum to one.
  for (std::size_t i = 0; i < 3; ++i) {
    double s = 0.0;
    for (auto k = r.start[i]; k < r.start[i + 1]; ++k) s += r.term_weight[static_cast<std::size_t>(k)];
    EXPECT_NEAR(s, 1.0, 1e-15);
  }
}

TEST(LinearConstraints, ResolverSubstitutesChainsAndRefusesCyclesAndDoubles) {
  // dof 0 = 2 * dof 1, dof 1 = dof 2 + 3 * dof 3: dof 0 = 2 dof 2 + 6 dof 3.
  std::vector<LinearConstraint> c = {{0, {{1, 2.0}}, "a"}, {1, {{2, 1.0}, {3, 3.0}}, "b"}};
  const auto r = femorax::compile::resolve_constraints(4, c);
  ASSERT_EQ(r.rows(), 2u);
  EXPECT_EQ(r.dof[0], 0);
  ASSERT_EQ(r.start[1] - r.start[0], 2);
  EXPECT_EQ(r.term_dof[0], 2);
  EXPECT_EQ(r.term_weight[0], 2.0);
  EXPECT_EQ(r.term_dof[1], 3);
  EXPECT_EQ(r.term_weight[1], 6.0);
  // A cycle.
  std::vector<LinearConstraint> cyc = {{0, {{1, 1.0}}, "first"}, {1, {{0, 1.0}}, "second"}};
  EXPECT_THROW(femorax::compile::resolve_constraints(2, cyc), femorax::io::ModelError);
  // A DOF constrained twice.
  std::vector<LinearConstraint> twice = {{0, {{1, 1.0}}, "first"}, {0, {{2, 1.0}}, "second"}};
  try {
    femorax::compile::resolve_constraints(3, twice);
    FAIL() << "expected a refusal";
  } catch (const femorax::io::ModelError& e) {
    EXPECT_NE(std::string(e.what()).find("constrained twice"), std::string::npos);
  }
}

}  // namespace
