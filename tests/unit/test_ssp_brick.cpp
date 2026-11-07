// SSPbrick properties that need no reference data: constant strain from linear fields,
// agreement with stdBrick for them, the patch test, rigid-body and hourglass modes, the
// element term against its entries, consistency of force, tangent, mass and damping, the
// geometry sources, and the stable step. The comparison with OpenSees SSPbrick is
// test_ssp_brick_reference.cpp.

#include "physics/elements/ssp_brick.hpp"
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"
#include "../reference/drivers/driver_common.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using femorax::ElasticIsotropic;
using femorax::ExplicitScheme;
using femorax::Rayleigh;
using Ssp = femorax::SSPbrick<ElasticIsotropic>;
using Brick = femorax::StdBrick<ElasticIsotropic>;
constexpr int NDOF = Ssp::NDOF;

static_assert(femorax::Element<Ssp>);
static_assert(femorax::ExplicitElement<Ssp>);
static_assert(femorax::PointwiseElement<Ssp>);
static_assert(femorax::MatrixFreeElement<Ssp>);
static_assert(femorax::ElementTermElement<Ssp>);
static_assert(femorax::CompatiblePair<Ssp, ElasticIsotropic>);
static_assert(femorax::initial_stiffness_is_tangent<Ssp>());
static_assert(!femorax::ElementTermElement<Brick>);
static_assert(!femorax::initial_stiffness_is_tangent<Brick>());

using Coords = std::array<std::array<double, 3>, 8>;

Ssp::Params params(double b3 = -19.9) { return Ssp::Params{{377000.0, 0.3, 2.0285}, {0.0, 0.0, b3}}; }

Coords box(double a = 1.0, double b = 1.0, double c = 1.0) {
  return {{{0, 0, 0}, {a, 0, 0}, {a, b, 0}, {0, b, 0}, {0, 0, c}, {a, 0, c}, {a, b, c}, {0, b, c}}};
}

Coords distorted(double amp = 0.12) {
  Coords x = box(1.3, 0.8, 1.1);
  refdrv::Lcg rng(7);
  for (auto& n : x)
    for (auto& v : n) v += amp * rng.symmetric();
  return x;
}

// An affine image of a box (a sheared, rotated parallelepiped): no hourglass geometry.
Coords parallelepiped() {
  Coords x = box(1.2, 0.7, 0.9);
  const double F[3][3] = {{1.0, 0.3, -0.2}, {0.1, 0.9, 0.25}, {-0.15, 0.2, 1.1}};
  for (auto& n : x) {
    const double X[3] = {n[0], n[1], n[2]};
    for (int i = 0; i < 3; ++i) n[i] = F[i][0] * X[0] + F[i][1] * X[1] + F[i][2] * X[2] + (i == 0 ? 3.0 : -1.0);
  }
  return x;
}

const double (*xe(const Coords& c))[3] { return reinterpret_cast<const double(*)[3]>(c.data()); }

struct Eval {
  double fe[NDOF];
  double Ke[NDOF][NDOF];
};

Eval eval(const Ssp::Params& p, const Coords& x, const double* u, const double* v, const Rayleigh& d = {}) {
  Eval e{};
  Ssp::compute<true>(p, d, xe(x), u, v, {}, {}, e.fe, e.Ke);
  return e;
}

std::vector<double> zeros() { return std::vector<double>(NDOF, 0.0); }

double max_abs(const double* a, int n) {
  double m = 0;
  for (int i = 0; i < n; ++i) m = std::max(m, std::abs(a[i]));
  return m;
}

const double kA[3][3] = {{1e-3, 2e-4, -3e-4}, {5e-4, -7e-4, 1e-4}, {-2e-4, 3e-4, 4e-4}};

std::vector<double> linear_field(const Coords& x) {
  std::vector<double> u(NDOF);
  for (int n = 0; n < 8; ++n)
    for (int i = 0; i < 3; ++i) u[3 * n + i] = 0.01 * i + kA[i][0] * x[n][0] + kA[i][1] * x[n][1] + kA[i][2] * x[n][2];
  return u;
}

// The element constants as the launch layer builds them.
struct Constants {
  double c[Ssp::NCONST];
  femorax::Strided<const double> view() const { return {c, 1}; }
};
Constants constants(const Ssp::Params& p, const Coords& x) {
  Constants k;
  double xl[3][8];
  Ssp::local_coords(xe(x), xl);
  Ssp::element_constants(p, xl, {k.c, 1});
  return k;
}

// Kstab v from the element-term functions.
std::vector<double> kstab_times(const Ssp::Params& p, const Constants& k, const double* v) {
  double z[Ssp::NTERM];
  Ssp::element_term_reduce(p, k.view(), v, z);
  std::vector<double> y(NDOF);
  for (int j = 0; j < 8; ++j) Ssp::element_term_node(p, k.view(), z, j, &y[3 * j]);
  return y;
}

// ---------------------------------------------------------------------------
TEST(SSPbrick, LinearFieldsGiveConstantStrainAndNoHourglassForce) {
  for (const Coords& x : {box(), distorted(), parallelepiped()}) {
    const auto u = linear_field(x);
    double xl[3][8];
    Ssp::local_coords(xe(x), xl);
    Ssp::PointShape s;
    Ssp::point_geometry(xl, 0, s);
    double eps[6];
    Ssp::point_strain(s, u.data(), eps);
    const double want[6] = {kA[0][0], kA[1][1], kA[2][2], kA[0][1] + kA[1][0], kA[1][2] + kA[2][1],
                            kA[0][2] + kA[2][0]};
    for (int c = 0; c < 6; ++c) EXPECT_NEAR(eps[c], want[c], 1e-15) << "strain " << c;
    // The hourglass vectors are orthogonal to linear fields: Kstab u = 0.
    const auto k = constants(params(), x);
    const auto y = kstab_times(params(), k, u.data());
    double kmax = 0;
    for (int a = 0; a < NDOF; ++a) kmax = std::max(kmax, std::abs(Ssp::element_term_entry(params(), k.view(), a, a)));
    for (double v : y) EXPECT_NEAR(v, 0.0, 1e-13 * kmax * max_abs(u.data(), NDOF));
  }
}

TEST(SSPbrick, MatchesStdBrickForLinearFields) {
  // Both elements integrate constant stress exactly (SSPbrick's dNmod is the uniform
  // gradient: mVol dNmod = the integral of grad N), so their forces agree.
  for (const Coords& x : {box(1.3, 0.8, 1.1), distorted(), parallelepiped()}) {
    const auto u = linear_field(x);
    const auto z = zeros();
    double fs[NDOF], fb[NDOF];
    Ssp::compute<false>(params(0.0), {}, xe(x), u.data(), z.data(), {}, {}, fs, nullptr);
    const Brick::Params pb{{377000.0, 0.3, 2.0285}, {0.0, 0.0, 0.0}, 1};
    Brick::compute<false>(pb, {}, xe(x), u.data(), z.data(), {}, {}, fb, nullptr);
    const double scale = max_abs(fb, NDOF);
    for (int a = 0; a < NDOF; ++a) EXPECT_NEAR(fs[a], fb[a], 1e-12 * scale) << "dof " << a;
  }
}

TEST(SSPbrick, PatchTestOnDistortedEightElementMesh) {
  double nodes[27][3];
  refdrv::Lcg rng(42);
  for (int k = 0; k < 3; ++k)
    for (int j = 0; j < 3; ++j)
      for (int i = 0; i < 3; ++i) {
        double* X = nodes[9 * k + 3 * j + i];
        X[0] = i * 1.0;
        X[1] = j * 0.9;
        X[2] = k * 1.2;
        if (i == 1 || j == 1 || k == 1)
          for (int c = 0; c < 3; ++c) X[c] += 0.15 * rng.symmetric();
      }
  auto lin = [&](const double* X, int i) { return kA[i][0] * X[0] + kA[i][1] * X[1] + kA[i][2] * X[2]; };
  const int interior = 13;
  const Ssp::Params p = params(0.0);
  Eigen::Matrix3d Kii = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
  for (int ek = 0; ek < 2; ++ek)
    for (int ej = 0; ej < 2; ++ej)
      for (int ei = 0; ei < 2; ++ei) {
        auto id = [&](int i, int j, int k) { return 9 * (ek + k) + 3 * (ej + j) + (ei + i); };
        const int conn[8] = {id(0, 0, 0), id(1, 0, 0), id(1, 1, 0), id(0, 1, 0),
                             id(0, 0, 1), id(1, 0, 1), id(1, 1, 1), id(0, 1, 1)};
        Coords x;
        double u[NDOF];
        int local = -1;
        for (int n = 0; n < 8; ++n) {
          for (int c = 0; c < 3; ++c) x[n][c] = nodes[conn[n]][c];
          for (int c = 0; c < 3; ++c) u[3 * n + c] = conn[n] == interior ? 0.0 : lin(nodes[conn[n]], c);
          if (conn[n] == interior) local = n;
        }
        ASSERT_GE(local, 0);
        const auto z = zeros();
        const Eval e = eval(p, x, u, z.data());
        for (int a = 0; a < 3; ++a) {
          rhs[a] -= e.fe[3 * local + a];
          for (int b = 0; b < 3; ++b) Kii(a, b) += e.Ke[3 * local + a][3 * local + b];
        }
      }
  const Eigen::Vector3d ui = Kii.fullPivLu().solve(rhs);
  for (int c = 0; c < 3; ++c) EXPECT_NEAR(ui[c], lin(nodes[interior], c), 1e-14) << "component " << c;
}

TEST(SSPbrick, TangentIsSymmetricWithSixRigidModesAndStabilizedHourglassModes) {
  for (const Coords& x : {box(), distorted(), parallelepiped()}) {
    const auto z = zeros();
    const Eval e = eval(params(), x, z.data(), z.data());
    const double scale = max_abs(&e.Ke[0][0], NDOF * NDOF);
    Eigen::MatrixXd K(NDOF, NDOF), Kpoint(NDOF, NDOF);
    const auto k = constants(params(), x);
    for (int a = 0; a < NDOF; ++a)
      for (int c = 0; c < NDOF; ++c) {
        EXPECT_NEAR(e.Ke[a][c], e.Ke[c][a], 1e-13 * scale);
        K(a, c) = e.Ke[a][c];
        Kpoint(a, c) = e.Ke[a][c] - Ssp::element_term_entry(params(), k.view(), a, c);
      }
    const Eigen::VectorXd ev = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(K).eigenvalues();
    const Eigen::VectorXd evp = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Kpoint).eigenvalues();
    const double top = ev[NDOF - 1];
    int zero = 0, zero_point = 0;
    for (int i = 0; i < NDOF; ++i) {
      EXPECT_GT(ev[i], -1e-12 * top);
      if (ev[i] < 1e-10 * top) ++zero;
      if (evp[i] < 1e-10 * top) ++zero_point;
    }
    EXPECT_EQ(zero, 6);         // three translations, three rotations
    EXPECT_EQ(zero_point, 18);  // one point alone: rank 6, twelve hourglass modes on top
  }
}

TEST(SSPbrick, ElementTermFunctionsAgreeWithItsEntries) {
  const Coords x = distorted();
  const auto k = constants(params(), x);
  refdrv::Lcg rng(9);
  double v[NDOF];
  for (double& a : v) a = rng.symmetric();
  const auto y = kstab_times(params(), k, v);
  double scale = 0;
  std::vector<double> want(NDOF, 0.0);
  for (int a = 0; a < NDOF; ++a) {
    for (int b = 0; b < NDOF; ++b) want[a] += Ssp::element_term_entry(params(), k.view(), a, b) * v[b];
    scale = std::max(scale, std::abs(want[a]));
  }
  for (int a = 0; a < NDOF; ++a) EXPECT_NEAR(y[a], want[a], 1e-13 * scale);
  // Exactly symmetric (interior is stored as its upper triangle).
  for (int a = 0; a < NDOF; ++a)
    for (int b = 0; b < NDOF; ++b)
      EXPECT_EQ(Ssp::element_term_entry(params(), k.view(), a, b), Ssp::element_term_entry(params(), k.view(), b, a));
}

TEST(SSPbrick, ForceIsTangentTimesDisplacementPlusBodyForce) {
  const Coords x = distorted();
  refdrv::Lcg rng(3);
  double u[NDOF];
  for (double& a : u) a = 1e-3 * rng.symmetric();
  const auto z = zeros();
  const Eval e = eval(params(), x, u, z.data());
  const Eval e0 = eval(params(), x, z.data(), z.data());  // body force alone
  double scale = 0;
  std::vector<double> want(NDOF);
  for (int a = 0; a < NDOF; ++a) {
    double s = e0.fe[a];
    for (int c = 0; c < NDOF; ++c) s += e.Ke[a][c] * u[c];
    want[a] = s;
    scale = std::max(scale, std::abs(s));
  }
  for (int a = 0; a < NDOF; ++a) EXPECT_NEAR(e.fe[a], want[a], 1e-12 * scale);
  // The body force is -b times the lumped mass over the density at each node.
  double m[8];
  Ssp::lumped_mass(params(), xe(x), m);
  for (int n = 0; n < 8; ++n) EXPECT_NEAR(e0.fe[3 * n + 2], 19.9 * m[n] / 2.0285, 1e-14 * 19.9 * m[n]);
}

TEST(SSPbrick, CachedGeometryMatchesRecomputedBitwise) {
  const Coords x = distorted();
  refdrv::Lcg rng(4);
  double u[NDOF], v[NDOF];
  for (double& a : u) a = 1e-3 * rng.symmetric();
  for (double& a : v) a = 1e-2 * rng.symmetric();
  const Rayleigh d{0.7, 2e-3, 5e-4, 1e-4};
  double xl[3][8];
  Ssp::local_coords(xe(x), xl);
  double geom[Ssp::NGEOM];
  Ssp::point_store_geometry(xl, 0, {geom, 1});
  const auto k = constants(params(), x);
  double f1[NDOF], K1[NDOF][NDOF], f2[NDOF], K2[NDOF][NDOF];
  Ssp::compute<true>(params(), d, xe(x), u, v, {}, {}, f1, K1);
  Ssp::compute_cached<true>(params(), d, {geom, 1, Ssp::NGEOM}, k.view(), u, v, {}, {}, f2, K2);
  EXPECT_EQ(std::memcmp(f1, f2, sizeof f1), 0);
  EXPECT_EQ(std::memcmp(K1, K2, sizeof K1), 0);
  // The force-only path gives the same force bits.
  double f3[NDOF];
  Ssp::compute<false>(params(), d, xe(x), u, v, {}, {}, f3, nullptr);
  EXPECT_EQ(std::memcmp(f1, f3, sizeof f1), 0);
}

TEST(SSPbrick, MassIsDiagonalAndExactForParallelepipeds) {
  for (const Coords& x : {box(1.3, 0.8, 1.1), parallelepiped()}) {
    std::vector<double> M(NDOF * NDOF);
    Ssp::mass(params(), xe(x), M.data());
    double xl[3][8];
    Ssp::local_coords(xe(x), xl);
    Ssp::PointShape s;
    Ssp::point_geometry(xl, 0, s);
    double total = 0;
    for (int a = 0; a < NDOF; ++a)
      for (int c = 0; c < NDOF; ++c) {
        if (a != c) EXPECT_EQ(M[a * NDOF + c], 0.0);
        else total += M[a * NDOF + c];
      }
    // Each node carries an eighth of the mass, and the volume is the stdBrick volume.
    const double rhoV = 2.0285 * s.mVol;
    EXPECT_NEAR(total / 3.0, rhoV, 1e-14 * rhoV);
    for (int a = 0; a < NDOF; ++a) EXPECT_NEAR(M[a * NDOF + a], rhoV / 8.0, 1e-14 * rhoV);
    const Brick::Params pb{{377000.0, 0.3, 2.0285}, {0, 0, 0}, 1};
    double mb[8];
    Brick::lumped_mass(pb, xe(x), mb);
    double vb = 0;
    for (double mm : mb) vb += mm;
    EXPECT_NEAR(rhoV, vb, 1e-13 * vb);
  }
  // OpenSees' getMass formula (density * J[0] * (1 + ...)) does not conserve the mass of a
  // distorted element; ported as is (docs/deviations.md, "SSPbrick"). Record the size.
  const Coords x = distorted(0.15);
  double m[8];
  Ssp::lumped_mass(params(), xe(x), m);
  double xl[3][8];
  Ssp::local_coords(xe(x), xl);
  Ssp::PointShape s;
  Ssp::point_geometry(xl, 0, s);
  double total = 0;
  for (double mm : m) total += mm;
  const double rel = total / (2.0285 * s.mVol) - 1.0;
  std::printf("[ SSPbrick mass of a distorted element: %.3e relative to rho * volume ]\n", rel);
  EXPECT_LT(std::abs(rel), 2e-2);
}

TEST(SSPbrick, DampingForceIsRayleighMatrixTimesVelocity) {
  const Coords x = distorted();
  refdrv::Lcg rng(11);
  double v[NDOF];
  for (double& a : v) a = 1e-2 * rng.symmetric();
  const auto z = zeros();
  const Ssp::Params p = params(0.0);
  const Rayleigh d{0.7, 2e-3, 5e-4, 1e-4};
  const Eval e = eval(p, x, z.data(), v, d);
  std::vector<double> M(NDOF * NDOF);
  Ssp::mass(p, xe(x), M.data());
  double scale = 0;
  std::vector<double> want(NDOF, 0.0);
  for (int a = 0; a < NDOF; ++a) {
    for (int c = 0; c < NDOF; ++c)
      want[a] += (d.alphaM * M[a * NDOF + c] + (d.betaK + d.betaKinit + d.betaKcomm) * e.Ke[a][c]) * v[c];
    scale = std::max(scale, std::abs(want[a]));
  }
  for (int a = 0; a < NDOF; ++a) EXPECT_NEAR(e.fe[a], want[a], 1e-13 * scale) << "dof " << a;
}

TEST(SSPbrick, MaxOmegaSquaredIsTheLargestEigenvalue) {
  for (const Coords& x : {box(), distorted()}) {
    const Ssp::Params p = params();
    const double w2 = Ssp::max_omega_sq(p, xe(x), {});
    const auto z = zeros();
    const Eval e = eval(p, x, z.data(), z.data());
    double m[8];
    Ssp::lumped_mass(p, xe(x), m);
    Eigen::MatrixXd A(NDOF, NDOF);
    for (int a = 0; a < NDOF; ++a)
      for (int c = 0; c < NDOF; ++c) A(a, c) = e.Ke[a][c] / std::sqrt(m[a / 3] * m[c / 3]);
    const double want = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(A).eigenvalues().maxCoeff();
    EXPECT_NEAR(w2, want, 1e-10 * want);
  }
}

TEST(SSPbrick, CriticalStepShrinksWithStiffnessDampingAndElementSize) {
  const Ssp::Params p = params();
  const double dt = Ssp::critical_dt(p, {}, ExplicitScheme::ExplicitDifference, xe(box()), {});
  const double dt_damped = Ssp::critical_dt(p, {0.0, 1e-3, 0.0, 0.0}, ExplicitScheme::ExplicitDifference, xe(box()), {});
  const double dt_half = Ssp::critical_dt(p, {}, ExplicitScheme::ExplicitDifference, xe(box(0.5, 0.5, 0.5)), {});
  EXPECT_LT(dt_damped, dt);
  EXPECT_NEAR(dt_half, 0.5 * dt, 1e-12 * dt);
  const double vp = std::sqrt(377000.0 * 0.7 / (1.3 * 0.4) / 2.0285);
  EXPECT_GT(dt, 0.2 / vp);
  EXPECT_LT(dt, 1.0 / vp);
  // One integration point and a lumped mass: no smaller than stdBrick's lumped step.
  const Brick::Params pb{{377000.0, 0.3, 2.0285}, {0, 0, -19.9}, 1};
  const double dtb = Brick::critical_dt(pb, {}, ExplicitScheme::ExplicitDifference, xe(box()), {});
  std::printf("[ unit cube, ExplicitDifference: SSPbrick dt_crit %.6e, stdBrick %.6e ]\n", dt, dtb);
  EXPECT_GE(dt, 0.99 * dtb);
}

}  // namespace
