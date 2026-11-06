// stdBrick properties that need no reference data: linear completeness (patch test), rigid
// body modes, symmetry, consistency of force, tangent, mass and damping, and the stable
// step. The comparison with OpenSees Brick is test_std_brick_reference.cpp.

#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"
#include "../reference/drivers/driver_common.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

using femorax::ElasticIsotropic;
using femorax::ExplicitScheme;
using femorax::Rayleigh;
using Brick = femorax::StdBrick<ElasticIsotropic>;
constexpr int NDOF = Brick::NDOF;

static_assert(femorax::Element<Brick>);
static_assert(femorax::ExplicitElement<Brick>);
static_assert(femorax::CompatiblePair<Brick, ElasticIsotropic>);
static_assert(femorax::PointwiseElement<Brick>);

using Coords = std::array<std::array<double, 3>, 8>;

Brick::Params params(int massType = 1, double b3 = -19.9) {
  return Brick::Params{{377000.0, 0.3, 2.0285}, {0.0, 0.0, b3}, massType};
}

// Unit cube in OpenSees node order: bottom face counter-clockwise, then top face.
Coords cube(double a = 1.0, double b = 1.0, double c = 1.0) {
  return {{{0, 0, 0}, {a, 0, 0}, {a, b, 0}, {0, b, 0}, {0, 0, c}, {a, 0, c}, {a, b, c}, {0, b, c}}};
}

Coords distorted() {
  Coords x = cube(1.3, 0.8, 1.1);
  refdrv::Lcg rng(7);
  for (auto& n : x)
    for (auto& v : n) v += 0.12 * rng.symmetric();
  return x;
}

const double (*xe(const Coords& c))[3] { return reinterpret_cast<const double(*)[3]>(c.data()); }

struct Eval {
  double fe[NDOF];
  double Ke[NDOF][NDOF];
};

Eval eval(const Brick::Params& p, const Coords& x, const double* u, const double* v,
          const Rayleigh& d = {}) {
  Eval e{};
  Brick::compute<true>(p, d, xe(x), u, v, {}, {}, e.fe, e.Ke);
  return e;
}

std::vector<double> zeros() { return std::vector<double>(NDOF, 0.0); }

double max_abs(const double* a, int n) {
  double m = 0;
  for (int i = 0; i < n; ++i) m = std::max(m, std::abs(a[i]));
  return m;
}

// ---------------------------------------------------------------------------
TEST(StdBrick, ReproducesLinearFieldsAtEveryGaussPoint) {
  // u = A x + c gives the constant strain of A at every point of a distorted element.
  const Coords x = distorted();
  const double A[3][3] = {{1e-3, 2e-4, -3e-4}, {5e-4, -7e-4, 1e-4}, {-2e-4, 3e-4, 4e-4}};
  double u[NDOF];
  for (int n = 0; n < 8; ++n)
    for (int i = 0; i < 3; ++i)
      u[3 * n + i] = 0.01 * (i + 1) + A[i][0] * x[n][0] + A[i][1] * x[n][1] + A[i][2] * x[n][2];
  const double want[6] = {A[0][0], A[1][1], A[2][2], A[0][1] + A[1][0], A[1][2] + A[2][1],
                          A[2][0] + A[0][2]};
  double xl[3][8];
  Brick::local_coords(xe(x), xl);
  for (int gp = 0; gp < Brick::NGP; ++gp) {
    Brick::PointShape s;
    Brick::point_geometry(xl, gp, s);
    EXPECT_GT(s.dvol, 0.0);
    double sumN = 0;
    for (int j = 0; j < 8; ++j) sumN += s.shp[3][j];
    EXPECT_NEAR(sumN, 1.0, 1e-15);
    double eps[6];
    Brick::point_strain(s, u, eps);
    for (int c = 0; c < 6; ++c) EXPECT_NEAR(eps[c], want[c], 1e-15) << "gp " << gp << " c " << c;
  }
}

TEST(StdBrick, VolumeIsSumOfPointWeights) {
  const Coords x = cube(2.0, 3.0, 0.5);
  double xl[3][8];
  Brick::local_coords(xe(x), xl);
  double vol = 0;
  for (int gp = 0; gp < 8; ++gp) {
    Brick::PointShape s;
    Brick::point_geometry(xl, gp, s);
    vol += s.dvol;
  }
  EXPECT_NEAR(vol, 3.0, 1e-14);
}

TEST(StdBrick, PatchTestOnDistortedEightElementMesh) {
  // 2x2x2 elements around one interior node, all 26 boundary nodes moved by a linear
  // field. Equilibrium at the interior node must recover the same linear field exactly.
  double nodes[27][3];
  refdrv::Lcg rng(42);
  for (int k = 0; k < 3; ++k)
    for (int j = 0; j < 3; ++j)
      for (int i = 0; i < 3; ++i) {
        double* X = nodes[9 * k + 3 * j + i];
        X[0] = i * 1.0;
        X[1] = j * 0.9;
        X[2] = k * 1.2;
        if (i == 1 || j == 1 || k == 1)  // perturb everything but the outer corners
          for (int c = 0; c < 3; ++c) X[c] += 0.15 * rng.symmetric();
      }
  const double A[3][3] = {{1e-3, 2e-4, -3e-4}, {5e-4, -7e-4, 1e-4}, {-2e-4, 3e-4, 4e-4}};
  auto lin = [&](const double* X, int i) {
    return A[i][0] * X[0] + A[i][1] * X[1] + A[i][2] * X[2];
  };
  const int interior = 13;
  const Brick::Params p = params(1, 0.0);  // no body force
  double Kii[3][3] = {}, rhs[3] = {};
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
        Eval e = eval(p, x, u, z.data());
        // f(u) with the interior node at zero: rhs = -f_interior; K_ii from the tangent.
        for (int a = 0; a < 3; ++a) {
          rhs[a] -= e.fe[3 * local + a];
          for (int b = 0; b < 3; ++b) Kii[a][b] += e.Ke[3 * local + a][3 * local + b];
        }
      }
  // Solve the 3x3 system (Cramer's rule).
  auto det3 = [](const double M[3][3]) {
    return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
           M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  };
  const double d = det3(Kii);
  for (int c = 0; c < 3; ++c) {
    double M[3][3];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) M[a][b] = b == c ? rhs[a] : Kii[a][b];
    const double ui = det3(M) / d;
    EXPECT_NEAR(ui, lin(nodes[interior], c), 1e-14) << "component " << c;
  }
}

TEST(StdBrick, TangentIsSymmetricAndHasSixRigidBodyModes) {
  const Coords x = distorted();
  const auto z = zeros();
  const Eval e = eval(params(), x, z.data(), z.data());
  const double scale = max_abs(&e.Ke[0][0], NDOF * NDOF);
  for (int a = 0; a < NDOF; ++a)
    for (int c = 0; c < NDOF; ++c) EXPECT_NEAR(e.Ke[a][c], e.Ke[c][a], 1e-13 * scale);
  // Rigid motions: 3 translations, 3 infinitesimal rotations about the centroid.
  for (int mode = 0; mode < 6; ++mode) {
    double r[NDOF];
    for (int n = 0; n < 8; ++n) {
      const double X = x[n][0], Y = x[n][1], Z = x[n][2];
      double v[3] = {0, 0, 0};
      if (mode < 3) v[mode] = 1.0;
      if (mode == 3) { v[1] = -Z; v[2] = Y; }
      if (mode == 4) { v[0] = Z; v[2] = -X; }
      if (mode == 5) { v[0] = -Y; v[1] = X; }
      for (int c = 0; c < 3; ++c) r[3 * n + c] = v[c];
    }
    for (int a = 0; a < NDOF; ++a) {
      double s = 0;
      for (int c = 0; c < NDOF; ++c) s += e.Ke[a][c] * r[c];
      EXPECT_NEAR(s, 0.0, 1e-12 * scale) << "mode " << mode;
    }
  }
}

TEST(StdBrick, ForceIsTangentTimesDisplacementPlusBodyForce) {
  const Coords x = distorted();
  refdrv::Lcg rng(3);
  double u[NDOF];
  for (double& v : u) v = 1e-3 * rng.symmetric();
  const auto z = zeros();
  const Eval loaded = eval(params(), x, u, z.data());
  const Eval at_rest = eval(params(), x, z.data(), z.data());  // body force only
  const double scale = max_abs(loaded.fe, NDOF);
  for (int a = 0; a < NDOF; ++a) {
    double Ku = 0;
    for (int c = 0; c < NDOF; ++c) Ku += loaded.Ke[a][c] * u[c];
    EXPECT_NEAR(loaded.fe[a], Ku + at_rest.fe[a], 1e-12 * scale);
  }
  // Body force: total load equals b * volume, shared out by the shape functions.
  double fz = 0;
  for (int n = 0; n < 8; ++n) fz += at_rest.fe[3 * n + 2];
  double xl[3][8], vol = 0;
  Brick::local_coords(xe(x), xl);
  for (int gp = 0; gp < 8; ++gp) {
    Brick::PointShape s;
    Brick::point_geometry(xl, gp, s);
    vol += s.dvol;
  }
  EXPECT_NEAR(fz, 19.9 * vol, 1e-12 * 19.9 * vol);  // resid = -b N dvol, b3 = -19.9
}

TEST(StdBrick, ForceOnlyPathMatchesFullPathBitwise) {
  const Coords x = distorted();
  refdrv::Lcg rng(9);
  double u[NDOF], v[NDOF];
  for (double& a : u) a = 1e-3 * rng.symmetric();
  for (double& a : v) a = 1e-2 * rng.symmetric();
  const Rayleigh d{0.8, 1e-3, 2e-4, 3e-4};
  const Eval full = eval(params(0), x, u, v, d);
  double fe[NDOF];
  Brick::compute<false>(params(0), d, xe(x), u, v, {}, {}, fe, nullptr);
  for (int a = 0; a < NDOF; ++a) EXPECT_EQ(fe[a], full.fe[a]);
}

TEST(StdBrick, CachedGeometryMatchesRecomputedBitwise) {
  const Coords x = distorted();
  refdrv::Lcg rng(21);
  double u[NDOF], v[NDOF];
  for (double& a : u) a = 1e-3 * rng.symmetric();
  for (double& a : v) a = 1e-2 * rng.symmetric();
  const Rayleigh d{0.8, 1e-3, 2e-4, 3e-4};
  double xl[3][8];
  Brick::local_coords(xe(x), xl);
  double geom[Brick::NGP * Brick::NGEOM];
  const femorax::PointArray<double> g{geom, Brick::NGP, 1};  // per-point layout, 1 element
  for (int gp = 0; gp < Brick::NGP; ++gp) Brick::point_store_geometry(xl, gp, g.point(gp));
  for (int massType : {0, 1}) {
    double f1[NDOF], f2[NDOF], K1[NDOF][NDOF], K2[NDOF][NDOF];
    Brick::compute<true>(params(massType), d, xe(x), u, v, {}, {}, f1, K1);
    Brick::compute_cached<true>(params(massType), d, g, u, v, {}, {}, f2, K2);
    EXPECT_EQ(std::memcmp(f1, f2, sizeof(f1)), 0);
    EXPECT_EQ(std::memcmp(K1, K2, sizeof(K1)), 0);
  }
  // The stored shape-function values equal shp3d's, bit for bit.
  for (int gp = 0; gp < Brick::NGP; ++gp) {
    Brick::PointShape a, b;
    Brick::point_geometry(xl, gp, a);
    Brick::point_load_geometry(gp, g.point(gp), b);
    EXPECT_EQ(std::memcmp(&a, &b, sizeof(a)), 0) << "gp " << gp;
  }
}

TEST(StdBrick, MassMatricesConserveTotalMass) {
  const Coords x = distorted();
  double xl[3][8], vol = 0;
  Brick::local_coords(xe(x), xl);
  for (int gp = 0; gp < 8; ++gp) {
    Brick::PointShape s;
    Brick::point_geometry(xl, gp, s);
    vol += s.dvol;
  }
  const double total = 2.0285 * vol;
  for (int massType : {0, 1}) {
    std::vector<double> M(NDOF * NDOF);
    Brick::mass(params(massType), xe(x), M.data());
    double sum = 0;
    for (double m : M) sum += m;
    EXPECT_NEAR(sum, 3.0 * total, 1e-13 * total) << "massType " << massType;
    for (int a = 0; a < NDOF; ++a)
      for (int c = 0; c < NDOF; ++c) {
        // Symmetric to rounding: entry (j, k) is (N_j dvol rho) N_k and (k, j) is
        // (N_k dvol rho) N_j, as in Brick::formInertiaTerms.
        EXPECT_NEAR(M[a * NDOF + c], M[c * NDOF + a], 2e-16 * std::abs(M[a * NDOF + c]));
        if (massType == 1 && a != c) {
          EXPECT_EQ(M[a * NDOF + c], 0.0);
        }
      }
  }
  // lumped_mass is the diagonal of the lumped matrix, and the row sums of the consistent one.
  double m[8];
  Brick::lumped_mass(params(0), xe(x), m);
  std::vector<double> Mc(NDOF * NDOF), Ml(NDOF * NDOF);
  Brick::mass(params(0), xe(x), Mc.data());
  Brick::mass(params(1), xe(x), Ml.data());
  for (int n = 0; n < 8; ++n)
    for (int q = 0; q < 3; ++q) {
      const int a = 3 * n + q;
      EXPECT_EQ(m[n], Ml[a * NDOF + a]);
      double row = 0;
      for (int c = 0; c < NDOF; ++c) row += Mc[a * NDOF + c];
      EXPECT_NEAR(row, m[n], 1e-14 * m[n]);
    }
}

TEST(StdBrick, DampingForceIsRayleighMatrixTimesVelocity) {
  const Coords x = distorted();
  refdrv::Lcg rng(11);
  double v[NDOF];
  for (double& a : v) a = 1e-2 * rng.symmetric();
  const auto z = zeros();
  for (int massType : {0, 1}) {
    const Brick::Params p = params(massType, 0.0);  // no body force, u = 0: f = C v only
    const Rayleigh d{0.7, 2e-3, 5e-4, 1e-4};
    const Eval e = eval(p, x, z.data(), v, d);
    std::vector<double> M(NDOF * NDOF);
    Brick::mass(p, xe(x), M.data());
    double scale = 0;
    std::vector<double> want(NDOF, 0.0);
    for (int a = 0; a < NDOF; ++a) {
      for (int c = 0; c < NDOF; ++c)
        want[a] += (d.alphaM * M[a * NDOF + c] +
                    (d.betaK + d.betaKinit + d.betaKcomm) * e.Ke[a][c]) * v[c];
      scale = std::max(scale, std::abs(want[a]));
    }
    for (int a = 0; a < NDOF; ++a)
      EXPECT_NEAR(e.fe[a], want[a], 1e-13 * scale) << "massType " << massType << " dof " << a;
  }
}

TEST(StdBrick, MaxOmegaSquaredIsTheLargestEigenvalue) {
  const Coords x = distorted();
  const Brick::Params p = params();
  const double w2 = Brick::max_omega_sq(p, xe(x), {});
  ASSERT_GT(w2, 0.0);
  // Independent check: power iteration on M^-1/2 K M^-1/2 converges to the same value.
  const auto z = zeros();
  const Eval e = eval(p, x, z.data(), z.data());
  double m[8];
  Brick::lumped_mass(p, xe(x), m);
  double A[NDOF][NDOF];
  for (int a = 0; a < NDOF; ++a)
    for (int c = 0; c < NDOF; ++c) A[a][c] = e.Ke[a][c] / std::sqrt(m[a / 3] * m[c / 3]);
  double y[NDOF];
  refdrv::Lcg rng(5);
  for (double& a : y) a = rng.symmetric();
  double lambda = 0;
  for (int it = 0; it < 5000; ++it) {
    double z2[NDOF], n2 = 0, yy = 0, yz = 0;
    for (int a = 0; a < NDOF; ++a) {
      z2[a] = 0;
      for (int c = 0; c < NDOF; ++c) z2[a] += A[a][c] * y[c];
      yz += y[a] * z2[a];
      yy += y[a] * y[a];
      n2 += z2[a] * z2[a];
    }
    lambda = yz / yy;
    for (int a = 0; a < NDOF; ++a) y[a] = z2[a] / std::sqrt(n2);
  }
  EXPECT_NEAR(w2, lambda, 1e-9 * lambda);
  // Trace check: Jacobi preserves the sum of eigenvalues, and the maximum is below it.
  double trace = 0;
  for (int a = 0; a < NDOF; ++a) trace += A[a][a];
  EXPECT_LT(w2, trace);
}

TEST(StdBrick, CriticalStepShrinksWithStiffnessDampingAndElementSize) {
  const Brick::Params p = params();
  const double dt = Brick::critical_dt(p, {}, ExplicitScheme::ExplicitDifference, xe(cube()), {});
  const double dt_damped = Brick::critical_dt(p, {0.0, 1e-3, 0.0, 0.0},
                                              ExplicitScheme::ExplicitDifference, xe(cube()), {});
  const double dt_half = Brick::critical_dt(p, {}, ExplicitScheme::ExplicitDifference,
                                            xe(cube(0.5, 0.5, 0.5)), {});
  EXPECT_LT(dt_damped, dt);
  EXPECT_NEAR(dt_half, 0.5 * dt, 1e-12 * dt);  // omega scales with 1/h
  // Of the order of h / P-wave speed for this soil (Vp ~ 500 m/s for E = 377 MPa).
  const double vp = std::sqrt(377000.0 * 0.7 / (1.3 * 0.4) / 2.0285);
  EXPECT_GT(dt, 0.2 / vp);
  EXPECT_LT(dt, 1.0 / vp);
}

}  // namespace
