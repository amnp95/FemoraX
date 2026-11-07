// CG with Jacobi on the matrix-free operator against Eigen on the host-assembled reference:
// static gravity (K) and implicit dynamic steps (K + c2 C + c3 M, Newmark at several dt) on
// the soil column; warm start, run-to-run bits, and the indefinite detection.

#include "solvers/cg.hpp"

#include <Eigen/SparseCholesky>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/reference_assembly.hpp"
#include "../common/synthetic_models.hpp"
#include "exec/force_settings.hpp"
#include "exec/matrix_free_operator.hpp"
#include "exec/runtime.hpp"
#include "solvers/jacobi.hpp"

namespace {

using femorax::OperatorCoefficients;
using femorax::compile::compile_model;
using femorax::compile::CompileOptions;
using femorax::compile::Compiled;
using femorax::exec::InternalForce;
using femorax::exec::MatrixFreeOperator;
using femorax::exec::Vec;
using femorax::solvers::ConjugateGradient;
using femorax::solvers::Jacobi;
using femorax::solvers::SolveOptions;

Vec upload(const std::vector<double>& h) {
  Vec d("v", h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}

std::vector<double> download(const Vec& d) {
  auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
  return std::vector<double>(m.data(), m.data() + m.extent(0));
}

double rel_error(const std::vector<double>& a, const Eigen::VectorXd& b) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num = std::max(num, std::abs(a[i] - b[static_cast<Eigen::Index>(i)]));
    den = std::max(den, std::abs(b[static_cast<Eigen::Index>(i)]));
  }
  return num / den;
}

struct Column {
  femorax::io::Model model;
  Compiled compiled;
  femorax::testing::ReferenceSystem ref;
  std::vector<double> body;  // the body-force load in equation space
};

Column make_column(int nxy, int nz) {
  femorax::testing::ColumnSpec s;
  s.nx = s.ny = nxy;
  s.nz = nz;
  s.lumped = false;
  Column c{femorax::testing::soil_column(s), {}, {}, {}};
  c.compiled = compile_model(c.model, CompileOptions{});
  c.ref = femorax::testing::assemble_reference(c.model, c.compiled);
  InternalForce k(c.compiled, femorax::exec::GeometryCache::Off);
  k.fill(0.0, 0.0);
  k.compute(femorax::exec::default_force_settings());
  c.body = download(k.f);
  for (auto& v : c.body) v = -v;
  return c;
}

TEST(ConjugateGradient, StaticGravityMatchesEigen) {
  const Column c = make_column(3, 16);
  InternalForce force(c.compiled, femorax::exec::GeometryCache::On);
  const auto n = c.body.size();
  const Vec u("u", n), x("x", n), b = upload(c.body);
  MatrixFreeOperator a(force, u, {1.0, 0.0, 0.0});
  Jacobi jacobi;
  jacobi.setup(a);
  SolveOptions o;
  o.relative_tolerance = 1e-13;
  ConjugateGradient cg(o, femorax::simd_width());
  cg.setup(a, &jacobi);
  const auto r = cg.solve(b, x);
  EXPECT_TRUE(r.converged);
  EXPECT_FALSE(r.indefinite);
  std::printf("[ CG static gravity 3x3x16, %zu equations: %d iterations ]\n", n, r.iterations);

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(c.ref.to_eigen(c.ref.K));
  Eigen::VectorXd F = Eigen::Map<const Eigen::VectorXd>(c.body.data(), static_cast<Eigen::Index>(n));
  const Eigen::VectorXd want = ldlt.solve(F);
  const auto got = download(x);
  EXPECT_LE(rel_error(got, want), 1e-10);

  // Warm start from the solution: nothing to do.
  const auto again = cg.solve(b, x);
  EXPECT_TRUE(again.converged);
  EXPECT_EQ(again.iterations, 0);

  // Run to run: the same bits.
  const Vec x2("x2", n);
  cg.solve(b, x2);
  EXPECT_EQ(download(x2), got);
}

TEST(ConjugateGradient, ImplicitDynamicStepsMatchEigen) {
  const Column c = make_column(3, 16);
  InternalForce force(c.compiled, femorax::exec::GeometryCache::On);
  const auto n = c.body.size();
  std::vector<double> hb(n);
  for (std::size_t i = 0; i < n; ++i) hb[i] = std::sin(0.17 * static_cast<double>(i)) + 0.1 * c.body[i];
  const Vec u("u", n), b = upload(hb);
  for (double dt : {2.5e-4, 1e-3, 4e-3, 1.6e-2}) {
    SCOPED_TRACE("dt = " + std::to_string(dt));
    // Newmark gamma = 1/2, beta = 1/4: K + gamma/(beta dt) C + 1/(beta dt^2) M.
    const OperatorCoefficients k{1.0, 0.5 / (0.25 * dt), 1.0 / (0.25 * dt * dt)};
    MatrixFreeOperator a(force, u, k);
    EXPECT_EQ(a.info().symmetry, femorax::solvers::Symmetry::SymmetricPositiveDefinite);
    Jacobi jacobi;
    jacobi.setup(a);
    SolveOptions o;
    o.relative_tolerance = 1e-13;
    ConjugateGradient cg(o, femorax::simd_width());
    cg.setup(a, &jacobi);
    const Vec x("x", n);
    const auto r = cg.solve(b, x);
    EXPECT_TRUE(r.converged);
    std::printf("[ CG dynamic step dt %.2g, %zu equations: %d iterations ]\n", dt, n, r.iterations);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(c.ref.to_eigen(c.ref.effective(k.stiffness, k.damping, k.mass)));
    const Eigen::VectorXd want = ldlt.solve(Eigen::Map<const Eigen::VectorXd>(hb.data(), static_cast<Eigen::Index>(n)));
    EXPECT_LE(rel_error(download(x), want), 1e-10);
  }
}

TEST(ConjugateGradient, DetectsAnIndefiniteOperator) {
  const Column c = make_column(2, 4);
  InternalForce force(c.compiled, femorax::exec::GeometryCache::Off);
  const auto n = c.body.size();
  const Vec u("u", n), x("x", n), b = upload(c.body);
  MatrixFreeOperator a(force, u, {-1.0, 0.0, 0.0});  // negative definite
  EXPECT_NE(a.info().symmetry, femorax::solvers::Symmetry::SymmetricPositiveDefinite);
  ConjugateGradient cg(SolveOptions{}, femorax::simd_width());
  cg.setup(a, nullptr);
  const auto r = cg.solve(b, x);
  EXPECT_FALSE(r.converged);
  EXPECT_TRUE(r.indefinite);
}

}  // namespace
