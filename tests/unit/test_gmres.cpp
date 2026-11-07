// GMRES (FGMRES(m), CGS2) and the automatic solver choice: a non-symmetric operator against
// Eigen SparseLU; the same answer as CG on the symmetric soil column; restarts; run-to-run
// bits; the fused block dot products against the documented order; the choice from the
// operator's symmetry and CG's fall-back on non-positive curvature.

#include "solvers/gmres.hpp"

#include <Eigen/SparseLU>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "../common/reference_assembly.hpp"
#include "../common/synthetic_models.hpp"
#include "exec/matrix_free_operator.hpp"
#include "exec/runtime.hpp"
#include "solvers/block_ops.hpp"
#include "solvers/cg.hpp"
#include "solvers/csr_operator.hpp"
#include "solvers/jacobi.hpp"
#include "solvers/krylov.hpp"

namespace {

using femorax::compile::compile_model;
using femorax::compile::CompileOptions;
using femorax::compile::Compiled;
using femorax::exec::InternalForce;
using femorax::exec::MatrixFreeOperator;
using femorax::exec::Vec;
using femorax::solvers::CsrOperator;
using femorax::solvers::Gmres;
using femorax::solvers::Jacobi;
using femorax::solvers::KrylovSolver;
using femorax::solvers::SolveOptions;
using femorax::solvers::Symmetry;

constexpr bool kHostBackend =
    std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;

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

double rel_error(const std::vector<double>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num = std::max(num, std::abs(a[i] - b[i]));
    den = std::max(den, std::abs(b[i]));
  }
  return num / den;
}

struct System {
  femorax::io::Model model;
  Compiled compiled;
  femorax::testing::ReferenceSystem ref;
};

System column() {
  femorax::testing::ColumnSpec s;
  s.nx = s.ny = 3;
  s.nz = 12;
  s.lumped = false;
  System c{femorax::testing::soil_column(s), {}, {}};
  c.compiled = compile_model(c.model, CompileOptions{});
  c.ref = femorax::testing::assemble_reference(c.model, c.compiled);
  return c;
}

const double kDt = 1e-3;
const femorax::OperatorCoefficients kNewmark{1.0, 0.5 / (0.25 * kDt), 1.0 / (0.25 * kDt * kDt)};

std::vector<double> rhs(std::size_t n) {
  std::vector<double> b(n);
  for (std::size_t i = 0; i < n; ++i) b[i] = std::sin(0.23 * static_cast<double>(i)) + 0.5;
  return b;
}

TEST(BlockOps, FusedDotsFollowTheDocumentedOrder) {
  const std::size_t n = 10007;
  const int k = 5;
  femorax::solvers::Block V("V", n, k);
  auto hv = Kokkos::create_mirror_view(V);
  std::vector<std::vector<double>> cols(k, std::vector<double>(n));
  std::vector<double> hw(n);
  for (std::size_t i = 0; i < n; ++i) {
    hw[i] = std::cos(0.01 * static_cast<double>(i));
    for (int c = 0; c < k; ++c) hv(i, c) = cols[c][i] = std::sin(0.3 * static_cast<double>(i * (c + 1)));
  }
  Kokkos::deep_copy(V, hv);
  const Vec w = upload(hw);
  femorax::solvers::BlockOps ops(static_cast<std::int64_t>(n), 8, femorax::simd_width());
  std::vector<double> h;
  ops.dots(V, k, w, h);
  for (int c = 0; c < k; ++c) {
    const double want = femorax::solvers::BlockOps::host_dot(cols[c], hw, femorax::simd_width());
    if (kHostBackend) {
      EXPECT_EQ(h[c], want);
    } else {
      EXPECT_NEAR(h[c], want, 1e-13 * static_cast<double>(n));
    }
  }
}

TEST(Gmres, NonSymmetricOperatorMatchesEigenSparseLU) {
  // An artificial non-symmetric operator: K_eff of the column plus a skew-symmetric part
  // (+0.3 |a_ij| above the diagonal, -0.3 |a_ij| below), assembled on the device.
  const System c = column();
  auto A = c.ref.effective(kNewmark.stiffness, kNewmark.damping, kNewmark.mass);
  const auto& p = c.ref.pattern;
  for (std::int32_t i = 0; i < p.rows; ++i)
    for (std::int64_t k = p.row_offsets[i]; k < p.row_offsets[i + 1]; ++k) {
      const std::int32_t j = p.columns[k];
      if (j == i) continue;
      const double s = 0.3 * std::abs(c.ref.effective(kNewmark.stiffness, kNewmark.damping, kNewmark.mass)[k]);
      A[k] += j > i ? s : -s;
    }
  CsrOperator op(p.row_offsets, p.columns, Symmetry::General);
  auto mv = Kokkos::create_mirror_view(op.values());
  for (std::size_t k = 0; k < A.size(); ++k) mv(k) = A[k];
  Kokkos::deep_copy(op.values(), mv);

  const auto n = static_cast<std::size_t>(p.rows);
  const auto hb = rhs(n);
  const Vec b = upload(hb), x("x", n);
  Jacobi jacobi;
  jacobi.setup(op);
  SolveOptions o;
  o.relative_tolerance = 1e-12;
  KrylovSolver solver(o, femorax::simd_width());
  solver.setup(op, &jacobi);
  EXPECT_EQ(solver.name(), "gmres");  // chosen from the symmetry flag
  const auto r = solver.solve(b, x);
  EXPECT_TRUE(r.converged);
  std::printf("[ GMRES non-symmetric, %zu equations: %d iterations ]\n", n, r.iterations);

  Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
  const auto M = c.ref.to_eigen(A);
  lu.analyzePattern(M);
  lu.factorize(M);
  ASSERT_EQ(lu.info(), Eigen::Success);
  const Eigen::VectorXd want = lu.solve(Eigen::Map<const Eigen::VectorXd>(hb.data(), static_cast<Eigen::Index>(n)));
  EXPECT_LE(rel_error(download(x), std::vector<double>(want.data(), want.data() + want.size())), 1e-9);

  // Run to run: the same bits; and a short restart still converges to the same answer.
  const Vec x2("x2", n);
  solver.solve(b, x2);
  EXPECT_EQ(download(x2), download(x));
  Gmres short_restart(o, femorax::simd_width(), 5);
  short_restart.setup(op, &jacobi);
  const Vec x3("x3", n);
  const auto r3 = short_restart.solve(b, x3);
  EXPECT_TRUE(r3.converged);
  EXPECT_GT(r3.iterations, r.iterations);
  EXPECT_LE(rel_error(download(x3), download(x)), 1e-9);
}

TEST(Gmres, SameAnswerAsCgOnTheSymmetricColumn) {
  const System c = column();
  InternalForce force(c.compiled, femorax::exec::GeometryCache::On);
  const auto n = static_cast<std::size_t>(c.compiled.dofs.num_equations);
  const Vec u("u", n), b = upload(rhs(n)), xc("xc", n), xg("xg", n);
  MatrixFreeOperator a(force, u, kNewmark);
  Jacobi jacobi;
  jacobi.setup(a);
  SolveOptions o;
  o.relative_tolerance = 1e-13;
  femorax::solvers::ConjugateGradient cg(o, femorax::simd_width());
  cg.setup(a, &jacobi);
  Gmres gmres(o, femorax::simd_width());
  gmres.setup(a, &jacobi);
  const auto rc = cg.solve(b, xc);
  const auto rg = gmres.solve(b, xg);
  EXPECT_TRUE(rc.converged);
  EXPECT_TRUE(rg.converged);
  std::printf("[ symmetric column, %zu equations: CG %d, GMRES %d iterations ]\n", n, rc.iterations, rg.iterations);
  EXPECT_LE(rel_error(download(xg), download(xc)), 1e-10);
  KrylovSolver automatic(o, femorax::simd_width());
  automatic.setup(a, &jacobi);
  EXPECT_EQ(automatic.name(), "cg");
}

// A diagonal operator with mixed signs that wrongly declares itself positive definite.
class LyingDiagonal final : public femorax::solvers::LinearOperator {
 public:
  explicit LyingDiagonal(Vec d) : d_(std::move(d)) {
    info_.rows = static_cast<std::int64_t>(d_.extent(0));
    info_.symmetry = Symmetry::SymmetricPositiveDefinite;
  }
  const femorax::solvers::OperatorInfo& info() const override { return info_; }
  void apply(const femorax::solvers::ConstVector& x, const Vec& y) const override {
    femorax::solvers::VectorOps::multiply(d_, x, y);
  }
  void diagonal(const Vec& d) const override { femorax::solvers::VectorOps::copy(d_, d); }

 private:
  Vec d_;
  femorax::solvers::OperatorInfo info_;
};

TEST(KrylovSolver, FallsBackToGmresOnNonPositiveCurvature) {
  const std::size_t n = 50;
  std::vector<double> hd(n), hb(n, 1.0);
  for (std::size_t i = 0; i < n; ++i) hd[i] = (i % 3 == 0 ? -1.0 : 1.0) * (1.0 + 0.1 * static_cast<double>(i));
  const LyingDiagonal a(upload(hd));
  const Vec b = upload(hb), x("x", n);
  SolveOptions o;
  o.relative_tolerance = 1e-12;
  KrylovSolver solver(o, femorax::simd_width());
  solver.setup(a, nullptr);
  EXPECT_EQ(solver.name(), "cg");
  const auto r = solver.solve(b, x);
  EXPECT_TRUE(solver.fell_back());
  EXPECT_TRUE(r.indefinite);
  EXPECT_TRUE(r.converged);
  const auto got = download(x);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(got[i], 1.0 / hd[i], 1e-10);
}

}  // namespace
