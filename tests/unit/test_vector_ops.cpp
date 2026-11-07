// Solver vector kernels: deterministic reductions (same bits for any thread count; on host
// backends the same bits as the documented order computed on the host) and the
// element-wise operations.

#include "solvers/vector_ops.hpp"
#include "solvers/linear_solver.hpp"

#include "exec/runtime.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

namespace {

using femorax::solvers::Vector;
using femorax::solvers::VectorOps;

// Host backends compute like the host reference, bit for bit. A GPU compiler fuses
// multiply-adds (nvcc --fmad=true), so there the reference holds to rounding only.
constexpr bool kHostBackend =
    std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;

// Exact on host backends; within the rounding of one fused multiply-add on GPUs.
void expect_fma_equal(double got, double want, double scale) {
  if (kHostBackend) {
    ASSERT_EQ(got, want);
  } else {
    ASSERT_NEAR(got, want, 2.3e-16 * scale);
  }
}

std::vector<double> random_vector(std::size_t n, unsigned seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(n);
  for (auto& x : v) x = u(g) * std::pow(10.0, static_cast<double>(g() % 7) - 3.0);  // mixed magnitudes
  return v;
}

Vector upload(const std::vector<double>& h) {
  Vector d("v", h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}

std::vector<double> download(const Vector& d) {
  auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
  return std::vector<double>(m.data(), m.data() + m.extent(0));
}

TEST(VectorOps, DotFollowsTheDocumentedOrder) {
  // Sizes around the chunk boundaries and one that needs two reduction levels.
  for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{2048}, std::size_t{2049},
                        std::size_t{100000}, std::size_t{5000003}}) {
    SCOPED_TRACE("n = " + std::to_string(n));
    const auto hx = random_vector(n, 1), hy = random_vector(n, 2);
    const Vector x = upload(hx), y = upload(hy);
    VectorOps ops(static_cast<std::int64_t>(n), femorax::simd_width());
    const double got = ops.dot(x, y);
    const double want = VectorOps::host_dot(hx, hy, ops.lanes());
    if (kHostBackend) {
      EXPECT_EQ(got, want);
    } else {
      double scale = 0.0;
      for (std::size_t i = 0; i < n; ++i) scale += std::abs(hx[i] * hy[i]);
      EXPECT_NEAR(got, want, 1e-15 * scale + 1e-300);
    }
    // Run to run: the same bits.
    EXPECT_EQ(ops.dot(x, y), got);
    EXPECT_EQ(ops.norm2(x), std::sqrt(ops.dot(x, x)));
  }
}

TEST(VectorOps, DotDoesNotDependOnTheThreadCount) {
  // The host order with lanes = 1 is the order of every CPU backend; the whole test binary
  // also runs with 1 and 8 OpenMP threads (unit_all.threads1/8), so a dependence on the
  // thread count would fail DotFollowsTheDocumentedOrder in one of them. Here: the order
  // must differ from a naive sequential sum, or the test above proves nothing.
  const std::size_t n = 100000;
  const auto hx = random_vector(n, 3), hy = random_vector(n, 4);
  double naive = 0.0;
  for (std::size_t i = 0; i < n; ++i) naive += hx[i] * hy[i];
  const double chunked = VectorOps::host_dot(hx, hy, 1);
  EXPECT_NE(naive, chunked);
  EXPECT_NEAR(naive, chunked, 1e-12 * std::abs(naive));
}

TEST(VectorOps, ElementwiseOperations) {
  const std::size_t n = 4099;
  const auto hx = random_vector(n, 5), hy = random_vector(n, 6);
  std::vector<double> hd(n);
  for (std::size_t i = 0; i < n; ++i) hd[i] = 1.0 + std::abs(hx[i]);
  const Vector x = upload(hx), d = upload(hd);

  Vector y = upload(hy);
  VectorOps::axpy(0.5, x, y);
  auto r = download(y);
  for (std::size_t i = 0; i < n; ++i) expect_fma_equal(r[i], hy[i] + 0.5 * hx[i], std::abs(hy[i]) + std::abs(hx[i]));

  y = upload(hy);
  VectorOps::xpay(x, -2.0, y);
  r = download(y);
  for (std::size_t i = 0; i < n; ++i) expect_fma_equal(r[i], hx[i] + -2.0 * hy[i], std::abs(hx[i]) + 2.0 * std::abs(hy[i]));

  VectorOps::multiply(d, x, y);
  r = download(y);
  for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(r[i], hd[i] * hx[i]);

  VectorOps::reciprocal(d, y);
  r = download(y);
  for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(r[i], 1.0 / hd[i]);

  VectorOps::copy(x, y);
  EXPECT_EQ(download(y), hx);
  VectorOps::fill(y, 3.0);
  for (double v : download(y)) ASSERT_EQ(v, 3.0);
}

// The interface compiles for a minimal matrix-free operator and reports what it is.
class DiagonalOperator final : public femorax::solvers::LinearOperator {
 public:
  explicit DiagonalOperator(Vector d) : d_(std::move(d)) {
    info_.rows = static_cast<std::int64_t>(d_.extent(0));
    info_.fields = {{"u", 0, info_.rows}};
  }
  const femorax::solvers::OperatorInfo& info() const override { return info_; }
  void apply(const femorax::solvers::ConstVector& x, const Vector& y) const override { VectorOps::multiply(d_, x, y); }
  void diagonal(const Vector& d) const override { VectorOps::copy(d_, d); }

 private:
  Vector d_;
  femorax::solvers::OperatorInfo info_;
};

TEST(LinearSolverInterface, MatrixFreeOperatorHasNoMatrix) {
  const DiagonalOperator a(upload({2.0, 3.0, 4.0}));
  EXPECT_EQ(a.assembled(), nullptr);
  EXPECT_EQ(a.info().rows, 3);
  EXPECT_EQ(a.info().symmetry, femorax::solvers::Symmetry::SymmetricPositiveDefinite);
  EXPECT_STREQ(femorax::solvers::to_string(a.info().symmetry), "SymmetricPositiveDefinite");
  Vector y("y", 3);
  a.apply(upload({1.0, 1.0, 2.0}), y);
  EXPECT_EQ(download(y), (std::vector<double>{2.0, 3.0, 8.0}));
}

}  // namespace
