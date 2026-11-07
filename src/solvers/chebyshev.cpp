#include "solvers/chebyshev.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace femorax::solvers {

namespace {

using Index = std::int64_t;

// d = s * d + t * inv .* res   (one kernel)
void chebyshev_direction(const Vector& d, double s, double t, const Vector& inv, const Vector& res) {
  Kokkos::parallel_for(
      "chebyshev_direction", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, d.extent(0)),
      KOKKOS_LAMBDA(const Index i) { d(i) = s * d(i) + t * inv(i) * res(i); });
}

}  // namespace

Chebyshev::Chebyshev(const ChebyshevOptions& options, int lanes) : o_(options), lanes_(lanes) {
  if (o_.degree < 1) throw std::invalid_argument("Chebyshev: degree must be at least 1");
  if (!(o_.ratio > 1.0)) throw std::invalid_argument("Chebyshev: ratio must exceed 1");
}

void Chebyshev::setup(const LinearOperator& a) {
  a_ = &a;
  const auto n = static_cast<std::size_t>(a.info().rows);
  if (inv_.extent(0) != n) {
    inv_ = Vector("chebyshev_inv", n);
    d_ = Vector("chebyshev_d", n);
    res_ = Vector("chebyshev_res", n);
    w_ = Vector("chebyshev_w", n);
    ops_ = std::make_unique<VectorOps>(static_cast<std::int64_t>(n), lanes_);
  }
  a.diagonal(d_);
  VectorOps::reciprocal(d_, inv_);
  // Power iteration on D^-1 A from a fixed start vector.
  const auto x = d_;
  Kokkos::parallel_for(
      "chebyshev_start", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, x.extent(0)),
      KOKKOS_LAMBDA(const Index i) { x(i) = 1.0 + 0.5 * Kokkos::sin(0.7 * static_cast<double>(i)); });
  double lambda = 0.0;
  for (int it = 0; it < o_.power_iterations; ++it) {
    const double nx = ops_->norm2(x);
    VectorOps::axpy(1.0 / nx - 1.0, x, x);  // x /= ||x||
    a.apply(x, res_);
    VectorOps::multiply(inv_, res_, w_);    // w = D^-1 A x
    lambda = ops_->dot(x, w_);              // Rayleigh quotient estimate (x normalized)
    VectorOps::copy(w_, x);
  }
  if (!(lambda > 0.0)) throw std::runtime_error("Chebyshev: non-positive eigenvalue estimate " + std::to_string(lambda));
  lmax_ = o_.safety * lambda;
  lmin_ = lmax_ / o_.ratio;
}

// Chebyshev iteration for A z = r from z = 0 (Saad, Iterative Methods, Alg. 12.1, with the
// Jacobi scaling): theta, delta the interval's centre and half width.
void Chebyshev::apply(const ConstVector& r, const Vector& z) const {
  const double theta = 0.5 * (lmax_ + lmin_), delta = 0.5 * (lmax_ - lmin_);
  const double sigma = theta / delta;
  double rho = 1.0 / sigma;
  // d = D^-1 r / theta; z = d
  VectorOps::multiply(inv_, r, d_);
  VectorOps::axpy(1.0 / theta - 1.0, d_, d_);
  VectorOps::copy(d_, z);
  for (int k = 1; k < o_.degree; ++k) {
    a_->apply(z, res_);
    VectorOps::xpay(r, -1.0, res_);  // res = r - A z
    const double rho_new = 1.0 / (2.0 * sigma - rho);
    chebyshev_direction(d_, rho_new * rho, 2.0 * rho_new / delta, inv_, res_);
    VectorOps::axpy(1.0, d_, z);
    rho = rho_new;
  }
}

}  // namespace femorax::solvers
