#include "solvers/cg.hpp"

#include <algorithm>
#include <cmath>

namespace femorax::solvers {

ConjugateGradient::ConjugateGradient(const SolveOptions& options, int lanes) : options_(options), lanes_(lanes) {}

void ConjugateGradient::setup(const LinearOperator& a, const Preconditioner* p) {
  a_ = &a;
  p_ = p;
  const auto n = a.info().rows;
  if (static_cast<std::int64_t>(r_.extent(0)) != n) {
    const auto un = static_cast<std::size_t>(n);
    r_ = Vector("cg_r", un);
    z_ = Vector("cg_z", un);
    p_dir_ = Vector("cg_p", un);
    ap_ = Vector("cg_ap", un);
    ops_ = std::make_unique<VectorOps>(n, lanes_);
  }
}

SolveResult ConjugateGradient::solve(const ConstVector& b, const Vector& x) {
  SolveResult res;
  const VectorOps& ops = *ops_;
  res.rhs_norm = ops.norm2(b);
  // r = b - A x
  a_->apply(x, ap_);
  VectorOps::copy(b, r_);
  VectorOps::axpy(-1.0, ap_, r_);
  double rnorm = ops.norm2(r_);
  res.initial_residual = res.final_residual = rnorm;
  const double tol = std::max(options_.relative_tolerance * res.rhs_norm, options_.absolute_tolerance);
  if (rnorm <= tol) {
    res.converged = true;
    return res;
  }
  auto precondition = [&] {
    if (p_) p_->apply(r_, z_);
    else VectorOps::copy(r_, z_);
  };
  precondition();
  VectorOps::copy(z_, p_dir_);
  double rz = ops.dot(r_, z_);
  for (int it = 1; it <= options_.max_iterations; ++it) {
    a_->apply(p_dir_, ap_);
    const double pap = ops.dot(p_dir_, ap_);
    if (!(pap > 0.0)) {
      res.indefinite = true;
      res.iterations = it - 1;
      return res;
    }
    const double alpha = rz / pap;
    VectorOps::axpy(alpha, p_dir_, x);
    VectorOps::axpy(-alpha, ap_, r_);
    rnorm = ops.norm2(r_);
    res.iterations = it;
    res.final_residual = rnorm;
    if (rnorm <= tol) {
      res.converged = true;
      return res;
    }
    precondition();
    const double rz_new = ops.dot(r_, z_);
    const double beta = rz_new / rz;
    rz = rz_new;
    VectorOps::xpay(z_, beta, p_dir_);  // p = z + beta p
  }
  return res;
}

}  // namespace femorax::solvers
