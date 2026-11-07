#pragma once

// Chebyshev-Jacobi preconditioner (docs/design.md, "Preconditioners"): z = p(D^-1 A) D^-1 r,
// the degree-k Chebyshev polynomial that best damps the Jacobi-scaled operator's spectrum on
// [lambda_max / ratio, lambda_max]. Needs only apply() and the diagonal, so it is
// matrix-free, and it is a fixed linear operator: symmetric positive definite for a
// symmetric positive definite A (CG may use it), and also usable inside GMRES.
//
// lambda_max of D^-1 A is estimated at setup by power iteration (a fixed start vector and
// a fixed number of iterations, so the estimate is deterministic), times a safety factor:
// an underestimate would make the polynomial negative above it.
//
// Each apply costs degree - 1 operator applications and no global reductions: it trades the
// dot products of extra CG iterations (one global reduction each, the scaling limit across
// GPUs) for local work.

#include <memory>

#include "solvers/linear_solver.hpp"
#include "solvers/vector_ops.hpp"

namespace femorax::solvers {

struct ChebyshevOptions {
  int degree = 4;                 // polynomial degree (operator applications per apply + 1)
  double ratio = 30.0;            // lambda_min = lambda_max / ratio
  int power_iterations = 20;      // for the lambda_max estimate
  double safety = 1.1;            // lambda_max = safety * estimate
};

class Chebyshev final : public Preconditioner {
 public:
  Chebyshev(const ChebyshevOptions& options, int lanes);
  std::string name() const override { return "chebyshev"; }
  void setup(const LinearOperator& a) override;
  void apply(const ConstVector& r, const Vector& z) const override;
  double lambda_max() const { return lmax_; }

 private:
  ChebyshevOptions o_;
  int lanes_;
  const LinearOperator* a_ = nullptr;
  Vector inv_, d_, res_, w_;
  std::unique_ptr<VectorOps> ops_;
  double lmax_ = 0.0, lmin_ = 0.0;
};

}  // namespace femorax::solvers
