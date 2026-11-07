#pragma once

// Preconditioned conjugate gradients for symmetric positive definite operators (docs/design.md,
// "Solvers"): our own, on any LinearOperator (matrix-free in production), with the
// deterministic reductions of VectorOps, so results are bit-identical for any thread count.
// x holds the initial guess (the previous solution in a time loop). Stops when
// ||r|| <= max(rtol ||b||, atol), with r the recursively updated residual. If p^T A p <= 0
// the operator is not positive definite: CG stops with indefinite = true and the caller
// falls back to GMRES.

#include <cstdint>
#include <memory>

#include "solvers/linear_solver.hpp"
#include "solvers/vector_ops.hpp"

namespace femorax::solvers {

class ConjugateGradient final : public LinearSolver {
 public:
  // lanes: exec::simd_width() (the reduction order, VectorOps).
  ConjugateGradient(const SolveOptions& options, int lanes);
  std::string name() const override { return "cg"; }
  void setup(const LinearOperator& a, const Preconditioner* p) override;
  SolveResult solve(const ConstVector& b, const Vector& x) override;
  const SolveOptions& options() const { return options_; }

 private:
  SolveOptions options_;
  int lanes_;
  const LinearOperator* a_ = nullptr;
  const Preconditioner* p_ = nullptr;
  std::unique_ptr<VectorOps> ops_;
  Vector r_, z_, p_dir_, ap_;
};

}  // namespace femorax::solvers
