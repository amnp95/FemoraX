#pragma once

// Restarted flexible GMRES, FGMRES(m), with right preconditioning, for symmetric indefinite
// and non-symmetric operators (docs/design.md, "Solvers"). Orthogonalization by classical
// Gram-Schmidt with one re-orthogonalization (CGS2): the dot products of one iteration are
// one fused deterministic reduction (BlockOps), two per iteration plus the norm. Flexible:
// the preconditioned vectors Z are kept, so a preconditioner that changes between iterations
// (Chebyshev, multigrid) is allowed. x holds the initial guess. Stops when the residual
// estimate ||r|| <= max(rtol ||b||, atol); after each restart the true residual is
// recomputed. Memory: 2m + 2 vectors.

#include <memory>
#include <vector>

#include "solvers/block_ops.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/vector_ops.hpp"

namespace femorax::solvers {

class Gmres final : public LinearSolver {
 public:
  // restart: m (default 30); lanes: exec::simd_width().
  Gmres(const SolveOptions& options, int lanes, int restart = 30);
  std::string name() const override { return "gmres"; }
  void setup(const LinearOperator& a, const Preconditioner* p) override;
  SolveResult solve(const ConstVector& b, const Vector& x) override;
  int restart() const { return m_; }

 private:
  SolveOptions options_;
  int lanes_, m_;
  const LinearOperator* a_ = nullptr;
  const Preconditioner* p_ = nullptr;
  std::unique_ptr<VectorOps> ops_;
  std::unique_ptr<BlockOps> block_;
  Block V_, Z_;
  Vector w_, r_;
};

}  // namespace femorax::solvers
