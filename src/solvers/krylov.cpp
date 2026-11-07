#include "solvers/krylov.hpp"

#include "solvers/vector_ops.hpp"

namespace femorax::solvers {

KrylovSolver::KrylovSolver(const SolveOptions& options, int lanes, int gmres_restart)
    : cg_(options, lanes), gmres_(options, lanes, gmres_restart) {}

void KrylovSolver::setup(const LinearOperator& a, const Preconditioner* p) {
  a_ = &a;
  p_ = p;
  use_cg_ = a.info().symmetry == Symmetry::SymmetricPositiveDefinite;
  gmres_ready_ = false;
  if (use_cg_) {
    cg_.setup(a, p);
    if (static_cast<std::int64_t>(x0_.extent(0)) != a.info().rows)
      x0_ = Vector("krylov_x0", static_cast<std::size_t>(a.info().rows));
  } else {
    gmres_.setup(a, p);
    gmres_ready_ = true;
  }
}

SolveResult KrylovSolver::solve(const ConstVector& b, const Vector& x) {
  fell_back_ = false;
  if (!use_cg_) return gmres_.solve(b, x);
  VectorOps::copy(x, x0_);
  SolveResult r = cg_.solve(b, x);
  if (!r.indefinite) return r;
  fell_back_ = true;
  if (!gmres_ready_) {
    gmres_.setup(*a_, p_);
    gmres_ready_ = true;
  }
  VectorOps::copy(x0_, x);
  SolveResult g = gmres_.solve(b, x);
  g.iterations += r.iterations;
  g.indefinite = true;
  return g;
}

}  // namespace femorax::solvers
