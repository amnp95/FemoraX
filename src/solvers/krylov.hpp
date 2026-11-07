#pragma once

// The implicit solver, chosen from the operator (docs/design.md, "Symmetry of the tangent,
// and the solver choice"): CG for symmetric positive definite operators, GMRES otherwise;
// users never choose. If CG meets non-positive curvature (a symmetry declaration was too
// optimistic), the solve is repeated with GMRES from the same initial guess, and the result
// says so (fell_back). GMRES's basis (2m + 2 vectors) is allocated only when GMRES runs: at
// setup for a non-symmetric or indefinite operator, at the first fall-back otherwise (the one
// allocation outside setup, on an event that also warns), so CG stages never pay for it.

#include <memory>
#include <string>

#include "solvers/cg.hpp"
#include "solvers/gmres.hpp"
#include "solvers/linear_solver.hpp"

namespace femorax::solvers {

class KrylovSolver final : public LinearSolver {
 public:
  KrylovSolver(const SolveOptions& options, int lanes, int gmres_restart = 30);
  std::string name() const override { return use_cg_ ? "cg" : "gmres"; }
  void setup(const LinearOperator& a, const Preconditioner* p) override;
  SolveResult solve(const ConstVector& b, const Vector& x) override;
  bool fell_back() const { return fell_back_; }  // the last solve needed GMRES after CG

 private:
  ConjugateGradient cg_;
  Gmres gmres_;
  bool use_cg_ = true, fell_back_ = false, gmres_ready_ = false;
  const LinearOperator* a_ = nullptr;
  const Preconditioner* p_ = nullptr;
  Vector x0_;
};

}  // namespace femorax::solvers
