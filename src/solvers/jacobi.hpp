#pragma once

// Jacobi preconditioner: z = r ./ diag(A), with diag(A) from the operator (matrix-free: the
// element-by-element diagonal). A zero diagonal entry is an error (an equation nothing
// stiffens or loads, which the compile step never produces).

#include <cstdint>

#include "solvers/linear_solver.hpp"

namespace femorax::solvers {

class Jacobi final : public Preconditioner {
 public:
  std::string name() const override { return "jacobi"; }
  void setup(const LinearOperator& a) override;
  void apply(const ConstVector& r, const Vector& z) const override;
  const Vector& inverse_diagonal() const { return inv_; }

 private:
  Vector diag_, inv_;
};

}  // namespace femorax::solvers
