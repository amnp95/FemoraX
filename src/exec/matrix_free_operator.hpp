#pragma once

// The matrix-free implicit operator behind the solver interface (docs/design.md, "The
// matrix-free operator"): K_eff = k.stiffness K_t + k.damping C + k.mass M, applied element by
// element by InternalForce's operator kernels. No global matrix exists anywhere.
//
// Symmetry (what the solver choice reads): the weakest declared tangent symmetry over the
// stage's element-material pairs; SymmetricPositiveDefinite only when, in addition, the
// coefficients are non-negative and the stiffness or mass coefficient is positive.

#include "exec/internal_force.hpp"
#include "solvers/linear_solver.hpp"

namespace femorax::exec {

class MatrixFreeOperator final : public solvers::LinearOperator {
 public:
  // force and u (the displacement whose tangent the operator uses) must outlive this.
  MatrixFreeOperator(InternalForce& force, const Vec& u, const OperatorCoefficients& k,
                     const OperatorSettings& settings = {});

  const solvers::OperatorInfo& info() const override { return info_; }
  void apply(const solvers::ConstVector& x, const solvers::Vector& y) const override;
  void diagonal(const solvers::Vector& d) const override;

  void set_coefficients(const OperatorCoefficients& k);
  const OperatorCoefficients& coefficients() const { return k_; }
  const OperatorSettings& settings() const { return settings_; }

 private:
  InternalForce& force_;
  Vec u_;
  OperatorCoefficients k_;
  OperatorSettings settings_;
  solvers::OperatorInfo info_;
};

solvers::Symmetry operator_symmetry(TangentSymmetry tangent, const OperatorCoefficients& k);

}  // namespace femorax::exec
