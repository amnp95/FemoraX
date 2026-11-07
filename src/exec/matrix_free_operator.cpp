#include "exec/matrix_free_operator.hpp"

namespace femorax::exec {

solvers::Symmetry operator_symmetry(TangentSymmetry tangent, const OperatorCoefficients& k) {
  if (tangent == TangentSymmetry::General) return solvers::Symmetry::General;
  const bool positive = k.stiffness >= 0.0 && k.damping >= 0.0 && k.mass >= 0.0 &&
                        (k.stiffness > 0.0 || k.mass > 0.0);
  if (tangent == TangentSymmetry::SymmetricPositiveDefinite && positive)
    return solvers::Symmetry::SymmetricPositiveDefinite;
  return solvers::Symmetry::Symmetric;
}

MatrixFreeOperator::MatrixFreeOperator(InternalForce& force, const Vec& u, const OperatorCoefficients& k,
                                       const OperatorSettings& settings)
    : force_(force), u_(u), k_(k), settings_(force.resolve(settings)) {
  info_.rows = force.num_equations();
  info_.fields = {{"displacement", 0, info_.rows}};
  info_.symmetry = operator_symmetry(force.tangent_symmetry(), k_);
}

void MatrixFreeOperator::set_coefficients(const OperatorCoefficients& k) {
  k_ = k;
  info_.symmetry = operator_symmetry(force_.tangent_symmetry(), k_);
}

void MatrixFreeOperator::apply(const solvers::ConstVector& x, const solvers::Vector& y) const {
  force_.apply_operator(k_, settings_, u_, x, y);
}

void MatrixFreeOperator::diagonal(const solvers::Vector& d) const {
  force_.operator_diagonal(k_, settings_, u_, d);
}

}  // namespace femorax::exec
