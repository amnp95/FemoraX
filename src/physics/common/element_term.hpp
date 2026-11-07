#pragma once

// The generic parts of an element-level stiffness term K_el (ElementTermElement in
// physics/contracts.hpp), shared by the elements that have one and by the launch layer's
// kernels, so every code path adds the same values in the same order.

#include "physics/common/strided.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"

namespace femorax {

// K_el's share of the element force at node j, before the point contributions:
//   resid0 = (K_el u)_j                                        (with the internal force)
//   fdamp0 = (betaK + betaKinit + betaKcomm) (K_el v)_j        (with the damping force)
// K_el is part of the current, initial and committed stiffness alike. zu and zv come from
// element_term_reduce of u and v.
template <class E>
FEMORAX_HD inline void element_term_force_node(const typename E::Params& p, const Rayleigh& damp,
                                               Strided<const double> c, const double* zu, const double* zv,
                                               int j, double* resid0, double* fdamp0) {
  E::element_term_node(p, c, zu, j, resid0);
  const double beta = damp.betaK + damp.betaKinit + damp.betaKcomm;
  if (beta != 0.0) {
    double kv[3];
    E::element_term_node(p, c, zv, j, kv);
    for (int q = 0; q < 3; ++q) fdamp0[q] = beta * kv[q];
  } else {
    for (int q = 0; q < 3; ++q) fdamp0[q] = 0.0;
  }
}

// The factor of K_el in K_eff = cK K_t + cC C + cM M: cK + cC (betaK + betaKinit + betaKcomm).
FEMORAX_HD inline double element_term_factor(const Rayleigh& damp, double stiffness, double damping) {
  return stiffness + damping * (damp.betaK + damp.betaKinit + damp.betaKcomm);
}

}  // namespace femorax
