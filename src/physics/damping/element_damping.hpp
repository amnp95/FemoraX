#pragma once

// Uniform damping inside solid elements (docs/design.md, "Frequency-independent damping",
// step 2b): the damping quantities of a solid element are the stresses at its integration
// points (OpenSees' Brick::formResidAndTangent: theDamping[i]->update(stress)). Each point
// filters its stress (physics/damping/uniform_damping.hpp) and the damping stress qd joins
// the element force through the point's B^T, the same path as the stress itself.
//
// Rigid motion has no strain, hence no stress change and no damping force; a yielding
// material's stress stops growing, and so does its damping. Element terms of their own
// (SSPbrick's hourglass stabilization) are not filtered: they carry no material stress.

#include "physics/contracts.hpp"
#include "physics/damping/uniform_damping.hpp"
#include "physics/macros.hpp"

namespace femorax {

// A solid element whose damping quantities are its point stresses: every pointwise element
// with a stress vector of its material's size.
template <class E>
concept UniformDampedSolid = MatrixFreeElement<E> && requires { E::NSTRESS; } &&
                             (E::NSTRESS == E::Material::NSTRAIN) && (E::NDOF == 3 * E::NNODE);

// The damping quantities per integration point and their number per element.
template <class E>
  requires UniformDampedSolid<E>
inline constexpr int uniform_damping_quantities_per_point = E::NSTRESS;
template <class E>
  requires UniformDampedSolid<E>
inline constexpr int uniform_damping_quantities = E::NGP * E::NSTRESS;

// The element's damping force from the damping stresses qd of its points:
// fe[3 j + c] = sum_gp (B_j^T (qd_gp dvol))_c. geometry.shape(gp, s) as in the element's
// compute (recomputed or cached).
template <class E, class Geometry>
  requires UniformDampedSolid<E>
FEMORAX_HD void uniform_damping_element_force(const Geometry& geometry, const double (*qd)[E::NSTRESS],
                                              double fe[E::NDOF]) {
  for (int a = 0; a < E::NDOF; ++a) fe[a] = 0.0;
  for (int gp = 0; gp < E::NGP; ++gp) {
    typename E::PointShape s;
    geometry.shape(gp, s);
    for (int j = 0; j < E::NNODE; ++j) {
      double fj[3];
      E::point_node_force(s, qd[gp], j, fj);
      for (int c = 0; c < 3; ++c) fe[3 * j + c] += fj[c];
    }
  }
}

}  // namespace femorax
