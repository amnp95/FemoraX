#pragma once

// Stable time step of the OpenSees explicit schemes for one mode with Rayleigh damping
// (docs/design.md, "Stable time step"). Elements call this from critical_dt with their
// max_omega_sq.

#include <cmath>

#include "physics/contracts.hpp"
#include "physics/macros.hpp"

namespace femorax {

// omega_sq: largest omega^2 of the element (or model). Returns +inf for omega_sq <= 0
// (a rigid or massless element sets no limit).
FEMORAX_HD inline double stable_dt(ExplicitScheme scheme, const Rayleigh& damp,
                                   double omega_sq) {
  if (!(omega_sq > 0.0)) return INFINITY;
  const double omega = std::sqrt(omega_sq);
  if (scheme == ExplicitScheme::CentralDifference) {
    // Centred, implicit damping: omega*dt <= 2 regardless of damping.
    return 2.0 / omega;
  }
  // ExplicitDifference: damping from the half-step velocity (lagged). For one mode,
  // omega^2*dt^2 + 2*c*dt - 4 <= 0 with c = alphaM + beta*omega^2, where beta sums all
  // stiffness-proportional terms (conservative).
  const double beta = damp.betaK + damp.betaKinit + damp.betaKcomm;
  const double c = damp.alphaM + beta * omega_sq;
  // (sqrt(c^2 + 4w^2) - c) / w^2, written to avoid cancellation when c >> w:
  // = 4 / (sqrt(c^2 + 4w^2) + c)
  return 4.0 / (std::sqrt(c * c + 4.0 * omega_sq) + c);
}

}  // namespace femorax
