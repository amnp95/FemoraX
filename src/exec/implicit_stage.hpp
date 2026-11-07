#pragma once

// Implicit analysis stages (milestone 1b), matrix-free (docs/design.md, "Linear solvers"):
// Newmark (Transient) and LoadControl (Static) with the Linear, Newton and ModifiedNewton
// algorithms and the NormUnbalance, NormDispIncr and EnergyIncr tests. Every linear solve
// runs the automatic Krylov solver (CG or GMRES, from the declared tangent symmetry) with a
// Jacobi preconditioner on the matrix-free operator K_eff = c1 K_t + c2 C + c3 M.
//
// Ported from OpenSees, commit 955ceb0:
//   Newmark.cpp (form D, displ = 1): newStep, update, revertToLastStep; c1 = 1,
//     c2 = gamma/(beta dt), c3 = 1/(beta dt^2); predictor U = U_t,
//     Udot = (1 - gamma/beta) Udot_t + dt (1 - gamma/(2 beta)) Udotdot_t,
//     Udotdot = (1 - 1/(2 beta)) Udotdot_t - Udot_t/(beta dt); update U += dU,
//     Udot += c2 dU, Udotdot += c3 dU. The residual is
//     B = P - f_int(U) - C Udot - M Udotdot - M r ag(t)  (TransientIntegrator::formUnbalance;
//     the mass is the element's own: consistent, or a true lumped mass, docs/deviations.md).
//   LoadControl.cpp: lambda += dlambda (dlambda scaled by numIter / iterations of the last
//     step, clamped to [min, max]); the domain time is lambda. Static residual
//     B = P - f_int(U) (StaticIntegrator: no damping, no inertia); tangent K_t.
//   Linear.cpp, NewtonRaphson.cpp, ModifiedNewton.cpp: Linear solves once per step;
//     Newton: B, test start, then (tangent, solve, update, B, test) until the test passes;
//     ModifiedNewton: the tangent once per step (factor_once: once per stage; initial: the
//     initial tangent).
//   CTestNormUnbalance / CTestNormDispIncr / CTestEnergyIncr: ||B||_2, ||dU||_2,
//     |dU . B_new| / 2 against the tolerance; failure after max iterations. Default test,
//     as OpenSees' analysis without one: NormUnbalance 1e-6, 25 iterations.
//
// Solver settings, automatic unless the model sets them (schema 0.3): relative tolerance
// 1e-12 (Newton-type algorithms accept a solve that stopped at the iteration limit after
// reducing the residual by six orders; the outer test governs); maximum iterations
// min(number of equations, 10000), at least 200; preconditioner Jacobi. A looser automatic
// tolerance for Newton (1e-10) made results differ from OpenSees' direct solves by 4.5e-8
// relative on the soil column, within the Newton test but more than necessary.

#include <cstdint>
#include <memory>
#include <string>

#include "compile/batches.hpp"
#include "exec/ground_motion.hpp"
#include "exec/internal_force.hpp"
#include "exec/recorders.hpp"
#include "io/model.hpp"
#include "solvers/linear_solver.hpp"

namespace femorax::exec {

struct ImplicitStageReport {
  std::int64_t steps = 0;
  std::int64_t newton_iterations = 0;  // total over the stage
  std::int64_t krylov_iterations = 0;  // total over the stage
  std::string solver;                  // "cg" or "gmres"
  std::string preconditioner;          // "jacobi", "chebyshev"
  int max_newton_iterations = 0;       // in one step
  bool fell_back = false;              // a CG solve needed GMRES
};

class ImplicitStage {
 public:
  // u, v, a: the model's committed displacement, velocity and acceleration (equation space,
  // device), updated in place; u is force.u. Allocates its work vectors once.
  ImplicitStage(InternalForce& force, const compile::Compiled& c, Vec u, Vec v, Vec a,
                const ForceSettings& force_settings);
  ~ImplicitStage();
  ImplicitStage(const ImplicitStage&) = delete;
  ImplicitStage& operator=(const ImplicitStage&) = delete;

  // Runs stage `an` from the domain time `time` (updated). Throws io::ModelError when a step
  // does not converge.
  ImplicitStageReport run(const io::AnalysisStep& an, double& time, GroundMotions& motions,
                          Recorders& recorders, bool verbose);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace femorax::exec
