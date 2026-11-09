#pragma once

// Runs a model's process (patterns, recorders, set_time, load_const, explicit and implicit
// analysis stages) on the device. Implicit stages: exec/implicit_stage.hpp.
//
// Integrators, ported from OpenSees (commit 955ceb0), on lumped (diagonal) mass:
//   ExplicitDifference  SRC/analysis/integrator/ExplicitDifference.cpp
//     newStep: v(n+1/2) = v(n-1/2) + dt a(n); u(n+1) = u(n) + dt v(n+1/2); t += dt
//     solve:   a(n+1) = R / m, R = -f(u(n+1), v(n+1/2)) - m r ag(t)   (accel set to 0)
//     update:  recorded v = v(n+1/2) + (dt/8) (3 a(n+1) + a(n))
//   CentralDifference   SRC/analysis/integrator/CentralDifference.cpp
//     loads at t(n); "garbage" v = -u(n-1)/(2dt), a = (u(n-1) - 2u(n))/dt^2;
//     u(n+1) = R / (m/dt^2 + c/(2dt)), R = -f(u(n), v) - m a - m r ag(t(n));
//     v = (3u(n+1) - 4u(n) + u(n-1))/(2dt), a = (v - v_prev)/dt; t += dt at commit.
//     Only with lumped mass and mass-proportional damping (io/support.cpp), so the system
//     is diagonal.
// Time accumulates as in OpenSees (t += dt in double, once per step), and a final_time stage
// steps while t < final_time, as Femora's Tcl loop does.
//
// Checks: the device memory estimate before anything is allocated; the stable step at the
// start of each stage and, when a material has history (PressureDependMultiYield stiffens
// with confinement), again every stable_step_check_interval steps (stops unless
// allow_unstable_dt, which warns once with the new limit); NaN/Inf in the displacements
// every nan_check_interval steps (and at the end of each stage).

#include <cstdint>
#include <string>
#include <vector>

#include "compile/batches.hpp"
#include "compile/renumber.hpp"
#include "io/model.hpp"

namespace femorax::exec {

struct RunOptions {
  compile::Renumbering renumbering = compile::Renumbering::RCM;
  int nan_check_interval = 100;
  int stable_step_check_interval = 100;  // 0: only at the start of each stage
  bool verbose = true;
};

struct StageReport {
  std::string name;
  std::string integrator;
  std::int64_t steps = 0;
  double dt = 0, critical_dt = 0;  // critical_dt: at the start of the stage
  double min_critical_dt = 0;       // the smallest stable step of the stage's checks
  double seconds = 0;  // wall time of the time loop (device work included)
  // Implicit stages: Newton iterations and Krylov iterations in total, and the solver.
  std::int64_t newton_iterations = 0, krylov_iterations = 0;
  // Implicit stages, Femora's recovery: retries, steps split into substeps, and commits
  // (steps plus substeps).
  std::int64_t retries = 0, substepped = 0, commits = 0;
  std::string solver;
};

struct RunReport {
  std::string force_settings, force_settings_source;
  double compile_seconds = 0, setup_seconds = 0, total_seconds = 0;
  std::int64_t equations = 0, elements = 0, records = 0;
  std::int64_t device_bytes_estimate = 0, device_bytes_free = -1;
  std::vector<StageReport> stages;
};

// Throws io::ModelError (instability, NaN, memory) and io::UnsupportedComponent.
RunReport run_model(const io::Model& model, const RunOptions& options);

// Device bytes a run of `model` allocates (checked against free memory before allocating):
// the compile step's data, the force object's vectors and gather, the geometry cache and
// element constants (assumed built), the implicit stages' vectors, stored tangents and
// Krylov work vectors (GMRES's basis when a stage's operator may be non-SPD), and the
// recorders. test_memory_estimate checks it against the allocated peak.
std::int64_t estimate_device_bytes(const io::Model& model, const compile::Compiled& c);

}  // namespace femorax::exec
