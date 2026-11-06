#pragma once

// Runs a model's process (patterns, recorders, set_time, explicit analysis stages) on the
// device.
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
// start of each stage (stops unless allow_unstable_dt); NaN/Inf in the displacements every
// nan_check_interval steps (and at the end of each stage).

#include <cstdint>
#include <string>
#include <vector>

#include "compile/renumber.hpp"
#include "io/model.hpp"

namespace femorax::exec {

struct RunOptions {
  compile::Renumbering renumbering = compile::Renumbering::RCM;
  int nan_check_interval = 100;
  bool verbose = true;
};

struct StageReport {
  std::string name;
  std::string integrator;
  std::int64_t steps = 0;
  double dt = 0, critical_dt = 0;
  double seconds = 0;  // wall time of the time loop (device work included)
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

}  // namespace femorax::exec
