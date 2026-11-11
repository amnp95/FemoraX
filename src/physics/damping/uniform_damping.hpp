#pragma once

// Frequency-independent ("Uniform") damping: port of OpenSees' UniformDamping
// (SRC/damping/UniformDamping.cpp, OpenSees commit 955ceb0; Yuli Huang). Reference:
// Y. Huang, R. Sturt, M. Willford, "A damping model for nonlinear dynamic analysis providing
// uniform damping over a frequency range", Computers & Structures 212 (2019) 101-109.
//
// A damping definition is a ratio eta over a band [freq1, freq2]: nFilter first-order filters
// at frequencies omegac[i] with weights alpha[i], fitted on the host in the compile step
// (compile/uniform_damping_fit.hpp: OpenSees' fit and FemoraX's). Each element filters
// its own damping quantities q (stresses at a Gauss point, a beam's basic forces, ...) and
// adds the damping quantity qd to them (uniform_damping_update); its stiffness is multiplied
// by uniform_damping_stiffness_multiplier. Static stages only track q. Names follow the
// OpenSees source (eta, freq1, freq2, alpha, omegac, qL, qd, q0, cd, c0, cL).
//
// State of one set of NQ damping quantities: NQ * (nFilter + 2) doubles, value k at ref[k]:
//   [0, NQ)                     q0, the damping quantities of the step
//   [NQ, 2 NQ)                  qd, the damping quantities added
//   [2 NQ + i NQ, 2 NQ + (i+1) NQ)  qL of filter i
// in the committed and the trial buffers like material state (docs/design.md, "Frequency-
// independent damping").

#include <cmath>

#include "physics/common/strided.hpp"
#include "physics/macros.hpp"

namespace femorax {

// Most filters a definition may need: OpenSees adds one until the fit is within 5 % of the
// ratio (about 2-3 per decade), so 32 covers ten decades.
inline constexpr int kUniformDampingMaxFilters = 32;

struct UniformDampingFilters {
  int nFilter = 0;
  double omegac[kUniformDampingMaxFilters] = {};
  double alpha[kUniformDampingMaxFilters] = {};
};

// The class works with eta = 2 * the damping ratio (the loss factor): OpenSees' command
// `damping Uniform tag zeta freq1 freq2 ...` constructs UniformDamping(tag, zeta*2.0, ...)
// (SRC/damping/TclDampingCommand.cpp). Every eta below is this value.
FEMORAX_HD inline double uniform_damping_eta(double zeta) { return zeta * 2.0; }

// When the damping acts, as UniformDamping::update and getStiffnessMultiplier decide it.
struct UniformDampingStep {
  bool dynamic = false;  // false in static stages (OpenSees: a StaticAnalysis exists)
  double t = 0.0;        // the trial time
  double dT = 0.0;       // the step size (trial time - committed time)
  double ta = 0.0;       // -activateTime (OpenSees default 0)
  double td = 1e20;      // -deactivateTime (OpenSees default 1e20)
  double fac = 1.0;      // the -fact time series' factor at t (1 without one)
};

// UniformDamping::getStiffnessMultiplier: 1 + sum_i 4 alpha_i eta / (2 + omegac_i dT) while
// the damping acts, else 1.
FEMORAX_HD inline double uniform_damping_stiffness_multiplier(const UniformDampingFilters& fl, double eta,
                                                              const UniformDampingStep& s) {
  double km = 0.0;
  if (s.dynamic && s.dT > 0.0 && s.t > s.ta && s.t < s.td) {
    for (int i = 0; i < fl.nFilter; ++i) km += 4.0 * fl.alpha[i] * eta / (2.0 + fl.omegac[i] * s.dT);
    km *= s.fac;
  }
  return 1.0 + km;
}

// UniformDamping::update for NQ damping quantities q: the trial state from the committed one,
// and the damping quantities qd (also stored in the trial state). The committed state must
// hold the last committed step (all zeros before the first update).
template <int NQ>
FEMORAX_HD void uniform_damping_update(const UniformDampingFilters& fl, double eta, const UniformDampingStep& s,
                                       Strided<const double> committed, Strided<double> trial, const double q[NQ],
                                       double qd[NQ]) {
  const auto q0C = [&](int j) { return committed[j]; };
  const auto qdC = [&](int j) { return committed[NQ + j]; };
  const auto qLC = [&](int j, int i) { return committed[2 * NQ + i * NQ + j]; };
  if (s.dynamic && !(s.dT > 0.0)) {
    // OpenSees changes nothing then: the trial state stays the committed one (after a
    // commit, OpenSees' trial and committed vectors are equal).
    const int n = NQ * (fl.nFilter + 2);
    for (int k = 0; k < n; ++k) trial[k] = committed[k];
    for (int j = 0; j < NQ; ++j) qd[j] = qdC(j);
    return;
  }
  for (int j = 0; j < NQ; ++j) trial[j] = q[j];  // *q0 = q
  for (int j = 0; j < NQ; ++j) qd[j] = 0.0;
  if (!s.dynamic) {
    for (int i = 0; i < fl.nFilter; ++i)
      for (int j = 0; j < NQ; ++j) trial[2 * NQ + i * NQ + j] = q[j];
  } else {
    if (s.t < s.td) {
      if (s.t > s.ta) {
        for (int i = 0; i < fl.nFilter; ++i) {
          const double dTomegac = s.dT * fl.omegac[i];
          const double cd = 4.0 * fl.alpha[i] * eta / (2.0 + dTomegac);
          const double c0 = dTomegac / (2.0 + dTomegac);
          const double cL = (2.0 - dTomegac) / (2.0 + dTomegac);
          for (int j = 0; j < NQ; ++j) {
            qd[j] += cd * (q0C(j) + q[j] - 2.0 * qLC(j, i));
            trial[2 * NQ + i * NQ + j] = c0 * (q0C(j) + q[j]) + cL * qLC(j, i);
          }
        }
        for (int j = 0; j < NQ; ++j) qd[j] -= qdC(j);
      } else {
        for (int i = 0; i < fl.nFilter; ++i)
          for (int j = 0; j < NQ; ++j) trial[2 * NQ + i * NQ + j] = q[j];
      }
      for (int j = 0; j < NQ; ++j) qd[j] *= s.fac;
    } else {
      // OpenSees leaves qL unchanged after the deactivation time (its trial filters keep
      // the committed values).
      for (int i = 0; i < fl.nFilter; ++i)
        for (int j = 0; j < NQ; ++j) trial[2 * NQ + i * NQ + j] = qLC(j, i);
    }
  }
  for (int j = 0; j < NQ; ++j) trial[NQ + j] = qd[j];
}

}  // namespace femorax

