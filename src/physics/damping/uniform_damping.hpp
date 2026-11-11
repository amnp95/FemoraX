#pragma once

// Frequency-independent ("Uniform") damping: port of OpenSees' UniformDamping
// (SRC/damping/UniformDamping.cpp, OpenSees commit 955ceb0; Yuli Huang). Reference:
// Y. Huang, R. Sturt, M. Willford, "A damping model for nonlinear dynamic analysis providing
// uniform damping over a frequency range", Computers & Structures 212 (2019) 101-109.
//
// A damping definition is a ratio eta over a band [freq1, freq2]: nFilter first-order filters
// at frequencies omegac[i] with weights alpha[i] (uniform_damping_fit). Each element filters
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

namespace uniform_damping_detail {

// Solve X a = y for a symmetric n x n system (n <= kUniformDampingMaxFilters) by Gaussian
// elimination with partial pivoting, on fixed-size storage. False if singular.
inline bool solve(int n, double (&X)[kUniformDampingMaxFilters][kUniformDampingMaxFilters],
                  double (&y)[kUniformDampingMaxFilters], double (&a)[kUniformDampingMaxFilters]) {
  for (int c = 0; c < n; ++c) {
    int p = c;
    for (int r = c + 1; r < n; ++r)
      if (std::fabs(X[r][c]) > std::fabs(X[p][c])) p = r;
    if (X[p][c] == 0.0) return false;
    if (p != c) {
      for (int k = 0; k < n; ++k) {
        const double t = X[c][k];
        X[c][k] = X[p][k];
        X[p][k] = t;
      }
      const double t = y[c];
      y[c] = y[p];
      y[p] = t;
    }
    for (int r = c + 1; r < n; ++r) {
      const double f = X[r][c] / X[c][c];
      for (int k = c; k < n; ++k) X[r][k] -= f * X[c][k];
      y[r] -= f * y[c];
    }
  }
  for (int r = n - 1; r >= 0; --r) {
    double s = y[r];
    for (int k = r + 1; k < n; ++k) s -= X[r][k] * a[k];
    a[r] = s / X[r][r];
  }
  return true;
}

}  // namespace uniform_damping_detail

// UniformDamping::Initialize: the fewest filters (from 2, log-spaced from freq1 to freq2)
// whose weighted response sum_j alpha_j phi_j(omega), phi_j = 2 w / (1 + w^2), w =
// omega / omegac_j, is within delta of 1 at 100 nFilter log-spaced frequencies of the band.
// Host code (called once per definition). False if no fit within kUniformDampingMaxFilters.
// OpenSees solves X alpha = y with its Matrix class; the elimination order here may change
// the last bits of alpha.
inline bool uniform_damping_fit(double freq1, double freq2, double delta, UniformDampingFilters& out) {
  constexpr int M = kUniformDampingMaxFilters;
  const double f1log = std::log10(freq1);
  const double f2log = std::log10(freq2);
  const double pi2 = 6.28318530718;  // as OpenSees
  const double w1 = pi2 * freq1;
  const double w2 = pi2 * freq2;
  for (int nFilter = 2; nFilter <= M; ++nFilter) {
    const double dfreq = (f2log - f1log) / (nFilter - 1);
    double omegac[M] = {}, psi[M] = {}, y[M] = {}, alpha[M] = {};
    double X[M][M] = {};
    for (int i = 0; i < nFilter; ++i) {
      const double wc = pi2 * std::pow(10.0, f1log + i * dfreq);
      omegac[i] = wc;
      psi[i] = wc * std::atan((wc * (w2 - w1)) / (wc * wc + w1 * w2));
      y[i] = wc * std::log((wc * wc + w2 * w2) / (wc * wc + w1 * w1));
      X[i][i] = 2.0 * psi[i] - ((2.0 * wc * wc * w2) / (wc * wc + w2 * w2) - (2.0 * wc * wc * w1) / (wc * wc + w1 * w1));
    }
    for (int i = 0; i < nFilter; ++i) {
      const double wi = omegac[i];
      for (int j = i + 1; j < nFilter; ++j) {
        const double wj = omegac[j];
        const double val = (4.0 * wi * wj * (psi[i] - psi[j])) / (wi * wi - wj * wj);
        X[i][j] = val;
        X[j][i] = val;
      }
    }
    if (!uniform_damping_detail::solve(nFilter, X, y, alpha)) return false;
    bool converged = true;
    const int nf = 100 * nFilter;
    const double df = (f2log - f1log) / (nf - 1);
    for (int i = 0; i < nf && converged; ++i) {
      const double omega = pi2 * std::pow(10.0, f1log + i * df);
      double err = 0.0;
      for (int j = 0; j < nFilter; ++j) {
        const double wjn = omega / omegac[j];
        const double phij = 2.0 * wjn / (1.0 + wjn * wjn);
        err += alpha[j] * phij;
      }
      err -= 1.0;
      if (err > delta || err < -delta) converged = false;
    }
    if (converged) {
      out.nFilter = nFilter;
      for (int i = 0; i < nFilter; ++i) {
        out.omegac[i] = omegac[i];
        out.alpha[i] = alpha[i];
      }
      return true;
    }
  }
  return false;
}

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
