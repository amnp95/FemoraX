#pragma once

// Uniform damping: the filter fits, on the host in the compile step (docs/design.md,
// "Frequency-independent damping"). The per-step filter and the stiffness multiplier are
// physics (physics/damping/uniform_damping.hpp); the fits use std::complex and large
// work arrays, which the physics layer does not allow.
//
// - uniform_damping_fit: OpenSees' UniformDamping::Initialize (fit = "opensees").
// - uniform_damping_mode: the damping and frequency of a single-mode oscillation through the
//   filters (the continuous limit of the step-by-step filter).
// - uniform_damping_fit_femorax: FemoraX's fit (fit = "femorax", the default): the
//   oscillation's damping equals the target over the band.

#include <cmath>
#include <complex>

#include "physics/damping/uniform_damping.hpp"

namespace femorax {

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

namespace uniform_damping_detail {

// UniformDamping::Initialize for a given number of filters: log-spaced omegac from freq1 to
// freq2 and the weights alpha of X alpha = y. False if the system is singular.
inline bool opensees_weights(double freq1, double freq2, int nFilter, UniformDampingFilters& out) {
  constexpr int M = kUniformDampingMaxFilters;
  const double f1log = std::log10(freq1);
  const double f2log = std::log10(freq2);
  const double pi2 = 6.28318530718;  // as OpenSees
  const double w1 = pi2 * freq1;
  const double w2 = pi2 * freq2;
  const double dfreq = (f2log - f1log) / (nFilter - 1);
  double psi[M] = {}, y[M] = {}, alpha[M] = {};
  double X[M][M] = {};
  for (int i = 0; i < nFilter; ++i) {
    const double wc = pi2 * std::pow(10.0, f1log + i * dfreq);
    out.omegac[i] = wc;
    psi[i] = wc * std::atan((wc * (w2 - w1)) / (wc * wc + w1 * w2));
    y[i] = wc * std::log((wc * wc + w2 * w2) / (wc * wc + w1 * w1));
    X[i][i] = 2.0 * psi[i] - ((2.0 * wc * wc * w2) / (wc * wc + w2 * w2) - (2.0 * wc * wc * w1) / (wc * wc + w1 * w1));
  }
  for (int i = 0; i < nFilter; ++i) {
    const double wi = out.omegac[i];
    for (int j = i + 1; j < nFilter; ++j) {
      const double wj = out.omegac[j];
      const double val = (4.0 * wi * wj * (psi[i] - psi[j])) / (wi * wi - wj * wj);
      X[i][j] = val;
      X[j][i] = val;
    }
  }
  if (!solve(nFilter, X, y, alpha)) return false;
  out.nFilter = nFilter;
  for (int i = 0; i < nFilter; ++i) out.alpha[i] = alpha[i];
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
  const double f1log = std::log10(freq1);
  const double f2log = std::log10(freq2);
  const double pi2 = 6.28318530718;
  for (int nFilter = 2; nFilter <= kUniformDampingMaxFilters; ++nFilter) {
    UniformDampingFilters fl;
    if (!uniform_damping_detail::opensees_weights(freq1, freq2, nFilter, fl)) return false;
    bool converged = true;
    const int nf = 100 * nFilter;
    const double df = (f2log - f1log) / (nf - 1);
    for (int i = 0; i < nf && converged; ++i) {
      const double omega = pi2 * std::pow(10.0, f1log + i * df);
      double err = 0.0;
      for (int j = 0; j < nFilter; ++j) {
        const double wjn = omega / fl.omegac[j];
        const double phij = 2.0 * wjn / (1.0 + wjn * wjn);
        err += fl.alpha[j] * phij;
      }
      err -= 1.0;
      if (err > delta || err < -delta) converged = false;
    }
    if (converged) {
      out = fl;
      return true;
    }
  }
  return false;
}

// The free vibration of a mass on a spring whose force passes through these filters: the
// root s near i omega0 of m s^2 + k (1 + sum_i 2 eta alpha_i s / (s + omegac_i)) = 0
// (k / m = omega0^2), the continuous limit of uniform_damping_update (the filter of step 1
// with dt -> 0; the discrete histories agree with it independently of dt). Host code.
struct UniformDampingMode {
  double zeta = 0;          // the damping ratio of the oscillation, -Re(s) / |s|
  double frequency_ratio = 1;  // its frequency over omega0, |s| / omega0 (the stiffening)
};
inline UniformDampingMode uniform_damping_mode(const UniformDampingFilters& fl, double eta, double omega0) {
  using C = std::complex<double>;
  C s(0.0, omega0);
  for (int it = 0; it < 100; ++it) {
    C g = s * s, dg = 2.0 * s, sum = 0.0, dsum = 0.0;
    for (int i = 0; i < fl.nFilter; ++i) {
      const double b = 2.0 * eta * fl.alpha[i];
      sum += b * s / (s + fl.omegac[i]);
      dsum += b * fl.omegac[i] / ((s + fl.omegac[i]) * (s + fl.omegac[i]));
    }
    g += omega0 * omega0 * (1.0 + sum);
    dg += omega0 * omega0 * dsum;
    const C ds = g / dg;
    s -= ds;
    if (std::abs(ds) <= 1e-15 * std::abs(s)) break;
  }
  return {-s.real() / std::abs(s), std::abs(s) / omega0};
}

// FemoraX's fit (fit = "femorax", docs/design.md): weights for which the oscillation's damping
// ratio (uniform_damping_mode, with eta = uniform_damping_eta(zeta)) is zeta within delta
// (relative) over [freq1, freq2]: Gauss-Newton on the weights of OpenSees' filters (the same
// log-spaced omegac), from OpenSees' weights; one more filter when that is not enough. Host
// code. False if no fit within kUniformDampingMaxFilters.
inline bool uniform_damping_fit_femorax(double zeta, double freq1, double freq2, double delta,
                                        UniformDampingFilters& out) {
  constexpr int M = kUniformDampingMaxFilters;
  const double eta = uniform_damping_eta(zeta);
  const double pi2 = 6.28318530718;
  const double f1log = std::log10(freq1), f2log = std::log10(freq2);
  UniformDampingFilters start;
  if (!uniform_damping_fit(freq1, freq2, delta, start)) return false;
  for (int nFilter = start.nFilter; nFilter <= M; ++nFilter) {
    UniformDampingFilters fl;
    if (!uniform_damping_detail::opensees_weights(freq1, freq2, nFilter, fl)) return false;
    const int K = 40 * nFilter;  // sample frequencies
    auto residuals = [&](const UniformDampingFilters& f, double* r) {
      double worst = 0.0;
      for (int k = 0; k < K; ++k) {
        const double w0 = pi2 * std::pow(10.0, f1log + (f2log - f1log) * k / (K - 1));
        r[k] = uniform_damping_mode(f, eta, w0).zeta / zeta - 1.0;
        worst = std::fabs(r[k]) > worst ? std::fabs(r[k]) : worst;
      }
      return worst;
    };
    double r[40 * M], J[40 * M][M];
    for (int iter = 0; iter < 50; ++iter) {
      const double worst = residuals(fl, r);
      if (worst <= delta) {
        out = fl;
        return true;
      }
      for (int j = 0; j < nFilter; ++j) {
        UniformDampingFilters g = fl;
        const double h = 1e-6 * (std::fabs(fl.alpha[j]) + 1e-3);
        g.alpha[j] += h;
        double rj[40 * M];
        residuals(g, rj);
        for (int k = 0; k < K; ++k) J[k][j] = (rj[k] - r[k]) / h;
      }
      double A[M][M] = {}, b[M] = {}, step[M] = {};
      for (int i = 0; i < nFilter; ++i) {
        for (int j = 0; j < nFilter; ++j) {
          double v = 0.0;
          for (int k = 0; k < K; ++k) v += J[k][i] * J[k][j];
          A[i][j] = v + (i == j ? 1e-12 : 0.0);
        }
        double v = 0.0;
        for (int k = 0; k < K; ++k) v += J[k][i] * r[k];
        b[i] = -v;
      }
      if (!uniform_damping_detail::solve(nFilter, A, b, step)) break;
      for (int j = 0; j < nFilter; ++j) fl.alpha[j] += step[j];
    }
  }
  return false;
}

}  // namespace femorax
