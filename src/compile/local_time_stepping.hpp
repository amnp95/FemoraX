#pragma once

// Stabilized leapfrog local time stepping (LF-LTS), host reference implementation.
//
// Grote, Michel & Sauter, "Stabilized leapfrog based local time-stepping method for the wave
// equation", Math. Comp. 90 (2021), doi:10.1090/mcom/3650, arXiv:2005.13350, section 3.3 (the
// algorithm, read from the arXiv source); with nu = 0 it is the original scheme of Diaz &
// Grote, SIAM J. Sci. Comput. 31 (2009), doi:10.1137/070709414. docs/design.md, "Local time
// stepping"; the 1D reproduction: docs/research/time_step_remedies/prototype/.
//
// The semi-discrete problem is u'' + A u = 0 with A = M^-1 K (lumped M). The DOFs are split by
// a 0/1 projection: fine (Pi_f) and coarse (Pi_c = I - Pi_f). One coarse step of size dt:
//   1. w = A Pi_c u (the coarse DOFs' contribution, once per coarse step);
//   2. z_1 = z_0 - 1/2 (dt/p)^2 (2 p^2 / (omega delta)) (w + A Pi_f z_0), z_0 = u;
//   3. z_{k+1} = (1 + beta_k) z_k - beta_k z_{k-1} - (dt/p)^2 (2 p^2 / omega) beta_{k+1/2}
//                (w + A Pi_f z_k),  k = 1 .. p-1;
//   4. u_next = -u_prev + 2 z_p,
// with delta = 1 + nu / p^2, omega = 2 T_p'(delta) / T_p(delta),
// beta_k = T_{k-1}(delta) / T_{k+1}(delta), beta_{k+1/2} = T_k(delta) / T_{k+1}(delta)
// (T_k: Chebyshev polynomials of the first kind). With nu = 0: delta = 1, omega = 2 p^2,
// beta = 1. Away from the fine region it is leapfrog with step dt. The scheme is equivalent to
// u_next = 2 u - u_prev - dt^2 A^{p,nu} u with A^{p,nu} = A P^dt_{p,nu}(Pi_f A), symmetric in
// the M inner product; stable when every eigenvalue of dt^2 A^{p,nu} lies in (0, 4); it
// conserves E = 1/2 (|(u_{n+1} - u_n) / dt|_M^2 + u_n^T M A^{p,nu} u_{n+1}).
//
// Host code (no Kokkos): the coefficients for the launch layer, and the reference step the
// device implementation is tested against.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace femorax::compile {

// Chebyshev polynomial of the first kind T_k(x), by the three-term recurrence.
inline double chebyshev_t(int k, double x) {
  if (k == 0) return 1.0;
  double t0 = 1.0, t1 = x;
  for (int j = 1; j < k; ++j) {
    const double t2 = 2.0 * x * t1 - t0;
    t0 = t1;
    t1 = t2;
  }
  return t1;
}

// T_k'(x) = k U_{k-1}(x), U the Chebyshev polynomials of the second kind.
inline double chebyshev_t_derivative(int k, double x) {
  if (k == 0) return 0.0;
  double u0 = 1.0, u1 = 2.0 * x;  // U_0, U_1
  if (k == 1) return 1.0;
  for (int j = 1; j < k - 1; ++j) {
    const double u2 = 2.0 * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  return k * u1;  // k U_{k-1}
}

struct LtsCoefficients {
  int p = 1;           // coarse-to-fine step ratio
  double nu = 0.0;     // stabilization, 0 <= nu <= 1/2 (0: Diaz & Grote's scheme)
  double delta = 1.0;  // 1 + nu / p^2
  double omega = 2.0;  // 2 T_p'(delta) / T_p(delta)
  double c_first = 1.0;              // 2 p^2 / (omega delta), step 2's factor
  double c_rest = 1.0;               // 2 p^2 / omega, step 3's factor
  std::vector<double> beta;          // beta_k, k = 1 .. p-1 (index k - 1)
  std::vector<double> beta_half;     // beta_{k+1/2}, k = 1 .. p-1 (index k - 1)
};

inline LtsCoefficients lts_coefficients(int p, double nu) {
  if (p < 1) throw std::invalid_argument("local time stepping: the step ratio p must be at least 1");
  if (!(nu >= 0.0 && nu <= 0.5)) throw std::invalid_argument("local time stepping: nu must be in [0, 1/2]");
  LtsCoefficients c;
  c.p = p;
  c.nu = nu;
  const double pp = static_cast<double>(p) * static_cast<double>(p);
  c.delta = 1.0 + nu / pp;
  c.omega = 2.0 * chebyshev_t_derivative(p, c.delta) / chebyshev_t(p, c.delta);
  c.c_first = 2.0 * pp / (c.omega * c.delta);
  c.c_rest = 2.0 * pp / c.omega;
  for (int k = 1; k < p; ++k) {
    const double tk1 = chebyshev_t(k + 1, c.delta);
    c.beta.push_back(chebyshev_t(k - 1, c.delta) / tk1);
    c.beta_half.push_back(chebyshev_t(k, c.delta) / tk1);
  }
  return c;
}

// One coarse step u_next from u_prev and u. apply(x, y) sets y = A x (A = M^-1 K) for vectors
// of size n; fine[i] = 1 marks the fine DOFs (Pi_f). Host reference: plain loops, the
// operator applied to full-length vectors.
template <class ApplyA>
void lts_step(const LtsCoefficients& c, double dt, const std::vector<std::uint8_t>& fine, ApplyA&& apply,
              const std::vector<double>& u_prev, const std::vector<double>& u, std::vector<double>& u_next) {
  const std::size_t n = u.size();
  std::vector<double> masked(n), w(n), af(n);
  for (std::size_t i = 0; i < n; ++i) masked[i] = fine[i] ? 0.0 : u[i];
  apply(masked, w);  // w = A Pi_c u
  const double tau2 = (dt / c.p) * (dt / c.p);
  std::vector<double> z_prev = u, z = u;
  for (std::size_t i = 0; i < n; ++i) masked[i] = fine[i] ? z[i] : 0.0;
  apply(masked, af);  // A Pi_f z_0
  for (std::size_t i = 0; i < n; ++i) z[i] = z_prev[i] - 0.5 * tau2 * c.c_first * (w[i] + af[i]);
  for (int k = 1; k < c.p; ++k) {
    const double bk = c.beta[static_cast<std::size_t>(k - 1)];
    const double bh = c.beta_half[static_cast<std::size_t>(k - 1)];
    for (std::size_t i = 0; i < n; ++i) masked[i] = fine[i] ? z[i] : 0.0;
    apply(masked, af);  // A Pi_f z_k
    for (std::size_t i = 0; i < n; ++i) {
      const double next = (1.0 + bk) * z[i] - bk * z_prev[i] - tau2 * c.c_rest * bh * (w[i] + af[i]);
      z_prev[i] = z[i];
      z[i] = next;
    }
  }
  u_next.resize(n);
  for (std::size_t i = 0; i < n; ++i) u_next[i] = -u_prev[i] + 2.0 * z[i];
}

}  // namespace femorax::compile
