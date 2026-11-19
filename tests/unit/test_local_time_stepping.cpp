// Stabilized leapfrog local time stepping, host reference (compile/local_time_stepping.hpp),
// against the properties Grote, Michel & Sauter prove and report (Math. Comp. 2021,
// arXiv:2005.13350, section 5), on their 1D test: the wave equation on (0, 1), c = 1,
// Dirichlet ends, P1 elements with mass lumping, a coarse mesh on (0, 0.9) with h_c and a fine
// mesh on [0.9, 1) with h_f = h_c / p; the fine DOFs are the nodes of the closed fine region.
//   - the coefficients reduce to Diaz & Grote's for nu = 0, and p = 1 is leapfrog;
//   - stability (every eigenvalue of dt^2 A^{p,nu} in (0, 4)): without stabilization an
//     unstable step near 0.7104 h_c for p = 2, none with nu = 0.01;
//   - the discrete energy is conserved over a long run;
//   - second-order convergence in h_c against d'Alembert's solution, with errors as small
//     as leapfrog's at the same coarse step (the nodal error at a full period, t = 2,
//     superconverges, and the paper's continuous L2 error is dominated by the interpolation
//     error: both hide the time integration, so the test measures at t = 1.33, after the
//     waves crossed and left the fine region, at 0.9 of the stable step, FemoraX's margin).

#include "compile/local_time_stepping.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using femorax::compile::lts_coefficients;
using femorax::compile::lts_step;
using femorax::compile::LtsCoefficients;

// The paper's 1D mesh: interior nodes x, element sizes h (n + 1 elements), lumped masses m,
// fine flags. A x = M^-1 K x for the P1 stiffness with Dirichlet ends.
struct Mesh1D {
  std::vector<double> x, h, m;
  std::vector<std::uint8_t> fine;
  std::size_t size() const { return x.size(); }
  void apply(const std::vector<double>& u, std::vector<double>& y) const {
    const std::size_t n = x.size();
    y.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const double left = i > 0 ? u[i - 1] : 0.0, right = i + 1 < n ? u[i + 1] : 0.0;
      y[i] = ((u[i] - left) / h[i] + (u[i] - right) / h[i + 1]) / m[i];
    }
  }
};

Mesh1D mesh(double hc, int p) {
  const int nc = static_cast<int>(std::lround(0.9 / hc)), nf = static_cast<int>(std::lround(0.1 / (hc / p)));
  std::vector<double> nodes;
  for (int i = 0; i <= nc; ++i) nodes.push_back(0.9 * i / nc);
  for (int i = 1; i <= nf; ++i) nodes.push_back(0.9 + 0.1 * i / nf);
  Mesh1D g;
  for (std::size_t e = 0; e + 1 < nodes.size(); ++e) g.h.push_back(nodes[e + 1] - nodes[e]);
  for (std::size_t i = 1; i + 1 < nodes.size(); ++i) {
    g.x.push_back(nodes[i]);
    g.m.push_back(0.5 * (g.h[i - 1] + g.h[i]));
    g.fine.push_back(nodes[i] >= 0.9 - 1e-12 ? 1 : 0);
  }
  return g;
}

// One coarse step with this mesh's operator.
std::vector<double> step(const LtsCoefficients& c, const Mesh1D& g, double dt, const std::vector<double>& u_prev,
                         const std::vector<double>& u) {
  std::vector<double> next;
  lts_step(c, dt, g.fine, [&](const std::vector<double>& a, std::vector<double>& b) { g.apply(a, b); }, u_prev, u,
           next);
  return next;
}

// dt^2 A^{p,nu}, symmetrized in the M inner product (M^1/2 A^{p,nu} M^-1/2), column by column
// from one step with u_prev = 0: u_next = 2 u - dt^2 A^{p,nu} u.
Eigen::MatrixXd scaled_operator(const LtsCoefficients& c, const Mesh1D& g, double dt) {
  const auto n = static_cast<Eigen::Index>(g.size());
  Eigen::MatrixXd S(n, n);
  std::vector<double> zero(g.size(), 0.0), e(g.size(), 0.0);
  for (Eigen::Index j = 0; j < n; ++j) {
    std::fill(e.begin(), e.end(), 0.0);
    e[static_cast<std::size_t>(j)] = 1.0 / std::sqrt(g.m[static_cast<std::size_t>(j)]);  // M^-1/2 e_j
    const auto next = step(c, g, dt, zero, e);
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto ii = static_cast<std::size_t>(i);
      S(i, j) = std::sqrt(g.m[ii]) * (2.0 * e[ii] - next[ii]);  // M^1/2 dt^2 A^{p,nu} M^-1/2 e_j
    }
  }
  return S;
}

// The smallest distance of the eigenvalues of dt^2 A^{p,nu} from the stability interval's
// ends 0 and 4 (negative: unstable), and how far the symmetrized operator is from symmetric.
struct Stability {
  double margin, asymmetry;
};
Stability stability(const LtsCoefficients& c, const Mesh1D& g, double dt) {
  const Eigen::MatrixXd S = scaled_operator(c, g, dt);
  const Eigen::MatrixXd sym = 0.5 * (S + S.transpose());
  const Eigen::VectorXd ev = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(sym).eigenvalues();
  return {std::min(ev.minCoeff(), 4.0 - ev.maxCoeff()), (S - S.transpose()).cwiseAbs().maxCoeff() / S.cwiseAbs().maxCoeff()};
}

double gaussian(double x) { return std::exp(-400.0 * (x - 0.5) * (x - 0.5)); }

TEST(LocalTimeStepping, CoefficientsReduceToDiazGroteWithoutStabilization) {
  for (int p : {1, 2, 3, 7}) {
    const auto c = lts_coefficients(p, 0.0);
    EXPECT_EQ(c.delta, 1.0);
    EXPECT_NEAR(c.omega, 2.0 * p * p, 1e-12 * p * p);  // T_p'(1) = p^2
    for (std::size_t k = 0; k + 1 < static_cast<std::size_t>(p); ++k) {
      EXPECT_EQ(c.beta[k], 1.0);
      EXPECT_EQ(c.beta_half[k], 1.0);
    }
  }
  // With stabilization the coefficients differ from 1 (and omega < 2 p^2).
  const auto s = lts_coefficients(3, 0.05);
  EXPECT_GT(s.delta, 1.0);
  EXPECT_LT(s.omega, 18.0);
  EXPECT_LT(s.beta[0], 1.0);
}

TEST(LocalTimeStepping, OneLevelIsLeapfrog) {
  // p = 1: A^{1,nu} = A, so the step is u_next = 2 u - u_prev - dt^2 A u (to rounding: the
  // scheme splits A u into its coarse and fine parts).
  const Mesh1D g = mesh(1.0 / 40, 1);
  std::vector<double> u(g.size()), u_prev(g.size()), au;
  for (std::size_t i = 0; i < g.size(); ++i) {
    u[i] = gaussian(g.x[i]);
    u_prev[i] = 0.9 * gaussian(g.x[i] + 0.01);
  }
  g.apply(u, au);
  const double dt = 0.9 / 40;
  for (const double nu : {0.0, 0.05}) {
    const auto next = step(lts_coefficients(1, nu), g, dt, u_prev, u);
    double worst = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i)
      worst = std::max(worst, std::abs(next[i] - (2.0 * u[i] - u_prev[i] - dt * dt * au[i])));
    EXPECT_LE(worst, 1e-14);
  }
}

TEST(LocalTimeStepping, StabilizationRemovesTheCriticalSteps) {
  // p = 2, h_c = 1/80. The paper's first critical step without stabilization is 0.7104 h_c (a
  // narrow band of unstable steps starts there); with nu = 0.01 every eigenvalue of
  // dt^2 A^{p,nu} stays inside (0, 4) over the same steps.
  const double hc = 1.0 / 80;
  const Mesh1D g = mesh(hc, 2);
  const auto plain = lts_coefficients(2, 0.0), stabilized = lts_coefficients(2, 0.01);
  double first_unstable = 0.0, stabilized_margin = 1.0, asymmetry = 0.0;
  for (int i = 0; i <= 125; ++i) {
    const double cfl = 0.700 + 0.0002 * i;  // 0.700 .. 0.725
    const Stability s = stability(plain, g, cfl * hc);
    if (first_unstable == 0.0 && s.margin < 0.0) first_unstable = cfl;
    const Stability t = stability(stabilized, g, cfl * hc);
    stabilized_margin = std::min(stabilized_margin, t.margin);
    asymmetry = std::max({asymmetry, s.asymmetry, t.asymmetry});
  }
  std::printf("[ p = 2, h_c = 1/80: first unstable step %.4f h_c without stabilization (the paper: 0.7104 h_c); "
              "smallest margin with nu = 0.01: %.2e ]\n",
              first_unstable, stabilized_margin);
  EXPECT_GE(first_unstable, 0.7080);
  EXPECT_LE(first_unstable, 0.7120);
  EXPECT_GT(stabilized_margin, 1e-4);  // inside (0, 4): stable (the plain scheme dips below 0)
  EXPECT_LE(asymmetry, 1e-12);  // A^{p,nu} is symmetric in the M inner product
}

TEST(LocalTimeStepping, ConservesTheDiscreteEnergy) {
  // h_c = 1/320, p = 2, nu = 0.01, dt = e^-nu h_c, to T = 100 (the paper's figure: constant to
  // machine precision). E_{n+1/2} = 1/2 (|(u_{n+1} - u_n) / dt|_M^2 + u_n^T M A^{p,nu} u_{n+1}).
  const double hc = 1.0 / 320, nu = 0.01, dt = std::exp(-nu) * hc;
  const Mesh1D g = mesh(hc, 2);
  const auto c = lts_coefficients(2, nu);
  const std::size_t n = g.size();
  std::vector<double> u0(n), u1(n), au, zero(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) u0[i] = gaussian(g.x[i]);
  g.apply(u0, au);
  for (std::size_t i = 0; i < n; ++i) u1[i] = u0[i] - 0.5 * dt * dt * au[i];  // v0 = 0
  auto energy = [&](const std::vector<double>& a, const std::vector<double>& b) {
    const auto next = step(c, g, dt, zero, b);  // dt^2 A^{p,nu} b = 2 b - next
    double kin = 0.0, pot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double v = (b[i] - a[i]) / dt;
      kin += g.m[i] * v * v;
      pot += g.m[i] * a[i] * (2.0 * b[i] - next[i]) / (dt * dt);
    }
    return 0.5 * (kin + pot);
  };
  const double e0 = energy(u0, u1);
  const int steps = static_cast<int>(100.0 / dt);
  double worst = 0.0;
  for (int s = 0; s < steps; ++s) {
    auto u2 = step(c, g, dt, u0, u1);
    u0.swap(u1);
    u1.swap(u2);
    if (s % 500 == 0 || s + 1 == steps) worst = std::max(worst, std::abs(energy(u0, u1) / e0 - 1.0));
  }
  std::printf("[ %d coarse steps (T = 100): largest relative energy change %.2e ]\n", steps, worst);
  EXPECT_LE(worst, 1e-11);
}

TEST(LocalTimeStepping, ConvergesAtSecondOrderWithLeapfrogsAccuracy) {
  // d'Alembert's solution with the odd, 2-periodic extension of the Gaussian (Dirichlet ends,
  // c = 1, v0 = 0), at t = 1.33, with the largest step at most 0.9 h_c that divides t.
  auto extension = [](double s) {
    s = std::fmod(s + 1.0, 2.0);
    if (s < 0.0) s += 2.0;
    s -= 1.0;
    return s >= 0.0 ? gaussian(s) : -gaussian(-s);
  };
  const double T = 1.33;
  auto error = [&](double hc, int p) {
    const Mesh1D g = mesh(hc, p);
    const auto c = lts_coefficients(p, 0.01);
    const int steps = static_cast<int>(std::ceil(T / (0.9 * hc)));
    const double dt = T / steps;
    const std::size_t n = g.size();
    std::vector<double> u0(n), u1(n), au;
    for (std::size_t i = 0; i < n; ++i) u0[i] = gaussian(g.x[i]);
    g.apply(u0, au);
    for (std::size_t i = 0; i < n; ++i) u1[i] = u0[i] - 0.5 * dt * dt * au[i];
    for (int s = 1; s < steps; ++s) {
      auto u2 = step(c, g, dt, u0, u1);
      u0.swap(u1);
      u1.swap(u2);
    }
    double e2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double exact = 0.5 * (extension(g.x[i] - T) + extension(g.x[i] + T));
      e2 += g.m[i] * (u1[i] - exact) * (u1[i] - exact);
    }
    return std::sqrt(e2);
  };
  const double hcs[] = {1.0 / 80, 1.0 / 160, 1.0 / 320, 1.0 / 640};
  for (const int p : {2, 3, 5}) {
    double err[4], ref[4];
    for (int k = 0; k < 4; ++k) {
      err[k] = error(hcs[k], p);
      ref[k] = error(hcs[k], 1);  // leapfrog on the coarse mesh's step
    }
    std::printf("[ p = %d: L2 errors at t = 1.33: %.3e %.3e %.3e %.3e (leapfrog %.3e ... %.3e), orders %.2f %.2f %.2f ]\n",
                p, err[0], err[1], err[2], err[3], ref[0], ref[3], std::log2(err[0] / err[1]),
                std::log2(err[1] / err[2]), std::log2(err[2] / err[3]));
    for (int k = 0; k < 3; ++k) {
      EXPECT_GT(std::log2(err[k] / err[k + 1]), 1.9);
      EXPECT_LT(std::abs(err[k] / ref[k] - 1.0), 0.15);  // as accurate as leapfrog
    }
  }
}

}  // namespace
