// Uniform damping, step 1 (docs/design.md, "Frequency-independent damping"): the port of
// OpenSees' UniformDamping (physics/damping/uniform_damping.hpp) against OpenSees itself.
//
// - The filter fit: 5 filters for 0.1-10 Hz, log-spaced, a response within 5 % of 1.
// - Single-mode free vibration exactly as OpenSees runs it (tests/reference/damping/
//   sdof_uniform.tcl: a mass of 1 on a zeroLength spring, released from u = 1, Newmark 0.5 /
//   0.25 with Newton, 5 % over 0.1-10 Hz): the displacement history of OpenSees'
//   ZeroLength + UniformDamping at 0.1, 1 and 10 Hz, to 1e-11 of the amplitude.
// - The damping ratio measured from those histories reproduces OpenSees' (4.69 % at 0.1 Hz,
//   4.45 % at 1 Hz, 2.79 % at 10 Hz for a 5 % target): the drop that step 2 corrects with
//   FemoraX's own fit (docs/design.md).
// - Static stages and inactive times: no damping and a stiffness multiplier of 1.
// - FemoraX's fit (step 2a) and the oscillation formula; inside solid elements (step 2b-1):
//   a stdBrick mode decays exactly like the single-mode spring, rigid motion is undamped.

#include "../common/synthetic_models.hpp"
#include "compile/batches.hpp"
#include "compile/uniform_damping_fit.hpp"
#include "exec/force_settings.hpp"
#include "exec/internal_force.hpp"
#include "physics/damping/element_damping.hpp"
#include "physics/damping/uniform_damped.hpp"
#include "physics/damping/uniform_damping.hpp"
#include "physics/elements/ssp_brick.hpp"
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"
#include "reference_data.hpp"

#include <Eigen/Dense>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace femorax;
constexpr double kPi = 3.14159265358979323846;

UniformDampingFilters fit_0p1_10() {
  UniformDampingFilters fl;
  EXPECT_TRUE(uniform_damping_fit(0.1, 10.0, 0.05, fl));
  return fl;
}

TEST(UniformDamping, FitIsLogSpacedAndWithinFivePercent) {
  const auto fl = fit_0p1_10();
  ASSERT_EQ(fl.nFilter, 5);
  for (int i = 0; i < fl.nFilter; ++i) {
    EXPECT_NEAR(fl.omegac[i] / (2 * kPi), std::pow(10.0, -1.0 + 0.5 * i), 1e-9);
    EXPECT_GT(fl.alpha[i], 0.0);
  }
  for (int k = 0; k <= 400; ++k) {
    const double omega = 2 * kPi * std::pow(10.0, -1.0 + 2.0 * k / 400);
    double r = 0;
    for (int j = 0; j < fl.nFilter; ++j) {
      const double w = omega / fl.omegac[j];
      r += fl.alpha[j] * 2 * w / (1 + w * w);
    }
    EXPECT_NEAR(r, 1.0, 0.05) << "at " << omega / (2 * kPi) << " Hz";
  }
}

TEST(UniformDamping, StaticAndInactiveStepsAddNothing) {
  const auto fl = fit_0p1_10();
  const int n = 1 * (fl.nFilter + 2);
  std::vector<double> C(n, 0.0), T(n, 0.0);
  const double q[1] = {3.0};
  double qd[1];
  UniformDampingStep st;  // static
  uniform_damping_update<1>(fl, 0.05, st, {C.data(), 1}, {T.data(), 1}, q, qd);
  EXPECT_EQ(qd[0], 0.0);
  for (int i = 0; i < fl.nFilter; ++i) EXPECT_EQ(T[2 + i], 3.0);  // the filters track q
  EXPECT_EQ(uniform_damping_stiffness_multiplier(fl, 0.05, st), 1.0);
  st.dynamic = true;
  st.dT = 0.01;
  st.t = 0.5;
  st.ta = 1.0;  // not yet active
  uniform_damping_update<1>(fl, 0.05, st, {C.data(), 1}, {T.data(), 1}, q, qd);
  EXPECT_EQ(qd[0], 0.0);
  EXPECT_EQ(uniform_damping_stiffness_multiplier(fl, 0.05, st), 1.0);
  st.ta = 0.0;
  EXPECT_GT(uniform_damping_stiffness_multiplier(fl, 0.05, st), 1.0);
}

struct Record {
  std::vector<double> t, u;
};

Record read_record(const std::string& name) {
  std::ifstream in(femorax::test::reference_data_dir() / "damping" / name);
  EXPECT_TRUE(in.good()) << name;
  Record r;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    double t, u;
    is >> t >> u;
    r.t.push_back(t);
    r.u.push_back(u);
  }
  return r;
}

// sdof_uniform.tcl as OpenSees runs it. Static step: u = 1, the filters take q = k u. Then
// Newmark (form D: OpenSees' Newmark::newStep predictor, accelerations from 0 since the
// script does not call initialize) with Newton to NormDispIncr 1e-14: the ZeroLength force
// k u + qd and tangent k (1 + km).
std::vector<double> simulate(double f, const UniformDampingFilters& fl, int steps) {
  const double eta = uniform_damping_eta(0.05), m = 1.0, gamma = 0.5, beta = 0.25;
  const double w = 2 * kPi * f, k = w * w, dt = 1.0 / f / 400.0;
  const int n = fl.nFilter + 2;
  std::vector<double> C(n, 0.0), T(n, 0.0);
  double u = 1.0, v = 0.0, a = 0.0, qd[1];
  {
    const double q[1] = {k * u};
    UniformDampingStep st;
    uniform_damping_update<1>(fl, eta, st, {C.data(), 1}, {T.data(), 1}, q, qd);
    C = T;
  }
  std::vector<double> out;
  double t = 0.0;
  for (int s = 0; s < steps; ++s) {
    UniformDampingStep st;
    st.dynamic = true;
    st.dT = dt;
    st.t = t + dt;
    const double un = u, vn = v, an = a;
    // predictor (Newmark::newStep): U unchanged
    double U = un;
    double Ud = (1.0 - gamma / beta) * vn + dt * (1.0 - 0.5 * gamma / beta) * an;
    double Udd = -1.0 / (beta * dt) * vn + (1.0 - 0.5 / beta) * an;
    const double c2 = gamma / (beta * dt), c3 = 1.0 / (beta * dt * dt);
    for (int it = 0; it < 20; ++it) {
      const double q[1] = {k * U};
      uniform_damping_update<1>(fl, eta, st, {C.data(), 1}, {T.data(), 1}, q, qd);
      const double R = -(m * Udd) - (k * U + qd[0]);
      const double K = c3 * m + k * uniform_damping_stiffness_multiplier(fl, eta, st);
      const double dU = R / K;
      U += dU;
      Ud += c2 * dU;
      Udd += c3 * dU;
      if (std::fabs(dU) <= 1e-14) break;
    }
    {
      const double q[1] = {k * U};
      uniform_damping_update<1>(fl, eta, st, {C.data(), 1}, {T.data(), 1}, q, qd);
    }
    C = T;  // commit
    u = U;
    v = Ud;
    a = Udd;
    t += dt;
    out.push_back(u);
  }
  return out;
}

double damping_ratio(const std::vector<double>& u) {
  std::vector<double> peaks;
  for (std::size_t i = 1; i + 1 < u.size(); ++i)
    if (u[i] > u[i - 1] && u[i] >= u[i + 1] && u[i] > 0) peaks.push_back(u[i]);
  const double d = std::log(peaks.front() / peaks.back()) / static_cast<double>(peaks.size() - 1);
  return d / std::sqrt(4 * kPi * kPi + d * d);
}

TEST(UniformDampingReference, SingleModeFreeVibrationMatchesOpenSees) {
  const auto fl = fit_0p1_10();
  const struct {
    const char* file;
    double f, zeta;  // OpenSees' measured ratio (docs/design.md)
  } cases[] = {{"sdof_uniform_0.1.txt", 0.1, 0.04691}, {"sdof_uniform_1.txt", 1.0, 0.04452},
               {"sdof_uniform_10.txt", 10.0, 0.02792}};
  for (const auto& c : cases) {
    const Record ref = read_record(c.file);
    ASSERT_EQ(ref.u.size(), 2400u) << c.file;
    const auto got = simulate(c.f, fl, static_cast<int>(ref.u.size()));
    double worst = 0;
    for (std::size_t i = 0; i < ref.u.size(); ++i) worst = std::max(worst, std::fabs(got[i] - ref.u[i]));
    std::printf("[ Uniform damping, %4g Hz: largest difference from OpenSees %.2e; damping %.3f %% (OpenSees %.3f %%) ]\n",
                c.f, worst, 100 * damping_ratio(got), 100 * damping_ratio(ref.u));
    EXPECT_LE(worst, 1e-11) << c.file;
    EXPECT_NEAR(damping_ratio(got), c.zeta, 5e-5) << c.file;
  }
}

// The damping of the oscillation in a step-by-step history, its slow part (the filters'
// own relaxation) removed: least squares of u on [e^-at cos(wd t), e^-at sin(wd t), 1, t,
// t^2] over a grid of (a, wd) around a guess, refined once. Returns the damping ratio.
double fitted_zeta(const std::vector<double>& u, double f, double zeta0, double wn0) {
  const double dt = 1.0 / f / 400.0;
  auto misfit = [&](double a, double wd) {
    double A[5][5] = {}, b[5] = {}, uu = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
      const double t = dt * static_cast<double>(i + 1), e = std::exp(-a * t);
      const double r[5] = {e * std::cos(wd * t), e * std::sin(wd * t), 1.0, t, t * t};
      for (int p = 0; p < 5; ++p) {
        for (int q = 0; q < 5; ++q) A[p][q] += r[p] * r[q];
        b[p] += r[p] * u[i];
      }
      uu += u[i] * u[i];
    }
    // residual = u.u - b^T A^-1 b (normal equations), by elimination
    double c[5], bb[5];
    for (int p = 0; p < 5; ++p) bb[p] = b[p];
    for (int col = 0; col < 5; ++col)
      for (int r2 = col + 1; r2 < 5; ++r2) {
        const double fct = A[r2][col] / A[col][col];
        for (int k = col; k < 5; ++k) A[r2][k] -= fct * A[col][k];
        bb[r2] -= fct * bb[col];
      }
    for (int r2 = 4; r2 >= 0; --r2) {
      double v = bb[r2];
      for (int k = r2 + 1; k < 5; ++k) v -= A[r2][k] * c[k];
      c[r2] = v / A[r2][r2];
    }
    double fit = 0;
    for (int p = 0; p < 5; ++p) fit += c[p] * b[p];
    return uu - fit;
  };
  double a0 = zeta0 * wn0, w0 = wn0, best = 1e300, ba = a0, bw = w0;
  for (double span : {0.3, 0.03}) {
    const double ac = ba, wc = bw;
    for (int i = -20; i <= 20; ++i)
      for (int j = -20; j <= 20; ++j) {
        const double a = ac * (1 + span * i / 20.0), wd = wc * (1 + 0.2 * span * j / 20.0);
        const double r = misfit(a, wd);
        if (r < best) {
          best = r;
          ba = a;
          bw = wd;
        }
      }
  }
  return ba / std::hypot(ba, bw);
}

TEST(UniformDamping, TheOscillationFormulaDescribesTheFilter) {
  const auto fl = fit_0p1_10();
  for (double f : {0.1, 1.0, 10.0}) {
    const auto mode = uniform_damping_mode(fl, uniform_damping_eta(0.05), 2 * kPi * f);
    const double z = fitted_zeta(simulate(f, fl, 2400), f, mode.zeta, mode.frequency_ratio * 2 * kPi * f);
    std::printf("[ OpenSees fit, %4g Hz: formula %.3f %%, frequency x%.3f; step by step %.3f %% ]\n", f, 100 * mode.zeta,
                mode.frequency_ratio, 100 * z);
    EXPECT_NEAR(z / mode.zeta, 1.0, 0.03) << f;
  }
}

TEST(UniformDamping, FemoraxFitGivesTheTargetAcrossTheBand) {
  for (const auto& band : {std::pair{0.1, 10.0}, std::pair{0.5, 20.0}, std::pair{0.2, 2.0}}) {
    for (double zeta : {0.02, 0.05, 0.10}) {
      UniformDampingFilters fl;
      ASSERT_TRUE(uniform_damping_fit_femorax(zeta, band.first, band.second, 0.05, fl));
      double lo = 1e9, hi = 0, shift = 0;
      for (int k = 0; k <= 200; ++k) {
        const double f = band.first * std::pow(band.second / band.first, k / 200.0);
        const auto mode = uniform_damping_mode(fl, uniform_damping_eta(zeta), 2 * kPi * f);
        lo = std::min(lo, mode.zeta / zeta);
        hi = std::max(hi, mode.zeta / zeta);
        shift = std::max(shift, mode.frequency_ratio);
      }
      std::printf("[ FemoraX fit %4.1f %% over %g-%g Hz: %d filters, damping %.3f-%.3f of the target, frequency up to x%.3f ]\n",
                  100 * zeta, band.first, band.second, fl.nFilter, lo, hi, shift);
      EXPECT_GE(lo, 0.95);
      EXPECT_LE(hi, 1.05);
    }
  }
  // and the step-by-step filter delivers it
  UniformDampingFilters fl;
  ASSERT_TRUE(uniform_damping_fit_femorax(0.05, 0.1, 10.0, 0.05, fl));
  for (double f : {0.1, 1.0, 10.0}) {
    const auto mode = uniform_damping_mode(fl, uniform_damping_eta(0.05), 2 * kPi * f);
    const double z = fitted_zeta(simulate(f, fl, 2400), f, mode.zeta, mode.frequency_ratio * 2 * kPi * f);
    std::printf("[ FemoraX fit 5 %% over 0.1-10 Hz, %4g Hz: step by step %.3f %% ]\n", f, 100 * z);
    EXPECT_NEAR(z / 0.05, 1.0, 0.06) << f;
  }
}

// ---------------------------------------------------------------------------------------
// Step 2b-1: inside solid elements. A unit stdBrick (ElasticIsotropic, lumped mass), its four
// base nodes fixed. Its stiffness is sum_gp B^T D B dvol; the damping filters each point's
// stress and returns B^T qd dvol. Started in one of its own modes, the element stays in that
// mode, and the mode's amplitude is exactly the single-mode spring of the same frequency
// (the same filter acting on k u): the element path matches simulate() to rounding.
using Brick = StdBrick<ElasticIsotropic>;
constexpr double kXe[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

Brick::Params brick_params() {
  Brick::Params p{};
  p.mat.E = 1.0e5;
  p.mat.v = 0.3;
  p.mat.rho = 70.0;  // heavy enough to put the first mode inside the 0.1-10 Hz band
  p.massType = 1;    // lumped
  return p;
}

// Point stresses D B u of the brick for nodal displacements u (24 values).
void brick_stresses(const Brick::Params& p, const double* u, double (*sig)[6]) {
  double xl[3][8], D[6][6];
  Brick::local_coords(kXe, xl);
  ElasticIsotropic::initial_tangent(p.mat, D);
  const auto g = Brick::recomputed_geometry(xl, p);
  for (int gp = 0; gp < 8; ++gp) {
    Brick::PointShape s;
    g.shape(gp, s);
    double eps[6];
    Brick::point_strain(s, u, eps);
    for (int a = 0; a < 6; ++a) {
      sig[gp][a] = 0;
      for (int c = 0; c < 6; ++c) sig[gp][a] += D[a][c] * eps[c];
    }
  }
}

// f = B^T sig dvol summed over the points (the element force of point stresses sig).
void brick_force(const Brick::Params& p, const double (*sig)[6], double* f) {
  double xl[3][8];
  Brick::local_coords(kXe, xl);
  uniform_damping_element_force<Brick>(Brick::recomputed_geometry(xl, p), sig, f);
}

TEST(UniformDampingElement, AStdBrickModeDecaysLikeTheSingleModeSpring) {
  const auto p = brick_params();
  // Free DOFs: the top nodes 4..7 (12 values); stiffness and lumped mass there.
  constexpr int NF = 12;
  Eigen::MatrixXd K(NF, NF), M = Eigen::MatrixXd::Zero(NF, NF);
  for (int a = 0; a < NF; ++a) {
    double u[24] = {}, sig[8][6], f[24];
    u[12 + a] = 1.0;
    brick_stresses(p, u, sig);
    brick_force(p, sig, f);
    for (int b = 0; b < NF; ++b) K(b, a) = f[12 + b];
  }
  {
    double Me[24 * 24];
    Brick::mass(p, kXe, Me);
    for (int a = 0; a < NF; ++a) M(a, a) = Me[(12 + a) * 24 + (12 + a)];
  }
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eig(K, M);
  const double lambda = eig.eigenvalues()(0);
  Eigen::VectorXd phi = eig.eigenvectors().col(0);
  phi /= std::sqrt(phi.dot(M * phi));
  const double f0 = std::sqrt(lambda) / (2 * kPi), dt = 1.0 / f0 / 400.0;
  for (const bool femorax_fit : {false, true}) {
    UniformDampingFilters fl;
    if (femorax_fit) ASSERT_TRUE(uniform_damping_fit_femorax(0.05, 0.1, 10.0, 0.05, fl));
    else fl = fit_0p1_10();
    const double eta = uniform_damping_eta(0.05);
    const int ns = 6 * (fl.nFilter + 2);
    std::vector<double> C(8 * ns, 0.0), T(8 * ns, 0.0);
    auto damping = [&](const Eigen::VectorXd& U, const UniformDampingStep& st, double* fd) {
      double u[24] = {}, sig[8][6], qd[8][6];
      for (int a = 0; a < NF; ++a) u[12 + a] = U(a);
      brick_stresses(p, u, sig);
      for (int gp = 0; gp < 8; ++gp)
        uniform_damping_update<6>(fl, eta, st, {C.data() + gp * ns, 1}, {T.data() + gp * ns, 1}, sig[gp], qd[gp]);
      brick_force(p, qd, fd);
    };
    // static step: u = phi, the filters take its stresses
    Eigen::VectorXd U = phi, V = Eigen::VectorXd::Zero(NF), A = Eigen::VectorXd::Zero(NF);
    {
      double fd[24];
      damping(U, UniformDampingStep{}, fd);
      C = T;
    }
    const int steps = 2400;
    const double g = 0.5, b = 0.25, c2 = g / (b * dt), c3 = 1.0 / (b * dt * dt);
    std::vector<double> amp;
    double t = 0;
    for (int n = 0; n < steps; ++n) {
      UniformDampingStep st;
      st.dynamic = true;
      st.dT = dt;
      st.t = t + dt;
      Eigen::VectorXd Ud = (1 - g / b) * V + dt * (1 - 0.5 * g / b) * A;
      Eigen::VectorXd Udd = -1.0 / (b * dt) * V + (1 - 0.5 / b) * A;
      Eigen::VectorXd Un = U;
      const double km = uniform_damping_stiffness_multiplier(fl, eta, st);
      const Eigen::MatrixXd Keff = c3 * M + km * K;
      for (int it = 0; it < 20; ++it) {
        double fd[24];
        damping(Un, st, fd);
        Eigen::VectorXd Fd(NF);
        for (int a = 0; a < NF; ++a) Fd(a) = fd[12 + a];
        const Eigen::VectorXd R = -(M * Udd) - (K * Un + Fd);
        const Eigen::VectorXd dU = Keff.ldlt().solve(R);
        Un += dU;
        Ud += c2 * dU;
        Udd += c3 * dU;
        if (dU.lpNorm<Eigen::Infinity>() <= 1e-14) break;
      }
      {
        double fd[24];
        damping(Un, st, fd);
      }
      C = T;
      U = Un;
      V = Ud;
      A = Udd;
      t += dt;
      amp.push_back(phi.dot(M * U));  // the mode's amplitude
    }
    const auto ref = simulate(f0, fl, steps);  // the single-mode spring at the brick's frequency
    double worst = 0;
    for (int n = 0; n < steps; ++n) worst = std::max(worst, std::fabs(amp[static_cast<std::size_t>(n)] - ref[static_cast<std::size_t>(n)]));
    std::printf("[ stdBrick mode at %.3f Hz, %s fit: amplitude within %.1e of the single-mode spring ]\n", f0,
                femorax_fit ? "FemoraX" : "OpenSees", worst);
    EXPECT_LE(worst, 1e-9);
  }
}

TEST(UniformDampingElement, RigidMotionGivesNoDampingForce) {
  // Every node moves by the same translation and a small rotation about z: no strain, so the
  // filtered stresses stay zero and so does the damping force, for stdBrick and SSPbrick.
  const auto fl = fit_0p1_10();
  const double eta = uniform_damping_eta(0.05);
  UniformDampingStep st;
  st.dynamic = true;
  st.dT = 0.001;
  st.t = 0.001;
  double u[24];
  for (int j = 0; j < 8; ++j) {
    u[3 * j] = 0.3 - 1e-3 * kXe[j][1];
    u[3 * j + 1] = -0.2 + 1e-3 * kXe[j][0];
    u[3 * j + 2] = 0.1;
  }
  const auto p = brick_params();
  double sig[8][6], qd[8][6], f[24];
  brick_stresses(p, u, sig);
  const int ns = 6 * (fl.nFilter + 2);
  std::vector<double> C(ns, 0.0), T(ns, 0.0);
  for (int gp = 0; gp < 8; ++gp) uniform_damping_update<6>(fl, eta, st, {C.data(), 1}, {T.data(), 1}, sig[gp], qd[gp]);
  brick_force(p, qd, f);
  double fmax = 0;
  for (double v : f) fmax = std::max(fmax, std::fabs(v));
  EXPECT_LE(fmax, 1e-9);

  using Ssp = SSPbrick<ElasticIsotropic>;
  static_assert(UniformDampedSolid<Brick> && UniformDampedSolid<Ssp>);
  static_assert(uniform_damping_quantities<Brick> == 48 && uniform_damping_quantities<Ssp> == 6);
  Ssp::Params sp{};
  sp.mat = p.mat;
  double xl[3][8];
  Ssp::local_coords(kXe, xl);
  const auto g = Ssp::recomputed_geometry(xl, sp);
  Ssp::PointShape s;
  g.shape(0, s);
  double eps[6], D[6][6], s0[6] = {}, q0[1][6];
  Ssp::point_strain(s, u, eps);
  ElasticIsotropic::initial_tangent(p.mat, D);
  for (int a = 0; a < 6; ++a)
    for (int c = 0; c < 6; ++c) s0[a] += D[a][c] * eps[c];
  uniform_damping_update<6>(fl, eta, st, {C.data(), 1}, {T.data(), 1}, s0, q0[0]);
  uniform_damping_element_force<Ssp>(g, q0, f);
  fmax = 0;
  for (double v : f) fmax = std::max(fmax, std::fabs(v));
  EXPECT_LE(fmax, 1e-9);
}

// ---------------------------------------------------------------------------------------
// Step 2b-2 (ii)-a: the damped-material wrapper, through the element's own compute (the path
// every kernel uses) with the step context in the Rayleigh struct.
using DampedEI = UniformDamped<ElasticIsotropic, 8>;
using DampedBrick = StdBrick<DampedEI>;
static_assert(Material<DampedEI>);

DampedBrick::Params damped_brick_params(const UniformDampingFilters& fl) {
  const auto pb = brick_params();
  DampedBrick::Params p{};
  p.mat.mat = pb.mat;
  p.mat.fl = fl;
  p.mat.eta = uniform_damping_eta(0.05);
  p.b[0] = p.b[1] = p.b[2] = 0.0;
  p.massType = pb.massType;
  return p;
}

TEST(UniformDampedMaterial, StaticStepsAreTheUndampedElement) {
  const auto fl = fit_0p1_10();
  const auto pd = damped_brick_params(fl);
  const auto pb = brick_params();
  double u[24], fe_d[24], fe_b[24], K_d[24][24], K_b[24][24];
  for (int a = 0; a < 24; ++a) u[a] = 1e-3 * std::sin(1.0 + a);
  const double v[24] = {};
  constexpr int NS = DampedEI::NSTATE;
  std::vector<double> C(8 * NS, 0.0), T(8 * NS, 0.0);
  Rayleigh damp;  // static: step.dynamic = 0
  DampedBrick::compute<true>(pd, damp, kXe, u, v, {C.data(), 1, NS}, {T.data(), 1, NS}, fe_d, K_d);
  Brick::compute<true>(pb, damp, kXe, u, v, {}, {}, fe_b, K_b);
  for (int a = 0; a < 24; ++a) {
    EXPECT_EQ(fe_d[a], fe_b[a]) << a;
    for (int c = 0; c < 24; ++c) EXPECT_EQ(K_d[a][c], K_b[a][c]);
  }
  // the filters took the stresses (static tracking), the damping stress stayed 0
  for (int gp = 0; gp < 8; ++gp) EXPECT_EQ(T[gp * NS + 6], 0.0);
}

TEST(UniformDampedMaterial, ADampedStdBrickModeDecaysLikeTheSingleModeSpring) {
  // The test of step 2b-1, now through DampedBrick::compute<true> (force with the damping,
  // tangent with the multiplier) and the step context, for both fits.
  const auto pb = brick_params();
  constexpr int NF = 12, NS = DampedEI::NSTATE;
  Eigen::MatrixXd K(NF, NF), M = Eigen::MatrixXd::Zero(NF, NF);
  {
    double u0[24] = {}, v0[24] = {}, fe[24], Ke[24][24];
    Brick::compute<true>(pb, Rayleigh{}, kXe, u0, v0, {}, {}, fe, Ke);
    for (int a = 0; a < NF; ++a)
      for (int b = 0; b < NF; ++b) K(a, b) = Ke[12 + a][12 + b];
    double Me[24 * 24];
    Brick::mass(pb, kXe, Me);
    for (int a = 0; a < NF; ++a) M(a, a) = Me[(12 + a) * 24 + (12 + a)];
  }
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eig(K, M);
  Eigen::VectorXd phi = eig.eigenvectors().col(0);
  phi /= std::sqrt(phi.dot(M * phi));
  const double f0 = std::sqrt(eig.eigenvalues()(0)) / (2 * kPi), dt = 1.0 / f0 / 400.0;
  for (const bool femorax_fit : {false, true}) {
    UniformDampingFilters fl;
    if (femorax_fit) ASSERT_TRUE(uniform_damping_fit_femorax(0.05, 0.1, 10.0, 0.05, fl));
    else fl = fit_0p1_10();
    const auto pd = damped_brick_params(fl);
    std::vector<double> C(8 * NS, 0.0), T(8 * NS, 0.0);
    auto element = [&](const Eigen::VectorXd& U, const StepContext& step, Eigen::VectorXd* F, Eigen::MatrixXd* Kt) {
      double u[24] = {}, v[24] = {}, fe[24], Ke[24][24];
      for (int a = 0; a < NF; ++a) u[12 + a] = U(a);
      Rayleigh damp;
      damp.step = step;
      DampedBrick::compute<true>(pd, damp, kXe, u, v, {C.data(), 1, NS}, {T.data(), 1, NS}, fe, Ke);
      if (F) {
        F->resize(NF);
        for (int a = 0; a < NF; ++a) (*F)(a) = fe[12 + a];
      }
      if (Kt) {
        Kt->resize(NF, NF);
        for (int a = 0; a < NF; ++a)
          for (int b = 0; b < NF; ++b) (*Kt)(a, b) = Ke[12 + a][12 + b];
      }
    };
    Eigen::VectorXd U = phi, V = Eigen::VectorXd::Zero(NF), A = Eigen::VectorXd::Zero(NF);
    element(U, StepContext{}, nullptr, nullptr);  // the static state: the filters take phi's stresses
    C = T;
    const double g = 0.5, b = 0.25, c2 = g / (b * dt), c3 = 1.0 / (b * dt * dt);
    const int steps = 2400;
    std::vector<double> amp;
    double t = 0;
    for (int n = 0; n < steps; ++n) {
      const StepContext step{t + dt, dt, 1};
      Eigen::VectorXd Ud = (1 - g / b) * V + dt * (1 - 0.5 * g / b) * A;
      Eigen::VectorXd Udd = -1.0 / (b * dt) * V + (1 - 0.5 / b) * A;
      Eigen::VectorXd Un = U;
      for (int it = 0; it < 20; ++it) {
        Eigen::VectorXd F;
        Eigen::MatrixXd Kt;
        element(Un, step, &F, &Kt);
        const Eigen::VectorXd R = -(M * Udd) - F;
        const Eigen::VectorXd dU = (c3 * M + Kt).ldlt().solve(R);
        Un += dU;
        Ud += c2 * dU;
        Udd += c3 * dU;
        if (dU.lpNorm<Eigen::Infinity>() <= 1e-14) break;
      }
      element(Un, step, nullptr, nullptr);
      C = T;
      U = Un;
      V = Ud;
      A = Udd;
      t += dt;
      amp.push_back(phi.dot(M * U));
    }
    const auto ref = simulate(f0, fl, steps);
    double worst = 0;
    for (int n = 0; n < steps; ++n)
      worst = std::max(worst, std::fabs(amp[static_cast<std::size_t>(n)] - ref[static_cast<std::size_t>(n)]));
    std::printf("[ damped stdBrick through compute, mode at %.3f Hz, %s fit: within %.1e of the single-mode spring ]\n",
                f0, femorax_fit ? "FemoraX" : "OpenSees", worst);
    EXPECT_LE(worst, 1e-9);
  }
}

TEST(UniformDampedMaterial, RecordersSeeTheMaterialsOwnStress) {
  const auto fl = fit_0p1_10();
  const auto pd = damped_brick_params(fl);
  const double eps[6] = {1e-4, -2e-4, 3e-5, 1e-4, 0, -5e-5};
  constexpr int NS = DampedEI::NSTATE;
  std::vector<double> C(NS, 0.0), scratch(NS, 0.0);
  double rec[6], ref[6], D[6][6];
  DampedEI::recorded_stress(pd.mat, {C.data(), 1}, {scratch.data(), 1}, eps, rec);
  ElasticIsotropic::update(pd.mat.mat, {}, {}, eps, ref, D);
  for (int a = 0; a < 6; ++a) EXPECT_EQ(rec[a], ref[a]);
}

// ---------------------------------------------------------------------------------------
// Step 2b-2 (ii)-b: damped batches from the model's Uniform damping regions, through the
// real force kernels.
io::Model damped_column(bool damped, io::ElementType type) {
  femorax::testing::ColumnSpec spec;
  spec.nx = 2;
  spec.ny = 2;
  spec.nz = 3;
  spec.damping = false;
  spec.element = type;
  io::Model m = femorax::testing::soil_column(spec);
  if (damped) {
    io::UniformDampingRegion r;
    r.id = "soil";
    r.ratio = 0.05;
    r.freq1 = 0.1;
    r.freq2 = 10.0;
    for (const auto& g : m.elements) r.elements.insert(r.elements.end(), g.tags.begin(), g.tags.end());
    m.uniform.push_back(r);
  }
  return m;
}

std::vector<double> force_at(exec::InternalForce& k, const std::vector<double>& u, const StepContext& step,
                             const exec::ForceSettings& s) {
  auto hu = Kokkos::create_mirror_view(k.u);
  for (std::size_t q = 0; q < u.size(); ++q) hu(q) = u[q];
  Kokkos::deep_copy(k.u, hu);
  Kokkos::deep_copy(k.v, 0.0);
  k.step = step;
  k.compute(s);
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, k.f);
  return std::vector<double>(hf.data(), hf.data() + hf.extent(0));
}

TEST(UniformDampedKernels, DampedBatchesStaticEqualUndampedAndEveryVariantAgrees) {
  for (const auto type : {io::ElementType::StdBrick, io::ElementType::SSPbrick}) {
    SCOPED_TRACE(io::to_string(type));
    const auto cd = compile::compile_model(damped_column(true, type), compile::CompileOptions{});
    const auto cu = compile::compile_model(damped_column(false, type), compile::CompileOptions{});
    ASSERT_EQ(cd.batches.size(), 1u);
    EXPECT_GT(cd.batches[0].nstate, 0);  // the wrapper's filter state
    exec::InternalForce kd(cd, exec::GeometryCache::On), ku(cu, exec::GeometryCache::On);
    std::vector<double> u(static_cast<std::size_t>(kd.num_equations()));
    for (std::size_t q = 0; q < u.size(); ++q) u[q] = 1e-3 * std::sin(0.7 + 1.3 * static_cast<double>(q));
    const auto def = exec::default_force_settings();
    // static: the damped column is the undamped one, bit for bit
    const auto fs = force_at(kd, u, StepContext{}, def), fu = force_at(ku, u, StepContext{}, def);
    ASSERT_EQ(fs.size(), fu.size());
    for (std::size_t q = 0; q < fs.size(); ++q) EXPECT_EQ(fs[q], fu[q]) << q;
    // dynamic, at a changed displacement: the damping force appears (the stresses changed
    // since the committed state), the same in every variant
    kd.commit_state();  // the static evaluation's filter state becomes committed
    std::vector<double> u2(u.size());
    for (std::size_t q = 0; q < u.size(); ++q) u2[q] = 1.2 * u[q] + 1e-4 * std::cos(0.3 * static_cast<double>(q));
    const StepContext step{0.001, 0.001, 1};
    const auto fu2 = force_at(ku, u2, StepContext{}, def);
    const auto variants = exec::tunable_variants(kd);
    const auto first = force_at(kd, u2, step, variants[0]);
    double diff = 0, scale = 0;
    for (std::size_t q = 0; q < first.size(); ++q) {
      diff = std::max(diff, std::fabs(first[q] - fu2[q]));
      scale = std::max(scale, std::fabs(fu2[q]));
    }
    EXPECT_GT(diff, 1e-6 * scale);
    for (const auto& s : variants) {
      SCOPED_TRACE(exec::to_string(s));
      const auto f = force_at(kd, u2, step, s);
      EXPECT_EQ(std::memcmp(f.data(), first.data(), f.size() * sizeof(double)), 0);
    }
  }
}

}  // namespace

