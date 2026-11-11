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

#include "compile/uniform_damping_fit.hpp"
#include "physics/damping/uniform_damping.hpp"
#include "reference_data.hpp"

#include <gtest/gtest.h>

#include <cmath>
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

}  // namespace

